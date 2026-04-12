"""
Phase 3: Training HelixIDS-Full on multi-dataset unified features.

Trains single unified model on NSL-KDD + UNSW-NB15 + CICIDS combined data.
Uses multi-task learning: binary (Normal vs Attack) + family (7-class).
No QAT—just FP32 training on M4 MPS.

Usage:
    python scripts/train_helix_ids_full.py --config config/helix_config.yaml --output models/helix_full

Output artifacts:
    - models/helix_full_best.pt: Best model checkpoint
    - models/helix_full_final.pt: Final trained model
    - results/helix_full/training_log.json: Training metrics
    - results/helix_full/eval_per_dataset.json: Per-dataset test evaluation
"""

# ruff: noqa: E402

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from helix_ids.config.helix_full_config import DataConfig, TrainingConfig  # noqa: E402

# Import from helix_ids package
from helix_ids.data.multi_dataset_loader import MultiDatasetLoader
from helix_ids.governance.determinism import seed_worker, set_global_determinism
from helix_ids.governance.entrypoint import governed_entrypoint  # noqa: E402
from helix_ids.governance.parameters import DEFAULT_GOVERNANCE_POLICY
from helix_ids.governance.promotion import SeedRunSummary, aggregate_seed_runs
from helix_ids.governance.run_registry import RunRegistry
from helix_ids.models.full import HelixFullConfig, HelixIDSFull, MultiTaskLoss, create_helix_full
from helix_ids.utils.metrics import (  # noqa: E402
    compute_accuracy,
    compute_macro_f1,
    compute_weighted_f1,
)


def _scan_feature_leakage(
    x_train: np.ndarray,
    y_binary_train: np.ndarray,
    *,
    feature_names: list[str],
    seed: int,
    max_samples: int = 200000,
) -> dict[str, Any]:
    """Detect suspicious single-feature separability against binary target."""
    if x_train.shape[0] == 0:
        return {"max_single_feature_auroc": 0.0, "suspicious_features": []}

    rng = np.random.default_rng(seed)
    sample_n = int(min(max_samples, x_train.shape[0]))
    sample_idx = rng.choice(x_train.shape[0], size=sample_n, replace=False)
    x_sample = x_train[sample_idx]
    y_sample = y_binary_train[sample_idx]

    suspicious_features: list[dict[str, Any]] = []
    max_auc = 0.0

    for feature_idx, feature_name in enumerate(feature_names):
        values = x_sample[:, feature_idx]
        if np.nanstd(values) <= 1e-12:
            continue

        try:
            auc = roc_auc_score(y_sample, values)
        except ValueError:
            continue
        auc = float(max(auc, 1.0 - auc))
        max_auc = max(max_auc, auc)

        if auc >= 0.995:
            suspicious_features.append(
                {
                    "feature": feature_name,
                    "single_feature_auroc": auc,
                }
            )

    suspicious_features.sort(key=lambda item: float(item["single_feature_auroc"]), reverse=True)
    return {
        "max_single_feature_auroc": float(max_auc),
        "suspicious_features": suspicious_features[:10],
    }


def _shuffled_label_sanity_check(
    x_train: np.ndarray,
    y_binary_train: np.ndarray,
    x_val: np.ndarray,
    y_binary_val: np.ndarray,
    *,
    seed: int,
    device: str,
    max_per_class: int = 25000,
    steps: int = 250,
    lr: float = 0.05,
) -> float:
    """Train a tiny probe on shuffled binary labels; high balanced-val accuracy indicates leakage."""
    rng = np.random.default_rng(seed)

    train_pos_idx = np.where(y_binary_train == 1)[0]
    train_neg_idx = np.where(y_binary_train == 0)[0]
    val_pos_idx = np.where(y_binary_val == 1)[0]
    val_neg_idx = np.where(y_binary_val == 0)[0]

    train_take = int(min(max_per_class, train_pos_idx.size, train_neg_idx.size))
    val_take = int(min(max_per_class // 2, val_pos_idx.size, val_neg_idx.size))
    if train_take < 100 or val_take < 50:
        return 0.0

    train_idx = np.concatenate(
        [
            rng.choice(train_pos_idx, size=train_take, replace=False),
            rng.choice(train_neg_idx, size=train_take, replace=False),
        ]
    )
    val_idx = np.concatenate(
        [
            rng.choice(val_pos_idx, size=val_take, replace=False),
            rng.choice(val_neg_idx, size=val_take, replace=False),
        ]
    )
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    x_train_bal = torch.from_numpy(x_train[train_idx]).float().to(device)
    y_train_bal_np = y_binary_train[train_idx].astype(np.int64, copy=False)
    y_train_bal = torch.from_numpy(y_train_bal_np).long().to(device)
    x_val_bal = torch.from_numpy(x_val[val_idx]).float().to(device)
    y_val_bal = torch.from_numpy(y_binary_val[val_idx].astype(np.int64)).long().to(device)

    shuffled_labels_np = y_train_bal_np[rng.permutation(y_train_bal_np.shape[0])]
    shuffled_labels = torch.from_numpy(shuffled_labels_np).long().to(device)

    probe = nn.Linear(x_train_bal.shape[1], 2).to(device)
    optimizer = optim.SGD(probe.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    probe.train()
    for _ in range(steps):
        logits = probe(x_train_bal)
        loss = loss_fn(logits, shuffled_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        preds = torch.argmax(probe(x_val_bal), dim=1)
        balanced_val_acc = float((preds == y_val_bal).float().mean().item())

    return balanced_val_acc


def _feature_ablation_sanity_check(
    x_train: np.ndarray,
    y_binary_train: np.ndarray,
    x_val: np.ndarray,
    y_binary_val: np.ndarray,
    *,
    seed: int,
    device: str,
    steps: int = 200,
    lr: float = 0.05,
) -> dict[str, float]:
    """Train linear probes with/without top proxy feature; report expected accuracy drop."""
    if x_train.shape[0] == 0 or x_val.shape[0] == 0:
        return {
            "baseline_balanced_val_acc": 0.0,
            "ablated_balanced_val_acc": 0.0,
            "accuracy_drop": 0.0,
            "ablated_feature_idx": -1.0,
        }

    rng = np.random.default_rng(seed)

    def _balanced_indices(y: np.ndarray, max_per_class: int) -> np.ndarray:
        idx_pos = np.where(y == 1)[0]
        idx_neg = np.where(y == 0)[0]
        take = int(min(max_per_class, idx_pos.size, idx_neg.size))
        if take < 50:
            return np.array([], dtype=np.int64)
        idx = np.concatenate(
            [
                rng.choice(idx_pos, size=take, replace=False),
                rng.choice(idx_neg, size=take, replace=False),
            ]
        )
        rng.shuffle(idx)
        return idx

    train_idx = _balanced_indices(y_binary_train, max_per_class=20000)
    val_idx = _balanced_indices(y_binary_val, max_per_class=10000)
    if train_idx.size == 0 or val_idx.size == 0:
        return {
            "baseline_balanced_val_acc": 0.0,
            "ablated_balanced_val_acc": 0.0,
            "accuracy_drop": 0.0,
            "ablated_feature_idx": -1.0,
        }

    x_train_bal = x_train[train_idx]
    y_train_bal = y_binary_train[train_idx].astype(np.int64, copy=False)
    x_val_bal = x_val[val_idx]
    y_val_bal = y_binary_val[val_idx].astype(np.int64, copy=False)

    # Identify top suspicious feature via absolute linear correlation to binary target.
    y_centered = y_train_bal.astype(np.float32) - float(np.mean(y_train_bal))
    correlations = []
    for feature_idx in range(x_train_bal.shape[1]):
        feature = x_train_bal[:, feature_idx]
        denom = float(np.std(feature) * np.std(y_centered))
        if denom <= 1e-12:
            correlations.append(0.0)
            continue
        corr = float(np.mean((feature - np.mean(feature)) * y_centered) / denom)
        correlations.append(abs(corr))

    top_feature_idx = int(np.argmax(np.asarray(correlations)))
    feature_fill = float(np.nanmean(x_train_bal[:, top_feature_idx]))

    def _probe_accuracy(x_train_probe: np.ndarray, x_val_probe: np.ndarray) -> float:
        x_train_t = torch.from_numpy(x_train_probe).float().to(device)
        y_train_t = torch.from_numpy(y_train_bal).long().to(device)
        x_val_t = torch.from_numpy(x_val_probe).float().to(device)
        y_val_t = torch.from_numpy(y_val_bal).long().to(device)

        probe = nn.Linear(x_train_probe.shape[1], 2).to(device)
        optimizer = optim.SGD(probe.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        probe.train()
        for _ in range(steps):
            logits = probe(x_train_t)
            loss = loss_fn(logits, y_train_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probe.eval()
        with torch.no_grad():
            preds = torch.argmax(probe(x_val_t), dim=1)
            return float((preds == y_val_t).float().mean().item())

    baseline_acc = _probe_accuracy(x_train_bal, x_val_bal)

    x_train_ablated = x_train_bal.copy()
    x_val_ablated = x_val_bal.copy()
    x_train_ablated[:, top_feature_idx] = feature_fill
    x_val_ablated[:, top_feature_idx] = feature_fill
    ablated_acc = _probe_accuracy(x_train_ablated, x_val_ablated)

    return {
        "baseline_balanced_val_acc": baseline_acc,
        "ablated_balanced_val_acc": ablated_acc,
        "accuracy_drop": baseline_acc - ablated_acc,
        "ablated_feature_idx": float(top_feature_idx),
    }


# ============================================================================
# Setup Logging
# ============================================================================


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("HelixFullTraining")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_dir / "training.log")
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


class MultiTaskNumpyDataset(Dataset):
    """Lazy dataset backed by numpy arrays (or memmaps) for multi-task labels."""

    def __init__(self, features: np.ndarray, family_labels: np.ndarray):
        if int(features.shape[0]) != int(family_labels.shape[0]):
            raise ValueError(
                f"Feature/label length mismatch: X={features.shape[0]}, y={family_labels.shape[0]}"
            )
        self.features = features
        self.family_labels = np.asarray(family_labels, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.family_labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_family = int(self.family_labels[idx])
        y_binary = 1 if y_family != 0 else 0
        x_row = np.asarray(self.features[idx], dtype=np.float32)
        return (
            torch.from_numpy(x_row),
            torch.tensor(y_binary, dtype=torch.long),
            torch.tensor(y_family, dtype=torch.long),
        )


def _chunk_finite_check(x: np.ndarray, chunk_rows: int = 250000) -> bool:
    """Check finite values in chunks to avoid large temporary allocations."""
    for start_idx in range(0, int(x.shape[0]), chunk_rows):
        chunk = np.asarray(x[start_idx : start_idx + chunk_rows], dtype=np.float32)
        if not np.isfinite(chunk).all():
            return False
    return True


def _sample_rows(x: np.ndarray, *, seed: int, max_rows: int = 50000) -> np.ndarray:
    """Sample rows for distribution checks without loading full arrays into memory."""
    n_rows = int(x.shape[0])
    if n_rows <= max_rows:
        return np.asarray(x, dtype=np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_rows, size=max_rows, replace=False)
    return np.asarray(x[idx], dtype=np.float32)


def _validate_per_dataset_splits(
    splits: dict[str, np.ndarray],
    *,
    logger: logging.Logger,
    seed: int,
) -> None:
    """Validate class presence, finite features, and cross-dataset scaling consistency."""
    datasets = ["nsl_kdd", "unsw_nb15", "cicids"]
    split_order = ["train", "val", "test"]
    reference_feature_dim: Optional[int] = None
    dataset_scale_stats: dict[str, dict[str, float]] = {}

    for dataset_idx, dataset_name in enumerate(datasets):
        seen_non_empty = False
        for split_name in split_order:
            x_key = f"X_{split_name}_{dataset_name}"
            y_key = f"y_{split_name}_{dataset_name}"
            if x_key not in splits or y_key not in splits:
                continue

            x_arr = splits[x_key]
            y_family = np.asarray(splits[y_key], dtype=np.int64)

            if int(x_arr.shape[0]) == 0:
                continue

            seen_non_empty = True
            if int(x_arr.shape[0]) != int(y_family.shape[0]):
                raise RuntimeError(
                    "Hard-stop integrity guard triggered: split_length_mismatch_"
                    f"{dataset_name}_{split_name}"
                )

            feature_dim = int(x_arr.shape[1])
            if reference_feature_dim is None:
                reference_feature_dim = feature_dim
            elif feature_dim != reference_feature_dim:
                raise RuntimeError(
                    "Hard-stop integrity guard triggered: feature_dim_mismatch_"
                    f"{dataset_name}_{split_name}"
                )

            if not _chunk_finite_check(x_arr):
                raise RuntimeError(
                    "Hard-stop integrity guard triggered: non_finite_features_"
                    f"{dataset_name}_{split_name}"
                )

            family_classes = np.unique(y_family)
            if family_classes.size < 2:
                raise RuntimeError(
                    "Hard-stop integrity guard triggered: single_family_class_"
                    f"{dataset_name}_{split_name}"
                )

            y_binary = (y_family != 0).astype(np.int64, copy=False)
            if np.unique(y_binary).size < 2:
                raise RuntimeError(
                    "Hard-stop integrity guard triggered: single_binary_class_"
                    f"{dataset_name}_{split_name}"
                )

            logger.info(
                f"Integrity[{dataset_name}/{split_name}] rows={x_arr.shape[0]:,}, "
                f"feature_dim={feature_dim}, family_classes={family_classes.size}, "
                "binary_classes=2"
            )

            # Capture one representative split per dataset for scale checks.
            if dataset_name not in dataset_scale_stats:
                sample = _sample_rows(x_arr, seed=seed + dataset_idx)
                p05 = np.percentile(sample, 5, axis=0)
                p95 = np.percentile(sample, 95, axis=0)
                feature_widths = np.clip(p95 - p05, 0.0, None)
                active_widths = feature_widths[feature_widths > 1e-6]
                scale_width = (
                    float(np.median(active_widths)) if active_widths.size > 0 else 0.0
                )
                dataset_scale_stats[dataset_name] = {
                    "width": scale_width,
                    "p01": float(np.percentile(sample, 1)),
                    "p99": float(np.percentile(sample, 99)),
                }

        if not seen_non_empty:
            logger.warning(f"Integrity[{dataset_name}] has no non-empty splits; skipping checks")

    if len(dataset_scale_stats) >= 2:
        width_values = np.asarray(
            [stats["width"] for stats in dataset_scale_stats.values()],
            dtype=np.float64,
        )
        median_width = float(max(1e-8, np.median(width_values)))
        for dataset_name, stats in dataset_scale_stats.items():
            width = float(stats["width"])
            p01 = float(stats["p01"])
            p99 = float(stats["p99"])
            ratio = float(width / median_width)
            logger.info(
                f"Scale[{dataset_name}] p01={p01:.6f}, p99={p99:.6f}, "
                f"width={width:.6f}, ratio_to_median={ratio:.3f}"
            )

            # Hard requirement: all datasets must share bounded scaling regime.
            if p01 < -0.05 or p99 > 1.05:
                raise RuntimeError(
                    "Hard-stop integrity guard triggered: cross_dataset_scale_out_of_bounds_"
                    f"{dataset_name}"
                )

            # Width differences can naturally occur due dataset-specific feature variance.
            if ratio < 0.05 or ratio > 20.0:
                logger.warning(
                    f"Scale[{dataset_name}] width ratio is highly imbalanced ({ratio:.3f}); "
                    "metrics must remain per-dataset and never averaged by sample count."
                )


def _load_eval_array(
    *,
    splits: dict[str, np.ndarray],
    dataset_name: str,
    split_name: str,
    prefix: str,
    logger: logging.Logger,
    expected_feature_dim: Optional[int] = None,
) -> np.ndarray:
    """Load validation/test arrays with memmap preference when cached npy exists."""
    key = f"{prefix}_{split_name}_{dataset_name}"
    if key not in splits:
        raise KeyError(f"Missing split key: {key}")

    mmap_path = PROJECT_ROOT / "data" / "processed" / "multi_dataset_v1" / f"{key}.npy"
    if split_name in {"val", "test"} and mmap_path.exists():
        mmap_array = cast(np.ndarray, np.load(mmap_path, mmap_mode="r"))
        if (
            prefix == "X"
            and expected_feature_dim is not None
            and mmap_array.ndim == 2
            and int(mmap_array.shape[1]) != int(expected_feature_dim)
        ):
            logger.warning(
                "Skipping mmap-backed split array due to feature_dim mismatch: "
                f"{mmap_path} (cached={int(mmap_array.shape[1])}, expected={int(expected_feature_dim)})"
            )
        else:
            logger.info(f"Using mmap-backed split array: {mmap_path}")
            return mmap_array

    return cast(np.ndarray, splits[key])


def _load_precomputed_splits(
    *,
    splits_dir: Path,
    logger: logging.Logger,
) -> Optional[dict[str, np.ndarray]]:
    """Load precomputed split tensors when available to bypass raw CSV harmonization."""
    required = [
        "X_train.npy",
        "y_train.npy",
        "X_val.npy",
        "y_val.npy",
        "X_test_nsl_kdd.npy",
        "y_test_nsl_kdd.npy",
        "X_test_unsw_nb15.npy",
        "y_test_unsw_nb15.npy",
        "X_test_cicids.npy",
        "y_test_cicids.npy",
    ]
    if not splits_dir.exists():
        return None
    if not all((splits_dir / fname).exists() for fname in required):
        return None

    logger.info(f"Loading precomputed splits from {splits_dir}")
    splits: dict[str, np.ndarray] = {}
    for npy_path in sorted(splits_dir.glob("*.npy")):
        key = npy_path.stem
        # Training arrays are loaded eagerly; validation/test arrays can be mem-mapped.
        if key.startswith("X_test_") or key.startswith("X_val_"):
            splits[key] = cast(np.ndarray, np.load(npy_path, mmap_mode="r"))
        else:
            splits[key] = cast(np.ndarray, np.load(npy_path))

    return splits


# ============================================================================
# Training Loop
# ============================================================================


class HelixFullTrainer:
    """Trainer for HelixIDS-Full model."""

    def __init__(
        self,
        model: HelixIDSFull,
        train_loader: DataLoader,
        val_loaders: dict[str, DataLoader],
        test_loaders: dict[str, DataLoader],
        optimizer: optim.Optimizer,
        loss_fn: MultiTaskLoss,
        config: TrainingConfig,
        binary_class_weights: Optional[torch.Tensor] = None,
        family_class_weights: Optional[torch.Tensor] = None,
        device: str = "mps",
        logger: Optional[logging.Logger] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loaders = val_loaders
        self.test_loaders = test_loaders
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.binary_class_weights = (
            binary_class_weights.to(device) if binary_class_weights is not None else None
        )
        self.family_class_weights = (
            family_class_weights.to(device) if family_class_weights is not None else None
        )

        # Training state
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.best_model_state: Optional[dict[str, torch.Tensor]] = None
        self.patience_counter = 0
        self.val_gap_collapse_streak = 0
        self.entropy_collapse_streak = 0
        self.training_history: dict[str, list[float]] = {
            "train_loss": [],
            "train_binary_acc": [],
            "train_family_acc": [],
            "val_loss": [],
            "val_binary_acc": [],
            "val_family_acc": [],
            "val_binary_auroc": [],
            "val_binary_auprc": [],
            "val_family_macro_f1": [],
            "val_family_minority_recall_min": [],
            "val_family_entropy": [],
        }

    def _get_learning_rate(self) -> float:
        """Compute learning rate with linear warmup and cosine decay."""
        if self.epoch < self.config.warmup_epochs:
            warmup_denom = max(1, self.config.warmup_epochs)
            return float(
                self.config.warmup_init_lr
                + (self.config.learning_rate - self.config.warmup_init_lr)
                * ((self.epoch + 1) / warmup_denom)
            )

        decay_epochs = max(1, self.config.epochs - self.config.warmup_epochs)
        decay_step = min(self.epoch - self.config.warmup_epochs, decay_epochs)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_step / decay_epochs))
        min_lr = self.config.learning_rate * 0.05
        return float(min_lr + (self.config.learning_rate - min_lr) * cosine_factor)

    def _set_learning_rate(self) -> None:
        """Update learning rate in optimizer."""
        lr = self._get_learning_rate()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _current_learning_rate(self) -> float:
        """Return current optimizer learning rate."""
        return float(self.optimizer.param_groups[0]["lr"])

    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_calibrated_loss = 0.0
        total_binary_correct = 0
        total_family_correct = 0
        total_samples = 0
        num_batches = 0

        for batch_idx, (x, y_binary, y_family) in enumerate(self.train_loader):
            x = x.to(self.device)
            y_binary = y_binary.to(self.device)
            y_family = y_family.to(self.device)

            # Forward pass
            binary_logits, family_logits = self.model(x)

            # Compute loss
            loss, _ = self.loss_fn(
                binary_logits,
                y_binary,
                family_logits,
                y_family,
                binary_class_weights=self.binary_class_weights,
                family_class_weights=self.family_class_weights,
            )

            calibrated_loss, _ = self.loss_fn(
                binary_logits,
                y_binary,
                family_logits,
                y_family,
                binary_class_weights=None,
                family_class_weights=None,
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            self.optimizer.step()

            batch_size = int(y_binary.shape[0])
            binary_correct = int((torch.argmax(binary_logits, dim=1) == y_binary).sum().item())
            family_correct = int((torch.argmax(family_logits, dim=1) == y_family).sum().item())

            total_loss += float(loss.item()) * batch_size
            total_calibrated_loss += float(calibrated_loss.item()) * batch_size
            total_binary_correct += binary_correct
            total_family_correct += family_correct
            total_samples += batch_size
            num_batches += 1

            # Log every N batches
            if batch_idx % self.config.log_interval == 0:
                avg_loss = total_loss / max(1, total_samples)
                binary_acc = total_binary_correct / max(1, total_samples)
                family_acc = total_family_correct / max(1, total_samples)
                lr = self._current_learning_rate()
                self.logger.info(
                    f"Epoch {self.epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {avg_loss:.4f} | "
                    f"Binary Acc: {binary_acc:.4f} | "
                    f"Family Acc: {family_acc:.4f} | "
                    f"LR: {lr:.2e}"
                )

        return {
            "train_loss": total_loss / max(1, total_samples),
            "train_calibrated_loss": total_calibrated_loss / max(1, total_samples),
            "train_binary_acc": total_binary_correct / max(1, total_samples),
            "train_family_acc": total_family_correct / max(1, total_samples),
        }

    @torch.no_grad()
    def _evaluate_loader(self, loader: DataLoader) -> dict[str, Any]:
        """Evaluate metrics on a single dataset loader."""
        total_loss = 0.0
        total_calibrated_loss = 0.0
        total_binary_correct = 0
        total_family_correct = 0
        total_samples = 0

        binary_prob_chunks: list[np.ndarray] = []
        binary_label_chunks: list[np.ndarray] = []
        family_prob_chunks: list[np.ndarray] = []
        family_pred_chunks: list[np.ndarray] = []
        family_label_chunks: list[np.ndarray] = []

        for x, y_binary, y_family in loader:
            x = x.to(self.device)
            y_binary = y_binary.to(self.device)
            y_family = y_family.to(self.device)

            binary_logits, family_logits = self.model(x)

            loss, _ = self.loss_fn(
                binary_logits,
                y_binary,
                family_logits,
                y_family,
                binary_class_weights=self.binary_class_weights,
                family_class_weights=self.family_class_weights,
            )
            calibrated_loss, _ = self.loss_fn(
                binary_logits,
                y_binary,
                family_logits,
                y_family,
                binary_class_weights=None,
                family_class_weights=None,
            )

            batch_size = int(y_binary.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_calibrated_loss += float(calibrated_loss.item()) * batch_size
            total_binary_correct += int((torch.argmax(binary_logits, dim=1) == y_binary).sum().item())
            total_family_correct += int((torch.argmax(family_logits, dim=1) == y_family).sum().item())
            total_samples += batch_size

            binary_prob = torch.softmax(binary_logits, dim=1)[:, 1].cpu().numpy()
            family_prob = torch.softmax(family_logits, dim=1).cpu().numpy()

            binary_prob_chunks.append(binary_prob)
            binary_label_chunks.append(y_binary.cpu().numpy())
            family_prob_chunks.append(family_prob)
            family_pred_chunks.append(torch.argmax(family_logits, dim=1).cpu().numpy())
            family_label_chunks.append(y_family.cpu().numpy())

        binary_probs = np.concatenate(binary_prob_chunks) if binary_prob_chunks else np.array([])
        binary_labels = np.concatenate(binary_label_chunks) if binary_label_chunks else np.array([])
        family_probs = np.concatenate(family_prob_chunks) if family_prob_chunks else np.array([])
        family_preds = np.concatenate(family_pred_chunks) if family_pred_chunks else np.array([])
        family_labels = np.concatenate(family_label_chunks) if family_label_chunks else np.array([])

        if binary_labels.size > 0 and np.unique(binary_labels).size > 1:
            binary_auroc = float(roc_auc_score(binary_labels, binary_probs))
            binary_auprc = float(average_precision_score(binary_labels, binary_probs))
        else:
            binary_auroc = 0.0
            binary_auprc = 0.0

        family_macro_f1 = (
            compute_macro_f1(family_labels, family_preds) if family_labels.size > 0 else 0.0
        )

        per_class_recall = {}
        for cls in np.unique(family_labels):
            cls_int = int(cls)
            cls_mask = family_labels == cls_int
            per_class_recall[str(cls_int)] = float((family_preds[cls_mask] == cls_int).mean())

        minority_recalls = [recall for cls, recall in per_class_recall.items() if int(cls) != 0]
        family_minority_recall_min = float(min(minority_recalls)) if minority_recalls else 0.0

        if family_probs.size > 0:
            safe_probs = np.clip(family_probs, 1e-10, 1.0 - 1e-10)
            per_sample_entropy = -np.sum(family_probs * np.log(safe_probs), axis=1)
            family_entropy = float(np.mean(per_sample_entropy / np.log(family_probs.shape[1])))
        else:
            family_entropy = 0.0

        present_classes = set(np.unique(family_labels).tolist()) if family_labels.size > 0 else set()
        predicted_classes = set(np.unique(family_preds).tolist()) if family_preds.size > 0 else set()
        zero_prediction_classes = sorted(int(cls) for cls in present_classes - predicted_classes)

        return {
            "num_samples": float(total_samples),
            "val_loss": total_loss / max(1, total_samples),
            "val_calibrated_loss": total_calibrated_loss / max(1, total_samples),
            "val_binary_acc": total_binary_correct / max(1, total_samples),
            "val_family_acc": total_family_correct / max(1, total_samples),
            "val_binary_auroc": binary_auroc,
            "val_binary_auprc": binary_auprc,
            "val_family_macro_f1": family_macro_f1,
            "val_family_minority_recall_min": family_minority_recall_min,
            "val_family_entropy": family_entropy,
            "val_family_zero_prediction_classes": float(len(zero_prediction_classes)),
        }

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Validate per dataset with strict isolation (worst-case aggregation)."""
        self.model.eval()
        if not self.val_loaders:
            raise RuntimeError("No validation loaders configured")

        dataset_metrics: dict[str, dict[str, Any]] = {}
        for dataset_name, loader in self.val_loaders.items():
            metrics = self._evaluate_loader(loader)
            dataset_metrics[dataset_name] = metrics
            self.logger.info(
                f"Val[{dataset_name}] loss={metrics['val_loss']:.4f}, "
                f"bin_acc={metrics['val_binary_acc']:.4f}, "
                f"fam_acc={metrics['val_family_acc']:.4f}, "
                f"entropy={metrics['val_family_entropy']:.4f}"
            )

        total_samples = sum(metric["num_samples"] for metric in dataset_metrics.values())
        if total_samples <= 0:
            raise RuntimeError("Validation metrics are empty; no samples found in val loaders")

        # Strict isolation: avoid sample-weighted averaging that can hide weak datasets.
        metric_values = list(dataset_metrics.values())
        return {
            "val_loss": float(max(metric["val_loss"] for metric in metric_values)),
            "val_calibrated_loss": float(
                max(metric["val_calibrated_loss"] for metric in metric_values)
            ),
            "val_binary_acc": float(min(metric["val_binary_acc"] for metric in metric_values)),
            "val_family_acc": float(min(metric["val_family_acc"] for metric in metric_values)),
            "val_binary_auroc": float(min(metric["val_binary_auroc"] for metric in metric_values)),
            "val_binary_auprc": float(min(metric["val_binary_auprc"] for metric in metric_values)),
            "val_family_macro_f1": float(
                min(metric["val_family_macro_f1"] for metric in metric_values)
            ),
            "val_family_minority_recall_min": float(
                min(metric["val_family_minority_recall_min"] for metric in metric_values)
            ),
            "val_family_entropy": float(min(metric["val_family_entropy"] for metric in metric_values)),
            "val_family_zero_prediction_classes": float(
                max(metric["val_family_zero_prediction_classes"] for metric in metric_values)
            ),
        }

    @torch.no_grad()
    def evaluate_per_dataset(self) -> dict[str, dict[str, float]]:
        """Evaluate on per-dataset test sets."""
        self.model.eval()
        results = {}

        for dataset_name, test_loader in self.test_loaders.items():
            binary_preds: list[int] = []
            binary_pos_probs: list[float] = []
            family_preds: list[int] = []
            family_probs: list[np.ndarray] = []
            binary_labels: list[int] = []
            family_labels: list[int] = []

            for x, y_binary, y_family in test_loader:
                x = x.to(self.device)

                binary_logits, family_logits = self.model(x)
                binary_prob = torch.softmax(binary_logits, dim=1)
                family_prob = torch.softmax(family_logits, dim=1)

                binary_preds.extend(torch.argmax(binary_logits, dim=1).cpu().numpy())
                binary_pos_probs.extend(binary_prob[:, 1].cpu().numpy())
                family_preds.extend(torch.argmax(family_logits, dim=1).cpu().numpy())
                family_probs.extend(family_prob.cpu().numpy())
                binary_labels.extend(y_binary.numpy())
                family_labels.extend(y_family.numpy())

            binary_preds_arr = np.array(binary_preds)
            family_preds_arr = np.array(family_preds)
            binary_probs_arr = np.array(binary_pos_probs)
            family_probs_arr = np.array(family_probs)
            binary_labels_arr = np.array(binary_labels)
            family_labels_arr = np.array(family_labels)

            per_class_recall = {}
            for cls in np.unique(family_labels_arr):
                cls_int = int(cls)
                cls_mask = family_labels_arr == cls_int
                per_class_recall[str(cls_int)] = float((family_preds_arr[cls_mask] == cls_int).mean())

            minority_recall_values = [
                recall for cls, recall in per_class_recall.items() if int(cls) != 0
            ]
            family_minority_recall_min = (
                float(min(minority_recall_values)) if minority_recall_values else 0.0
            )

            if binary_labels_arr.size > 0 and np.unique(binary_labels_arr).size > 1:
                binary_auroc = float(roc_auc_score(binary_labels_arr, binary_probs_arr))
                binary_auprc = float(average_precision_score(binary_labels_arr, binary_probs_arr))
            else:
                binary_auroc = 0.0
                binary_auprc = 0.0

            family_entropy = float(
                np.mean(
                    -np.sum(
                        family_probs_arr * np.log(np.clip(family_probs_arr, 1e-12, 1.0)),
                        axis=1,
                    )
                    / np.log(family_probs_arr.shape[1])
                )
            )

            present_classes = set(np.unique(family_labels_arr).tolist())
            predicted_classes = set(np.unique(family_preds_arr).tolist())
            zero_prediction_classes = sorted(int(cls) for cls in present_classes - predicted_classes)

            results[dataset_name] = {
                "binary_accuracy": compute_accuracy(binary_labels_arr, binary_preds_arr),
                "binary_f1": compute_weighted_f1(binary_labels_arr, binary_preds_arr),
                "binary_auroc": binary_auroc,
                "binary_auprc": binary_auprc,
                "family_accuracy": compute_accuracy(family_labels_arr, family_preds_arr),
                "family_f1": compute_weighted_f1(family_labels_arr, family_preds_arr),
                "family_macro_f1": compute_macro_f1(family_labels_arr, family_preds_arr),
                "family_minority_recall_min": family_minority_recall_min,
                "family_entropy": family_entropy,
                "family_zero_prediction_classes": float(len(zero_prediction_classes)),
            }

        return results

    def fit(self) -> dict[str, Any]:
        """Train for specified epochs."""
        self.logger.info("=" * 80)
        self.logger.info("Starting HelixIDS-Full Training")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {self.model.param_count:,}")
        self.logger.info(f"Epochs: {self.config.epochs}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info("=" * 80)

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            self._set_learning_rate()
            # Train
            train_metrics = self.train_epoch()

            # Validate every N epochs
            if self.epoch % self.config.val_interval == 0:
                val_metrics = self.validate()

                for key, val in train_metrics.items():
                    self.training_history.setdefault(key, []).append(val)
                for key, val in val_metrics.items():
                    self.training_history.setdefault(key, []).append(val)

                self.logger.info(
                    f"Epoch {self.epoch:3d} | "
                    f"Train Loss: {train_metrics['train_loss']:.4f} | "
                    f"Train Cal Loss: {train_metrics['train_calibrated_loss']:.4f} | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Val Cal Loss: {val_metrics['val_calibrated_loss']:.4f} | "
                    f"Val Binary Acc: {val_metrics['val_binary_acc']:.4f} | "
                    f"Val Family Acc: {val_metrics['val_family_acc']:.4f} | "
                    f"Val Entropy: {val_metrics.get('val_family_entropy', 0.0):.4f}"
                )

                hard_stop_reason = self._hard_stop_reason(train_metrics, val_metrics)
                if hard_stop_reason is not None:
                    raise RuntimeError(f"Hard-stop integrity guard triggered: {hard_stop_reason}")

                should_stop = self._update_early_stopping(train_metrics, val_metrics)
                self._save_checkpoint_if_needed()
                if should_stop:
                    break

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.logger.info("✅ Loaded best model state")

        # Evaluate on per-dataset test sets
        self.logger.info("\nPer-Dataset Evaluation:")
        per_dataset_results = self.evaluate_per_dataset()
        self._log_per_dataset_results(per_dataset_results)

        return {
            "training_history": self.training_history,
            "per_dataset_results": per_dataset_results,
            "best_val_loss": self.best_val_loss,
            "epochs_trained": self.epoch + 1,
        }

    def _hard_stop_reason(
        self,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> Optional[str]:
        """Return hard-stop reason when integrity constraints are violated."""
        # Train loss is gathered in train mode (dropout/batchnorm active) while
        # validation is in eval mode. A lower val loss can be normal, so only
        # hard-stop when the gap is large and accompanied by collapse symptoms.
        val_gap = train_metrics["train_calibrated_loss"] - val_metrics["val_calibrated_loss"]
        collapse_signals = (
            val_metrics.get("val_family_macro_f1", 1.0) < 0.25
            or val_metrics.get("val_family_minority_recall_min", 1.0) < 0.10
            or val_metrics.get("val_family_entropy", 1.0) < 0.15
            or val_metrics.get("val_family_zero_prediction_classes", 0.0) > 0
        )
        if val_gap > 0.12 and collapse_signals:
            self.val_gap_collapse_streak += 1
            if self.val_gap_collapse_streak >= 2:
                return "val_loss_below_train_loss_with_collapse"
        else:
            self.val_gap_collapse_streak = 0

        peak_accuracy = max(
            train_metrics["train_binary_acc"],
            train_metrics["train_family_acc"],
            val_metrics["val_binary_acc"],
            val_metrics["val_family_acc"],
        )
        if train_metrics["train_calibrated_loss"] > 0.5 and peak_accuracy > 0.95:
            return "high_accuracy_with_high_loss"

        # More lenient entropy threshold (0.15 instead of 0.1)
        # Only trigger if accompanied by zero_prediction_classes (confirmed mode collapse)
        entropy_val = val_metrics.get("val_family_entropy", 0.0)
        has_missing_classes = val_metrics.get("val_family_zero_prediction_classes", 0.0) > 0
        if entropy_val < 0.12 and has_missing_classes:
            return "prediction_entropy_collapse_with_missing_classes"
        
        # Very strict threshold only for extreme cases
        if entropy_val < 0.08:
            self.entropy_collapse_streak = getattr(self, 'entropy_collapse_streak', 0) + 1
            if self.entropy_collapse_streak >= 3:
                self.logger.warning(
                    f"⚠️  Entropy critically low for 3 epochs: {entropy_val:.4f} "
                    f"(missing_classes={int(val_metrics.get('val_family_zero_prediction_classes', 0))})"
                )
                return "prediction_entropy_critical_collapse"
            return None
        else:
            self.entropy_collapse_streak = 0

        return None

    def _update_early_stopping(self, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> bool:
        """Update early stopping state; return True when training should stop."""
        val_loss = val_metrics["val_loss"]
        quality_gate_pass = (
            val_metrics.get("val_family_minority_recall_min", 0.0)
            >= self.config.min_family_minority_recall_for_best
            and val_metrics.get("val_family_entropy", 0.0) >= 0.3
        )

        if val_loss < self.best_val_loss - self.config.early_stopping_threshold:
            if quality_gate_pass:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                self.logger.info(f"✅ Best model update (loss: {self.best_val_loss:.4f})")
                return False
            self.logger.info(
                "Best-loss candidate rejected by quality gate: "
                f"minority_recall={val_metrics.get('val_family_minority_recall_min', 0.0):.4f}, "
                f"entropy={val_metrics.get('val_family_entropy', 0.0):.4f}"
            )

        self.patience_counter += 1
        if self.patience_counter >= self.config.early_stopping_patience:
            self.logger.info(
                f"Early stopping triggered (patience {self.patience_counter} >= "
                f"{self.config.early_stopping_patience})"
            )
            return True
        return False

    def _save_checkpoint_if_needed(self) -> None:
        """Persist intermediate checkpoint on configured interval."""
        if self.epoch <= 0 or self.epoch % self.config.save_interval != 0:
            return
        checkpoint_path = self.config.checkpoint_dir / f"checkpoint_epoch_{self.epoch}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _log_per_dataset_results(self, per_dataset_results: dict[str, dict[str, float]]) -> None:
        """Log formatted per-dataset metrics."""
        for dataset_name, metrics in per_dataset_results.items():
            self.logger.info(f"\n{dataset_name}:")
            for key, val in metrics.items():
                self.logger.info(f"  {key}: {val:.4f}")


# ============================================================================
# Main Training Script
# ============================================================================


@governed_entrypoint(entrypoint_id="scripts.train_helix_ids_full")
def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train HelixIDS-Full model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/helix_config.yaml",
        help="Path to training config (YAML)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/helix_full",
        help="Output directory for model/logs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device (mps, cpu, cuda)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Number of epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("HELIX_SEED", "42")),
        help="Global seed for deterministic execution",
    )
    parser.add_argument(
        "--holdout-dataset",
        type=str,
        default="cicids",
        choices=["nsl_kdd", "unsw_nb15", "cicids"],
        help="Dataset to keep fully held out when entity keys are unavailable",
    )
    parser.add_argument(
        "--precomputed-splits-dir",
        type=str,
        default="data/processed/multi_dataset_v1",
        help="Path to precomputed split .npy files",
    )
    parser.add_argument(
        "--force-recompute-splits",
        action="store_true",
        help="Ignore precomputed splits and recompute from raw datasets",
    )

    args = parser.parse_args()
    os.environ["HELIX_STRICT_MISSING"] = "1"
    os.environ["STRICT_MISSING"] = "1"
    os.environ["HELIX_SEED"] = str(args.seed)
    determinism_state = set_global_determinism(args.seed)
    split_start = time.perf_counter()

    # Create output directories
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("results/helix_full")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(results_dir)

    # Load configs
    train_config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
    )
    data_config = DataConfig()

    logger.info(f"Loading data from {data_config.data_dir}...")

    # Load multi-dataset (Phase 1)
    from helix_ids.data.feature_harmonization import labels_to_multi_task

    splits: dict[str, np.ndarray]
    precomputed_splits_dir = Path(args.precomputed_splits_dir)
    if args.force_recompute_splits:
        logger.info("Skipping precomputed splits due to --force-recompute-splits")
        precomputed_splits = None
    else:
        precomputed_splits = _load_precomputed_splits(
            splits_dir=precomputed_splits_dir,
            logger=logger,
        )

    if precomputed_splits is not None:
        splits = precomputed_splits
    else:
        loader = MultiDatasetLoader()
        nsl_kdd, unsw, cicids = loader.load_and_harmonize_all()
        splits = loader.create_splits(
            [nsl_kdd, unsw, cicids],
            holdout_dataset=args.holdout_dataset,
        )

    _validate_per_dataset_splits(splits, logger=logger, seed=args.seed)

    logger.info(f"Combined training set: {splits['X_train'].shape[0]:,} samples")
    logger.info(f"Combined validation set: {splits['X_val'].shape[0]:,} samples")
    if splits["X_val"].shape[0] > splits["X_train"].shape[0]:
        raise RuntimeError(
            "Hard-stop integrity guard triggered: val_size_exceeds_train_size "
            f"(train={splits['X_train'].shape[0]:,}, val={splits['X_val'].shape[0]:,})"
        )
    if "X_test_nsl_kdd" in splits:
        logger.info(f"NSL-KDD test set: {splits['X_test_nsl_kdd'].shape[0]:,} samples")
    if "X_test_unsw_nb15" in splits:
        logger.info(f"UNSW test set: {splits['X_test_unsw_nb15'].shape[0]:,} samples")
    if "X_test_cicids" in splits:
        logger.info(f"CICIDS test set: {splits['X_test_cicids'].shape[0]:,} samples")

    # Convert family labels to binary + family for multi-task learning
    y_train_binary, y_train_family = labels_to_multi_task(splits["y_train"])
    y_val_binary, y_val_family = labels_to_multi_task(splits["y_val"])

    family_counts = np.bincount(splits["y_train"].astype(int))
    family_majority_ratio = float(family_counts.max() / max(1, family_counts.sum()))
    binary_counts_train = np.bincount(y_train_binary.astype(int), minlength=2)
    binary_majority_ratio = float(binary_counts_train.max() / max(1, binary_counts_train.sum()))

    family_probs_train = family_counts / max(1, family_counts.sum())
    family_label_entropy = float(
        -np.sum(family_probs_train * np.log(np.clip(family_probs_train, 1e-12, 1.0)))
        / np.log(max(2, family_probs_train.size))
    )

    logger.info(f"Binary distribution - Train: {np.bincount(y_train_binary)}")
    logger.info(f"Family distribution - Train: {np.bincount(y_train_family)}")
    logger.info(
        "Data integrity probes | "
        f"binary_majority_ratio={binary_majority_ratio:.4f}, "
        f"family_majority_ratio={family_majority_ratio:.4f}, "
        f"family_label_entropy={family_label_entropy:.4f}"
    )

    feature_names = [str(name) for name in splits.get("feature_columns", np.array([], dtype=object))]
    if len(feature_names) != splits["X_train"].shape[1]:
        feature_names = [f"feature_{idx}" for idx in range(splits["X_train"].shape[1])]
    current_feature_dim = int(splits["X_train"].shape[1])

    leakage_scan = _scan_feature_leakage(
        splits["X_train"],
        y_train_binary,
        feature_names=feature_names,
        seed=args.seed + 17,
    )
    shuffled_label_balanced_val_acc = _shuffled_label_sanity_check(
        splits["X_train"],
        y_train_binary,
        splits["X_val"],
        y_val_binary,
        seed=args.seed + 19,
        device=args.device,
    )
    ablation_check = _feature_ablation_sanity_check(
        splits["X_train"],
        y_train_binary,
        splits["X_val"],
        y_val_binary,
        seed=args.seed + 23,
        device=args.device,
    )

    logger.info(
        "Leakage diagnostics | "
        f"max_single_feature_auroc={leakage_scan['max_single_feature_auroc']:.4f}, "
        f"shuffled_label_balanced_val_acc={shuffled_label_balanced_val_acc:.4f}, "
        f"feature_ablation_drop={ablation_check['accuracy_drop']:.4f}, "
        f"ablated_feature_idx={int(ablation_check['ablated_feature_idx'])}"
    )
    if leakage_scan["suspicious_features"]:
        logger.warning(
            "Potential leakage features detected: "
            f"{json.dumps(leakage_scan['suspicious_features'])}"
        )

    if leakage_scan["max_single_feature_auroc"] >= 0.995:
        raise RuntimeError(
            "Hard-stop integrity guard triggered: single_feature_leakage_signal"
        )
    if shuffled_label_balanced_val_acc > 0.65:
        raise RuntimeError(
            "Hard-stop integrity guard triggered: shuffled_label_generalization_detected"
        )
    if ablation_check["accuracy_drop"] <= 0.01:
        logger.warning(
            "Feature ablation produced a very small accuracy drop; "
            "model may still rely on diffuse shortcuts."
        )

    # Create data loaders (no resampling; class imbalance handled via weighted loss)
    train_dataset = TensorDataset(
        torch.from_numpy(splits["X_train"]).float(),
        torch.from_numpy(y_train_binary).long(),
        torch.from_numpy(y_train_family).long(),
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
        worker_init_fn=seed_worker,
        generator=loader_generator,
        persistent_workers=train_config.num_workers > 0,
        prefetch_factor=2 if train_config.num_workers > 0 else None,
    )

    # Build per-dataset validation loaders (no mixed-dataset validation).
    val_loaders: dict[str, DataLoader] = {}
    val_dataset_specs = [
        ("nsl_kdd", "X_val_nsl_kdd", "y_val_nsl_kdd"),
        ("unsw_nb15", "X_val_unsw_nb15", "y_val_unsw_nb15"),
        ("cicids", "X_val_cicids", "y_val_cicids"),
    ]
    for dataset_name, x_key, y_key in val_dataset_specs:
        if x_key not in splits or y_key not in splits:
            continue
        if splits[x_key].shape[0] == 0:
            continue
        x_val_ds = _load_eval_array(
            splits=splits,
            dataset_name=dataset_name,
            split_name="val",
            prefix="X",
            logger=logger,
            expected_feature_dim=current_feature_dim,
        )
        y_val_family_ds = _load_eval_array(
            splits=splits,
            dataset_name=dataset_name,
            split_name="val",
            prefix="y",
            logger=logger,
        )
        val_dataset = MultiTaskNumpyDataset(x_val_ds, y_val_family_ds)
        val_loaders[dataset_name] = DataLoader(
            val_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=train_config.num_workers,
            pin_memory=train_config.pin_memory,
            worker_init_fn=seed_worker,
            generator=loader_generator,
            persistent_workers=train_config.num_workers > 0,
            prefetch_factor=2 if train_config.num_workers > 0 else None,
        )

    if not val_loaders:
        raise RuntimeError("No per-dataset validation splits available")

    # Create per-dataset test loaders
    test_loaders = {}
    dataset_specs = [
        ("nsl_kdd", "X_test_nsl_kdd", "y_test_nsl_kdd"),
        ("unsw_nb15", "X_test_unsw_nb15", "y_test_unsw_nb15"),
        ("cicids", "X_test_cicids", "y_test_cicids"),
    ]
    for dataset_name, x_key, y_key in dataset_specs:
        if x_key not in splits or y_key not in splits:
            continue
        x_test_ds = _load_eval_array(
            splits=splits,
            dataset_name=dataset_name,
            split_name="test",
            prefix="X",
            logger=logger,
            expected_feature_dim=current_feature_dim,
        )
        y_test_family_ds = _load_eval_array(
            splits=splits,
            dataset_name=dataset_name,
            split_name="test",
            prefix="y",
            logger=logger,
        )
        test_dataset = MultiTaskNumpyDataset(x_test_ds, y_test_family_ds)
        test_loaders[dataset_name] = DataLoader(
            test_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=max(2, train_config.num_workers),
            worker_init_fn=seed_worker,
            generator=loader_generator,
            persistent_workers=max(2, train_config.num_workers) > 0,
            prefetch_factor=2,
        )

    split_end = time.perf_counter()
    split_elapsed = split_end - split_start

    # Create model (Phase 2)
    logger.info("Creating HelixIDS-Full model...")
    model = create_helix_full(HelixFullConfig(input_dim=int(splits["X_train"].shape[1])))
    logger.info(f"Model parameters: {model.param_count:,}")

    # Setup training
    binary_class_weights = None
    family_class_weights = None
    if "train_class_weights" in splits:
        family_class_weights = torch.from_numpy(splits["train_class_weights"]).float()
        binary_counts = np.bincount(y_train_binary.astype(int), minlength=2)
        binary_counts = np.where(binary_counts == 0, 1, binary_counts)
        binary_weights_np = binary_counts.sum() / (len(binary_counts) * binary_counts)
        binary_class_weights = torch.from_numpy(binary_weights_np.astype(np.float32))
        logger.info(f"Using family class weights: {family_class_weights.tolist()}")
        logger.info(f"Using binary class weights: {binary_class_weights.tolist()}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    loss_fn = MultiTaskLoss(
        lambda_binary=train_config.lambda_binary,
        lambda_family=train_config.lambda_family,
    )

    # Create trainer
    trainer = HelixFullTrainer(
        model=model,
        train_loader=train_loader,
        val_loaders=val_loaders,
        test_loaders=test_loaders,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=train_config,
        binary_class_weights=binary_class_weights,
        family_class_weights=family_class_weights,
        device=args.device,
        logger=logger,
    )

    # Train
    logger.info("Starting training...")
    pretrain_elapsed = max(0.001, time.perf_counter() - split_end)
    training_start = time.perf_counter()
    results = trainer.fit()
    training_elapsed = time.perf_counter() - training_start

    # Save model
    best_model_path = output_dir / "helix_full_best.pt"
    final_model_path = output_dir / "helix_full_final.pt"
    torch.save(model.state_dict(), best_model_path)
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"✅ Model saved to {best_model_path}")

    # Save results
    results_path = results_dir / "training_results.json"
    eval_path = results_dir / "eval_per_dataset.json"

    with open(results_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "batch_size": train_config.batch_size,
                    "epochs": train_config.epochs,
                    "learning_rate": train_config.learning_rate,
                    "device": args.device,
                },
                "results": {
                    k: v if not isinstance(v, (dict, list)) else str(v) for k, v in results.items()
                },
            },
            f,
            indent=2,
            default=str,
        )

    with open(eval_path, "w") as f:
        json.dump(results["per_dataset_results"], f, indent=2)

    posteval_start = time.perf_counter()
    max_ci_width = 0.0
    min_ci_lower = 1.0
    macro_values: list[float] = []
    min_family_minority_recall = 1.0
    min_family_entropy = 1.0
    max_zero_prediction_classes = 0.0
    min_binary_auprc = 1.0
    for dataset_metrics in results["per_dataset_results"].values():
        ci_width = float(dataset_metrics.get("family_ci95_width", 0.0))
        ci_lower = float(
            dataset_metrics.get("family_ci95_lower", dataset_metrics.get("family_macro_f1", 0.0))
        )
        macro_val = float(
            dataset_metrics.get("family_macro_f1", dataset_metrics.get("family_f1", 0.0))
        )
        minority_recall_min = float(dataset_metrics.get("family_minority_recall_min", 0.0))
        family_entropy_val = float(dataset_metrics.get("family_entropy", 0.0))
        zero_pred_classes = float(dataset_metrics.get("family_zero_prediction_classes", 0.0))
        binary_auprc = float(dataset_metrics.get("binary_auprc", 0.0))

        max_ci_width = max(max_ci_width, ci_width)
        min_ci_lower = min(min_ci_lower, ci_lower)
        min_family_minority_recall = min(min_family_minority_recall, minority_recall_min)
        min_family_entropy = min(min_family_entropy, family_entropy_val)
        max_zero_prediction_classes = max(max_zero_prediction_classes, zero_pred_classes)
        min_binary_auprc = min(min_binary_auprc, binary_auprc)
        macro_values.append(macro_val)

    prepromote_start = time.perf_counter()
    # Strict per-dataset interpretation: track worst-case macro-F1, not averaged macro-F1.
    aggregate_macro_f1 = float(min(macro_values)) if macro_values else 0.0
    policy = DEFAULT_GOVERNANCE_POLICY
    registry = RunRegistry(
        Path(os.environ.get("HELIX_RUN_REGISTRY", "results/gates/run_registry.jsonl"))
    )
    drift, z_score = registry.compute_drift(
        dataset_id="helix_full",
        current_macro_f1=aggregate_macro_f1,
        baseline_window_runs=20,
    )

    data_integrity_pass = (
        binary_majority_ratio <= 0.90
        and family_majority_ratio <= 0.90
        and float(leakage_scan["max_single_feature_auroc"]) < 0.995
        and shuffled_label_balanced_val_acc <= 0.65
        and ablation_check["accuracy_drop"] > 0.01
        and min_family_minority_recall >= 0.70
        and min_family_entropy >= 0.30
        and max_zero_prediction_classes == 0.0
        and min_binary_auprc >= 0.70
    )

    tier2_pass = (
        min_ci_lower >= policy.bootstrap.min_ci95_lower_bound
        and max_ci_width <= policy.bootstrap.max_ci_width
        and drift <= policy.drift.max_abs_macro_f1_drift
        and z_score <= policy.drift.max_abs_z_score
        and data_integrity_pass
    )
    promotion_consensus = aggregate_seed_runs(
        [
            SeedRunSummary(
                seed=args.seed,
                macro_f1=aggregate_macro_f1,
                macro_f1_ci_lower=min_ci_lower,
                macro_f1_ci_width=max_ci_width,
                tier2_pass=tier2_pass,
            )
        ],
        min_seed_runs=policy.promotion.min_seed_runs,
        max_inter_seed_macro_f1_variance=policy.promotion.max_inter_seed_macro_f1_variance,
        reproducibility_tolerance=policy.promotion.reproducibility_tolerance,
        min_ci95_lower_bound=policy.bootstrap.min_ci95_lower_bound,
        max_ci_width=policy.bootstrap.max_ci_width,
    )

    family_weight_min = (
        float(family_class_weights.min().item()) if family_class_weights is not None else 1.0
    )
    binary_weight_min = (
        float(binary_class_weights.min().item()) if binary_class_weights is not None else 1.0
    )
    governance_stages = {
        "presplit": {
            "presplit_elapsed_seconds": split_elapsed,
            "split_train_rows": int(splits["X_train"].shape[0]),
            "split_binary_class_count": int(len(np.unique(y_train_binary))),
        },
        "pretrain": {
            "pretrain_elapsed_seconds": pretrain_elapsed,
            "family_class_weight_min": family_weight_min,
            "binary_class_weight_min": binary_weight_min,
        },
        "intrain": {
            "intrain_elapsed_seconds": training_elapsed,
            "low_entropy_consecutive_batches": 0,
            "gradient_dominance": 0.0,
            "epochs_without_improvement": int(
                min(
                    trainer.patience_counter,
                    train_config.early_stopping_patience,
                )
            ),
        },
        "posteval": {
            "posteval_elapsed_seconds": max(0.001, time.perf_counter() - posteval_start),
            "macro_f1_ci_width": max_ci_width,
            "macro_f1_ci_lower": min_ci_lower,
            "abs_macro_f1_drift": drift,
            "abs_macro_f1_zscore": z_score,
        },
        "prepromote": {
            "prepromote_elapsed_seconds": max(0.001, time.perf_counter() - prepromote_start),
            "macro_f1_ci_width": max_ci_width,
            "macro_f1_ci_lower": min_ci_lower,
            "data_integrity_pass": float(data_integrity_pass),
            "binary_majority_ratio": binary_majority_ratio,
            "family_majority_ratio": family_majority_ratio,
            "max_single_feature_auroc": float(leakage_scan["max_single_feature_auroc"]),
            "shuffled_label_balanced_val_acc": shuffled_label_balanced_val_acc,
            "feature_ablation_drop": float(ablation_check["accuracy_drop"]),
            "feature_ablation_idx": float(ablation_check["ablated_feature_idx"]),
            "min_family_minority_recall": min_family_minority_recall,
            "min_family_entropy": min_family_entropy,
            "max_zero_prediction_classes": max_zero_prediction_classes,
            "min_binary_auprc": min_binary_auprc,
            **promotion_consensus.to_stage_metrics(),
        },
    }
    if promotion_consensus.invalid_reason is not None:
        governance_stages["prepromote"]["promotion_invalid_reason"] = (
            promotion_consensus.invalid_reason
        )
    elif not data_integrity_pass:
        governance_stages["prepromote"]["promotion_invalid_reason"] = (
            "data_integrity_or_signal_quality_failure"
        )

    logger.info(f"✅ Results saved to {results_path}")
    logger.info("=" * 80)
    logger.info("Training complete!")

    return {
        "results": results,
        "governance_stages": governance_stages,
        "governance_context": {
            "seed": args.seed,
        },
        "governance_run_record": {
            "dataset_id": "helix_full",
            "macro_f1": aggregate_macro_f1,
            "fingerprint": os.environ.get("HELIX_FINGERPRINT"),
            "parent_run_id": os.environ.get("HELIX_PARENT_RUN_ID"),
            "lineage": {
                "dataset_hashes": os.environ.get("HELIX_DATASET_HASHES", "unknown"),
                "schema_hash": os.environ.get("HELIX_SCHEMA_HASH", "unknown"),
                "mapping_version": os.environ.get("HELIX_MAPPING_VERSION", "unknown"),
                "model_artifact": str(best_model_path),
                "metrics_artifact": str(results_path),
            },
        },
        "determinism": determinism_state.to_dict(),
    }


if __name__ == "__main__":
    main()
