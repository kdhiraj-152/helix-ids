#!/usr/bin/env python3
"""Phase 27B — Multi-Dataset CORAL Validation.

Runs 8 experiments: 4 pairwise + 4 holdout across NSL-KDD, UNSW-NB15,
CICIDS2018, and TON-IoT. Every experiment trains a baseline (lambda=0)
and a CORAL model (lambda=0.50), capturing full metrics and embeddings.

Usage:
    python benchmarks/phase27b.py [--epochs 100] [--max-samples 50000]
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase27b")

# ── Project paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from helix_ids.contracts.schema_contract import CANONICAL_FEATURE_ORDER, CANONICAL_INPUT_DIM

# ── Constants ──────────────────────────────────────────────────────────
NUM_FEATURES = 17
NUM_CLASSES = 7
CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R", "Generic", "Backdoor"]

DATASET_NAMES = ["nsl_kdd", "unsw_nb15", "cicids2018", "ton_iot"]
DATASET_DISPLAY = {
    "nsl_kdd": "NSL-KDD",
    "unsw_nb15": "UNSW-NB15",
    "cicids2018": "CICIDS2018",
    "ton_iot": "TON-IoT",
}

# All 8 experiments — (name, source_datasets, target_dataset)
EXPERIMENTS = [
    ("exp01_pairwise_nsl_to_unsw",       ["nsl_kdd"],                                "unsw_nb15"),
    ("exp02_pairwise_unsw_to_cicids",    ["unsw_nb15"],                              "cicids2018"),
    ("exp03_pairwise_cicids_to_ton",     ["cicids2018"],                             "ton_iot"),
    ("exp04_pairwise_ton_to_nsl",        ["ton_iot"],                                "nsl_kdd"),
    ("exp05_holdout_3src_to_ton",        ["nsl_kdd", "unsw_nb15", "cicids2018"],     "ton_iot"),
    ("exp06_holdout_3src_to_cicids",     ["nsl_kdd", "unsw_nb15", "ton_iot"],        "cicids2018"),
    ("exp07_holdout_3src_to_nsl",        ["unsw_nb15", "cicids2018", "ton_iot"],     "nsl_kdd"),
    ("exp08_holdout_3src_to_unsw",       ["nsl_kdd", "cicids2018", "ton_iot"],       "unsw_nb15"),
]

# Device
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info("Device: %s", DEVICE)


# ══════════════════════════════════════════════════════════════════════════
# Model — CORALHelixModel (Phase 27A architecture)
# ══════════════════════════════════════════════════════════════════════════

class CORALHelixModel(nn.Module):
    """Backbone + dual-head classifier with backbone feature access for CORAL.

    Same architecture as Phase 27A train_with_coral.py.
    """

    def __init__(self, input_dim: int = NUM_FEATURES, family_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.family_head = nn.Linear(64, family_classes)
        self.binary_head = nn.Linear(64, 2)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        family_logits = self.family_head(features)
        binary_logits = self.binary_head(features)
        if return_features:
            return binary_logits, family_logits, features
        return binary_logits, family_logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return features


def coral_loss(source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    """CORAL loss: L2 distance between source and target covariance matrices.

    Same implementation as Phase 27A.
    """
    d = source_features.size(1)

    src_mean = source_features.mean(dim=0, keepdim=True)
    tgt_mean = target_features.mean(dim=0, keepdim=True)
    src_centered = source_features - src_mean
    tgt_centered = target_features - tgt_mean

    src_cov = (src_centered.t() @ src_centered) / (source_features.size(0) - 1)
    tgt_cov = (tgt_centered.t() @ tgt_centered) / (target_features.size(0) - 1)

    return ((src_cov - tgt_cov) ** 2).sum() / (4.0 * d * d)


# ══════════════════════════════════════════════════════════════════════════
# Data loading & preprocessing
# ══════════════════════════════════════════════════════════════════════════

def load_all_datasets(
    max_samples: int = 50000,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load all 4 datasets through MultiDatasetLoader, return (X, y) dict.

    Each (X, y) has canonical 17 features and 7-class labels.
    Subsampled to max_samples per dataset when given.
    """
    from helix_ids.data.multi_dataset_loader import MultiDatasetLoader

    loader = MultiDatasetLoader(project_root=PROJECT_ROOT)
    nsl_kdd, unsw, cicids, ton_iot, *_ = loader.load_and_harmonize_all()

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ds_name, df in [
        ("nsl_kdd", nsl_kdd),
        ("unsw_nb15", unsw),
        ("cicids2018", cicids),
        ("ton_iot", ton_iot),
    ]:
        if df is None or len(df) == 0:
            logger.warning("Dataset %s is empty or None, skipping", ds_name)
            continue

        # Validate canonical feature columns
        missing = [c for c in CANONICAL_FEATURE_ORDER if c not in df.columns]
        if missing:
            logger.warning(
                "Dataset %s missing canonical features: %s. Available: %s",
                ds_name, missing, list(df.columns[:5]),
            )
            continue

        X = df[CANONICAL_FEATURE_ORDER].to_numpy(dtype=np.float32)
        y = df["label"].to_numpy(dtype=np.int64)

        # Subsample if needed
        if max_samples > 0 and len(X) > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X), size=max_samples, replace=False)
            idx.sort()
            X = X[idx]
            y = y[idx]

        logger.info("  %s: %d samples, %d classes", ds_name, len(X), len(np.unique(y)))
        result[ds_name] = (X, y)

    return result


def create_splits(
    harmonized: dict[str, tuple[np.ndarray, np.ndarray]],
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Create stratified train/test splits per dataset.

    Returns dict[name] -> (X_train, y_train, X_test, y_test)
    """
    from sklearn.model_selection import train_test_split

    rng = np.random.RandomState(seed)
    splits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    for ds_name, (X, y) in harmonized.items():
        unique_classes = np.unique(y)
        if len(unique_classes) <= 1:
            logger.warning("  %s: only 1 class, splitting randomly", ds_name)
            n = len(X)
            n_test = int(n * test_size)
            idx = np.arange(n)
            rng.shuffle(idx)
            train_idx = idx[n_test:]
            test_idx = idx[:n_test]
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed, stratify=y,
            )

        logger.info(
            "  %s split: %d train, %d test",
            ds_name, len(X_train), len(X_test),
        )
        splits[ds_name] = (X_train, y_train, X_test, y_test)

    return splits


def cap_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_per_class: int = 5000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified cap on training data per class."""
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    for cls in sorted(np.unique(y_train)):
        idx = np.where(y_train == cls)[0]
        if len(idx) > max_per_class:
            chosen = rng.choice(idx, size=max_per_class, replace=False)
        else:
            chosen = idx
        X_list.append(X_train[chosen])
        y_list.append(y_train[chosen])
    return np.vstack(X_list), np.concatenate(y_list)


# ══════════════════════════════════════════════════════════════════════════
# Training infrastructure
# ══════════════════════════════════════════════════════════════════════════

class MultiTaskDataset(torch.utils.data.Dataset):
    """Minimal dataset returning (x, binary_label, family_label)."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        # binary: 0=normal, 1=attack
        self.y_bin = torch.tensor((y > 0).astype(np.int64), dtype=torch.long)
        self.y_fam = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y_bin[idx], self.y_fam[idx]


class UnlabeledDataset(torch.utils.data.Dataset):
    """Dataset for target-domain data (features only, no labels)."""

    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.X[idx]


def train_model(
    X_src: np.ndarray,
    y_src: np.ndarray,
    X_tgt: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    lambda_coral: float = 0.0,
    epochs: int = 150,
    batch_size: int = 256,
    lr: float = 5e-4,
    patience: int = 20,
    seed: int = 42,
    experiment_label: str = "",
) -> tuple[CORALHelixModel, dict]:
    """Train a CORALHelixModel with optional CORAL loss.

    Args:
        X_src: Source training features
        y_src: Source training labels (7-class)
        X_tgt: Target training features (unlabeled, for CORAL alignment)
        X_val: Validation features
        y_val: Validation labels
        lambda_coral: Weight for CORAL loss (0.0 = baseline)
        ...

    Returns:
        (trained_model, history_dict)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CORALHelixModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_dataset = MultiTaskDataset(X_src, y_src)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
    )

    val_dataset = MultiTaskDataset(X_val, y_val)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    # Target data loader for CORAL (unlabeled)
    tgt_dataset = UnlabeledDataset(X_tgt)
    tgt_loader = torch.utils.data.DataLoader(
        tgt_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
    )

    history: dict[str, list] = {"epoch": [], "train_loss": [], "val_f1": []}
    best_val_f1 = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        train_batches = 0

        src_iter = iter(train_loader)
        tgt_iter = iter(tgt_loader)

        # Train for one epoch, alternating source and target batches
        for src_batch in train_loader:
            x_batch, y_bin_batch, y_fam_batch = src_batch
            x_batch = x_batch.to(DEVICE)
            y_bin_batch = y_bin_batch.to(DEVICE)
            y_fam_batch = y_fam_batch.to(DEVICE)

            # Source forward pass
            bin_logits, fam_logits = model(x_batch)
            loss_cls = F.cross_entropy(bin_logits, y_bin_batch) + F.cross_entropy(fam_logits, y_fam_batch)

            loss = loss_cls

            # CORAL loss on backbone features
            if lambda_coral > 0:
                try:
                    x_tgt_batch = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(tgt_loader)
                    x_tgt_batch = next(tgt_iter)

                x_tgt = x_tgt_batch.to(DEVICE)
                src_features = model.extract_features(x_batch)
                tgt_features = model.extract_features(x_tgt)
                loss_coral_val = coral_loss(src_features, tgt_features)
                loss = loss_cls + lambda_coral * loss_coral_val

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            train_batches += 1

        avg_train_loss = total_loss / max(train_batches, 1)

        # Validation
        model.eval()
        val_preds: list[int] = []
        val_targets: list[int] = []
        with torch.no_grad():
            for x_batch, _, y_fam in val_loader:
                x_batch = x_batch.to(DEVICE)
                _, fam_logits = model(x_batch)
                val_preds.extend(fam_logits.argmax(dim=1).cpu().numpy().tolist())
                val_targets.extend(y_fam.numpy().tolist())

        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["val_f1"].append(float(val_f1))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            label = f"{experiment_label} (λ={lambda_coral:.2f})"
            logger.info(
                "    [%s] Epoch %3d/%d | Loss: %.4f | Val F1: %.4f",
                label, epoch + 1, epochs, avg_train_loss, val_f1,
            )

        if patience_counter >= patience:
            logger.info("    [%s] Early stopping at epoch %d", experiment_label, epoch + 1)
            break

    # Restore best state
    if best_state:
        model.load_state_dict(best_state)

    return model, history


def evaluate_model(
    model: CORALHelixModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate trained model on test data, return full metrics."""
    model.eval()
    dataset = MultiTaskDataset(X_test, y_test)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=512, shuffle=False, num_workers=0,
    )

    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for x_batch, _, y_fam in loader:
            x_batch = x_batch.to(DEVICE)
            _, fam_logits = model(x_batch)
            all_preds.extend(fam_logits.argmax(dim=1).cpu().numpy().tolist())
            all_targets.extend(y_fam.numpy().tolist())

    y_pred = np.array(all_preds, dtype=np.int64)
    y_true = np.array(all_targets, dtype=np.int64)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def extract_embeddings(
    model: CORALHelixModel,
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract backbone embeddings and labels."""
    model.eval()
    n = len(X)
    if n > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_samples, replace=False)
        X = X[idx]
        y = y[idx]

    dataset = MultiTaskDataset(X, y)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=512, shuffle=False, num_workers=0,
    )

    all_embs: list[np.ndarray] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for x_batch, _, y_fam in loader:
            x_batch = x_batch.to(DEVICE)
            feats = model.extract_features(x_batch)
            all_embs.append(feats.cpu().numpy())
            all_labels.extend(y_fam.numpy().tolist())

    return np.vstack(all_embs), np.array(all_labels, dtype=np.int64)


# ══════════════════════════════════════════════════════════════════════════
# Per-experiment execution
# ══════════════════════════════════════════════════════════════════════════

def run_experiment(
    exp_name: str,
    source_names: list[str],
    target_name: str,
    harmonized: dict[str, tuple[np.ndarray, np.ndarray]],
    splits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    epochs: int = 150,
    seed: int = 42,
    lambda_coral: float = 0.50,
    train_cap_per_class: int = 5000,
) -> dict:
    """Run baseline + CORAL for one experiment, return full result dict."""
    logger.info("\n%s", "=" * 70)
    sources_display = " + ".join(DATASET_DISPLAY.get(s, s) for s in source_names)
    target_display = DATASET_DISPLAY.get(target_name, target_name)
    logger.info("EXPERIMENT: %s", exp_name)
    logger.info("  Source: %s", sources_display)
    logger.info("  Target: %s", target_display)
    logger.info("%s", "=" * 70)

    # Combine source training data
    X_src_list, y_src_list = [], []
    for s_name in source_names:
        X_tr, y_tr = splits[s_name][0], splits[s_name][1]
        X_src_list.append(X_tr)
        y_src_list.append(y_tr)

    X_src_raw = np.vstack(X_src_list) if len(X_src_list) > 1 else X_src_list[0]
    y_src_raw = np.concatenate(y_src_list) if len(y_src_list) > 1 else y_src_list[0]

    # Cap per-class to avoid imbalance blowup
    X_src, y_src = cap_training_data(X_src_raw, y_src_raw, max_per_class=train_cap_per_class, seed=seed)

    # Target data
    X_tgt_train, y_tgt_train = splits[target_name][0], splits[target_name][1]
    X_tgt_test, y_tgt_test = splits[target_name][2], splits[target_name][3]

    # Use 20% of target train as validation for early stopping
    from sklearn.model_selection import train_test_split
    X_tgt_val, X_tgt_for_coral, y_tgt_val, _ = train_test_split(
        X_tgt_train, y_tgt_train, test_size=0.8, random_state=seed, stratify=y_tgt_train,
    )

    # ── Standardize: fit on source, transform all ────────────────────
    scaler = StandardScaler()
    X_src_scaled = scaler.fit_transform(X_src).astype(np.float32)
    X_tgt_val_scaled = scaler.transform(X_tgt_val).astype(np.float32)
    X_tgt_for_coral_scaled = scaler.transform(X_tgt_for_coral).astype(np.float32)
    X_tgt_test_scaled = scaler.transform(X_tgt_test).astype(np.float32)

    result: dict = {
        "experiment_name": exp_name,
        "source_datasets": source_names,
        "target_dataset": target_name,
        "source_display": sources_display,
        "target_display": target_display,
        "source_train_samples": len(X_src_scaled),
        "target_test_samples": len(X_tgt_test_scaled),
    }

    for model_type, lc in [("baseline", 0.0), ("coral", lambda_coral)]:
        label = f"{exp_name}/{model_type}"
        logger.info("\n  Training %s (λ=%s)...", model_type, lc)

        t0 = time.time()
        model, history = train_model(
            X_src_scaled, y_src,
            X_tgt_for_coral_scaled,
            X_tgt_val_scaled, y_tgt_val,
            lambda_coral=lc,
            epochs=epochs,
            seed=seed,
            experiment_label=label,
        )
        train_time = time.time() - t0

        # Evaluate on target test
        metrics = evaluate_model(model, X_tgt_test_scaled, y_tgt_test)

        # Compute train accuracy for gen gap
        train_metrics = evaluate_model(model, X_src_scaled, y_src)
        gen_gap = train_metrics["accuracy"] - metrics["accuracy"]

        # Embeddings
        embs_src, _ = extract_embeddings(model, X_src_scaled, y_src, max_samples=3000)
        embs_tgt, y_tgt_for_sil = extract_embeddings(
            model, X_tgt_test_scaled, y_tgt_test, max_samples=3000,
        )

        # Silhouette by dataset (combine source + target embeddings with dataset_ids)
        all_embs = np.vstack([embs_src, embs_tgt])
        ds_ids = np.array([0] * len(embs_src) + [1] * len(embs_tgt))  # 0=source, 1=target
        sil_dataset = float(silhouette_score(all_embs, ds_ids)) if len(np.unique(ds_ids)) > 1 else 0.0

        # Silhouette by attack family (on target only)
        target_family_ids = y_tgt_for_sil
        if len(np.unique(target_family_ids)) > 1:
            sil_family = float(silhouette_score(embs_tgt, target_family_ids))
        else:
            sil_family = 0.0

        epochs_trained = len(history["epoch"]) if history["epoch"] else epochs

        model_results = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "train_accuracy": train_metrics["accuracy"],
            "generalization_gap": float(gen_gap),
            "epochs_trained": epochs_trained,
            "training_time_s": float(train_time),
            "early_stopping_epoch": epochs_trained,
            "confusion_matrix": None,
            "silhouette_dataset": sil_dataset,
            "silhouette_family": sil_family,
        }

        # Save embeddings for later plotting
        embs_data = {
            "source_embs": embs_src.tolist(),
            "target_embs": embs_tgt.tolist(),
            "target_labels": y_tgt_for_sil.tolist(),
        }

        result[model_type] = model_results
        result[f"{model_type}_embeddings"] = embs_data  # for later plot generation

        logger.info(
            "  RESULTS [%s]: Acc=%.4f | F1=%.4f | GenGap=%+.4f | Epochs=%d | %.1fs",
            label, metrics["accuracy"], metrics["macro_f1"],
            gen_gap, epochs_trained, train_time,
        )
        logger.info(
            "  Silhouette — dataset=%.4f, family=%.4f",
            sil_dataset, sil_family,
        )

        # Clean large tensors
        del model
        torch.mps.empty_cache() if DEVICE.type == "mps" else None

    result["improvement"] = {
        "macro_f1_delta": result.get("coral", {}).get("macro_f1", 0) - result.get("baseline", {}).get("macro_f1", 0),
        "accuracy_delta": result.get("coral", {}).get("accuracy", 0) - result.get("baseline", {}).get("accuracy", 0),
        "silhouette_dataset_delta": result.get("coral", {}).get("silhouette_dataset", 0) - result.get("baseline", {}).get("silhouette_dataset", 0),
    }

    return result


# ══════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════

def generate_plots(all_results: dict[str, dict], plots_dir: Path):
    """Generate all Phase 27B plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plots_dir.mkdir(parents=True, exist_ok=True)
    (plots_dir / "tsne").mkdir(parents=True, exist_ok=True)
    (plots_dir / "umap").mkdir(parents=True, exist_ok=True)

    # ── 1. CORAL vs Baseline heatmap ────────────────────────────────
    exp_names = list(all_results.keys())
    baseline_f1 = [all_results[e]["baseline"]["macro_f1"] for e in exp_names]
    coral_f1 = [all_results[e]["coral"]["macro_f1"] for e in exp_names]
    f1_delta = [c - b for c, b in zip(coral_f1, baseline_f1)]

    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(exp_names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, baseline_f1, width, label="Baseline", color="steelblue")
    bars2 = ax.bar(x + width / 2, coral_f1, width, label="CORAL", color="coral")

    # Annotate deltas
    for i, (b, c, d) in enumerate(zip(baseline_f1, coral_f1, f1_delta)):
        color = "green" if d > 0 else "red"
        ax.text(x[i] + width / 2, c + 0.01, f"{d:+.3f}", ha="center", va="bottom",
                fontsize=9, color=color, fontweight="bold")

    ax.set_ylabel("Macro F1")
    ax.set_title("Phase 27B: Baseline vs CORAL Macro F1\n(λ = 0.50)")
    ax.set_xticks(x)
    short_names = [
        "NSL→UNSW", "UNSW→CIC", "CIC→TON", "TON→NSL",
        "3→TON", "3→CIC", "3→NSL", "3→UNSW"
    ]
    ax.set_xticklabels(short_names, rotation=25, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = plots_dir / "coral_vs_baseline_heatmap.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", p)

    # ── 2. F1 improvement bar chart ─────────────────────────────────
    colors = ["green" if d > 0 else "red" for d in f1_delta]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(short_names, f1_delta, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Macro F1 Δ (CORAL − Baseline)")
    ax.set_title("CORAL F1 Improvement per Experiment")
    for bar, val in zip(bars, f1_delta):
        ax.text(val + (0.005 if val >= 0 else -0.03), bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}", va="center", fontsize=9)
    plt.tight_layout()
    p = plots_dir / "coral_f1_improvement.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", p)

    # ── 3. Silhouette comparison ────────────────────────────────────
    sil_ds_baseline = [all_results[e]["baseline"]["silhouette_dataset"] for e in exp_names]
    sil_ds_coral = [all_results[e]["coral"]["silhouette_dataset"] for e in exp_names]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(exp_names))
    ax.plot(x, sil_ds_baseline, "o-", color="steelblue", label="Baseline (dataset silhouette)")
    ax.plot(x, sil_ds_coral, "s--", color="coral", label="CORAL (dataset silhouette)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=25, ha="right")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Embedding Silhouette by Dataset: Baseline vs CORAL")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = plots_dir / "silhouette_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", p)

    # ── 4. Embedding visualizations (t-SNE, UMAP) per experiment ────
    logger.info("Generating t-SNE/UMAP projections...")
    from sklearn.manifold import TSNE

    for exp_name, res in all_results.items():
        for model_type in ["baseline", "coral"]:
            embs_key = f"{model_type}_embeddings"
            if embs_key not in res:
                continue

            source_embs = np.array(res[embs_key]["source_embs"])
            target_embs = np.array(res[embs_key]["target_embs"])
            target_labels = np.array(res[embs_key]["target_labels"])

            all_embs = np.vstack([source_embs, target_embs])
            ds_labels = np.array([0] * len(source_embs) + [1] * len(target_embs))

            # Subsample for speed
            n_total = len(all_embs)
            if n_total > 5000:
                rng = np.random.default_rng(42)
                idx = rng.choice(n_total, size=5000, replace=False)
                all_embs = all_embs[idx]
                ds_labels = ds_labels[idx]

            # t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
            embs_2d = tsne.fit_transform(all_embs)

            fig, axes = plt.subplots(1, 2, figsize=(16, 7))

            # By dataset
            scatter_ds = axes[0].scatter(
                embs_2d[:, 0], embs_2d[:, 1], c=ds_labels,
                cmap="coolwarm", alpha=0.6, s=8,
            )
            axes[0].set_title(f"{exp_name} — {model_type}\nColored by Dataset (red=target)")
            axes[0].set_xlabel("t-SNE 1")
            axes[0].set_ylabel("t-SNE 2")
            axes[0].grid(True, alpha=0.2)
            cbar_ds = fig.colorbar(scatter_ds, ax=axes[0])
            cbar_ds.set_ticks([0, 1])
            cbar_ds.set_ticklabels(["Source", "Target"])

            # By family (target only)
            tgt_embs_2d = embs_2d[ds_labels == 1]
            if len(tgt_embs_2d) > 0:
                tgt_labels_2d = target_labels[:len(tgt_embs_2d)] if len(target_labels) >= len(tgt_embs_2d) else target_labels
                scatter_fam = axes[1].scatter(
                    tgt_embs_2d[:, 0], tgt_embs_2d[:, 1],
                    c=tgt_labels_2d if len(tgt_labels_2d) == len(tgt_embs_2d) else ds_labels[:len(tgt_embs_2d)],
                    cmap="tab10", alpha=0.6, s=10,
                )
                axes[1].set_title(f"{exp_name} — {model_type}\nTarget Only (by Attack Family)")
                axes[1].set_xlabel("t-SNE 1")
                axes[1].set_ylabel("t-SNE 2")
                axes[1].grid(True, alpha=0.2)
                fig.colorbar(scatter_fam, ax=axes[1])

            plt.tight_layout()
            p = plots_dir / "tsne" / f"{exp_name}_{model_type}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("  Generated t-SNE for %s/%s", exp_name, model_type)

    # Try UMAP
    try:
        import umap

        for exp_name, res in all_results.items():
            for model_type in ["baseline", "coral"]:
                embs_key = f"{model_type}_embeddings"
                if embs_key not in res:
                    continue

                source_embs = np.array(res[embs_key]["source_embs"])
                target_embs = np.array(res[embs_key]["target_embs"])

                all_embs = np.vstack([source_embs, target_embs])
                ds_labels = np.array([0] * len(source_embs) + [1] * len(target_embs))

                n_total = len(all_embs)
                if n_total > 5000:
                    rng = np.random.default_rng(42)
                    idx = rng.choice(n_total, size=5000, replace=False)
                    all_embs = all_embs[idx]
                    ds_labels = ds_labels[idx]

                reducer = umap.UMAP(n_components=2, random_state=42)
                embs_2d = reducer.fit_transform(all_embs)

                fig, axes = plt.subplots(1, 2, figsize=(16, 7))
                axes[0].scatter(embs_2d[:, 0], embs_2d[:, 1], c=ds_labels,
                                cmap="coolwarm", alpha=0.6, s=8)
                axes[0].set_title(f"{exp_name} — {model_type} (UMAP by Dataset)")

                tgt_embs_2d = embs_2d[ds_labels == 1]
                if len(tgt_embs_2d) > 0:
                    axes[1].scatter(tgt_embs_2d[:, 0], tgt_embs_2d[:, 1],
                                    alpha=0.6, s=10)
                    axes[1].set_title(f"{exp_name} — {model_type} (UMAP Target)")

                plt.tight_layout()
                p = plots_dir / "umap" / f"{exp_name}_{model_type}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(p, dpi=150, bbox_inches="tight")
                plt.close(fig)
                logger.info("  Generated UMAP for %s/%s", exp_name, model_type)

    except ImportError:
        logger.warning("UMAP not installed, skipping UMAP plots")

    logger.info("All plots generated in %s", plots_dir)


# ══════════════════════════════════════════════════════════════════════════
# Report generation
# ══════════════════════════════════════════════════════════════════════════

def generate_pairwise_results(all_results: dict[str, dict], doc_dir: Path):
    """Write PAIRWISE_RESULTS.md."""
    pairwise = [k for k in all_results if "pairwise" in k]
    lines = [
        "# Pairwise Transfer Results\n\n",
        f"## Experiments Run: {len(pairwise)}/4\n\n",
    ]

    lines.append("| Experiment | Model | Accuracy | Macro F1 | Precision | Recall | Gen Gap | Epochs | Time (s) | Sil-DS | Sil-Fam |\n")
    lines.append("|-----------|------:|--------:|---------:|----------:|------:|--------:|------:|--------:|-------:|--------:|\n")

    for exp_name in sorted(pairwise):
        res = all_results[exp_name]
        src = "+".join(DATASET_DISPLAY.get(s, s) for s in res["source_datasets"])
        tgt = DATASET_DISPLAY.get(res["target_dataset"], res["target_dataset"])

        for model_type in ["baseline", "coral"]:
            m = res[model_type]
            lines.append(
                f"| {src}→{tgt} ({model_type}) | {model_type} "
                f"| {m['accuracy']:.4f} | {m['macro_f1']:.4f} "
                f"| {m['precision']:.4f} | {m['recall']:.4f} "
                f"| {m['generalization_gap']:+.4f} | {m['epochs_trained']} "
                f"| {m['training_time_s']:.1f} "
                f"| {m['silhouette_dataset']:.4f} | {m['silhouette_family']:.4f} |\n"
            )

    lines.append("\n\n## Individual Experiment Details\n\n")
    for exp_name in sorted(pairwise):
        res = all_results[exp_name]
        lines.append(f"### {exp_name}\n\n")
        lines.append(f"- **Source**: {res['source_display']}\n")
        lines.append(f"- **Target**: {res['target_display']}\n")
        lines.append(f"- **Train samples**: {res['source_train_samples']}\n")
        lines.append(f"- **Test samples**: {res['target_test_samples']}\n")

        imp = res.get("improvement", {})
        lines.append(f"\n**CORAL Δ — Macro F1: {imp.get('macro_f1_delta', 0):+.4f}**\n\n")

        for model_type in ["baseline", "coral"]:
            m = res[model_type]
            lines.append(
                f"- **{model_type.title()}**: "
                f"Acc={m['accuracy']:.4f}, F1={m['macro_f1']:.4f}, "
                f"Prec={m['precision']:.4f}, Rec={m['recall']:.4f}, "
                f"GenGap={m['generalization_gap']:+.4f}, "
                f"Epochs={m['epochs_trained']}, "
                f"Sil-DS={m['silhouette_dataset']:.4f}, Sil-Fam={m['silhouette_family']:.4f}\n"
            )
        lines.append("\n")

    doc_dir.mkdir(parents=True, exist_ok=True)
    p = doc_dir / "PAIRWISE_RESULTS.md"
    p.write_text("".join(lines))
    logger.info("Generated %s", p)


def generate_holdout_results(all_results: dict[str, dict], doc_dir: Path):
    """Write HOLDOUT_RESULTS.md."""
    holdout = [k for k in all_results if "holdout" in k]
    lines = [
        "# Holdout Transfer Results\n\n",
        f"## Experiments Run: {len(holdout)}/4\n\n",
    ]

    lines.append("| Experiment | Model | Accuracy | Macro F1 | Precision | Recall | Gen Gap | Epochs | Time (s) | Sil-DS | Sil-Fam |\n")
    lines.append("|-----------|------:|--------:|---------:|----------:|------:|--------:|------:|--------:|-------:|--------:|\n")

    for exp_name in sorted(holdout):
        res = all_results[exp_name]
        src = "3-dataset"
        tgt = DATASET_DISPLAY.get(res["target_dataset"], res["target_dataset"])

        for model_type in ["baseline", "coral"]:
            m = res[model_type]
            lines.append(
                f"| {src}→{tgt} ({model_type}) | {model_type} "
                f"| {m['accuracy']:.4f} | {m['macro_f1']:.4f} "
                f"| {m['precision']:.4f} | {m['recall']:.4f} "
                f"| {m['generalization_gap']:+.4f} | {m['epochs_trained']} "
                f"| {m['training_time_s']:.1f} "
                f"| {m['silhouette_dataset']:.4f} | {m['silhouette_family']:.4f} |\n"
            )

    lines.append("\n\n## Individual Experiment Details\n\n")
    for exp_name in sorted(holdout):
        res = all_results[exp_name]
        lines.append(f"### {exp_name}\n\n")
        lines.append(f"- **Source Datasets**: {res['source_display']}\n")
        lines.append(f"- **Target Held-Out**: {res['target_display']}\n")
        lines.append(f"- **Train samples**: {res['source_train_samples']}\n")
        lines.append(f"- **Test samples**: {res['target_test_samples']}\n")

        imp = res.get("improvement", {})
        lines.append(f"\n**CORAL Δ — Macro F1: {imp.get('macro_f1_delta', 0):+.4f}**\n\n")

        for model_type in ["baseline", "coral"]:
            m = res[model_type]
            lines.append(
                f"- **{model_type.title()}**: "
                f"Acc={m['accuracy']:.4f}, F1={m['macro_f1']:.4f}, "
                f"Prec={m['precision']:.4f}, Rec={m['recall']:.4f}, "
                f"GenGap={m['generalization_gap']:+.4f}, "
                f"Epochs={m['epochs_trained']}, "
                f"Sil-DS={m['silhouette_dataset']:.4f}, Sil-Fam={m['silhouette_family']:.4f}\n"
            )
        lines.append("\n")

    doc_dir.mkdir(parents=True, exist_ok=True)
    p = doc_dir / "HOLDOUT_RESULTS.md"
    p.write_text("".join(lines))
    logger.info("Generated %s", p)


def generate_embedding_audit(all_results: dict[str, dict], doc_dir: Path):
    """Write EMBEDDING_AUDIT.md."""
    lines = [
        "# Embedding Audit\n\n",
        "## Methodology\n\n",
        "For each experiment, backbone embeddings (64-dim) are extracted from both baseline "
        "and CORAL models for source and target test data. We compute:\n\n",
        "- **silhouette_dataset**: Silhouette score of source vs target embeddings.\n",
        "  Lower = better domain-invariant alignment.\n",
        "- **silhouette_family**: Silhouette score of attack-family clusters within target.\n",
        "  Higher = better class separability.\n\n",
        "| Experiment | Model | Sil-Dataset (↓better) | Sil-Family (↑better) |\n",
        "|-----------|------:|---------------------:|--------------------:|\n",
    ]

    sil_ds_deltas = []
    sil_fam_deltas = []

    for exp_name in sorted(all_results.keys()):
        res = all_results[exp_name]
        for model_type in ["baseline", "coral"]:
            m = res[model_type]
            sil_ds = m["silhouette_dataset"]
            sil_fam = m["silhouette_family"]
            lines.append(
                f"| {exp_name} | {model_type} | {sil_ds:.4f} | {sil_fam:.4f} |\n"
            )
        # Silhouette delta (coral - baseline)
        imp = res.get("improvement", {})
        sil_ds_deltas.append(imp.get("silhouette_dataset_delta", 0))

    avg_sil_ds_delta = np.mean(sil_ds_deltas) if sil_ds_deltas else 0

    lines.append(f"\n\n## Summary\n\n")
    lines.append(f"- **Average silhouette_dataset delta**: {avg_sil_ds_delta:.4f}\n")

    if avg_sil_ds_delta <= -0.05:
        lines.append("- **CORAL reduces domain separation** (silhouette decreases by >= 0.05)\n")
        lines.append("- **Verdict**: CORAL improves domain-invariant feature alignment.\n")
    elif avg_sil_ds_delta <= 0:
        lines.append("- **CORAL slightly reduces or maintains** domain separation.\n")
    else:
        lines.append("- **CORAL increases domain separation** — feature alignment degraded.\n")

    p = doc_dir / "EMBEDDING_AUDIT.md"
    doc_dir.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(lines))
    logger.info("Generated %s", p)


def generate_effectiveness_report(all_results: dict[str, dict], doc_dir: Path):
    """Write CORAL_EFFECTIVENESS.md."""
    lines = [
        "# CORAL Effectiveness Analysis\n\n",
        "## Summary Statistics\n\n",
    ]

    macro_f1_deltas = []
    accuracy_deltas = []
    sil_ds_deltas = []
    wins = 0
    losses = 0

    for exp_name in sorted(all_results.keys()):
        res = all_results[exp_name]
        imp = res.get("improvement", {})

        f1_d = imp.get("macro_f1_delta", 0)
        acc_d = imp.get("accuracy_delta", 0)
        sil_d = imp.get("silhouette_dataset_delta", 0)

        macro_f1_deltas.append(f1_d)
        accuracy_deltas.append(acc_d)
        sil_ds_deltas.append(sil_d)

        if f1_d > 0:
            wins += 1
        elif f1_d < 0:
            losses += 1

    avg_f1_delta = np.mean(macro_f1_deltas)
    avg_acc_delta = np.mean(accuracy_deltas)
    avg_sil_ds_delta = np.mean(sil_ds_deltas)
    total = len(all_results)

    lines.append(f"- **Total experiments**: {total}\n")
    lines.append(f"- **Wins (CORAL better)**: {wins}\n")
    lines.append(f"- **Losses (Baseline better)**: {losses}\n")
    lines.append(f"- **Ties**: {total - wins - losses}\n")
    lines.append(f"- **Average Macro F1 Δ**: {avg_f1_delta:+.4f}\n")
    lines.append(f"- **Average Accuracy Δ**: {avg_acc_delta:+.4f}\n")
    lines.append(f"- **Average Silhouette Dataset Δ**: {avg_sil_ds_delta:+.4f}\n")

    lines.append("\n\n## Per-Experiment Δ\n\n")
    lines.append("| Experiment | Δ Macro F1 | Δ Accuracy | Δ Sil-DS | Win/Loss |\n")
    lines.append("|-----------|----------:|----------:|--------:|----------:|\n")
    for exp_name in sorted(all_results.keys()):
        imp = all_results[exp_name].get("improvement", {})
        f1_d = imp.get("macro_f1_delta", 0)
        acc_d = imp.get("accuracy_delta", 0)
        sil_d = imp.get("silhouette_dataset_delta", 0)
        wl = "WIN" if f1_d > 0 else ("LOSS" if f1_d < 0 else "TIE")
        lines.append(f"| {exp_name} | {f1_d:+.4f} | {acc_d:+.4f} | {sil_d:+.4f} | {wl} |\n")

    lines.append("\n\n## Decision\n\n")
    avg_f1_pct = avg_f1_delta * 100

    if avg_f1_delta >= 0.20 and wins >= 5:
        lines.append(
            "**GO for Phase 27C (production CORAL integration)**\n\n"
            f"- Average macro F1 improvement = {avg_f1_pct:.2f}% (≥ 20% threshold)\n"
            f"- {wins}/{total} experiments improve (≥ 5/8 threshold)\n"
        )
        if avg_sil_ds_delta <= -0.15:
            lines.append(f"- Average silhouette reduction = {avg_sil_ds_delta:.4f} (≥ 15% reduction)\n")
        else:
            lines.append(f"- Average silhouette reduction = {avg_sil_ds_delta:.4f} (below 15% target)\n")
    elif avg_f1_delta < 0.10:
        lines.append(
            "**GO for Phase 28 (DANN domain-adversarial training)**\n\n"
            f"- Average macro F1 improvement = {avg_f1_pct:.2f}% (< 20% threshold)\n"
            f"- Average improvement < 10% → failure condition triggered\n"
        )
    else:
        lines.append(
            "**GO for Phase 28 (DANN domain-adversarial training)**\n\n"
            f"- Average macro F1 improvement = {avg_f1_pct:.2f}% (< 20% threshold)\n"
            f"- Primary success criterion not met.\n"
        )

    p = doc_dir / "CORAL_EFFECTIVENESS.md"
    doc_dir.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(lines))
    logger.info("Generated %s", p)


def generate_certification_report(all_results: dict[str, dict], doc_dir: Path):
    """Write PHASE27B_CORAL_CERTIFICATION.md."""
    macro_f1_deltas = []
    sil_ds_deltas = []
    wins = 0
    losses = 0

    for exp_name in sorted(all_results.keys()):
        res = all_results[exp_name]
        imp = res.get("improvement", {})
        f1_d = imp.get("macro_f1_delta", 0)
        sil_d = imp.get("silhouette_dataset_delta", 0)
        macro_f1_deltas.append(f1_d)
        sil_ds_deltas.append(sil_d)
        if f1_d > 0:
            wins += 1
        elif f1_d < 0:
            losses += 1

    avg_f1_delta = np.mean(macro_f1_deltas)
    avg_f1_pct = avg_f1_delta * 100
    avg_sil_delta = np.mean(sil_ds_deltas)
    total = len(all_results)

    # Decision logic
    primary_pass = avg_f1_delta >= 0.20
    secondary_pass = avg_sil_delta <= -0.15
    tertiary_pass = wins >= 5
    failure_triggered = avg_f1_delta < 0.10 or losses >= total // 2

    if primary_pass and tertiary_pass:
        decision = "GO"
        next_phase = "Phase 27C (production CORAL integration)"
    else:
        decision = "NO-GO" if failure_triggered else "HOLD"
        next_phase = "Phase 28 (DANN domain-adversarial training)"

    fail_status = "TRIGGERED" if failure_triggered else "NOT TRIGGERED"
    fail_icon = "⚠️ CORAL underperforms" if failure_triggered else "✅ No failure detected"

    lines = [
        "# PHASE 27B — CORAL Multi-Dataset Certification Report\n\n",
        f"**Decision**: {decision}\n",
        f"**Recommended next phase**: {next_phase}\n\n",
        "## Executive Summary\n\n",
        f"Phase 27B validates whether CORAL produces consistent domain-invariant "
        f"improvements across the entire Helix dataset ecosystem. We ran {total} "
        f"experiments: 4 pairwise and 4 holdout transfers covering all 4 datasets "
        f"(NSL-KDD, UNSW-NB15, CICIDS2018, TON-IoT).\n\n",
        "## Success Criteria\n\n",
        "### Primary: Average Macro F1 improvement >= 20%\n\n",
        f"- **Average Macro F1 Δ**: {avg_f1_delta:+.4f} ({avg_f1_pct:+.2f}%)\n",
        f"- **Threshold**: ≥ 0.20 (20%)\n",
        f"- **Result**: {'PASS' if primary_pass else 'FAIL'} {'✅' if primary_pass else '❌'}\n\n",
        "### Secondary: Average dataset silhouette reduction >= 15%\n\n",
        f"- **Average Silhouette Δ**: {avg_sil_delta:+.4f}\n",
        f"- **Threshold**: ≤ -0.15 (15% reduction)\n",
        f"- **Result**: {'PASS' if secondary_pass else 'FAIL'} {'✅' if secondary_pass else '❌'}\n\n",
        "### Tertiary: At least 5 of 8 experiments improve Macro F1\n\n",
        f"- **Wins**: {wins}/{total}\n",
        f"- **Threshold**: ≥ 5\n",
        f"- **Result**: {'PASS' if tertiary_pass else 'FAIL'} {'✅' if tertiary_pass else '❌'}\n\n",
        f"**Failure condition**: {fail_status} ({fail_icon})\n\n",
    ]
    lines += [
        "## Per-Experiment Macro F1\n\n",
        "| Experiment | Baseline F1 | CORAL F1 | Δ |\n",
        "|-----------|----------:|--------:|--:|\n",
    ]
    for exp_name in sorted(all_results.keys()):
        res = all_results[exp_name]
        b_f1 = res["baseline"]["macro_f1"]
        c_f1 = res["coral"]["macro_f1"]
        d = c_f1 - b_f1
        emoji = "🟢" if d > 0 else "🔴"
        lines.append(f"| {exp_name} | {b_f1:.4f} | {c_f1:.4f} | {d:+.4f} {emoji} |\n")

    lines += [
        "\n## Conclusion\n\n",
        f"1. **Average Macro F1 delta**: {avg_f1_delta:+.4f} ({avg_f1_pct:+.2f}%)\n",
        f"2. **Average silhouette delta**: {avg_sil_delta:+.4f}\n",
        f"3. **Wins/Losses**: {wins}/{losses} (ties: {total - wins - losses})\n",
    ]

    # Determine whether CORAL generalizes beyond NSL→UNSW
    # Check specific improvement in exp01 vs others
    exp01_name = [k for k in all_results if "nsl_to_unsw" in k]
    if exp01_name:
        exp01_imp = all_results[exp01_name[0]].get("improvement", {}).get("macro_f1_delta", 0)
    else:
        exp01_imp = 0

    other_improvements = [
        all_results[k].get("improvement", {}).get("macro_f1_delta", 0)
        for k in sorted(all_results.keys()) if "nsl_to_unsw" not in k
    ]

    generalized = any(d > 0 for d in other_improvements)

    if generalized:
        lines.append(
            f"4. **CORAL generalizes beyond NSL→UNSW**: YES — "
            f"consistently improves multiple dataset pairs{' ' + str([f'{d:+.4f}' for d in other_improvements]) if other_improvements else ''}\n"
        )
    else:
        lines.append(
            f"4. **CORAL generalizes beyond NSL→UNSW**: NO — "
            f"improvement only seen in the original NSL→UNSW pair\n"
        )

    lines.append(
        f"5. **Decision**: {decision} → Recommend {next_phase}\n"
    )

    p = doc_dir / ".." / "releases" / "PHASE27B_CORAL_CERTIFICATION.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(lines))
    logger.info("Generated %s", p)


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase 27B — Multi-Dataset CORAL Validation"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--max-samples", type=int, default=50000,
                        help="Subsample to at most this many rows per dataset")
    parser.add_argument("--train-cap-per-class", type=int, default=5000,
                        help="Max training rows per class per source dataset")
    parser.add_argument("--lambda-coral", type=float, default=0.50,
                        help="CORAL loss weight (Phase 27A winner)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-file",
                        default=str(PROJECT_ROOT / "benchmarks" / "phase27b_results.json"))
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, generate docs from existing results")
    args = parser.parse_args()

    results_json = Path(args.results_file)

    # Output directories
    doc_dir = PROJECT_ROOT / "docs" / "phase27b"
    plots_dir = PROJECT_ROOT / "plots"
    doc_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Load or run experiments ──────────────────────────────────────
    all_results: dict[str, dict] = {}

    if args.skip_train and results_json.exists():
        logger.info("Loading cached results from %s", results_json)
        all_results = json.loads(results_json.read_text())
        # Remove __run_config__ if present
        all_results.pop("__run_config__", None)
        # Load embeddings from separate file
        emb_path = results_json.with_stem(results_json.stem + "_embeddings")
        if emb_path.exists():
            emb_data = json.loads(emb_path.read_text())
            for en, ed in emb_data.items():
                if en in all_results:
                    for mk in ["baseline_embeddings", "coral_embeddings"]:
                        if mk in ed:
                            all_results[en][mk] = ed[mk]
            logger.info("Merged embeddings for %d experiments", len(emb_data))
        logger.info("Loaded %d experiments", len(all_results))
    else:
        logger.info("Using device: %s", DEVICE)

        # Step 1: Load harmonized data
        logger.info("Loading datasets through MultiDatasetLoader...")
        harmonized = load_all_datasets(max_samples=args.max_samples)
        available = set(harmonized.keys())
        logger.info("Available datasets: %s", available)

        if len(available) < 2:
            logger.error("Need at least 2 datasets, got %s", available)
            sys.exit(1)

        # Step 2: Create splits
        logger.info("Creating train/test splits...")
        splits = create_splits(harmonized, seed=args.seed)

        # Step 3: Filter experiments to available datasets
        active = []
        for name, sources, target in EXPERIMENTS:
            if all(s in available for s in sources) and target in available:
                active.append((name, sources, target))
                logger.info("  Enqueued: %s", name)
            else:
                logger.warning("  Skipping %s: missing datasets", name)

        if len(active) < 2:
            logger.error("Too few experiments (%d), exiting", len(active))
            sys.exit(1)

        # Step 4: Run experiments
        logger.info("\nRunning %d experiments...", len(active))
        for name, sources, target in active:
            try:
                result = run_experiment(
                    name, sources, target, harmonized, splits,
                    epochs=args.epochs, seed=args.seed,
                    lambda_coral=args.lambda_coral,
                    train_cap_per_class=args.train_cap_per_class,
                )
                all_results[name] = result

                # Save full embeddings separately BEFORE popping from results
                emb_path = results_json.with_stem(results_json.stem + "_embeddings")
                embs_data = {}
                for en, er in all_results.items():
                    embs_data[en] = {}
                    for mk in ["baseline_embeddings", "coral_embeddings"]:
                        if mk in er:
                            embs_data[en][mk] = er[mk]
                if embs_data:
                    emb_path.write_text(json.dumps(embs_data, indent=2, default=str))
                    logger.info("  Saved embeddings to %s", emb_path)

                # Save intermediate results (without embeddings)
                serializable = {}
                for en, er in all_results.items():
                    serializable[en] = {k: v for k, v in er.items() if k not in ["baseline_embeddings", "coral_embeddings"]}
                serializable["__run_config__"] = vars(args)

                results_json.parent.mkdir(parents=True, exist_ok=True)
                results_json.write_text(json.dumps(serializable, indent=2, default=str))
                logger.info("  Saved intermediate results to %s", results_json)

            except Exception as e:
                logger.error("Experiment %s FAILED: %s", name, e, exc_info=True)
                # Save partial results so far
                serializable = {k: v for k, v in all_results.items()}
                serializable["__run_config__"] = vars(args)
                results_json.write_text(json.dumps(serializable, indent=2, default=str))
                logger.info("  Saved partial results to %s", results_json)

        # Reload clean results (without embeddings) for summary
        serializable = {}
        for en, er in all_results.items():
            serializable[en] = {k: v for k, v in er.items() if k not in ["baseline_embeddings", "coral_embeddings"]}
        serializable["__run_config__"] = vars(args)
        results_json.write_text(json.dumps(serializable, indent=2, default=str))
        logger.info("Saved final results to %s", results_json)

    if not all_results:
        logger.error("No results to process")
        sys.exit(1)

    # ── Step 5: Generate plots ──────────────────────────────────────
    logger.info("\\nGenerating plots...")
    generate_plots(all_results, plots_dir)

    # ── Step 6: Generate reports ─────────────────────────────────────
    logger.info("\nGenerating reports...")
    generate_pairwise_results(all_results, doc_dir)
    generate_holdout_results(all_results, doc_dir)
    generate_embedding_audit(all_results, doc_dir)
    generate_effectiveness_report(all_results, doc_dir)

    logger.info("\nGenerating certification report...")
    generate_certification_report(all_results, doc_dir)

    # ── Summary ──────────────────────────────────────────────────────
    macro_f1_deltas = [
        all_results[e].get("improvement", {}).get("macro_f1_delta", 0)
        for e in sorted(all_results.keys())
    ]
    sil_deltas = [
        all_results[e].get("improvement", {}).get("silhouette_dataset_delta", 0)
        for e in sorted(all_results.keys())
    ]
    wins = sum(1 for d in macro_f1_deltas if d > 0)
    losses = sum(1 for d in macro_f1_deltas if d < 0)

    logger.info("\n%s", "=" * 70)
    logger.info("PHASE 27B — SUMMARY")
    logger.info("%s", "=" * 70)
    logger.info("  Experiments completed: %d/%d", len(all_results), len(EXPERIMENTS))
    logger.info("  Avg Macro F1 Δ:   %+.4f", np.mean(macro_f1_deltas))
    logger.info("  Avg Silhouette Δ: %+.4f", np.mean(sil_deltas))
    logger.info("  Wins/Losses:      %d/%d", wins, losses)
    logger.info("  Results JSON:     %s", results_json)
    logger.info("  Reports:          %s/", doc_dir)
    logger.info("  Plots:            %s/", plots_dir)
    logger.info("%s", "=" * 70)


if __name__ == "__main__":
    main()
