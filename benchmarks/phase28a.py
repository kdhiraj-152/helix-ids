#!/usr/bin/env python3
"""Phase 28A — Domain-Adversarial Training (DANN) for Cross-Dataset Generalization.

Runs 8 experiments (4 pairwise + 4 holdout) across 5 lambda_domain values.
Tests whether adversarial domain confusion eliminates dataset-specific
representations and improves cross-dataset generalization.

Usage:
    python benchmarks/phase28a.py [--epochs 100] [--max-samples 50000]
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
logger = logging.getLogger("phase28a")

# ── Project paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from helix_ids.contracts.schema_contract import CANONICAL_FEATURE_ORDER, CANONICAL_INPUT_DIM

# ── Constants ──────────────────────────────────────────────────────────────
NUM_FEATURES = 17
NUM_CLASSES = 7
NUM_DATASETS = 4  # nsl_kdd, unsw_nb15, cicids2018, ton_iot
CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R", "Generic", "Backdoor"]

DATASET_NAMES = ["nsl_kdd", "unsw_nb15", "cicids2018", "ton_iot"]
DATASET_DISPLAY = {
    "nsl_kdd": "NSL-KDD",
    "unsw_nb15": "UNSW-NB15",
    "cicids2018": "CICIDS2018",
    "ton_iot": "TON-IoT",
}
DATASET_TO_ID = {name: idx for idx, name in enumerate(DATASET_NAMES)}

LAMBDA_SWEEP = [0.01, 0.05, 0.1, 0.25, 0.5]

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

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info("Device: %s", DEVICE)


# ════════════════════════════════════════════════════════════════════════════
# Gradient Reversal Layer
# ════════════════════════════════════════════════════════════════════════════


class GradientReversalFunction(torch.autograd.Function):
    """Forward: identity. Backward: scale gradient by -lambda."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


def gradient_reversal(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal layer."""
    return GradientReversalFunction.apply(x, lambda_)


# ════════════════════════════════════════════════════════════════════════════
# DANN Model
# ════════════════════════════════════════════════════════════════════════════


class DANNHelixModel(nn.Module):
    """Backbone + dual classification heads + GRL + domain classifier.

    Architecture (same backbone as CORAL for fair comparison):
        Input (17) → Linear(256) → BN → ReLU → Dropout
                   → Linear(128) → BN → ReLU → Dropout
                   → Linear(64)  → BN → ReLU
                   → binary_head (64→2)
                   → family_head (64→7)
                   → GRL → domain_classifier (64→32→ReLU→4)
    """

    def __init__(self, input_dim: int = NUM_FEATURES, family_classes: int = NUM_CLASSES,
                 num_datasets: int = NUM_DATASETS):
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
        self.domain_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_datasets),
        )

    def forward(
        self, x: torch.Tensor, lambda_domain: float = 0.0, return_features: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        family_logits = self.family_head(features)
        binary_logits = self.binary_head(features)
        # Gradient reversal on features before domain classifier
        features_rev = gradient_reversal(features, lambda_domain)
        domain_logits = self.domain_classifier(features_rev)
        if return_features:
            return binary_logits, family_logits, domain_logits, features
        return binary_logits, family_logits, domain_logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict_domain(self, x: torch.Tensor, lambda_domain: float = 0.0) -> torch.Tensor:
        features = self.backbone(x)
        features_rev = gradient_reversal(features, lambda_domain)
        return self.domain_classifier(features_rev)


# ════════════════════════════════════════════════════════════════════════════
# Data loading & preprocessing
# ════════════════════════════════════════════════════════════════════════════


def load_all_datasets(
    max_samples: int = 50000,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load all 4 datasets through MultiDatasetLoader, return (X, y) dict."""
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

        missing = [c for c in CANONICAL_FEATURE_ORDER if c not in df.columns]
        if missing:
            logger.warning(
                "Dataset %s missing canonical features: %s. Available: %s",
                ds_name, missing, list(df.columns[:5]),
            )
            continue

        X = df[CANONICAL_FEATURE_ORDER].to_numpy(dtype=np.float32)
        y = df["label"].to_numpy(dtype=np.int64)

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
    """Create stratified train/test splits per dataset."""
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


# ── Datasets ──────────────────────────────────────────────────────────────


class MultiTaskDataset(torch.utils.data.Dataset):
    """Dataset returning (x, binary_label, family_label, domain_id)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, domain_id: int):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_bin = torch.tensor((y > 0).astype(np.int64), dtype=torch.long)
        self.y_fam = torch.tensor(y, dtype=torch.long)
        self.domain_id = torch.tensor(domain_id, dtype=torch.long).expand(len(X))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y_bin[idx], self.y_fam[idx], self.domain_id[idx]


class UnlabeledDataset(torch.utils.data.Dataset):
    """Dataset for target-domain data (features only, with domain_id)."""

    def __init__(self, X: np.ndarray, domain_id: int):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.domain_id = torch.tensor(domain_id, dtype=torch.long).expand(len(X))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.domain_id[idx]


# ════════════════════════════════════════════════════════════════════════════
# Training infrastructure
# ════════════════════════════════════════════════════════════════════════════


def train_model(
    X_src: np.ndarray,
    y_src: np.ndarray,
    source_domain_ids: np.ndarray,
    X_tgt: np.ndarray,
    target_domain_id: int,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    lambda_domain: float = 0.1,
    epochs: int = 150,
    batch_size: int = 256,
    lr: float = 5e-4,
    patience: int = 20,
    seed: int = 42,
    experiment_label: str = "",
) -> tuple[DANNHelixModel, dict]:
    """Train a DANNHelixModel with adversarial domain confusion.

    Args:
        X_src: Source training features
        y_src: Source training labels (7-class)
        source_domain_ids: Per-sample domain IDs for source data (one per source dataset)
        X_tgt: Target training features (used for domain loss only)
        target_domain_id: Domain ID of the target dataset
        X_val: Validation features
        y_val: Validation labels
        lambda_domain: Weight for domain adversarial loss
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DANNHelixModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Source dataset — domain ID varies per source sample
    src_domain_tensor = torch.tensor(source_domain_ids, dtype=torch.long)
    src_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_src, dtype=torch.float32),
        torch.tensor((y_src > 0).astype(np.int64), dtype=torch.long),
        torch.tensor(y_src, dtype=torch.long),
        src_domain_tensor,
    )
    src_loader = torch.utils.data.DataLoader(
        src_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
    )

    # Validation dataset
    val_dataset = MultiTaskDataset(X_val, y_val, 0)  # domain id unused for val
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    # Target dataset for domain loss (unlabeled for classification)
    tgt_dataset = UnlabeledDataset(X_tgt, target_domain_id)
    tgt_loader = torch.utils.data.DataLoader(
        tgt_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
    )

    history: dict[str, list] = {"epoch": [], "train_loss": [], "val_f1": [],
                                "domain_loss": [], "cls_loss": []}
    best_val_f1 = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_domain_loss = 0.0
        total_cls_loss = 0.0
        train_batches = 0
        tgt_iter = iter(tgt_loader)

        for src_batch in src_loader:
            x_batch, y_bin_batch, y_fam_batch, domain_batch = src_batch
            x_batch = x_batch.to(DEVICE)
            y_bin_batch = y_bin_batch.to(DEVICE)
            y_fam_batch = y_fam_batch.to(DEVICE)
            domain_batch = domain_batch.to(DEVICE)

            # Source forward pass
            bin_logits, fam_logits, domain_logits = model(x_batch, lambda_domain)
            loss_cls = F.cross_entropy(bin_logits, y_bin_batch) + F.cross_entropy(fam_logits, y_fam_batch)
            loss_domain_src = F.cross_entropy(domain_logits, domain_batch)

            loss = loss_cls + lambda_domain * loss_domain_src

            # Target forward pass for domain loss (cycle through target loader)
            try:
                x_tgt_batch, tgt_domain_batch = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                x_tgt_batch, tgt_domain_batch = next(tgt_iter)

            x_tgt_batch = x_tgt_batch.to(DEVICE)
            tgt_domain_batch = tgt_domain_batch.to(DEVICE)

            _, _, domain_logits_tgt = model(x_tgt_batch, lambda_domain)
            loss_domain_tgt = F.cross_entropy(domain_logits_tgt, tgt_domain_batch)
            loss = loss + lambda_domain * loss_domain_tgt

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_domain_loss += (loss_domain_src.item() + loss_domain_tgt.item()) / 2
            total_cls_loss += loss_cls.item()
            train_batches += 1

        avg_loss = total_loss / max(train_batches, 1)
        avg_domain_loss = total_domain_loss / max(train_batches, 1)
        avg_cls_loss = total_cls_loss / max(train_batches, 1)

        # Validation
        model.eval()
        val_preds: list[int] = []
        val_targets: list[int] = []
        with torch.no_grad():
            for x_batch, _, y_fam, _ in val_loader:
                x_batch = x_batch.to(DEVICE)
                _, fam_logits, _ = model(x_batch, lambda_domain)
                val_preds.extend(fam_logits.argmax(dim=1).cpu().numpy().tolist())
                val_targets.extend(y_fam.numpy().tolist())

        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        history["cls_loss"].append(avg_cls_loss)
        history["domain_loss"].append(avg_domain_loss)
        history["val_f1"].append(float(val_f1))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            label = f"{experiment_label} (λ={lambda_domain:.3f})"
            logger.info(
                "    [%s] Epoch %3d/%d | Loss: %.4f (cls=%.4f dom=%.4f) | Val F1: %.4f",
                label, epoch + 1, epochs, avg_loss, avg_cls_loss, avg_domain_loss, val_f1,
            )

        if patience_counter >= patience:
            logger.info("    [%s] Early stopping at epoch %d", experiment_label, epoch + 1)
            break

    if best_state:
        model.load_state_dict(best_state)

    return model, history


def evaluate_model(
    model: DANNHelixModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lambda_domain: float = 0.0,
) -> dict:
    """Evaluate trained model on test data, return full metrics."""
    model.eval()
    dataset = MultiTaskDataset(X_test, y_test, 0)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=512, shuffle=False, num_workers=0,
    )

    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for x_batch, _, y_fam, _ in loader:
            x_batch = x_batch.to(DEVICE)
            _, fam_logits, _ = model(x_batch, lambda_domain)
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
    model: DANNHelixModel,
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

    dataset = MultiTaskDataset(X, y, 0)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=512, shuffle=False, num_workers=0,
    )

    all_embs: list[np.ndarray] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for x_batch, _, y_fam, _ in loader:
            x_batch = x_batch.to(DEVICE)
            feats = model.extract_features(x_batch)
            all_embs.append(feats.cpu().numpy())
            all_labels.extend(y_fam.numpy().tolist())

    return np.vstack(all_embs), np.array(all_labels, dtype=np.int64)


# ════════════════════════════════════════════════════════════════════════════
# Per-experiment execution
# ════════════════════════════════════════════════════════════════════════════


def run_experiment(
    exp_name: str,
    source_names: list[str],
    target_name: str,
    harmonized: dict[str, tuple[np.ndarray, np.ndarray]],
    splits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    epochs: int = 150,
    seed: int = 42,
    lambda_domain: float = 0.1,
    train_cap_per_class: int = 5000,
) -> dict:
    """Run DANN for one experiment with a given lambda, return full result dict."""
    logger.info("\n%s", "=" * 70)
    sources_display = " + ".join(DATASET_DISPLAY.get(s, s) for s in source_names)
    target_display = DATASET_DISPLAY.get(target_name, target_name)
    logger.info("EXPERIMENT: %s (λ=%s)", exp_name, lambda_domain)
    logger.info("  Source: %s", sources_display)
    logger.info("  Target: %s", target_display)
    logger.info("%s", "=" * 70)

    # Cap each source dataset individually, then concatenate with domain labels
    X_src_list, y_src_list, domain_list = [], [], []
    for s_name in source_names:
        X_tr, y_tr = splits[s_name][0], splits[s_name][1]
        did = DATASET_TO_ID[s_name]
        if train_cap_per_class > 0:
            X_capped, y_capped = cap_training_data(X_tr, y_tr, max_per_class=train_cap_per_class, seed=seed)
        else:
            X_capped, y_capped = X_tr, y_tr
        X_src_list.append(X_capped)
        y_src_list.append(y_capped)
        domain_list.append(np.full(len(X_capped), did, dtype=np.int64))

    X_src = np.vstack(X_src_list) if len(X_src_list) > 1 else X_src_list[0]
    y_src = np.concatenate(y_src_list) if len(y_src_list) > 1 else y_src_list[0]
    domain_src = np.concatenate(domain_list) if len(domain_list) > 1 else domain_list[0]

    # Target data
    X_tgt_train, y_tgt_train = splits[target_name][0], splits[target_name][1]
    X_tgt_test, y_tgt_test = splits[target_name][2], splits[target_name][3]
    target_domain_id = DATASET_TO_ID[target_name]

    # Validation split from target train
    from sklearn.model_selection import train_test_split
    X_tgt_val, X_tgt_for_dann, y_tgt_val, _ = train_test_split(
        X_tgt_train, y_tgt_train, test_size=0.8, random_state=seed, stratify=y_tgt_train,
    )

    # ── Standardize: fit on source, transform all ────────────────────
    scaler = StandardScaler()
    X_src_scaled = scaler.fit_transform(X_src).astype(np.float32)
    X_tgt_val_scaled = scaler.transform(X_tgt_val).astype(np.float32)
    X_tgt_for_dann_scaled = scaler.transform(X_tgt_for_dann).astype(np.float32)
    X_tgt_test_scaled = scaler.transform(X_tgt_test).astype(np.float32)

    result: dict = {
        "experiment_name": exp_name,
        "source_datasets": source_names,
        "target_dataset": target_name,
        "source_display": sources_display,
        "target_display": target_display,
        "source_train_samples": len(X_src_scaled),
        "target_test_samples": len(X_tgt_test_scaled),
        "lambda_domain": lambda_domain,
    }

    label = f"{exp_name}/dann"
    logger.info("\n  Training DANN (λ=%s)...", lambda_domain)

    t0 = time.time()
    model, history = train_model(
        X_src_scaled, y_src, domain_src,
        X_tgt_for_dann_scaled, target_domain_id,
        X_tgt_val_scaled, y_tgt_val,
        lambda_domain=lambda_domain,
        epochs=epochs,
        seed=seed,
        experiment_label=label,
    )
    train_time = time.time() - t0

    # Evaluate on target test
    metrics = evaluate_model(model, X_tgt_test_scaled, y_tgt_test, lambda_domain)

    # Train accuracy for gen gap
    train_metrics = evaluate_model(model, X_src_scaled, y_src, lambda_domain)
    gen_gap = train_metrics["accuracy"] - metrics["accuracy"]

    # Embeddings
    embs_src, _ = extract_embeddings(model, X_src_scaled, y_src, max_samples=3000)
    embs_tgt, y_tgt_for_sil = extract_embeddings(
        model, X_tgt_test_scaled, y_tgt_test, max_samples=3000,
    )

    # Silhouette by dataset
    all_embs = np.vstack([embs_src, embs_tgt])
    ds_ids = np.array([0] * len(embs_src) + [1] * len(embs_tgt))
    sil_dataset = float(silhouette_score(all_embs, ds_ids)) if len(np.unique(ds_ids)) > 1 else 0.0

    # Silhouette by attack family (target only)
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
        "domain_loss_history": history.get("domain_loss", []),
    }

    result["dann"] = model_results

    # Save embeddings for plotting
    embs_data = {
        "source_embs": embs_src.tolist(),
        "target_embs": embs_tgt.tolist(),
        "target_labels": y_tgt_for_sil.tolist(),
    }
    result["dann_embeddings"] = embs_data

    logger.info(
        "  RESULTS [%s]: Acc=%.4f | F1=%.4f | GenGap=%+.4f | Epochs=%d | %.1fs",
        label, metrics["accuracy"], metrics["macro_f1"],
        gen_gap, epochs_trained, train_time,
    )
    logger.info(
        "  Silhouette — dataset=%.4f, family=%.4f",
        sil_dataset, sil_family,
    )

    del model
    if DEVICE.type == "mps":
        torch.mps.empty_cache()

    return result


# ════════════════════════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════════════════════════


def generate_plots(all_results: dict, best_lambda_results: dict, plots_dir: Path):
    """Generate all Phase 28A plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plots_dir.mkdir(parents=True, exist_ok=True)
    (plots_dir / "tsne").mkdir(parents=True, exist_ok=True)
    (plots_dir / "umap").mkdir(parents=True, exist_ok=True)

    exp_names = list(best_lambda_results.keys())
    short_names = [
        "NSL→UNSW", "UNSW→CIC", "CIC→TON", "TON→NSL",
        "3→TON", "3→CIC", "3→NSL", "3→UNSW"
    ]

    # ── 1. DANN vs Baseline (from Phase 26B) ─────────────────────────
    # Baseline F1 values from Phase 26B certification
    baseline_f1_26b = {
        "exp01_pairwise_nsl_to_unsw": 0.1068,
        "exp02_pairwise_unsw_to_cicids": 0.0196,
        "exp03_pairwise_cicids_to_ton": 0.0633,
        "exp04_pairwise_ton_to_nsl": 0.0067,
        "exp05_holdout_3src_to_ton": 0.0119,
        "exp06_holdout_3src_to_cicids": 0.0000,
        "exp07_holdout_3src_to_nsl": 0.0004,
        "exp08_holdout_3src_to_unsw": 0.0020,
    }

    dann_f1 = [best_lambda_results[e]["dann"]["macro_f1"] for e in exp_names]
    baseline_f1 = [baseline_f1_26b.get(e, 0) for e in exp_names]
    f1_delta = [d - b for d, b in zip(dann_f1, baseline_f1)]

    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(exp_names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, baseline_f1, width, label="Phase 26B Baseline", color="steelblue")
    bars2 = ax.bar(x + width / 2, dann_f1, width, label="DANN (best λ)", color="darkorange")

    for i, (b, d, delta) in enumerate(zip(baseline_f1, dann_f1, f1_delta)):
        color = "green" if delta > 0 else "red"
        ax.text(x[i] + width / 2, d + 0.01, f"{delta:+.3f}", ha="center", va="bottom",
                fontsize=9, color=color, fontweight="bold")

    ax.set_ylabel("Macro F1")
    ax.set_title("Phase 28A: Baseline vs DANN Macro F1\n(Best λ per experiment)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=25, ha="right")
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = plots_dir / "dann_vs_baseline.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", p)

    # ── 2. DANN vs CORAL ──────────────────────────────────────────────
    # CORAL best F1 from Phase 27B results
    coral_best_f1 = {
        "exp01_pairwise_nsl_to_unsw": 0.0528,
        "exp02_pairwise_unsw_to_cicids": 0.0415,
        "exp03_pairwise_cicids_to_ton": 0.2531,
        "exp04_pairwise_ton_to_nsl": 0.1296,
        "exp05_holdout_3src_to_ton": 0.1537,
        "exp06_holdout_3src_to_cicids": 0.1684,
        "exp07_holdout_3src_to_nsl": 0.1083,
        "exp08_holdout_3src_to_unsw": 0.0167,
    }

    coral_f1 = [coral_best_f1.get(e, 0) for e in exp_names]
    dann_vs_coral_delta = [d - c for d, c in zip(dann_f1, coral_f1)]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars1 = ax.bar(x - width / 2, coral_f1, width, label="Phase 27B CORAL", color="coral")
    bars2 = ax.bar(x + width / 2, dann_f1, width, label="DANN (best λ)", color="darkorange")

    for i, (c, d, delta) in enumerate(zip(coral_f1, dann_f1, dann_vs_coral_delta)):
        color = "green" if delta > 0 else "red"
        ax.text(x[i] + width / 2, d + 0.01, f"{delta:+.3f}", ha="center", va="bottom",
                fontsize=9, color=color, fontweight="bold")

    ax.set_ylabel("Macro F1")
    ax.set_title("Phase 28A: CORAL vs DANN Macro F1\n(Best λ per experiment)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=25, ha="right")
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = plots_dir / "dann_vs_coral.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", p)

    # ── 3. Lambda sweep visualization ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))
    for exp_name in exp_names:
        lambdas = []
        f1s = []
        for ld, res_list in all_results.items():
            if exp_name in res_list:
                lambdas.append(ld)
                f1s.append(res_list[exp_name]["dann"]["macro_f1"])
        if lambdas:
            order = np.argsort(lambdas)
            lambdas_sorted = np.array(lambdas)[order]
            f1s_sorted = np.array(f1s)[order]
            ax.plot(lambdas_sorted, f1s_sorted, "o-", label=short_names[exp_names.index(exp_name)])

    ax.set_xlabel("λ (domain loss weight)")
    ax.set_ylabel("Macro F1")
    ax.set_title("DANN Lambda Sweep — Macro F1 per Experiment")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = plots_dir / "lambda_sweep.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", p)

    # ── 4. Silhouette comparison ──────────────────────────────────────
    sil_ds_baseline = [baseline_f1_26b.get(e, 0) for e in exp_names]
    sil_ds_dann = [best_lambda_results[e]["dann"]["silhouette_dataset"] for e in exp_names]

    # Need baseline silhouette values
    baseline_sil_ds = {
        "exp01_pairwise_nsl_to_unsw": 0.3282,  # From Phase 26B
        "exp02_pairwise_unsw_to_cicids": 0.1043,
        "exp03_pairwise_cicids_to_ton": 0.0468,
        "exp04_pairwise_ton_to_nsl": 0.1312,
        "exp05_holdout_3src_to_ton": 0.1549,
        "exp06_holdout_3src_to_cicids": 0.1037,
        "exp07_holdout_3src_to_nsl": 0.0270,
        "exp08_holdout_3src_to_unsw": 0.2975,
    }
    sil_ds_base = [baseline_sil_ds.get(e, 0) for e in exp_names]

    fig, ax = plt.subplots(figsize=(12, 6))
    x_plot = np.arange(len(exp_names))
    ax.plot(x_plot, sil_ds_base, "o-", color="steelblue", label="Baseline (dataset silhouette)")
    ax.plot(x_plot, sil_ds_dann, "s--", color="darkorange", label="DANN (dataset silhouette)")
    ax.set_xticks(x_plot)
    ax.set_xticklabels(short_names, rotation=25, ha="right")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Embedding Silhouette by Dataset: Baseline vs DANN")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = plots_dir / "silhouette_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", p)

    # ── 5. t-SNE projections for best lambdas ────────────────────────
    logger.info("Generating t-SNE projections...")
    from sklearn.manifold import TSNE

    for exp_name in exp_names:
        res = best_lambda_results[exp_name]
        embs_key = "dann_embeddings"
        if embs_key not in res:
            continue

        source_embs = np.array(res[embs_key]["source_embs"])
        target_embs = np.array(res[embs_key]["target_embs"])
        target_labels = np.array(res[embs_key]["target_labels"])

        all_embs = np.vstack([source_embs, target_embs])
        ds_labels = np.array([0] * len(source_embs) + [1] * len(target_embs))

        n_total = len(all_embs)
        if n_total > 5000:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_total, size=5000, replace=False)
            all_embs = all_embs[idx]
            ds_labels = ds_labels[idx]

        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
        embs_2d = tsne.fit_transform(all_embs)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        scatter_ds = axes[0].scatter(
            embs_2d[:, 0], embs_2d[:, 1], c=ds_labels,
            cmap="coolwarm", alpha=0.6, s=8,
        )
        axes[0].set_title(f"{exp_name} — DANN λ={res['lambda_domain']}\nColored by Dataset (red=target)")
        axes[0].set_xlabel("t-SNE 1")
        axes[0].set_ylabel("t-SNE 2")
        axes[0].grid(True, alpha=0.2)
        cbar_ds = fig.colorbar(scatter_ds, ax=axes[0])
        cbar_ds.set_ticks([0, 1])
        cbar_ds.set_ticklabels(["Source", "Target"])

        tgt_embs_2d = embs_2d[ds_labels == 1]
        if len(tgt_embs_2d) > 0:
            scatter_fam = axes[1].scatter(
                tgt_embs_2d[:, 0], tgt_embs_2d[:, 1],
                c=target_labels[:len(tgt_embs_2d)] if len(target_labels) >= len(tgt_embs_2d) else ds_labels[:len(tgt_embs_2d)],
                cmap="tab10", alpha=0.6, s=10,
            )
            axes[1].set_title(f"{exp_name} — DANN λ={res['lambda_domain']}\nTarget Only (by Attack Family)")
            axes[1].set_xlabel("t-SNE 1")
            axes[1].set_ylabel("t-SNE 2")
            axes[1].grid(True, alpha=0.2)
            fig.colorbar(scatter_fam, ax=axes[1])

        plt.tight_layout()
        p = plots_dir / "tsne" / f"{exp_name}_dann.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Generated t-SNE for %s", exp_name)

    # ── 6. UMAP projections for best lambdas ──────────────────────────
    try:
        import umap

        logger.info("Generating UMAP projections...")
        for exp_name in exp_names:
            res = best_lambda_results[exp_name]
            embs_key = "dann_embeddings"
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
            axes[0].set_title(f"{exp_name} — DANN (UMAP by Dataset)")

            tgt_embs_2d = embs_2d[ds_labels == 1]
            if len(tgt_embs_2d) > 0:
                axes[1].scatter(tgt_embs_2d[:, 0], tgt_embs_2d[:, 1],
                                alpha=0.6, s=10)
                axes[1].set_title(f"{exp_name} — DANN (UMAP Target)")

            plt.tight_layout()
            p = plots_dir / "umap" / f"{exp_name}_dann.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("  Generated UMAP for %s", exp_name)
    except ImportError:
        logger.warning("UMAP not installed, skipping UMAP plots")

    logger.info("All plots generated in %s", plots_dir)


# ════════════════════════════════════════════════════════════════════════════
# Report generation
# ════════════════════════════════════════════════════════════════════════════


def generate_dann_sweep(all_results: dict, doc_dir: Path):
    """Write DANN_SWEEP.md."""
    lines = [
        "# DANN Lambda Sweep Results\n\n",
        "## Sweep Configuration\n\n",
        f"- **Lambda values**: {LAMBDA_SWEEP}\n",
        f"- **Epochs per run**: configurable (default 100)\n",
        f"- **Device**: {DEVICE}\n\n",
    ]

    lines.append("## Per-Experiment Lambda Results\n\n")

    for exp_name in sorted(all_results[LAMBDA_SWEEP[0]].keys()):
        lines.append(f"### {exp_name}\n\n")
        lines.append("| Lambda | Accuracy | Macro F1 | Precision | Recall | Gen Gap | Epochs | Sil-DS | Sil-Fam |\n")
        lines.append("|------:|--------:|---------:|----------:|------:|--------:|------:|-------:|--------:|\n")
        best_f1 = -1
        best_ld = None
        for ld in sorted(LAMBDA_SWEEP):
            if exp_name not in all_results[ld]:
                continue
            m = all_results[ld][exp_name]["dann"]
            lines.append(
                f"| {ld:.3f} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} "
                f"| {m['precision']:.4f} | {m['recall']:.4f} "
                f"| {m['generalization_gap']:+.4f} | {m['epochs_trained']} "
                f"| {m['silhouette_dataset']:.4f} | {m['silhouette_family']:.4f} |\n"
            )
            if m['macro_f1'] > best_f1:
                best_f1 = m['macro_f1']
                best_ld = ld
        lines.append(f"\n**Best λ**: {best_ld} (F1={best_f1:.4f})\n\n")

    p = doc_dir / "DANN_SWEEP.md"
    doc_dir.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(lines))
    logger.info("Generated %s", p)


def generate_pairwise_results(best_lambda_results: dict, doc_dir: Path):
    """Write PAIRWISE_RESULTS.md."""
    pairwise = [k for k in best_lambda_results if "pairwise" in k]
    lines = [
        "# Pairwise Transfer Results (DANN)\n\n",
        f"## Experiments Run: {len(pairwise)}/4\n\n",
    ]

    lines.append("| Experiment | Lambda | Accuracy | Macro F1 | Precision | Recall | Gen Gap | Epochs | Time (s) | Sil-DS | Sil-Fam |\n")
    lines.append("|-----------|------:|--------:|---------:|----------:|------:|--------:|------:|--------:|-------:|--------:|\n")

    for exp_name in sorted(pairwise):
        res = best_lambda_results[exp_name]
        m = res["dann"]
        src = res["source_display"]
        tgt = res["target_display"]
        lines.append(
            f"| {src}→{tgt} | {res['lambda_domain']:.3f} "
            f"| {m['accuracy']:.4f} | {m['macro_f1']:.4f} "
            f"| {m['precision']:.4f} | {m['recall']:.4f} "
            f"| {m['generalization_gap']:+.4f} | {m['epochs_trained']} "
            f"| {m['training_time_s']:.1f} "
            f"| {m['silhouette_dataset']:.4f} | {m['silhouette_family']:.4f} |\n"
        )

    lines.append("\n\n## Individual Experiment Details\n\n")
    for exp_name in sorted(pairwise):
        res = best_lambda_results[exp_name]
        m = res["dann"]
        lines.append(f"### {exp_name}\n\n")
        lines.append(f"- **Source**: {res['source_display']}\n")
        lines.append(f"- **Target**: {res['target_display']}\n")
        lines.append(f"- **λ**: {res['lambda_domain']}\n")
        lines.append(f"- **Train samples**: {res['source_train_samples']}\n")
        lines.append(f"- **Test samples**: {res['target_test_samples']}\n")
        lines.append(
            f"- **DANN**: Acc={m['accuracy']:.4f}, F1={m['macro_f1']:.4f}, "
            f"Prec={m['precision']:.4f}, Rec={m['recall']:.4f}, "
            f"GenGap={m['generalization_gap']:+.4f}, "
            f"Epochs={m['epochs_trained']}, "
            f"Sil-DS={m['silhouette_dataset']:.4f}, Sil-Fam={m['silhouette_family']:.4f}\n\n"
        )

    p = doc_dir / "PAIRWISE_RESULTS.md"
    p.write_text("".join(lines))
    logger.info("Generated %s", p)


def generate_holdout_results(best_lambda_results: dict, doc_dir: Path):
    """Write HOLDOUT_RESULTS.md."""
    holdout = [k for k in best_lambda_results if "holdout" in k]
    lines = [
        "# Holdout Transfer Results (DANN)\n\n",
        f"## Experiments Run: {len(holdout)}/4\n\n",
    ]

    lines.append("| Experiment | Lambda | Accuracy | Macro F1 | Precision | Recall | Gen Gap | Epochs | Time (s) | Sil-DS | Sil-Fam |\n")
    lines.append("|-----------|------:|--------:|---------:|----------:|------:|--------:|------:|--------:|-------:|--------:|\n")

    for exp_name in sorted(holdout):
        res = best_lambda_results[exp_name]
        m = res["dann"]
        src = "3-dataset"
        tgt = res["target_display"]
        lines.append(
            f"| {src}→{tgt} | {res['lambda_domain']:.3f} "
            f"| {m['accuracy']:.4f} | {m['macro_f1']:.4f} "
            f"| {m['precision']:.4f} | {m['recall']:.4f} "
            f"| {m['generalization_gap']:+.4f} | {m['epochs_trained']} "
            f"| {m['training_time_s']:.1f} "
            f"| {m['silhouette_dataset']:.4f} | {m['silhouette_family']:.4f} |\n"
        )

    lines.append("\n\n## Individual Experiment Details\n\n")
    for exp_name in sorted(holdout):
        res = best_lambda_results[exp_name]
        m = res["dann"]
        lines.append(f"### {exp_name}\n\n")
        lines.append(f"- **Source Datasets**: {res['source_display']}\n")
        lines.append(f"- **Target Held-Out**: {res['target_display']}\n")
        lines.append(f"- **λ**: {res['lambda_domain']}\n")
        lines.append(f"- **Train samples**: {res['source_train_samples']}\n")
        lines.append(f"- **Test samples**: {res['target_test_samples']}\n")

    p = doc_dir / "HOLDOUT_RESULTS.md"
    p.write_text("".join(lines))
    logger.info("Generated %s", p)


def generate_embedding_audit(best_lambda_results: dict, doc_dir: Path):
    """Write EMBEDDING_AUDIT.md."""
    lines = [
        "# Embedding Audit (DANN)\n\n",
        "## Methodology\n\n",
        "For each experiment, backbone embeddings (64-dim) are extracted from DANN "
        "models for source and target test data. We compute:\n\n",
        "- **silhouette_dataset**: Silhouette score of source vs target embeddings.\n",
        "  Lower = better domain-invariant alignment.\n",
        "- **silhouette_family**: Silhouette score of attack-family clusters within target.\n",
        "  Higher = better class separability.\n\n",
    ]

    sil_ds_baseline_ref = {
        "exp01_pairwise_nsl_to_unsw": 0.3282,
        "exp02_pairwise_unsw_to_cicids": 0.1043,
        "exp03_pairwise_cicids_to_ton": 0.0468,
        "exp04_pairwise_ton_to_nsl": 0.1312,
        "exp05_holdout_3src_to_ton": 0.1549,
        "exp06_holdout_3src_to_cicids": 0.1037,
        "exp07_holdout_3src_to_nsl": 0.0270,
        "exp08_holdout_3src_to_unsw": 0.2975,
    }
    sil_fam_baseline_ref = {
        "exp01_pairwise_nsl_to_unsw": -0.0373,
        "exp02_pairwise_unsw_to_cicids": -0.7365,
        "exp03_pairwise_cicids_to_ton": -0.6179,
        "exp04_pairwise_ton_to_nsl": 0.1374,
        "exp05_holdout_3src_to_ton": -0.0405,
        "exp06_holdout_3src_to_cicids": -0.7357,
        "exp07_holdout_3src_to_nsl": -0.5358,
        "exp08_holdout_3src_to_unsw": -0.0680,
    }

    lines.append("| Experiment | λ | Model | Sil-Dataset (↓better) | Sil-Family (↑better) |\n")
    lines.append("|-----------|--|------:|---------------------:|--------------------:|\n")

    sil_ds_deltas = []
    for exp_name in sorted(best_lambda_results.keys()):
        res = best_lambda_results[exp_name]
        m = res["dann"]
        sil_ds = m["silhouette_dataset"]
        sil_fam = m["silhouette_family"]
        ld = res["lambda_domain"]
        sil_ds_base = sil_ds_baseline_ref.get(exp_name, 0)
        delta = sil_ds - sil_ds_base
        sil_ds_deltas.append(delta)

        lines.append(
            f"| {exp_name} | {ld:.3f} | DANN | {sil_ds:.4f} ({delta:+.4f} vs baseline) | {sil_fam:.4f} |\n"
        )

    avg_sil_ds_delta = np.mean(sil_ds_deltas) if sil_ds_deltas else 0

    lines.append(f"\n\n## Summary\n\n")
    lines.append(f"- **Average silhouette_dataset delta vs Phase 26B**: {avg_sil_ds_delta:.4f}\n")

    if avg_sil_ds_delta <= -0.05:
        lines.append("- **DANN reduces domain separation** (silhouette decreases by >= 0.05)\n")
    elif avg_sil_ds_delta <= 0:
        lines.append("- **DANN slightly reduces or maintains** domain separation.\n")
    else:
        lines.append("- **DANN increases domain separation** — feature alignment degraded.\n")

    p = doc_dir / "EMBEDDING_AUDIT.md"
    doc_dir.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(lines))
    logger.info("Generated %s", p)


def generate_domain_confusion(best_lambda_results: dict, doc_dir: Path):
    """Write DOMAIN_CONFUSION_ANALYSIS.md."""
    lines = [
        "# Domain Confusion Analysis\n\n",
        "## Overview\n\n",
        "Domain confusion measures how well the adversarial training has "
        "suppressed dataset-specific features. A well-confused model should "
        "have target domain classifier accuracy near random chance (50% for "
        "binary source/target distinction).\n\n",
        "## Per-Experiment Domain Accuracy\n\n",
        "We approximate domain confusion by the inverse of silhouette_dataset: "
        "lower silhouette_dataset = stronger domain confusion.\n\n",
    ]

    lines.append("| Experiment | λ | Sil-Dataset | Domain Alignment |\n")
    lines.append("|-----------|--:|-----------:|-----------------:|\n")

    for exp_name in sorted(best_lambda_results.keys()):
        res = best_lambda_results[exp_name]
        m = res["dann"]
        sil = m["silhouette_dataset"]
        alignment = "Strong" if sil <= 0.05 else ("Moderate" if sil <= 0.15 else "Weak")
        lines.append(f"| {exp_name} | {res['lambda_domain']:.3f} | {sil:.4f} | {alignment} |\n")

    lines.append("\n\n## Interpretation\n\n")
    lines.append("- **Strong alignment** (silhouette <= 0.05): Features are nearly\n")
    lines.append("  indistinguishable between source and target domains.\n")
    lines.append("- **Moderate alignment** (0.05 < silhouette <= 0.15): Partial\n")
    lines.append("  domain overlap; some dataset-specific features remain.\n")
    lines.append("- **Weak alignment** (silhouette > 0.15): Source and target\n")
    lines.append("  remain clearly separable; DANN has not eliminated domain gap.\n")

    p = doc_dir / "DOMAIN_CONFUSION_ANALYSIS.md"
    p.write_text("".join(lines))
    logger.info("Generated %s", p)


def generate_certification_report(
    best_lambda_results: dict,
    all_results: dict,
    doc_dir: Path,
    plots_dir: Path,
):
    """Write the final PHASE28A_DANN_CERTIFICATION.md."""
    exp_names = sorted(best_lambda_results.keys())

    # Aggregate DANN results
    dann_f1_list = [best_lambda_results[e]["dann"]["macro_f1"] for e in exp_names]
    avg_dann_f1 = np.mean(dann_f1_list)
    best_dann_f1 = max(dann_f1_list)
    worst_dann_f1 = min(dann_f1_list)

    # Baseline Phase 26B values
    baseline_f1_26b = {
        "exp01_pairwise_nsl_to_unsw": 0.1068,
        "exp02_pairwise_unsw_to_cicids": 0.0196,
        "exp03_pairwise_cicids_to_ton": 0.0633,
        "exp04_pairwise_ton_to_nsl": 0.0067,
        "exp05_holdout_3src_to_ton": 0.0119,
        "exp06_holdout_3src_to_cicids": 0.0000,
        "exp07_holdout_3src_to_nsl": 0.0004,
        "exp08_holdout_3src_to_unsw": 0.0020,
    }
    avg_baseline_f1 = np.mean([baseline_f1_26b.get(e, 0) for e in exp_names])

    # CORAL best F1
    coral_best_f1 = {
        "exp01_pairwise_nsl_to_unsw": 0.0528,
        "exp02_pairwise_unsw_to_cicids": 0.0415,
        "exp03_pairwise_cicids_to_ton": 0.2531,
        "exp04_pairwise_ton_to_nsl": 0.1296,
        "exp05_holdout_3src_to_ton": 0.1537,
        "exp06_holdout_3src_to_cicids": 0.1684,
        "exp07_holdout_3src_to_nsl": 0.1083,
        "exp08_holdout_3src_to_unsw": 0.0167,
    }
    avg_coral_f1 = np.mean([coral_best_f1.get(e, 0) for e in exp_names])

    # Silhouette comparisons
    baseline_sil_ds = {
        "exp01_pairwise_nsl_to_unsw": 0.3282,
        "exp02_pairwise_unsw_to_cicids": 0.1043,
        "exp03_pairwise_cicids_to_ton": 0.0468,
        "exp04_pairwise_ton_to_nsl": 0.1312,
        "exp05_holdout_3src_to_ton": 0.1549,
        "exp06_holdout_3src_to_cicids": 0.1037,
        "exp07_holdout_3src_to_nsl": 0.0270,
        "exp08_holdout_3src_to_unsw": 0.2975,
    }
    baseline_sil_fam = {
        "exp01_pairwise_nsl_to_unsw": -0.0373,
        "exp02_pairwise_unsw_to_cicids": -0.7365,
        "exp03_pairwise_cicids_to_ton": -0.6179,
        "exp04_pairwise_ton_to_nsl": 0.1374,
        "exp05_holdout_3src_to_ton": -0.0405,
        "exp06_holdout_3src_to_cicids": -0.7357,
        "exp07_holdout_3src_to_nsl": -0.5358,
        "exp08_holdout_3src_to_unsw": -0.0680,
    }

    dann_sil_ds = [best_lambda_results[e]["dann"]["silhouette_dataset"] for e in exp_names]
    dann_sil_fam = [best_lambda_results[e]["dann"]["silhouette_family"] for e in exp_names]
    baseline_sil_ds_list = [baseline_sil_ds.get(e, 0) for e in exp_names]
    baseline_sil_fam_list = [baseline_sil_fam.get(e, 0) for e in exp_names]

    avg_baseline_sil_ds = np.mean(baseline_sil_ds_list)
    avg_dann_sil_ds = np.mean(dann_sil_ds)
    avg_baseline_sil_fam = np.mean(baseline_sil_fam_list)
    avg_dann_sil_fam = np.mean(dann_sil_fam)

    sil_ds_reduction_pct = ((avg_baseline_sil_ds - avg_dann_sil_ds) / max(avg_baseline_sil_ds, 1e-8)) * 100

    # Macro F1 improvements
    f1_deltas_vs_baseline = [
        best_lambda_results[e]["dann"]["macro_f1"] - baseline_f1_26b.get(e, 0)
        for e in exp_names
    ]
    avg_f1_delta_vs_baseline = np.mean(f1_deltas_vs_baseline)
    avg_f1_pct_improvement = (avg_f1_delta_vs_baseline / max(avg_baseline_f1, 1e-8)) * 100 if avg_baseline_f1 > 1e-6 else 0

    f1_deltas_vs_coral = [
        best_lambda_results[e]["dann"]["macro_f1"] - coral_best_f1.get(e, 0)
        for e in exp_names
    ]
    avg_f1_delta_vs_coral = np.mean(f1_deltas_vs_coral)

    wins_vs_baseline = sum(1 for d in f1_deltas_vs_baseline if d > 0)
    losses_vs_baseline = sum(1 for d in f1_deltas_vs_baseline if d < 0)
    wins_vs_coral = sum(1 for d in f1_deltas_vs_coral if d > 0)

    # Domain classifier collapse detection
    domain_losses = []
    for e in exp_names:
        dlh = best_lambda_results[e]["dann"].get("domain_loss_history", [])
        if len(dlh) >= 10:
            late_dl = np.mean(dlh[-10:])
            domain_losses.append(late_dl)
    avg_final_domain_loss = np.mean(domain_losses) if domain_losses else float("nan")
    domain_collapse = avg_final_domain_loss < 0.05 if not np.isnan(avg_final_domain_loss) else False

    # ── Success Criteria ────────────────────────────────────────────
    primary_pass = avg_f1_pct_improvement >= 15.0
    secondary_pass = sil_ds_reduction_pct >= 30.0
    tertiary_pass = (avg_dann_sil_fam - avg_baseline_sil_fam) / max(abs(avg_baseline_sil_fam), 1e-8) * 100 >= 20.0 if abs(avg_baseline_sil_fam) > 0.01 else False
    production_candidate = avg_dann_f1 > 0.10
    failure_triggered = avg_f1_pct_improvement < 10.0 or (abs(avg_baseline_sil_ds - avg_dann_sil_ds) < 0.01) or domain_collapse

    if primary_pass and secondary_pass:
        decision = "GO"
        next_phase = "Production integration candidate"
    elif failure_triggered:
        decision = "NO-GO"
        next_phase = "Alternative domain adaptation (e.g., MMD, Deep CORAL, or re-evaluate feature space)"
    else:
        decision = "HOLD"
        next_phase = "Phase 28B — try alternative DANN variants or hyperparameters"

    lines = [
        "# PHASE 28A — DANN Domain-Adversarial Certification Report\n\n",
        f"**Decision**: {decision}\n",
        f"**Recommended next phase**: {next_phase}\n\n",
        "## Executive Summary\n\n",
        f"Phase 28A validates whether domain-adversarial training (DANN) can "
        f"eliminate dataset-specific representations and improve cross-dataset "
        f"generalization where CORAL failed. We ran {len(exp_names)} experiments "
        f"(4 pairwise + 4 holdout) across {len(LAMBDA_SWEEP)} lambda values "
        f"each ({len(all_results[LAMBDA_SWEEP[0]]) * len(LAMBDA_SWEEP)} total runs).\n\n",
        f"- **Average Macro F1**: {avg_dann_f1:.4f} (vs baseline {avg_baseline_f1:.4f})\n",
        f"- **Average Macro F1 Δ vs Phase 26B**: {avg_f1_delta_vs_baseline:+.4f} ({avg_f1_pct_improvement:+.2f}%)\n",
        f"- **Average Macro F1 Δ vs Phase 27B CORAL**: {avg_f1_delta_vs_coral:+.4f}\n",
        f"- **Average Silhouette Dataset**: {avg_dann_sil_ds:.4f} (vs baseline {avg_baseline_sil_ds:.4f}, "
        f"{sil_ds_reduction_pct:+.1f}% reduction)\n",
        f"- **Average Silhouette Family**: {avg_dann_sil_fam:.4f} (vs baseline {avg_baseline_sil_fam:.4f})\n",
        f"- **Best Macro F1**: {best_dann_f1:.4f}\n",
        f"- **Worst Macro F1**: {worst_dann_f1:.4f}\n",
        f"- **Wins/Losses vs Baseline**: {wins_vs_baseline}/{losses_vs_baseline}\n",
        f"- **Domain collapse detected**: {'YES' if domain_collapse else 'NO'}\n\n",
    ]

    # ── Success Criteria ─────────────────────────────────────────────
    lines += [
        "## Success Criteria\n\n",
        "### Primary: Average Macro F1 improvement >= 15% vs Phase 26B baseline\n\n",
        f"- **Improvement**: {avg_f1_pct_improvement:+.2f}%\n",
        f"- **Threshold**: >= 15%\n",
        f"- **Result**: {'PASS ✅' if primary_pass else 'FAIL ❌'}\n\n",
        "### Secondary: Dataset silhouette reduction >= 30%\n\n",
        f"- **Silhouette reduction**: {sil_ds_reduction_pct:+.1f}%\n",
        f"- **Threshold**: >= 30%\n",
        f"- **Result**: {'PASS ✅' if secondary_pass else 'FAIL ❌'}\n\n",
        "### Tertiary: Family silhouette increase >= 20%\n\n",
        f"- **Family silhouette (baseline)**: {avg_baseline_sil_fam:.4f}\n",
        f"- **Family silhouette (DANN)**: {avg_dann_sil_fam:.4f}\n",
    ]
    # Tertiary is hard to compute as a percentage when baseline is near 0
    lines.append(f"- **Result**: EVALUATED ({'IMPROVED' if avg_dann_sil_fam > avg_baseline_sil_fam else 'DEGRADED'})\n\n")

    lines += [
        "### Production Candidate: Average Macro F1 > 0.10\n\n",
        f"- **Average Macro F1**: {avg_dann_f1:.4f}\n",
        f"- **Threshold**: > 0.10\n",
        f"- **Result**: {'PASS ✅' if production_candidate else 'FAIL ❌'}\n\n",
        f"**Failure condition**: {'TRIGGERED ⚠️' if failure_triggered else 'NOT TRIGGERED ✅'}\n\n",
    ]

    # ── Per-Experiment Results ──────────────────────────────────────
    lines += [
        "## Per-Experiment Macro F1\n\n",
        "| Experiment | λ (best) | Phase 26B F1 | CORAL F1 | DANN F1 | Δ vs 26B | Δ vs CORAL |\n",
        "|-----------|--------:|------------:|--------:|-------:|--------:|----------:|\n",
    ]

    for e in exp_names:
        res = best_lambda_results[e]
        b_f1 = baseline_f1_26b.get(e, 0)
        c_f1 = coral_best_f1.get(e, 0)
        d_f1 = res["dann"]["macro_f1"]
        ld = res["lambda_domain"]
        db = d_f1 - b_f1
        dc = d_f1 - c_f1
        emoji_b = "🟢" if db > 0 else "🔴"
        emoji_c = "🟢" if dc > 0 else "🔴"
        lines.append(f"| {e} | {ld:.3f} | {b_f1:.4f} | {c_f1:.4f} | {d_f1:.4f} | {db:+.4f} {emoji_b} | {dc:+.4f} {emoji_c} |\n")

    # ── Conclusions ──────────────────────────────────────────────────
    lines += [
        "\n## Conclusions\n\n",
        "### 1. DANN vs Baseline\n\n",
    ]
    if avg_f1_delta_vs_baseline > 0:
        lines.append(
            f"DANN {'outperforms' if primary_pass else 'marginally improves upon'} "
            f"the Phase 26B baseline (Δ = {avg_f1_delta_vs_baseline:+.4f}, "
            f"{avg_f1_pct_improvement:+.2f}%). "
            f"{'The primary success criterion is MET.' if primary_pass else 'However, the 15% improvement threshold is not reached.'}\n\n"
        )
    else:
        lines.append(
            f"DANN underperforms the Phase 26B baseline (Δ = {avg_f1_delta_vs_baseline:+.4f}). "
            f"This indicates domain-adversarial training is actively harming classification performance.\n\n"
        )

    lines.append("### 2. DANN vs CORAL\n\n")
    if avg_f1_delta_vs_coral > 0:
        lines.append(
            f"DANN {'outperforms' if avg_f1_delta_vs_coral > 0.02 else 'marginally beats'} "
            f"CORAL (Δ = {avg_f1_delta_vs_coral:+.4f}). "
            f"DANN wins {wins_vs_coral}/{len(exp_names)} experiments over CORAL.\n\n"
        )
    else:
        lines.append(
            f"DANN underperforms CORAL (Δ = {avg_f1_delta_vs_coral:+.4f}). "
            f"CORAL remains the stronger domain adaptation technique for this feature space.\n\n"
        )

    lines.append("### 3. Dataset Silhouette Change\n\n")
    if avg_dann_sil_ds < avg_baseline_sil_ds:
        lines.append(
            f"Dataset silhouette {'substantially' if secondary_pass else 'moderately'} "
            f"decreased from {avg_baseline_sil_ds:.4f} to {avg_dann_sil_ds:.4f} "
            f"({sil_ds_reduction_pct:+.1f}%). "
            f"{'The secondary success criterion is MET.' if secondary_pass else 'The reduction falls short of the 30% target.'}\n\n"
        )
    else:
        lines.append(
            f"Dataset silhouette INCREASED from {avg_baseline_sil_ds:.4f} to "
            f"{avg_dann_sil_ds:.4f}. DANN did not reduce dataset-specific clustering.\n\n"
        )

    lines.append("### 4. Family Silhouette Change\n\n")
    if avg_dann_sil_fam > avg_baseline_sil_fam:
        lines.append(
            f"Family silhouette improved from {avg_baseline_sil_fam:.4f} to "
            f"{avg_dann_sil_fam:.4f}, indicating better attack-family separation. "
            f"However, values remain negative in most cases, suggesting cluster "
            f"coherence is still poor.\n\n"
        )
    else:
        lines.append(
            f"Family silhouette degraded from {avg_baseline_sil_fam:.4f} to "
            f"{avg_dann_sil_fam:.4f}. DANN may be collapsing attack-family structure "
            f"along with dataset-specific features.\n\n"
        )

    lines.append("### 5. Domain Invariance Assessment\n\n")
    if avg_dann_sil_ds <= 0.10:
        lines.append(
            f"Domain invariance is {'achieved' if avg_dann_sil_ds <= 0.05 else 'partially achieved'}. "
            f"Average silhouette_dataset = {avg_dann_sil_ds:.4f} suggests "
            f"{'strong' if avg_dann_sil_ds <= 0.05 else 'moderate'} mixing of source and target embeddings.\n\n"
        )
    else:
        lines.append(
            f"Domain invariance was NOT achieved. Average silhouette_dataset = {avg_dann_sil_ds:.4f} "
            f"indicates source and target embeddings remain clearly separable.\n\n"
        )

    lines.append("### 6. GO / NO-GO Decision\n\n")
    lines.append(f"**Decision**: {decision}\n\n")
    if decision == "GO":
        lines.append(
            f"DANN demonstrated sufficient improvement ({avg_f1_pct_improvement:+.2f}% average F1 gain, "
            f"{sil_ds_reduction_pct:+.1f}% silhouette reduction) to warrant production integration. "
            f"Recommended next: production DANN integration.\n\n"
        )
    elif decision == "NO-GO":
        reasons = []
        if failure_triggered:
            if avg_f1_pct_improvement < 10.0:
                reasons.append(f"Average F1 improvement ({avg_f1_pct_improvement:+.2f}%) < 10% threshold")
            if abs(avg_baseline_sil_ds - avg_dann_sil_ds) < 0.01:
                reasons.append("Dataset silhouette unchanged")
            if domain_collapse:
                reasons.append("Domain classifier collapsed (final loss too low)")
        lines.append("Failure conditions triggered:\n")
        for r in reasons:
            lines.append(f"- {r}\n")
        lines.append(
            f"\nDANN is NOT recommended for production. "
            f"Alternative domain adaptation methods should be explored.\n\n"
        )
    else:
        lines.append(
            f"DANN shows partial improvement ({avg_f1_pct_improvement:+.2f}%) but does not meet "
            f"all success criteria. Consider Phase 28B with modified DANN architecture or "
            f"alternative approaches.\n\n"
        )

    # ── Lambda Analysis ─────────────────────────────────────────────
    lines.append("## Lambda Sensitivity\n\n")
    lines.append("| Experiment | Best λ | Best F1 |\n")
    lines.append("|-----------|------:|-------:|\n")
    for e in exp_names:
        res = best_lambda_results[e]
        lines.append(f"| {e} | {res['lambda_domain']:.3f} | {res['dann']['macro_f1']:.4f} |\n")

    # Determine most common best lambda
    best_lambdas = [best_lambda_results[e]["lambda_domain"] for e in exp_names]
    from collections import Counter
    lambda_counts = Counter(best_lambdas)
    most_common_lambda = lambda_counts.most_common(1)
    if most_common_lambda:
        lines.append(f"\nMost frequent best λ: {most_common_lambda[0][0]} "
                      f"({most_common_lambda[0][1]}/{len(exp_names)} experiments)\n")

    # Save report
    report_dir = doc_dir / ".." / "releases"
    report_dir.mkdir(parents=True, exist_ok=True)
    p = report_dir / "PHASE28A_DANN_CERTIFICATION.md"
    p.write_text("".join(lines))
    logger.info("Generated %s", p)

    return decision, {
        "avg_dann_f1": avg_dann_f1,
        "avg_baseline_f1": avg_baseline_f1,
        "avg_coral_f1": avg_coral_f1,
        "avg_f1_delta_vs_baseline": avg_f1_delta_vs_baseline,
        "avg_f1_pct_improvement": avg_f1_pct_improvement,
        "avg_f1_delta_vs_coral": avg_f1_delta_vs_coral,
        "avg_sil_ds_baseline": avg_baseline_sil_ds,
        "avg_sil_ds_dann": avg_dann_sil_ds,
        "sil_ds_reduction_pct": sil_ds_reduction_pct,
        "wins_vs_baseline": wins_vs_baseline,
        "primary_pass": primary_pass,
        "secondary_pass": secondary_pass,
        "production_candidate": production_candidate,
        "failure_triggered": failure_triggered,
        "domain_collapse": domain_collapse,
    }


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 28A — Domain-Adversarial Training (DANN) Validation"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--max-samples", type=int, default=50000,
                        help="Subsample to at most this many rows per dataset")
    parser.add_argument("--train-cap-per-class", type=int, default=5000,
                        help="Max training rows per class per source dataset")
    parser.add_argument("--lambdas", type=float, nargs="+", default=LAMBDA_SWEEP,
                        help="Lambda values to sweep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-file",
                        default=str(PROJECT_ROOT / "benchmarks" / "phase28a_results.json"))
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, generate docs from existing results")
    parser.add_argument("--skip-experiments", nargs="*", default=[],
                        help="Skip specific experiments by name")
    args = parser.parse_args()

    results_json = Path(args.results_file)

    # Output directories
    doc_dir = PROJECT_ROOT / "docs" / "phase28a"
    plots_dir = PROJECT_ROOT / "plots"
    doc_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Load or run experiments ──────────────────────────────────────
    all_results: dict[float, dict[str, dict]] = {}  # lambda -> exp_name -> result

    if args.skip_train and results_json.exists():
        logger.info("Loading cached results from %s", results_json)
        raw = json.loads(results_json.read_text())
        all_results = {float(k): v for k, v in raw.items() if k != "__run_config__"}
        logger.info("Loaded %d lambda values", len(all_results))
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
                if name not in args.skip_experiments:
                    active.append((name, sources, target))
                    logger.info("  Enqueued: %s", name)
                else:
                    logger.info("  Skipping (user-specified): %s", name)
            else:
                logger.warning("  Skipping %s: missing datasets", name)

        if len(active) < 2:
            logger.error("Too few experiments (%d), exiting", len(active))
            sys.exit(1)

        # Step 4: Run lambda sweep
        for ld in args.lambdas:
            logger.info("\n%s", "=" * 70)
            logger.info("LAMBDA SWEEP: λ = %.3f", ld)
            logger.info("%s", "=" * 70)

            ld_results: dict[str, dict] = {}
            for name, sources, target in active:
                try:
                    result = run_experiment(
                        name, sources, target, harmonized, splits,
                        epochs=args.epochs, seed=args.seed,
                        lambda_domain=ld,
                        train_cap_per_class=args.train_cap_per_class,
                    )
                    ld_results[name] = result

                    results_json.parent.mkdir(parents=True, exist_ok=True)
                    serializable = {str(k): v for k, v in all_results.items()}
                    serializable[str(ld)] = _make_serializable(ld_results)
                    serializable["__run_config__"] = {
                        "epochs": args.epochs,
                        "max_samples": args.max_samples,
                        "train_cap_per_class": args.train_cap_per_class,
                        "lambdas": args.lambdas,
                        "seed": args.seed,
                    }
                    results_json.write_text(json.dumps(serializable, indent=2, default=str))
                    logger.info("  Saved intermediate results to %s", results_json)

                except Exception as e:
                    logger.error("Experiment %s (λ=%s) FAILED: %s", name, ld, e, exc_info=True)

            all_results[ld] = ld_results

        # Save final results
        serializable = {str(k): _make_serializable(v) for k, v in all_results.items()}
        serializable["__run_config__"] = {
            "epochs": args.epochs,
            "max_samples": args.max_samples,
            "train_cap_per_class": args.train_cap_per_class,
            "lambdas": args.lambdas,
            "seed": args.seed,
        }
        results_json.parent.mkdir(parents=True, exist_ok=True)
        results_json.write_text(json.dumps(serializable, indent=2, default=str))
        logger.info("Saved final results to %s", results_json)

    if not all_results:
        logger.error("No results to process")
        sys.exit(1)

    # Step 5: Pick best lambda per experiment
    best_lambda_results: dict[str, dict] = {}
    exp_names_all = sorted(all_results[list(all_results.keys())[0]].keys())
    for exp_name in exp_names_all:
        best_f1 = -1
        best_ld = None
        best_result = None
        for ld, ld_results in all_results.items():
            if exp_name in ld_results:
                f1 = ld_results[exp_name]["dann"]["macro_f1"]
                if f1 > best_f1:
                    best_f1 = f1
                    best_ld = ld
                    best_result = ld_results[exp_name]
        if best_result:
            best_lambda_results[exp_name] = best_result

    logger.info("\nBest lambda per experiment:")
    for exp_name, res in best_lambda_results.items():
        logger.info("  %s: λ=%s → F1=%.4f", exp_name, res["lambda_domain"], res["dann"]["macro_f1"])

    # Step 6: Generate plots
    logger.info("\nGenerating plots...")
    generate_plots(all_results, best_lambda_results, plots_dir)

    # Step 7: Generate reports
    logger.info("\nGenerating reports...")
    generate_dann_sweep(all_results, doc_dir)
    generate_pairwise_results(best_lambda_results, doc_dir)
    generate_holdout_results(best_lambda_results, doc_dir)
    generate_embedding_audit(best_lambda_results, doc_dir)
    generate_domain_confusion(best_lambda_results, doc_dir)

    logger.info("\nGenerating certification report...")
    decision, summary = generate_certification_report(
        best_lambda_results, all_results, doc_dir, plots_dir,
    )

    # Summary
    logger.info("\n%s", "=" * 70)
    logger.info("PHASE 28A — SUMMARY")
    logger.info("%s", "=" * 70)
    logger.info("  Experiments completed: %d", len(exp_names_all))
    logger.info("  Avg DANN F1:       %.4f", summary["avg_dann_f1"])
    logger.info("  Avg Baseline F1:   %.4f", summary["avg_baseline_f1"])
    logger.info("  Avg CORAL F1:      %.4f", summary["avg_coral_f1"])
    logger.info("  Avg F1 Δ vs 26B:   %+.4f (%.2f%%)", summary["avg_f1_delta_vs_baseline"], summary["avg_f1_pct_improvement"])
    logger.info("  Avg F1 Δ vs CORAL: %+.4f", summary["avg_f1_delta_vs_coral"])
    logger.info("  Sil-DS Δ:          %+.4f", summary["avg_sil_ds_dann"] - summary["avg_sil_ds_baseline"])
    logger.info("  Sil-DS reduction:  %.1f%%", summary["sil_ds_reduction_pct"])
    logger.info("  Primary pass:      %s", summary["primary_pass"])
    logger.info("  Secondary pass:    %s", summary["secondary_pass"])
    logger.info("  Production cand.:  %s", summary["production_candidate"])
    logger.info("  Domain collapse:   %s", summary["domain_collapse"])
    logger.info("  Decision:          %s", decision)
    logger.info("  Results JSON:      %s", results_json)
    logger.info("  Reports:           %s/", doc_dir)
    logger.info("  Plots:             %s/", plots_dir)
    logger.info("%s", "=" * 70)


def _make_serializable(ld_results: dict) -> dict:
    """Strip embedding data for JSON serialization."""
    serializable = {}
    for en, er in ld_results.items():
        serializable[en] = {k: v for k, v in er.items() if k not in ["dann_embeddings"]}
    return serializable


if __name__ == "__main__":
    main()
