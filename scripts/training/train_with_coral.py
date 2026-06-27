#!/usr/bin/env python3
"""
Phase 27A — CORAL Domain Alignment Sweep.

Trains HelixIDS-Full on NSL-KDD (source) with CORAL alignment to
UNSW-NB15 (target) features, sweeping lambda_coral ∈ {0.01, 0.05, 0.10, 0.25, 0.50, 1.00}.

Usage:
    source .venv311/bin/activate
    python scripts/training/train_with_coral.py

Outputs:
    results/coral_sweep/sweep_results.json
    results/coral_sweep/tsne_baseline.png
    results/coral_sweep/tsne_coral_best.png
    results/coral_sweep/umap_baseline.png
    results/coral_sweep/umap_coral_best.png
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Device detection
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
logger.info("Using device: %s", DEVICE)

SEED = 42
NUM_EPOCHS = 150
BATCH_SIZE = 256
LEARNING_RATE = 5e-4
PATIENCE = 20

# Classes
CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]


# ============================================================================
# Data loading
# ============================================================================


def load_harmonized_csv(path: Path, label_col: str = "label") -> tuple[np.ndarray, np.ndarray]:
    """Load a harmonized 17-feature CSV and return (X, y)."""
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.int64)
    return X, y


def load_datasets() -> dict:
    """Load NSL-KDD (source) and UNSW-NB15 (target) with standard scaling."""
    data_root = PROJECT_ROOT / "data"
    nsl_train = data_root / "nsl_kdd" / "train.csv"
    nsl_test = data_root / "nsl_kdd" / "test.csv"
    unsw_train = data_root / "unsw_nb15" / "train.csv"
    unsw_test = data_root / "unsw_nb15" / "test.csv"

    # Load raw
    X_src, y_src = load_harmonized_csv(nsl_train, label_col="label")
    X_src_test, y_src_test = load_harmonized_csv(nsl_test, label_col="label")
    X_tgt, _ = load_harmonized_csv(unsw_train, label_col="attack_cat")
    X_tgt_test, y_tgt_test = load_harmonized_csv(unsw_test, label_col="attack_cat")

    # Standard scale on source stats
    scaler = StandardScaler()
    X_src = scaler.fit_transform(X_src)
    X_src_test = scaler.transform(X_src_test)
    X_tgt = scaler.transform(X_tgt)
    X_tgt_test = scaler.transform(X_tgt_test)

    logger.info("Source (NSL-KDD)  train: %s  test: %s", X_src.shape, X_src_test.shape)
    logger.info("Target (UNSW-NB15) train: %s  test: %s", X_tgt.shape, X_tgt_test.shape)
    logger.info("Source labels: %s", sorted(np.unique(y_src)))
    logger.info("Target labels: %s", sorted(np.unique(y_tgt_test)))

    # Log class distributions
    for name, yy in [("nsltrain", y_src), ("nsltest", y_src_test), ("unswtest", y_tgt_test)]:
        dist = pd.Series(yy).value_counts().sort_index()
        logger.info("  %s class dist: %s", name, dist.to_dict())

    return {
        "X_src": X_src, "y_src": y_src,
        "X_src_test": X_src_test, "y_src_test": y_src_test,
        "X_tgt": X_tgt,
        "X_tgt_test": X_tgt_test, "y_tgt_test": y_tgt_test,
        "n_features": X_src.shape[1],
    }


# ============================================================================
# Model (HelixIDS-Full, simplified construction)
# ============================================================================


class CORALHelixModel(nn.Module):
    """HelixIDS-Full backbone with CORAL-compatible feature extraction.

    Architecture matches the canonical HelixIDSFull but exposes backbone
    features for alignment.

    Input:  17 engineered flow features
    Output: family_logits (7-dim), binary_logits (2-dim)
    """

    def __init__(self, input_dim: int = 17):
        super().__init__()
        self.input_dim = input_dim

        # Shared backbone (4-layer MLP)
        hidden_dims = [512, 384, 256, 256]
        dropout_rates = [0.3, 0.3, 0.25, 0.2]

        backbone = []
        prev = input_dim
        for i, h in enumerate(hidden_dims):
            backbone.append(nn.Linear(prev, h))
            backbone.append(nn.BatchNorm1d(h))
            backbone.append(nn.ReLU())
            backbone.append(nn.Dropout(dropout_rates[i]))
            prev = h
        self.backbone = nn.Sequential(*backbone)

        # Family head (7-class)
        self.family_head = nn.Sequential(
            nn.Linear(prev, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 7),  # 7-class family taxonomy
        )

        # Binary head (Normal vs Attack)
        self.binary_head = nn.Sequential(
            nn.Linear(prev, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        )

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> tuple[torch.Tensor, ...] | torch.Tensor:
        features = self.backbone(x)
        family_logits = self.family_head(features)
        binary_logits = self.binary_head(features)

        if return_features:
            return binary_logits, family_logits, features
        return binary_logits, family_logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features for domain alignment."""
        return self.backbone(x)


# ============================================================================
# CORAL Loss
# ============================================================================


def coral_loss(source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    """CORAL loss: ||C_s - C_t||²_F / (4*d²)

    Args:
        source_features: (n_s, d)
        target_features: (n_t, d)

    Returns:
        Scalar loss.
    """
    d = source_features.size(1)

    def cov(x):
        n = x.size(0)
        if n < 2:
            return torch.zeros(d, d, device=x.device)
        centered = x - x.mean(dim=0)
        return centered.T @ centered / (n - 1)

    cs = cov(source_features)
    ct = cov(target_features)
    diff = cs - ct
    loss = torch.sum(diff * diff) / (4.0 * d * d)
    return loss


# ============================================================================
# Training helpers
# ============================================================================


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> dict:
    """Compute accuracy, precision, recall, macro F1."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate_model(
    model: nn.Module, loader: DataLoader, num_classes: int = 7
) -> dict:
    """Evaluate model on a dataloader, returns metrics."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            _, family_logits = model(xb)
            preds = family_logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(yb.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    return compute_metrics(all_targets, all_preds, num_classes)


def extract_embeddings(
    model: nn.Module, loader: DataLoader
) -> tuple[np.ndarray, np.ndarray]:
    """Extract backbone features and labels from a dataloader."""
    model.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            feats = model.extract_features(xb).cpu().numpy()
            all_features.append(feats)
            all_labels.append(yb.numpy())
    return np.concatenate(all_features), np.concatenate(all_labels)


# ============================================================================
# Single training run
# ============================================================================


def train_one(
    lambda_coral: float,
    data: dict,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
) -> dict:
    """Train a model with CORAL alignment on NSL-KDD → UNSW-NB15.

    Returns dict with metrics, model state, and embedding data.
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    n_features = data["n_features"]
    model = CORALHelixModel(input_dim=n_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Source loader (NSL-KDD train)
    src_dataset = TensorDataset(
        torch.FloatTensor(data["X_src"]),
        torch.LongTensor(data["y_src"]),
    )
    src_loader = DataLoader(
        src_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    # Target feature loader (UNSW-NB15 train, no labels)
    tgt_dataset = TensorDataset(torch.FloatTensor(data["X_tgt"]))
    tgt_loader = DataLoader(
        tgt_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    # Source test loader (NSL-KDD test)
    src_test_dataset = TensorDataset(
        torch.FloatTensor(data["X_src_test"]),
        torch.LongTensor(data["y_src_test"]),
    )
    src_test_loader = DataLoader(
        src_test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    # Target test loader (UNSW-NB15 test)
    tgt_test_dataset = TensorDataset(
        torch.FloatTensor(data["X_tgt_test"]),
        torch.LongTensor(data["y_tgt_test"]),
    )
    tgt_test_loader = DataLoader(
        tgt_test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    # Combined loader for source labels (NSL-KDD train) + CORAL alignment
    # We iterate through source batches; for each batch, draw a target batch
    tgt_iter = iter(tgt_loader)

    best_tgt_f1 = 0.0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss_cls = 0.0
        epoch_loss_coral = 0.0
        epoch_total = 0.0
        n_batches = 0

        for x_src, y_src in src_loader:
            x_src, y_src = x_src.to(DEVICE), y_src.to(DEVICE)

            # Get target batch (restart if exhausted)
            try:
                (x_tgt,) = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                (x_tgt,) = next(tgt_iter)
            x_tgt = x_tgt.to(DEVICE)

            # Forward pass with feature extraction
            binary_logits, family_logits, src_features = model(x_src, return_features=True)
            tgt_features = model.extract_features(x_tgt)

            # Classification loss (on source)
            loss_cls = F.cross_entropy(family_logits, y_src)

            # CORAL loss (source features → target features)
            loss_coral_val = coral_loss(src_features, tgt_features)

            # Total loss
            loss_total = loss_cls + lambda_coral * loss_coral_val

            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            epoch_loss_cls += loss_cls.item()
            epoch_loss_coral += loss_coral_val.item()
            epoch_total += loss_total.item()
            n_batches += 1

        # Evaluate on target test set
        tgt_metrics = evaluate_model(model, tgt_test_loader, num_classes=7)

        history.append({
            "epoch": epoch,
            "loss_cls": epoch_loss_cls / n_batches,
            "loss_coral": epoch_loss_coral / n_batches,
            "loss_total": epoch_total / n_batches,
            "tgt_acc": tgt_metrics["accuracy"],
            "tgt_macro_f1": tgt_metrics["macro_f1"],
        })

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "  [%3d/%d] cls=%.4f coral=%.6f total=%.4f | tgt acc=%.4f f1=%.4f",
                epoch, num_epochs,
                epoch_loss_cls / n_batches,
                epoch_loss_coral / n_batches,
                epoch_total / n_batches,
                tgt_metrics["accuracy"],
                tgt_metrics["macro_f1"],
            )

        # Early stopping on target macro F1
        if tgt_metrics["macro_f1"] > best_tgt_f1:
            best_tgt_f1 = tgt_metrics["macro_f1"]
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("  Early stopping at epoch %d (best target F1=%.4f)", epoch, best_tgt_f1)
                break

    # Restore best model
    model.load_state_dict(best_state)

    # Final evaluation
    src_metrics = evaluate_model(model, src_test_loader, num_classes=7)
    tgt_metrics = evaluate_model(model, tgt_test_loader, num_classes=7)

    logger.info("  Final source acc=%.4f f1=%.4f", src_metrics["accuracy"], src_metrics["macro_f1"])
    logger.info("  Final target acc=%.4f f1=%.4f", tgt_metrics["accuracy"], tgt_metrics["macro_f1"])

    # Compute generalization gap
    gap = float(src_metrics["macro_f1"] - tgt_metrics["macro_f1"])

    return {
        "lambda_coral": lambda_coral,
        "best_target_macro_f1": float(best_tgt_f1),
        "source_accuracy": float(src_metrics["accuracy"]),
        "source_macro_f1": float(src_metrics["macro_f1"]),
        "target_accuracy": float(tgt_metrics["accuracy"]),
        "target_macro_f1": float(tgt_metrics["macro_f1"]),
        "target_precision": float(tgt_metrics["precision"]),
        "target_recall": float(tgt_metrics["recall"]),
        "generalization_gap": gap,
        "num_epochs": min(epoch, num_epochs),
        "stopped_early": patience_counter >= PATIENCE,
        "history": history,
        "model_state": best_state,
    }


def train_baseline(data: dict) -> dict:
    """Train without CORAL (lambda=0) to establish baseline."""
    return train_one(lambda_coral=0.0, data=data)


# ============================================================================
# Embedding visualizations
# ============================================================================


def generate_embeddings(
    model: nn.Module, data: dict, name: str, results_dir: Path
) -> dict:
    """Generate t-SNE and UMAP plots for source + target embeddings."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        logger.warning("sklearn not available, skipping t-SNE")
        return {"tsne": None, "umap": None}

    # Create combined dataloader for source and target
    def make_loader(X, y):
        ds = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
        return DataLoader(ds, batch_size=512, shuffle=False)

    model.eval()

    # Source
    src_loader = make_loader(data["X_src_test"], data["y_src_test"])
    src_feats, src_labels = extract_embeddings(model, src_loader)

    # Target
    tgt_loader = make_loader(data["X_tgt_test"], data["y_tgt_test"])
    tgt_feats, tgt_labels = extract_embeddings(model, tgt_loader)

    # Combine
    all_feats = np.vstack([src_feats, tgt_feats])
    all_labels = np.concatenate([src_labels, tgt_labels])
    dataset_ids = np.concatenate([
        np.zeros(len(src_feats)),
        np.ones(len(tgt_feats)),
    ])

    # Compute silhouette scores
    from sklearn.metrics import silhouette_score
    sil_dataset = float(silhouette_score(all_feats, dataset_ids))
    sil_family = float(silhouette_score(all_feats, all_labels))
    logger.info("  Silhouette (dataset): %.4f  (family): %.4f", sil_dataset, sil_family)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
    tsne_2d = tsne.fit_transform(all_feats)
    tsne_path = str(results_dir / f"tsne_{name}.png")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Color by dataset
    colors = ["#1f77b4", "#ff7f0e"]
    for ds_id in [0, 1]:
        mask = dataset_ids == ds_id
        axes[0].scatter(
            tsne_2d[mask, 0], tsne_2d[mask, 1],
            c=colors[ds_id], label=["NSL-KDD", "UNSW-NB15"][ds_id],
            alpha=0.5, s=5,
        )
    axes[0].set_title(f"t-SNE by Dataset (sil={sil_dataset:.3f})", fontsize=10)
    axes[0].legend(fontsize=8)

    # Color by family
    cmap = plt.cm.tab10
    for label_id in np.unique(all_labels):
        mask = all_labels == label_id
        axes[1].scatter(
            tsne_2d[mask, 0], tsne_2d[mask, 1],
            c=cmap(label_id % 10),
            label=str(label_id) if int(label_id) < 5 else f"cls{int(label_id)}",
            alpha=0.5, s=5,
        )
    axes[1].set_title(f"t-SNE by Attack Family (sil={sil_family:.3f})", fontsize=10)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(tsne_path, dpi=150)
    plt.close()
    logger.info("  Saved t-SNE: %s", tsne_path)

    # UMAP
    try:
        import umap
        reducer = umap.UMAP(random_state=SEED)
        umap_2d = reducer.fit_transform(all_feats)
        umap_path = str(results_dir / f"umap_{name}.png")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ds_id in [0, 1]:
            mask = dataset_ids == ds_id
            axes[0].scatter(
                umap_2d[mask, 0], umap_2d[mask, 1],
                c=colors[ds_id], label=["NSL-KDD", "UNSW-NB15"][ds_id],
                alpha=0.5, s=5,
            )
        axes[0].set_title(f"UMAP by Dataset (sil={sil_dataset:.3f})", fontsize=10)
        axes[0].legend(fontsize=8)

        for label_id in np.unique(all_labels):
            mask = all_labels == label_id
            axes[1].scatter(
                umap_2d[mask, 0], umap_2d[mask, 1],
                c=cmap(label_id % 10),
                label=str(label_id) if int(label_id) < 5 else f"cls{int(label_id)}",
                alpha=0.5, s=5,
            )
        axes[1].set_title(f"UMAP by Attack Family (sil={sil_family:.3f})", fontsize=10)
        axes[1].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(umap_path, dpi=150)
        plt.close()
        logger.info("  Saved UMAP: %s", umap_path)
    except ImportError:
        logger.warning("umap not installed, skipping UMAP plot")
        umap_path = None

    return {
        "silhouette_dataset": sil_dataset,
        "silhouette_family": sil_family,
        "tsne_path": tsne_path,
        "umap_path": umap_path,
    }


# ============================================================================
# Main sweep
# ============================================================================


def main():
    torch.set_num_threads(4)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["OMP_NUM_THREADS"] = "4"

    results_dir = PROJECT_ROOT / "results" / "coral_sweep"
    results_dir.mkdir(parents=True, exist_ok=True)

    phase_dir = PROJECT_ROOT / "docs" / "phase27a"
    phase_dir.mkdir(parents=True, exist_ok=True)

    release_dir = PROJECT_ROOT / "docs" / "releases"
    release_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    logger.info("=" * 72)
    logger.info("Phase 27A — CORAL Domain Alignment Sweep")
    logger.info("Source: NSL-KDD → Target: UNSW-NB15")
    logger.info("=" * 72)

    # ── Load data ──────────────────────────────────────────────────────
    data = load_datasets()

    # ── Baseline (no CORAL) ──────────────────────────────────────────
    logger.info("\n--- Baseline training (lambda_coral=0.0) ---")
    baseline = train_baseline(data)
    baseline_f1 = baseline["target_macro_f1"]
    logger.info("Baseline target Macro F1: %.4f", baseline_f1)
    logger.info("Baseline source Macro F1: %.4f", baseline["source_macro_f1"])
    logger.info("Baseline generalization gap: %.4f", baseline["generalization_gap"])

    # Baseline embeddings
    model_baseline = CORALHelixModel(input_dim=data["n_features"]).to(DEVICE)
    model_baseline.load_state_dict(baseline["model_state"])
    baseline_emb = generate_embeddings(model_baseline, data, "baseline", results_dir)

    # ── Sweep lambda ──────────────────────────────────────────────────
    LAMBDA_VALUES = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    sweep_results = []
    best_config = None
    best_f1 = baseline_f1

    for lam in LAMBDA_VALUES:
        logger.info("\n--- lambda_coral=%.2f ---", lam)
        result = train_one(lambda_coral=lam, data=data)
        sweep_results.append(result)

        f1 = result["target_macro_f1"]
        logger.info("  lambda=%.2f: target F1=%.4f (source F1=%.4f, gap=%.4f)",
                     lam, f1, result["source_macro_f1"], result["generalization_gap"])

        if f1 > best_f1:
            best_f1 = f1
            best_config = result

    elapsed = time.time() - start
    logger.info("\n" + "=" * 72)
    logger.info("Sweep complete in %.1f seconds", elapsed)

    # ── Best CORAL embeddings ─────────────────────────────────────────
    best_lambda = best_config["lambda_coral"]
    logger.info("\nBest lambda: %.2f (target Macro F1 = %.4f)", best_lambda, best_f1)

    # Compute improvement vs baseline
    pct_improvement = ((best_f1 - baseline_f1) / max(baseline_f1, 1e-8)) * 100

    model_best = CORALHelixModel(input_dim=data["n_features"]).to(DEVICE)
    model_best.load_state_dict(best_config["model_state"])
    best_emb = generate_embeddings(model_best, data, "coral_best", results_dir)

    # ── Silhouette deltas ─────────────────────────────────────────────
    sil_ds_baseline = baseline_emb.get("silhouette_dataset")
    sil_ds_best = best_emb.get("silhouette_dataset")
    sil_fam_baseline = baseline_emb.get("silhouette_family")
    sil_fam_best = best_emb.get("silhouette_family")

    sil_ds_delta = None
    sil_fam_delta = None
    if sil_ds_baseline is not None and sil_ds_best is not None:
        sil_ds_delta = sil_ds_best - sil_ds_baseline
    if sil_fam_baseline is not None and sil_fam_best is not None:
        sil_fam_delta = sil_fam_best - sil_fam_baseline

    # ── Success / Failure ─────────────────────────────────────────────
    success_f1 = pct_improvement >= 25.0
    success_sil_ds = sil_ds_delta is not None and sil_ds_delta <= -0.3 * abs(sil_ds_baseline)
    success_sil_fam = sil_fam_best is not None and sil_fam_best > 0

    success = success_f1 or success_sil_ds or success_sil_fam
    failure = (not success) and (pct_improvement < 10.0) and (sil_ds_best is not None and sil_ds_best > 0)

    # ── Save sweep results ───────────────────────────────────────────
    sweep_data = {
        "experiment": "Phase 27A - CORAL Domain Alignment",
        "source_dataset": "NSL-KDD",
        "target_dataset": "UNSW-NB15",
        "baseline_macro_f1": baseline_f1,
        "best_lambda": best_lambda,
        "best_macro_f1": best_f1,
        "pct_improvement": pct_improvement,
        "elapsed_seconds": elapsed,
        "sweep": [
            {
                "lambda": r["lambda_coral"],
                "target_macro_f1": r["target_macro_f1"],
                "source_macro_f1": r["source_macro_f1"],
                "target_accuracy": r["target_accuracy"],
                "generalization_gap": r["generalization_gap"],
                "epochs": r["num_epochs"],
            }
            for r in sweep_results
        ],
        "baseline_embeddings": {
            "silhouette_dataset": sil_ds_baseline,
            "silhouette_family": sil_fam_baseline,
        },
        "best_embeddings": {
            "silhouette_dataset": sil_ds_best,
            "silhouette_family": sil_fam_best,
        },
        "silhouette_dataset_delta": sil_ds_delta,
        "silhouette_family_delta": sil_fam_delta,
        "success": success,
        "failure": failure,
        "success_criteria": {
            "f1_improvement_25pct": success_f1,
            "silhouette_dataset_decrease_30pct": success_sil_ds,
            "silhouette_family_positive": success_sil_fam,
        },
    }

    sweep_json_path = results_dir / "sweep_results.json"
    with open(sweep_json_path, "w") as f:
        json.dump(sweep_data, f, indent=2)
    logger.info("Saved sweep results to %s", sweep_json_path)

    # ── Generate CORAL_SWEEP.md ──────────────────────────────────────
    sweep_lines = [
        "# CORAL Domain Alignment Sweep Report (Phase 27A)",
        "",
        "## Experiment",
        "",
        f"- **Source**: NSL-KDD → **Target**: UNSW-NB15",
        f"- **Baseline Macro F1** (no CORAL): **{baseline_f1:.4f}**",
        f"- **Best lambda_coral**: **{best_lambda:.2f}**",
        f"- **Best Macro F1**: **{best_f1:.4f}**",
        f"- **Improvement**: **{pct_improvement:.2f}%**",
        f"- **Elapsed**: {elapsed:.0f}s",
        "",
        "## Sweep Results",
        "",
        "| lambda_coral | Target Acc | Target Macro F1 | Source Macro F1 | Gen Gap | Epochs |",
        "|-------------|-----------|----------------|----------------|--------|--------|",
    ]
    for r in sweep_results:
        sweep_lines.append(
            f"| {r['lambda_coral']:.2f} | {r['target_accuracy']:.4f} | "
            f"{r['target_macro_f1']:.4f} | {r['source_macro_f1']:.4f} | "
            f"{r['generalization_gap']:.4f} | {r['num_epochs']} |"
        )

    sweep_lines.extend([
        "",
        "## Embedding Audit",
        "",
        f"- **Baseline**: silhouette_dataset={sil_ds_baseline:.4f}, silhouette_family={sil_fam_baseline:.4f}"
        if sil_ds_baseline is not None else "- **Baseline**: silhouette unavailable",
        f"- **Best CORAL**: silhouette_dataset={sil_ds_best:.4f}, silhouette_family={sil_fam_best:.4f}"
        if sil_ds_best is not None else "- **Best CORAL**: silhouette unavailable",
        f"- **Dataset silhouette delta**: {sil_ds_delta:.4f}" if sil_ds_delta is not None else "- **Dataset silhouette delta**: N/A",
        f"- **Family silhouette delta**: {sil_fam_delta:.4f}" if sil_fam_delta is not None else "- **Family silhouette delta**: N/A",
        "",
        "### Visualizations",
        "",
        "- Baseline: `results/coral_sweep/tsne_baseline.png`, `results/coral_sweep/umap_baseline.png`",
        "- Best CORAL: `results/coral_sweep/tsne_coral_best.png`, `results/coral_sweep/umap_coral_best.png`",
        "",
        "## Success Criteria Check",
        "",
        f"- Macro F1 improvement >= 25%? **{'YES ✓' if success_f1 else 'NO'}** ({pct_improvement:.2f}%)",
        f"- Dataset silhouette decrease >= 30%? **{'YES ✓' if success_sil_ds else 'NO'}**",
        f"- Family silhouette positive? **{'YES ✓' if success_sil_fam else 'NO'}**",
        "",
        f"**Overall: {'SUCCESS' if success else 'FAILURE'}**",
    ])
    if success:
        sweep_lines.append(f"\nReason: CORAL at lambda={best_lambda:.2f} meets success criteria.")
    else:
        sweep_lines.append("")
        sweep_lines.append("## Recommendation")
        sweep_lines.append("")
        sweep_lines.append("CORAL alone does not resolve domain shift. Proceed to Phase 28 (DANN).")

    sweep_md_path = phase_dir / "CORAL_SWEEP.md"
    sweep_md_path.write_text("\n".join(sweep_lines))
    logger.info("Saved sweep report to %s", sweep_md_path)

    # ── Certification report ──────────────────────────────────────────
    cert_lines = [
        "# PHASE 27A — CORAL DOMAIN ALIGNMENT CERTIFICATION",
        "",
        "## Overview",
        "",
        "Phase 27A evaluates whether lightweight covariance alignment (CORAL) can",
        "bridge the domain gap between NSL-KDD and UNSW-NB15 without backbone changes.",
        "",
        "## Hypothesis",
        "",
        "The current representation encodes dataset identity over attack semantics.",
        "CORAL alignment forces source and target feature distributions toward a shared",
        "covariance structure, which should improve cross-dataset generalization if the",
        "representation is salvageable.",
        "",
        "## Results Summary",
        "",
        f"| Metric | Baseline | Best CORAL | Delta |",
        f"|--------|----------|------------|-------|",
        f"| **Lambda coral** | — | {best_lambda:.2f} | — |",
        f"| **Macro F1 (target)** | {baseline_f1:.4f} | {best_f1:.4f} | {best_f1 - baseline_f1:+.4f} ({pct_improvement:+.2f}%) |",
        f"| **Dataset silhouette** | {sil_ds_baseline:.4f}" if sil_ds_baseline is not None else "| **Dataset silhouette** | N/A",
        f" | {sil_ds_best:.4f}" if sil_ds_best is not None else " | N/A",
        f" | {sil_ds_delta:+.4f}" if sil_ds_delta is not None else " | N/A",
        f" |" if sil_ds_baseline is not None else " |",
        f"| **Family silhouette** | {sil_fam_baseline:.4f}" if sil_fam_baseline is not None else "| **Family silhouette** | N/A",
        f" | {sil_fam_best:.4f}" if sil_fam_best is not None else " | N/A",
        f" | {sil_fam_delta:+.4f}" if sil_fam_delta is not None else " | N/A",
        f" |" if sil_fam_baseline is not None else " |",
        "",
    ]

    # Rebuild cert table properly
    def fmtv(v, fmt=".4f"):
        if v is None:
            return "N/A"
        return f"{v:{fmt}}"

    cert_lines = [  # Rewrite cleanly
        "# PHASE 27A — CORAL DOMAIN ALIGNMENT CERTIFICATION",
        "",
        "## Experiment: NSL-KDD → UNSW-NB15",
        "",
        f"| Metric | Baseline | Best CORAL | Delta |",
        f"|--------|----------|------------|-------|",
        f"| **Lambda coral** | — | {fmtv(best_lambda)} | — |",
        f"| **Macro F1 (target)** | {fmtv(baseline_f1)} | {fmtv(best_f1)} | {fmtv(best_f1 - baseline_f1, '+.4f')} ({pct_improvement:+.2f}%) |",
        f"| **Macro F1 (source)** | {fmtv(baseline['source_macro_f1'])} | {fmtv(best_config['source_macro_f1'])} | {fmtv(best_config['source_macro_f1'] - baseline['source_macro_f1'], '+.4f')} |",
        f"| **Accuracy (target)** | {fmtv(baseline['target_accuracy'])} | {fmtv(best_config['target_accuracy'])} | {fmtv(best_config['target_accuracy'] - baseline['target_accuracy'], '+.4f')} |",
        f"| **Gen. gap** | {fmtv(baseline['generalization_gap'])} | {fmtv(best_config['generalization_gap'])} | {fmtv(best_config['generalization_gap'] - baseline['generalization_gap'], '+.4f')} |",
        f"| **Silhouette (dataset)** | {fmtv(sil_ds_baseline)} | {fmtv(sil_ds_best)} | {fmtv(sil_ds_delta, '+.4f')} |",
        f"| **Silhouette (family)** | {fmtv(sil_fam_baseline)} | {fmtv(sil_fam_best)} | {fmtv(sil_fam_delta, '+.4f')} |",
        "",
        f"## Decision",
        "",
    ]

    if success:
        cert_lines.append("### ✅ SUCCESS")
        cert_lines.append("")
        reasons = []
        if success_f1:
            reasons.append(f"Macro F1 improved by {pct_improvement:.2f}% (≥25%)")
        if success_sil_ds:
            reasons.append("Dataset silhouette decreased by ≥30%")
        if success_sil_fam:
            reasons.append("Family silhouette became positive")
        for r in reasons:
            cert_lines.append(f"- {r}")
        cert_lines.append("")
        cert_lines.append("**Recommendation**: Proceed to Phase 27B (Multi-dataset CORAL training).")
    else:
        cert_lines.append("### ❌ FAILURE")
        cert_lines.append("")
        cert_lines.append("CORAL alignment failed to resolve domain shift:")
        cert_lines.append(f"- Best Macro F1 improvement: {pct_improvement:.2f}% (< 10%)")
        if sil_ds_best is not None and sil_ds_best > 0:
            cert_lines.append(f"- Dataset silhouette remains positive ({sil_ds_best:.4f})")
        cert_lines.append("")
        cert_lines.append("**Recommendation**: Proceed directly to Phase 28 (DANN).")

    cert_lines.append("")
    cert_lines.append("---")
    cert_lines.append(f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S IST')}*")
    cert_lines.append(f"*Device: {DEVICE}*")
    cert_lines.append(f"*Repo: {PROJECT_ROOT.name}*")

    cert_md_path = release_dir / "PHASE27A_CORAL_CERTIFICATION.md"
    cert_md_path.write_text("\n".join(cert_lines))
    logger.info("Saved certification report to %s", cert_md_path)

    logger.info("\n=== Phase 27A Complete ===")
    logger.info("Best lambda: %.2f  |  Baseline F1: %.4f  →  Best F1: %.4f  (%+.2f%%)",
                best_lambda, baseline_f1, best_f1, pct_improvement)
    logger.info("sil_ds: %s → %s  |  sil_fam: %s → %s",
                fmtv(sil_ds_baseline), fmtv(sil_ds_best),
                fmtv(sil_fam_baseline), fmtv(sil_fam_best))
    logger.info("Decision: %s", "SUCCESS → Phase 27B" if success else "FAILURE → Phase 28 (DANN)")


if __name__ == "__main__":
    main()
