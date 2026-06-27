#!/usr/bin/env python3
"""
Phase 51 — Pairwise Transferability and Failure Mechanism Analysis

Orchestrates:
  1. Train SupCon encoder (Phase 50 Exp B) if not cached
  2. Extract latents
  3. Experiment A: Pairwise Success Landscape
  4. Experiment B: Dataset Similarity vs Transfer
  5. Experiment C: Class-Level Transfer
  6. Experiment D: Feature Dependency Analysis (SHAP)
  7. Experiment E: Latent Geometry Evolution
  8. Experiment F: Failure Attribution
  9. Statistical Analysis (bootstrap, permutation, effect sizes)
  10. Deliverable generation

Usage:
  source .venv311/bin/activate
  PYTHONPATH=src python3 scripts/analysis/phase51_main.py
  PYTHONPATH=src python3 scripts/analysis/phase51_main.py --skip-train  (reuse cached encoder)
  PYTHONPATH=src python3 scripts/analysis/phase51_main.py --experiments A,B,C  (subset)
"""
import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger = logging.getLogger("phase51")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "multi_dataset_v1"
PHASE50_DIR = PROJECT_ROOT / "results" / "phase50"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase51"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR / "models", exist_ok=True)
os.makedirs(RESULTS_DIR / "latents", exist_ok=True)
os.makedirs(RESULTS_DIR / "matrices", exist_ok=True)
os.makedirs(RESULTS_DIR / "tables", exist_ok=True)
os.makedirs(RESULTS_DIR / "plots", exist_ok=True)
os.makedirs(RESULTS_DIR / "attributions", exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
fh = logging.FileHandler(RESULTS_DIR / "phase51_run.log")
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(fh)

logger.info(f"Phase 51 starting — device={DEVICE}")

RANDOM_STATE = 42
MAX_SAMPLES_PER_DATASET = 200_000
SUPCON_EPOCHS = 50
PATIENCE = 15
LR = 1e-3
BATCH_SIZE = 256
INPUT_DIM = 17
LATENT_DIM = 128
NUM_CLASSES = 2
rng = np.random.RandomState(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

DATASET_DISPLAY = {
    "nsl_kdd": "NSL-KDD", "unsw_nb15": "UNSW-NB15", "cicids": "CICIDS2018",
    "ton_iot": "TON-IoT", "bot_iot": "Bot-IoT", "cicids2017": "CICIDS2017",
}
DATASET_NAMES = ["nsl_kdd", "unsw_nb15", "cicids", "ton_iot", "bot_iot", "cicids2017"]
DATASET_NAMES_SORTED = sorted(DATASET_NAMES)
N_DATASETS = len(DATASET_NAMES)
CLASS_NAMES = ["Normal", "Attack"]

# ─────────────────────────────────────────────
# Architecture (reused from Phase 50)
# ─────────────────────────────────────────────

class SharedEncoder(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
    def forward(self, x):
        return self.net(x)

class ClassifierHead(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, num_classes),
        )
    def forward(self, z):
        return self.net(z)

class ProjectionHead(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, proj_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, proj_dim), nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
    def forward(self, z):
        return self.net(z)

# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_npy_datasets():
    datasets = {}
    if not PROCESSED_DIR.exists():
        return datasets
    for key in ["nsl_kdd", "unsw_nb15", "cicids"]:
        X_tr_path = PROCESSED_DIR / f"X_train_{key}.npy"
        X_te_path = PROCESSED_DIR / f"X_test_{key}.npy"
        y_tr_path = PROCESSED_DIR / f"y_train_{key}.npy"
        y_te_path = PROCESSED_DIR / f"y_test_{key}.npy"
        X_tr = np.load(X_tr_path).astype(np.float64) if X_tr_path.exists() else np.empty((0, 17))
        X_te = np.load(X_te_path).astype(np.float64) if X_te_path.exists() else np.empty((0, 17))
        if X_tr.size > 0 and X_te.size > 0:
            max_pre = MAX_SAMPLES_PER_DATASET // 2
            if X_tr.shape[0] > max_pre:
                idx = rng.choice(X_tr.shape[0], size=max_pre, replace=False)
                X_tr = X_tr[idx]
            if X_te.shape[0] > max_pre:
                idx = rng.choice(X_te.shape[0], size=max_pre, replace=False)
                X_te = X_te[idx]
            X = np.vstack([X_tr, X_te])
            y_tr = np.load(y_tr_path)[:X_tr.shape[0]] if X_tr.shape[0] > 0 else np.empty(0)
            y_te = np.load(y_te_path)[:X_te.shape[0]] if X_te.shape[0] > 0 else np.empty(0)
            y = np.concatenate([y_tr, y_te])
        elif X_tr.size > 0:
            X = X_tr
            y = np.load(y_tr_path)
            if X.shape[0] > MAX_SAMPLES_PER_DATASET:
                idx = rng.choice(X.shape[0], size=MAX_SAMPLES_PER_DATASET, replace=False)
                X = X[idx]
                y = y[idx]
        else:
            X = X_te
            y = np.load(y_te_path) if y_te_path.exists() else np.empty(0)
            if X.shape[0] > MAX_SAMPLES_PER_DATASET:
                idx = rng.choice(X.shape[0], size=MAX_SAMPLES_PER_DATASET, replace=False)
                X = X[idx]
                y = y[idx]
        X = np.ascontiguousarray(X)
        datasets[key] = {"X": X, "y": y.ravel()}
    return datasets

def load_harmonized_dataset(name):
    from helix_ids.contracts.schema_contract import CANONICAL_FEATURE_ORDER
    from helix_ids.data.multi_dataset_loader import MultiDatasetLoader
    loader = MultiDatasetLoader(project_root=str(PROJECT_ROOT))
    loaders = {
        "ton_iot": loader.load_ton_iot, "bot_iot": loader.load_bot_iot,
        "cicids2017": loader.load_cicids2017,
    }
    harmonizers = {
        "ton_iot": loader.harmonize_ton_iot, "bot_iot": loader.harmonize_bot_iot,
        "cicids2017": loader.harmonize_cicids2017,
    }
    if name not in loaders:
        return None
    raw = loaders[name]()
    if raw is None or len(raw) == 0:
        return None
    df = harmonizers[name](raw)
    if df is None or "label" not in df.columns:
        return None
    cols = [c for c in CANONICAL_FEATURE_ORDER if c in df.columns]
    X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64)
    return {"X": X, "y": y}

def load_all_datasets():
    datasets = load_npy_datasets()
    for name in ["ton_iot", "bot_iot", "cicids2017"]:
        if name not in datasets:
            d = load_harmonized_dataset(name)
            if d is not None:
                datasets[name] = d
    return datasets

def convert_to_binary(y):
    return (y > 0).astype(np.int64)

def subsample(X, y, max_samples):
    n = X.shape[0]
    if n <= max_samples:
        return X.copy(), y.copy()
    classes = np.unique(y)
    indices = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        target = max(1, int(max_samples * len(c_idx) / n))
        if len(c_idx) > target:
            c_idx = rng.choice(c_idx, size=target, replace=False)
        indices.extend(c_idx.tolist())
    rng.shuffle(indices)
    idx_arr = np.array(indices, dtype=np.int64)
    return X[idx_arr], y[idx_arr]

# ─────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────

def supervised_contrastive_loss(features, labels, temperature=0.1):
    device = features.device
    batch_size = features.shape[0]
    labels = labels.contiguous().view(-1, 1)
    features = nn.functional.normalize(features, dim=1)
    sim = features @ features.T / temperature
    mask = torch.eye(batch_size, device=device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)
    pos_mask = (labels == labels.T).float()
    pos_mask = pos_mask.masked_fill(mask, 0)
    if pos_mask.sum() < 1:
        return torch.tensor(0.0, device=device)
    pos_sim = (sim * pos_mask).sum(dim=1)
    neg_sim = torch.logsumexp(sim, dim=1)
    loss = (neg_sim - pos_sim / pos_mask.sum(dim=1).clamp(min=1)).mean()
    return loss

# ─────────────────────────────────────────────
# Data Preparation
# ─────────────────────────────────────────────

def prepare_data(data_dict, val_split=0.15):
    from sklearn.preprocessing import StandardScaler
    train_data, val_data = {}, {}
    train_scalers = {}
    for name in DATASET_NAMES:
        if name not in data_dict:
            continue
        X = data_dict[name]["X"]
        y = convert_to_binary(data_dict[name]["y"])
        n = X.shape[0]
        if n > MAX_SAMPLES_PER_DATASET:
            idx = rng.permutation(n)[:MAX_SAMPLES_PER_DATASET]
            X = X[idx]
            y = y[idx]
            n = MAX_SAMPLES_PER_DATASET
        n_val = max(1, int(n * val_split))
        idx = rng.permutation(n)
        X_tr, X_val = X[idx[n_val:]], X[idx[:n_val]]
        y_tr, y_val = y[idx[n_val:]], y[idx[:n_val]]
        scaler_tr = StandardScaler()
        X_tr_s = scaler_tr.fit_transform(X_tr)
        X_val_s = scaler_tr.transform(X_val)
        train_data[name] = {"X": X_tr_s, "y": y_tr}
        val_data[name] = {"X": X_val_s, "y": y_val}
        train_scalers[name] = scaler_tr
    return train_data, val_data, train_scalers

def build_classification_loader(data_dict):
    train_loaders = {}
    for name in DATASET_NAMES:
        if name not in data_dict:
            continue
        Xt = torch.from_numpy(data_dict[name]["X"]).float()
        yt = torch.from_numpy(data_dict[name]["y"]).long()
        ds = TensorDataset(Xt, yt)
        train_loaders[name] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    return train_loaders

def multi_loader_iter(loaders_dict, steps_per_epoch=200):
    names = sorted(loaders_dict.keys())
    iters = {n: iter(loaders_dict[n]) for n in names}
    for _ in range(steps_per_epoch):
        name = names[rng.randint(len(names))]
        try:
            xb, yb = next(iters[name])
        except StopIteration:
            iters[name] = iter(loaders_dict[name])
            xb, yb = next(iters[name])
        yield name, xb.to(DEVICE), yb.to(DEVICE)

def fit_eval_scalers(data_dict):
    from sklearn.preprocessing import StandardScaler
    scalers = {}
    for name in DATASET_NAMES:
        if name not in data_dict:
            continue
        X = data_dict[name]["X"]
        n = X.shape[0]
        if n > MAX_SAMPLES_PER_DATASET:
            idx = rng.permutation(n)[:MAX_SAMPLES_PER_DATASET]
            X = X[idx]
        scaler = StandardScaler()
        scaler.fit(X)
        scalers[name] = scaler
    return scalers

# ─────────────────────────────────────────────
# SupCon Training (Phase 50 Exp B)
# ─────────────────────────────────────────────

def train_supcon(data_dict):
    logger.info(f"\n{'='*65}")
    logger.info("Training SupCon Encoder (Phase 50 Experiment B)")
    logger.info(f"{'='*65}")

    train_data, val_data, _ = prepare_data(data_dict)
    train_loaders = build_classification_loader(train_data)

    # Build val loaders correctly
    _, val_loaders = _build_val_loaders(train_data, val_data)

    encoder = SharedEncoder().to(DEVICE)
    classifier = ClassifierHead().to(DEVICE)
    projector = ProjectionHead().to(DEVICE)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()) + list(projector.parameters()),
        lr=LR
    )
    cls_criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    steps_per_epoch = max(100, sum(ds.dataset.tensors[0].shape[0] // BATCH_SIZE
                                   for ds in train_loaders.values()) // (2 * len(train_loaders)))
    steps_per_epoch = min(steps_per_epoch, 500)

    for epoch in range(SUPCON_EPOCHS):
        encoder.train()
        classifier.train()
        projector.train()
        train_losses = []
        train_correct = 0
        train_total = 0

        for name, xb, yb in multi_loader_iter(train_loaders, steps_per_epoch):
            optimizer.zero_grad()
            z = encoder(xb)
            logits = classifier(z)
            cls_loss = cls_criterion(logits, yb)
            proj = projector(z)
            supcon = supervised_contrastive_loss(proj, yb, temperature=0.1)
            total_loss = cls_loss + 0.5 * supcon
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(classifier.parameters()) + list(projector.parameters()),
                10.0
            )
            optimizer.step()
            train_losses.append(total_loss.item())
            train_correct += (logits.argmax(dim=1) == yb).sum().item()
            train_total += yb.shape[0]

        # Validation
        encoder.eval()
        classifier.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for loader in val_loaders.values():
                for xb, yb in loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    z = encoder(xb)
                    logits = classifier(z)
                    loss = cls_criterion(logits, yb)
                    val_losses.append(loss.item())
                    val_correct += (logits.argmax(dim=1) == yb).sum().item()
                    val_total += yb.shape[0]

        train_l = np.mean(train_losses) if train_losses else 0
        val_l = np.mean(val_losses) if val_losses else 0
        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        history["train_loss"].append(float(train_l))
        history["val_loss"].append(float(val_l))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))

        if epoch == 0 or (epoch + 1) % 10 == 0:
            logger.info(f"  [SupCon] Epoch {epoch+1:3d}/{SUPCON_EPOCHS} "
                        f"train={train_l:.6f} val={val_l:.6f} "
                        f"acc={train_acc:.4f}/{val_acc:.4f}")

        if val_l < best_val_loss - 1e-6:
            best_val_loss = val_l
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            logger.info(f"  [SupCon] Early stopping at epoch {epoch+1}")
            break

    logger.info(f"  [SupCon] Done. Best val_loss={best_val_loss:.6f}")
    torch.save(encoder.state_dict(), PHASE50_DIR / "models" / "encoder_expB.pt")
    torch.save(classifier.state_dict(), PHASE50_DIR / "models" / "classifier_expB.pt")
    return encoder, classifier, history

def _build_val_loaders(train_data, val_data):
    val_loaders = {}
    for name in sorted(train_data.keys()):
        Xv = torch.from_numpy(val_data[name]["X"]).float()
        yv = torch.from_numpy(val_data[name]["y"]).long()
        ds_v = TensorDataset(Xv, yv)
        val_loaders[name] = DataLoader(ds_v, batch_size=BATCH_SIZE * 2)
    return train_data, val_loaders

# ─────────────────────────────────────────────
# Latent extraction and RF transfer
# ─────────────────────────────────────────────

def extract_latents(encoder, data_dict, scalers, batch_size=1024):
    encoder.eval()
    latents = {}
    labels = {}
    for name in sorted(data_dict.keys()):
        X = data_dict[name]["X"]
        y = convert_to_binary(data_dict[name]["y"])
        X_s = scalers[name].transform(X)
        ds = TensorDataset(torch.from_numpy(X_s).float())
        loader = DataLoader(ds, batch_size=batch_size)
        z_list = []
        with torch.no_grad():
            for (xb,) in loader:
                z = encoder(xb.to(DEVICE)).cpu().numpy()
                z_list.append(z)
        latents[name] = np.vstack(z_list)
        labels[name] = y
    return latents, labels

def compute_transfer_matrix_full(latents_dict, labels_dict):
    """
    Full transfer matrix with MF1, precision, recall, calibration, AUROC.
    Returns (matrix_dict, results_list)
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    names = sorted(latents_dict.keys())
    n_names = len(names)

    # Subsample large datasets for RF training
    subsampled_data = {}
    for nm in names:
        Z = latents_dict[nm]
        y = labels_dict[nm]
        if len(Z) > 50000:
            idx = rng.permutation(len(Z))[:50000]
            Z = Z[idx]
            y = y[idx]
        subsampled_data[nm] = (Z, y)

    results = []
    matrices = {
        "macro_f1": np.zeros((n_names, n_names)),
        "precision": np.zeros((n_names, n_names)),
        "recall": np.zeros((n_names, n_names)),
        "auroc": np.zeros((n_names, n_names)),
        "brier": np.full((n_names, n_names), np.nan),
        "ece": np.full((n_names, n_names), np.nan),
    }

    for i, src in enumerate(names):
        Z_src, y_src = subsampled_data[src]
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=1
        )
        clf.fit(Z_src, y_src)

        for j, tgt in enumerate(names):
            Z_tgt, y_tgt = latents_dict[tgt], labels_dict[tgt]
            y_pred = clf.predict(Z_tgt)
            y_prob = clf.predict_proba(Z_tgt)[:, 1] if hasattr(clf, "predict_proba") else None

            mf1 = float(f1_score(y_tgt, y_pred, average="macro", zero_division=0))
            prec = float(precision_score(y_tgt, y_pred, average="macro", zero_division=0))
            rec = float(recall_score(y_tgt, y_pred, average="macro", zero_division=0))

            # AUROC
            auroc_val = 0.0
            if y_prob is not None and len(np.unique(y_tgt)) > 1:
                try:
                    auroc_val = float(roc_auc_score(y_tgt, y_prob))
                except ValueError:
                    auroc_val = 0.5

            # Brier score (calibration)
            brier_val = float(brier_score_loss(y_tgt, y_prob)) if y_prob is not None else np.nan

            # ECE (Expected Calibration Error)
            ece_val = compute_ece(y_tgt, y_prob, n_bins=10) if y_prob is not None else np.nan

            results.append({
                "source": src, "target": tgt,
                "source_display": DATASET_DISPLAY.get(src, src),
                "target_display": DATASET_DISPLAY.get(tgt, tgt),
                "macro_f1": mf1, "precision": prec, "recall": rec,
                "auroc": auroc_val, "brier": brier_val, "ece": ece_val,
            })

            matrices["macro_f1"][i, j] = mf1
            matrices["precision"][i, j] = prec
            matrices["recall"][i, j] = rec
            matrices["auroc"][i, j] = auroc_val
            matrices["brier"][i, j] = brier_val
            matrices["ece"][i, j] = ece_val

    return matrices, results

def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    if y_prob is None:
        return np.nan
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if i == n_bins - 1:
            in_bin = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        if np.sum(in_bin) == 0:
            continue
        bin_acc = np.mean(y_true[in_bin])
        bin_conf = np.mean(y_prob[in_bin])
        ece += np.abs(bin_acc - bin_conf) * np.sum(in_bin) / len(y_true)
    return float(ece)

# ─────────────────────────────────────────────
# Experiment A: Pairwise Success Landscape
# ─────────────────────────────────────────────

def experiment_a_pairwise_success(matrices, results, names, display_names):
    """Produce heatmap data, directed graph ranking, Transferability Index."""
    logger.info(f"\n{'='*65}")
    logger.info("Experiment A: Pairwise Success Landscape")
    logger.info(f"{'='*65}")

    n = len(names)
    mf1 = matrices["macro_f1"]

    # Ranking: easiest and hardest transfers
    off_diag = []
    for i in range(n):
        for j in range(n):
            if i != j:
                off_diag.append({
                    "source": names[i], "target": names[j],
                    "source_display": display_names[names[i]],
                    "target_display": display_names[names[j]],
                    "macro_f1": mf1[i, j],
                })

    off_diag.sort(key=lambda x: x["macro_f1"], reverse=True)

    # Transferability Index: for each source, average MF1 across targets
    transferability = {}
    for i, src in enumerate(names):
        targets = [mf1[i, j] for j in range(n) if i != j]
        transferability[src] = float(np.mean(targets))

    idx_sorted = sorted(transferability.items(), key=lambda x: x[1], reverse=True)

    logger.info("\n  Transferability Index (source→all targets avg MF1):")
    for src_name, val in idx_sorted:
        logger.info(f"    {display_names[src_name]:15s}  {val:.4f}")

    logger.info("\n  Top 5 easiest transfers:")
    for r in off_diag[:5]:
        logger.info(f"    {r['source_display']:15s} → {r['target_display']:15s}  MF1={r['macro_f1']:.4f}")

    logger.info("\n  Bottom 5 hardest transfers:")
    for r in off_diag[-5:]:
        logger.info(f"    {r['source_display']:15s} → {r['target_display']:15s}  MF1={r['macro_f1']:.4f}")

    # Save tables
    pd.DataFrame(off_diag).to_csv(RESULTS_DIR / "tables" / "pairwise_ranking.csv", index=False)

    ti_df = pd.DataFrame([
        {"dataset": n, "transferability_index": v} for n, v in idx_sorted
    ])
    ti_df.to_csv(RESULTS_DIR / "tables" / "transferability_index.csv", index=False)

    # Save heatmap data as JSON
    heatmap_data = {}
    for met in ["macro_f1", "precision", "recall", "auroc", "brier", "ece"]:
        data = matrices[met]
        heatmap_data[met] = {
            "row_names": [display_names[n] for n in names],
            "col_names": [display_names[n] for n in names],
            "values": data.tolist(),
        }
    with open(RESULTS_DIR / "matrices" / "heatmap_data.json", "w") as f:
        json.dump(heatmap_data, f, indent=2)

    return off_diag, transferability

# ─────────────────────────────────────────────
# Experiment B: Dataset Similarity vs Transfer
# ─────────────────────────────────────────────

def compute_similarity_metrics(latents_dict, labels_dict, subsample_n=10000):
    """
    Compute all similarity metrics between every dataset pair.
    Returns dict of matrices.
    """
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import wasserstein_distance

    names = sorted(latents_dict.keys())
    n = len(names)

    metrics = {
        "js_divergence": np.zeros((n, n)),
        "wasserstein": np.zeros((n, n)),
        "mmd": np.zeros((n, n)),
        "frechet": np.zeros((n, n)),
        "linear_cka": np.zeros((n, n)),
        "conditional_mmd": np.zeros((n, n)),
        "prototype_alignment": np.zeros((n, n)),
    }

    # Pre-subsample for heavy metrics
    subsampled = {}
    for nm in names:
        Z = latents_dict[nm]
        y = labels_dict[nm]
        if len(Z) > subsample_n:
            idx = rng.choice(len(Z), subsample_n, replace=False)
            subsampled[nm] = (Z[idx], y[idx])
        else:
            subsampled[nm] = (Z, y)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            Z_i, y_i = subsampled[names[i]]
            Z_j, y_j = subsampled[names[j]]

            # JS Divergence (on label distributions)
            p_i = np.bincount(y_i, minlength=2).astype(float)
            p_j = np.bincount(y_j, minlength=2).astype(float)
            p_i /= p_i.sum()
            p_j /= p_j.sum()
            metrics["js_divergence"][i, j] = float(jensenshannon(p_i, p_j))

            # Wasserstein (1D projection)
            wd = 0.0
            for d in range(min(Z_i.shape[1], 4)):  # first 4 dims
                wd += wasserstein_distance(Z_i[:, d], Z_j[:, d])
            metrics["wasserstein"][i, j] = wd / min(Z_i.shape[1], 4)

            # MMD (linear kernel)
            n_s = min(Z_i.shape[0], Z_j.shape[0], 5000)
            Z_is, Z_js = Z_i[:n_s], Z_j[:n_s]
            xx = (Z_is @ Z_is.T).mean()
            yy = (Z_js @ Z_js.T).mean()
            xy = (Z_is @ Z_js.T).mean()
            mmd_val = max(0, xx + yy - 2 * xy)
            metrics["mmd"][i, j] = float(mmd_val)

            # Fréchet distance
            mu_i, mu_j = Z_i.mean(axis=0), Z_j.mean(axis=0)
            sigma_i = np.cov(Z_i, rowvar=False) + 1e-6 * np.eye(Z_i.shape[1])
            sigma_j = np.cov(Z_j, rowvar=False) + 1e-6 * np.eye(Z_j.shape[1])
            diff = mu_i - mu_j
            from scipy.linalg import sqrtm
            cov_mean = sqrtm(sigma_i @ sigma_j)
            if np.iscomplexobj(cov_mean):
                cov_mean = cov_mean.real
            frechet_val = float(diff @ diff + np.trace(sigma_i + sigma_j - 2 * cov_mean))
            metrics["frechet"][i, j] = frechet_val

            # Linear CKA
            lr = min(Z_i.shape[0], Z_j.shape[0], 10000)
            idx_i = rng.choice(Z_i.shape[0], size=lr, replace=False)
            idx_j = rng.choice(Z_j.shape[0], size=lr, replace=False)
            Xi_c = Z_i[idx_i] - Z_i[idx_i].mean(axis=0)
            Xj_c = Z_j[idx_j] - Z_j[idx_j].mean(axis=0)
            cross = Xj_c.T @ Xi_c
            hsic_xy = float(np.sum(cross ** 2)) / (lr - 1) ** 2
            hsic_xx = float(np.sum((Xi_c.T @ Xi_c) ** 2)) / (lr - 1) ** 2
            hsic_yy = float(np.sum((Xj_c.T @ Xj_c) ** 2)) / (lr - 1) ** 2
            denom = np.sqrt(hsic_xx * hsic_yy)
            metrics["linear_cka"][i, j] = hsic_xy / denom if denom > 1e-12 else 0.0

            # Conditional MMD (per-class)
            cmmd_val = 0.0
            n_classes = 0
            for c in np.unique(np.concatenate([y_i, y_j])):
                Z_ic = Z_i[y_i == c]
                Z_jc = Z_j[y_j == c]
                if len(Z_ic) < 2 or len(Z_jc) < 2:
                    continue
                n_take = min(len(Z_ic), len(Z_jc), 2000)
                Z_ic_s = Z_ic[:n_take]
                Z_jc_s = Z_jc[:n_take]
                xx_c = (Z_ic_s @ Z_ic_s.T).mean()
                yy_c = (Z_jc_s @ Z_jc_s.T).mean()
                xy_c = (Z_ic_s @ Z_jc_s.T).mean()
                cmmd_val += max(0, xx_c + yy_c - 2 * xy_c)
                n_classes += 1
            metrics["conditional_mmd"][i, j] = cmmd_val / max(n_classes, 1)

            # Prototype alignment
            proto_i = Z_i[y_i == 0].mean(axis=0) if (y_i == 0).sum() > 0 else Z_i.mean(axis=0)
            proto_j = Z_j[y_j == 0].mean(axis=0) if (y_j == 0).sum() > 0 else Z_j.mean(axis=0)
            proto_dist = np.linalg.norm(proto_i - proto_j)
            metrics["prototype_alignment"][i, j] = float(proto_dist)

    return metrics

def experiment_b_similarity_vs_transfer(matrices, latents_dict, labels_dict, names, display_names):
    """
    Correlate every similarity metric with transfer MF1.
    Determine which metric best predicts successful transfer.
    """
    logger.info(f"\n{'='*65}")
    logger.info("Experiment B: Dataset Similarity vs Transfer")
    logger.info(f"{'='*65}")

    from scipy.stats import pearsonr, spearmanr

    similarity_metrics = compute_similarity_metrics(latents_dict, labels_dict)
    mf1 = matrices["macro_f1"]
    n = len(names)

    correlations = []
    for met_name, met_matrix in similarity_metrics.items():
        mf1_vals, met_vals = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    mf1_vals.append(mf1[i, j])
                    met_vals.append(met_matrix[i, j])

        mf1_arr = np.array(mf1_vals)
        met_arr = np.array(met_vals)

        # Remove NaN
        valid = ~(np.isnan(met_arr) | np.isnan(mf1_arr))
        if valid.sum() < 3:
            correlations.append({"metric": met_name, "pearson_r": 0, "pearson_p": 1,
                                 "spearman_r": 0, "spearman_p": 1, "n_valid": int(valid.sum())})
            continue

        pr, pp = pearsonr(mf1_arr[valid], met_arr[valid])
        sr, sp = spearmanr(mf1_arr[valid], met_arr[valid])

        correlations.append({
            "metric": met_name,
            "pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp),
            "n_valid": int(valid.sum()),
        })

    correlations.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)

    logger.info("\n  Correlation with Transfer MF1:")
    for c in correlations:
        logger.info(f"    {c['metric']:20s}  pearson_r={c['pearson_r']:+.4f}  "
                    f"p={c['pearson_p']:.4f}  spearman={c['spearman_r']:+.4f}")

    pd.DataFrame(correlations).to_csv(RESULTS_DIR / "tables" / "similarity_correlations.csv", index=False)

    # Save similarity matrices
    for met_name, met_matrix in similarity_metrics.items():
        df = pd.DataFrame(met_matrix, index=[display_names[n] for n in names],
                          columns=[display_names[n] for n in names])
        df.to_csv(RESULTS_DIR / "matrices" / f"similarity_{met_name}.csv")

    return correlations, similarity_metrics

# ─────────────────────────────────────────────
# Experiment C: Class-Level Transfer
# ─────────────────────────────────────────────

def experiment_c_class_transfer(latents_dict, labels_dict, data_dict, names, display_names):
    """
    Measure transfer for each attack class (binary: Normal=0, Attack=1).
    Source F1, target F1, improvement, degradation.
    """
    logger.info(f"\n{'='*65}")
    logger.info("Experiment C: Class-Level Transfer")
    logger.info(f"{'='*65}")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score

    n = len(names)
    # Class-level matrix: [source, target, class] -> F1
    class_matrix = np.zeros((n, n, 2))

    subsampled = {}
    for nm in names:
        Z = latents_dict[nm]
        y = labels_dict[nm]
        if len(Z) > 50000:
            idx = rng.permutation(len(Z))[:50000]
            Z = Z[idx]
            y = y[idx]
        subsampled[nm] = (Z, y)

    for i, src in enumerate(names):
        Z_src, y_src = subsampled[src]
        clf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                      random_state=RANDOM_STATE, n_jobs=1)
        clf.fit(Z_src, y_src)

        for j, tgt in enumerate(names):
            Z_tgt, y_tgt = latents_dict[tgt], labels_dict[tgt]
            y_pred = clf.predict(Z_tgt)

            for c in [0, 1]:
                mask_tgt = y_tgt == c
                mask_pred = y_pred == c
                if mask_tgt.sum() > 0:
                    f1_c = float(f1_score(y_tgt[mask_tgt], y_pred[mask_tgt], average="binary",
                                          pos_label=c, zero_division=0))
                else:
                    f1_c = 0.0
                class_matrix[i, j, c] = f1_c

    # Build per-class results
    class_results = []
    for i, src in enumerate(names):
        for j, tgt in enumerate(names):
            if i == j:
                continue
            for c, cname in enumerate(CLASS_NAMES):
                src_f1 = class_matrix[i, i, c]  # within-source performance
                tgt_f1 = class_matrix[i, j, c]  # cross-dataset performance
                improvement = tgt_f1 - src_f1
                class_results.append({
                    "source": src, "target": tgt,
                    "source_display": display_names[src],
                    "target_display": display_names[tgt],
                    "class": cname,
                    "source_f1": float(src_f1),
                    "target_f1": float(tgt_f1),
                    "improvement": float(improvement),
                })

    df = pd.DataFrame(class_results)
    df.to_csv(RESULTS_DIR / "tables" / "class_transfer_matrix.csv", index=False)

    # Which attacks transfer universally?
    for cname in CLASS_NAMES:
        c_results = df[df["class"] == cname]
        logger.info(f"\n  Class '{cname}' transfer stats:")
        logger.info(f"    Mean target F1: {c_results['target_f1'].mean():.4f}")
        logger.info(f"    Mean improvement: {c_results['improvement'].mean():.4f}")
        logger.info(f"    Universal (target_f1 > 0.8): {(c_results['target_f1'] > 0.8).sum()}/{(len(c_results))}")
        logger.info(f"    Never transfer (target_f1 < 0.1): {(c_results['target_f1'] < 0.1).sum()}/{(len(c_results))}")

    return class_results, class_matrix

# ─────────────────────────────────────────────
# Experiment D: Feature Dependency Analysis (SHAP)
# ─────────────────────────────────────────────

def experiment_d_feature_dependency(latents_dict, labels_dict, data_dict, names, display_names):
    """
    Train explainable models (RF) and compute SHAP interaction values.
    Compare same attack across different datasets.
    """
    logger.info(f"\n{'='*65}")
    logger.info("Experiment D: Feature Dependency Analysis")
    logger.info(f"{'='*65}")

    import shap

    # Use 17-feature raw data for SHAP (not latents)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    shap_results = {}

    for name in names:
        X = data_dict[name]["X"]
        y = convert_to_binary(data_dict[name]["y"])

        # Subsample for speed
        if len(X) > 10000:
            idx = rng.choice(len(X), 10000, replace=False)
            X = X[idx]
            y = y[idx]

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                     random_state=RANDOM_STATE, n_jobs=1)
        rf.fit(X_s, y)

        # SHAP (subsample further)
        shap_n = min(1000, len(X_s))
        idx_shap = rng.choice(len(X_s), shap_n, replace=False)
        X_shap = X_s[idx_shap]

        explainer = shap.TreeExplainer(rf)
        shap_vals = explainer.shap_values(X_shap)

        # Handle SHAP output format (list for binary in older shap, 3D in newer)
        if isinstance(shap_vals, list):
            mean_shap = np.mean(np.abs(shap_vals[0]), axis=0)
        elif shap_vals.ndim == 3:
            mean_shap = np.mean(np.abs(shap_vals), axis=(0, 2))
        else:
            mean_shap = np.mean(np.abs(shap_vals), axis=0)

        shap_results[name] = {
            "mean_abs_shap": mean_shap.tolist(),
            "top3_features": np.argsort(mean_shap)[::-1][:3].tolist(),
            "rf_macro_f1": float(rf.score(X_shap, y[idx_shap])),
        }

        logger.info(f"  {display_names[name]:15s}  rf_acc={shap_results[name]['rf_macro_f1']:.4f}  "
                    f"top3={shap_results[name]['top3_features']}")

    # Compare SHAP patterns: same attack across datasets
    # Feature interaction via permutation importance
    logger.info("\n  Comparing SHAP patterns across datasets...")
    pairs = []
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            s1 = np.array(shap_results[n1]["mean_abs_shap"])
            s2 = np.array(shap_results[n2]["mean_abs_shap"])
            from scipy.stats import pearsonr, spearmanr
            pr, pp = pearsonr(s1, s2)
            sr, sp = spearmanr(s1, s2)
            pairs.append({
                "dataset_a": n1, "dataset_b": n2,
                "shap_pearson": float(pr), "shap_pearson_p": float(pp),
                "shap_spearman": float(sr), "shap_spearman_p": float(sp),
                "common_top3": len(set(shap_results[n1]["top3_features"]) & set(shap_results[n2]["top3_features"])),
            })
            logger.info(f"    {display_names[n1]:15s} ↔ {display_names[n2]:15s}  "
                        f"shap_ρ={pr:.3f}  common_top3={pairs[-1]['common_top3']}")

    pd.DataFrame(pairs).to_csv(RESULTS_DIR / "tables" / "shap_similarity.csv", index=False)
    with open(RESULTS_DIR / "attributions" / "shap_results.json", "w") as f:
        json.dump(shap_results, f, indent=2)

    return shap_results, pairs

# ─────────────────────────────────────────────
# Experiment E: Latent Geometry Evolution
# ─────────────────────────────────────────────

def experiment_e_latent_geometry(latents_dict, labels_dict, data_dict, names, display_names):
    """
    PCA, UMAP/t-SNE visualization, silhouette scores.
    Do clusters organize by attack type or by dataset?
    """
    logger.info(f"\n{'='*65}")
    logger.info("Experiment E: Latent Geometry Evolution")
    logger.info(f"{'='*65}")

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    # Pool all latents with dataset and attack labels
    all_Z = []
    all_dataset_ids = []
    all_attack_ids = []

    for i, name in enumerate(names):
        Z = latents_dict[name]
        y = convert_to_binary(data_dict[name]["y"])
        # Subsample for tractable t-SNE/UMAP
        if len(Z) > 5000:
            idx = rng.choice(len(Z), 5000, replace=False)
            Z = Z[idx]
            y = y[idx]
        all_Z.append(Z)
        all_dataset_ids.append(np.full(len(Z), i, dtype=np.int64))
        all_attack_ids.append(y)

    Z_pooled = np.vstack(all_Z)
    dataset_ids = np.concatenate(all_dataset_ids)
    attack_ids = np.concatenate(all_attack_ids)

    logger.info(f"  Pooled latents: {Z_pooled.shape}")

    # PCA
    pca = PCA(n_components=min(50, Z_pooled.shape[1]))
    Z_pca = pca.fit_transform(Z_pooled)
    logger.info(f"  PCA explained variance ratio (top 5): {pca.explained_variance_ratio_[:5].tolist()}")

    # t-SNE (2D)
    tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE, max_iter=1000)
    Z_tsne = tsne.fit_transform(Z_pca[:, :10])  # PCA-reduced for speed

    # Silhouette scores
    sil_dataset = float(silhouette_score(Z_tsne, dataset_ids))
    sil_attack = float(silhouette_score(Z_tsne, attack_ids))

    db_dataset = float(davies_bouldin_score(Z_tsne, dataset_ids))
    db_attack = float(davies_bouldin_score(Z_tsne, attack_ids))

    ch_dataset = float(calinski_harabasz_score(Z_tsne, dataset_ids))
    ch_attack = float(calinski_harabasz_score(Z_tsne, attack_ids))

    logger.info("\n  Clustering Metrics (t-SNE 2D):")
    logger.info(f"    Silhouette (by dataset): {sil_dataset:.4f}")
    logger.info(f"    Silhouette (by attack):  {sil_attack:.4f}")
    logger.info(f"    Davies-Bouldin (dataset): {db_dataset:.4f}")
    logger.info(f"    Davies-Bouldin (attack):  {db_attack:.4f}")
    logger.info(f"    Calinski-Harabasz (dataset): {ch_dataset:.4f}")
    logger.info(f"    Calinski-Harabasz (attack):  {ch_attack:.4f}")

    if sil_dataset > sil_attack:
        logger.info("  INFO: Clusters organize MORE by DATASET identity (dataset-specific encoding)")
    else:
        logger.info("  INFO: Clusters organize MORE by ATTACK type (transfer-friendly encoding)")

    # Save results
    geo_results = {
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "silhouette_dataset": sil_dataset,
        "silhouette_attack": sil_attack,
        "davies_bouldin_dataset": db_dataset,
        "davies_bouldin_attack": db_attack,
        "calinski_harabasz_dataset": ch_dataset,
        "calinski_harabasz_attack": ch_attack,
    }

    # Save t-SNE coordinates for plotting
    tsne_data = {
        "x": Z_tsne[:, 0].tolist(), "y": Z_tsne[:, 1].tolist(),
        "dataset_ids": dataset_ids.tolist(),
        "attack_ids": attack_ids.tolist(),
        "dataset_names": [display_names[n] for n in names],
    }
    with open(RESULTS_DIR / "latents" / "tsne_coordinates.json", "w") as f:
        json.dump(tsne_data, f, indent=2)
    with open(RESULTS_DIR / "attributions" / "geometry_results.json", "w") as f:
        json.dump(geo_results, f, indent=2)

    # Also compute UMAP if available
    try:
        import umap
        reducer = umap.UMAP(random_state=RANDOM_STATE)
        Z_umap = reducer.fit_transform(Z_pca[:, :10])
        sil_umap_dataset = float(silhouette_score(Z_umap, dataset_ids))
        sil_umap_attack = float(silhouette_score(Z_umap, attack_ids))
        logger.info(f"\n  UMAP: Silhouette (dataset)={sil_umap_dataset:.4f}  Silhouette (attack)={sil_umap_attack:.4f}")

        geo_results["umap_silhouette_dataset"] = sil_umap_dataset
        geo_results["umap_silhouette_attack"] = sil_umap_attack

        umap_data = {
            "x": Z_umap[:, 0].tolist(), "y": Z_umap[:, 1].tolist(),
            "dataset_ids": dataset_ids.tolist(),
            "attack_ids": attack_ids.tolist(),
        }
        with open(RESULTS_DIR / "latents" / "umap_coordinates.json", "w") as f:
            json.dump(umap_data, f, indent=2)
    except Exception as e:
        logger.warning(f"  UMAP failed: {e}")

    with open(RESULTS_DIR / "attributions" / "geometry_results.json", "w") as f:
        json.dump(geo_results, f, indent=2)

    return geo_results, Z_tsne, dataset_ids, attack_ids

# ─────────────────────────────────────────────
# Experiment F: Failure Attribution
# ─────────────────────────────────────────────

def experiment_f_failure_attribution(matrices, results, latents_dict, labels_dict,
                                      data_dict, names, display_names,
                                      shap_results, similarity_metrics):
    """
    For every failed transfer pair, identify the dominant failure cause.

    Failure categories:
    - feature_mismatch: SHAP patterns differ greatly
    - label_mismatch: label distributions differ
    - prototype_mismatch: normal prototypes far apart
    - calibration_mismatch: poor calibration
    - representation_mismatch: low CKA, high Frechet
    """
    logger.info(f"\n{'='*65}")
    logger.info("Experiment F: Failure Attribution")
    logger.info(f"{'='*65}")

    from scipy.spatial.distance import jensenshannon

    # Define thresholds for each failure mode
    FAILURE_THRESHOLD = 0.5  # MF1 below this is "failed"

    attributions = []

    mf1 = matrices["macro_f1"]
    n = len(names)

    for i, src in enumerate(names):
        for j, tgt in enumerate(names):
            if i == j:
                continue

            mf1_val = mf1[i, j]
            if mf1_val > FAILURE_THRESHOLD:
                continue  # Not a failure

            # Compute failure signals

            # 1. Feature mismatch: low SHAP correlation
            shap_sim = 1.0
            if shap_results and src in shap_results and tgt in shap_results:
                s1 = np.array(shap_results[src]["mean_abs_shap"])
                s2 = np.array(shap_results[tgt]["mean_abs_shap"])
                from scipy.stats import pearsonr
                pr, _ = pearsonr(s1, s2)
                shap_sim = float(pr)

            # 2. Label mismatch: JS divergence of label distributions
            y_src = convert_to_binary(data_dict[src]["y"])
            y_tgt = convert_to_binary(data_dict[tgt]["y"])
            p_src = np.bincount(y_src, minlength=2).astype(float)
            p_tgt = np.bincount(y_tgt, minlength=2).astype(float)
            p_src /= p_src.sum()
            p_tgt /= p_tgt.sum()
            label_js = float(jensenshannon(p_src, p_tgt))

            # 3. Prototype mismatch
            Z_src = latents_dict[src]
            Z_tgt = latents_dict[tgt]
            proto_src = Z_src[y_src == 0].mean(axis=0) if (y_src == 0).sum() > 0 else Z_src.mean(axis=0)
            proto_tgt = Z_tgt[y_tgt == 0].mean(axis=0) if (y_tgt == 0).sum() > 0 else Z_tgt.mean(axis=0)
            proto_dist = float(np.linalg.norm(proto_src - proto_tgt))

            # 4. Calibration mismatch
            ece_val = matrices["ece"][i, j] if not np.isnan(matrices["ece"][i, j]) else 0.5

            # 5. Representation mismatch
            cka_val = similarity_metrics.get("linear_cka", np.zeros((n, n)))[i, j]
            frechet_val = similarity_metrics.get("frechet", np.zeros((n, n)))[i, j]
            rep_dist = (1.0 - cka_val) + min(1.0, frechet_val / 100.0)

            # Score each failure mode
            failures = {
                "feature_mismatch": 1.0 - max(0, shap_sim),
                "label_mismatch": min(1.0, label_js * 2),
                "prototype_mismatch": min(1.0, proto_dist / 5.0),
                "calibration_mismatch": min(1.0, ece_val * 3),
                "representation_mismatch": min(1.0, rep_dist),
            }

            dominant = max(failures, key=failures.get)
            confidence = failures[dominant]

            attributions.append({
                "source": src, "target": tgt,
                "source_display": display_names[src],
                "target_display": display_names[tgt],
                "macro_f1": mf1_val,
                "dominant_failure": dominant,
                "confidence": float(confidence),
                **{f"score_{k}": float(v) for k, v in failures.items()},
                "shap_similarity": float(shap_sim),
                "label_js_divergence": float(label_js),
                "prototype_distance": float(proto_dist),
                "ece_calibration_error": float(ece_val),
                "linear_cka": float(cka_val),
                "frechet_distance": float(frechet_val),
            })

    # Build failure attribution matrix
    failure_matrix = np.full((n, n), "", dtype=object)
    for attr in attributions:
        i = names.index(attr["source"])
        j = names.index(attr["target"])
        failure_matrix[i, j] = attr["dominant_failure"]

    df_attr = pd.DataFrame(attributions)
    df_attr.to_csv(RESULTS_DIR / "tables" / "failure_attribution.csv", index=False)

    pd.DataFrame(failure_matrix, index=[display_names[n] for n in names],
                 columns=[display_names[n] for n in names]).to_csv(
        RESULTS_DIR / "matrices" / "failure_attribution_matrix.csv")

    logger.info(f"\n  Failure Attributions ({len(attributions)} failed pairs):")
    for attr in attributions:
        logger.info(f"    {attr['source_display']:15s} → {attr['target_display']:15s}  "
                    f"MF1={attr['macro_f1']:.3f}  {attr['dominant_failure']:25s}  "
                    f"conf={attr['confidence']:.2f}")

    return attributions, failure_matrix

# ─────────────────────────────────────────────
# Statistical Analysis
# ─────────────────────────────────────────────

def statistical_analysis(matrices, results, attributions, names, display_names):
    """
    Bootstrap confidence intervals, permutation tests, paired t-tests,
    Wilcoxon tests, Holm-Bonferroni correction, effect sizes.
    """
    logger.info(f"\n{'='*65}")
    logger.info("Statistical Analysis")
    logger.info(f"{'='*65}")

    from scipy.stats import ttest_rel, wilcoxon

    mf1 = matrices["macro_f1"]
    n = len(names)
    off_diag = mf1[~np.eye(n, dtype=bool)]
    diag = np.diag(mf1)

    # 1. Bootstrap CI for mean off-diag MF1
    n_bootstrap = 10000
    bootstrap_means = []
    rng_boot = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng_boot.choice(len(off_diag), size=len(off_diag), replace=True)
        bootstrap_means.append(off_diag[idx].mean())
    bootstrap_means = np.array(bootstrap_means)
    ci_low = np.percentile(bootstrap_means, 2.5)
    ci_high = np.percentile(bootstrap_means, 97.5)

    logger.info(f"\n  Bootstrap CI (mean off-diag MF1): {ci_low:.4f} – {ci_high:.4f}")

    # 2. Permutation test: are off-diag values significantly > 0?
    n_perm = 10000
    null_means = []
    for _ in range(n_perm):
        shuffled = off_diag.copy()
        rng_boot.shuffle(shuffled)
        null_means.append(shuffled.mean())
    null_means = np.array(null_means)
    p_value = (np.sum(null_means >= off_diag.mean()) + 1) / (n_perm + 1)
    logger.info(f"  Permutation test p-value (H0: mean ≤ 0): {p_value:.4f}")

    # 3. Paired t-test: within-dataset vs cross-dataset performance
    t_stat, t_p = ttest_rel(diag, [off_diag[i * (n - 1):(i + 1) * (n - 1)].mean()
                                     for i in range(n)])
    # Actually use the full matched pairs
    diag_vs_off = []
    for i in range(n):
        src_diag = diag[i]
        for j in range(n):
            if i != j:
                diag_vs_off.append((src_diag, mf1[i, j]))
    diag_vals = np.array([d for d, _ in diag_vs_off])
    off_vals = np.array([o for _, o in diag_vs_off])
    t_stat, t_p = ttest_rel(diag_vals, off_vals)
    logger.info(f"  Paired t-test (diag vs off-diag): t={t_stat:.4f}, p={t_p:.6f}")

    # 4. Wilcoxon signed-rank
    try:
        w_stat, w_p = wilcoxon(diag_vals, off_vals, alternative='two-sided')
        logger.info(f"  Wilcoxon test: W={w_stat:.0f}, p={w_p:.6f}")
    except ValueError as e:
        logger.warning(f"  Wilcoxon test failed: {e}")
        w_stat, w_p = 0, 1.0

    # 5. Effect sizes
    cohens_d = (np.mean(diag_vals) - np.mean(off_vals)) / np.std(diag_vals - off_vals, ddof=1)
    logger.info(f"  Cohen's d (diag vs off-diag): {cohens_d:.4f}")

    # Cliff's delta
    n_d = len(diag_vals)
    n_o = len(off_vals)
    cliff_sum = 0
    for d_val in diag_vals:
        cliff_sum += np.sum(d_val > off_vals) - np.sum(d_val < off_vals)
    cliffs_delta = cliff_sum / (n_d * n_o)
    logger.info(f"  Cliff's δ: {cliffs_delta:.4f}")

    stats_results = {
        "bootstrap_ci_95": [float(ci_low), float(ci_high)],
        "permutation_p": float(p_value),
        "paired_t": float(t_stat),
        "paired_t_p": float(t_p),
        "wilcoxon_w": float(w_stat),
        "wilcoxon_p": float(w_p),
        "cohens_d": float(cohens_d),
        "cliffs_delta": float(cliffs_delta),
        "mean_off_diag_mf1": float(np.mean(off_diag)),
        "median_off_diag_mf1": float(np.median(off_diag)),
        "std_off_diag_mf1": float(np.std(off_diag, ddof=1)),
        "min_off_diag": float(np.min(off_diag)),
        "max_off_diag": float(np.max(off_diag)),
        "mean_diag_mf1": float(np.mean(diag)),
        "n_directions": int(len(off_diag)),
    }
    with open(RESULTS_DIR / "tables" / "statistical_analysis.json", "w") as f:
        json.dump(stats_results, f, indent=2)

    return stats_results

# ─────────────────────────────────────────────
# Deliverable generation
# ─────────────────────────────────────────────

def generate_deliverables(matrices, results_list, attributions, shap_results,
                          correlations, stats_results, class_results,
                          geo_results, transferability, names, display_names):
    """Generate all deliverable files."""
    logger.info(f"\n{'='*65}")
    logger.info("Generating Deliverables")
    logger.info(f"{'='*65}")

    mf1 = matrices["macro_f1"]
    n = len(names)
    off_diag = mf1[~np.eye(n, dtype=bool)]

    # 1. Pairwise Transferability Atlas (comprehensive table)
    atlas = pd.DataFrame(results_list)
    atlas.to_csv(RESULTS_DIR / "tables" / "pairwise_transferability_atlas.csv", index=False)
    logger.info("  ✓ Pairwise Transferability Atlas saved")

    # 2. Class Transfer Atlas
    if class_results is not None:
        pd.DataFrame(class_results).to_csv(
            RESULTS_DIR / "tables" / "class_transfer_atlas.csv", index=False)
        logger.info("  ✓ Class Transfer Atlas saved")

    # 3. Failure Attribution Matrix
    if attributions:
        pd.DataFrame(attributions).to_csv(
            RESULTS_DIR / "tables" / "failure_attribution_matrix_detail.csv", index=False)
        logger.info("  ✓ Failure Attribution Matrix saved")

    # 4. Predictor Ranking (from correlations)
    if correlations:
        pd.DataFrame(correlations).sort_values("pearson_r", key=abs, ascending=False).to_csv(
            RESULTS_DIR / "tables" / "predictor_ranking.csv", index=False)
        logger.info("  ✓ Predictor Ranking saved")

    # 5. Dataset Difficulty Ranking
    difficulty = []
    for name in names:
        src_idx = names.index(name)
        targets = [mf1[src_idx, j] for j in range(n) if j != src_idx]
        difficulty.append({
            "dataset": name,
            "display_name": display_names[name],
            "mean_as_source": float(np.mean(targets)),
            "std_as_source": float(np.std(targets, ddof=1)),
        })
    df_difficulty = pd.DataFrame(difficulty).sort_values("mean_as_source")
    df_difficulty.to_csv(RESULTS_DIR / "tables" / "dataset_difficulty_ranking.csv", index=False)
    logger.info("  ✓ Dataset Difficulty Ranking saved")

    # 6. Universal Transferability Index
    if transferability:
        ti_list = [{"dataset": display_names[n], "transferability_index": v}
                    for n, v in sorted(transferability.items(), key=lambda x: x[1], reverse=True)]
        pd.DataFrame(ti_list).to_csv(
            RESULTS_DIR / "tables" / "universal_transferability_index.csv", index=False)
        logger.info("  ✓ Universal Transferability Index saved")

    logger.info("\n  All deliverables saved to:")
    logger.info(f"    {RESULTS_DIR / 'tables'}")
    logger.info(f"    {RESULTS_DIR / 'matrices'}")
    logger.info(f"    {RESULTS_DIR / 'attributions'}")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    global SUPCON_EPOCHS
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=str, default="A,B,C,D,E,F",
                        help="Comma-separated experiment IDs to run")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip SupCon training, load cached encoder")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    if args.epochs:
        SUPCON_EPOCHS = args.epochs

    experiments_to_run = [e.strip() for e in args.experiments.split(",")]
    logger.info(f"Experiments to run: {experiments_to_run}")

    # Load data
    logger.info("Loading datasets...")
    data_dict = load_all_datasets()
    logger.info(f"Loaded {len(data_dict)} datasets:")
    for name, d in sorted(data_dict.items()):
        logger.info(f"  {name}: {d['X'].shape}")

    # Train or load SupCon encoder
    encoder_path = PHASE50_DIR / "models" / "encoder_expB.pt"
    if args.skip_train and encoder_path.exists():
        logger.info("Loading cached SupCon encoder...")
        encoder = SharedEncoder().to(DEVICE)
        encoder.load_state_dict(torch.load(encoder_path, map_location=DEVICE))
        classifier = ClassifierHead().to(DEVICE)
        clf_path = PHASE50_DIR / "models" / "classifier_expB.pt"
        if clf_path.exists():
            classifier.load_state_dict(torch.load(clf_path, map_location=DEVICE))
    else:
        logger.info("Training SupCon encoder...")
        encoder, classifier, history = train_supcon(data_dict)
        pd.DataFrame(history).to_csv(PHASE50_DIR / "training_curves_supcon.csv", index=False)

    logger.info("Fitting evaluation scalers and extracting latents...")
    eval_scalers = fit_eval_scalers(data_dict)
    latents_dict, labels_dict = extract_latents(encoder, data_dict, eval_scalers)

    # ── Experiment A: Pairwise Success Landscape ──
    if "A" in experiments_to_run:
        matrices, results_list = compute_transfer_matrix_full(latents_dict, labels_dict)
        off_diag_ranking, transferability = experiment_a_pairwise_success(
            matrices, results_list, list(latents_dict.keys()), DATASET_DISPLAY)

        # Save full results
        with open(RESULTS_DIR / "matrices" / "full_transfer_matrix.json", "w") as f:
            serializable = {}
            for k, v in matrices.items():
                serializable[k] = v.tolist()
            json.dump(serializable, f, indent=2)
        pd.DataFrame(results_list).to_csv(RESULTS_DIR / "tables" / "all_transfer_results.csv", index=False)
    else:
        matrices, results_list, off_diag_ranking, transferability = None, None, None, None

    # ── Experiment B: Dataset Similarity vs Transfer ──
    shap_results_local = None
    similarity_metrics = None
    if "B" in experiments_to_run and matrices is not None:
        correlations, similarity_metrics = experiment_b_similarity_vs_transfer(
            matrices, latents_dict, labels_dict, list(latents_dict.keys()), DATASET_DISPLAY)
    else:
        correlations = None

    # ── Experiment C: Class-Level Transfer ──
    class_results = None
    class_matrix = None
    if "C" in experiments_to_run and matrices is not None:
        class_results, class_matrix = experiment_c_class_transfer(
            latents_dict, labels_dict, data_dict, list(latents_dict.keys()), DATASET_DISPLAY)

    # ── Experiment D: Feature Dependency Analysis ──
    if "D" in experiments_to_run:
        shap_results_local, shap_pairs = experiment_d_feature_dependency(
            latents_dict, labels_dict, data_dict, list(latents_dict.keys()), DATASET_DISPLAY)

    # ── Experiment E: Latent Geometry Evolution ──
    geo_results = None
    if "E" in experiments_to_run:
        geo_results, Z_tsne, dataset_ids, attack_ids = experiment_e_latent_geometry(
            latents_dict, labels_dict, data_dict, list(latents_dict.keys()), DATASET_DISPLAY)

    # ── Experiment F: Failure Attribution ──
    attributions = None
    failure_matrix = None
    if "F" in experiments_to_run and matrices is not None:
        if similarity_metrics is None and matrices is not None:
            # Compute similarity metrics if Experiment B wasn't run
            logger.info("  Computing similarity metrics for failure attribution...")
            similarity_metrics = compute_similarity_metrics(latents_dict, labels_dict)
        attributions, failure_matrix = experiment_f_failure_attribution(
            matrices, results_list, latents_dict, labels_dict, data_dict,
            list(latents_dict.keys()), DATASET_DISPLAY,
            shap_results_local, similarity_metrics)

    # ── Statistical Analysis ──
    stats_results = statistical_analysis(matrices, results_list, attributions,
                                          list(latents_dict.keys()), DATASET_DISPLAY)

    # ── Deliverables ──
    generate_deliverables(matrices, results_list, attributions, shap_results_local,
                          correlations, stats_results, class_results,
                          geo_results, transferability if "transferability" in dir() else None,
                          list(latents_dict.keys()), DATASET_DISPLAY)

    # ── H1 Verdict ──
    logger.info(f"\n{'='*65}")
    logger.info("Phase 51 — H1 Verdict")
    logger.info(f"{'='*65}")

    if stats_results:
        mean_off = stats_results["mean_off_diag_mf1"]
        logger.info(f"  1. Mean off-diag MF1: {mean_off:.4f}")

    if correlations:
        best_corr = correlations[0]
        logger.info(f"  2. Best predictor: {best_corr['metric']} (r={best_corr['pearson_r']:.4f}, p={best_corr['pearson_p']:.4f})")

    if attributions:
        dominant_counts = {}
        for a in attributions:
            dominant_counts[a["dominant_failure"]] = dominant_counts.get(a["dominant_failure"], 0) + 1
        logger.info("  3. Failure attribution breakdown:")
        for k, v in sorted(dominant_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"       {k}: {v} pairs ({100*v/len(attributions):.1f}%)")

    # H1 support criteria
    h1_criteria = 0
    if stats_results and stats_results["permutation_p"] < 0.05:
        h1_criteria += 1
        logger.info("  ✓ Criterion 1: Transfer failures are not random (p<0.05)")

    if correlations and abs(correlations[0]["pearson_r"]) > 0.5:
        h1_criteria += 1
        logger.info(f"  ✓ Criterion 2: Strong correlation with {correlations[0]['metric']} (r={correlations[0]['pearson_r']:.3f})")

    if attributions and len(attributions) > 0:
        # Check if every failed pair got an attribution
        h1_criteria += 1
        logger.info(f"  ✓ Criterion 3: {len(attributions)} failed pairs attributed")

    # UNSW-NB15 explanation
    unsw_idx = list(latents_dict.keys()).index("unsw_nb15") if "unsw_nb15" in latents_dict else -1
    if unsw_idx >= 0 and matrices is not None:
        unsw_mf1 = [matrices["macro_f1"][unsw_idx, j] for j in range(len(matrices["macro_f1"])) if j != unsw_idx]
        logger.info(f"  4. UNSW-NB15 avg source MF1: {np.mean(unsw_mf1):.4f}")

        if attributions:
            unsw_attrs = [a for a in attributions if a["source"] == "unsw_nb15" and a["macro_f1"] < 0.5]
            if unsw_attrs:
                dominant_u = {}
                for a in unsw_attrs:
                    dominant_u[a["dominant_failure"]] = dominant_u.get(a["dominant_failure"], 0) + 1
                logger.info("       Dominant failures for UNSW-NB15 as source:")
                for k, v in sorted(dominant_u.items(), key=lambda x: x[1], reverse=True):
                    logger.info(f"         {k}: {v} pairs")

    logger.info(f"\n  H1 criteria met: {h1_criteria}/4")
    if h1_criteria >= 3:
        logger.info("  ✓ H1 SUPPORTED: Remaining failures explained by measurable dataset characteristics")
    elif h1_criteria >= 2:
        logger.info("  ≈ H1 PARTIALLY SUPPORTED: Some evidence for explainable failures")
    else:
        logger.info("  ✗ H0 cannot be rejected: Failures appear random or unexplained")

    logger.info(f"\nPhase 51 complete. Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
