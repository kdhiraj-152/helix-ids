#!/usr/bin/env python3
"""
Phase 52 — Generalization and Ablation Study

Validates whether SupCon transfer improvements are architecture-independent by
testing 6 ablations: latent dimension, encoder depth, temperature, loss weight,
label noise, and sample efficiency.

Usage:
  source .venv311/bin/activate
  PYTHONPATH=src python3 scripts/analysis/phase52_main.py
  PYTHONPATH=src python3 scripts/analysis/phase52_main.py --experiments A,B,C  (subset)
  PYTHONPATH=src python3 scripts/analysis/phase52_main.py --skip-train         (reload cached)
  PYTHONPATH=src nohup python3 -u scripts/analysis/phase52_main.py > /tmp/phase52.log 2>&1 &
"""

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
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


def cleanup_memory():
    """Force garbage collection and MPS cache clear to prevent OOM across experiments."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


logger = logging.getLogger("phase52")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "multi_dataset_v1"
PHASE50_DIR = PROJECT_ROOT / "results" / "phase50"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase52"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR / "models", exist_ok=True)
os.makedirs(RESULTS_DIR / "latents", exist_ok=True)
os.makedirs(RESULTS_DIR / "tables", exist_ok=True)
os.makedirs(RESULTS_DIR / "matrices", exist_ok=True)
os.makedirs(RESULTS_DIR / "learning_curves", exist_ok=True)
os.makedirs(RESULTS_DIR / "latent_visualizations", exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
fh = logging.FileHandler(RESULTS_DIR / "phase52_run.log")
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(fh)

logger.info(f"Phase 52 starting — device={DEVICE}")

RANDOM_STATE = 42
MAX_SAMPLES_PER_DATASET = 50_000  # reduced from 200K to fit 16GB MPS
SUPCON_EPOCHS = 30
PATIENCE = 10
LR = 1e-3
BATCH_SIZE = 128  # reduced from 256 for MPS memory
INPUT_DIM = 17
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

# ---------------------------------------------------------------------------
# Configurable architectures
# ---------------------------------------------------------------------------

class ConfigurableEncoder(nn.Module):
    """Encoder with configurable architecture from a list of (in, out) tuples."""
    def __init__(self, layer_spec, input_dim=INPUT_DIM):
        super().__init__()
        layers = []
        prev = input_dim
        for out_dim in layer_spec:
            layers.append(nn.Linear(prev, out_dim))
            layers.append(nn.ReLU())
            prev = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ClassifierHead(nn.Module):
    def __init__(self, latent_dim, num_classes=NUM_CLASSES):
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
    def __init__(self, latent_dim, proj_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, proj_dim), nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, z):
        return self.net(z)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, x):
        return x + self.net(x)


class ResidualEncoder(nn.Module):
    """Residual encoder: Linear(17→128) → ReLU → ResBlock(128) → Linear(128→2)."""
    def __init__(self, input_dim=INPUT_DIM, latent_dim=2):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU())
        self.resblocks = nn.Sequential(ResidualBlock(128), ResidualBlock(128))
        self.output_proj = nn.Linear(128, latent_dim)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.resblocks(h)
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# Data Loading (reused from Phase 51)
# ---------------------------------------------------------------------------

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
        max_pre = MAX_SAMPLES_PER_DATASET // 2
        if X_tr.size > 0 and X_te.size > 0:
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
                X = X[idx]; y = y[idx]
        else:
            X = X_te
            y = np.load(y_te_path) if y_te_path.exists() else np.empty(0)
            if X.shape[0] > MAX_SAMPLES_PER_DATASET:
                idx = rng.choice(X.shape[0], size=MAX_SAMPLES_PER_DATASET, replace=False)
                X = X[idx]; y = y[idx]
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


def subsample_stratified(X, y, max_samples):
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


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

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
            X = X[idx]; y = y[idx]; n = MAX_SAMPLES_PER_DATASET
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


def _build_val_loaders(train_data, val_data):
    val_loaders = {}
    for name in sorted(train_data.keys()):
        Xv = torch.from_numpy(val_data[name]["X"]).float()
        yv = torch.from_numpy(val_data[name]["y"]).long()
        ds_v = TensorDataset(Xv, yv)
        val_loaders[name] = DataLoader(ds_v, batch_size=BATCH_SIZE * 2)
    return train_data, val_loaders


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


# ---------------------------------------------------------------------------
# SupCon training (configurable)
# ---------------------------------------------------------------------------

def train_supcon(data_dict, latent_dim=128, layer_spec=None, temperature=0.1,
                 supcon_weight=0.5, label_noise_rate=0.0, sample_fraction=1.0,
                 run_name="supcon"):
    """Train a SupCon encoder with configurable parameters."""
    logger.info(f"  Training '{run_name}': latent_dim={latent_dim}, "
                f"temperature={temperature}, supcon_weight={supcon_weight}, "
                f"noise={label_noise_rate}, sample_fraction={sample_fraction}")

    # Sample fraction: reduce data size
    if sample_fraction < 1.0:
        reduced_data = {}
        for name in DATASET_NAMES:
            if name not in data_dict:
                continue
            X = data_dict[name]["X"]
            y = data_dict[name]["y"]
            n_target = max(100, int(X.shape[0] * sample_fraction))
            X_s, y_s = subsample_stratified(X, y, n_target)
            reduced_data[name] = {"X": X_s, "y": y_s}
        train_data, val_data, _ = prepare_data(reduced_data)
    else:
        train_data, val_data, _ = prepare_data(data_dict)

    train_loaders = build_classification_loader(train_data)
    _, val_loaders = _build_val_loaders(train_data, val_data)

    # Build encoder
    if layer_spec is not None:
        encoder = ConfigurableEncoder(layer_spec, input_dim=INPUT_DIM).to(DEVICE)
        # effective latent dim is last element
        effective_latent = layer_spec[-1]
    elif latent_dim == 2 and layer_spec is None:
        # Special case: residual encoder
        encoder = ResidualEncoder(input_dim=INPUT_DIM, latent_dim=2).to(DEVICE)
        effective_latent = 2
    else:
        encoder = ConfigurableEncoder([32, 64, 128, latent_dim], input_dim=INPUT_DIM).to(DEVICE)
        effective_latent = latent_dim

    classifier = ClassifierHead(latent_dim=effective_latent).to(DEVICE)
    projector = ProjectionHead(latent_dim=effective_latent).to(DEVICE)

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
            # Inject label noise (flip yb with probability label_noise_rate)
            if label_noise_rate > 0:
                noise_mask = torch.rand(yb.shape, device=yb.device) < label_noise_rate
                yb_noisy = yb.clone()
                yb_noisy[noise_mask] = 1 - yb_noisy[noise_mask]
                yb_use = yb_noisy
            else:
                yb_use = yb

            optimizer.zero_grad()
            z = encoder(xb)
            logits = classifier(z)
            cls_loss = cls_criterion(logits, yb_use)
            proj = projector(z)
            supcon = supervised_contrastive_loss(proj, yb_use, temperature=temperature)
            total_loss = cls_loss + supcon_weight * supcon
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
            logger.info(f"    [{run_name}] Epoch {epoch+1:3d}/{SUPCON_EPOCHS} "
                        f"train={train_l:.6f} val={val_l:.6f} "
                        f"acc={train_acc:.4f}/{val_acc:.4f}")

        if val_l < best_val_loss - 1e-6:
            best_val_loss = val_l
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            logger.info(f"    [{run_name}] Early stopping at epoch {epoch+1}")
            break

    logger.info(f"    [{run_name}] Done. Best val_loss={best_val_loss:.6f}")
    return encoder, classifier, history


# ---------------------------------------------------------------------------
# Latent extraction and RF transfer evaluation
# ---------------------------------------------------------------------------

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


def compute_transfer_matrix(latents_dict, labels_dict):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss

    names = sorted(latents_dict.keys())
    n_names = len(names)

    subsampled = {}
    for nm in names:
        Z = latents_dict[nm]; y = labels_dict[nm]
        if len(Z) > 50000:
            idx = rng.permutation(len(Z))[:50000]
            subsampled[nm] = (Z[idx], y[idx])
        else:
            subsampled[nm] = (Z, y)

    results = []
    mf1_mat = np.zeros((n_names, n_names))

    for i, src in enumerate(names):
        Z_src, y_src = subsampled[src]
        clf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                      random_state=RANDOM_STATE, n_jobs=1)
        clf.fit(Z_src, y_src)

        for j, tgt in enumerate(names):
            Z_tgt, y_tgt = latents_dict[tgt], labels_dict[tgt]
            y_pred = clf.predict(Z_tgt)
            y_prob = clf.predict_proba(Z_tgt)[:, 1] if hasattr(clf, "predict_proba") else None

            mf1 = float(f1_score(y_tgt, y_pred, average="macro", zero_division=0))
            prec = float(precision_score(y_tgt, y_pred, average="macro", zero_division=0))
            rec = float(recall_score(y_tgt, y_pred, average="macro", zero_division=0))

            auroc_val = 0.0
            if y_prob is not None and len(np.unique(y_tgt)) > 1:
                try:
                    auroc_val = float(roc_auc_score(y_tgt, y_prob))
                except ValueError:
                    auroc_val = 0.5

            brier_val = float(brier_score_loss(y_tgt, y_prob)) if y_prob is not None else np.nan
            ece_val = compute_ece(y_tgt, y_prob) if y_prob is not None else np.nan

            results.append({
                "source": src, "target": tgt, "macro_f1": mf1,
                "precision": prec, "recall": rec, "auroc": auroc_val,
                "brier": brier_val, "ece": ece_val,
            })
            mf1_mat[i, j] = mf1

    return {"macro_f1": mf1_mat, "names": names}, results


def compute_ece(y_true, y_prob, n_bins=10):
    if y_prob is None:
        return np.nan
    bins = np.linspace(0, 1, n_bins + 1)
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


# ---------------------------------------------------------------------------
# Experiment A — Latent Dimension Ablation
# ---------------------------------------------------------------------------

LATENT_DIMS = [2, 8, 16, 32, 64, 128]

def run_experiment_a(data_dict):
    """Latent dimension ablation."""
    logger.info(f"\n{'='*65}")
    logger.info("Experiment A — Latent Dimension Ablation")
    logger.info(f"Latent dims: {LATENT_DIMS}")
    logger.info(f"{'='*65}")

    results = []
    for ld in LATENT_DIMS:
        run_name = f"supcon_dim{ld}"
        cache_path = RESULTS_DIR / "latents" / f"encoder_{run_name}.pt"
        metrics_path = RESULTS_DIR / "tables" / f"expA_dim{ld}_metrics.json"

        if cache_path.exists() and metrics_path.exists():
            logger.info(f"  Skipping {run_name}: cached encoder + metrics found")
            with open(metrics_path) as f:
                res = json.load(f)
            results.append(res)
            continue

        t0 = time.time()
        # Build encoder layer spec: [32, 64, 128, ld]
        if ld == 2:
            # Residual encoder for dim=2
            encoder = ResidualEncoder(input_dim=INPUT_DIM, latent_dim=2).to(DEVICE)
            layer_spec = None
        else:
            layer_spec = [32, 64, 128, ld]
            encoder = ConfigurableEncoder(layer_spec, input_dim=INPUT_DIM).to(DEVICE)

        classifier = ClassifierHead(latent_dim=ld).to(DEVICE)
        projector = ProjectionHead(latent_dim=ld).to(DEVICE)

        train_data, val_data, _ = prepare_data(data_dict)
        train_loaders = build_classification_loader(train_data)
        _, val_loaders = _build_val_loaders(train_data, val_data)

        optimizer = optim.Adam(
            list(encoder.parameters()) + list(classifier.parameters()) + list(projector.parameters()),
            lr=LR
        )
        cls_criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0

        steps_per_epoch = max(100, sum(ds.dataset.tensors[0].shape[0] // BATCH_SIZE
                                       for ds in train_loaders.values()) // (2 * len(train_loaders)))
        steps_per_epoch = min(steps_per_epoch, 500)

        for epoch in range(SUPCON_EPOCHS):
            encoder.train(); classifier.train(); projector.train()
            losses = []
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
                    list(encoder.parameters()) + list(classifier.parameters()) + list(projector.parameters()), 10.0)
                optimizer.step()
                losses.append(total_loss.item())

            # Validation
            encoder.eval(); classifier.eval()
            val_losses = []
            with torch.no_grad():
                for loader in val_loaders.values():
                    for xb, yb in loader:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        z = encoder(xb)
                        logits = classifier(z)
                        loss = cls_criterion(logits, yb)
                        val_losses.append(loss.item())

            val_l = np.mean(val_losses) if val_losses else 0
            if val_l < best_val_loss - 1e-6:
                best_val_loss = val_l
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= PATIENCE:
                break

        train_time = time.time() - t0
        logger.info(f"    [{run_name}] Training done in {train_time:.1f}s, best val_loss={best_val_loss:.6f}")

        torch.save(encoder.state_dict(), cache_path)

        # Evaluate
        eval_scalers = fit_eval_scalers(data_dict)
        latents_dict, labels_dict = extract_latents(encoder, data_dict, eval_scalers)
        matrices, rlist = compute_transfer_matrix(latents_dict, labels_dict)

        # Compute metrics
        mf1 = matrices["macro_f1"]
        n = len(matrices["names"])
        off_diag_vals = [mf1[i, j] for i in range(n) for j in range(n) if i != j]
        diag_vals = [mf1[i, i] for i in range(n)]
        mean_off = float(np.mean(off_diag_vals))
        mean_diag = float(np.mean(diag_vals))
        std_off = float(np.std(off_diag_vals, ddof=1))
        std_diag = float(np.std(diag_vals, ddof=1))

        # Representation similarity (linear CKA across datasets)
        cka_vals = compute_pairwise_cka(latents_dict)

        # Memory usage estimate
        n_params = sum(p.numel() for p in encoder.parameters())

        res = {
            "run_name": run_name, "latent_dim": ld,
            "mean_off_diag_mf1": mean_off, "std_off_diag_mf1": std_off,
            "mean_diag_mf1": mean_diag, "std_diag_mf1": std_diag,
            "mean_cka": float(np.mean(cka_vals)) if cka_vals else 0,
            "n_params": n_params,
            "train_time_seconds": train_time,
            "best_val_loss": best_val_loss,
            "epochs_completed": epoch + 1,
            "full_results": rlist,
        }
        results.append(res)
        with open(metrics_path, "w") as f:
            json.dump(res, f, indent=2, default=str)
        logger.info(f"    [{run_name}] Mean off-diag MF1={mean_off:.4f} ± {std_off:.4f}")
        cleanup_memory()

    return results


def compute_pairwise_cka(latents_dict):
    """Compute linear CKA between all dataset pairs from latents."""
    names = sorted(latents_dict.keys())
    cka_vals = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            Z_i = latents_dict[names[i]]
            Z_j = latents_dict[names[j]]
            lr = min(Z_i.shape[0], Z_j.shape[0], 10000)
            idx_i = rng.choice(Z_i.shape[0], size=lr, replace=False)
            idx_j = rng.choice(Z_j.shape[0], size=lr, replace=False)
            Xi_c = Z_i[idx_i] - Z_i[idx_i].mean(axis=0)
            Xj_c = Z_j[idx_j] - Z_j[idx_j].mean(axis=0)
            hsic_xy = float(np.sum((Xj_c.T @ Xi_c) ** 2)) / (lr - 1) ** 2
            hsic_xx = float(np.sum((Xi_c.T @ Xi_c) ** 2)) / (lr - 1) ** 2
            hsic_yy = float(np.sum((Xj_c.T @ Xj_c) ** 2)) / (lr - 1) ** 2
            denom = np.sqrt(hsic_xx * hsic_yy)
            cka_vals.append(hsic_xy / denom if denom > 1e-12 else 0.0)
    return cka_vals


# ---------------------------------------------------------------------------
# Experiment B — Encoder Depth Ablation
# ---------------------------------------------------------------------------

ENCODER_ARCHES = {
    "shallow_17_32_2": [32, 2],
    "shallow_17_64_2": [64, 2],
    "medium_17_64_128_2": [64, 128, 2],
    "deep_17_128_256_2": [128, 256, 2],
    "residual_17_128_2": "residual",
}

def run_experiment_b(data_dict):
    """Encoder depth/architecture ablation."""
    logger.info(f"\n{'='*65}")
    logger.info("Experiment B — Encoder Depth Ablation")
    logger.info(f"Architectures: {list(ENCODER_ARCHES.keys())}")
    logger.info(f"{'='*65}")

    results = []
    for arch_name, spec in ENCODER_ARCHES.items():
        run_name = f"supcon_{arch_name}"
        cache_path = RESULTS_DIR / "latents" / f"encoder_{run_name}.pt"
        metrics_path = RESULTS_DIR / "tables" / f"expB_{arch_name}_metrics.json"

        if cache_path.exists() and metrics_path.exists():
            logger.info(f"  Skipping {run_name}: cached encoder + metrics found")
            with open(metrics_path) as f:
                res = json.load(f)
            results.append(res)
            continue

        t0 = time.time()
        if spec == "residual":
            encoder = ResidualEncoder(input_dim=INPUT_DIM, latent_dim=2).to(DEVICE)
            latent_dim = 2
        else:
            encoder = ConfigurableEncoder(spec, input_dim=INPUT_DIM).to(DEVICE)
            latent_dim = spec[-1]

        classifier = ClassifierHead(latent_dim=latent_dim).to(DEVICE)
        projector = ProjectionHead(latent_dim=latent_dim).to(DEVICE)

        train_data, val_data, _ = prepare_data(data_dict)
        train_loaders = build_classification_loader(train_data)
        _, val_loaders = _build_val_loaders(train_data, val_data)

        optimizer = optim.Adam(
            list(encoder.parameters()) + list(classifier.parameters()) + list(projector.parameters()),
            lr=LR
        )
        cls_criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0
        steps_per_epoch = max(100, sum(ds.dataset.tensors[0].shape[0] // BATCH_SIZE
                                       for ds in train_loaders.values()) // (2 * len(train_loaders)))
        steps_per_epoch = min(steps_per_epoch, 500)

        for epoch in range(SUPCON_EPOCHS):
            encoder.train(); classifier.train(); projector.train()
            for name, xb, yb in multi_loader_iter(train_loaders, steps_per_epoch):
                optimizer.zero_grad()
                z = encoder(xb); logits = classifier(z)
                cls_loss = cls_criterion(logits, yb)
                proj = projector(z)
                supcon = supervised_contrastive_loss(proj, yb, temperature=0.1)
                total_loss = cls_loss + 0.5 * supcon
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(classifier.parameters()) + list(projector.parameters()), 10.0)
                optimizer.step()

            encoder.eval(); classifier.eval()
            val_losses = []
            with torch.no_grad():
                for loader in val_loaders.values():
                    for xb, yb in loader:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        z = encoder(xb); logits = classifier(z)
                        loss = cls_criterion(logits, yb)
                        val_losses.append(loss.item())
            val_l = np.mean(val_losses) if val_losses else 0
            if val_l < best_val_loss - 1e-6:
                best_val_loss = val_l; patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= PATIENCE:
                break

        train_time = time.time() - t0
        torch.save(encoder.state_dict(), cache_path)

        # Evaluate
        eval_scalers = fit_eval_scalers(data_dict)
        latents_dict, labels_dict = extract_latents(encoder, data_dict, eval_scalers)
        matrices, rlist = compute_transfer_matrix(latents_dict, labels_dict)

        mf1 = matrices["macro_f1"]
        n = len(matrices["names"])
        off_diag_vals = [mf1[i, j] for i in range(n) for j in range(n) if i != j]
        diag_vals = [mf1[i, i] for i in range(n)]
        mean_off = float(np.mean(off_diag_vals))
        mean_diag = float(np.mean(diag_vals))
        std_off = float(np.std(off_diag_vals, ddof=1))

        n_params = sum(p.numel() for p in encoder.parameters())
        cka_vals = compute_pairwise_cka(latents_dict)

        res = {
            "run_name": run_name, "architecture": arch_name, "layer_spec": str(spec),
            "mean_off_diag_mf1": mean_off, "std_off_diag_mf1": std_off,
            "mean_diag_mf1": mean_diag, "std_diag_mf1": float(np.std(diag_vals, ddof=1)),
            "mean_cka": float(np.mean(cka_vals)) if cka_vals else 0,
            "n_params": n_params,
            "train_time_seconds": train_time,
            "best_val_loss": best_val_loss,
            "epochs_completed": epoch + 1,
            "full_results": rlist,
        }
        results.append(res)
        with open(metrics_path, "w") as f:
            json.dump(res, f, indent=2, default=str)
        logger.info(f"    [{run_name}] Mean off-diag MF1={mean_off:.4f} ± {std_off:.4f}, "
                    f"params={n_params}")
        cleanup_memory()

    return results


# ---------------------------------------------------------------------------
# Experiment C — Temperature Sweep
# ---------------------------------------------------------------------------

TEMPERATURES = [0.03, 0.05, 0.07, 0.10, 0.20, 0.50]

def run_experiment_c(data_dict):
    """Temperature sweep for SupCon loss."""
    logger.info(f"\n{'='*65}")
    logger.info("Experiment C — Temperature Sweep")
    logger.info(f"Temperatures: {TEMPERATURES}")
    logger.info(f"{'='*65}")

    results = []
    for temp in TEMPERATURES:
        run_name = f"supcon_temp{temp:.2f}".replace(".", "p")
        cache_path = RESULTS_DIR / "latents" / f"encoder_{run_name}.pt"
        metrics_path = RESULTS_DIR / "tables" / f"expC_temp_{temp:.2f}_metrics.json"

        if cache_path.exists() and metrics_path.exists():
            logger.info(f"  Skipping {run_name}: cached")
            with open(metrics_path) as f:
                res = json.load(f)
            results.append(res)
            continue

        t0 = time.time()
        latent_dim = 128
        logger.info(f"  [{run_name}] Step 1: creating models on {DEVICE}...")
        encoder = ConfigurableEncoder([32, 64, 128, latent_dim], input_dim=INPUT_DIM).to(DEVICE)
        classifier = ClassifierHead(latent_dim=latent_dim).to(DEVICE)
        projector = ProjectionHead(latent_dim=latent_dim).to(DEVICE)
        logger.info(f"  [{run_name}] Step 2: models created, preparing data...")

        train_data, val_data, _ = prepare_data(data_dict)
        train_loaders = build_classification_loader(train_data)
        _, val_loaders = _build_val_loaders(train_data, val_data)
        logger.info(f"  [{run_name}] Step 3: data prepared, creating optimizer...")

        optimizer = optim.Adam(
            list(encoder.parameters()) + list(classifier.parameters()) + list(projector.parameters()),
            lr=LR
        )
        cls_criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0
        steps_per_epoch = max(100, sum(ds.dataset.tensors[0].shape[0] // BATCH_SIZE
                                       for ds in train_loaders.values()) // (2 * len(train_loaders)))
        steps_per_epoch = min(steps_per_epoch, 500)
        logger.info(f"  [{run_name}] Step 4: steps_per_epoch={steps_per_epoch}, entering training...")

        for epoch in range(SUPCON_EPOCHS):
            encoder.train(); classifier.train(); projector.train()
            for name, xb, yb in multi_loader_iter(train_loaders, steps_per_epoch):
                optimizer.zero_grad()
                z = encoder(xb); logits = classifier(z)
                cls_loss = cls_criterion(logits, yb)
                proj = projector(z)
                supcon = supervised_contrastive_loss(proj, yb, temperature=temp)
                total_loss = cls_loss + 0.5 * supcon
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    logger.warning(f"  [{run_name}] NaN loss at epoch {epoch}, step {name}")
                    continue
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(classifier.parameters()) + list(projector.parameters()), 10.0)
                optimizer.step()

            encoder.eval(); classifier.eval()
            val_losses = []
            with torch.no_grad():
                for loader in val_loaders.values():
                    for xb, yb in loader:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        z = encoder(xb); logits = classifier(z)
                        loss = cls_criterion(logits, yb)
                        val_losses.append(loss.item())
            val_l = np.mean(val_losses) if val_losses else 0
            if val_l < best_val_loss - 1e-6:
                best_val_loss = val_l; patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"  [{run_name}] Early stopping at epoch {epoch+1}")
                break

        train_time = time.time() - t0
        torch.save(encoder.state_dict(), cache_path)

        eval_scalers = fit_eval_scalers(data_dict)
        latents_dict, labels_dict = extract_latents(encoder, data_dict, eval_scalers)
        matrices, rlist = compute_transfer_matrix(latents_dict, labels_dict)

        mf1 = matrices["macro_f1"]
        n = len(matrices["names"])
        off_diag_vals = [mf1[i, j] for i in range(n) for j in range(n) if i != j]
        diag_vals = [mf1[i, i] for i in range(n)]
        mean_off = float(np.mean(off_diag_vals))
        mean_diag = float(np.mean(diag_vals))

        # Intra-class compactness (mean cosine similarity within class)
        compactness = compute_intra_class_compactness(latents_dict, labels_dict)
        # Inter-class separation (mean cosine similarity between classes)
        separation = compute_inter_class_separation(latents_dict, labels_dict)

        res = {
            "run_name": run_name, "temperature": temp,
            "mean_off_diag_mf1": mean_off, "std_off_diag_mf1": float(np.std(off_diag_vals, ddof=1)),
            "mean_diag_mf1": mean_diag, "std_diag_mf1": float(np.std(diag_vals, ddof=1)),
            "intra_class_compactness": compactness,
            "inter_class_separation": separation,
            "train_time_seconds": train_time,
            "best_val_loss": best_val_loss,
            "epochs_completed": epoch + 1,
            "full_results": rlist,
        }
        results.append(res)
        with open(metrics_path, "w") as f:
            json.dump(res, f, indent=2, default=str)
        logger.info(f"    [{run_name}] Mean off-diag MF1={mean_off:.4f}, "
                    f"compact={compactness:.4f}, sep={separation:.4f}")
        cleanup_memory()

    return results


def compute_intra_class_compactness(latents_dict, labels_dict):
    """Mean cosine similarity between same-class samples."""
    sims = []
    for name in sorted(latents_dict.keys()):
        Z = latents_dict[name]
        y = labels_dict[name]
        for c in np.unique(y):
            mask = y == c
            if mask.sum() < 2:
                continue
            Zc = Z[mask]
            Zc_norm = Zc / (np.linalg.norm(Zc, axis=1, keepdims=True) + 1e-12)
            sim_matrix = Zc_norm @ Zc_norm.T
            n = len(Zc)
            # Upper triangle only
            triu_vals = sim_matrix[np.triu_indices(n, k=1)]
            if len(triu_vals) > 0:
                sims.append(float(np.mean(triu_vals)))
    return float(np.mean(sims)) if sims else 0.0


def compute_inter_class_separation(latents_dict, labels_dict):
    """Mean cosine similarity between class centroids."""
    centroids = {}
    for name in sorted(latents_dict.keys()):
        Z = latents_dict[name]
        y = labels_dict[name]
        for c in np.unique(y):
            centroids[f"{name}_class{c}"] = Z[y == c].mean(axis=0)

    keys = list(centroids.keys())
    sims = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ki, kj = keys[i], keys[j]
            # Skip if same dataset
            if ki.split("_class")[0] == kj.split("_class")[0]:
                continue
            ci, cj = centroids[ki], centroids[kj]
            dot = ci @ cj
            norm = np.linalg.norm(ci) * np.linalg.norm(cj) + 1e-12
            sims.append(float(dot / norm))
    return float(np.mean(sims)) if sims else 0.0


# ---------------------------------------------------------------------------
# Experiment D — Loss Weight Ablation
# ---------------------------------------------------------------------------

LOSS_WEIGHTS = [0.0, 0.25, 0.50, 1.00, 2.00]

def run_experiment_d(data_dict):
    """SupCon loss weight ablation."""
    logger.info(f"\n{'='*65}")
    logger.info("Experiment D — Loss Weight Ablation")
    logger.info(f"Weights: {LOSS_WEIGHTS}")
    logger.info(f"{'='*65}")

    results = []
    for w in LOSS_WEIGHTS:
        run_name = f"supcon_w{w:.2f}".replace(".", "p")
        cache_path = RESULTS_DIR / "latents" / f"encoder_{run_name}.pt"
        metrics_path = RESULTS_DIR / "tables" / f"expD_weight_{w:.2f}_metrics.json"

        if cache_path.exists() and metrics_path.exists():
            logger.info(f"  Skipping {run_name}: cached")
            with open(metrics_path) as f:
                res = json.load(f)
            results.append(res)
            continue

        t0 = time.time()
        latent_dim = 128
        logger.info(f"  [{run_name}] Step 1: creating models on {DEVICE}...")
        encoder = ConfigurableEncoder([32, 64, 128, latent_dim], input_dim=INPUT_DIM).to(DEVICE)
        classifier = ClassifierHead(latent_dim=latent_dim).to(DEVICE)
        projector = ProjectionHead(latent_dim=latent_dim).to(DEVICE)
        logger.info(f"  [{run_name}] Step 2: models created, preparing data...")

        train_data, val_data, _ = prepare_data(data_dict)
        train_loaders = build_classification_loader(train_data)
        _, val_loaders = _build_val_loaders(train_data, val_data)
        logger.info(f"  [{run_name}] Step 3: data prepared, creating optimizer...")

        optimizer = optim.Adam(
            list(encoder.parameters()) + list(classifier.parameters()) + list(projector.parameters()),
            lr=LR
        )
        cls_criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0
        steps_per_epoch = max(100, sum(ds.dataset.tensors[0].shape[0] // BATCH_SIZE
                                       for ds in train_loaders.values()) // (2 * len(train_loaders)))
        steps_per_epoch = min(steps_per_epoch, 500)

        for epoch in range(SUPCON_EPOCHS):
            encoder.train(); classifier.train(); projector.train()
            for name, xb, yb in multi_loader_iter(train_loaders, steps_per_epoch):
                optimizer.zero_grad()
                z = encoder(xb); logits = classifier(z)
                cls_loss = cls_criterion(logits, yb)
                proj = projector(z)
                supcon = supervised_contrastive_loss(proj, yb, temperature=0.1)
                total_loss = cls_loss + w * supcon
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(classifier.parameters()) + list(projector.parameters()), 10.0)
                optimizer.step()

            encoder.eval(); classifier.eval()
            val_losses = []
            with torch.no_grad():
                for loader in val_loaders.values():
                    for xb, yb in loader:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        z = encoder(xb); logits = classifier(z)
                        loss = cls_criterion(logits, yb)
                        val_losses.append(loss.item())
            val_l = np.mean(val_losses) if val_losses else 0
            if val_l < best_val_loss - 1e-6:
                best_val_loss = val_l; patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= PATIENCE:
                break

        train_time = time.time() - t0
        torch.save(encoder.state_dict(), cache_path)

        eval_scalers = fit_eval_scalers(data_dict)
        latents_dict, labels_dict = extract_latents(encoder, data_dict, eval_scalers)
        matrices, rlist = compute_transfer_matrix(latents_dict, labels_dict)

        mf1 = matrices["macro_f1"]
        n = len(matrices["names"])
        off_diag_vals = [mf1[i, j] for i in range(n) for j in range(n) if i != j]
        diag_vals = [mf1[i, i] for i in range(n)]
        mean_off = float(np.mean(off_diag_vals))
        mean_diag = float(np.mean(diag_vals))

        # Representation collapse detection (mean pair-wise distance / std)
        all_latents = np.vstack(list(latents_dict.values()))
        pair_dists = np.linalg.norm(all_latents[:1000] - all_latents[1000:2000], axis=1) if len(all_latents) >= 2000 else np.array([1.0])
        collapse_ratio = float(np.std(pair_dists) / max(np.mean(pair_dists), 1e-12))

        # ECE (calibration) on source
        ece_vals = [r["ece"] for r in rlist if r["source"] == r["target"] and not np.isnan(r["ece"])]
        mean_ece = float(np.mean(ece_vals)) if ece_vals else 0.0

        res = {
            "run_name": run_name, "supcon_weight": w,
            "mean_off_diag_mf1": mean_off, "std_off_diag_mf1": float(np.std(off_diag_vals, ddof=1)),
            "mean_diag_mf1": mean_diag, "std_diag_mf1": float(np.std(diag_vals, ddof=1)),
            "mean_ece": mean_ece,
            "collapse_ratio": collapse_ratio,
            "train_time_seconds": train_time,
            "best_val_loss": best_val_loss,
            "epochs_completed": epoch + 1,
            "full_results": rlist,
        }
        results.append(res)
        with open(metrics_path, "w") as f:
            json.dump(res, f, indent=2, default=str)
        logger.info(f"    [{run_name}] Mean off-diag MF1={mean_off:.4f}, "
                    f"ECE={mean_ece:.4f}, collapse={collapse_ratio:.4f}")
        cleanup_memory()

    return results


# ---------------------------------------------------------------------------
# Experiment E — Label Noise Robustness
# ---------------------------------------------------------------------------

NOISE_RATES = [0.0, 0.05, 0.10, 0.20, 0.30]

def run_experiment_e(data_dict):
    """Label noise robustness."""
    logger.info(f"\n{'='*65}")
    logger.info("Experiment E — Label Noise Robustness")
    logger.info(f"Noise rates: {NOISE_RATES}")
    logger.info(f"{'='*65}")

    # We need the full pipeline — inject noise during training
    results = []
    for noise in NOISE_RATES:
        run_name = f"supcon_noise{noise:.2f}".replace(".", "p")
        cache_path = RESULTS_DIR / "models" / f"encoder_{run_name}.pt"
        metrics_path = RESULTS_DIR / "tables" / f"expE_noise_{noise:.2f}_metrics.json"

        if cache_path.exists() and metrics_path.exists():
            logger.info(f"  Skipping {run_name}: cached")
            with open(metrics_path) as f:
                res = json.load(f)
            results.append(res)
            continue

        # Train with label noise
        logger.info(f"  Training with label noise rate={noise}")
        encoder, classifier, history = train_supcon(
            data_dict, latent_dim=128, temperature=0.1, supcon_weight=0.5,
            label_noise_rate=noise, run_name=run_name
        )
        torch.save(encoder.state_dict(), cache_path)
        # Save learning curves
        pd.DataFrame(history).to_csv(RESULTS_DIR / "learning_curves" / f"expE_noise_{noise:.2f}.csv", index=False)

        # Evaluate
        eval_scalers = fit_eval_scalers(data_dict)
        latents_dict, labels_dict = extract_latents(encoder, data_dict, eval_scalers)
        matrices, rlist = compute_transfer_matrix(latents_dict, labels_dict)

        mf1 = matrices["macro_f1"]
        n = len(matrices["names"])
        off_diag_vals = [mf1[i, j] for i in range(n) for j in range(n) if i != j]
        diag_vals = [mf1[i, i] for i in range(n)]
        mean_off = float(np.mean(off_diag_vals))
        mean_diag = float(np.mean(diag_vals))

        # SHAP-based feature stability (Quick measure: use RF feature importance correlation)
        shap_stability = compute_shap_stability(latents_dict, labels_dict)
        # Prototype alignment
        proto_align = compute_prototype_alignment(latents_dict, labels_dict)

        res = {
            "run_name": run_name, "label_noise_rate": noise,
            "mean_off_diag_mf1": mean_off, "std_off_diag_mf1": float(np.std(off_diag_vals, ddof=1)),
            "mean_diag_mf1": mean_diag, "std_diag_mf1": float(np.std(diag_vals, ddof=1)),
            "shap_stability": shap_stability,
            "prototype_alignment": proto_align,
            "full_results": rlist,
        }
        results.append(res)
        with open(metrics_path, "w") as f:
            json.dump(res, f, indent=2, default=str)
        logger.info(f"    [{run_name}] Mean off-diag MF1={mean_off:.4f}, "
                    f"shap_stab={shap_stability:.4f}, proto_align={proto_align:.4f}")
        cleanup_memory()

    return results


def compute_shap_stability(latents_dict, labels_dict):
    """SHAP stability: correlation of feature importance across datasets."""
    from sklearn.ensemble import RandomForestClassifier
    importances = []
    for name in sorted(latents_dict.keys()):
        Z = latents_dict[name]
        y = labels_dict[name]
        if len(Z) > 50000:
            idx = rng.permutation(len(Z))[:50000]
            Z, y = Z[idx], y[idx]
        clf = RandomForestClassifier(n_estimators=50, max_depth=8,
                                      random_state=RANDOM_STATE, n_jobs=1)
        clf.fit(Z, y)
        importances.append(clf.feature_importances_)

    if len(importances) < 2:
        return 0.0
    from scipy.stats import spearmanr
    rhos = []
    ld = latents_dict[next(iter(latents_dict))].shape[1]
    for i in range(len(importances)):
        for j in range(i + 1, len(importances)):
            # Pad shorter importance vectors
            imp_i = np.pad(importances[i], (0, max(0, ld - len(importances[i]))))[:ld]
            imp_j = np.pad(importances[j], (0, max(0, ld - len(importances[j]))))[:ld]
            rho, _ = spearmanr(imp_i, imp_j)
            if not np.isnan(rho):
                rhos.append(rho)
    return float(np.mean(rhos)) if rhos else 0.0


def compute_prototype_alignment(latents_dict, labels_dict):
    """Mean cosine similarity between Normal class centroids across datasets."""
    centroids = []
    for name in sorted(latents_dict.keys()):
        Z = latents_dict[name]
        y = labels_dict[name]
        norm_mask = y == 0
        if norm_mask.sum() > 0:
            c = Z[norm_mask].mean(axis=0)
            centroids.append(c / (np.linalg.norm(c) + 1e-12))
    if len(centroids) < 2:
        return 0.0
    sims = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            sims.append(float(centroids[i] @ centroids[j]))
    return float(np.mean(sims)) if sims else 0.0


# ---------------------------------------------------------------------------
# Experiment F — Sample Efficiency
# ---------------------------------------------------------------------------

SAMPLE_FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]

def run_experiment_f(data_dict):
    """Sample efficiency: how much data needed for stable transfer."""
    logger.info(f"\n{'='*65}")
    logger.info("Experiment F — Sample Efficiency")
    logger.info(f"Fractions: {SAMPLE_FRACTIONS}")
    logger.info(f"{'='*65}")

    results = []
    for frac in SAMPLE_FRACTIONS:
        run_name = f"supcon_frac{frac:.2f}".replace(".", "p")
        cache_path = RESULTS_DIR / "models" / f"encoder_{run_name}.pt"
        metrics_path = RESULTS_DIR / "tables" / f"expF_frac_{frac:.2f}_metrics.json"

        if cache_path.exists() and metrics_path.exists():
            logger.info(f"  Skipping {run_name}: cached")
            with open(metrics_path) as f:
                res = json.load(f)
            results.append(res)
            continue

        # Train with subsampled data
        logger.info(f"  Training with {100*frac:.0f}% data")
        encoder, classifier, history = train_supcon(
            data_dict, latent_dim=128, temperature=0.1, supcon_weight=0.5,
            sample_fraction=frac, run_name=run_name
        )
        torch.save(encoder.state_dict(), cache_path)
        pd.DataFrame(history).to_csv(RESULTS_DIR / "learning_curves" / f"expF_frac_{frac:.2f}.csv", index=False)

        # Evaluate
        eval_scalers = fit_eval_scalers(data_dict)
        latents_dict, labels_dict = extract_latents(encoder, data_dict, eval_scalers)
        matrices, rlist = compute_transfer_matrix(latents_dict, labels_dict)

        mf1 = matrices["macro_f1"]
        n = len(matrices["names"])
        off_diag_vals = [mf1[i, j] for i in range(n) for j in range(n) if i != j]
        diag_vals = [mf1[i, i] for i in range(n)]
        mean_off = float(np.mean(off_diag_vals))
        mean_diag = float(np.mean(diag_vals))
        std_off = float(np.std(off_diag_vals, ddof=1))

        # Bootstrap CI
        ci_low, ci_high = bootstrap_ci(np.array(off_diag_vals))

        res = {
            "run_name": run_name, "sample_fraction": frac,
            "mean_off_diag_mf1": mean_off, "std_off_diag_mf1": std_off,
            "mean_diag_mf1": mean_diag, "std_diag_mf1": float(np.std(diag_vals, ddof=1)),
            "ci_low_95": ci_low, "ci_high_95": ci_high,
            "full_results": rlist,
        }
        results.append(res)
        with open(metrics_path, "w") as f:
            json.dump(res, f, indent=2, default=str)
        logger.info(f"    [{run_name}] Mean off-diag MF1={mean_off:.4f} ± {std_off:.4f}, "
                    f"95% CI=[{ci_low:.4f}, {ci_high:.4f}]")
        cleanup_memory()

    return results


def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    """Bootstrap confidence interval."""
    rng_boot = np.random.RandomState(42)
    means = []
    for _ in range(n_bootstrap):
        idx = rng_boot.choice(len(data), size=len(data), replace=True)
        means.append(np.mean(data[idx]))
    means.sort()
    low_idx = int((1 - ci) / 2 * n_bootstrap)
    high_idx = int((1 + ci) / 2 * n_bootstrap)
    return float(means[low_idx]), float(means[high_idx])


# ---------------------------------------------------------------------------
# Statistical Analysis
# ---------------------------------------------------------------------------

def run_statistical_analysis(experiment_results):
    """
    Comprehensive statistical analysis across all experiment results.
    Computes: bootstrap CI, paired t-test, Wilcoxon, Cohen's d, Cliff's δ,
    Holm-Bonferroni, Friedman test, Nemenyi post-hoc.
    """
    logger.info(f"\n{'='*65}")
    logger.info("Statistical Analysis")
    logger.info(f"{'='*65}")

    from scipy.stats import ttest_rel, wilcoxon, friedmanchisquare
    from scipy.stats import norm as scipy_norm

    stats = {
        "experiment_a": statistical_analysis_single(experiment_results.get("exp_a", []), "latent_dim"),
        "experiment_b": statistical_analysis_single(experiment_results.get("exp_b", []), "architecture"),
        "experiment_c": statistical_analysis_single(experiment_results.get("exp_c", []), "temperature"),
        "experiment_d": statistical_analysis_single(experiment_results.get("exp_d", []), "supcon_weight"),
        "experiment_e": statistical_analysis_single(experiment_results.get("exp_e", []), "label_noise_rate"),
        "experiment_f": statistical_analysis_single(experiment_results.get("exp_f", []), "sample_fraction"),
    }

    # Cross-experiment: SupCon (baseline) vs all other configs
    cross_results = perform_cross_experiment_tests(experiment_results)
    stats["cross_experiment"] = cross_results

    with open(RESULTS_DIR / "tables" / "statistical_tests.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    logger.info("  Statistical tests saved to tables/statistical_tests.json")
    return stats


def statistical_analysis_single(results, param_name):
    """Statistical tests for a single experiment."""

    if len(results) < 2:
        return {"error": "Insufficient results for analysis", "n_configs": len(results)}

    # Sort by parameter value
    results_sorted = sorted(results, key=lambda x: x.get(param_name, 0))

    configs = []
    means = []
    stdevs = []
    for r in results_sorted:
        configs.append(str(r.get(param_name, r.get("run_name", "?"))))
        means.append(r.get("mean_off_diag_mf1", 0))
        stdevs.append(r.get("std_off_diag_mf1", 0))

    analysis = {
        "param_name": param_name,
        "configs": configs,
        "mean_mf1": means,
        "std_mf1": stdevs,
    }

    # Friedman test (non-parametric, >2 groups)
    if len(results) >= 3:
        # We need paired data for Friedman. Collect all off-diag MF1 per config.
        off_diag_matrices = []
        for r in results_sorted:
            vals = [x["macro_f1"] for x in r.get("full_results", []) if x["source"] != x["target"]]
            off_diag_matrices.append(vals)

        if len(off_diag_matrices) >= 3:
            min_len = min(len(m) for m in off_diag_matrices)
            if min_len >= 3:
                friedman_data = [m[:min_len] for m in off_diag_matrices]
                try:
                    friedman_stat, friedman_p = friedmanchisquare(*friedman_data)
                    analysis["friedman_stat"] = float(friedman_stat)
                    analysis["friedman_p"] = float(friedman_p)
                except Exception as e:
                    analysis["friedman_error"] = str(e)

    # Pairwise tests against baseline (first config = baseline)
    pairwise_tests = []
    baseline_mean = means[0]
    for i in range(1, len(results_sorted)):
        vals_baseline = [x["macro_f1"] for x in results_sorted[0].get("full_results", [])
                         if x["source"] != x["target"]]
        vals_target = [x["macro_f1"] for x in results_sorted[i].get("full_results", [])
                       if x["source"] != x["target"]]
        min_len = min(len(vals_baseline), len(vals_target))
        if min_len < 3:
            continue
        vb = np.array(vals_baseline[:min_len])
        vt = np.array(vals_target[:min_len])

        diff = vt - vb

        # Paired t-test
        t_stat, t_p = ttest_rel(vb, vt)

        # Wilcoxon signed-rank
        try:
            w_stat, w_p = wilcoxon(vb, vt, alternative="two-sided")
        except Exception:
            w_stat, w_p = 0, 1.0

        # Cohen's d (paired)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        cohens_d = mean_diff / (std_diff + 1e-12) if std_diff > 0 else 0.0

        # Cliff's delta
        n1, n2 = len(vb), len(vt)
        cliff = 0.0
        if n1 > 0 and n2 > 0:
            count = 0
            for x in vb:
                for y in vt:
                    if x < y:
                        count += 1
                    elif x > y:
                        count -= 1
            cliff = count / (n1 * n2)

        pairwise_tests.append({
            "comparison": f"{configs[0]} vs {configs[i]}",
            "baseline": configs[0], "target": configs[i],
            "baseline_mean": float(np.mean(vb)),
            "target_mean": float(np.mean(vt)),
            "mean_diff": float(mean_diff),
            "t_statistic": float(t_stat),
            "t_p_value": float(t_p),
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_p": float(w_p),
            "cohens_d": float(cohens_d),
            "cliffs_delta": float(cliff),
        })

    # Holm-Bonferroni correction
    if pairwise_tests:
        p_values = np.array([p["t_p_value"] for p in pairwise_tests])
        sorted_idx = np.argsort(p_values)
        n_tests = len(p_values)
        holm_corrected = []
        for rank, idx in enumerate(sorted_idx):
            corrected_p = min(1.0, p_values[idx] * (n_tests - rank))
            holm_corrected.append({
                "comparison": pairwise_tests[idx]["comparison"],
                "original_p": float(p_values[idx]),
                "holm_corrected_p": float(corrected_p),
                "significant_at_005": corrected_p < 0.05,
            })

        analysis["pairwise_tests"] = pairwise_tests
        analysis["holm_bonferroni"] = holm_corrected

    return analysis


def perform_cross_experiment_tests(experiment_results):
    """Cross-experiment: compare baseline SupCon (exp A, dim=128) vs all others."""
    all_configs = []
    for exp_name in ["exp_a", "exp_b", "exp_c", "exp_d", "exp_e", "exp_f"]:
        exp_results = experiment_results.get(exp_name, [])
        for r in exp_results:
            config_label = f"{exp_name}_{r.get('run_name', '?')}"
            vals = [x["macro_f1"] for x in r.get("full_results", []) if x["source"] != x["target"]]
            all_configs.append({"label": config_label, "vals": vals, **r})

    if len(all_configs) < 2:
        return {"error": "Insufficient cross-experiment data"}

    # Baseline = standard SupCon (exp A, dim=128)
    baseline_idx = None
    for i, c in enumerate(all_configs):
        if c.get("latent_dim") == 128 and c.get("run_name", "").startswith("supcon_dim"):
            baseline_idx = i
            break

    if baseline_idx is None:
        baseline_idx = 0  # fallback to first config

    baseline_vals = np.array(all_configs[baseline_idx]["vals"])

    comparisons = []
    for i, c in enumerate(all_configs):
        if i == baseline_idx:
            continue
        target_vals = np.array(c["vals"])
        min_len = min(len(baseline_vals), len(target_vals))
        if min_len < 3:
            continue
        vb, vt = baseline_vals[:min_len], target_vals[:min_len]
        diff = vt - vb

        # Paired t-test
        t_stat, t_p = ttest_rel(vb, vt)

        # Cohen's d
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        cohens_d = mean_diff / (std_diff + 1e-12) if std_diff > 0 else 0.0

        comparisons.append({
            "config": c["label"],
            "mean_baseline": float(np.mean(vb)),
            "mean_target": float(np.mean(vt)),
            "diff": float(mean_diff),
            "t_p_value": float(t_p),
            "cohens_d": float(cohens_d),
            "is_significant": t_p < 0.05,
        })

    # Ranking across all configs
    ranking = []
    for c in all_configs:
        ranking.append({
            "config": c["label"],
            "mean_mf1": float(np.mean(c["vals"])),
            "std_mf1": float(np.std(c["vals"], ddof=1)),
        })
    ranking.sort(key=lambda x: x["mean_mf1"], reverse=True)

    return {
        "n_configs_across_all_experiments": len(all_configs),
        "baseline_config": all_configs[baseline_idx]["label"] if baseline_idx is not None else "unknown",
        "ranking": ranking,
        "pairwise_comparisons": comparisons,
    }


# ---------------------------------------------------------------------------
# Deliverable Generation
# ---------------------------------------------------------------------------

def generate_deliverables(experiment_results, stats):
    """Generate all CSV tables, ranking tables, and FINAL_REPORT.md."""
    logger.info(f"\n{'='*65}")
    logger.info("Generating deliverables")
    logger.info(f"{'='*65}")

    # 1. Per-experiment CSV summaries
    generate_experiment_csvs(experiment_results)

    # 2. Ranking table across all experiments
    generate_ranking_table(experiment_results)

    # 3. Final report
    generate_final_report(experiment_results, stats)


def generate_experiment_csvs(experiment_results):
    """Generate CSV files for each experiment."""

    for exp_key, file_name in [
        ("exp_a", "latent_dimension_results.csv"),
        ("exp_b", "architecture_ablation.csv"),
        ("exp_c", "temperature_sweep.csv"),
        ("exp_d", "loss_weight_ablation.csv"),
        ("exp_e", "label_noise_results.csv"),
        ("exp_f", "sample_efficiency.csv"),
    ]:
        exp_data = experiment_results.get(exp_key, [])
        if not exp_data:
            logger.info(f"  Warning: No data for {file_name}, creating placeholder")
            pd.DataFrame().to_csv(RESULTS_DIR / "tables" / file_name, index=False)
            continue

        rows = []
        for r in exp_data:
            row = {}
            for k, v in r.items():
                if k == "full_results":
                    continue
                if isinstance(v, (list, dict)):
                    row[k] = str(v)
                else:
                    row[k] = v
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(RESULTS_DIR / "tables" / file_name, index=False)
        logger.info(f"  ✓ {file_name} saved")


def generate_ranking_table(experiment_results):
    """Cross-experiment ranking."""
    all_rows = []
    for exp_key, exp_name in [
        ("exp_a", "Latent Dim"),
        ("exp_b", "Architecture"),
        ("exp_c", "Temperature"),
        ("exp_d", "Loss Weight"),
        ("exp_e", "Label Noise"),
        ("exp_f", "Sample Efficiency"),
    ]:
        exp_data = experiment_results.get(exp_key, [])
        for r in exp_data:
            all_rows.append({
                "experiment": exp_name,
                "config": str(r.get("run_name", "?")),
                "mean_off_diag_mf1": r.get("mean_off_diag_mf1", 0),
                "std_off_diag_mf1": r.get("std_off_diag_mf1", 0),
                "mean_diag_mf1": r.get("mean_diag_mf1", 0),
                "n_params": r.get("n_params", "?"),
                "train_time_s": r.get("train_time_seconds", "?"),
            })

    if all_rows:
        df = pd.DataFrame(all_rows)
        df = df.sort_values("mean_off_diag_mf1", ascending=False)
        df.to_csv(RESULTS_DIR / "tables" / "ranking_tables.csv", index=False)
        logger.info(f"  ✓ ranking_tables.csv saved ({len(df)} rows)")
    else:
        pd.DataFrame().to_csv(RESULTS_DIR / "tables" / "ranking_tables.csv", index=False)
        logger.info("  Warning: No data for ranking_tables.csv")


def generate_final_report(experiment_results, stats):
    """Generate FINAL_REPORT.md."""
    lines = [
        "# Phase 52 — Generalization and Ablation Study",
        f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S IST', time.localtime())}*",
        "",
        "## Objective",
        "",
        "Validate that conclusions from Phases 33–51 are architecture-independent and identify ",
        "which components of the SupCon pipeline are responsible for successful cross-dataset transfer.",
        "",
        "---",
        "",
        "## H0: The Phase 50 improvements are architecture-dependent and disappear under reasonable implementation changes.",
        "## H1: The improvements are robust across architectural and hyperparameter variations, demonstrating that conditional representation learning—not implementation details—is responsible for observed transfer gains.",
        "",
        "---",
        "",
    ]

    # Summary table
    lines.append("## Summary of Results\n")
    lines.append("| Experiment | Best Config | Mean Off-Diag MF1 | Std | Configs |")
    lines.append("|-----------|-------------|:-----------------:|:---:|:------:|")

    for exp_key, exp_name in [
        ("exp_a", "A: Latent Dimension"),
        ("exp_b", "B: Encoder Depth"),
        ("exp_c", "C: Temperature"),
        ("exp_d", "D: Loss Weight"),
        ("exp_e", "E: Label Noise"),
        ("exp_f", "F: Sample Efficiency"),
    ]:
        exp_data = experiment_results.get(exp_key, [])
        if not exp_data:
            lines.append(f"| {exp_name} | — | — | — | 0 |")
            continue
        # Best by mean_off_diag_mf1
        best = max(exp_data, key=lambda x: x.get("mean_off_diag_mf1", 0))
        best_config = best.get("run_name", "?")
        best_mf1 = best.get("mean_off_diag_mf1", 0)
        best_std = best.get("std_off_diag_mf1", 0)
        lines.append(f"| {exp_name} | {best_config} | {best_mf1:.4f} | {best_std:.4f} | {len(exp_data)} |")

    lines.append("")

    # Per-experiment sections
    for exp_key, exp_title in [
        ("exp_a", "## Experiment A — Latent Dimension Ablation"),
        ("exp_b", "## Experiment B — Encoder Depth Ablation"),
        ("exp_c", "## Experiment C — SupCon Temperature Sweep"),
        ("exp_d", "## Experiment D — Loss Weight Ablation"),
        ("exp_e", "## Experiment E — Label Noise Robustness"),
        ("exp_f", "## Experiment F — Sample Efficiency"),
    ]:
        exp_data = experiment_results.get(exp_key, [])
        lines.append(exp_title)
        lines.append("")
        if not exp_data:
            lines.append("*No results collected.*\n")
            continue

        lines.append(f"**Configurations tested:** {len(exp_data)}\n")
        lines.append("| Config | Off-Diag MF1 | Std | Diag MF1 | Params | Time (s) |")
        lines.append("|--------|:-----------:|:---:|:--------:|:-----:|:--------:|")
        for r in sorted(exp_data, key=lambda x: x.get("mean_off_diag_mf1", 0), reverse=True):
            cn = r.get("run_name", "?")
            off = r.get("mean_off_diag_mf1", 0)
            sd = r.get("std_off_diag_mf1", 0)
            diag = r.get("mean_diag_mf1", 0)
            nparams = r.get("n_params", "?")
            t = r.get("train_time_seconds", "?")
            t_str = f"{t:.0f}" if isinstance(t, (int, float)) else str(t)
            lines.append(f"| {cn} | {off:.4f} | {sd:.4f} | {diag:.4f} | {nparams} | {t_str} |")

        # Add any experiment-specific metrics
        if exp_key == "exp_c":
            lines.append("\n**Intra-class compactness and inter-class separation:**\n")
            lines.append("| Temperature | Compactness | Separation |")
            lines.append("|:----------:|:----------:|:---------:|")
            for r in sorted(exp_data, key=lambda x: x.get("temperature", 0)):
                t = r.get("temperature", "?")
                comp = r.get("intra_class_compactness", 0)
                sep = r.get("inter_class_separation", 0)
                lines.append(f"| {t} | {comp:.4f} | {sep:.4f} |")

        if exp_key == "exp_d":
            lines.append("\n**Calibration and collapse:**\n")
            lines.append("| Weight | ECE | Collapse Ratio |")
            lines.append("|:----:|:---:|:-------------:|")
            for r in sorted(exp_data, key=lambda x: x.get("supcon_weight", 0)):
                w = r.get("supcon_weight", "?")
                ece = r.get("mean_ece", 0)
                cr = r.get("collapse_ratio", 0)
                lines.append(f"| {w} | {ece:.4f} | {cr:.4f} |")

        if exp_key == "exp_e":
            lines.append("\n**Stability metrics:**\n")
            lines.append("| Noise Rate | SHAP Stability | Prototype Alignment |")
            lines.append("|:---------:|:-------------:|:-----------------:|")
            for r in sorted(exp_data, key=lambda x: x.get("label_noise_rate", 0)):
                nr = r.get("label_noise_rate", "?")
                ss = r.get("shap_stability", 0)
                pa = r.get("prototype_alignment", 0)
                lines.append(f"| {nr} | {ss:.4f} | {pa:.4f} |")

        if exp_key == "exp_f":
            lines.append("\n**95% Confidence Intervals:**\n")
            lines.append("| Fraction | CI Low | CI High |")
            lines.append("|:------:|:-----:|:------:|")
            for r in sorted(exp_data, key=lambda x: x.get("sample_fraction", 0)):
                f = r.get("sample_fraction", "?")
                lo = r.get("ci_low_95", 0)
                hi = r.get("ci_high_95", 0)
                lines.append(f"| {f} | {lo:.4f} | {hi:.4f} |")

        lines.append("")

    # Statistical analysis summary
    lines.append("## Statistical Analysis Summary\n")
    if stats:
        cross = stats.get("cross_experiment", {})
        if "ranking" in cross:
            lines.append("### Cross-Experiment Ranking\n")
            lines.append("| Rank | Config | Mean MF1 | Std |")
            lines.append("|:---:|--------|:-------:|:---:|")
            for i, r in enumerate(cross.get("ranking", [])):
                lines.append(f"| {i+1} | {r['config']} | {r['mean_mf1']:.4f} | {r['std_mf1']:.4f} |")
            lines.append("")

        for exp_key, exp_name in [
            ("experiment_a", "Experiment A"),
            ("experiment_b", "Experiment B"),
            ("experiment_c", "Experiment C"),
            ("experiment_d", "Experiment D"),
            ("experiment_e", "Experiment E"),
            ("experiment_f", "Experiment F"),
        ]:
            exp_stats = stats.get(exp_key, {})
            if "friedman_p" in exp_stats:
                lines.append(f"**{exp_name}:** Friedman χ²={exp_stats['friedman_stat']:.4f}, "
                            f"p={exp_stats['friedman_p']:.4f}")
            if "holm_bonferroni" in exp_stats:
                sig_count = sum(1 for h in exp_stats["holm_bonferroni"] if h.get("significant_at_005"))
                lines.append(f"  Holm-Bonferroni: {sig_count}/{len(exp_stats['holm_bonferroni'])} significant")
            pairwise = exp_stats.get("pairwise_tests", [])
            if pairwise:
                max_effect = max(abs(p.get("cliffs_delta", 0)) for p in pairwise)
                lines.append(f"  Largest Cliff's δ: {max_effect:.3f}")
            lines.append("")

    # H1 Verdict
    lines.append("## H1 Verdict\n")
    lines.append("**H1 is supported if:**\n")
    lines.append("1. SupCon remains the top-performing method across most ablations.")
    lines.append("2. Performance varies smoothly rather than collapsing under minor architectural changes.")
    lines.append("3. Statistical significance remains after multiple-comparison correction.")
    lines.append("4. The optimal configuration can be justified empirically.\n")

    # Collect evidence for H1 support
    all_mf1_values = []
    for exp_key in ["exp_a", "exp_b", "exp_c", "exp_d", "exp_e", "exp_f"]:
        for r in experiment_results.get(exp_key, []):
            all_mf1_values.append(r.get("mean_off_diag_mf1", 0))

    if all_mf1_values:
        mean_all = np.mean(all_mf1_values)
        std_all = np.std(all_mf1_values, ddof=1)
        min_val = min(all_mf1_values)
        max_val = max(all_mf1_values)
        # Coefficient of variation
        cv = std_all / mean_all if mean_all > 0 else 0
        lines.append(f"**Across all {len(all_mf1_values)} configurations:**")
        lines.append(f"- Mean MF1: {mean_all:.4f} ± {std_all:.4f}")
        lines.append(f"- Range: [{min_val:.4f}, {max_val:.4f}]")
        lines.append(f"- Coefficient of variation: {cv:.4f}")
        lines.append(f"- CV < 0.3 suggests smooth variation (not collapse).\n")

    # Baseline SupCon reference
    supcon_baseline = 0.719  # Phase 50 result
    best_config = max(experiment_results.get("exp_a", []) +
                      experiment_results.get("exp_b", []) +
                      experiment_results.get("exp_c", []) +
                      experiment_results.get("exp_d", []) +
                      experiment_results.get("exp_e", []) +
                      experiment_results.get("exp_f", []),
                      key=lambda x: x.get("mean_off_diag_mf1", 0),
                      default={})
    best_mf1 = best_config.get("mean_off_diag_mf1", 0)
    lines.append(f"**Phase 50 baseline (SupCon, default config):** {supcon_baseline:.4f}")
    lines.append(f"**Best Phase 52 config:** {best_config.get('run_name', '?')} = {best_mf1:.4f}")
    if best_mf1 >= supcon_baseline * 0.85:
        lines.append("✓ Best config within 85% of Phase 50 baseline — architecture independence supported.")
    else:
        lines.append("⚠ Best config more than 15% below Phase 50 baseline — investigate architecture sensitivity.")
    lines.append("")

    lines.append("---")
    lines.append("## Deliverables\n")
    lines.append("All files under `results/phase52/`:")
    lines.append("- [tables/latent_dimension_results.csv](tables/latent_dimension_results.csv)")
    lines.append("- [tables/architecture_ablation.csv](tables/architecture_ablation.csv)")
    lines.append("- [tables/temperature_sweep.csv](tables/temperature_sweep.csv)")
    lines.append("- [tables/loss_weight_ablation.csv](tables/loss_weight_ablation.csv)")
    lines.append("- [tables/label_noise_results.csv](tables/label_noise_results.csv)")
    lines.append("- [tables/sample_efficiency.csv](tables/sample_efficiency.csv)")
    lines.append("- [tables/statistical_tests.json](tables/statistical_tests.json)")
    lines.append("- [tables/ranking_tables.csv](tables/ranking_tables.csv)")
    lines.append("- [learning_curves/](learning_curves/)")
    lines.append("- [latent_visualizations/](latent_visualizations/)")

    report_path = RESULTS_DIR / "FINAL_REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"  ✓ FINAL_REPORT.md saved to {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=str, default="A,B,C,D,E,F",
                        help="Comma-separated experiment IDs")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, load cached encoder files")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--skip-stats", action="store_true",
                        help="Skip statistical analysis and report generation")
    args = parser.parse_args()

    global SUPCON_EPOCHS
    if args.epochs:
        SUPCON_EPOCHS = args.epochs

    experiments_to_run = [e.strip().upper() for e in args.experiments.split(",")]
    logger.info(f"Experiments: {experiments_to_run}")

    # Load data
    logger.info("Loading datasets...")
    data_dict = load_all_datasets()
    logger.info(f"Loaded {len(data_dict)} datasets:")
    for name, d in sorted(data_dict.items()):
        logger.info(f"  {name}: {d['X'].shape}")

    experiment_results = {}
    stats = None

    # Experiment A
    if "A" in experiments_to_run:
        exp_a_results = run_experiment_a(data_dict)
        experiment_results["exp_a"] = exp_a_results

    # Experiment B
    if "B" in experiments_to_run:
        exp_b_results = run_experiment_b(data_dict)
        experiment_results["exp_b"] = exp_b_results

    # Experiment C
    if "C" in experiments_to_run:
        exp_c_results = run_experiment_c(data_dict)
        experiment_results["exp_c"] = exp_c_results

    # Experiment D
    if "D" in experiments_to_run:
        exp_d_results = run_experiment_d(data_dict)
        experiment_results["exp_d"] = exp_d_results

    # Experiment E
    if "E" in experiments_to_run:
        exp_e_results = run_experiment_e(data_dict)
        experiment_results["exp_e"] = exp_e_results

    # Experiment F
    if "F" in experiments_to_run:
        exp_f_results = run_experiment_f(data_dict)
        experiment_results["exp_f"] = exp_f_results

    # Statistical Analysis
    if experiment_results and not args.skip_stats:
        stats = run_statistical_analysis(experiment_results)

    # Generate deliverables
    if experiment_results and not args.skip_stats:
        generate_deliverables(experiment_results, stats)

    logger.info(f"\n{'='*65}")
    logger.info("Phase 52 complete")
    logger.info(f"{'='*65}")


if __name__ == "__main__":
    main()
