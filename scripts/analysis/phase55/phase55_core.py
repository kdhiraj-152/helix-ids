#!/usr/bin/env python3
"""Phase 55 Core — Shared infrastructure for all experiments.

Data loading, model architectures, loss functions, transfer evaluation,
geometry metrics, information theory estimates, and statistical analysis.
"""

import gc, json, logging, math, os, sys, time, warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.spatial import procrustes as sp_procrustes
from sklearn import metrics as sk_metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    davies_bouldin_score, silhouette_score, f1_score,
    roc_auc_score, brier_score_loss, mutual_info_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

SEED = 42
rng = np.random.RandomState(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float32

PROJ = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJ / "data" / "processed" / "multi_dataset_v1"
RESULTS = PROJ / "results" / "phase55"

INPUT_DIM = 17
NUM_CLASSES = 2
MAX_SAMPLES_PER_DATASET = 15000
SUPCON_EPOCHS = 25
PATIENCE = 8
LR = 1e-3
BATCH_SIZE = 256
LATENT_DIM = 32
TEMPERATURE = 0.1
SUPCON_WEIGHT = 0.5

DATASET_NAMES = ["nsl_kdd", "unsw_nb15", "cicids", "ton_iot", "bot_iot", "cicids2017"]
DATASET_DISPLAY = {
    "nsl_kdd": "NSL-KDD", "unsw_nb15": "UNSW-NB15", "cicids": "CICIDS2018",
    "ton_iot": "TON-IoT", "bot_iot": "Bot-IoT", "cicids2017": "CICIDS2017",
}
CLASS_NAMES = ["Normal", "Attack"]

# ═══════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("phase55")


def setup_logging():
    RESULTS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    fh = logging.FileHandler(RESULTS / "phase55_run.log", mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    logger.info(f"Phase 55 starting — device={DEVICE}")

# ═══════════════════════════════════════════════════════════════════════════
# Memory
# ═══════════════════════════════════════════════════════════════════════════


def cleanup_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def to_binary(y):
    return (y > 0).astype(np.int64)


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def subsample_stratified(X, y, mx, rng_=None):
    if rng_ is None:
        rng_ = rng
    n = X.shape[0]
    if n <= mx:
        return X.copy(), y.copy()
    classes = np.unique(y)
    idx = []
    for c in classes:
        ci = np.where(y == c)[0]
        t = max(1, int(mx * len(ci) / n))
        if len(ci) > t:
            ci = rng_.choice(ci, size=t, replace=False)
        idx.extend(ci.tolist())
    rng_.shuffle(idx)
    a = np.array(idx)
    return X[a], y[a]


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
    loader = MultiDatasetLoader(project_root=str(PROJ))
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


# ═══════════════════════════════════════════════════════════════════════════
# Data Preparation
# ═══════════════════════════════════════════════════════════════════════════

def prepare_data(data_dict, val_split=0.15):
    train_data, val_data = {}, {}
    for name in sorted(data_dict.keys()):
        X = data_dict[name]["X"]
        y = to_binary(data_dict[name]["y"])
        n = X.shape[0]
        if n > MAX_SAMPLES_PER_DATASET:
            idx = rng.permutation(n)[:MAX_SAMPLES_PER_DATASET]
            X = X[idx]; y = y[idx]; n = MAX_SAMPLES_PER_DATASET
        nv = max(1, int(n * val_split))
        idx = rng.permutation(n)
        X_tr, X_vl = X[idx[nv:]], X[idx[:nv]]
        y_tr, y_vl = y[idx[nv:]], y[idx[:nv]]
        sc = StandardScaler()
        train_data[name] = {"X": sc.fit_transform(X_tr), "y": y_tr}
        val_data[name] = {"X": sc.transform(X_vl), "y": y_vl}
    return train_data, val_data


def build_loaders(train_data):
    loaders = {}
    for n, d in sorted(train_data.items()):
        ds = TensorDataset(
            torch.from_numpy(d["X"]).float(), torch.from_numpy(d["y"]).long())
        loaders[n] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    return loaders


def build_val_loaders(val_data):
    loaders = {}
    for n, d in sorted(val_data.items()):
        ds = TensorDataset(
            torch.from_numpy(d["X"]).float(), torch.from_numpy(d["y"]).long())
        loaders[n] = DataLoader(ds, batch_size=BATCH_SIZE * 2)
    return loaders


def loader_iter(loaders_dict, steps=200):
    names = sorted(loaders_dict.keys())
    iters = {n: iter(loaders_dict[n]) for n in names}
    for _ in range(steps):
        n = names[rng.randint(len(names))]
        try:
            xb, yb = next(iters[n])
        except StopIteration:
            iters[n] = iter(loaders_dict[n])
            xb, yb = next(iters[n])
        yield n, xb.to(DEVICE), yb.to(DEVICE)


def fit_scalers(data_dict):
    sc = {}
    for name in sorted(data_dict.keys()):
        X = data_dict[name]["X"]
        n = X.shape[0]
        if n > MAX_SAMPLES_PER_DATASET:
            idx = rng.permutation(n)[:MAX_SAMPLES_PER_DATASET]
            X = X[idx]
        sc[name] = StandardScaler().fit(X)
    return sc

def extract_latents(encoder, data_dict, scalers, batch_size=1024):
    """Extract latents, subsampling large datasets for speed."""
    encoder.eval()
    latents, labels = {}, {}
    for name in sorted(data_dict.keys()):
        X = scalers[name].transform(data_dict[name]["X"])
        y = to_binary(data_dict[name]["y"])
        # Subsample for fast evaluation
        if len(X) > 15000:
            idx = rng.permutation(len(X))[:15000]
            X, y = X[idx], y[idx]
        dl = DataLoader(TensorDataset(torch.from_numpy(X).float()), batch_size=batch_size)
        zs = []
        with torch.no_grad():
            for (xb,) in dl:
                zs.append(encoder(xb.to(DEVICE)).cpu().numpy())
        latents[name] = np.vstack(zs)
        labels[name] = y
    return latents, labels


# ═══════════════════════════════════════════════════════════════════════════
# Model Architectures
# ═══════════════════════════════════════════════════════════════════════════

class MLPEncoder(nn.Module):
    """Basic MLP encoder: 17 → hidden → latent."""
    def __init__(self, inp=INPUT_DIM, latent=LATENT_DIM, n_layers=3, hidden=64):
        super().__init__()
        layers = [nn.Linear(inp, hidden), nn.ReLU(), nn.BatchNorm1d(hidden)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.ReLU(), nn.BatchNorm1d(hidden)])
        layers.append(nn.Linear(hidden, latent))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPEncoderFlex(nn.Module):
    """Encoder returning backbone features + optional activations."""
    def __init__(self, inp=INPUT_DIM, latent=LATENT_DIM):
        super().__init__()
        self.fc1 = nn.Linear(inp, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.out = nn.Linear(64, latent)

    def forward(self, x, return_activations=False):
        h1 = F.relu(self.bn1(self.fc1(x)))
        h2 = F.relu(self.bn2(self.fc2(h1)))
        h3 = F.relu(self.bn3(self.fc3(h2))) + h2
        z = self.out(h3)
        if return_activations:
            return z, (h1, h2, h3)
        return z

    def forward_activations(self, x):
        acts = {}
        h1 = F.relu(self.bn1(self.fc1(x))); acts["layer1"] = h1
        h2 = F.relu(self.bn2(self.fc2(h1))); acts["layer2"] = h2
        h3 = F.relu(self.bn3(self.fc3(h2))) + h2; acts["layer3"] = h3
        z = self.out(h3); acts["latent"] = z
        return acts


class ClassifierHead(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, z):
        return self.net(z)


class ProjectionHead(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, proj_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, proj_dim),
        )

    def forward(self, z):
        return self.net(z)


class IdentityProjection(nn.Module):
    """No projection — passes latent directly."""
    def forward(self, z):
        return z


# ═══════════════════════════════════════════════════════════════════════════
# Loss Functions (Experiment A — Objective Decomposition)
# ═══════════════════════════════════════════════════════════════════════════

def supcon_loss(features, labels, temp=TEMPERATURE):
    """Supervised Contrastive Loss."""
    dev = features.device
    bs = features.shape[0]
    lbl = labels.contiguous().view(-1, 1)
    feat = F.normalize(features, dim=1)
    sim = feat @ feat.T / temp
    mask = torch.eye(bs, device=dev, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)
    pos = (lbl == lbl.T).float().masked_fill(mask, 0)
    if pos.sum() < 1:
        return torch.tensor(0.0, device=dev)
    return (sim.logsumexp(dim=1) - (sim * pos).sum(dim=1) / pos.sum(dim=1).clamp(min=1)).mean()


def triplet_loss(features, labels, margin=1.0):
    """Batch-hard triplet loss — fully vectorized."""
    dev = features.device
    feat = F.normalize(features, dim=1)
    sim = feat @ feat.T  # bs × bs
    bs = feat.shape[0]

    # For each anchor, find hardest positive and negative
    # pos_mask[i, j] = 1 if labels[i] == labels[j] and i != j
    eye = torch.eye(bs, device=dev, dtype=torch.bool)
    pos_mask = (labels[:, None] == labels[None, :]).float()
    pos_mask = pos_mask.masked_fill(eye, 0)
    neg_mask = 1.0 - pos_mask - torch.eye(bs, device=dev)

    # Hardest positive: max similarity among same-class (closest positive)
    hardest_pos = (sim * pos_mask).max(dim=1)[0]
    # Hardest negative: min similarity among different-class (closest negative)
    # For triplet, we need closest negative = highest sim among negatives
    hardest_neg = (sim * neg_mask).max(dim=1)[0]

    # Triplet: d(a,p) - d(a,n) + margin = (1-s_max_pos) - (1-s_min_neg) + margin = s_min_neg - s_max_pos + margin
    # Wait: hardest_pos is max sim among positives (closest positive)
    # hardest_neg is max sim among negatives (closest negative = most similar negative)
    # We want: sim(a, positive) < sim(a, negative) + margin
    # hinge = max(0, hardest_neg - hardest_pos + margin)
    hinge = hardest_neg - hardest_pos + margin
    valid = pos_mask.sum(dim=1) > 0
    loss = torch.clamp(hinge[valid], min=0).mean() if valid.any() else torch.tensor(0.0, device=dev)
    return loss


def arcface_loss(features, labels, s=30.0, m=0.5):
    """ArcFace additive angular margin loss."""
    dev = features.device
    n_classes = len(torch.unique(labels))
    W = getattr(arcface_loss, "W", None)
    if W is None or W.shape[0] != n_classes:
        W = torch.randn(n_classes, features.shape[1], device=dev) * 0.01
        arcface_loss.W = nn.Parameter(W, requires_grad=True)
    W = F.normalize(arcface_loss.W, dim=1)
    feat = F.normalize(features, dim=1)
    logits = feat @ W.T * s
    target_logits = logits.gather(1, labels.unsqueeze(1))
    target_logits = target_logits - m * s
    one_hot = F.one_hot(labels, num_classes=n_classes).float()
    logits = logits - logits.max(dim=1, keepdim=True)[0]
    logits = logits * (1 - one_hot) + target_logits * one_hot
    loss = F.cross_entropy(logits, labels)
    return loss

# Weight matrix for ArcFace
arcface_loss.W = None


def center_loss(features, labels, num_classes=NUM_CLASSES, alpha=0.5):
    """Center loss: L_c = 0.5 * ||z - c_y||²."""
    dev = features.device
    centers = getattr(center_loss, "centers", None)
    if centers is None or centers.shape[0] != num_classes:
        centers = torch.randn(num_classes, features.shape[1], device=dev) * 0.1
        center_loss.centers = nn.Parameter(centers, requires_grad=True)
    batch_centers = center_loss.centers[labels]
    loss = (features - batch_centers).pow(2).sum(dim=1).mean() * alpha
    return loss

center_loss.centers = None


def proxynca_loss(features, labels, temperature=1.0):
    """Proxy-NCA loss."""
    dev = features.device
    feat = F.normalize(features, dim=1)
    n_classes = len(torch.unique(labels))
    proxies = getattr(proxynca_loss, "proxies", None)
    if proxies is None or proxies.shape[0] != n_classes:
        proxies = torch.randn(n_classes, features.shape[1], device=dev) * 0.1
        proxynca_loss.proxies = nn.Parameter(proxies, requires_grad=True)
    proxies = F.normalize(proxynca_loss.proxies, dim=1)
    sim = feat @ proxies.T  # n_samples × n_classes
    pos_sim = sim.gather(1, labels.unsqueeze(1))
    loss = -torch.log(
        torch.exp(pos_sim / temperature) /
        torch.exp(sim / temperature).sum(dim=1, keepdim=True).clamp(min=1e-12)
    ).mean()
    return loss

proxynca_loss.proxies = None


def circle_loss(features, labels, m=0.25, gamma=80):
    """Circle loss."""
    dev = features.device
    feat = F.normalize(features, dim=1)
    sim = feat @ feat.T
    bs = feat.shape[0]
    eye = torch.eye(bs, device=dev, dtype=torch.bool)
    pos_mask = (labels == labels.T.unsqueeze(1)).float()
    pos_mask = pos_mask.masked_fill(eye, 0)
    neg_mask = (labels != labels.T.unsqueeze(1)).float()
    if pos_mask.sum() < 1 or neg_mask.sum() < 1:
        return torch.tensor(0.0, device=dev)
    sp = pos_mask.sum(dim=1).clamp(min=1)
    sn = neg_mask.sum(dim=1).clamp(min=1)
    op = 1 + m  # positive margin
    on = -m     # negative margin
    ap = torch.clamp(op - sim.detach(), min=0)
    an = torch.clamp(sim.detach() - on, min=0)
    p_logits = gamma * ap * (sim - op)
    n_logits = gamma * an * (sim - on)
    p_logsum = torch.logsumexp(p_logits.masked_fill(pos_mask == 0, -1e9), dim=1)
    n_logsum = torch.logsumexp(n_logits.masked_fill(neg_mask == 0, -1e9), dim=1)
    loss = F.softplus(p_logsum + n_logsum).mean()
    return loss


def infonce_loss(features, labels, temp=TEMPERATURE):
    """InfoNCE loss (self-supervised contrastive)."""
    dev = features.device
    bs = features.shape[0]
    feat = F.normalize(features, dim=1)
    sim = feat @ feat.T / temp
    eye = torch.eye(bs, device=dev, dtype=torch.bool)
    # InfoNCE: treat other views of same class as positives
    pos_mask = (labels == labels.T.unsqueeze(1)).float()
    pos_mask = pos_mask.masked_fill(eye, 0)
    neg_mask = (~eye).float() - pos_mask  # all non-self, non-positives
    if pos_mask.sum() < 1 or neg_mask.sum() < 1:
        return torch.tensor(0.0, device=dev)
    # For each anchor: -log( sum(exp(pos_sim)) / sum(exp(all_sim)) )
    pos_exp = torch.exp(sim * pos_mask)
    neg_exp = torch.exp(sim * neg_mask)
    numerator = pos_exp.sum(dim=1).clamp(min=1e-12)
    denominator = numerator + neg_exp.sum(dim=1)
    loss = -torch.log(numerator / denominator).mean()
    return loss


LOSS_FUNCTIONS = {
    "ce": ("ce", None),  # CrossEntropyLoss is handled separately
    "supcon": ("supcon", lambda proj, yb: supcon_loss(proj, yb)),
    "triplet": ("triplet", lambda proj, yb: triplet_loss(proj, yb)),
    "arcface": ("arcface", lambda proj, yb: arcface_loss(proj, yb)),
    "center": ("center", lambda proj, yb: center_loss(proj, yb)),
    "proxynca": ("proxynca", lambda proj, yb: proxynca_loss(proj, yb)),
    "circle": ("circle", lambda proj, yb: circle_loss(proj, yb)),
    "infonce": ("infonce", lambda proj, yb: infonce_loss(proj, yb)),
}
LOSS_NAMES = list(LOSS_FUNCTIONS.keys())
LOSS_DISPLAY = {
    "ce": "Cross Entropy", "supcon": "SupCon", "triplet": "Triplet",
    "arcface": "ArcFace", "center": "Center Loss", "proxynca": "ProxyNCA",
    "circle": "Circle Loss", "infonce": "InfoNCE",
}


def reset_loss_weights():
    """Reset learnable parameters in loss functions for a fresh training run."""
    arcface_loss.W = None
    center_loss.centers = None
    proxynca_loss.proxies = None


# ═══════════════════════════════════════════════════════════════════════════
# Generic Training Function
# ═══════════════════════════════════════════════════════════════════════════

def train_encoder(data_dict, loss_name="supcon", latent_dim=LATENT_DIM,
                  temperature=TEMPERATURE, loss_weight=SUPCON_WEIGHT,
                  epochs=SUPCON_EPOCHS, lr=LR, run_name="model", track_geom=False,
                  swap_to_ce_at=None, use_projection=True):
    """Generic encoder training with any loss function. Returns encoder, clf, hist, latents.

    If swap_to_ce_at is set, switch from contrastive to CE-only at that epoch.
    If track_geom, per-epoch geometry metrics are tracked.
    """
    reset_loss_weights()
    train_data, val_data = prepare_data(data_dict)
    loaders = build_loaders(train_data)
    vloaders = build_val_loaders(val_data)
    scalers = fit_scalers(data_dict)

    encoder = MLPEncoderFlex(inp=INPUT_DIM, latent=latent_dim).to(DEVICE)
    clf = ClassifierHead(latent_dim=latent_dim).to(DEVICE)
    if use_projection and loss_name != "ce":
        proj = ProjectionHead(latent_dim=latent_dim).to(DEVICE)
    else:
        proj = IdentityProjection().to(DEVICE)

    params = list(encoder.parameters()) + list(clf.parameters())
    if use_projection and loss_name != "ce":
        params += list(proj.parameters())
    opt = optim.Adam(params, lr=lr)
    crit = nn.CrossEntropyLoss()

    loss_fn = LOSS_FUNCTIONS.get(loss_name, (None, None))[1]

    best_vl = float("inf")
    patience = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    if loss_name != "ce":
        history[f"{loss_name}_loss"] = []

    # Optional geometry tracking
    if track_geom:
        eval_subset_Z, eval_subset_y = _build_eval_subset(data_dict, scalers, n_per_dataset=2000)
        epoch_metrics = defaultdict(list)
    else:
        eval_subset_Z = eval_subset_y = None
        epoch_metrics = None

    steps = max(80, sum(ds.dataset.tensors[0].shape[0] // BATCH_SIZE
                         for ds in loaders.values()) // (3 * max(len(loaders), 1)))
    steps = min(steps, 300)

    for ep in range(epochs):
        encoder.train()
        clf.train()
        if use_projection and loss_name != "ce":
            proj.train()
        losses = []
        extra_losses = []
        corr = 0
        tot = 0

        # Determine if we should use only CE this epoch (for swap experiments)
        use_ce_only = (swap_to_ce_at is not None and ep >= swap_to_ce_at and loss_name != "ce")

        for _, xb, yb in loader_iter(loaders, steps):
            opt.zero_grad()
            z = encoder(xb)
            logits = clf(z)
            cls_l = crit(logits, yb)

            if use_ce_only or loss_name == "ce":
                total_loss = cls_l
            else:
                z_proj = proj(z)
                extra_l = loss_fn(z_proj, yb) if loss_fn is not None else 0
                extra_losses.append(extra_l.item() if isinstance(extra_l, torch.Tensor) else 0)
                total_loss = cls_l + loss_weight * extra_l

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10)
            opt.step()
            losses.append(total_loss.item())
            corr += (logits.argmax(1) == yb).sum().item()
            tot += yb.shape[0]

        # Validation
        encoder.eval()
        clf.eval()
        vlosses = []
        vcorr = 0
        vtot = 0
        with torch.no_grad():
            for loader in vloaders.values():
                for xb, yb in loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    loss = crit(clf(encoder(xb)), yb)
                    vlosses.append(loss.item())
                    vcorr += (clf(encoder(xb)).argmax(1) == yb).sum().item()
                    vtot += yb.shape[0]

        tl = float(np.mean(losses)) if losses else 0
        vl = float(np.mean(vlosses)) if vlosses else 0
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(corr / max(tot, 1))
        history["val_acc"].append(vcorr / max(vtot, 1))
        if loss_name != "ce" and extra_losses:
            history.setdefault(f"{loss_name}_loss", []).append(float(np.mean(extra_losses)))

        # Track geometry
        if track_geom and eval_subset_Z is not None:
            metrics_ep, _, _ = _compute_epoch_metrics(eval_subset_Z, eval_subset_y, encoder)
            for k, v in metrics_ep.items():
                epoch_metrics[k].append(v)

        if (ep + 1) % 10 == 0 or ep == 0:
            log_extra = f" {loss_name}={np.mean(extra_losses):.4f}" if extra_losses else ""
            logger.info(f"  [{run_name}] Ep {ep+1:2d}/{epochs} train={tl:.6f} val={vl:.6f} "
                        f"acc={corr/max(tot,1):.4f}{log_extra}")

        if vl < best_vl - 1e-6:
            best_vl = vl
            patience = 0
        else:
            patience += 1
        if patience >= PATIENCE:
            logger.info(f"  [{run_name}] Early stopping at epoch {ep+1}")
            break

    logger.info(f"  [{run_name}] Done. best_val_loss={best_vl:.6f}")

    # Extract final latents
    latents_dict, labels_dict = extract_latents(encoder, data_dict, scalers)

    if track_geom and epoch_metrics:
        return encoder, clf, history, latents_dict, labels_dict, scalers, dict(epoch_metrics)
    return encoder, clf, history, latents_dict, labels_dict, scalers


def _build_eval_subset(data_dict, scalers, n_per_dataset=2000):
    subset_Z, subset_y = {}, {}
    for name in sorted(data_dict.keys()):
        X = data_dict[name]["X"]
        y = to_binary(data_dict[name]["y"])
        X_s, y_s = subsample_stratified(X, y, n_per_dataset, rng_=np.random.RandomState(42))
        X_t = torch.from_numpy(scalers[name].transform(X_s)).float()
        subset_Z[name] = X_t
        subset_y[name] = y_s
    return subset_Z, subset_y


def _compute_epoch_metrics(subset_Z, subset_y, encoder):
    encoder.eval()
    latents, labels = {}, {}
    with torch.no_grad():
        for name in sorted(subset_Z.keys()):
            xb = subset_Z[name].to(DEVICE)
            z = encoder(xb).cpu().numpy()
            latents[name] = z
            labels[name] = subset_y[name]
    all_z = np.vstack(list(latents.values()))
    all_y = np.concatenate(list(labels.values()))
    metrics = {}
    if len(np.unique(all_y)) > 1:
        metrics["intra_class_var"] = float(_intra_class_variance(all_z, all_y))
        metrics["inter_class_dist"] = float(_inter_class_distance(all_z, all_y))
        metrics["margin_ratio"] = metrics["intra_class_var"] / max(metrics["inter_class_dist"], 1e-12)
        metrics["fisher_ratio"] = float(_fisher_discriminant_ratio(all_z, all_y))
        try:
            metrics["davies_bouldin"] = float(davies_bouldin_score(all_z, all_y))
        except Exception:
            metrics["davies_bouldin"] = float("nan")
        try:
            metrics["silhouette"] = float(silhouette_score(all_z, all_y))
        except Exception:
            metrics["silhouette"] = float("nan")
    return metrics, latents, labels


# ═══════════════════════════════════════════════════════════════════════════
# Transfer Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def compute_transfer_matrix(latents_dict, labels_dict):
    """Compute cross-dataset transfer matrix using LogisticRegression (fast)."""
    names = sorted(latents_dict.keys())
    n = len(names)
    ss = {}
    for nm in names:
        Z, y = latents_dict[nm], labels_dict[nm]
        # Subsample to max 20000 for speed
        if len(Z) > 20000:
            idx = rng.permutation(len(Z))[:20000]
            ss[nm] = (Z[idx], y[idx])
        else:
            ss[nm] = (Z, y)
    mf1_mat = np.zeros((n, n))
    for i, src in enumerate(names):
        clf = LogisticRegression(max_iter=200, random_state=SEED, C=1.0, n_jobs=1)
        Z_src, y_src = ss[src]
        clf.fit(Z_src, y_src)
        for j, tgt in enumerate(names):
            Zt, yt = latents_dict[tgt], labels_dict[tgt]
            # Also subsample target for speed if large
            if len(Zt) > 20000:
                idx = rng.permutation(len(Zt))[:20000]
                Zt = Zt[idx]; yt = yt[idx]
            yp = clf.predict(Zt)
            mf1_mat[i, j] = float(f1_score(yt, yp, average="macro", zero_division=0))
    return mf1_mat, names


def compute_transfer_offdiag(mf1_mat, names):
    n = len(names)
    off = [mf1_mat[i, j] for i in range(n) for j in range(n) if i != j]
    mean_off = float(np.mean(off)) if off else 0
    std_off = float(np.std(off, ddof=1)) if len(off) > 1 else 0
    return mean_off, std_off


def evaluate_transfer(latents_dict, labels_dict):
    """Full transfer evaluation returning matrix + summary."""
    mf1_mat, names = compute_transfer_matrix(latents_dict, labels_dict)
    mean_off, std_off = compute_transfer_offdiag(mf1_mat, names)
    n = len(names)
    diag = [mf1_mat[i, i] for i in range(n)]
    return {
        "mf1_matrix": mf1_mat.tolist(),
        "names": names,
        "mean_off_diag_mf1": mean_off,
        "std_off_diag_mf1": std_off,
        "mean_diag_mf1": float(np.mean(diag)) if diag else 0,
        "std_diag_mf1": float(np.std(diag, ddof=1)) if len(diag) > 1 else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Geometry Metrics
# ═══════════════════════════════════════════════════════════════════════════

def _intra_class_variance(Z, y):
    classes = np.unique(y)
    variances = []
    for c in classes:
        mask = y == c
        if mask.sum() <= 1:
            continue
        variances.append(np.mean(np.var(Z[mask], axis=0, ddof=1)))
    return float(np.mean(variances)) if variances else 0.0


def _inter_class_distance(Z, y):
    classes = np.unique(y)
    centroids = []
    for c in classes:
        mask = y == c
        if mask.sum() == 0:
            continue
        centroids.append(np.mean(Z[mask], axis=0))
    if len(centroids) < 2:
        return 0.0
    dists = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dists.append(np.linalg.norm(centroids[i] - centroids[j]))
    return float(np.mean(dists)) if dists else 0.0


def _fisher_discriminant_ratio(Z, y):
    classes = np.unique(y)
    overall_mean = np.mean(Z, axis=0)
    s_w = np.zeros(Z.shape[1])
    s_b = np.zeros(Z.shape[1])
    for c in classes:
        mask = y == c
        if mask.sum() == 0:
            continue
        class_mean = np.mean(Z[mask], axis=0)
        s_w += np.sum((Z[mask] - class_mean) ** 2, axis=0)
        n_c = mask.sum()
        diff = class_mean - overall_mean
        s_b += n_c * diff ** 2
    sw_trace = np.sum(s_w) / Z.shape[0]
    sb_trace = np.sum(s_b) / Z.shape[0]
    return float(sb_trace / max(sw_trace, 1e-12))


def compute_all_geometry(Z, y):
    """Compute all geometry metrics for a latent space."""
    metrics = {}
    if len(np.unique(y)) > 1:
        metrics["intra_class_var"] = _intra_class_variance(Z, y)
        metrics["inter_class_dist"] = _inter_class_distance(Z, y)
        metrics["fisher_ratio"] = _fisher_discriminant_ratio(Z, y)
        try:
            metrics["silhouette"] = float(silhouette_score(Z, y))
        except Exception:
            metrics["silhouette"] = float("nan")
        try:
            metrics["davies_bouldin"] = float(davies_bouldin_score(Z, y))
        except Exception:
            metrics["davies_bouldin"] = float("nan")
    return metrics


def compute_domain_invariance(Z_all, D_all):
    """Predict dataset ID from latents → lower accuracy = more domain invariance."""
    le = LabelEncoder()
    D = le.fit_transform(D_all)
    clf = LogisticRegression(max_iter=200, random_state=SEED)
    split = int(len(Z_all) * 0.8)
    clf.fit(Z_all[:split], D[:split])
    acc = clf.score(Z_all[split:], D[split:])
    return float(acc)


# ═══════════════════════════════════════════════════════════════════════════
# Information Theory Estimates
# ═══════════════════════════════════════════════════════════════════════════

def estimate_hscore(Z, Y, n_samples=2000):
    """H-score: 0.5 * tr(cov(Z)^{-1} @ cov(E[Z|Y]))."""
    Z = Z if Z.ndim >= 2 else Z.reshape(-1, 1)
    n = Z.shape[0]
    if n > n_samples:
        idx = rng.choice(n, size=n_samples, replace=False)
        Z = Z[idx]; Y = Y[idx]
    Z_c = Z - Z.mean(axis=0)
    cov_z = np.atleast_2d(np.cov(Z_c, rowvar=False))
    cov_z_inv = np.linalg.pinv(cov_z + 1e-6 * np.eye(cov_z.shape[0]))
    classes = np.unique(Y)
    cond_means = np.zeros_like(Z_c)
    for c in classes:
        mask = Y == c
        if mask.sum() > 0:
            cond_means[mask] = Z_c[mask].mean(axis=0)
    cov_cond = np.atleast_2d(np.cov(cond_means, rowvar=False))
    h_score = 0.5 * np.trace(cov_z_inv @ cov_cond)
    return float(h_score)


def estimate_mutual_info_disc(Z, Y, n_bins=20):
    """Discretized mutual information estimate."""
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    dim = Z.shape[1]
    if dim > 10:
        dims = rng.choice(dim, size=10, replace=False)
    else:
        dims = np.arange(dim)
    mis = []
    for d in dims:
        z_d = Z[:, d]
        z_disc = np.digitize(z_d, np.percentile(z_d, np.linspace(0, 100, n_bins + 1)[1:-1]))
        mi = mutual_info_score(z_disc, Y)
        mis.append(mi)
    return float(np.mean(mis)) if mis else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Representation Similarity (Experiment E)
# ═══════════════════════════════════════════════════════════════════════════

def compute_cka(Z1, Z2):
    """Linear CKA between two representation matrices."""
    lr = min(Z1.shape[0], Z2.shape[0], 10000)
    idx = rng.choice(Z1.shape[0], size=lr, replace=False)
    X1_c = Z1[idx] - Z1[idx].mean(axis=0)
    X2_c = Z2[idx] - Z2[idx].mean(axis=0)
    hsic_xy = float(np.sum((X2_c.T @ X1_c) ** 2)) / (lr - 1) ** 2
    hsic_xx = float(np.sum((X1_c.T @ X1_c) ** 2)) / (lr - 1) ** 2
    hsic_yy = float(np.sum((X2_c.T @ X2_c) ** 2)) / (lr - 1) ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    return hsic_xy / denom if denom > 1e-12 else 0.0


def compute_svcca(Z1, Z2, dim=16):
    """SVCCA: CCA after SVD truncation — whitened principal components."""
    from scipy.linalg import svd
    lr = min(Z1.shape[0], Z2.shape[0], 10000)
    idx = rng.choice(Z1.shape[0], size=lr, replace=False)
    Z1_s = Z1[idx] - Z1[idx].mean(axis=0)
    Z2_s = Z2[idx] - Z2[idx].mean(axis=0)
    U1, S1, _ = svd(Z1_s, full_matrices=False)
    U2, S2, _ = svd(Z2_s, full_matrices=False)
    d = min(dim, U1.shape[1], U2.shape[1])
    # Whitened PCs: left singular vectors are already orthonormal
    _, S, _ = svd(U1[:, :d].T @ U2[:, :d], full_matrices=False)
    return float(np.mean(S))


def compute_pwcca(Z1, Z2, dim=16):
    """Projection-weighted CCA — whitened principal components."""
    from scipy.linalg import svd
    lr = min(Z1.shape[0], Z2.shape[0], 10000)
    idx = rng.choice(Z1.shape[0], size=lr, replace=False)
    Z1_s = Z1[idx] - Z1[idx].mean(axis=0)
    Z2_s = Z2[idx] - Z2[idx].mean(axis=0)
    U1, S1, _ = svd(Z1_s, full_matrices=False)
    U2, S2, _ = svd(Z2_s, full_matrices=False)
    d = min(dim, U1.shape[1], U2.shape[1])
    Z1_t = U1[:, :d]
    Z2_t = U2[:, :d]
    Q1, _ = np.linalg.qr(Z1_t)
    Q2, _ = np.linalg.qr(Z2_t)
    _, S, _ = svd(Q1.T @ Q2, full_matrices=False)
    alphas = S / S.sum()
    return float(np.sum(alphas * S))

def compute_procrustes_similarity(Z1, Z2):
    """Procrustes similarity: lower disparity = more similar."""
    lr = min(Z1.shape[0], Z2.shape[0], 5000)
    idx = rng.choice(Z1.shape[0], size=lr, replace=False)
    Z1_s = Z1[idx]
    Z2_s = Z2[idx]
    # Normalize to same shape for Procrustes
    d = min(Z1_s.shape[1], Z2_s.shape[1])
    m1, m2, disparity = sp_procrustes(Z1_s[:, :d], Z2_s[:, :d])
    return 1.0 / (1.0 + disparity)  # 1 = identical, 0 = completely different


def compute_orthogonal_similarity(Z1, Z2):
    """Orthogonal similarity: how well can we align Z1 to Z2 via rotation."""
    from scipy.linalg import orthogonal_procrustes
    lr = min(Z1.shape[0], Z2.shape[0], 5000)
    idx = rng.choice(Z1.shape[0], size=lr, replace=False)
    Z1_s = Z1[idx]
    Z2_s = Z2[idx]
    d = min(Z1_s.shape[1], Z2_s.shape[1])
    R, _ = orthogonal_procrustes(Z1_s[:, :d], Z2_s[:, :d])
    # How well Z1 aligns with Z2 after optimal rotation
    Z1_aligned = Z1_s[:, :d] @ R
    cos_sim = np.mean(np.sum(Z1_aligned * Z2_s[:, :d], axis=1) /
                      (np.linalg.norm(Z1_aligned, axis=1) * np.linalg.norm(Z2_s[:, :d], axis=1) + 1e-12))
    return float(cos_sim)


def compute_subspace_overlap(Z1, Z2, dim=16):
    """Subspace overlap via principal angles."""
    d = min(dim, Z1.shape[1], Z2.shape[1])
    U1, _, _ = np.linalg.svd(Z1 - Z1.mean(axis=0), full_matrices=False)
    U2, _, _ = np.linalg.svd(Z2 - Z2.mean(axis=0), full_matrices=False)
    # Principal angles
    _, S, _ = np.linalg.svd(U1[:, :d].T @ U2[:, :d], full_matrices=False)
    return float(np.mean(S))  # mean cosine of principal angles


def compute_all_similarity_metrics(Z1, Z2):
    """Compute all representation similarity metrics between two latents."""
    return {
        "cka": compute_cka(Z1, Z2),
        "svcca": compute_svcca(Z1, Z2),
        "pwcca": compute_pwcca(Z1, Z2),
        "procrustes": compute_procrustes_similarity(Z1, Z2),
        "orthogonal": compute_orthogonal_similarity(Z1, Z2),
        "subspace_overlap": compute_subspace_overlap(Z1, Z2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Statistical Analysis
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_ci(data, func=np.mean, n_bootstrap=2000, ci=0.95):
    """Bootstrap confidence interval for a function of data."""
    estimates = []
    n = len(data)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        estimates.append(func([data[i] for i in idx]))
    estimates = np.array(sorted(estimates))
    lower_idx = max(0, int((1 - ci) / 2 * n_bootstrap))
    upper_idx = min(n_bootstrap - 1, int((1 + ci) / 2 * n_bootstrap))
    return float(estimates[lower_idx]), float(estimates[upper_idx])


def bootstrap_paired_test(x, y, n_bootstrap=5000):
    """Bootstrap test for paired difference. Returns p-value (H0: diff <= 0)."""
    diff = np.array(x) - np.array(y)
    n = len(diff)
    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_diffs.append(np.mean(diff[idx]))
    # One-sided p: proportion of bootstraps where diff <= 0
    p_val = np.mean(np.array(boot_diffs) <= 0)
    return float(p_val)


def cohens_d(x, y):
    """Cohen's d effect size."""
    n1, n2 = len(x), len(y)
    s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return float((np.mean(x) - np.mean(y)) / max(sp, 1e-12))


def bayesian_effect_size(x, y, n_samples=10000):
    """Bayesian estimation of effect size using normal approximation."""
    mu_diff = np.mean(x) - np.mean(y)
    se_diff = np.sqrt(np.var(x, ddof=1)/len(x) + np.var(y, ddof=1)/len(y))
    posterior_samples = mu_diff + np.random.randn(n_samples) * se_diff
    prob_direction = np.mean(posterior_samples > 0)
    return {
        "mean_effect": float(mu_diff),
        "sd_effect": float(se_diff),
        "prob_direction": float(prob_direction),
    }


def tost_equivalence(x, y, epsilon=0.05):
    """Two One-Sided Tests for equivalence within ±epsilon."""
    from scipy.stats import ttest_ind
    t1, p1 = ttest_ind(x, y, alternative="greater")
    # H0: mean(x) - mean(y) >= epsilon  (test if diff < epsilon)
    n1, n2 = len(x), len(y)
    s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    se = sp * np.sqrt(1/n1 + 1/n2)
    t_lower = (np.mean(x) - np.mean(y) + epsilon) / max(se, 1e-12)
    p_lower = scipy_stats.t.sf(t_lower, n1 + n2 - 2)
    t_upper = (np.mean(x) - np.mean(y) - epsilon) / max(se, 1e-12)
    p_upper = scipy_stats.t.cdf(t_upper, n1 + n2 - 2)
    p_eq = max(p_lower, p_upper)
    return {
        "equivalent": bool(p_eq < 0.05),
        "p_value": float(p_eq),
        "epsilon": epsilon,
        "mean_diff": float(np.mean(x) - np.mean(y)),
    }


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"  Saved {path}")


def create_result_dirs():
    subdirs = [
        "objective_decomposition", "latent_surgery",
        "representation_isomorphism", "mediation_analysis",
        "stress_tests", "intrinsic_transfer_dimension",
        "causal_graphs", "models", "latents", "stats", "tables",
    ]
    for s in subdirs:
        (RESULTS / s).mkdir(parents=True, exist_ok=True)
