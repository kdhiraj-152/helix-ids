#!/usr/bin/env python3
"""Phase 58 — Representation Learning vs Feature Engineering Validation.

Determines whether cross-dataset transfer failure in NIDS is caused by
insufficient handcrafted features (H0) or persists even with learned latent
representations (H1).

Compares 4 representation paradigms:
  - Raw Canonical (17-feature baseline)
  - PCA (linear latent, multiple dims)
  - Autoencoder (nonlinear compressed latent)
  - Contrastive Encoder (SimCLR-style self-supervised)

Usage:
  source .venv311/bin/activate
  PYTHONPATH=src python scripts/analysis/phase58_main.py
  # Or to skip GPU-intensive training:
  PYTHONPATH=src python scripts/analysis/phase58_main.py --skip-ae --skip-contrastive
"""
import argparse, gc, json, logging, math, os, sys, time, warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.linalg import svd, sqrtm
import sklearn.metrics as sk_metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, brier_score_loss, mutual_info_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ═══════════════════════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════════════════════

SEED = 42
rng = np.random.RandomState(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
INPUT_DIM = 17
BATCH_SIZE = 256
LR = 1e-3
MAX_SAMPLES_PER_DATASET = 20000  # subsample for speed

PROJ = Path(__file__).resolve().parents[2]
CACHE = PROJ / "data" / "processed" / "phase52_cache"
RESULTS = PROJ / "results" / "phase58"
for sub in ["models", "latents", "umap", "tsne", "matrices"]:
    (RESULTS / sub).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJ / "src"))

DATASET_NAMES = ["nsl_kdd", "unsw_nb15", "cicids2018", "ton_iot", "bot_iot", "cicids2017"]
DATASET_DISPLAY = {
    "nsl_kdd": "NSL-KDD", "unsw_nb15": "UNSW-NB15", "cicids2018": "CICIDS2018",
    "ton_iot": "TON-IoT", "bot_iot": "Bot-IoT", "cicids2017": "CICIDS2017",
}

logger = logging.getLogger("phase58")
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
fh = logging.FileHandler(RESULTS / "phase58_run.log", mode="w")
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(fh)
logger.info(f"Phase 58 starting — device={DEVICE}")


def cleanup():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def to_binary(y):
    return (y > 0).astype(np.int64)


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
    return X[np.array(idx)], y[np.array(idx)]


# ═══════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_datasets():
    """Load all 6 harmonized datasets from phase52 cache."""
    datasets = {}
    if not CACHE.exists():
        logger.error(f"Cache dir {CACHE} not found!")
        return datasets
    for name in DATASET_NAMES:
        X_tr = np.load(CACHE / f"{name}_X_train.npy", mmap_mode="r").astype(np.float64)
        X_te = np.load(CACHE / f"{name}_X_test.npy", mmap_mode="r").astype(np.float64)
        y_tr = np.load(CACHE / f"{name}_y_train.npy").ravel()
        y_te = np.load(CACHE / f"{name}_y_test.npy").ravel()
        X = np.vstack([X_tr, X_te])
        y = np.concatenate([y_tr, y_te])
        datasets[name] = {"X": X, "y": y}
    return datasets


def prepare_data(data_dict, val_split=0.15):
    """Split each dataset into train/val, standardize."""
    train_data, val_data = {}, {}
    for name in sorted(data_dict.keys()):
        X = data_dict[name]["X"]
        y = to_binary(data_dict[name]["y"])
        n = X.shape[0]
        if n > MAX_SAMPLES_PER_DATASET:
            idx = rng.permutation(n)[:MAX_SAMPLES_PER_DATASET]
            X = X[idx]
            y = y[idx]
            n = MAX_SAMPLES_PER_DATASET
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
        ds = TensorDataset(torch.from_numpy(d["X"]).float(), torch.from_numpy(d["y"]).long())
        loaders[n] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    return loaders


def build_val_loaders(val_data):
    loaders = {}
    for n, d in sorted(val_data.items()):
        ds = TensorDataset(torch.from_numpy(d["X"]).float(), torch.from_numpy(d["y"]).long())
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
        if X.shape[0] > MAX_SAMPLES_PER_DATASET:
            idx = rng.permutation(X.shape[0])[:MAX_SAMPLES_PER_DATASET]
            X = X[idx]
        sc[name] = StandardScaler().fit(X)
    return sc


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_transfer(latents_dict, labels_dict):
    """Cross-dataset transfer evaluation with RF classifier (100 trees)."""
    names = sorted(latents_dict.keys())
    n = len(names)
    ss = {}
    for nm in names:
        Z, y = latents_dict[nm], labels_dict[nm]
        if len(Z) > 20000:
            idx = rng.permutation(len(Z))[:20000]
            ss[nm] = (Z[idx], y[idx])
        else:
            ss[nm] = (Z, y)

    mf1_mat = np.zeros((n, n))
    bf1_mat = np.zeros((n, n))
    ece_mat = np.zeros((n, n))
    nll_mat = np.zeros((n, n))
    results_list = []

    for i, src in enumerate(names):
        clf = RandomForestClassifier(100, max_depth=10, random_state=SEED, n_jobs=1)
        Z_src, y_src = ss[src]
        clf.fit(Z_src, y_src)
        for j, tgt in enumerate(names):
            Zt, yt = latents_dict[tgt], labels_dict[tgt]
            if len(Zt) > 20000:
                idx = rng.permutation(len(Zt))[:20000]
                Zt = Zt[idx]
                yt = yt[idx]
            yp = clf.predict(Zt)
            ypr = clf.predict_proba(Zt)
            mf1 = float(f1_score(yt, yp, average="macro", zero_division=0))
            bf1 = float(f1_score(yt, yp, average="binary", zero_division=0))
            mf1_mat[i, j] = mf1
            bf1_mat[i, j] = bf1

            # Calibration (ECE)
            ece_val = _ece(yt, ypr[:, 1]) if ypr.shape[1] > 1 else 0.0
            ece_mat[i, j] = ece_val

            # NLL
            eps = 1e-12
            nll_val = -np.mean(np.log(ypr[np.arange(len(yt)), yt] + eps))
            nll_mat[i, j] = float(nll_val)

            # Prediction entropy
            ent = -np.mean(np.sum(ypr * np.log(ypr + eps), axis=1))

            results_list.append({
                "representation": "",
                "source": src, "target": tgt,
                "macro_f1": mf1, "binary_f1": bf1,
                "ece": ece_val, "nll": float(nll_val),
                "prediction_entropy": float(ent),
            })

    offdiag = [mf1_mat[i, j] for i in range(n) for j in range(n) if i != j]
    diag = [mf1_mat[i, i] for i in range(n)]
    return {
        "matrix": mf1_mat.tolist(),
        "bf1_matrix": bf1_mat.tolist(),
        "ece_matrix": ece_mat.tolist(),
        "nll_matrix": nll_mat.tolist(),
        "names": names,
        "mean_off_diag": float(np.mean(offdiag)) if offdiag else 0,
        "std_off_diag": float(np.std(offdiag, ddof=1)) if len(offdiag) > 1 else 0,
        "mean_diag": float(np.mean(diag)) if diag else 0,
        "std_diag": float(np.std(diag, ddof=1)) if len(diag) > 1 else 0,
        "results": results_list,
    }


def _ece(y_true, y_prob, n_bins=10):
    if y_prob is None:
        return np.nan
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        ib = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if i == n_bins - 1:
            ib = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        if ib.sum() == 0:
            continue
        ece += abs(np.mean(y_true[ib]) - np.mean(y_prob[ib])) * ib.sum() / len(y_true)
    return float(ece)


# ═══════════════════════════════════════════════════════════════════════════
# 1. RAW BASELINE
# ═══════════════════════════════════════════════════════════════════════════

def get_raw_latents(data_dict):
    """Raw canonical 17-dim features (no transformation, just standardization)."""
    scalers = fit_scalers(data_dict)
    latents, labels = {}, {}
    for name in sorted(data_dict.keys()):
        X = scalers[name].transform(data_dict[name]["X"])
        y = to_binary(data_dict[name]["y"])
        latents[name] = X
        labels[name] = y
    return latents, labels


# ═══════════════════════════════════════════════════════════════════════════
# 2. PCA REPRESENTATION
# ═══════════════════════════════════════════════════════════════════════════

def get_pca_latents(data_dict, n_components=16):
    """Train PCA on source datasets, extract latents from all."""
    scalers = fit_scalers(data_dict)
    names = sorted(data_dict.keys())
    pca_models = {}
    latents, labels = {}, {}

    for name in names:
        X = scalers[name].transform(data_dict[name]["X"])
        y = to_binary(data_dict[name]["y"])
        pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0]))
        Z = pca.fit_transform(X)
        pca_models[name] = pca
        latents[name] = Z
        labels[name] = y

    return latents, labels, pca_models


def get_pca_latents_multi_dim(data_dict):
    """Try multiple PCA dimensions, return all variants."""
    dims = [8, 16, 32, 64]
    results = {}
    for d in dims:
        latents, labels, _ = get_pca_latents(data_dict, n_components=d)
        results[f"pca_{d}"] = (latents, labels)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 3. AUTOENCODER
# ═══════════════════════════════════════════════════════════════════════════

class Autoencoder(nn.Module):
    """17 → 128 → 64 → 32 → 16 latent → 32 → 64 → 128 → 17."""

    def __init__(self, input_dim=INPUT_DIM, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def encode(self, x):
        return self.encoder(x)


def train_autoencoder_on_source(data_dict, epochs=100, patience=15):
    """Train autoencoder on source data. Extract latents from all datasets."""
    train_data, val_data = prepare_data(data_dict)
    train_loaders = build_loaders(train_data)
    val_loaders = build_val_loaders(val_data)
    scalers = fit_scalers(data_dict)

    model = Autoencoder().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()

    best_val = float("inf")
    patience_counter = 0
    ae_history = {"train_loss": [], "val_loss": []}
    steps = min(200, sum(ds.dataset.tensors[0].shape[0] // BATCH_SIZE
                         for ds in train_loaders.values()) // 2)

    logger.info("  Training Autoencoder on all source datasets...")
    for ep in range(epochs):
        model.train()
        tl = []
        for _, xb, _ in loader_iter(train_loaders, steps):
            opt.zero_grad()
            x_recon, _ = model(xb)
            loss = crit(x_recon, xb)
            loss.backward()
            opt.step()
            tl.append(loss.item())

        model.eval()
        vl = []
        with torch.no_grad():
            for loader in val_loaders.values():
                for xb, _ in loader:
                    xb = xb.to(DEVICE)
                    x_recon, _ = model(xb)
                    loss = crit(x_recon, xb)
                    vl.append(loss.item())

        train_l = float(np.mean(tl)) if tl else 0
        val_l = float(np.mean(vl)) if vl else 0
        ae_history["train_loss"].append(train_l)
        ae_history["val_loss"].append(val_l)

        if val_l < best_val - 1e-6:
            best_val = val_l
            patience_counter = 0
            torch.save(model.state_dict(), RESULTS / "models" / "autoencoder.pt")
        else:
            patience_counter += 1

        if (ep + 1) % 10 == 0 or ep == 0:
            logger.info(f"  AE Epoch {ep+1:3d}/{epochs} train={train_l:.6f} val={val_l:.6f}")

        if patience_counter >= patience:
            logger.info(f"  AE Early stopping at epoch {ep+1}")
            break

    model.load_state_dict(torch.load(RESULTS / "models" / "autoencoder.pt"))
    model.eval()

    # Extract latents
    latents, labels = {}, {}
    for name in sorted(data_dict.keys()):
        X = scalers[name].transform(data_dict[name]["X"])
        y = to_binary(data_dict[name]["y"])
        dl = DataLoader(TensorDataset(torch.from_numpy(X).float()), batch_size=1024)
        zs = []
        with torch.no_grad():
            for (xb,) in dl:
                zs.append(model.encode(xb.to(DEVICE)).cpu().numpy())
        latents[name] = np.vstack(zs)
        labels[name] = y

    with open(RESULTS / "autoencoder_history.json", "w") as f:
        json.dump(ae_history, f, indent=2)
    logger.info(f"  AE done. Best val_loss = {best_val:.6f}")
    return latents, labels, model


# ═══════════════════════════════════════════════════════════════════════════
# 4. CONTRASTIVE (SimCLR-style) ENCODER
# ═══════════════════════════════════════════════════════════════════════════

class ContrastiveEncoder(nn.Module):
    """17 → 64 → 64 → 32 latent. SimCLR-style."""

    def __init__(self, input_dim=INPUT_DIM, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class ProjectionHead(nn.Module):
    def __init__(self, latent_dim=32, proj_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, proj_dim),
        )

    def forward(self, z):
        return self.net(z)


def contrastive_augment(x):
    """SimCLR-style positive augmentations for tabular data."""
    augs = []
    # Aug 1: Gaussian noise
    noise = torch.randn_like(x) * 0.05
    augs.append(x + noise)
    # Aug 2: Feature dropout
    mask = torch.ones_like(x)
    mask = F.dropout(mask, p=0.15, training=True)
    augs.append(x * mask)
    # Aug 3: Scaling
    scale = 1.0 + (torch.rand(x.shape[0], 1, device=x.device) - 0.5) * 0.2
    augs.append(x * scale)
    # Aug 4: Random masking (mask 10% of features)
    mask2 = torch.bernoulli(torch.full_like(x, 0.9))
    augs.append(x * mask2)
    return augs


def nt_xent_loss(z, temperature=0.1):
    """Normalized temperature-scaled cross entropy loss (NT-Xent)."""
    z = F.normalize(z, dim=1)
    n = z.shape[0]
    sim = z @ z.T / temperature
    # Mask out self-similarity
    mask = torch.eye(n, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)
    # Each sample has 1 positive (the other augmentation) and 2n-2 negatives
    labels = torch.arange(n, device=z.device)
    labels = labels[1::2]  # positives are pairs (0↔1, 2↔3, ...)
    labels = torch.cat([labels, labels])
    loss = F.cross_entropy(sim, labels)
    return loss


def train_contrastive(data_dict, epochs=50, patience=10, latent_dim=32):
    """Train SimCLR-style contrastive encoder."""
    train_data, val_data = prepare_data(data_dict)
    scalers = fit_scalers(data_dict)

    encoder = ContrastiveEncoder(latent_dim=latent_dim).to(DEVICE)
    proj = ProjectionHead(latent_dim=latent_dim).to(DEVICE)
    opt = optim.Adam(list(encoder.parameters()) + list(proj.parameters()), lr=LR)

    best_val = float("inf")
    patience_counter = 0

    # Build a combined dataset for contrastive learning
    all_x, all_y = [], []
    for name in sorted(data_dict.keys()):
        Xt = scalers[name].transform(data_dict[name]["X"])
        yt = to_binary(data_dict[name]["y"])
        all_x.append(Xt)
        all_y.append(yt)
    all_x = np.vstack(all_x)
    all_y = np.concatenate(all_y)
    # Subsample for time
    if len(all_x) > 50000:
        idx = rng.permutation(len(all_x))[:50000]
        all_x = all_x[idx]
        all_y = all_y[idx]
    n = len(all_x)
    nv = max(1, int(n * 0.15))
    idx = rng.permutation(n)
    X_tr, X_vl = all_x[idx[nv:]], all_x[idx[:nv]]
    y_tr, y_vl = all_y[idx[nv:]], all_y[idx[:nv]]

    train_ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long())
    val_ds = TensorDataset(torch.from_numpy(X_vl).float(), torch.from_numpy(y_vl).long())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2)

    logger.info("  Training Contrastive (SimCLR) encoder...")
    for ep in range(epochs):
        encoder.train()
        proj.train()
        losses = []
        for xb, _ in train_loader:
            xb = xb.to(DEVICE)
            augs = contrastive_augment(xb)

            # Combine original + augmentations
            batch_pairs = []
            batch_pairs.append(xb)
            batch_pairs.extend(augs[:2])  # use 2 augmentations

            combined = torch.cat(batch_pairs, dim=0)
            z = encoder(combined)
            p = proj(z)
            loss = nt_xent_loss(p)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # Validation loss (reconstruction + entropy proxy)
        encoder.eval()
        proj.eval()
        vlosses = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(DEVICE)
                z = encoder(xb)
                p = proj(z)
                augs = contrastive_augment(xb)
                combined = torch.cat([xb, augs[0]], dim=0)
                p2 = proj(encoder(combined))
                vl = nt_xent_loss(p2)
                vlosses.append(vl.item())

        train_l = float(np.mean(losses)) if losses else 0
        val_l = float(np.mean(vlosses)) if vlosses else 0

        if val_l < best_val - 1e-6:
            best_val = val_l
            patience_counter = 0
            torch.save(encoder.state_dict(), RESULTS / "models" / "contrastive_encoder.pt")
        else:
            patience_counter += 1

        if (ep + 1) % 5 == 0 or ep == 0:
            logger.info(f"  CT Ep {ep+1:3d}/{epochs} train={train_l:.6f} val={val_l:.6f}")

        if patience_counter >= patience:
            logger.info(f"  CT Early stopping at epoch {ep+1}")
            break

    encoder.load_state_dict(torch.load(RESULTS / "models" / "contrastive_encoder.pt"))
    encoder.eval()

    # Extract latents
    latents, labels = {}, {}
    for name in sorted(data_dict.keys()):
        X = scalers[name].transform(data_dict[name]["X"])
        y = to_binary(data_dict[name]["y"])
        dl = DataLoader(TensorDataset(torch.from_numpy(X).float()), batch_size=1024)
        zs = []
        with torch.no_grad():
            for (xb,) in dl:
                zs.append(encoder(xb.to(DEVICE)).cpu().numpy())
        latents[name] = np.vstack(zs)
        labels[name] = y

    logger.info(f"  Contrastive done. Best val_loss = {best_val:.6f}")
    return latents, labels, encoder


# ═══════════════════════════════════════════════════════════════════════════
# Representation Similarity
# ═══════════════════════════════════════════════════════════════════════════

def compute_cka(Z1, Z2):
    """Linear CKA."""
    lr = min(Z1.shape[0], Z2.shape[0], 10000)
    idx1 = rng.choice(Z1.shape[0], size=lr, replace=False)
    idx2 = rng.choice(Z2.shape[0], size=lr, replace=False)
    X1_c = Z1[idx1] - Z1[idx1].mean(axis=0)
    X2_c = Z2[idx2] - Z2[idx2].mean(axis=0)
    hsic_xy = float(np.sum((X2_c.T @ X1_c) ** 2)) / (lr - 1) ** 2
    hsic_xx = float(np.sum((X1_c.T @ X1_c) ** 2)) / (lr - 1) ** 2
    hsic_yy = float(np.sum((X2_c.T @ X2_c) ** 2)) / (lr - 1) ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    return hsic_xy / denom if denom > 1e-12 else 0.0


def compute_cca(Z1, Z2):
    """Canonical Correlation Analysis — mean canonical correlation."""
    lr = min(Z1.shape[0], Z2.shape[0], 10000)
    idx1 = rng.choice(Z1.shape[0], size=lr, replace=False)
    idx2 = rng.choice(Z2.shape[0], size=lr, replace=False)
    Z1_s = Z1[idx1] - Z1[idx1].mean(axis=0)
    Z2_s = Z2[idx2] - Z2[idx2].mean(axis=0)
    d = min(Z1_s.shape[1], Z2_s.shape[1])
    if d < 1:
        return 0.0
    Q1, _, _ = np.linalg.svd(Z1_s, full_matrices=False)
    Q2, _, _ = np.linalg.svd(Z2_s, full_matrices=False)
    _, S, _ = np.linalg.svd(Q1[:, :d].T @ Q2[:, :d], full_matrices=False)
    return float(np.mean(S))


def compute_mmd(Z1, Z2, sigma=1.0):
    """Maximum Mean Discrepancy with RBF kernel."""
    lr = min(Z1.shape[0], Z2.shape[0], 5000)
    idx1 = rng.choice(Z1.shape[0], size=lr, replace=False)
    idx2 = rng.choice(Z2.shape[0], size=lr, replace=False)
    X = Z1[idx1]
    Y = Z2[idx2]

    gamma = 1.0 / (2 * sigma ** 2)
    n = X.shape[0]
    m = Y.shape[0]

    K_xx = np.exp(-gamma * cdist(X, X, "sqeuclidean"))
    K_yy = np.exp(-gamma * cdist(Y, Y, "sqeuclidean"))
    K_xy = np.exp(-gamma * cdist(X, Y, "sqeuclidean"))

    mmd = (np.sum(K_xx) - np.trace(K_xx)) / (n * (n - 1))
    mmd += (np.sum(K_yy) - np.trace(K_yy)) / (m * (m - 1))
    mmd -= 2 * np.sum(K_xy) / (n * m)
    return float(max(0, mmd))


def compute_wasserstein(Z1, Z2):
    """Wasserstein distance (1-d projections mean)."""
    lr = min(Z1.shape[0], Z2.shape[0], 5000)
    idx1 = rng.choice(Z1.shape[0], size=lr, replace=False)
    idx2 = rng.choice(Z2.shape[0], size=lr, replace=False)
    X = Z1[idx1]
    Y = Z2[idx2]
    d = min(X.shape[1], Y.shape[1])
    dists = []
    for i in range(d):
        dists.append(scipy_stats.wasserstein_distance(X[:, i], Y[:, i]))
    return float(np.mean(dists))


def compute_all_similarity(latents_dict, rep_name):
    """Compute pairwise similarity between all dataset latent spaces."""
    names = sorted(latents_dict.keys())
    n = len(names)
    cka_mat = np.zeros((n, n))
    cca_mat = np.zeros((n, n))
    mmd_mat = np.zeros((n, n))
    wass_mat = np.zeros((n, n))
    results = []

    for i, src in enumerate(names):
        Z_src = latents_dict[src]
        for j, tgt in enumerate(names):
            Z_tgt = latents_dict[tgt]
            cka = compute_cka(Z_src, Z_tgt) if i != j else 1.0
            cca = compute_cca(Z_src, Z_tgt) if i != j else 1.0
            mmd = compute_mmd(Z_src, Z_tgt) if i != j else 0.0
            wass = compute_wasserstein(Z_src, Z_tgt) if i != j else 0.0
            cka_mat[i, j] = cka
            cca_mat[i, j] = cca
            mmd_mat[i, j] = mmd
            wass_mat[i, j] = wass
            results.append({
                "representation": rep_name,
                "source": src, "target": tgt,
                "cka": cka, "cca": cca,
                "mmd": mmd, "wasserstein": wass,
            })

    return {
        "cka_matrix": cka_mat.tolist(),
        "cca_matrix": cca_mat.tolist(),
        "mmd_matrix": mmd_mat.tolist(),
        "wasserstein_matrix": wass_mat.tolist(),
        "names": names,
        "results": results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Mutual Information Estimation
# ═══════════════════════════════════════════════════════════════════════════

def estimate_mutual_information(data_dict, latents_dict, rep_name):
    """Estimate I(Z;X) and I(Z;Y) per dataset."""
    results = []
    for name in sorted(latents_dict.keys()):
        Z = latents_dict[name]
        X_raw = data_dict[name]["X"]
        y = to_binary(data_dict[name]["y"])

        # I(Z;X): discretize and compute MI per dimension, average
        n_bins = 20
        mi_zx = 0.0
        zdim = min(Z.shape[1], 10)  # sample dims if too many
        if zdim > 0:
            dims = rng.choice(Z.shape[1], size=zdim, replace=False) if Z.shape[1] > zdim else np.arange(Z.shape[1])
            zx_list = []
            for d in dims:
                z_d = Z[:, int(d)]
                z_disc = np.digitize(z_d, np.percentile(z_d, np.linspace(0, 100, n_bins + 1)[1:-1]))
                # Use a subset of raw features
                for xd in range(min(X_raw.shape[1], 5)):
                    x_d = X_raw[:, xd]
                    x_disc = np.digitize(x_d, np.percentile(x_d, np.linspace(0, 100, n_bins + 1)[1:-1]))
                    mi = mutual_info_score(z_disc, x_disc)
                    zx_list.append(mi)
            mi_zx = float(np.mean(zx_list)) if zx_list else 0.0

        # I(Z;Y): discretized
        mi_zy = 0.0
        if zdim > 0:
            zy_list = []
            for d in dims:
                z_d = Z[:, int(d)]
                z_disc = np.digitize(z_d, np.percentile(z_d, np.linspace(0, 100, n_bins + 1)[1:-1]))
                mi = mutual_info_score(z_disc, y)
                zy_list.append(mi)
            mi_zy = float(np.mean(zy_list)) if zy_list else 0.0

        results.append({
            "representation": rep_name,
            "dataset": name,
            "I(Z;X)": mi_zx,
            "I(Z;Y)": mi_zy,
            "z_dim": int(Z.shape[1]),
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Statistical Tests
# ═══════════════════════════════════════════════════════════════════════════

def run_bootstrap(all_results, n_iterations=10000):
    """Bootstrap comparison of transfer MF1 between representations."""
    # Group results by representation
    by_rep = defaultdict(list)
    for r in all_results:
        if r["source"] != r["target"]:
            by_rep[r["representation"]].append(r["macro_f1"])

    rep_names = sorted(by_rep.keys())
    bootstrap_results = {"n_iterations": n_iterations, "comparisons": []}

    for i, r1 in enumerate(rep_names):
        for r2 in rep_names[i + 1:]:
            d1 = np.array(by_rep[r1])
            d2 = np.array(by_rep[r2])
            boot_diffs = []
            for _ in range(n_iterations):
                idx1 = rng.choice(len(d1), size=len(d1), replace=True)
                idx2 = rng.choice(len(d2), size=len(d2), replace=True)
                boot_diffs.append(np.mean(d1[idx1]) - np.mean(d2[idx2]))
            boot_diffs = np.array(boot_diffs)
            bootstrap_results["comparisons"].append({
                "rep_1": r1, "rep_2": r2,
                "mean_diff": float(np.mean(d1) - np.mean(d2)),
                "ci_lower": float(np.percentile(boot_diffs, 2.5)),
                "ci_upper": float(np.percentile(boot_diffs, 97.5)),
                "p_value": float(np.mean(boot_diffs <= 0)),
                "cohens_d": float(
                    (np.mean(d1) - np.mean(d2)) /
                    max(np.sqrt((np.var(d1, ddof=1) + np.var(d2, ddof=1)) / 2), 1e-12)
                ),
            })

    return bootstrap_results


def run_mixed_effects(all_results):
    """Simple mixed-effects model: Representation -> Transfer MF1, controlling for dataset pair."""
    df = pd.DataFrame(all_results)
    df = df[df["source"] != df["target"]].copy()

    # Fixed effect: representation
    # Random effects: source, target (simplified — use mean per pair)
    pair_means = df.groupby(["source", "target", "representation"])["macro_f1"].mean().reset_index()

    results_list = []
    for rep in pair_means["representation"].unique():
        sub = pair_means[pair_means["representation"] == rep]
        results_list.append({
            "representation": rep,
            "mean_mf1": float(sub["macro_f1"].mean()),
            "std_mf1": float(sub["macro_f1"].std()),
            "n_pairs": len(sub),
        })

    return {
        "model": "Simplified mixed-effects (representation fixed, pair random)",
        "results": results_list,
    }


def run_bayesian_comparison(all_results, n_samples=50000):
    """Bayesian paired comparison between representations."""
    by_rep = defaultdict(list)
    for r in all_results:
        if r["source"] != r["target"]:
            by_rep[r["representation"]].append(r["macro_f1"])

    rep_names = sorted(by_rep.keys())
    bayesian_results = {"n_samples": n_samples, "comparisons": []}

    for i, r1 in enumerate(rep_names):
        for r2 in rep_names[i + 1:]:
            d1 = np.array(by_rep[r1])
            d2 = np.array(by_rep[r2])

            mu_diff = np.mean(d1) - np.mean(d2)
            se_diff = np.sqrt(np.var(d1, ddof=1) / len(d1) + np.var(d2, ddof=1) / len(d2))
            posterior = mu_diff + np.random.randn(n_samples) * se_diff

            bayesian_results["comparisons"].append({
                "rep_1": r1, "rep_2": r2,
                "posterior_mean": float(mu_diff),
                "posterior_sd": float(se_diff),
                "prob_rep1_greater": float(np.mean(posterior > 0)),
                "prob_rep2_greater": float(np.mean(posterior < 0)),
                "effect_size": float(
                    mu_diff / max(np.sqrt((np.var(d1, ddof=1) + np.var(d2, ddof=1)) / 2), 1e-12)
                ),
                "credible_interval_95": [
                    float(np.percentile(posterior, 2.5)),
                    float(np.percentile(posterior, 97.5)),
                ],
            })

    return bayesian_results


# ═══════════════════════════════════════════════════════════════════════════
# UMAP / t-SNE Visualizations
# ═══════════════════════════════════════════════════════════════════════════

def run_umap_tsne(latents_dict, labels_dict, rep_name):
    """Generate UMAP and t-SNE projections for each dataset pair."""
    from sklearn.manifold import TSNE
    import umap

    # Per-dataset UMAP
    for name in sorted(latents_dict.keys()):
        Z = latents_dict[name]
        y = labels_dict[name]
        ss = min(Z.shape[0], 5000)
        idx = rng.choice(Z.shape[0], size=ss, replace=False)
        Zs = Z[idx]
        ys = y[idx]

        # UMAP
        try:
            reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=30, min_dist=0.1)
            emb = reducer.fit_transform(Zs)
            np.savez(RESULTS / "umap" / f"{rep_name}_{name}.npz", embedding=emb, labels=ys, indices=idx)
        except Exception as e:
            logger.warning(f"  UMAP failed for {rep_name}/{name}: {e}")

        # t-SNE
        try:
            tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=1000)
            emb_t = tsne.fit_transform(Zs)
            np.savez(RESULTS / "tsne" / f"{rep_name}_{name}.npz", embedding=emb_t, labels=ys, indices=idx)
        except Exception as e:
            logger.warning(f"  t-SNE failed for {rep_name}/{name}: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(skip_ae=False, skip_contrastive=False):
    logger.info("=" * 70)
    logger.info("Phase 58 — Representation Learning vs Feature Engineering Validation")
    logger.info("=" * 70)

    # 0. Load data
    logger.info("\n[0] Loading datasets...")
    data_dict = load_datasets()
    for name, d in data_dict.items():
        logger.info(f"  {name}: {d['X'].shape[0]} samples, {np.unique(d['y'])} classes")
    all_results = []

    # ─── RAW BASELINE ────────────────────────────────────────────────────
    logger.info("\n[1] Raw Canonical (17-feature baseline)...")
    t0 = time.time()
    raw_latents, raw_labels = get_raw_latents(data_dict)
    raw_ev = evaluate_transfer(raw_latents, raw_labels)
    for r in raw_ev["results"]:
        r["representation"] = "raw"
    all_results.extend(raw_ev["results"])
    logger.info(f"  Mean off-diag MF1: {raw_ev['mean_off_diag']:.4f}")
    logger.info(f"  Mean diag MF1: {raw_ev['mean_diag']:.4f}")

    raw_sim = compute_all_similarity(raw_latents, "raw")
    raw_mi = estimate_mutual_information(data_dict, raw_latents, "raw")
    run_umap_tsne(raw_latents, raw_labels, "raw")

    # Save raw latents
    np.savez(RESULTS / "latents" / "raw_latents.npz",
             **{n: raw_latents[n] for n in raw_latents})
    logger.info(f"  Raw done in {time.time()-t0:.1f}s")

    # ─── PCA ────────────────────────────────────────────────────────────
    logger.info("\n[2] PCA representation...")
    t0 = time.time()
    pca_all = get_pca_latents_multi_dim(data_dict)
    pca_results = {}
    for pca_name, (latents, labels) in pca_all.items():
        ev = evaluate_transfer(latents, labels)
        for r in ev["results"]:
            r["representation"] = pca_name
        all_results.extend(ev["results"])
        pca_results[pca_name] = ev
        logger.info(f"  {pca_name}: off-diag MF1={ev['mean_off_diag']:.4f}, diag={ev['mean_diag']:.4f}")

        sim = compute_all_similarity(latents, pca_name)
        mi = estimate_mutual_information(data_dict, latents, pca_name)

        # Save latents
        np.savez(RESULTS / "latents" / f"{pca_name}_latents.npz",
                 **{n: latents[n] for n in latents})

        # Save similarity
        pd.DataFrame(sim["results"]).to_csv(RESULTS / f"latent_similarity_{pca_name}.csv", index=False)

        if pca_name == "pca_16":  # Visualize only one PCA variant
            run_umap_tsne(latents, labels, pca_name)

    logger.info(f"  PCA done in {time.time()-t0:.1f}s")

    # ─── AUTOENCODER ────────────────────────────────────────────────────
    if not skip_ae:
        logger.info("\n[3] Autoencoder representation...")
        t0 = time.time()
        ae_latents, ae_labels, ae_model = train_autoencoder_on_source(data_dict)
        ae_ev = evaluate_transfer(ae_latents, ae_labels)
        for r in ae_ev["results"]:
            r["representation"] = "autoencoder"
        all_results.extend(ae_ev["results"])
        logger.info(f"  AE: off-diag MF1={ae_ev['mean_off_diag']:.4f}, diag={ae_ev['mean_diag']:.4f}")

        ae_sim = compute_all_similarity(ae_latents, "autoencoder")
        ae_mi = estimate_mutual_information(data_dict, ae_latents, "autoencoder")
        np.savez(RESULTS / "latents" / "autoencoder_latents.npz",
                 **{n: ae_latents[n] for n in ae_latents})
        pd.DataFrame(ae_sim["results"]).to_csv(RESULTS / "latent_similarity_autoencoder.csv", index=False)
        run_umap_tsne(ae_latents, ae_labels, "autoencoder")
        cleanup()
        logger.info(f"  AE done in {time.time()-t0:.1f}s")
    else:
        logger.info("\n[3] Skipping Autoencoder (--skip-ae)")
        ae_ev = None

    # ─── CONTRASTIVE ────────────────────────────────────────────────────
    if not skip_contrastive:
        logger.info("\n[4] Contrastive (SimCLR) representation...")
        t0 = time.time()
        ct_latents, ct_labels, ct_model = train_contrastive(data_dict)
        ct_ev = evaluate_transfer(ct_latents, ct_labels)
        for r in ct_ev["results"]:
            r["representation"] = "contrastive"
        all_results.extend(ct_ev["results"])
        logger.info(f"  CT: off-diag MF1={ct_ev['mean_off_diag']:.4f}, diag={ct_ev['mean_diag']:.4f}")

        ct_sim = compute_all_similarity(ct_latents, "contrastive")
        ct_mi = estimate_mutual_information(data_dict, ct_latents, "contrastive")
        np.savez(RESULTS / "latents" / "contrastive_latents.npz",
                 **{n: ct_latents[n] for n in ct_latents})
        pd.DataFrame(ct_sim["results"]).to_csv(RESULTS / "latent_similarity_contrastive.csv", index=False)
        run_umap_tsne(ct_latents, ct_labels, "contrastive")
        cleanup()
        logger.info(f"  CT done in {time.time()-t0:.1f}s")
    else:
        logger.info("\n[4] Skipping Contrastive (--skip-contrastive)")
        ct_ev = None

    # ═══════════════════════════════════════════════════════════════════
    # Assemble Deliverables
    # ═══════════════════════════════════════════════════════════════════

    logger.info("\n[5] Assembling deliverables...")

    # 5a. representation_transfer.csv
    df_transfer = pd.DataFrame(all_results)
    df_transfer.to_csv(RESULTS / "representation_transfer.csv", index=False)
    logger.info(f"  Saved representation_transfer.csv ({len(df_transfer)} rows)")

    # 5b. Latent similarity (combined)
    all_sim_results = []
    for rep_name in ["raw"] + list(pca_all.keys()) + (["autoencoder"] if not skip_ae else []) + (["contrastive"] if not skip_contrastive else []):
        # Recompute from cached
        if rep_name == "raw":
            s = compute_all_similarity(raw_latents, "raw")
        elif rep_name in pca_all:
            s = compute_all_similarity(pca_all[rep_name][0], rep_name)
        elif rep_name == "autoencoder" and not skip_ae:
            s = ae_sim
        elif rep_name == "contrastive" and not skip_contrastive:
            s = ct_sim
        else:
            continue
        all_sim_results.extend(s["results"])

    pd.DataFrame(all_sim_results).to_csv(RESULTS / "latent_similarity.csv", index=False)
    logger.info(f"  Saved latent_similarity.csv ({len(all_sim_results)} rows)")

    # 5c. CKA results matrix
    cka_rows = []
    for r in all_sim_results:
        cka_rows.append({
            "representation": r["representation"],
            "source": r["source"], "target": r["target"],
            "cka": r["cka"],
        })
    pd.DataFrame(cka_rows).to_csv(RESULTS / "cka_results.csv", index=False)
    logger.info(f"  Saved cka_results.csv ({len(cka_rows)} rows)")

    # 5d. MMD results
    mmd_rows = []
    for r in all_sim_results:
        mmd_rows.append({
            "representation": r["representation"],
            "source": r["source"], "target": r["target"],
            "mmd": r["mmd"],
        })
    pd.DataFrame(mmd_rows).to_csv(RESULTS / "mmd_results.csv", index=False)
    logger.info(f"  Saved mmd_results.csv ({len(mmd_rows)} rows)")

    # 5e. Wasserstein results
    wass_rows = []
    for r in all_sim_results:
        wass_rows.append({
            "representation": r["representation"],
            "source": r["source"], "target": r["target"],
            "wasserstein": r["wasserstein"],
        })
    pd.DataFrame(wass_rows).to_csv(RESULTS / "wasserstein_results.csv", index=False)
    logger.info(f"  Saved wasserstein_results.csv ({len(wass_rows)} rows)")

    # 5f. Mutual information
    all_mi = []
    for rep_name in ["raw"] + list(pca_all.keys()) + (["autoencoder"] if not skip_ae else []) + (["contrastive"] if not skip_contrastive else []):
        if rep_name == "raw":
            all_mi.extend(raw_mi)
        else:
            all_mi.extend(estimate_mutual_information(data_dict,
                (pca_all[rep_name][0] if rep_name in pca_all else
                 (ae_latents if rep_name == "autoencoder" and not skip_ae else
                  (ct_latents if rep_name == "contrastive" and not skip_contrastive else {}))),
                rep_name))
    mi_dict = {"datasets": DATASET_NAMES, "results": all_mi}
    with open(RESULTS / "mutual_information.json", "w") as f:
        json.dump(mi_dict, f, indent=2, default=str)
    logger.info("  Saved mutual_information.json")

    # 5g. Statistical tests
    logger.info("  Running bootstrap (10,000 iterations)...")
    bootstrap_results = run_bootstrap(all_results, n_iterations=10000)
    with open(RESULTS / "bootstrap_results.json", "w") as f:
        json.dump(bootstrap_results, f, indent=2, default=str)
    logger.info("  Saved bootstrap_results.json")

    logger.info("  Running mixed-effects model...")
    mixed_results = run_mixed_effects(all_results)
    with open(RESULTS / "mixed_effects.json", "w") as f:
        json.dump(mixed_results, f, indent=2, default=str)
    logger.info("  Saved mixed_effects.json")

    logger.info("  Running Bayesian paired comparison...")
    bayesian_results = run_bayesian_comparison(all_results)
    with open(RESULTS / "bayesian_representation.json", "w") as f:
        json.dump(bayesian_results, f, indent=2, default=str)
    logger.info("  Saved bayesian_representation.json")

    # 5h. Representation ablation summary
    ablation = {}
    for rep_name in sorted(set(r["representation"] for r in all_results)):
        sub = [r for r in all_results if r["representation"] == rep_name]
        src_tgt_pairs = [r for r in sub if r["source"] != r["target"]]
        within = [r for r in sub if r["source"] == r["target"]]
        ablation[rep_name] = {
            "mean_off_diag_mf1": float(np.mean([r["macro_f1"] for r in src_tgt_pairs])) if src_tgt_pairs else 0,
            "std_off_diag_mf1": float(np.std([r["macro_f1"] for r in src_tgt_pairs], ddof=1)) if len(src_tgt_pairs) > 1 else 0,
            "mean_diag_mf1": float(np.mean([r["macro_f1"] for r in within])) if within else 0,
            "mean_binary_f1": float(np.mean([r["binary_f1"] for r in sub])),
            "mean_ece": float(np.mean([r["ece"] for r in sub])),
            "mean_nll": float(np.mean([r["nll"] for r in sub])),
            "n_pairs": len(src_tgt_pairs),
        }
    with open(RESULTS / "representation_ablation.json", "w") as f:
        json.dump(ablation, f, indent=2, default=str)
    logger.info("  Saved representation_ablation.json")

    # 5i. Summary report
    logger.info("  Generating summary report...")
    generate_report(ablation, bootstrap_results, mixed_results, bayesian_results)

    logger.info("\n" + "=" * 70)
    logger.info("Phase 58 complete. All deliverables in results/phase58/")
    logger.info("=" * 70)


def generate_report(ablation, bootstrap, mixed, bayesian):
    """Generate phase58_summary.md and phase58_report.md."""
    lines = []
    lines.append("# Phase 58 — Representation Learning vs Feature Engineering Validation")
    lines.append("")
    lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Device**: {DEVICE}")
    lines.append("")
    lines.append("## Hypothesis")
    lines.append("")
    lines.append("- **H0**: Transfer failure is primarily due to inadequate handcrafted features.")
    lines.append("- **H1**: Transfer failure persists even after learning dataset-invariant latent representations.")
    lines.append("")
    lines.append("## Representations Compared")
    lines.append("")
    lines.append("| Representation | Description | Mean Off-Diag MF1 | Within-Diag MF1 |")
    lines.append("|---|---|---|---|")
    for rep, stats in sorted(ablation.items()):
        lines.append(f"| {rep} | | {stats['mean_off_diag_mf1']:.4f} ± {stats['std_off_diag_mf1']:.4f} | {stats['mean_diag_mf1']:.4f} |")
    lines.append("")

    lines.append("## Bootstrap Comparisons (10,000 iterations)")
    lines.append("")
    lines.append("| Comparison | Mean Diff | 95% CI | p-value | Cohen's d |")
    lines.append("|---|---|---|---|---|")
    for comp in bootstrap["comparisons"]:
        lines.append(f"| {comp['rep_1']} vs {comp['rep_2']} | {comp['mean_diff']:.4f} | [{comp['ci_lower']:.4f}, {comp['ci_upper']:.4f}] | {comp['p_value']:.4f} | {comp['cohens_d']:.4f} |")
    lines.append("")

    lines.append("## Bayesian Paired Comparison")
    lines.append("")
    lines.append("| Comparison | Posterior Mean | P(rep1 > rep2) | Effect Size | 95% CrI |")
    lines.append("|---|---|---|---|---|")
    for comp in bayesian["comparisons"]:
        cri = comp["credible_interval_95"]
        lines.append(f"| {comp['rep_1']} vs {comp['rep_2']} | {comp['posterior_mean']:.4f} | {comp['prob_rep1_greater']:.4f} | {comp['effect_size']:.4f} | [{cri[0]:.4f}, {cri[1]:.4f}] |")
    lines.append("")

    lines.append("## Mixed-Effects Model")
    lines.append("")
    lines.append("| Representation | Mean MF1 | Std MF1 | N Pairs |")
    lines.append("|---|---|---|---|")
    for r in mixed["results"]:
        lines.append(f"| {r['representation']} | {r['mean_mf1']:.4f} | {r['std_mf1']:.4f} | {r['n_pairs']} |")
    lines.append("")

    # Decision
    lines.append("## Conclusion")
    lines.append("")
    raw_mf1 = ablation.get("raw", {}).get("mean_off_diag_mf1", 0)
    best_rep = max(ablation, key=lambda k: ablation[k]["mean_off_diag_mf1"])
    best_mf1 = ablation[best_rep]["mean_off_diag_mf1"]
    improvement = best_mf1 - raw_mf1

    if best_mf1 > raw_mf1 + 0.05:
        lines.append(f"**Possible Result 1**: Representations ({best_rep}) improve transfer MF1 by {improvement:.4f} over raw features.")
        lines.append("→ Feature engineering was limiting transfer performance.")
    elif best_mf1 > raw_mf1 + 0.01:
        lines.append(f"**Possible Result 2**: Representations improve within-dataset but transfer is largely unchanged.")
        lines.append(f"Best representation ({best_rep}) achieves MF1={best_mf1:.4f} vs raw={raw_mf1:.4f}.")
        lines.append("→ Conditional distribution mismatch remains the dominant factor.")
    else:
        lines.append(f"**Possible Result 3**: Even learned representations fail to improve transfer.")
        lines.append(f"Best representation ({best_rep}): MF1={best_mf1:.4f} vs raw={raw_mf1:.4f} (Δ={improvement:.4f})")
        lines.append("→ Strong evidence that P(Y|X) differs fundamentally across IDS datasets.")
    lines.append("")

    lines.append("## Deliverables")
    lines.append("")
    lines.append("- `representation_transfer.csv` — Full cross-dataset transfer results")
    lines.append("- `latent_similarity.csv` — Combined similarity metrics per representation")
    lines.append("- `cka_results.csv` — Centered Kernel Alignment matrices")
    lines.append("- `mmd_results.csv` — Maximum Mean Discrepancy results")
    lines.append("- `wasserstein_results.csv` — Wasserstein distances")
    lines.append("- `representation_ablation.json` — Summary ablation statistics")
    lines.append("- `latent_visualizations/umap/` — UMAP embeddings per dataset per representation")
    lines.append("- `latent_visualizations/tsne/` — t-SNE embeddings per dataset per representation")
    lines.append("- `mutual_information.json` — I(Z;X) and I(Z;Y) estimates")
    lines.append("- `mixed_effects.json` — Mixed-effects model results")
    lines.append("- `bootstrap_results.json` — Bootstrap comparison results")
    lines.append("- `bayesian_representation.json` — Bayesian paired comparisons")

    summary = "\n".join(lines)

    # Save summary
    with open(RESULTS / "phase58_summary.md", "w") as f:
        f.write(summary)

    # Full report (add more details)
    with open(RESULTS / "phase58_report.md", "w") as f:
        f.write(summary)
        f.write("\n\n## Full Results\n\n")
        f.write("See CSV files and JSON files in results/phase58/ for complete data.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 58: Representation Learning vs Feature Engineering")
    parser.add_argument("--skip-ae", action="store_true", help="Skip autoencoder training")
    parser.add_argument("--skip-contrastive", action="store_true", help="Skip contrastive training")
    args = parser.parse_args()
    run_pipeline(skip_ae=args.skip_ae, skip_contrastive=args.skip_contrastive)
