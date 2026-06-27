#!/usr/bin/env python3
"""
Phase 52 — Generalization and Ablation Study.

Validates whether Phase 50 SupCon transfer gains are architecture-independent.

Usage:
  python scripts/phase52_experiments.py [--experiment A] [--epochs 50]
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "phase52"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "phase52_cache"

DATASETS = ["nsl_kdd", "unsw_nb15", "cicids2018", "cicids2017", "bot_iot", "ton_iot"]

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ============================================================================
# Data Loading
# ============================================================================

def load_cached_data(max_samples=15000):
    result = {}
    for ds_name in DATASETS:
        paths = {k: CACHE_DIR / f"{ds_name}_{k}.npy" for k in
                 ["X_train", "y_train", "X_test", "y_test"]}
        if not all(p.exists() for p in paths.values()):
            print(f"  SKIP {ds_name}: cache not found")
            continue
        X_train = np.load(paths["X_train"])
        y_train = np.load(paths["y_train"], allow_pickle=True)
        X_test = np.load(paths["X_test"])
        y_test = np.load(paths["y_test"], allow_pickle=True)
        for tag, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
            if len(X) > max_samples:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(X), max_samples, replace=False)
                if tag == "train":
                    X_train = X_train[idx]; y_train = y_train[idx]
                else:
                    X_test = X_test[idx]; y_test = y_test[idx]
        result[ds_name] = {"X_train": X_train, "y_train": y_train,
                           "X_test": X_test, "y_test": y_test}
        print(f"  {ds_name}: train={X_train.shape[0]}, test={X_test.shape[0]}")
    return result


# ============================================================================
# Model
# ============================================================================

class SupConEncoder(nn.Module):
    def __init__(self, input_dim=17, hidden_dims=None, latent_dim=32,
                 use_bn=True, dropout=0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_bn and h > 1:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class SupConModel(nn.Module):
    def __init__(self, input_dim=17, hidden_dims=None, latent_dim=32,
                 num_classes=7, temperature=0.07, supcon_weight=1.0):
        super().__init__()
        self.supcon_weight = supcon_weight
        self.temperature = temperature
        self.encoder = SupConEncoder(input_dim, hidden_dims, latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        z = self.encoder(x)
        return z, self.classifier(z)


def supcon_loss(features, labels, temperature=0.07):
    B = features.shape[0]
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(features.device)
    sim = torch.mm(features, features.T) / temperature
    diag = torch.eye(B, device=features.device).bool()
    sim = sim.masked_fill(diag, -1e9)
    pos = (torch.exp(sim) * mask).masked_fill(diag, 0)
    neg = torch.exp(sim) * (1 - mask)
    pos_sum = pos.sum(1)
    neg_sum = neg.sum(1)
    denom = pos_sum + neg_sum + 1e-8
    log_prob = torch.log(pos_sum.clamp(min=1e-8) / denom)
    n_pos = pos.sum(1)
    valid = n_pos > 0
    loss = torch.where(valid, -log_prob / (n_pos + 1e-8), torch.zeros_like(log_prob))
    return loss.mean()


# ============================================================================
# Training
# ============================================================================

def train_supcon_model(model, X_train, y_train, X_val, y_val,
                       epochs=25, batch_size=256, lr=1e-3, wd=1e-4,
                       label_noise=0.0, sample_frac=1.0, verbose=True):
    if sample_frac < 1.0:
        n_keep = max(int(len(X_train) * sample_frac), 100)
        idx = np.random.default_rng(42).choice(len(X_train), n_keep, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    if label_noise > 0:
        rng = np.random.default_rng(42)
        n_noise = int(len(y_train) * label_noise)
        idx = rng.choice(len(y_train), n_noise, replace=False)
        y_noisy = y_train.copy()
        classes = np.unique(y_train)
        for i in idx:
            alt = [c for c in classes if c != y_train[i]]
            if alt:
                y_noisy[i] = rng.choice(alt)
        y_train = y_noisy

    loader = DataLoader(TensorDataset(torch.FloatTensor(X_train),
                                       torch.LongTensor(y_train)),
                        batch_size=batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_mf1, patience = 0.0, 10
    X_v_t = torch.FloatTensor(X_val).to(DEVICE)
    y_v_t = torch.LongTensor(y_val).to(DEVICE)

    for ep in range(epochs):
        model.train()
        loss_sum = 0.0
        nb = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            z, logits = model(xb)
            ce = F.cross_entropy(logits, yb)
            loss = ce
            if model.supcon_weight > 0:
                loss = ce + model.supcon_weight * supcon_loss(z, yb, model.temperature)
            if torch.isnan(loss) or torch.isinf(loss):
                return {"train_loss": [float('nan')], "val_mf1": [0.0]}
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item()
            nb += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            preds = model(X_v_t)[1].argmax(1).cpu().numpy()
            val_mf1 = f1_score(y_val, preds, average="macro", zero_division=0)
        if val_mf1 > best_mf1:
            best_mf1 = val_mf1; patience = 10
        else:
            patience -= 1
        if verbose and (ep + 1) % 10 == 0:
            print(f"  ep {ep+1}: loss={loss_sum/max(nb,1):.4f} val_mf1={val_mf1:.4f}")
        if patience <= 0:
            break
    return {"train_loss": [loss_sum / max(nb, 1)], "val_mf1": [best_mf1]}


def evaluate_transfer(model, X_train, y_train, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_tr_f = model.encoder(torch.FloatTensor(X_train).to(DEVICE)).cpu().numpy()
        X_te_f = model.encoder(torch.FloatTensor(X_test).to(DEVICE)).cpu().numpy()
    lr = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    lr.fit(X_tr_f, y_train)
    y_pred = lr.predict(X_te_f)
    return {
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }


def compute_all_transfer(model, all_data):
    rows = []
    for src in DATASETS:
        if src not in all_data:
            continue
        s = all_data[src]
        for tgt in DATASETS:
            if tgt not in all_data:
                continue
            m = evaluate_transfer(model, s["X_train"], s["y_train"],
                                  all_data[tgt]["X_test"], all_data[tgt]["y_test"])
            rows.append({"source": src, "target": tgt, **m})
    return pd.DataFrame(rows)


def make_model(**kw):
    d = dict(input_dim=17, hidden_dims=[64], latent_dim=32,
             num_classes=7, temperature=0.07, supcon_weight=1.0)
    d.update(kw)
    return SupConModel(**d).to(DEVICE)


# ============================================================================
# Experiments
# ============================================================================

def safe_split(X, y):
    """Stratified split with fallback for tiny classes."""
    if pd.Series(y).value_counts().min() >= 2:
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def run_experiment_a(all_data, epochs):
    """Latent dimension ablation."""
    print(f"\n{'='*70}\nEXPERIMENT A — Latent Dimension Ablation\n{'='*70}")
    dims = [2, 8, 16, 32, 64, 128]
    all_rows = []
    for ld in dims:
        print(f"\n  latent_dim = {ld}")
        offs = []
        for src_name in DATASETS:
            if src_name not in all_data:
                continue
            src = all_data[src_name]
            model = make_model(latent_dim=ld)
            X_tr, X_v, y_tr, y_v = safe_split(src["X_train"], src["y_train"])
            train_supcon_model(model, X_tr, y_tr, X_v, y_v, epochs=epochs,
                               verbose=(ld == 32))
            tdf = compute_all_transfer(model, all_data)
            for _, r in tdf.iterrows():
                all_rows.append({"latent_dim": ld, "experiment": "A_latent_dim",
                                 **r.to_dict()})
            offs.extend(r["macro_f1"] for _, r in tdf.iterrows()
                        if r["source"] != r["target"])
        if offs:
            print(f"    mean off-diag MF1 = {np.mean(offs):.4f} ± {np.std(offs):.4f}")
    pd.DataFrame(all_rows).to_csv(RESULTS_DIR / "latent_dimension_results.csv", index=False)


def run_experiment_b(all_data, epochs):
    """Encoder depth ablation."""
    print(f"\n{'='*70}\nEXPERIMENT B — Encoder Depth Ablation\n{'='*70}")
    archs = [("17->32->2", [32], 2), ("17->64->2", [64], 2),
             ("17->64->128->2", [64, 128], 2), ("17->128->256->2", [128, 256], 2)]
    all_rows = []
    for aname, hdims, ld in archs:
        print(f"\n  {aname}")
        offs = []
        for src_name in DATASETS:
            if src_name not in all_data:
                continue
            src = all_data[src_name]
            model = make_model(hidden_dims=hdims, latent_dim=ld)
            X_tr, X_v, y_tr, y_v = safe_split(src["X_train"], src["y_train"])
            train_supcon_model(model, X_tr, y_tr, X_v, y_v, epochs=epochs,
                               verbose=(aname == "17->64->2"))
            tdf = compute_all_transfer(model, all_data)
            for _, r in tdf.iterrows():
                all_rows.append({"architecture": aname, "hidden_dims": str(hdims),
                                 "latent_dim": ld, "experiment": "B_architecture",
                                 **r.to_dict()})
            offs.extend(r["macro_f1"] for _, r in tdf.iterrows()
                        if r["source"] != r["target"])
        if offs:
            print(f"    mean off-diag MF1 = {np.mean(offs):.4f} ± {np.std(offs):.4f}")
    pd.DataFrame(all_rows).to_csv(RESULTS_DIR / "architecture_ablation.csv", index=False)


def run_experiment_c(all_data, epochs):
    """Temperature sweep."""
    print(f"\n{'='*70}\nEXPERIMENT C — SupCon Temperature Sweep\n{'='*70}")
    temps = [0.03, 0.05, 0.07, 0.10, 0.20, 0.50]
    all_rows = []
    for temp in temps:
        print(f"\n  temperature = {temp}")
        offs = []
        for src_name in DATASETS:
            if src_name not in all_data:
                continue
            src = all_data[src_name]
            model = make_model(temperature=temp)
            X_tr, X_v, y_tr, y_v = safe_split(src["X_train"], src["y_train"])
            train_supcon_model(model, X_tr, y_tr, X_v, y_v, epochs=epochs,
                               verbose=(temp == 0.07))
            tdf = compute_all_transfer(model, all_data)
            for _, r in tdf.iterrows():
                all_rows.append({"temperature": temp, "experiment": "C_temperature",
                                 **r.to_dict()})
            offs.extend(r["macro_f1"] for _, r in tdf.iterrows()
                        if r["source"] != r["target"])
        if offs:
            print(f"    mean off-diag MF1 = {np.mean(offs):.4f} ± {np.std(offs):.4f}")
    pd.DataFrame(all_rows).to_csv(RESULTS_DIR / "temperature_sweep.csv", index=False)


def run_experiment_d(all_data, epochs):
    """Loss weight ablation."""
    print(f"\n{'='*70}\nEXPERIMENT D — Loss Weight Ablation\n{'='*70}")
    weights = [0.0, 0.25, 0.50, 1.0, 2.0]
    all_rows = []
    for sw in weights:
        label = "CE_only" if sw == 0.0 else f"CE+{sw:.2f}SupCon"
        print(f"\n  supcon_weight = {sw} ({label})")
        offs = []
        for src_name in DATASETS:
            if src_name not in all_data:
                continue
            src = all_data[src_name]
            model = make_model(supcon_weight=sw)
            X_tr, X_v, y_tr, y_v = safe_split(src["X_train"], src["y_train"])
            train_supcon_model(model, X_tr, y_tr, X_v, y_v, epochs=epochs,
                               verbose=(sw == 1.0))
            tdf = compute_all_transfer(model, all_data)
            for _, r in tdf.iterrows():
                all_rows.append({"supcon_weight": sw, "experiment": "D_loss_weight",
                                 **r.to_dict()})
            offs.extend(r["macro_f1"] for _, r in tdf.iterrows()
                        if r["source"] != r["target"])
        if offs:
            print(f"    mean off-diag MF1 = {np.mean(offs):.4f} ± {np.std(offs):.4f}")
    pd.DataFrame(all_rows).to_csv(RESULTS_DIR / "loss_weight_ablation.csv", index=False)


def run_experiment_e(all_data, epochs):
    """Label noise robustness."""
    print(f"\n{'='*70}\nEXPERIMENT E — Label Noise Robustness\n{'='*70}")
    noises = [0.0, 0.05, 0.10, 0.20, 0.30]
    all_rows = []
    for noise in noises:
        print(f"\n  label_noise = {noise*100:.0f}%")
        offs, selfs = [], []
        for src_name in DATASETS:
            if src_name not in all_data:
                continue
            src = all_data[src_name]
            model = make_model()
            X_tr, X_v, y_tr, y_v = safe_split(src["X_train"], src["y_train"])
            train_supcon_model(model, X_tr, y_tr, X_v, y_v, epochs=epochs,
                               label_noise=noise, verbose=(noise == 0.0))
            tdf = compute_all_transfer(model, all_data)
            for _, r in tdf.iterrows():
                all_rows.append({"label_noise": noise, "experiment": "E_label_noise",
                                 **r.to_dict()})
                if r["source"] == r["target"]:
                    selfs.append(r["macro_f1"])
                else:
                    offs.append(r["macro_f1"])
        if offs:
            print(f"    off-diag MF1 = {np.mean(offs):.4f} self MF1 = {np.mean(selfs):.4f}")
    pd.DataFrame(all_rows).to_csv(RESULTS_DIR / "label_noise_results.csv", index=False)


def run_experiment_f(all_data, epochs):
    """Sample efficiency."""
    print(f"\n{'='*70}\nEXPERIMENT F — Sample Efficiency\n{'='*70}")
    fracs = [0.10, 0.25, 0.50, 0.75, 1.0]
    all_rows = []
    for frac in fracs:
        print(f"\n  sample_fraction = {frac*100:.0f}%")
        offs = []
        for src_name in DATASETS:
            if src_name not in all_data:
                continue
            src = all_data[src_name]
            model = make_model()
            X_tr, X_v, y_tr, y_v = safe_split(src["X_train"], src["y_train"])
            train_supcon_model(model, X_tr, y_tr, X_v, y_v, epochs=epochs,
                               sample_frac=frac, verbose=(frac == 1.0))
            tdf = compute_all_transfer(model, all_data)
            for _, r in tdf.iterrows():
                all_rows.append({"sample_fraction": frac, "experiment": "F_sample_efficiency",
                                 **r.to_dict()})
            offs.extend(r["macro_f1"] for _, r in tdf.iterrows()
                        if r["source"] != r["target"])
        if offs:
            print(f"    mean off-diag MF1 = {np.mean(offs):.4f} ± {np.std(offs):.4f}")
    pd.DataFrame(all_rows).to_csv(RESULTS_DIR / "sample_efficiency.csv", index=False)


# ============================================================================
# Statistical Analysis
# ============================================================================

def compute_statistics(result_paths):
    import scipy.stats as stats
    res = {}
    for ename, csv_path in result_paths.items():
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if "macro_f1" not in df.columns:
            continue
        pcol = next((c for c in ["latent_dim", "temperature", "supcon_weight",
                                  "label_noise", "sample_fraction", "architecture"]
                     if c in df.columns), None)
        if not pcol:
            continue
        print(f"\n{'='*60}\nStats: {ename}\n{'='*60}")
        ed = {"groups": {}, "pairwise": [], "global": {}}
        groups = {}
        for name, grp in df.groupby(pcol):
            vals = grp["macro_f1"].dropna().values
            if len(vals) <= 1:
                continue
            groups[str(name)] = vals
            rng = np.random.default_rng(42)
            boot = np.array([vals[rng.choice(len(vals), len(vals), replace=True)].mean()
                             for _ in range(2000)])
            ci = np.percentile(boot, [2.5, 97.5])
            ed["groups"][str(name)] = {"mean": float(vals.mean()), "std": float(vals.std()),
                                        "n": int(len(vals)), "ci_95_low": float(ci[0]),
                                        "ci_95_high": float(ci[1])}
            print(f"  {pcol}={name}: MF1={vals.mean():.4f}±{vals.std():.4f} "
                  f"95%CI=[{ci[0]:.4f},{ci[1]:.4f}] n={len(vals)}")

        gnames = sorted(groups.keys(), key=lambda x: groups[x].mean(), reverse=True)
        if len(gnames) >= 3:
            min_n = min(len(groups[g]) for g in gnames)
            if min_n >= 2:
                try:
                    f_stat, f_p = stats.friedmanchisquare(*[groups[g][:min_n] for g in gnames])
                    ed["global"]["friedman_stat"] = float(f_stat)
                    ed["global"]["friedman_p"] = float(f_p)
                    print(f"  Friedman: χ²={f_stat:.4f}, p={f_p:.6f}")
                except Exception:
                    pass

        best_g = gnames[0]
        print(f"  Pairwise vs {best_g}:")
        pairwise = []
        for g in gnames[1:]:
            a, b = groups[best_g], groups[g]
            ml = min(len(a), len(b))
            if ml < 2:
                continue
            sp = np.sqrt((a.std()**2 + b.std()**2) / 2 + 1e-10)
            cd = abs(a.mean() - b.mean()) / sp
            n_ab = min(len(a), 200) * min(len(b), 200)
            cl = sum((1 if va > vb else -1 if vb > va else 0)
                     for va in a[:200] for vb in b[:200]) / max(n_ab, 1)
            try:
                _, w_p = stats.wilcoxon(a[:ml], b[:ml])
            except Exception:
                w_p = 1.0
            try:
                _, t_p = stats.ttest_rel(a[:ml], b[:ml])
            except Exception:
                t_p = 1.0
            pairwise.append({"group1": best_g, "group2": g, "cohens_d": float(cd),
                              "cliffs_delta": float(cl), "wilcoxon_p": float(w_p),
                              "ttest_p": float(t_p)})
            print(f"    {best_g} vs {g}: d={cd:.3f}, δ={cl:.3f}, W_p={w_p:.4f}, t_p={t_p:.4f}")

        ed["pairwise"] = pairwise
        if pairwise:
            p_vals = [p["wilcoxon_p"] for p in pairwise]
            sidx = np.argsort(p_vals)
            n_t = len(p_vals)
            holm = []
            for rk, idx in enumerate(sidx):
                adj = min(1.0, p_vals[idx] * (n_t - rk))
                holm.append({"comparison": f"{pairwise[idx]['group1']} vs {pairwise[idx]['group2']}",
                             "original_p": p_vals[idx], "holm_corrected_p": float(adj),
                             "reject_H0": adj < 0.05})
            ed["holm_bonferroni"] = holm
            print("  Holm-Bonferroni:")
            for h in holm:
                print(f"    {h['comparison']}: p_adj={h['holm_corrected_p']:.4f} reject={h['reject_H0']}")
        res[ename] = ed
    return res


def compute_rankings(result_paths):
    rows = []
    for ename, csv_path in result_paths.items():
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if "macro_f1" not in df.columns:
            continue
        pcol = next((c for c in ["latent_dim", "temperature", "supcon_weight",
                                  "label_noise", "sample_fraction", "architecture"]
                     if c in df.columns), None)
        if not pcol:
            continue
        for name, grp in df.groupby(pcol):
            rows.append({"experiment": ename, "parameter": pcol,
                         "configuration": str(name),
                         "mean_macro_f1": grp["macro_f1"].mean(),
                         "std_macro_f1": grp["macro_f1"].std(),
                         "n": len(grp)})
    rdf = pd.DataFrame(rows).sort_values("mean_macro_f1", ascending=False)
    rdf.to_csv(RESULTS_DIR / "ranking_tables.csv", index=False)
    print(f"\nRankings:\n{rdf.to_string(index=False)}")
    return rdf


# ============================================================================
# Main
# ============================================================================

def run_stats():
    rfs = {k: RESULTS_DIR / v for k, v in [
        ("A_latent_dim", "latent_dimension_results.csv"),
        ("B_architecture", "architecture_ablation.csv"),
        ("C_temperature", "temperature_sweep.csv"),
        ("D_loss_weight", "loss_weight_ablation.csv"),
        ("E_label_noise", "label_noise_results.csv"),
        ("F_sample_efficiency", "sample_efficiency.csv"),
    ]}
    print("\n" + "=" * 60)
    print("Statistical analysis...")
    print("=" * 60)
    stats = compute_statistics(rfs)
    with open(RESULTS_DIR / "statistical_tests.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
    compute_rankings(rfs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=None,
                        choices=["A", "B", "C", "D", "E", "F", "all", "stats"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--max-samples", type=int, default=15000)
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()

    if args.stats_only:
        return run_stats()

    print(f"Phase 52: {args.epochs} epochs, {args.max_samples} samples/dataset")
    t0 = time.time()

    all_data = load_cached_data(max_samples=args.max_samples)
    if not all_data:
        print("No data loaded! Run scripts/phase52_preprocess.py first.")
        return
    print(f"Loaded {len(all_data)} datasets: {list(all_data.keys())}")

    if args.experiment in (None, "all", "A"):
        run_experiment_a(all_data, args.epochs)
    if args.experiment in (None, "all", "B"):
        run_experiment_b(all_data, args.epochs)
    if args.experiment in (None, "all", "C"):
        run_experiment_c(all_data, args.epochs)
    if args.experiment in (None, "all", "D"):
        run_experiment_d(all_data, args.epochs)
    if args.experiment in (None, "all", "E"):
        run_experiment_e(all_data, args.epochs)
    if args.experiment in (None, "all", "F"):
        run_experiment_f(all_data, args.epochs)

    run_stats()
    print(f"\nDone in {time.time()-t0:.0f}s — results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
