#!/usr/bin/env python3
"""Task 5 — Domain Generalization Audit (leave-one-dataset-out)."""
import sys, json, logging, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
for p in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("domain_audit")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
EPOCHS = 100
PATIENCE = 20
BATCH_SIZE = 512
LR = 5e-4
LAMBDA_DOMAIN = 0.5

NUM_FEATURES = 17
NUM_CLASSES = 7
NUM_DATASETS = 3
CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R", "Generic", "Backdoor"]
DATASET_NAMES = ["nsl_kdd", "unsw_nb15", "cicids2018"]
DATASET_TO_ID = {n: i for i, n in enumerate(DATASET_NAMES)}

from scripts.training.train_dann_production import (
    load_all_datasets, create_splits, DANNHelixModel,
)

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def gradient_reversal(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)

def train_dann(X_train, y_train, domain_train, X_val, y_val, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DANNHelixModel(NUM_FEATURES, NUM_CLASSES, NUM_DATASETS).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    src_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor((y_train > 0).astype(np.int64), dtype=torch.long),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(domain_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor((y_val > 0).astype(np.int64), dtype=torch.long),
        torch.tensor(y_val, dtype=torch.long),
    )
    src_loader = DataLoader(src_ds, BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=0)
    best_val_f1 = 0.0
    patience_counter = 0
    for epoch in range(EPOCHS):
        model.train()
        for batch in src_loader:
            x_b, yb, yf, d = [t.to(DEVICE) for t in batch]
            bl, fl, dl = model(x_b, LAMBDA_DOMAIN)
            loss_cls = F.cross_entropy(bl, yb) + F.cross_entropy(fl, yf)
            loss_dom = F.cross_entropy(dl, d)
            loss = loss_cls + LAMBDA_DOMAIN * loss_dom
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x_b, _, yf in val_loader:
                bl, fl, _ = model(x_b.to(DEVICE), 0.0)
                preds.extend(fl.argmax(dim=1).cpu().numpy().tolist())
                targets.extend(yf.numpy().tolist())
        val_f1 = f1_score(targets, preds, average="macro", zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            break
    if best_state:
        model.load_state_dict(best_state)
    return model

def evaluate(model, X_test, y_test):
    model.eval()
    ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    loader = DataLoader(ds, BATCH_SIZE, shuffle=False)
    preds, targets = [], []
    with torch.no_grad():
        for x_b, y_b in loader:
            _, fl, _ = model(x_b.to(DEVICE), 0.0)
            preds.extend(fl.argmax(dim=1).cpu().numpy().tolist())
            targets.extend(y_b.numpy().tolist())
    y_true = np.array(targets)
    y_pred = np.array(preds)
    y_bin_true = (y_true > 0).astype(np.int64)
    y_bin_pred = (y_pred > 0).astype(np.int64)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bin_f1 = f1_score(y_bin_true, y_bin_pred, average="binary", zero_division=0)
    try:
        y_true_onehot = np.eye(NUM_CLASSES)[y_true]
        probs = torch.cat([
            F.softmax(model(torch.tensor(X_test[i:i+BATCH_SIZE], dtype=torch.float32).to(DEVICE), 0.0)[1], dim=1).cpu()
            for i in range(0, len(X_test), BATCH_SIZE)
        ]).numpy()
        roc_auc = roc_auc_score(y_true_onehot, probs, multi_class="ovr")
    except Exception:
        roc_auc = 0.0
    return {"accuracy": float(acc), "macro_f1": float(macro_f1), "binary_f1": float(bin_f1),
            "roc_auc": float(roc_auc), "n_test": len(y_true)}

def main():
    os.makedirs("docs/phase30", exist_ok=True)
    logger.info("Loading datasets (with 500k CICIDS cap)...")
    harmonized = load_all_datasets()
    seeds = [42, 1337, 2026]
    results = {}
    for held_out in DATASET_NAMES:
        logger.info(f"\n{'='*60}")
        logger.info(f"HOLDING OUT: {held_out}")
        logger.info(f"{'='*60}")
        train_splits = {}
        for ds_name in DATASET_NAMES:
            if ds_name == held_out or ds_name not in harmonized:
                continue
            X, y = harmonized[ds_name]
            rng = np.random.RandomState(42)
            n = len(X)
            n_test = int(n * 0.2)
            idx = np.arange(n)
            rng.shuffle(idx)
            train_splits[ds_name] = (
                X[idx[n_test:]], y[idx[n_test:]],
                X[idx[:n_test]], y[idx[:n_test]],
                X[idx[:n_test]], y[idx[:n_test]],
            )
        X_tr_list, y_tr_list, d_tr_list = [], [], []
        X_val_list, y_val_list = [], []
        for ds_name, ds in train_splits.items():
            did = DATASET_TO_ID[ds_name]
            X_tr_list.append(ds[0]); y_tr_list.append(ds[1])
            d_tr_list.append(np.full(len(ds[0]), did, dtype=np.int64))
            X_val_list.append(ds[2]); y_val_list.append(ds[3])
        X_train = np.vstack(X_tr_list).astype(np.float32)
        y_train = np.concatenate(y_tr_list)
        d_train = np.concatenate(d_tr_list)
        X_val = np.vstack(X_val_list).astype(np.float32)
        y_val = np.concatenate(y_val_list)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)
        logger.info(f"  Train: {len(X_train)}, Val: {len(X_val)}")
        # Test on held-out dataset
        test_ds = train_splits.get(held_out, harmonized.get(held_out))
        if test_ds is None:
            logger.warning(f"  No test data for {held_out}")
            continue
        X_test = harmonized[held_out][0]
        y_test = harmonized[held_out][1]
        X_test = scaler.transform(X_test).astype(np.float32)
        logger.info(f"  Test ({held_out}): {len(X_test)}")
        for seed in seeds:
            logger.info(f"  Training with seed {seed}...")
            model = train_dann(X_train, y_train, d_train, X_val, y_val, seed=seed)
            eval_metrics = evaluate(model, X_test, y_test)
            logger.info(f"  Seed {seed}: Acc={eval_metrics['accuracy']:.4f}, MF1={eval_metrics['macro_f1']:.4f}, BinF1={eval_metrics['binary_f1']:.4f}, AUC={eval_metrics['roc_auc']:.4f}")
            results.setdefault(held_out, {})[seed] = eval_metrics
            del model
            if DEVICE.type == "mps":
                torch.mps.empty_cache()
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DOMAIN GENERALIZATION RESULTS")
    logger.info("=" * 60)
    lines = []
    for held_out in DATASET_NAMES:
        if held_out not in results:
            continue
        lines.append(f"\n### Test on {held_out.upper()}")
        mf1s = [r["macro_f1"] for r in results[held_out].values()]
        bf1s = [r["binary_f1"] for r in results[held_out].values()]
        aucs = [r["roc_auc"] for r in results[held_out].values()]
        lines.append(f"  Train: {', '.join(d for d in DATASET_NAMES if d != held_out)}")
        lines.append(f"  Macro F1: {np.mean(mf1s):.4f} ± {np.std(mf1s, ddof=1):.4f}")
        lines.append(f"  Binary F1: {np.mean(bf1s):.4f} ± {np.std(bf1s, ddof=1):.4f}")
        lines.append(f"  ROC-AUC: {np.mean(aucs):.4f} ± {np.std(aucs, ddof=1):.4f}")
        lines.append(f"  Seeds: {list(results[held_out].keys())}")
    report = "\n".join([
        "# Phase 30 — Strict Domain Generalization Audit\n",
        f"**Device**: {DEVICE}\n",
        "**Protocol**: Leave-one-dataset-out cross-validation\n",
        f"**Epochs**: {EPOCHS}, **Patience**: {PATIENCE}\n\n",
        "## Results\n",
        *[l + "\n" for l in lines],
        "\n## Comparison\n\n",
        f"| Test Set | Macro F1 (μ±σ) | Binary F1 (μ±σ) | ROC-AUC (μ±σ) | Phase 29 (in-dist) |\n",
        f"|----------|:------------:|:--------------:|:-------------:|:-----------------:|\n",
    ])
    for held_out in DATASET_NAMES:
        if held_out not in results:
            continue
        mf1s = [r["macro_f1"] for r in results[held_out].values()]
        bf1s = [r["binary_f1"] for r in results[held_out].values()]
        aucs = [r["roc_auc"] for r in results[held_out].values()]
        report += f"| {held_out.upper()} | {np.mean(mf1s):.4f}±{np.std(mf1s, ddof=1):.4f} | {np.mean(bf1s):.4f}±{np.std(bf1s, ddof=1):.4f} | {np.mean(aucs):.4f}±{np.std(aucs, ddof=1):.4f} | 0.5757±0.0034 |\n"
    report += "\n---\n*Phase 30 Audit — Domain Generalization*\n"
    Path("docs/phase30/STRICT_DOMAIN_GENERALIZATION.md").write_text(report)
    logger.info(f"Report saved to docs/phase30/STRICT_DOMAIN_GENERALIZATION.md")
    print(report)

if __name__ == "__main__":
    main()
