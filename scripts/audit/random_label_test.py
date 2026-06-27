#!/usr/bin/env python3
"""Task 6 — Random Label Sanity Check."""
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("random_label_test")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
EPOCHS = 80
PATIENCE = 15
BATCH_SIZE = 512
LR = 5e-4
LAMBDA_DOMAIN = 0.5

NUM_FEATURES = 17
NUM_CLASSES = 7
NUM_DATASETS = 3

from scripts.training.train_dann_production import (
    load_all_datasets, create_splits, combine_sources, DANNHelixModel,
)

def train_model(X_train, y_train, domain_train, X_val, y_val, seed=42, label="random"):
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
    src_loader = DataLoader(src_ds, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor((y_val > 0).astype(np.int64), dtype=torch.long),
            torch.tensor(y_val, dtype=torch.long),
        ), BATCH_SIZE, shuffle=False)
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
                preds.extend(fl.argmax(dim=1).cpu().tolist())
                targets.extend(yf.cpu().tolist())
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
    loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)), BATCH_SIZE, shuffle=False)
    preds, targets = [], []
    with torch.no_grad():
        for x_b, y_b in loader:
            _, fl, _ = model(x_b.to(DEVICE), 0.0)
            preds.extend(fl.argmax(dim=1).cpu().tolist())
            targets.extend(y_b.tolist())
    y_true, y_pred = np.array(targets), np.array(preds)
    y_bin = (y_true > 0).astype(np.int64)
    y_bin_p = (y_pred > 0).astype(np.int64)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bin_f1 = f1_score(y_bin, y_bin_p, average="binary", zero_division=0)
    try:
        y_onehot = np.eye(NUM_CLASSES)[y_true]
        probs = torch.cat([F.softmax(model(torch.tensor(X_test[i:i+BATCH_SIZE], dtype=torch.float32).to(DEVICE), 0.0)[1], dim=1).cpu() for i in range(0, len(X_test), BATCH_SIZE)]).numpy()
        roc_auc = roc_auc_score(y_onehot, probs, multi_class="ovr")
    except:
        roc_auc = 0.0
    return {"accuracy": float(acc), "macro_f1": float(macro_f1), "binary_f1": float(bin_f1), "roc_auc": float(roc_auc), "n_test": len(y_true)}

def main():
    os.makedirs("docs/phase30", exist_ok=True)
    harmonized = load_all_datasets()
    splits = create_splits(harmonized, seed=42)
    X_train, y_train, domain_train, X_val, y_val, X_test, y_test, scaler = combine_sources(splits, seed=42)
    
    logger.info(f"Original data: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    real_label_results = {}
    random_label_results = {}
    
    seeds = [42, 1337, 2026]
    CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R", "Generic", "Backdoor"]
    
    for seed in seeds:
        logger.info(f"\nSeed {seed}:")
        # Real labels
        logger.info("  Training with REAL labels...")
        model_real = train_model(X_train, y_train, domain_train, X_val, y_val, seed=seed, label="real")
        real_metrics = evaluate(model_real, X_test, y_test)
        logger.info(f"  REAL: MF1={real_metrics['macro_f1']:.4f}, BinF1={real_metrics['binary_f1']:.4f}, AUC={real_metrics['roc_auc']:.4f}")
        real_label_results[seed] = real_metrics
        del model_real; torch.mps.empty_cache() if DEVICE.type == "mps" else None
        
        # Random labels
        rng = np.random.RandomState(seed)
        y_train_permuted = y_train.copy()
        rng.shuffle(y_train_permuted)
        logger.info("  Training with PERMUTED labels...")
        model_random = train_model(X_train, y_train_permuted, domain_train, X_val, y_val, seed=seed, label="random")
        rand_metrics = evaluate(model_random, X_test, y_test)
        logger.info(f"  RANDOM: MF1={rand_metrics['macro_f1']:.4f}, BinF1={rand_metrics['binary_f1']:.4f}, AUC={rand_metrics['roc_auc']:.4f}")
        random_label_results[seed] = rand_metrics
        del model_random; torch.mps.empty_cache() if DEVICE.type == "mps" else None
    
    # Report
    lines = [
        "# Phase 30 — Random Label Sanity Check\n",
        f"**Device**: {DEVICE}\n\n",
        "## Protocol\n",
        "1. Train DANNHelixModel with REAL labels\n",
        "2. Train identical model with PERMUTED labels (random shuffle)\n",
        "3. If model achieves high performance with random labels: LEAKAGE EXISTS\n\n",
        "## Expected baseline (random labels)\n",
        "- Macro F1 ≈ 1/N_classes = 0.1429 (chance level)\n",
        "- ROC-AUC ≈ 0.5 (chance level)\n",
        "- Binary F1 ≈ Normal prevalence rate\n\n",
        "## Results\n\n",
        "| Seed | Real MF1 | Real AUC | Random MF1 | Random AUC | Leakage? |\n",
        "|-----:|--------:|--------:|----------:|----------:|--------:|\n",
    ]
    for seed in seeds:
        r = real_label_results[seed]
        p = random_label_results[seed]
        leak = "❌ YES" if p["macro_f1"] > 0.2 else "✅ NO"
        lines.append(f"| {seed} | {r['macro_f1']:.4f} | {r['roc_auc']:.4f} | {p['macro_f1']:.4f} | {p['roc_auc']:.4f} | {leak} |\n")
    lines.append("\n## Detailed Random Label Metrics\n\n")
    for seed in seeds:
        p = random_label_results[seed]
        lines.append(f"### Seed {seed}\n")
        lines.append(f"- Accuracy: {p['accuracy']:.4f}\n")
        lines.append(f"- Macro F1: {p['macro_f1']:.4f}\n")
        lines.append(f"- Binary F1: {p['binary_f1']:.4f}\n")
        lines.append(f"- ROC-AUC: {p['roc_auc']:.4f}\n")
    lines.append("\n## Verdict\n\n")
    rand_mf1s = [random_label_results[s]["macro_f1"] for s in seeds]
    avg_rand = np.mean(rand_mf1s)
    if avg_rand < 0.18:
        lines.append(f"✅ **PASS**: Random label Macro F1 ({avg_rand:.4f}) is near chance level (~0.143). No label leakage detected.\n")
    else:
        lines.append(f"❌ **FAIL**: Random label Macro F1 ({avg_rand:.4f}) exceeds chance level. Possible leakage.\n")
    Path("docs/phase30/RANDOM_LABEL_TEST.md").write_text("".join(lines))
    print("".join(lines))

if __name__ == "__main__":
    main()
