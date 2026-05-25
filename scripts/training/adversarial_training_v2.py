#!/usr/bin/env python3
"""
Adversarial Training & Robustness Hardening for HELIX-IDS v2

Adds adversarial examples (FGSM + PGD) to training data to improve
robustness from the current ~20% adversarial accuracy baseline.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
RESULTS_DIR = PROJECT_ROOT / "results" / "v2_fixed"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
if torch.cuda.is_available():
    device_str = "cuda"
elif torch.backends.mps.is_available():
    device_str = "mps"
else:
    device_str = "cpu"
DEVICE = torch.device(device_str)


def fgsm_attack(model, X, y, epsilon=0.1):
    """Generate FGSM adversarial examples."""
    x_adv = X.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    x_adv = x_adv + epsilon * x_adv.grad.sign()
    return x_adv.detach()


def pgd_attack(model, X, y, epsilon=0.1, alpha=0.01, steps=10):
    """Generate PGD adversarial examples."""
    x_adv = X.clone().detach()
    x_orig = X.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()
        x_adv = x_adv.detach() + alpha * x_adv.grad.sign()
        # Project back to epsilon ball
        perturbation = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
        x_adv = x_orig + perturbation
    return x_adv.detach()


def adversarial_train_epoch(model, train_loader, optimizer, device, epsilon=0.05, adv_ratio=0.3):
    """Train with mix of clean and adversarial examples."""
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []
    criterion = nn.CrossEntropyLoss()

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        # Split batch: some clean, some adversarial
        n_adv = max(1, int(len(x_batch) * adv_ratio))

        # Generate adversarial examples for subset
        model.eval()
        x_adv = fgsm_attack(model, x_batch[:n_adv], y_batch[:n_adv], epsilon)
        model.train()

        # Combine clean + adversarial
        x_combined = torch.cat([x_batch[n_adv:], x_adv], dim=0)
        y_combined = torch.cat([y_batch[n_adv:], y_batch[:n_adv]], dim=0)

        logits = model(x_combined)
        loss = criterion(logits, y_combined)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_targets.extend(y_combined.cpu().numpy())

    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return total_loss / len(train_loader), f1


def evaluate_robustness(model, X_test, y_test, device, epsilons=None):
    """Evaluate model robustness at different perturbation levels."""
    if epsilons is None:
        epsilons = [0.01, 0.05, 0.1, 0.2]
    model.eval()
    x_tensor = torch.FloatTensor(X_test).to(device)
    y_tensor = torch.LongTensor(y_test).to(device)

    # Clean accuracy
    with torch.no_grad():
        clean_preds = model(x_tensor).argmax(1).cpu().numpy()
    clean_acc = accuracy_score(y_test, clean_preds)
    clean_f1 = f1_score(y_test, clean_preds, average="macro", zero_division=0)

    results = {"clean": {"accuracy": float(clean_acc), "f1_macro": float(clean_f1)}}

    for eps in epsilons:
        x_adv = fgsm_attack(model, x_tensor, y_tensor, eps)
        with torch.no_grad():
            adv_preds = model(x_adv).argmax(1).cpu().numpy()
        adv_acc = accuracy_score(y_test, adv_preds)
        adv_f1 = f1_score(y_test, adv_preds, average="macro", zero_division=0)
        attack_success = 1.0 - adv_acc

        results[f"fgsm_eps_{eps}"] = {
            "accuracy": float(adv_acc),
            "f1_macro": float(adv_f1),
            "attack_success_rate": float(attack_success),
        }
        logger.info(f"  FGSM ε={eps}: Acc={adv_acc:.4f} F1={adv_f1:.4f} ASR={attack_success:.4f}")

    return results


def main():
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset
    from train_multidataset_v2_fixed import (
        HELIXMLP5Class,
        SafeDataLoader,
    )

    logger.info("=" * 80)
    logger.info("ADVERSARIAL TRAINING FOR HELIX-IDS")
    logger.info("=" * 80)

    # Load data
    loader = SafeDataLoader()
    data = loader.prepare_data(PROJECT_ROOT / "data")
    X_train, y_train = data["X_train"], data["y_train"]
    x_tr, x_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(x_tr), torch.LongTensor(y_tr)),
        batch_size=128,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(x_val), torch.LongTensor(y_val)),
        batch_size=256,
        num_workers=0,
    )

    n_features = data["n_features"]

    # Train adversarially hardened production model
    model = HELIXMLP5Class(
        input_dim=n_features, hidden_dims=[256, 128, 64, 32], num_classes=5, dropout=0.35
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    model = model.to(DEVICE)

    logger.info("\nAdversarial training: 60 epochs, adv_ratio=0.3, epsilon=0.05")
    best_f1 = 0
    best_state = None

    for epoch in range(60):
        loss, train_f1 = adversarial_train_epoch(
            model, train_loader, optimizer, DEVICE, epsilon=0.05, adv_ratio=0.3
        )

        # Validate
        model.eval()
        all_p, all_t = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                p = model(xb.to(DEVICE)).argmax(1).cpu().numpy()
                all_p.extend(p)
                all_t.extend(yb.numpy())
        val_f1 = f1_score(all_t, all_p, average="macro", zero_division=0)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1}: loss={loss:.4f} train_f1={train_f1:.4f} val_f1={val_f1:.4f}"
            )

    if best_state:
        model.load_state_dict(best_state)
    logger.info(f"\nBest val F1: {best_f1:.4f}")

    # Evaluate robustness
    logger.info("\n--- Robustness Evaluation (NSL-KDD) ---")
    nsl_robustness = evaluate_robustness(model, data["X_nsl_test"], data["y_nsl_test"], DEVICE)
    logger.info("\n--- Robustness Evaluation (UNSW-NB15) ---")
    unsw_robustness = evaluate_robustness(model, data["X_unsw_test"], data["y_unsw_test"], DEVICE)

    # Save
    model_dir = PROJECT_ROOT / "models" / "v2_fixed" / "adversarial"
    model_dir.mkdir(parents=True, exist_ok=True)
    # Save canonical checkpoint including runtime contract and sidecars
    from helix_ids.contracts.schema_contract import runtime_contract_payload
    import json

    model_path = model_dir / "model_adversarial.pt"
    payload = {"model_state_dict": model.state_dict()}
    payload.update(runtime_contract_payload())
    torch.save(payload, model_path)
    (model_path.with_suffix(model_path.suffix + ".contract.json")).write_text(json.dumps(runtime_contract_payload(), indent=2), encoding="utf-8")
    (model_path.with_suffix(model_path.suffix + ".feature_order.json")).write_text(json.dumps(runtime_contract_payload()["feature_order"], indent=2), encoding="utf-8")
    (model_path.with_suffix(model_path.suffix + ".schema_hash.txt")).write_text(str(runtime_contract_payload()["schema_hash"]) + "\n", encoding="utf-8")

    results = {
        "nsl_kdd_robustness": nsl_robustness,
        "unsw_robustness": unsw_robustness,
        "best_val_f1": float(best_f1),
        "training_date": datetime.now().isoformat(),
    }
    with open(RESULTS_DIR / "adversarial_robustness_v2.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\nDone — saved to results/v2_fixed/adversarial_robustness_v2.json")


if __name__ == "__main__":
    main()
