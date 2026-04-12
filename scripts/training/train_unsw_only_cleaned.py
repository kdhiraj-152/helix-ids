#!/usr/bin/env python3
"""Train HelixIDS-Full on UNSW-only data after anomaly filtering.

Phase 2: remove anomalous UNSW train samples using saved indices.
Phase 3: train/evaluate UNSW-only HelixIDS-Full model.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from helix_ids.config.helix_full_config import TrainingConfig
from helix_ids.data.multi_dataset_loader import MultiDatasetLoader
from helix_ids.data.feature_harmonization import (
    COMMON_FEATURES,
    labels_to_multi_task,
)
from helix_ids.models.full import MultiTaskLoss, create_helix_full
from train_helix_ids_full import HelixFullTrainer, setup_logging


def load_unsw_split(loader: MultiDatasetLoader):
    """Reproduce UNSW split logic used in anomaly phase for index consistency."""
    unsw_raw = loader.load_unsw()
    unsw = loader.harmonize_unsw(unsw_raw)

    y = unsw["label"].values
    X = unsw.drop(columns=["label"]).values
    feat_cols = unsw.drop(columns=["label"]).columns

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=loader.random_state,
        stratify=loader._safe_stratify(y, "unsw-train-test"),
    )

    X_train = loader.normalize_per_dataset(
        pd.DataFrame(X_train, columns=feat_cols),
        dataset_code=1,
        fit=True,
    ).values

    X_test = loader.normalize_per_dataset(
        pd.DataFrame(X_test, columns=feat_cols),
        dataset_code=1,
        fit=False,
    ).values

    val_ratio = 0.15 / (1 - 0.15)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_ratio,
        random_state=loader.random_state,
        stratify=loader._safe_stratify(y_train, "unsw-train-val"),
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_flagged_indices(mode: str):
    file_name = (
        "flagged_samples_conservative.json"
        if mode == "conservative"
        else "flagged_samples_aggressive.json"
    )
    path = PROJECT_ROOT / "results" / "unsw_anomaly_analysis" / file_name
    if not path.exists():
        raise FileNotFoundError(f"Missing anomaly file: {path}")
    with open(path, "r") as f:
        payload = json.load(f)
    return np.array(payload["indices"], dtype=np.int64), payload


def main():
    parser = argparse.ArgumentParser(description="Train UNSW-only cleaned model")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--output", type=str, default="models/helix_full_unsw_cleaned")
    parser.add_argument(
        "--anomaly-mode",
        type=str,
        choices=["conservative", "aggressive"],
        default="conservative",
        help="Conservative=remove samples flagged by both IQR+IsolationForest",
    )
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = PROJECT_ROOT / "results" / "unsw_only_cleaned"
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(results_dir)

    logger.info("Loading UNSW splits and anomaly indices...")
    loader = MultiDatasetLoader(project_root=PROJECT_ROOT)
    X_train, y_train, X_val, y_val, X_test, y_test = load_unsw_split(loader)

    flagged_idx, flagged_meta = load_flagged_indices(args.anomaly_mode)
    valid_mask = (flagged_idx >= 0) & (flagged_idx < len(X_train))
    flagged_idx = flagged_idx[valid_mask]

    keep_mask = np.ones(len(X_train), dtype=bool)
    keep_mask[flagged_idx] = False

    X_train_clean = X_train[keep_mask]
    y_train_clean = y_train[keep_mask]

    logger.info(
        "UNSW clean split: train %s -> %s (removed %s / %.2f%%), val %s, test %s",
        f"{len(X_train):,}",
        f"{len(X_train_clean):,}",
        f"{len(flagged_idx):,}",
        (len(flagged_idx) / len(X_train)) * 100,
        f"{len(X_val):,}",
        f"{len(X_test):,}",
    )
    logger.info("Anomaly mode: %s | %s", args.anomaly_mode, flagged_meta.get("description", ""))

    y_train_bin, y_train_fam = labels_to_multi_task(y_train_clean)
    y_val_bin, y_val_fam = labels_to_multi_task(y_val)
    y_test_bin, y_test_fam = labels_to_multi_task(y_test)

    logger.info("Binary distribution - Train: %s", np.bincount(y_train_bin).tolist())
    logger.info("Family distribution - Train: %s", np.bincount(y_train_fam).tolist())

    train_dataset = TensorDataset(
        torch.from_numpy(X_train_clean).float(),
        torch.from_numpy(y_train_bin).long(),
        torch.from_numpy(y_train_fam).long(),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val_bin).long(),
        torch.from_numpy(y_val_fam).long(),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test_bin).long(),
        torch.from_numpy(y_test_fam).long(),
    )

    train_config = TrainingConfig(
        batch_size=args.batch_size, epochs=args.epochs, device=args.device
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=train_config.batch_size, shuffle=False, num_workers=0
    )

    model = create_helix_full()
    logger.info("Model parameters: %s", f"{model.param_count:,}")

    fam_counts = np.bincount(y_train_fam.astype(int), minlength=7)
    fam_counts = np.where(fam_counts == 0, 1, fam_counts)
    fam_weights = fam_counts.sum() / (len(fam_counts) * fam_counts)
    family_class_weights = torch.from_numpy(fam_weights.astype(np.float32))

    bin_counts = np.bincount(y_train_bin.astype(int), minlength=2)
    bin_counts = np.where(bin_counts == 0, 1, bin_counts)
    bin_weights = bin_counts.sum() / (len(bin_counts) * bin_counts)
    binary_class_weights = torch.from_numpy(bin_weights.astype(np.float32))

    logger.info("Family class weights: %s", family_class_weights.tolist())
    logger.info("Binary class weights: %s", binary_class_weights.tolist())

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    loss_fn = MultiTaskLoss(
        lambda_binary=train_config.lambda_binary,
        lambda_family=train_config.lambda_family,
    )

    trainer = HelixFullTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loaders={"unsw_cleaned": test_loader},
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=train_config,
        binary_class_weights=binary_class_weights,
        family_class_weights=family_class_weights,
        device=args.device,
        logger=logger,
    )

    logger.info("Starting UNSW-only cleaned training...")
    results = trainer.fit()

    best_model_path = output_dir / "helix_full_unsw_cleaned_best.pt"
    final_model_path = output_dir / "helix_full_unsw_cleaned_final.pt"
    torch.save(model.state_dict(), best_model_path)
    torch.save(model.state_dict(), final_model_path)
    logger.info("Saved model: %s", best_model_path)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "device": args.device,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "anomaly_mode": args.anomaly_mode,
        },
        "dataset": {
            "train_original": int(len(X_train)),
            "train_cleaned": int(len(X_train_clean)),
            "removed": int(len(flagged_idx)),
            "removed_pct": float((len(flagged_idx) / len(X_train)) * 100),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "results": {
            "best_val_loss": float(results["best_val_loss"]),
            "epochs_trained": int(results["epochs_trained"]),
            "unsw_cleaned": results["per_dataset_results"]["unsw_cleaned"],
        },
    }

    with open(results_dir / "training_results_unsw_cleaned.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Training results saved to %s", results_dir / "training_results_unsw_cleaned.json")


if __name__ == "__main__":
    main()
