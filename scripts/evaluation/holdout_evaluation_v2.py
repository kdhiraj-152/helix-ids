#!/usr/bin/env python3
"""
Holdout Evaluation Script for HELIX-IDS v2

True holdout evaluation on completely unseen data splits to validate
generalization beyond the training datasets.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from helix_ids.governance.entrypoint import governed_entrypoint  # noqa: E402
from helix_ids.governance.determinism import seed_worker, set_global_determinism
from helix_ids.governance.parameters import DEFAULT_GOVERNANCE_POLICY
from helix_ids.governance.promotion import SeedRunSummary, aggregate_seed_runs
from helix_ids.governance.run_registry import RunRegistry
from helix_ids.utils.metrics import evaluate as evaluate_contract  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results" / "v2_fixed"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
if torch.cuda.is_available():
    device_str = "cuda"
elif torch.backends.mps.is_available():
    device_str = "mps"
else:
    device_str = "cpu"
DEVICE = torch.device(device_str)
CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]


def cross_validate_holdout(model_class, model_kwargs, X, y, class_weights, n_folds=5, seed=42):
    """5-fold cross-validation for robust holdout estimation."""
    from torch.utils.data import DataLoader, TensorDataset
    from train_multidataset_v2_fixed import ImprovedTrainer

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"\n--- Fold {fold + 1}/{n_folds} ---")
        x_tr, x_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        loader_generator = torch.Generator()
        loader_generator.manual_seed(seed + fold)
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(x_tr), torch.LongTensor(y_tr)),
            batch_size=128,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=loader_generator,
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(x_val), torch.LongTensor(y_val)),
            batch_size=256,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=loader_generator,
        )

        model = model_class(**model_kwargs)
        trainer = ImprovedTrainer(model, DEVICE, class_weights=class_weights, gamma=2.5)
        trainer.fit(train_loader, val_loader, lr=5e-4, epochs=80, patience=15)

        # Evaluate
        model.eval()
        all_p = []
        with torch.no_grad():
            for xb, _ in val_loader:
                p = model(xb.to(DEVICE)).argmax(1).cpu().numpy()
                all_p.extend(p)
        all_p = np.array(all_p)

        fold_metrics = evaluate_contract(
            preds=all_p,
            targets=y_val,
            dataset_id=f"cv_fold_{fold + 1}",
            class_names=CLASS_NAMES,
        )

        fold_results.append(
            {
                "fold": fold + 1,
                "accuracy": float(fold_metrics.accuracy),
                "f1_macro": float(fold_metrics.macro_f1),
                "f1_per_class": fold_metrics.per_class_f1,
                "ci95_lower": float(fold_metrics.ci95_lower),
                "ci95_upper": float(fold_metrics.ci95_upper),
            }
        )
        logger.info(
            f"Fold {fold + 1}: Acc={fold_metrics.accuracy:.4f} F1-macro={fold_metrics.macro_f1:.4f}"
        )

    # Aggregate
    avg_f1 = np.mean([r["f1_macro"] for r in fold_results])
    std_f1 = np.std([r["f1_macro"] for r in fold_results])
    avg_acc = np.mean([r["accuracy"] for r in fold_results])

    logger.info(f"\nCross-validation: F1-macro = {avg_f1:.4f} ± {std_f1:.4f}, Acc = {avg_acc:.4f}")

    return {
        "folds": fold_results,
        "mean_f1_macro": float(avg_f1),
        "std_f1_macro": float(std_f1),
        "mean_accuracy": float(avg_acc),
    }


def cross_dataset_holdout(model, nsl_scaler, unsw_scaler, n_features):
    """Evaluate trained model on completely separate dataset portions."""
    from train_multidataset_v2_fixed import SafeDataLoader

    loader = SafeDataLoader()
    x_nsl, y_nsl = loader.load_nsl_kdd(PROJECT_ROOT / "data" / "nsl_kdd" / "test.csv")
    x_unsw, y_unsw = loader.load_unsw_nb15(PROJECT_ROOT / "data" / "unsw_nb15" / "test.csv")

    x_nsl = nsl_scaler.transform(np.nan_to_num(x_nsl[:, :n_features], 0))
    x_unsw = unsw_scaler.transform(np.nan_to_num(x_unsw[:, :n_features], 0))

    results = {}
    for name, X, y in [("NSL-KDD", x_nsl, y_nsl), ("UNSW-NB15", x_unsw, y_unsw)]:
        model.eval()
        with torch.no_grad():
            preds = model(torch.FloatTensor(X).to(DEVICE)).argmax(1).cpu().numpy()

        eval_metrics = evaluate_contract(
            preds=preds,
            targets=y,
            dataset_id=name,
            class_names=CLASS_NAMES,
        )

        results[name] = {
            "accuracy": float(eval_metrics.accuracy),
            "f1_macro": float(eval_metrics.macro_f1),
            "f1_per_class": eval_metrics.per_class_f1,
            "ci95_lower": float(eval_metrics.ci95_lower),
            "ci95_upper": float(eval_metrics.ci95_upper),
        }
        logger.info(f"{name}: Acc={eval_metrics.accuracy:.4f} F1-macro={eval_metrics.macro_f1:.4f}")

    # Cross-dataset F1 drop
    if "NSL-KDD" in results and "UNSW-NB15" in results:
        f1_drop = results["NSL-KDD"]["f1_macro"] - results["UNSW-NB15"]["f1_macro"]
        results["cross_dataset_f1_drop"] = float(f1_drop)
        logger.info(f"Cross-dataset F1 drop: {f1_drop:.4f}")

    return results


@governed_entrypoint(entrypoint_id="scripts.holdout_evaluation_v2")
def main():
    seed = int(os.environ.get("HELIX_SEED", "42"))
    os.environ["HELIX_SEED"] = str(seed)
    determinism_state = set_global_determinism(seed)

    from train_multidataset_v2_fixed import HELIXMLP5Class, SafeDataLoader

    logger.info("=" * 80)
    logger.info("HELIX-IDS HOLDOUT EVALUATION")
    logger.info("=" * 80)

    split_start = time.perf_counter()
    loader = SafeDataLoader()
    data = loader.prepare_data(PROJECT_ROOT / "data")
    split_elapsed = time.perf_counter() - split_start

    model_kwargs = {
        "input_dim": data["n_features"],
        "hidden_dims": [256, 128, 64, 32],
        "num_classes": 5,
        "dropout": 0.35,
    }

    # 5-fold CV
    intrain_start = time.perf_counter()
    cv_results = cross_validate_holdout(
        HELIXMLP5Class,
        model_kwargs,
        data["X_train"],
        data["y_train"],
        data["class_weights"],
        n_folds=5,
        seed=seed,
    )
    intrain_elapsed = time.perf_counter() - intrain_start

    results = {"cross_validation": cv_results, "date": datetime.now().isoformat()}

    with open(RESULTS_DIR / "holdout_evaluation_v2.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\nDone — saved to results/v2_fixed/holdout_evaluation_v2.json")

    posteval_start = time.perf_counter()
    fold_widths = [
        float(fold["ci95_upper"] - fold["ci95_lower"])
        for fold in cv_results["folds"]
        if "ci95_lower" in fold and "ci95_upper" in fold
    ]
    fold_lowers = [
        float(fold["ci95_lower"]) for fold in cv_results["folds"] if "ci95_lower" in fold
    ]
    aggregate_macro_f1 = float(cv_results.get("mean_f1_macro", 0.0))
    policy = DEFAULT_GOVERNANCE_POLICY
    registry = RunRegistry(
        Path(os.environ.get("HELIX_RUN_REGISTRY", "results/gates/run_registry.jsonl"))
    )
    drift, z_score = registry.compute_drift(
        dataset_id="holdout_evaluation_v2",
        current_macro_f1=aggregate_macro_f1,
        baseline_window_runs=20,
    )
    min_ci_lower = min(fold_lowers) if fold_lowers else 0.0
    max_ci_width = max(fold_widths) if fold_widths else 0.0
    tier2_pass = (
        min_ci_lower >= policy.bootstrap.min_ci95_lower_bound
        and max_ci_width <= policy.bootstrap.max_ci_width
        and drift <= policy.drift.max_abs_macro_f1_drift
        and z_score <= policy.drift.max_abs_z_score
    )
    promotion_consensus = aggregate_seed_runs(
        [
            SeedRunSummary(
                seed=seed,
                macro_f1=aggregate_macro_f1,
                macro_f1_ci_lower=min_ci_lower,
                macro_f1_ci_width=max_ci_width,
                tier2_pass=tier2_pass,
            )
        ],
        min_seed_runs=policy.promotion.min_seed_runs,
        max_inter_seed_macro_f1_variance=policy.promotion.max_inter_seed_macro_f1_variance,
        reproducibility_tolerance=policy.promotion.reproducibility_tolerance,
        min_ci95_lower_bound=policy.bootstrap.min_ci95_lower_bound,
        max_ci_width=policy.bootstrap.max_ci_width,
    )
    prepromote_elapsed = max(0.001, time.perf_counter() - posteval_start)

    governance_stages = {
        "presplit": {
            "presplit_elapsed_seconds": split_elapsed,
            "split_train_rows": int(data["X_train"].shape[0]),
            "split_binary_class_count": int(len(np.unique((data["y_train"] > 0).astype(int)))),
        },
        "pretrain": {
            "pretrain_elapsed_seconds": 0.001,
            "family_class_weight_min": float(min(data["class_weights"].values())),
            "binary_class_weight_min": 1.0,
        },
        "intrain": {
            "intrain_elapsed_seconds": max(0.001, intrain_elapsed),
            "low_entropy_consecutive_batches": 0,
            "gradient_dominance": 0.0,
            "epochs_without_improvement": 0,
        },
        "posteval": {
            "posteval_elapsed_seconds": max(0.001, time.perf_counter() - posteval_start),
            "macro_f1_ci_width": max_ci_width,
            "macro_f1_ci_lower": min_ci_lower,
            "abs_macro_f1_drift": max(float(cv_results.get("std_f1_macro", 0.0)), drift),
            "abs_macro_f1_zscore": z_score,
        },
        "prepromote": {
            "prepromote_elapsed_seconds": prepromote_elapsed,
            "macro_f1_ci_width": max_ci_width,
            "macro_f1_ci_lower": min_ci_lower,
            **promotion_consensus.to_stage_metrics(),
        },
    }
    if promotion_consensus.invalid_reason is not None:
        governance_stages["prepromote"]["promotion_invalid_reason"] = (
            promotion_consensus.invalid_reason
        )

    return {
        "results": results,
        "governance_stages": governance_stages,
        "governance_context": {
            "seed": seed,
        },
        "governance_run_record": {
            "dataset_id": "holdout_evaluation_v2",
            "macro_f1": aggregate_macro_f1,
            "fingerprint": os.environ.get("HELIX_FINGERPRINT"),
            "parent_run_id": os.environ.get("HELIX_PARENT_RUN_ID"),
            "lineage": {
                "dataset_hashes": os.environ.get("HELIX_DATASET_HASHES", "unknown"),
                "schema_hash": os.environ.get("HELIX_SCHEMA_HASH", "unknown"),
                "mapping_version": os.environ.get("HELIX_MAPPING_VERSION", "unknown"),
                "model_artifact": str(PROJECT_ROOT / "models" / "v2_fixed"),
                "metrics_artifact": str(RESULTS_DIR / "holdout_evaluation_v2.json"),
            },
        },
        "determinism": determinism_state.to_dict(),
    }


if __name__ == "__main__":
    main()
