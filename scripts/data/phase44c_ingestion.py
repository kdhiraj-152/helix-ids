#!/usr/bin/env python3
"""
Phase 44C — Dataset Ingestion & Verification.

Loads and harmonizes all 6 datasets, generates:
- feature coverage report
- class distribution report
- entropy report

Usage:
    cd /Users/kdhiraj/Downloads/RP-2
    source .venv311/bin/activate
    PYTHONPATH=src python scripts/data/phase44c_ingestion.py
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase44c_ingestion")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "phase44c"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from helix_ids.contracts.schema_contract import CANONICAL_FEATURE_ORDER, CANONICAL_INPUT_DIM
from helix_ids.contracts.attack_taxonomy import HELIX_CLASSES
from helix_ids.data.multi_dataset_loader import MultiDatasetLoader

CANONICAL_FEATURES = list(CANONICAL_FEATURE_ORDER)
N_FEATURES = len(CANONICAL_FEATURES)  # 17
MAX_SAMPLES = 200_000
RANDOM_STATE = 42
rng = np.random.RandomState(RANDOM_STATE)

DATASET_DISPLAY = {
    "nsl_kdd": "NSL-KDD",
    "unsw_nb15": "UNSW-NB15",
    "cicids2018": "CICIDS2018",
    "ton_iot": "TON-IoT",
    "bot_iot": "Bot-IoT",
    "cicids2017": "CIC-IDS2017",
}
CLASS_NAMES_7 = {0: "Normal", 1: "DoS", 2: "Probe", 3: "R2L", 4: "U2R", 5: "Generic", 6: "Backdoor"}


def subsample_stratified(X, y, max_samples):
    n = X.shape[0]
    if n <= max_samples:
        return X, y
    classes = np.unique(y)
    indices = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        target = max(1, int(max_samples * len(c_idx) / n))
        if len(c_idx) > target:
            c_idx = rng.choice(c_idx, size=target, replace=False)
        indices.extend(c_idx.tolist())
    rng.shuffle(indices)
    idx = np.array(indices, dtype=np.int64)
    return X[idx], y[idx]


def verify_harmonization(datasets: dict[str, dict]) -> dict:
    report = {}

    # 1. Feature coverage
    feat_report = {}
    for name, d in datasets.items():
        if d is None:
            feat_report[name] = {"status": "MISSING"}
            continue
        X = d["X"]
        n_nan = int(np.isnan(X).sum())
        n_inf = int(np.isinf(X).sum())
        # Count columns with zero variance
        stds = np.nanstd(X, axis=0)
        n_zero_var = int((stds == 0).sum())
        # Count columns with all NaN
        n_all_nan = int(np.isnan(X).all(axis=0).sum())
        feat_report[name] = {
            "shape": list(X.shape),
            "n_nan": n_nan,
            "n_inf": n_inf,
            "n_zero_variance_cols": n_zero_var,
            "n_all_nan_cols": n_all_nan,
            "dtype": str(X.dtype),
            "range": [float(np.nanmin(X)), float(np.nanmax(X))],
            "has_finite_range": bool(np.isfinite(X).all()),
        }
    report["feature_coverage"] = feat_report

    # 2. Class distribution
    class_report = {}
    for name, d in datasets.items():
        if d is None:
            class_report[name] = {"status": "MISSING"}
            continue
        y = d["y"]
        unique, counts = np.unique(y, return_counts=True)
        dist = {}
        for cls, cnt in zip(unique, counts):
            cls_name = CLASS_NAMES_7.get(int(cls), f"Class-{cls}")
            dist[cls_name] = {"count": int(cnt), "pct": round(float(cnt / len(y) * 100), 2)}
        sorted_classes = sorted(unique)
        class_report[name] = {
            "n_samples": int(len(y)),
            "n_classes": int(len(unique)),
            "classes_present": [int(c) for c in sorted_classes],
            "distribution": dist,
            "majority_pct": round(max(cnt / len(y) * 100 for cnt in counts), 2),
            "minority_pct": round(min(cnt / len(y) * 100 for cnt in counts), 2),
        }
    report["class_distribution"] = class_report

    # 3. Entropy / information report
    entropy_report = {}
    for name, d in datasets.items():
        if d is None:
            entropy_report[name] = {"status": "MISSING"}
            continue
        X, y = d["X"], d["y"]

        # Per-feature entropy (discretized into quartile-based bins)
        feat_entropies = []
        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            col = col[~np.isnan(col)]
            if len(col) < 2:
                feat_entropies.append(0.0)
                continue
            p1, p99 = np.percentile(col, [1, 99])
            bins = np.linspace(p1, p99, 20) if p99 > p1 else np.array([p1])
            discretized = np.digitize(col, bins)
            vals, cnts = np.unique(discretized, return_counts=True)
            probs = cnts / cnts.sum()
            entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
            feat_entropies.append(entropy)

        # Label entropy
        vals, cnts = np.unique(y, return_counts=True)
        probs = cnts / cnts.sum()
        label_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

        entropy_report[name] = {
            "mean_feature_entropy": round(float(np.mean(feat_entropies)), 4),
            "min_feature_entropy": round(float(np.min(feat_entropies)), 4),
            "max_feature_entropy": round(float(np.max(feat_entropies)), 4),
            "label_entropy": round(label_entropy, 4),
            "n_classes": int(len(vals)),
        }
    report["entropy"] = entropy_report

    return report


def main():
    start_time = time.time()

    # ============================================================
    # Step 1: Load and harmonize all 6 datasets
    # ============================================================
    logger.info("=" * 60)
    logger.info("Step 1: Loading and harmonizing all 6 datasets")
    logger.info("=" * 60)

    loader = MultiDatasetLoader(project_root=PROJECT_ROOT)
    dfs = loader.load_and_harmonize_all()
    # Returns: nsl_kdd, unsw, cicids, ton_iot, bot_iot, cicids2017
    dataset_keys = ["nsl_kdd", "unsw_nb15", "cicids2018", "ton_iot", "bot_iot", "cicids2017"]
    datasets_raw = dict(zip(dataset_keys, dfs))

    for key, df in datasets_raw.items():
        display = DATASET_DISPLAY.get(key, key)
        if df is not None:
            logger.info(f"  {display:18s} | {len(df):>8,} samples | shape={list(df.shape)} | "
                        f"cols={list(df.columns[:6])}...")
        else:
            logger.warning(f"  {display:18s} | NOT AVAILABLE")

    # ============================================================
    # Step 2: Convert to numpy, augment derived features
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Converting to numpy, verifying canonical features")
    logger.info("=" * 60)

    datasets_np = {}
    for key, df in datasets_raw.items():
        if df is None:
            datasets_np[key] = None
            continue

        missing = [f for f in CANONICAL_FEATURES if f not in df.columns]
        if missing:
            logger.warning(f"  {DATASET_DISPLAY[key]}: MISSING canonical: {missing}")

        X = (
            df[CANONICAL_FEATURES]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .to_numpy(dtype=np.float64)
        )
        y = df["label"].to_numpy()

        logger.info(f"  {DATASET_DISPLAY[key]:18s} | X={str(X.shape):20s} | y={len(y):>8,} | "
                    f"NaN={np.isnan(X).sum():>6d} | Inf={np.isinf(X).sum():>4d}")

        datasets_np[key] = {"X": X, "y": y, "feat_cols": CANONICAL_FEATURES}

    # ============================================================
    # Step 3: Subsample for reports
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Subsampling for reports")
    logger.info("=" * 60)

    datasets_sample = {}
    for key, d in datasets_np.items():
        if d is None:
            datasets_sample[key] = None
            continue
        Xs, ys = subsample_stratified(d["X"], d["y"], MAX_SAMPLES)
        datasets_sample[key] = {"X": Xs, "y": ys, "feat_cols": d["feat_cols"]}
        pct = len(Xs) / len(d["X"]) * 100 if len(d["X"]) > MAX_SAMPLES else 100
        logger.info(f"  {DATASET_DISPLAY[key]:18s} | {len(Xs):>8,}/{len(d['X']):<8,} rows ({pct:.0f}%)")

    # ============================================================
    # Step 4: Generate reports
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Generating reports")
    logger.info("=" * 60)

    report = verify_harmonization(datasets_sample)

    report_path = OUTPUT_DIR / "ingestion_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nFull JSON report saved to {report_path}")

    # ============================================================
    # Step 5: Print summary tables
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 60)

    logger.info("\n--- Feature Coverage ---")
    logger.info(f"  {'Dataset':18s} {'Shape':>22s} {'NaN':>8s} {'Inf':>5s} {'ZeroVarCols':>12s}")
    for name, stats in report["feature_coverage"].items():
        if stats.get("status") == "MISSING":
            logger.info(f"  {name:18s} MISSING")
            continue
        display = DATASET_DISPLAY.get(name, name)
        shape = f"{stats['shape'][0]}x{stats['shape'][1]}"
        logger.info(f"  {display:18s} {shape:>22s} {stats['n_nan']:>8d} {stats['n_inf']:>5d} "
                    f"{stats['n_zero_variance_cols']:>12d}")

    logger.info("\n--- Class Distribution ---")
    logger.info(f"  {'Dataset':18s} {'N':>8s} {'Classes':>8s} {'Maj%':>8s} {'Min%':>8s}  {'Distribution'}")
    for name, stats in report["class_distribution"].items():
        if stats.get("status") == "MISSING":
            logger.info(f"  {name:18s} MISSING")
            continue
        display = DATASET_DISPLAY.get(name, name)
        dist_str = " | ".join(f"{k}: {v['pct']:.1f}%" for k, v in stats["distribution"].items())
        logger.info(f"  {display:18s} {stats['n_samples']:>8d} {stats['n_classes']:>8d} "
                    f"{stats['majority_pct']:>7.1f}% {stats['minority_pct']:>7.2f}%  {dist_str}")

    logger.info("\n--- Entropy Report ---")
    logger.info(f"  {'Dataset':18s} {'MeanFeatEnt':>12s} {'MinFeatEnt':>12s} {'MaxFeatEnt':>12s} {'LabelEnt':>10s}")
    for name, stats in report["entropy"].items():
        if stats.get("status") == "MISSING":
            logger.info(f"  {name:18s} MISSING")
            continue
        display = DATASET_DISPLAY.get(name, name)
        logger.info(f"  {display:18s} {stats['mean_feature_entropy']:>12.4f} "
                    f"{stats['min_feature_entropy']:>12.4f} {stats['max_feature_entropy']:>12.4f} "
                    f"{stats['label_entropy']:>10.4f}")

    # Print canonical features verification
    logger.info(f"\n--- Canonical Features: {N_FEATURES} features ---")
    logger.info(f"  {CANONICAL_FEATURES}")

    elapsed = time.time() - start_time
    logger.info(f"\nTotal time: {elapsed:.0f}s")
    logger.info(f"Ingestion complete. Reports in {OUTPUT_DIR}/")

    # Save a summary report
    summary = {
        "metadata": {
            "phase": "44C",
            "experiment": "Dataset Ingestion & Verification",
            "n_datasets": 6,
            "canonical_features": N_FEATURES,
            "max_samples_per_report": MAX_SAMPLES,
        },
        "datasets": {k: {"available": v is not None, "display": DATASET_DISPLAY.get(k, k)}
                     for k, v in datasets_np.items()},
        "feature_coverage": report["feature_coverage"],
        "class_distribution": {k: v for k, v in report["class_distribution"].items()},
        "entropy": report["entropy"],
    }
    summary_path = OUTPUT_DIR / "summary_report.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
