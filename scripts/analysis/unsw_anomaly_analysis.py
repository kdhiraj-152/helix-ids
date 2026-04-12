#!/usr/bin/env python3
"""
UNSW-NB15 Anomaly Analysis & Data Quality Assessment

Loads harmonized UNSW test data and identifies statistical outliers,
anomalies via Isolation Forest, and class distribution issues.

Output:
- Detailed anomaly report (console + JSON)
- Flagged sample indices for removal
- Per-feature statistics & outlier distribution
"""

import os
import json
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "unsw_anomaly_analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Add src to path for imports
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from helix_ids.data.multi_dataset_loader import MultiDatasetLoader
from helix_ids.data.feature_harmonization import COMMON_FEATURES, labels_to_multi_task

FAMILY_CLASS_DISTRIBUTION_LABEL = "  Family class distribution:"


def load_harmonized_unsw():
    """Load harmonized UNSW data from multi-dataset loader."""
    print("=" * 80)
    print("Loading harmonized UNSW datasets...")
    print("=" * 80)

    loader = MultiDatasetLoader(project_root=PROJECT_ROOT)
    nsl_kdd, unsw, _ = loader.load_and_harmonize_all()

    # Reproduce train/val/test logic from loader for UNSW with dataset-specific normalization.
    y_unsw = unsw["label"].values
    X_unsw = unsw.drop(columns=["label"]).values
    feat_cols = unsw.drop(columns=["label"]).columns

    X_train, X_test, y_train, y_test = train_test_split(
        X_unsw,
        y_unsw,
        test_size=0.15,
        random_state=loader.random_state,
        stratify=loader._safe_stratify(y_unsw, "unsw-train-test"),
    )

    x_train_norm = loader.normalize_per_dataset(
        pd.DataFrame(X_train, columns=feat_cols),
        dataset_code=1,
        fit=True,
    )
    X_train = x_train_norm.values

    x_test_norm = loader.normalize_per_dataset(
        pd.DataFrame(X_test, columns=feat_cols),
        dataset_code=1,
        fit=False,
    )
    X_test = x_test_norm.values

    val_ratio = 0.15 / (1 - 0.15)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_ratio,
        random_state=loader.random_state,
        stratify=loader._safe_stratify(y_train, "unsw-train-val"),
    )

    # Keep NSL labels for cross-dataset distribution checks.
    y_nsl = nsl_kdd["label"].values

    print(f"✓ UNSW train set: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"✓ UNSW val set:   {X_val.shape[0]:,} samples, {X_val.shape[1]} features")
    print(f"✓ UNSW test set:  {X_test.shape[0]:,} samples, {X_test.shape[1]} features")

    return X_train, y_train, X_val, y_val, X_test, y_test, y_nsl


def compute_feature_statistics(data, feature_names):
    """Compute comprehensive statistics for each feature."""
    stats = {}
    for i, feat in enumerate(feature_names):
        col = data.iloc[:, i]
        stats[feat] = {
            "mean": float(col.mean()),
            "std": float(col.std()),
            "min": float(col.min()),
            "max": float(col.max()),
            "median": float(col.median()),
            "q25": float(col.quantile(0.25)),
            "q75": float(col.quantile(0.75)),
            "iqr": float(col.quantile(0.75) - col.quantile(0.25)),
            "skew": float(col.skew()),
            "kurtosis": float(col.kurtosis()),
        }
    return stats


def detect_iqr_outliers(data, feature_names, iqr_multiplier=1.5):
    """Detect IQR-based outliers per feature."""
    outlier_mask = pd.DataFrame(False, index=data.index, columns=data.columns)
    outlier_counts = {}

    for i, feat in enumerate(feature_names):
        col = data.iloc[:, i]
        Q1 = col.quantile(0.25)
        Q3 = col.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR

        is_outlier = (col < lower_bound) | (col > upper_bound)
        outlier_mask.iloc[:, i] = is_outlier

        outlier_counts[feat] = {
            "count": int(is_outlier.sum()),
            "percentage": float((is_outlier.sum() / len(col)) * 100),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "Q1": float(Q1),
            "Q3": float(Q3),
            "IQR": float(IQR),
        }

    # Sample-level outlier detection: flag if outlier in ANY feature
    any_outlier = outlier_mask.any(axis=1)

    return outlier_mask, any_outlier, outlier_counts


def detect_isolation_forest_anomalies(data, contamination=0.05):
    """Detect anomalies using Isolation Forest."""
    print(f"\n  Running Isolation Forest (contamination={contamination})...")
    iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)

    predictions = iso_forest.fit_predict(data)
    scores = iso_forest.score_samples(data)

    is_anomaly = predictions == -1

    return is_anomaly, scores


def analyze_class_distribution(y_unsw_train, y_unsw_val, y_unsw_test, y_nsl_train):
    """Compare attack class distributions between UNSW and NSL-KDD."""
    print("\n" + "=" * 80)
    print("Attack Class Distribution Analysis")
    print("=" * 80)

    unsw_train_binary, unsw_train_family = labels_to_multi_task(y_unsw_train)
    unsw_val_binary, _ = labels_to_multi_task(y_unsw_val)
    unsw_test_binary, unsw_test_family = labels_to_multi_task(y_unsw_test)
    nsl_train_binary, nsl_train_family = labels_to_multi_task(y_nsl_train)

    family_names = ["Normal", "DoS", "Probe", "R2L", "U2R", "Generic", "Backdoor"]

    print("\nUNSW Train Set:")
    print(f"  Binary: {dict(zip(*np.unique(unsw_train_binary, return_counts=True)))}")
    print(FAMILY_CLASS_DISTRIBUTION_LABEL)
    for cls, name in enumerate(family_names):
        count = (unsw_train_family == cls).sum()
        pct = (count / len(unsw_train_family)) * 100
        print(f"    {name:12s}: {count:6,} ({pct:5.2f}%)")

    print("\nUNSW Val Set:")
    print(f"  Binary: {dict(zip(*np.unique(unsw_val_binary, return_counts=True)))}")

    print("\nUNSW Test Set:")
    print(f"  Binary: {dict(zip(*np.unique(unsw_test_binary, return_counts=True)))}")
    print(FAMILY_CLASS_DISTRIBUTION_LABEL)
    for cls, name in enumerate(family_names):
        count = (unsw_test_family == cls).sum()
        pct = (count / len(unsw_test_family)) * 100
        print(f"    {name:12s}: {count:6,} ({pct:5.2f}%)")

    print("\nNSL-KDD Train Set (Baseline):")
    print(f"  Binary: {dict(zip(*np.unique(nsl_train_binary, return_counts=True)))}")
    print(FAMILY_CLASS_DISTRIBUTION_LABEL)
    for cls, name in enumerate(family_names):
        count = (nsl_train_family == cls).sum()
        pct = (count / len(nsl_train_family)) * 100
        print(f"    {name:12s}: {count:6,} ({pct:5.2f}%)")

    return unsw_train_binary, unsw_test_binary, nsl_train_binary


def generate_report(
    unsw_train,
    unsw_val,
    unsw_test,
    feature_stats,
    iqr_any_outlier,
    iqr_counts,
    iso_anomalies,
    iso_scores,
):
    """Generate comprehensive anomaly analysis report."""

    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset_summary": {
            "unsw_train_count": len(unsw_train),
            "unsw_val_count": len(unsw_val),
            "unsw_test_count": len(unsw_test),
            "total_features": unsw_train.shape[1],
            "feature_names": list(COMMON_FEATURES) + ["is_nsl_kdd", "is_unsw", "is_cicids"],
        },
        "feature_statistics": feature_stats,
        "iqr_analysis": {
            "per_feature": iqr_counts,
            "any_feature_outlier_count": int(iqr_any_outlier.sum()),
            "any_feature_outlier_percentage": float(
                (iqr_any_outlier.sum() / len(unsw_train)) * 100
            ),
        },
        "isolation_forest_analysis": {
            "contamination_rate": 0.05,
            "anomaly_count": int(iso_anomalies.sum()),
            "anomaly_percentage": float((iso_anomalies.sum() / len(unsw_train)) * 100),
            "mean_anomaly_score": float(iso_scores[iso_anomalies].mean()),
            "median_anomaly_score": float(np.median(iso_scores[iso_anomalies])),
        },
        "combined_analysis": {
            "both_iqr_and_isof": int((iqr_any_outlier & iso_anomalies).sum()),
            "either_iqr_or_isof": int((iqr_any_outlier | iso_anomalies).sum()),
            "only_iqr": int((iqr_any_outlier & ~iso_anomalies).sum()),
            "only_isof": int((~iqr_any_outlier & iso_anomalies).sum()),
        },
    }

    return report


def main():
    # Load data
    unsw_train, y_unsw_train, unsw_val, y_unsw_val, unsw_test, y_unsw_test, y_nsl_train = (
        load_harmonized_unsw()
    )

    unsw_train_df = pd.DataFrame(
        unsw_train,
        columns=list(COMMON_FEATURES) + ["is_nsl_kdd", "is_unsw", "is_cicids"],
    )

    # Feature names (28 common + 3 dataset origin + 2 labels = 33 total)
    feature_names_base = list(COMMON_FEATURES) + ["is_nsl_kdd", "is_unsw", "is_cicids"]

    print("\n" + "=" * 80)
    print("Phase 1: Feature Statistics & Outlier Detection")
    print("=" * 80)

    # Compute statistics on train set (use for baseline)
    print("\nComputing feature statistics on UNSW train set...")
    feature_stats = compute_feature_statistics(unsw_train_df, feature_names_base)

    # Print top insights
    print("\n  Top 10 features with highest standard deviation:")
    sorted_std = sorted(feature_stats.items(), key=lambda x: x[1]["std"], reverse=True)
    for feat, stats in sorted_std[:10]:
        print(
            f"    {feat:20s}: std={stats['std']:10.4f}, range=[{stats['min']:8.4f}, {stats['max']:8.4f}]"
        )

    # IQR-based outlier detection
    print("\nDetecting IQR-based outliers (1.5×IQR method) on UNSW train set...")
    _, iqr_any, iqr_counts = detect_iqr_outliers(unsw_train_df, feature_names_base)

    print(
        f"  ✓ Samples with outliers in ANY feature: {iqr_any.sum():,} ({(iqr_any.sum() / len(unsw_train)) * 100:.2f}%)"
    )

    print("\n  Top 10 features with highest outlier percentage:")
    sorted_outliers = sorted(iqr_counts.items(), key=lambda x: x[1]["percentage"], reverse=True)
    for feat, counts in sorted_outliers[:10]:
        print(f"    {feat:20s}: {counts['count']:6,} ({counts['percentage']:6.2f}%)")

    # Isolation Forest anomaly detection
    print("\nDetecting anomalies via Isolation Forest on UNSW train set...")
    iso_anomalies, iso_scores = detect_isolation_forest_anomalies(unsw_train_df, contamination=0.05)
    print(
        f"  ✓ Anomalies detected: {iso_anomalies.sum():,} ({(iso_anomalies.sum() / len(unsw_train)) * 100:.2f}%)"
    )

    # Combined analysis
    both = (iqr_any & iso_anomalies).sum()
    either = (iqr_any | iso_anomalies).sum()
    only_iqr = (iqr_any & ~iso_anomalies).sum()
    only_iso = (~iqr_any & iso_anomalies).sum()

    print("\nCombined Anomaly Detection:")
    print(f"  Both IQR and Isolation Forest: {both:,} ({(both / len(unsw_train)) * 100:.2f}%)")
    print(f"  Either IQR or Isolation Forest: {either:,} ({(either / len(unsw_train)) * 100:.2f}%)")
    print(f"  Only IQR outliers: {only_iqr:,} ({(only_iqr / len(unsw_train)) * 100:.2f}%)")
    print(
        f"  Only Isolation Forest anomalies: {only_iso:,} ({(only_iso / len(unsw_train)) * 100:.2f}%)"
    )

    # Class distribution analysis
    print()
    _, _, _ = analyze_class_distribution(y_unsw_train, y_unsw_val, y_unsw_test, y_nsl_train)

    # Generate and save report
    print("\n" + "=" * 80)
    print("Generating detailed report...")
    print("=" * 80)

    report = generate_report(
        unsw_train,
        unsw_val,
        unsw_test,
        feature_stats,
        iqr_any,
        iqr_counts,
        iso_anomalies,
        iso_scores,
    )

    # Save report as JSON
    report_path = RESULTS_DIR / "unsw_anomaly_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Report saved to: {report_path}")

    # Save flagged sample indices for removal (conservative: both IQR AND IsFor)
    flagged_conservative = np.nonzero(iqr_any & iso_anomalies)[0]
    flagged_aggressive = np.nonzero(iqr_any | iso_anomalies)[0]

    conservative_path = RESULTS_DIR / "flagged_samples_conservative.json"
    with open(conservative_path, "w") as f:
        json.dump(
            {
                "description": "Samples flagged by BOTH IQR AND Isolation Forest (conservative)",
                "count": int(len(flagged_conservative)),
                "percentage": float((len(flagged_conservative) / len(unsw_train)) * 100),
                "indices": flagged_conservative.tolist(),
            },
            f,
            indent=2,
        )
    print(f"✓ Conservative flagged samples saved: {conservative_path}")

    aggressive_path = RESULTS_DIR / "flagged_samples_aggressive.json"
    with open(aggressive_path, "w") as f:
        json.dump(
            {
                "description": "Samples flagged by EITHER IQR OR Isolation Forest (aggressive)",
                "count": int(len(flagged_aggressive)),
                "percentage": float((len(flagged_aggressive) / len(unsw_train)) * 100),
                "indices": flagged_aggressive.tolist(),
            },
            f,
            indent=2,
        )
    print(f"✓ Aggressive flagged samples saved: {aggressive_path}")

    # Save feature statistics
    stats_path = RESULTS_DIR / "feature_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(feature_stats, f, indent=2)
    print(f"✓ Feature statistics saved: {stats_path}")

    print("\n" + "=" * 80)
    print("✅ Phase 1 Complete: UNSW Anomaly Analysis")
    print("=" * 80)
    print("\nNext Steps:")
    print(f"  1. Review report at: {report_path}")
    print("  2. Use conservative flagged samples to filter UNSW training data")
    print("  3. Create cleaned UNSW dataset (Phase 2)")
    print("  4. Train UNSW-only model with cleaned data (Phase 3)")

    return report, flagged_conservative, flagged_aggressive


if __name__ == "__main__":
    report, flagged_cons, flagged_agg = main()
