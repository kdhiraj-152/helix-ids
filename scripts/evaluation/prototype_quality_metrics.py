#!/usr/bin/env python3
"""
Phase 42 — Prototype Validation: Empirical Quality Metrics (DOS, LCS, SOS, DIC)

Implements the four Phase 36 quality metrics and runs them on:
    NSL-KDD, UNSW-NB15, CICIDS2018, TON-IoT

Usage:
    cd helix-ids
    source .venv311/bin/activate
    PYTHONPATH=src python scripts/evaluation/prototype_quality_metrics.py

Reference: docs/phase36/QUALITY_METRICS.md
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from helix_ids.data.multi_dataset_loader import MultiDatasetLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("prototype")

# Canonical 7-class names (matching ATTACK_TAXONOMY_7CLASS)
CLASS_NAMES_7 = ["Normal", "DoS", "Probe", "R2L", "U2R", "Generic", "Backdoor"]

# Subsample cap for DOS (logistic regression is O(n) and large CICIDS files can be slow)
DOS_MAX_SAMPLES = 100_000

# Number of folds for DOS cross-validation
DOS_FOLDS = 5

# Subsample cap for SOS (per-class Wasserstein on 16M CICIDS rows is OOM)
SOS_MAX_SAMPLES = 20_000  # per dataset, before per-class splitting
SOS_MAX_CLASS_SAMPLES = 5_000  # per class


def compute_pairwise_dos(
    datasets: dict[str, np.ndarray],
    dataset_labels: dict[str, np.ndarray],
) -> dict[tuple[str, str], float]:
    """
    DOS (Domain Overlap Score) for every pair of datasets.

    DOS = 1 - Dataset-ID Accuracy

    Dataset-ID Accuracy = accuracy of a logistic regression (L2, 5-fold CV)
    trained to distinguish which dataset a sample came from, using only the
    canonical features.

    Returns dict mapping (ds1, ds2) -> DOS.
    """
    results: dict[tuple[str, str], float] = {}
    names = list(datasets.keys())

    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            logger.info("  DOS: %s <-> %s", name_a, name_b)

            X_a = datasets[name_a]
            X_b = datasets[name_b]

            # Subsample if too large
            if len(X_a) > DOS_MAX_SAMPLES:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(X_a), DOS_MAX_SAMPLES, replace=False)
                X_a = X_a[idx]
            if len(X_b) > DOS_MAX_SAMPLES:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(X_b), DOS_MAX_SAMPLES, replace=False)
                X_b = X_b[idx]

            # Build binary classification problem: dataset A vs dataset B
            n_a, n_b = len(X_a), len(X_b)
            X_pair = np.vstack([X_a, X_b])
            y_pair = np.array([0] * n_a + [1] * n_b, dtype=np.int64)

            # Normalise features for logistic regression
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_pair)

            # 5-fold stratified cross-validation
            cv = StratifiedKFold(n_splits=DOS_FOLDS, shuffle=True, random_state=42)
            accuracies = cross_val_score(
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=42,
                ),
                X_scaled,
                y_pair,
                cv=cv,
                scoring="accuracy",
                n_jobs=-1,
            )
            dos = float(1.0 - accuracies.mean())
            results[(name_a, name_b)] = dos
            logger.info(
                "    → Dataset-ID Accuracy: %.4f ± %.4f  DOS: %.4f",
                accuracies.mean(), accuracies.std(), dos,
            )

    return results


def compute_lcs(
    datasets: dict[str, np.ndarray],
) -> dict[tuple[str, str], float]:
    """
    LCS (Label Consistency Score) for every pair of datasets.

    LCS = (Number of shared classes) / (Total unique classes across both)

    NOTE: Phase 36 defines LCS at Level-2 (specific attack tool+configuration).
    This prototype computes LCS at the class-family level since Level-2
    annotations are not available in the existing datasets. This overestimates
    LCS relative to the spec (families are broader than Level-2 types).

    Returns dict mapping (ds1, ds2) -> LCS.
    """
    results: dict[tuple[str, str], float] = {}
    names = list(datasets.keys())
    label_sets = {name: set(labels.astype(int)) for name, labels in datasets.items()}

    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            classes_a = label_sets[name_a]
            classes_b = label_sets[name_b]
            shared = classes_a & classes_b
            total = classes_a | classes_b
            lcs = len(shared) / len(total) if total else 0.0
            results[(name_a, name_b)] = lcs
            logger.info(
                "  LCS %s <-> %s: shared=%d unique=%d  LCS=%.4f",
                name_a, name_b, len(shared), len(total), lcs,
            )

    return results


def compute_sos(
    datasets: dict[str, np.ndarray],
    dataset_labels: dict[str, np.ndarray],
) -> dict[tuple[str, str], float]:
    """
    SOS (Semantic Overlap Score) for every pair of datasets.

    For each shared class c across collections A and B:
      SOS_c = 1 - W_distance(features_A_c, features_B_c) / max_possible_W

    Overall SOS is the macro-average across all shared classes.

    Uses per-feature 1D Wasserstein distances (robust, memory-safe) averaged
    across features. max_possible_W is the theoretical maximum for 1D W_dist
    on standardized features: E[|Z1 - Z2|] for standard normals ≈ 1.128,
    so max_possible_W ≈ 6 standard deviations = 6.0 per feature.
    """
    from scipy.stats import wasserstein_distance

    results: dict[tuple[str, str], float] = {}
    names = list(datasets.keys())
    rng = np.random.RandomState(42)
    n_features = datasets[names[0]].shape[1]

    # For standardized features, max_possible_W per feature is ~6 sigma
    # (covering 99.999% of probability mass for normal distributions)
    max_possible_w_per_feature = 6.0

    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            logger.info("  SOS: %s <-> %s", name_a, name_b)

            X_a_all = datasets[name_a]
            y_a_all = dataset_labels[name_a].astype(int)
            X_b_all = datasets[name_b]
            y_b_all = dataset_labels[name_b].astype(int)

            # --- Subsample whole datasets first (avoid loading 16M rows into masks) ---
            if len(X_a_all) > SOS_MAX_SAMPLES:
                idx = rng.choice(len(X_a_all), SOS_MAX_SAMPLES, replace=False)
                X_a_all = X_a_all[idx]
                y_a_all = y_a_all[idx]

            if len(X_b_all) > SOS_MAX_SAMPLES:
                idx = rng.choice(len(X_b_all), SOS_MAX_SAMPLES, replace=False)
                X_b_all = X_b_all[idx]
                y_b_all = y_b_all[idx]

            # --- Normalize features jointly for stability ---
            combined = np.vstack([X_a_all, X_b_all])
            mean = combined.mean(axis=0)
            std = combined.std(axis=0) + 1e-8
            X_a_norm = (X_a_all - mean) / std
            X_b_norm = (X_b_all - mean) / std

            shared_classes = sorted(set(y_a_all) & set(y_b_all))
            if not shared_classes:
                logger.info("    → No shared classes, SOS = 0.0")
                results[(name_a, name_b)] = 0.0
                continue

            sos_per_class: list[float] = []
            for c in shared_classes:
                mask_a = y_a_all == c
                mask_b = y_b_all == c
                if mask_a.sum() < 5 or mask_b.sum() < 5:
                    logger.info("    Class %d: insufficient samples (n_a=%d, n_b=%d) — skipping", c, mask_a.sum(), mask_b.sum())
                    continue
                feat_a = X_a_norm[mask_a]
                feat_b = X_b_norm[mask_b]

                # Subsample per class
                if len(feat_a) > SOS_MAX_CLASS_SAMPLES:
                    idx = rng.choice(len(feat_a), SOS_MAX_CLASS_SAMPLES, replace=False)
                    feat_a = feat_a[idx]
                if len(feat_b) > SOS_MAX_CLASS_SAMPLES:
                    idx = rng.choice(len(feat_b), SOS_MAX_CLASS_SAMPLES, replace=False)
                    feat_b = feat_b[idx]

                # Per-feature 1D Wasserstein distance, then average
                w_dists = np.array([
                    wasserstein_distance(feat_a[:, f], feat_b[:, f])
                    for f in range(n_features)
                ])
                mean_w = float(w_dists.mean())
                sos_c = max(0.0, 1.0 - mean_w / max_possible_w_per_feature)
                sos_per_class.append(sos_c)
                class_name = CLASS_NAMES_7[c] if c < len(CLASS_NAMES_7) else f"class_{c}"
                logger.info(
                    "    %s: mean_W=%.4f  SOS=%.4f  (n_a=%d, n_b=%d)",
                    class_name, mean_w, sos_c, len(feat_a), len(feat_b),
                )

            sos = float(np.mean(sos_per_class)) if sos_per_class else 0.0
            results[(name_a, name_b)] = sos
            logger.info("    → Macro SOS: %.4f", sos)

    return results


def compute_dic(
    dos: dict[tuple[str, str], float],
    lcs: dict[tuple[str, str], float],
    sos: dict[tuple[str, str], float],
    oracle_mf1: float,
) -> dict[tuple[str, str], float]:
    """
    DIC (Dataset Incompatibility Coefficient).

    DIC = Oracle MF1 × min(1.0, LCS / 0.80, DOS / 0.30, SOS / 0.60)

    Returns dict mapping (ds1, ds2) -> DIC.
    """
    results: dict[tuple[str, str], float] = {}
    all_pairs = set(dos.keys()) | set(lcs.keys()) | set(sos.keys())

    for pair in all_pairs:
        d = dos.get(pair, 0.0)
        l = lcs.get(pair, 0.0)
        s = sos.get(pair, 0.0)
        ratio = min(1.0, l / 0.80, d / 0.30, s / 0.60)
        dic = oracle_mf1 * ratio
        results[pair] = dic
        logger.info(
            "  DIC %s <-> %s: DOS=%.4f LCS=%.4f SOS=%.4f ratio=%.4f DIC=%.4f",
            *pair, d, l, s, ratio, dic,
        )

    return results


def compute_oracle_mf1(
    datasets: dict[str, np.ndarray],
    dataset_labels: dict[str, np.ndarray],
) -> float:
    """
    Oracle Macro F1: best in-distribution Macro F1 across all collections.

    Trains a RandomForest classifier within each dataset (5-fold CV),
    reports the best Macro F1 as the Oracle.
    """
    best_mf1 = 0.0
    best_dataset = None

    for name in datasets:
        X = datasets[name]
        y = dataset_labels[name].astype(int)
        logger.info("  Oracle MF1: training in-distribution classifier for %s (%d samples)", name, len(X))

        # Subsample large datasets
        if len(X) > 50_000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X), 50_000, replace=False)
            X = X[idx]
            y = y[idx]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model = RandomForestClassifier(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1,
        )

        # Use cross_val_score with f1_macro
        scores = cross_val_score(
            model, X_scaled, y, cv=cv, scoring="f1_macro", n_jobs=-1,
        )
        mean_mf1 = float(scores.mean())
        logger.info("    → MF1: %.4f ± %.4f", mean_mf1, scores.std())

        if mean_mf1 > best_mf1:
            best_mf1 = mean_mf1
            best_dataset = name

    logger.info("  Oracle MF1: %.4f (from %s)", best_mf1, best_dataset)
    return best_mf1


def load_and_prepare_datasets() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Load all 4 datasets via the existing MultiDatasetLoader pipeline.

    Returns (features_dict, labels_dict) where keys are dataset names.
    """
    logger.info("Loading and harmonizing datasets via MultiDatasetLoader...")
    loader = MultiDatasetLoader(random_state=42)

    nsl_kdd, unsw, cicids, ton_iot, *_ = loader.load_and_harmonize_all()

    # TON-IoT may be None if only test.csv exists (no train.csv)
    if ton_iot is None:
        logger.info("Attempting TON-IoT from test.csv (single-file corpus)...")
        test_path = loader.data_dir / "ton_iot" / "raw" / "test.csv"
        if test_path.exists():
            raw = pd.read_csv(test_path, low_memory=False)
            raw = loader._clean_ton_iot_frame(raw)
            ton_iot = loader.harmonize_ton_iot(raw)
            logger.info("  → TON-IoT from test.csv: %d samples, shape %s", len(ton_iot), ton_iot.shape)
        else:
            logger.warning("TON-IoT not available (neither train.csv nor test.csv found)")

    datasets: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}

    # Helper: extract X, y from harmonized DataFrame
    def extract(name: str, df: pd.DataFrame) -> None:
        feature_cols = [c for c in df.columns if c != "label"]
        datasets[name] = df[feature_cols].to_numpy(dtype=np.float32)
        labels[name] = df["label"].to_numpy(dtype=np.int64)
        logger.info(
            "  %s: %d samples, %d features, classes=%s",
            name, len(datasets[name]), datasets[name].shape[1],
            sorted(set(labels[name].astype(int))),
        )

    extract("NSL-KDD", nsl_kdd)
    extract("UNSW-NB15", unsw)

    if cicids is not None:
        extract("CICIDS2018", cicids)
    else:
        logger.warning("CICIDS2018 not available — will be excluded from analysis")

    if ton_iot is not None:
        extract("TON-IoT", ton_iot)
    else:
        logger.warning("TON-IoT not available — will be excluded from analysis")

    return datasets, labels


def print_quality_report(
    datasets: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    dos: dict[tuple[str, str], float],
    lcs: dict[tuple[str, str], float],
    sos: dict[tuple[str, str], float],
    dic: dict[tuple[str, str], float],
    oracle_mf1: float,
) -> None:
    """Print a formatted quality report similar to the Phase 36 template."""
    names = list(datasets.keys())

    print()
    print("=" * 72)
    print("  PHASE 36 QUALITY METRICS — PROTOTYPE VALIDATION REPORT")
    print("=" * 72)
    print(f"  Datasets: {', '.join(names)}")
    print(f"  Canonical features: {datasets[names[0]].shape[1]}")
    print(f"  Oracle MF1: {oracle_mf1:.4f}")
    print()

    # Per-dataset summary
    print("-" * 72)
    print("  Dataset Summary")
    print("-" * 72)
    for name in names:
        X, y = datasets[name], labels[name]
        classes = np.bincount(y.astype(int))
        class_dist = ", ".join(
            f"{CLASS_NAMES_7[i] if i < len(CLASS_NAMES_7) else f'C{i}'}={c}"
            for i, c in enumerate(classes) if c > 0
        )
        print(f"  {name:12s}: {len(X):>8,d} samples, {X.shape[1]} features, [{class_dist}]")
    print()

    # Pairwise metrics table
    print("-" * 72)
    print("  Pairwise Quality Metrics")
    print("-" * 72)
    print(f"  {'Pair':24s} {'DOS':>8s} {'LCS':>8s} {'SOS':>8s} {'DIC':>8s}  {'Status':>8s}")
    print(f"  {'-'*24} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  {'-'*8}")

    for pair in sorted(dos.keys()):
        d = dos.get(pair, 0.0)
        l = lcs.get(pair, 0.0)
        s = sos.get(pair, 0.0)
        di = dic.get(pair, 0.0)

        pass_dos = d >= 0.30
        pass_lcs = l >= 0.80
        pass_sos = s >= 0.60
        pass_dic = di >= 0.50
        passes = all([pass_dos, pass_lcs, pass_sos, pass_dic])
        status = "PASS" if passes else "FAIL"

        pair_label = f"{pair[0]:10s} ↔ {pair[1]:10s}"
        print(
            f"  {pair_label:24s} {d:8.4f} {l:8.4f} {s:8.4f} {di:8.4f}  {status:>8s}"
        )
    print()

    # Gate summary
    print("-" * 72)
    print("  Gate Summary")
    print("-" * 72)
    avg_dos = np.mean(list(dos.values())) if dos else 0
    avg_lcs = np.mean(list(lcs.values())) if lcs else 0
    avg_sos = np.mean(list(sos.values())) if sos else 0
    avg_dic = np.mean(list(dic.values())) if dic else 0

    print(f"  Metric     Average    Target     Result")
    print(f"  {'-----':12s} {'-------':>8s} {'-------':>8s} {'-------':>8s}")
    print(f"  {'DOS':12s} {avg_dos:8.4f} {'≥ 0.30':>8s} {'PASS' if avg_dos >= 0.30 else 'FAIL':>8s}")
    print(f"  {'LCS':12s} {avg_lcs:8.4f} {'≥ 0.80':>8s} {'PASS' if avg_lcs >= 0.80 else 'FAIL':>8s}")
    print(f"  {'SOS':12s} {avg_sos:8.4f} {'≥ 0.60':>8s} {'PASS' if avg_sos >= 0.60 else 'FAIL':>8s}")
    print(f"  {'DIC':12s} {avg_dic:8.4f} {'≥ 0.50':>8s} {'PASS' if avg_dic >= 0.50 else 'FAIL':>8s}")
    print()

    # Comparison to existing benchmarks
    print("-" * 72)
    print("  Comparison to Existing Benchmarks (from QUALITY_METRICS.md §6)")
    print("-" * 72)
    print(f"  {'Metric':12s} {'Phase 30-34 Avg':>16s} {'Phase 36 Target':>16s} {'This Prototype Avg':>18s}")
    print(f"  {'-----':12s} {'---------------':>16s} {'---------------':>16s} {'-----------------':>18s}")
    print(f"  {'DOS':12s} {'~0.01':>16s} {'≥ 0.30':>16s} {avg_dos:18.4f}")
    print(f"  {'LCS':12s} {'~0.35':>16s} {'≥ 0.80':>16s} {avg_lcs:18.4f}")
    print(f"  {'SOS':12s} {'~0.25':>16s} {'≥ 0.60':>16s} {avg_sos:18.4f}")
    print(f"  {'DIC':12s} {'0.37':>16s} {'≥ 0.50':>16s} {avg_dic:18.4f}")
    print()

    # Caveats
    print("-" * 72)
    print("  Caveats & Deviations from Phase 36 Spec")
    print("-" * 72)
    print("  1. Features: 17 canonical (not 22 as Phase 36 specifies)")
    print("  2. LCS: computed at class-family level, not Level-2 attack")
    print("     types. Overestimates LCS vs spec (families are broader).")
    print("  3. SOS: per-feature 1D Wasserstein averaged across 17 features.")
    print("     Max_possible_W=6.0σ per feature (theoretical bound for")
    print("     standardized data). Does not capture multivariate interactions.")
    print("  4. Oracle MF1: RandomForest (not the full HelixIDS-Full")
    print("     model pipeline). Tractable but not exhaustive.")
    print("  5. TON-IoT loaded from test.csv (single-file corpus).")
    print()

    print("=" * 72)
    print()


def main():
    t_start = time.time()

    # ── Step 1: Load datasets ────────────────────────────────────────────
    logger.info("Step 1/4: Loading datasets...")
    datasets, labels = load_and_prepare_datasets()

    if len(datasets) < 2:
        logger.error("Need at least 2 datasets to compute pairwise metrics")
        sys.exit(1)

    # ── Step 2: Oracle MF1 ────────────────────────────────────────────────
    logger.info("Step 2/4: Computing Oracle MF1...")
    oracle_mf1 = compute_oracle_mf1(datasets, labels)

    # ── Step 3: Pairwise metrics ──────────────────────────────────────────
    logger.info("Step 3/4: Computing pairwise DOS...")
    dos_results = compute_pairwise_dos(datasets, labels)

    logger.info("Computing pairwise LCS...")
    lcs_results = compute_lcs(labels)

    logger.info("Computing pairwise SOS...")
    sos_results = compute_sos(datasets, labels)

    # ── Step 4: DIC ──────────────────────────────────────────────────────
    logger.info("Step 4/4: Computing DIC...")
    dic_results = compute_dic(dos_results, lcs_results, sos_results, oracle_mf1)

    # ── Report ────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print_quality_report(datasets, labels, dos_results, lcs_results, sos_results, dic_results, oracle_mf1)
    logger.info("Total time: %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
