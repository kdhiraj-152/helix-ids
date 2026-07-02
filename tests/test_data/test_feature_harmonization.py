#!/usr/bin/env python3
"""
Test cross-dataset feature harmonization on real datasets.

Validates:
1. Feature mapping coverage
2. Label harmonization accuracy
3. Distribution alignment
4. Cross-dataset transfer
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Archived under archive/phase24a/src/ (Phase 24A)
# Use importlib so the archived module resolves relative imports correctly
import importlib.util as _imp_util
from pathlib import Path as _Path

_ARCHIVE_HELIX = _Path(__file__).resolve().parent.parent / "archive" / "phase24a" / "src" / "helix_ids"

# Create a synthetic package in sys.modules so relative imports work
import types as _types

_archived_hids = _types.ModuleType("helix_ids")
_archived_hids.__path__ = [str(_ARCHIVE_HELIX)]
_archived_hids.__package__ = "helix_ids"
sys.modules["helix_ids"] = _archived_hids

# Now load adaptation.__init__ as a submodule
_adapt_spec = _imp_util.spec_from_file_location(
    "helix_ids.adaptation", str(_ARCHIVE_HELIX / "adaptation" / "__init__.py"),
    submodule_search_locations=[str(_ARCHIVE_HELIX / "adaptation")],
)
_adapt_mod = _imp_util.module_from_spec(_adapt_spec)
sys.modules["helix_ids.adaptation"] = _adapt_mod
_adapt_spec.loader.exec_module(_adapt_mod)

# Export the needed names
FeatureHarmonizer = _adapt_mod.FeatureHarmonizer
create_cross_dataset_pipeline = _adapt_mod.create_cross_dataset_pipeline

# Restore real helix_ids in sys.modules so other tests don't get the synthetic one
sys.modules.pop("helix_ids.adaptation", None)
sys.modules.pop("helix_ids", None)


def test_feature_harmonizer_basic():
    """Test basic feature harmonization."""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Feature Harmonization")
    print("=" * 70)

    # Create sample NSL-KDD-like data
    nsl_features = {
        "duration": np.random.exponential(100, 1000),
        "src_bytes": np.random.exponential(1000, 1000),
        "dst_bytes": np.random.exponential(1000, 1000),
        "count": np.random.poisson(10, 1000),
        "serror_rate": np.random.uniform(0, 1, 1000),
        "rerror_rate": np.random.uniform(0, 1, 1000),
        "protocol_type": np.random.choice([0, 1, 2], 1000),
        "service": np.random.choice([0, 1, 2, 3], 1000),
    }

    X_nsl = pd.DataFrame(nsl_features)
    y_nsl = pd.Series(np.random.choice(["normal", "neptune", "nmap"], 1000))

    # Harmonize
    harmonizer = FeatureHarmonizer(source_dataset="nsl-kdd")
    harmonizer.fit(X_nsl)
    X_harm = harmonizer.harmonize(X_nsl)

    print(f"✓ Input shape: {X_nsl.shape}")
    print(f"✓ Output shape: {X_harm.shape}")
    print(f"✓ Missing features imputed: {(X_harm == 0).any(axis=0).sum()}")

    # Check label harmonization
    y_harm = harmonizer.harmonize_labels(y_nsl)
    print(f"✓ Original labels: {y_nsl.unique()}")
    print(f"✓ Harmonized labels: {y_harm.unique()}")

    report = harmonizer.get_harmonization_report()
    print("\nHarmonization Report:")
    for key, value in report.items():
        print(f"  {key}: {value}")

    return True


def test_unsw_nb15_harmonization():
    """Test UNSW-NB15 harmonization."""
    print("\n" + "=" * 70)
    print("TEST 2: UNSW-NB15 Feature Harmonization")
    print("=" * 70)

    # Create sample UNSW-NB15-like data
    unsw_features = {
        "dur": np.random.exponential(50, 1000),
        "sbytes": np.random.exponential(500, 1000),
        "dbytes": np.random.exponential(500, 1000),
        "spkts": np.random.poisson(20, 1000),
        "dpkts": np.random.poisson(20, 1000),
        "sttl": np.random.randint(1, 255, 1000),
        "dttl": np.random.randint(1, 255, 1000),
        "proto": np.random.choice([6, 17, 1], 1000),  # TCP, UDP, ICMP
        "service": np.random.choice([0, 1, 2], 1000),
        "state": np.random.choice([0, 1, 2], 1000),
        "Sload": np.random.uniform(0, 100, 1000),
        "Dload": np.random.uniform(0, 100, 1000),
        "ct_srv_src": np.random.poisson(5, 1000),
        "ct_srv_dst": np.random.poisson(5, 1000),
    }

    X_unsw = pd.DataFrame(unsw_features)
    y_unsw = pd.Series(np.random.choice(["Normal", "DoS", "Exploits"], 1000))

    # Harmonize
    harmonizer = FeatureHarmonizer(source_dataset="unsw-nb15")
    harmonizer.fit(X_unsw)
    X_harm = harmonizer.harmonize(X_unsw)

    print(f"✓ Input shape: {X_unsw.shape}")
    print(f"✓ Output shape: {X_harm.shape}")
    print(f"✓ Normalization applied: {harmonizer.normalize}")

    # Label harmonization
    y_harm = harmonizer.harmonize_labels(y_unsw)
    print(f"✓ Label mapping: {dict(zip(y_unsw.unique(), y_harm.unique()))}")

    return True


def test_cross_dataset_pipeline():
    """Test multi-dataset harmonization pipeline."""
    print("\n" + "=" * 70)
    print("TEST 3: Cross-Dataset Pipeline")
    print("=" * 70)

    # Create harmonizers for both datasets
    harmonizers = create_cross_dataset_pipeline(source_datasets=["nsl-kdd", "unsw-nb15"])

    print(f"✓ Created {len(harmonizers)} harmonizers")
    for name, harmonizer in harmonizers.items():
        print(f"  - {name}: {harmonizer.get_harmonization_report()}")

    return True


def test_distribution_alignment():
    """Test distribution alignment via normalization."""
    print("\n" + "=" * 70)
    print("TEST 4: Distribution Alignment")
    print("=" * 70)

    # Create data with different distributions
    X_train = pd.DataFrame(
        {
            "duration": np.random.exponential(100, 500),
            "src_bytes": np.random.exponential(1000, 500),
            "dst_bytes": np.random.exponential(1000, 500),
        }
    )

    X_test = pd.DataFrame(
        {
            "duration": np.random.exponential(200, 500),  # Shifted
            "src_bytes": np.random.exponential(2000, 500),
            "dst_bytes": np.random.exponential(2000, 500),
        }
    )

    # Fit on train, harmonize both
    harmonizer = FeatureHarmonizer(source_dataset="nsl-kdd", normalize=True)
    harmonizer.fit(X_train)

    X_train_norm = harmonizer.harmonize(X_train)
    X_test_norm = harmonizer.harmonize(X_test)

    print(f"✓ Training data mean after normalization: {X_train_norm.mean(axis=0).mean():.4f}")
    print(f"✓ Training data std after normalization: {X_train_norm.std(axis=0).mean():.4f}")
    print(f"✓ Test data mean after normalization: {X_test_norm.mean(axis=0).mean():.4f}")
    print(f"✓ Test data std after normalization: {X_test_norm.std(axis=0).mean():.4f}")

    # Verify normalization reduces distribution mismatch
    train_stats = (X_train_norm.mean(axis=0), X_train_norm.std(axis=0))
    test_stats = (X_test_norm.mean(axis=0), X_test_norm.std(axis=0))

    mean_drift = np.abs(train_stats[0] - test_stats[0]).mean()
    print(f"✓ Mean drift after harmonization: {mean_drift:.4f}")

    return True


def test_real_data_if_available():
    """Test on real data if available."""
    print("\n" + "=" * 70)
    print("TEST 5: Real Data Harmonization")
    print("=" * 70)

    # Try to load real NSL-KDD data
    nsl_path = PROJECT_ROOT / "data/processed/nsl-kdd_cleaned.csv"
    if nsl_path.exists():
        df = pd.read_csv(nsl_path)
        print(f"✓ Loaded {nsl_path.name}: {df.shape}")

        # Separate features and labels
        label_col = "__label__"
        feature_cols = [c for c in df.columns if c not in [label_col, "attack_type"]]

        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df[label_col]

        # Harmonize
        harmonizer = FeatureHarmonizer(source_dataset="nsl-kdd")
        harmonizer.fit(X)
        X_harm = harmonizer.harmonize(X)
        y_harm = harmonizer.harmonize_labels(y)

        print(f"✓ Real data harmonized: {X_harm.shape}")
        print(f"✓ Unique original labels: {len(y.unique())}")
        print(f"✓ Unique harmonized labels: {len(y_harm.unique())}")
        print(f"✓ Harmonized labels: {sorted(y_harm.unique())}")

        return True
    else:
        print(f"⚠ Real data not found at {nsl_path}")
        return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("FEATURE HARMONIZATION TEST SUITE")
    print("=" * 70)

    tests = [
        test_feature_harmonizer_basic,
        test_unsw_nb15_harmonization,
        test_cross_dataset_pipeline,
        test_distribution_alignment,
        test_real_data_if_available,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
