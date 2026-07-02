"""Tests for FeatureEngineer and error rate consolidation.

Covers:
  - FeatureEngineer initialization
  - get_attack_features / get_feature_importance
  - extract_attack_features
  - throughput features computation
  - cross-dataset alignment
  - normalization (standard, minmax, robust)
  - error rate consolidation
  - schema helpers
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from helix_ids.data.feature_engineering import (
    ATTACK_FEATURES,
    CICIDS_TO_NSL_MAPPING,
    CONSOLIDATED_ERROR_FEATURES,
    FEATURE_IMPORTANCE,
    NSL_KDD_SCHEMA,
    NSL_KDD_SCHEMA_CONSOLIDATED,
    REDUNDANT_ERROR_FEATURES,
    THROUGHPUT_FEATURES,
    UNSW_TO_NSL_MAPPING,
    ErrorRateConsolidationConfig,
    FeatureEngineer,
    consolidate_error_features,
    get_schema_with_error_consolidation,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def nsl_sample() -> pd.DataFrame:
    """Small NSL-KDD-like DataFrame with all 41 features."""
    rng = np.random.default_rng(42)
    data = {col: rng.random(20).astype(np.float32) for col in NSL_KDD_SCHEMA}
    return pd.DataFrame(data)


@pytest.fixture
def unsw_sample() -> pd.DataFrame:
    """Small UNSW-NB15-like DataFrame with mapped columns."""
    return pd.DataFrame(
        {
            "sbytes": [100, 200, 300],
            "dbytes": [50, 150, 250],
            "sttl": [64, 128, 255],
            "ct_srv_src": [1, 2, 3],
            "ct_dst_ltm": [5, 10, 15],
            "label": [0, 1, 0],
        }
    )


@pytest.fixture
def engineer() -> FeatureEngineer:
    return FeatureEngineer()


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level constants
# ═══════════════════════════════════════════════════════════════════════════════


class TestModuleConstants:
    def test_throughput_features(self) -> None:
        """THROUGHPUT_FEATURES has expected entries."""
        assert "bytes_per_sec" in THROUGHPUT_FEATURES
        assert "packets_per_sec" in THROUGHPUT_FEATURES

    def test_nsl_schema_length(self) -> None:
        """NSL-KDD schema has 41 features."""
        assert len(NSL_KDD_SCHEMA) == 41

    def test_nsl_consolidated_length(self) -> None:
        """Consolidated schema removes 8 error features, adds 2 = 35."""
        assert len(NSL_KDD_SCHEMA_CONSOLIDATED) == 35

    def test_redundant_error_features(self) -> None:
        """REDUNDANT_ERROR_FEATURES has 8 entries."""
        assert len(REDUNDANT_ERROR_FEATURES) == 8

    def test_consolidated_error_features(self) -> None:
        """CONSOLIDATED_ERROR_FEATURES has 2 entries."""
        assert len(CONSOLIDATED_ERROR_FEATURES) == 2

    def test_unsw_mapping(self) -> None:
        """UNSW-to-NSL mapping has expected entries."""
        assert UNSW_TO_NSL_MAPPING["sbytes"] == "src_bytes"
        assert UNSW_TO_NSL_MAPPING["ct_srv_src"] == "srv_count"

    def test_cicids_mapping(self) -> None:
        """CICIDS-to-NSL mapping has expected entries."""
        assert CICIDS_TO_NSL_MAPPING["Flow Duration"] == "duration"
        assert CICIDS_TO_NSL_MAPPING["Total Fwd Packets"] == "count"


# ═══════════════════════════════════════════════════════════════════════════════
# ErrorRateConsolidationConfig
# ═══════════════════════════════════════════════════════════════════════════════


class TestErrorRateConsolidationConfig:
    def test_default_weights(self) -> None:
        """Default weights are equal."""
        cfg = ErrorRateConsolidationConfig()
        weights = cfg.get_serror_weights()
        assert all(v == pytest.approx(0.25) for v in weights.values())

    def test_custom_weights(self) -> None:
        """Custom weights propagate."""
        cfg = ErrorRateConsolidationConfig(
            weights={"serror_rate": 0.5, "srv_serror_rate": 0.5}
        )
        w = cfg.get_serror_weights()
        assert w["serror_rate"] == 0.5

    def test_disabled_returns_original(self, nsl_sample: pd.DataFrame) -> None:
        """When config.enabled=False, returns data unchanged."""
        cfg = ErrorRateConsolidationConfig(enabled=False)
        result = consolidate_error_features(nsl_sample, config=cfg)
        assert result.shape == nsl_sample.shape


# ═══════════════════════════════════════════════════════════════════════════════
# consolidate_error_features
# ═══════════════════════════════════════════════════════════════════════════════


class TestConsolidateErrorFeatures:
    def test_adds_weighted_serror(self, nsl_sample: pd.DataFrame) -> None:
        """Consolidation adds weighted_serror column."""
        result = consolidate_error_features(nsl_sample)
        assert "weighted_serror" in result.columns

    def test_adds_weighted_rerror(self, nsl_sample: pd.DataFrame) -> None:
        """Consolidation adds weighted_rerror column."""
        result = consolidate_error_features(nsl_sample)
        assert "weighted_rerror" in result.columns

    def test_drops_original_by_default(self, nsl_sample: pd.DataFrame) -> None:
        """Original error features are dropped by default."""
        result = consolidate_error_features(nsl_sample)
        for f in REDUNDANT_ERROR_FEATURES:
            assert f not in result.columns

    def test_keeps_original_when_drop_false(self, nsl_sample: pd.DataFrame) -> None:
        """When drop_original=False, original features remain."""
        result = consolidate_error_features(nsl_sample, drop_original=False)
        for f in REDUNDANT_ERROR_FEATURES:
            assert f in result.columns

    def test_partial_columns(self) -> None:
        """Works when only some error features present."""
        X = pd.DataFrame({"serror_rate": [0.1, 0.2, 0.3]})
        result = consolidate_error_features(X)
        assert "weighted_serror" in result.columns


# ═══════════════════════════════════════════════════════════════════════════════
# get_schema_with_error_consolidation
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetSchema:
    def test_consolidated(self) -> None:
        """Consolidated schema has 35 features."""
        assert len(get_schema_with_error_consolidation(True)) == 35

    def test_original(self) -> None:
        """Original schema has 41 features."""
        assert len(get_schema_with_error_consolidation(False)) == 41

    def test_returns_copy(self) -> None:
        """Returns a copy, not the original list."""
        s1 = get_schema_with_error_consolidation(True)
        s2 = get_schema_with_error_consolidation(True)
        assert s1 is not s2


# ═══════════════════════════════════════════════════════════════════════════════
# FeatureEngineer
# ═══════════════════════════════════════════════════════════════════════════════


class TestFeatureEngineerInit:
    def test_default_init(self) -> None:
        """Default FeatureEngineer loads expected mappings."""
        eng = FeatureEngineer()
        assert "DoS" in eng.attack_features
        assert "Probe" in eng.attack_features
        assert not eng._fitted

    def test_custom_attack_features(self) -> None:
        """Custom attack_features override defaults."""
        custom = {"CustomAttack": ["src_bytes", "dst_bytes"]}
        eng = FeatureEngineer(attack_features=custom)
        assert "CustomAttack" in eng.attack_features
        assert "DoS" not in eng.attack_features


class TestGetAttackFeatures:
    def test_known_attack(self, engineer: FeatureEngineer) -> None:
        """Known attack type returns feature list."""
        features = engineer.get_attack_features("DoS")
        assert isinstance(features, list)
        assert "src_bytes" in features

    def test_unknown_attack_raises(self, engineer: FeatureEngineer) -> None:
        """Unknown attack type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown attack type"):
            engineer.get_attack_features("UnknownAttack")

    def test_returns_copy(self, engineer: FeatureEngineer) -> None:
        """Returns a copy, not internal reference."""
        f1 = engineer.get_attack_features("DoS")
        f2 = engineer.get_attack_features("DoS")
        assert f1 is not f2


class TestGetFeatureImportance:
    def test_specific_attack(self, engineer: FeatureEngineer) -> None:
        """Returns importance for a specific attack."""
        imp = engineer.get_feature_importance("U2R")
        assert "num_root" in imp
        assert imp["num_root"] == 0.35

    def test_all_attacks(self, engineer: FeatureEngineer) -> None:
        """Returns combined importance across all attacks."""
        imp = engineer.get_feature_importance()
        assert isinstance(imp, dict)

    def test_top_k(self, engineer: FeatureEngineer) -> None:
        """top_k limits returned features."""
        imp = engineer.get_feature_importance(top_k=3)
        assert len(imp) <= 3

    def test_unknown_raises(self, engineer: FeatureEngineer) -> None:
        """Unknown attack raises ValueError."""
        with pytest.raises(ValueError, match="Unknown attack type"):
            engineer.get_feature_importance("Bogus")


class TestExtractAttackFeatures:
    def test_extracts_columns(self, engineer: FeatureEngineer, nsl_sample: pd.DataFrame) -> None:
        """Extracting DoS features returns only DoS-relevant columns."""
        extracted = engineer.extract_attack_features(nsl_sample, "DoS")
        assert extracted.shape[1] > 0
        assert extracted.shape[1] < nsl_sample.shape[1]

    def test_no_available_cols_raises(self, engineer: FeatureEngineer) -> None:
        """No matching columns raises ValueError."""
        X = pd.DataFrame({"unrelated": [1.0, 2.0]})
        with pytest.raises(ValueError, match="None of the required features"):
            engineer.extract_attack_features(X, "DoS")


class TestComputeThroughputFeatures:
    def test_adds_throughput_columns(self, nsl_sample: pd.DataFrame) -> None:
        """Throughput computation adds expected columns."""
        eng = FeatureEngineer()
        result = eng.compute_throughput_features(nsl_sample)
        for col in THROUGHPUT_FEATURES:
            assert col in result.columns

    def test_no_inf_values(self, nsl_sample: pd.DataFrame) -> None:
        """Throughput features have no inf values."""
        eng = FeatureEngineer()
        result = eng.compute_throughput_features(nsl_sample)
        for col in THROUGHPUT_FEATURES:
            assert not result[col].isin([np.inf, -np.inf]).any()

    def test_missing_duration_defaults_to_min(self) -> None:
        """When duration column is missing, uses min_duration default."""
        X = pd.DataFrame({"src_bytes": [100, 200]})
        eng = FeatureEngineer()
        result = eng.compute_throughput_features(X)
        assert "bytes_per_sec" in result.columns


class TestNormalize:
    @pytest.mark.parametrize("method", ["standard", "minmax", "robust"])
    def test_normalize_preserves_shape(self, engineer: FeatureEngineer, nsl_sample: pd.DataFrame, method: str) -> None:
        """Normalization preserves DataFrame shape."""
        result = engineer.normalize(nsl_sample, method=method)
        assert result.shape == nsl_sample.shape

    def test_standard_normalize(self, engineer: FeatureEngineer, nsl_sample: pd.DataFrame) -> None:
        """Standard normalization produces ~zero mean, unit variance (float32 approx)."""
        result = engineer.normalize(nsl_sample, method="standard")
        col = nsl_sample.columns[0]
        assert abs(result[col].mean()) < 1e-5
        assert abs(result[col].std() - 1.0) < 0.06

    def test_minmax_range(self, engineer: FeatureEngineer, nsl_sample: pd.DataFrame) -> None:
        """MinMax normalization produces values in [0, 1]."""
        result = engineer.normalize(nsl_sample, method="minmax")
        for col in result.columns:
            assert result[col].min() >= -1e-6
            assert result[col].max() <= 1.0 + 1e-6

    def test_transform_without_fit_raises(self, engineer: FeatureEngineer, nsl_sample: pd.DataFrame) -> None:
        """Calling normalize with fit=False before fitting raises."""
        with pytest.raises(ValueError, match="Scaler not fitted"):
            engineer.normalize(nsl_sample, method="standard", fit=False)

    def test_fit_then_transform(self, engineer: FeatureEngineer, nsl_sample: pd.DataFrame) -> None:
        """Fitted scaler can transform new data."""
        engineer.normalize(nsl_sample, method="standard", fit=True)
        new_data = nsl_sample * 2.0
        result = engineer.normalize(new_data, method="standard", fit=False)
        assert result.shape == new_data.shape

    def test_unknown_method_raises(self, engineer: FeatureEngineer, nsl_sample: pd.DataFrame) -> None:
        """Unknown normalization method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown normalization method"):
            engineer.normalize(nsl_sample, method="unknown")

    def test_numpy_array_input(self, engineer: FeatureEngineer) -> None:
        """Numpy array input returns numpy array."""
        X = np.random.default_rng(42).random((10, 4)).astype(np.float32)
        result = engineer.normalize(X, method="standard")
        assert isinstance(result, np.ndarray)


class TestAlignFeatures:
    def test_unsw_alignment(self, engineer: FeatureEngineer, unsw_sample: pd.DataFrame) -> None:
        """UNSW features align to NSL-KDD schema."""
        result = engineer.align_features(unsw_sample, source_dataset="unsw")
        assert result.shape[0] == 3
        assert "src_bytes" in result.columns

    def test_nsl_alignment(self, engineer: FeatureEngineer, nsl_sample: pd.DataFrame) -> None:
        """NSL alignment preserves columns."""
        result = engineer.align_features(nsl_sample, source_dataset="nsl")
        assert set(result.columns) == set(NSL_KDD_SCHEMA)

    def test_fill_zeros(self, engineer: FeatureEngineer, unsw_sample: pd.DataFrame) -> None:
        """Missing columns filled with zeros."""
        result = engineer.align_features(unsw_sample, source_dataset="unsw")
        # 'hot' is not mapped from UNSW - should be 0
        if "hot" in result.columns:
            assert (result["hot"] == 0).all()

    def test_unknown_dataset_raises(self, engineer: FeatureEngineer) -> None:
        """Unknown source dataset raises ValueError."""
        X = pd.DataFrame({"a": [1.0]})
        with pytest.raises(ValueError, match="Unknown source dataset"):
            engineer.align_features(X, source_dataset="unknown")


class TestAttackFeaturesAndImportance:
    def test_all_attacks_defined(self) -> None:
        """All 4 attack types have feature lists."""
        for attack in ["DoS", "Probe", "R2L", "U2R"]:
            assert attack in ATTACK_FEATURES
            assert len(ATTACK_FEATURES[attack]) > 0

    def test_all_attacks_have_importance(self) -> None:
        """All 4 attack types have importance rankings."""
        for attack in ["DoS", "Probe", "R2L", "U2R"]:
            assert attack in FEATURE_IMPORTANCE
            assert len(FEATURE_IMPORTANCE[attack]) > 0
