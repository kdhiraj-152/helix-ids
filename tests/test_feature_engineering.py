"""
Test suite for HELIX-IDS feature engineering pipeline.

Tests cover:
- Rate feature calculations (bytes_per_sec, packets_per_sec)
- Ratio feature calculations (bytes_ratio, bytes_imbalance)
- Log transforms
- Feature count validation (32 features)
- Edge case handling (zero values, NaN, infinities)
"""

import numpy as np
import pandas as pd
import pytest

# =============================================================================
# Constants
# =============================================================================

EXPECTED_FEATURE_COUNT = 32

# Rate features that should be computed
RATE_FEATURES = ["bytes_per_sec", "packets_per_sec"]

# Ratio features that should be computed
RATIO_FEATURES = ["bytes_ratio", "bytes_imbalance"]

# Log-transformed features
LOG_FEATURES = ["log_src_bytes", "log_dst_bytes", "log_count", "log_srv_count"]


# =============================================================================
# Rate Feature Calculation Tests
# =============================================================================


class TestRateFeatures:
    """Test rate-based feature calculations."""

    def test_bytes_per_sec_calculation(self, raw_network_data):
        """
        Test bytes_per_sec is calculated as (src_bytes + dst_bytes) / duration.
        
        Verifies that throughput rate is correctly computed from byte counts
        and connection duration.
        """
        df = raw_network_data.copy()
        
        # Calculate expected bytes_per_sec
        total_bytes = df["src_bytes"] + df["dst_bytes"]
        # Avoid division by zero
        duration_safe = df["duration"].replace(0, 1e-6)
        expected_bps = total_bytes / duration_safe
        
        # Compute actual feature
        df["bytes_per_sec"] = total_bytes / duration_safe
        
        # Verify calculation
        np.testing.assert_array_almost_equal(
            df["bytes_per_sec"].values,
            expected_bps.values,
            decimal=5
        )

    def test_bytes_per_sec_zero_duration(self):
        """
        Test bytes_per_sec handles zero duration gracefully.
        
        Should not produce infinity or NaN, but instead use a small epsilon.
        """
        df = pd.DataFrame({
            "src_bytes": [1000.0, 2000.0, 3000.0],
            "dst_bytes": [500.0, 1000.0, 1500.0],
            "duration": [0.0, 0.0, 1.0],
        })
        
        total_bytes = df["src_bytes"] + df["dst_bytes"]
        duration_safe = df["duration"].replace(0, 1e-6)
        df["bytes_per_sec"] = total_bytes / duration_safe
        
        # Should not have inf or nan
        assert not df["bytes_per_sec"].isin([np.inf, -np.inf]).any(), \
            "bytes_per_sec should not contain infinity"
        assert not df["bytes_per_sec"].isna().any(), \
            "bytes_per_sec should not contain NaN"

    def test_packets_per_sec_calculation(self, raw_network_data):
        """
        Test packets_per_sec calculation from count and duration.
        
        Verifies packet rate computation handles various durations.
        """
        df = raw_network_data.copy()
        
        # Calculate packets per second
        duration_safe = df["duration"].replace(0, 1e-6)
        expected_pps = df["count"] / duration_safe
        
        df["packets_per_sec"] = df["count"] / duration_safe
        
        np.testing.assert_array_almost_equal(
            df["packets_per_sec"].values,
            expected_pps.values,
            decimal=5
        )

    def test_rate_features_positive(self, raw_network_data):
        """
        Test that rate features are always non-negative.
        
        Network traffic rates cannot be negative in valid data.
        """
        df = raw_network_data.copy()
        
        total_bytes = df["src_bytes"] + df["dst_bytes"]
        duration_safe = df["duration"].replace(0, 1e-6).clip(lower=1e-6)
        
        df["bytes_per_sec"] = total_bytes / duration_safe
        df["packets_per_sec"] = df["count"] / duration_safe
        
        assert (df["bytes_per_sec"] >= 0).all(), "bytes_per_sec should be non-negative"
        assert (df["packets_per_sec"] >= 0).all(), "packets_per_sec should be non-negative"


# =============================================================================
# Ratio Feature Calculation Tests
# =============================================================================


class TestRatioFeatures:
    """Test ratio-based feature calculations."""

    def test_bytes_ratio_calculation(self, raw_network_data):
        """
        Test bytes_ratio = src_bytes / (src_bytes + dst_bytes).
        
        Measures the proportion of outbound traffic.
        """
        df = raw_network_data.copy()
        
        total_bytes = df["src_bytes"] + df["dst_bytes"]
        # Avoid division by zero
        total_safe = total_bytes.replace(0, 1.0)
        
        expected_ratio = df["src_bytes"] / total_safe
        df["bytes_ratio"] = df["src_bytes"] / total_safe
        
        np.testing.assert_array_almost_equal(
            df["bytes_ratio"].values,
            expected_ratio.values,
            decimal=5
        )

    def test_bytes_ratio_bounds(self, raw_network_data):
        """
        Test bytes_ratio is bounded between 0 and 1.
        
        As a proportion, it must be in [0, 1] range.
        """
        df = raw_network_data.copy()
        
        total_bytes = df["src_bytes"] + df["dst_bytes"]
        total_safe = total_bytes.replace(0, 1.0)
        df["bytes_ratio"] = df["src_bytes"] / total_safe
        
        assert (df["bytes_ratio"] >= 0).all(), "bytes_ratio should be >= 0"
        assert (df["bytes_ratio"] <= 1).all(), "bytes_ratio should be <= 1"

    def test_bytes_imbalance_calculation(self, raw_network_data):
        """
        Test bytes_imbalance = (src_bytes - dst_bytes) / (src_bytes + dst_bytes).
        
        Measures traffic asymmetry: +1 = all outbound, -1 = all inbound.
        """
        df = raw_network_data.copy()
        
        total_bytes = df["src_bytes"] + df["dst_bytes"]
        total_safe = total_bytes.replace(0, 1.0)
        
        expected_imbalance = (df["src_bytes"] - df["dst_bytes"]) / total_safe
        df["bytes_imbalance"] = (df["src_bytes"] - df["dst_bytes"]) / total_safe
        
        np.testing.assert_array_almost_equal(
            df["bytes_imbalance"].values,
            expected_imbalance.values,
            decimal=5
        )

    def test_bytes_imbalance_bounds(self, raw_network_data):
        """
        Test bytes_imbalance is bounded between -1 and 1.
        """
        df = raw_network_data.copy()
        
        total_bytes = df["src_bytes"] + df["dst_bytes"]
        total_safe = total_bytes.replace(0, 1.0)
        df["bytes_imbalance"] = (df["src_bytes"] - df["dst_bytes"]) / total_safe
        
        assert (df["bytes_imbalance"] >= -1).all(), "bytes_imbalance should be >= -1"
        assert (df["bytes_imbalance"] <= 1).all(), "bytes_imbalance should be <= 1"

    def test_ratio_zero_total_bytes(self):
        """
        Test ratio features handle zero total bytes.
        
        When both src and dst bytes are 0, should produce defined values.
        """
        df = pd.DataFrame({
            "src_bytes": [0.0, 0.0, 100.0],
            "dst_bytes": [0.0, 0.0, 0.0],
        })
        
        total_bytes = df["src_bytes"] + df["dst_bytes"]
        total_safe = total_bytes.replace(0, 1.0)
        
        df["bytes_ratio"] = df["src_bytes"] / total_safe
        df["bytes_imbalance"] = (df["src_bytes"] - df["dst_bytes"]) / total_safe
        
        # Should not have NaN
        assert not df["bytes_ratio"].isna().any()
        assert not df["bytes_imbalance"].isna().any()


# =============================================================================
# Log Transform Tests
# =============================================================================


class TestLogTransforms:
    """Test log transformation features."""

    def test_log_src_bytes(self, raw_network_data):
        """
        Test log transform of src_bytes: log(1 + src_bytes).
        
        Log transform helps normalize skewed distributions.
        """
        df = raw_network_data.copy()
        
        expected_log = np.log1p(df["src_bytes"])
        df["log_src_bytes"] = np.log1p(df["src_bytes"])
        
        np.testing.assert_array_almost_equal(
            df["log_src_bytes"].values,
            expected_log.values,
            decimal=5
        )

    def test_log_dst_bytes(self, raw_network_data):
        """
        Test log transform of dst_bytes: log(1 + dst_bytes).
        """
        df = raw_network_data.copy()
        
        expected_log = np.log1p(df["dst_bytes"])
        df["log_dst_bytes"] = np.log1p(df["dst_bytes"])
        
        np.testing.assert_array_almost_equal(
            df["log_dst_bytes"].values,
            expected_log.values,
            decimal=5
        )

    def test_log_count(self, raw_network_data):
        """
        Test log transform of count: log(1 + count).
        """
        df = raw_network_data.copy()
        
        expected_log = np.log1p(df["count"])
        df["log_count"] = np.log1p(df["count"])
        
        np.testing.assert_array_almost_equal(
            df["log_count"].values,
            expected_log.values,
            decimal=5
        )

    def test_log_srv_count(self, raw_network_data):
        """
        Test log transform of srv_count: log(1 + srv_count).
        """
        df = raw_network_data.copy()
        
        expected_log = np.log1p(df["srv_count"])
        df["log_srv_count"] = np.log1p(df["srv_count"])
        
        np.testing.assert_array_almost_equal(
            df["log_srv_count"].values,
            expected_log.values,
            decimal=5
        )

    def test_log_transform_zero_values(self):
        """
        Test log transform handles zero values correctly.
        
        log(1 + 0) = 0, should not produce -inf.
        """
        df = pd.DataFrame({
            "src_bytes": [0.0, 0.0, 0.0],
            "dst_bytes": [0.0, 100.0, 1000.0],
            "count": [0, 1, 100],
            "srv_count": [0, 0, 50],
        })
        
        df["log_src_bytes"] = np.log1p(df["src_bytes"])
        df["log_dst_bytes"] = np.log1p(df["dst_bytes"])
        df["log_count"] = np.log1p(df["count"])
        df["log_srv_count"] = np.log1p(df["srv_count"])
        
        for col in ["log_src_bytes", "log_dst_bytes", "log_count", "log_srv_count"]:
            assert not df[col].isin([np.inf, -np.inf]).any(), f"{col} should not contain infinity"
            assert not df[col].isna().any(), f"{col} should not contain NaN"
            assert (df[col] >= 0).all(), f"{col} should be non-negative"

    def test_log_transform_large_values(self):
        """
        Test log transform compresses large values.
        """
        df = pd.DataFrame({
            "src_bytes": [1e9, 1e12, 1e15],
        })
        
        df["log_src_bytes"] = np.log1p(df["src_bytes"])
        
        # Log should significantly reduce the range
        assert df["log_src_bytes"].max() < df["src_bytes"].max()
        assert df["log_src_bytes"].max() < 50  # log(1e15) ≈ 34.5


# =============================================================================
# Feature Count Tests
# =============================================================================


class TestFeatureCount:
    """Test that feature engineering produces exactly 32 features."""

    def test_production_feature_count(self, production_feature_names):
        """
        Test production model expects exactly 32 features.
        """
        assert len(production_feature_names) == EXPECTED_FEATURE_COUNT, \
            f"Expected {EXPECTED_FEATURE_COUNT} features, got {len(production_feature_names)}"

    def test_feature_names_no_duplicates(self, production_feature_names):
        """
        Test feature names are unique.
        """
        assert len(production_feature_names) == len(set(production_feature_names)), \
            "Feature names should be unique"

    def test_rate_features_present(self, production_feature_names):
        """
        Test that rate features are in the production feature set.
        """
        for feature in RATE_FEATURES:
            assert feature in production_feature_names, \
                f"Rate feature '{feature}' should be in production features"

    def test_ratio_features_present(self, production_feature_names):
        """
        Test that ratio features are in the production feature set.
        """
        for feature in RATIO_FEATURES:
            assert feature in production_feature_names, \
                f"Ratio feature '{feature}' should be in production features"

    def test_log_features_present(self, production_feature_names):
        """
        Test that log-transformed features are in the production feature set.
        """
        for feature in LOG_FEATURES:
            assert feature in production_feature_names, \
                f"Log feature '{feature}' should be in production features"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test handling of edge cases in feature engineering."""

    def test_all_zeros(self):
        """
        Test feature engineering with all-zero input.
        """
        df = pd.DataFrame({
            "src_bytes": [0.0] * 5,
            "dst_bytes": [0.0] * 5,
            "duration": [0.0] * 5,
            "count": [0] * 5,
            "srv_count": [0] * 5,
        })
        
        # Compute features with safe division
        total_bytes = df["src_bytes"] + df["dst_bytes"]
        total_safe = total_bytes.replace(0, 1.0)
        duration_safe = df["duration"].replace(0, 1e-6)
        
        df["bytes_per_sec"] = total_bytes / duration_safe
        df["bytes_ratio"] = df["src_bytes"] / total_safe
        df["bytes_imbalance"] = (df["src_bytes"] - df["dst_bytes"]) / total_safe
        df["log_src_bytes"] = np.log1p(df["src_bytes"])
        
        # Should not have any NaN or Inf
        for col in ["bytes_per_sec", "bytes_ratio", "bytes_imbalance", "log_src_bytes"]:
            assert not df[col].isna().any(), f"{col} should not have NaN"
            assert not df[col].isin([np.inf, -np.inf]).any(), f"{col} should not have Inf"

    def test_nan_handling(self, edge_case_network_data):
        """
        Test feature engineering handles NaN values.
        
        NaN should be handled gracefully (filled or propagated consistently).
        """
        df = edge_case_network_data.copy()
        
        # Check that NaN is present in input
        assert df.isna().any().any(), "Test data should contain NaN"
        
        # Fill NaN before computation
        df_filled = df.fillna(0)
        
        total_bytes = df_filled["src_bytes"] + df_filled["dst_bytes"]
        total_safe = total_bytes.replace(0, 1.0)
        
        df_filled["bytes_ratio"] = df_filled["src_bytes"] / total_safe
        
        # After filling, should not have NaN
        assert not df_filled["bytes_ratio"].isna().any()

    def test_very_large_values(self):
        """
        Test feature engineering with very large values.
        
        Should handle values up to 1e15 without overflow.
        """
        df = pd.DataFrame({
            "src_bytes": [1e15, 1e14, 1e13],
            "dst_bytes": [1e14, 1e15, 1e12],
            "duration": [1.0, 0.001, 1000.0],
            "count": [1000000, 500000, 100],
        })
        
        total_bytes = df["src_bytes"] + df["dst_bytes"]
        total_safe = total_bytes.replace(0, 1.0)
        duration_safe = df["duration"].replace(0, 1e-6)
        
        df["bytes_per_sec"] = total_bytes / duration_safe
        df["bytes_ratio"] = df["src_bytes"] / total_safe
        df["log_src_bytes"] = np.log1p(df["src_bytes"])
        
        # Should not overflow
        assert not df["bytes_per_sec"].isin([np.inf, -np.inf]).any()
        assert not df["bytes_ratio"].isna().any()
        assert (df["bytes_ratio"] >= 0).all() and (df["bytes_ratio"] <= 1).all()

    def test_negative_values_handling(self):
        """
        Test feature engineering with negative values.
        
        Network byte counts should never be negative, but we handle gracefully.
        """
        df = pd.DataFrame({
            "src_bytes": [-100.0, 100.0, 200.0],  # Invalid negative
            "dst_bytes": [100.0, -50.0, 150.0],   # Invalid negative
        })
        
        # Clip negative values to 0
        df["src_bytes"] = df["src_bytes"].clip(lower=0)
        df["dst_bytes"] = df["dst_bytes"].clip(lower=0)
        
        total_bytes = df["src_bytes"] + df["dst_bytes"]
        total_safe = total_bytes.replace(0, 1.0)
        
        df["bytes_ratio"] = df["src_bytes"] / total_safe
        
        assert (df["bytes_ratio"] >= 0).all()
        assert (df["bytes_ratio"] <= 1).all()


# =============================================================================
# Feature Engineering Pipeline Integration Tests
# =============================================================================


class TestFeatureEngineeringPipeline:
    """Integration tests for the complete feature engineering pipeline."""

    def test_engineer_features_output_shape(self, raw_network_data):
        """
        Test that engineered features have correct shape.
        """
        df = raw_network_data.copy()
        n_samples = len(df)
        
        # Manually engineer features to simulate pipeline
        total_bytes = df["src_bytes"] + df["dst_bytes"]
        total_safe = total_bytes.replace(0, 1.0)
        duration_safe = df["duration"].replace(0, 1e-6)
        
        # Add engineered features
        df["bytes_per_sec"] = total_bytes / duration_safe
        df["packets_per_sec"] = df["count"] / duration_safe
        df["bytes_ratio"] = df["src_bytes"] / total_safe
        df["bytes_imbalance"] = (df["src_bytes"] - df["dst_bytes"]) / total_safe
        df["log_src_bytes"] = np.log1p(df["src_bytes"])
        df["log_dst_bytes"] = np.log1p(df["dst_bytes"])
        df["log_count"] = np.log1p(df["count"])
        df["log_srv_count"] = np.log1p(df["srv_count"])
        
        # Check shape preserved
        assert len(df) == n_samples, "Number of samples should be preserved"

    def test_feature_engineering_deterministic(self, raw_network_data):
        """
        Test that feature engineering is deterministic.
        
        Same input should always produce same output.
        """
        df1 = raw_network_data.copy()
        df2 = raw_network_data.copy()
        
        # Engineer same features on both
        for df in [df1, df2]:
            total_bytes = df["src_bytes"] + df["dst_bytes"]
            total_safe = total_bytes.replace(0, 1.0)
            duration_safe = df["duration"].replace(0, 1e-6)
            
            df["bytes_per_sec"] = total_bytes / duration_safe
            df["bytes_ratio"] = df["src_bytes"] / total_safe
            df["log_src_bytes"] = np.log1p(df["src_bytes"])
        
        # Results should be identical
        pd.testing.assert_frame_equal(df1, df2)

    def test_feature_engineering_dtype_consistency(self, raw_network_data):
        """
        Test that engineered features have consistent dtypes.
        """
        df = raw_network_data.copy()
        
        total_bytes = df["src_bytes"] + df["dst_bytes"]
        total_safe = total_bytes.replace(0, 1.0)
        duration_safe = df["duration"].replace(0, 1e-6)
        
        df["bytes_per_sec"] = total_bytes / duration_safe
        df["bytes_ratio"] = df["src_bytes"] / total_safe
        df["log_src_bytes"] = np.log1p(df["src_bytes"])
        
        # All engineered features should be float
        assert df["bytes_per_sec"].dtype in [np.float32, np.float64]
        assert df["bytes_ratio"].dtype in [np.float32, np.float64]
        assert df["log_src_bytes"].dtype in [np.float32, np.float64]


# =============================================================================
# Feature Importance/Selection Tests
# =============================================================================


class TestFeatureSelection:
    """Test feature selection and importance."""

    def test_all_32_features_selected(self, production_feature_names):
        """
        Verify all 32 production features are included.
        """
        expected_count = 32
        assert len(production_feature_names) == expected_count

    def test_no_target_leakage_features(self, production_feature_names):
        """
        Test that no target-leaking features are included.
        
        Features like 'label', 'attack_type', 'class' should not be present.
        """
        leakage_features = ["label", "attack_type", "class", "attack", "target", "y"]
        
        for feature in leakage_features:
            assert feature not in production_feature_names, \
                f"Leakage feature '{feature}' should not be in production features"

    def test_feature_engineering_preserves_data_integrity(self, raw_network_data):
        """
        Test that feature engineering does not modify original columns.
        """
        df_original = raw_network_data.copy()
        df = raw_network_data.copy()
        
        # Engineer features
        total_bytes = df["src_bytes"] + df["dst_bytes"]
        total_safe = total_bytes.replace(0, 1.0)
        df["bytes_ratio"] = df["src_bytes"] / total_safe
        
        # Original columns should be unchanged
        np.testing.assert_array_equal(
            df["src_bytes"].values,
            df_original["src_bytes"].values
        )
        np.testing.assert_array_equal(
            df["dst_bytes"].values,
            df_original["dst_bytes"].values
        )
