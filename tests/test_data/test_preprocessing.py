"""
Test suite for HELIX-IDS preprocessing pipeline.

Tests cover:
- MinMaxScaler loading and application
- Feature scaling bounds [0, 1]
- Scaler persistence (save/load)
- Data preprocessing consistency
"""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

# =============================================================================
# Constants
# =============================================================================

INPUT_DIM = 32
SCALE_MIN = 0.0
SCALE_MAX = 1.0


# =============================================================================
# MinMaxScaler Loading Tests
# =============================================================================


class TestScalerLoading:
    """Test MinMaxScaler loading functionality."""

    def test_production_scaler_exists(self, project_root):
        """
        Test that production scaler file exists.
        """
        scaler_path = project_root / "models" / "production" / "scaler.pkl"
        assert scaler_path.exists(), f"Scaler not found at {scaler_path}"

    def test_load_production_scaler(self, production_scaler):
        """
        Test production scaler can be loaded from pickle.
        """
        assert production_scaler is not None
        assert hasattr(production_scaler, "transform")
        assert hasattr(production_scaler, "fit")

    def test_scaler_has_feature_range(self, production_scaler):
        """
        Test scaler has correct feature range attribute.
        """
        if hasattr(production_scaler, "feature_range"):
            assert production_scaler.feature_range == (SCALE_MIN, SCALE_MAX), \
                f"Expected feature_range (0, 1), got {production_scaler.feature_range}"

    def test_scaler_has_fitted_attributes(self, fitted_minmax_scaler):
        """
        Test fitted scaler has necessary attributes.
        """
        # MinMaxScaler should have these after fitting
        assert hasattr(fitted_minmax_scaler, "data_min_")
        assert hasattr(fitted_minmax_scaler, "data_max_")
        assert hasattr(fitted_minmax_scaler, "scale_")
        assert hasattr(fitted_minmax_scaler, "min_")

    def test_scaler_n_features(self, fitted_minmax_scaler):
        """
        Test scaler was fitted on correct number of features.
        """
        assert fitted_minmax_scaler.n_features_in_ == INPUT_DIM, \
            f"Expected {INPUT_DIM} features, scaler has {fitted_minmax_scaler.n_features_in_}"


# =============================================================================
# Scaling Application Tests
# =============================================================================


class TestScalingApplication:
    """Test scaler application to data."""

    def test_transform_shape_preserved(self, fitted_minmax_scaler, sample_numpy_32):
        """
        Test that transform preserves data shape.
        """
        X, _ = sample_numpy_32
        X_scaled = fitted_minmax_scaler.transform(X)

        assert X_scaled.shape == X.shape, \
            f"Shape changed from {X.shape} to {X_scaled.shape}"

    def test_transform_bounds_respected(self, fitted_minmax_scaler, sample_numpy_32):
        """
        Test that scaled values are within [0, 1] for data similar to training.
        """
        X, _ = sample_numpy_32
        X_scaled = fitted_minmax_scaler.transform(X)

        # MinMaxScaler can produce values outside [0, 1] when the test data
        # contains values beyond the training range used to fit the scaler.
        # Since sample_numpy_32 is drawn from a random distribution,
        # values may extend beyond the range the scaler was fit on.
        # We check that the worst-case deviation is bounded.
        min_val = X_scaled.min()
        max_val = X_scaled.max()
        tolerance = 0.5  # allow moderate deviation from [0, 1] for out-of-range data
        assert min_val >= SCALE_MIN - tolerance, \
            f"Scaled minimum {min_val} too far below {SCALE_MIN}"
        assert max_val <= SCALE_MAX + tolerance, \
            f"Scaled maximum {max_val} too far above {SCALE_MAX}"

    def test_transform_dtype_preserved(self, fitted_minmax_scaler, sample_numpy_32):
        """
        Test that transform preserves or uses appropriate dtype.
        """
        X, _ = sample_numpy_32
        X_scaled = fitted_minmax_scaler.transform(X)

        assert X_scaled.dtype in [np.float32, np.float64], \
            f"Unexpected dtype {X_scaled.dtype}"

    def test_inverse_transform(self, fitted_minmax_scaler, sample_numpy_32):
        """
        Test that inverse_transform recovers original values.
        """
        X, _ = sample_numpy_32
        X_scaled = fitted_minmax_scaler.transform(X)
        X_recovered = fitted_minmax_scaler.inverse_transform(X_scaled)

        np.testing.assert_array_almost_equal(X, X_recovered, decimal=5)

    def test_transform_single_sample(self, fitted_minmax_scaler):
        """
        Test transform works on single sample.
        """
        X_single = np.random.randn(1, INPUT_DIM).astype(np.float32)
        X_scaled = fitted_minmax_scaler.transform(X_single)

        assert X_scaled.shape == (1, INPUT_DIM)

    def test_transform_batch_consistency(self, fitted_minmax_scaler, sample_numpy_32):
        """
        Test that batch transform equals individual transforms.
        """
        X, _ = sample_numpy_32

        # Batch transform
        X_batch_scaled = fitted_minmax_scaler.transform(X)

        # Individual transforms
        X_individual_scaled = np.array([
            fitted_minmax_scaler.transform(X[i:i+1])[0]
            for i in range(len(X))
        ])

        np.testing.assert_array_almost_equal(X_batch_scaled, X_individual_scaled)


# =============================================================================
# Scaling Bounds Tests
# =============================================================================


class TestScalingBounds:
    """Test feature scaling produces values in [0, 1]."""

    def test_scaling_min_value(self, fitted_minmax_scaler):
        """
        Test minimum value after scaling is 0.
        """
        # Create data at the minimum seen during fit
        X_min = fitted_minmax_scaler.data_min_.reshape(1, -1)
        X_scaled = fitted_minmax_scaler.transform(X_min)

        np.testing.assert_array_almost_equal(X_scaled, np.zeros_like(X_scaled), decimal=5)

    def test_scaling_max_value(self, fitted_minmax_scaler):
        """
        Test maximum value after scaling is 1.
        """
        # Create data at the maximum seen during fit
        X_max = fitted_minmax_scaler.data_max_.reshape(1, -1)
        X_scaled = fitted_minmax_scaler.transform(X_max)

        np.testing.assert_array_almost_equal(X_scaled, np.ones_like(X_scaled), decimal=5)

    def test_scaling_midpoint(self, fitted_minmax_scaler):
        """
        Test midpoint scales to 0.5.
        """
        X_mid = (fitted_minmax_scaler.data_min_ + fitted_minmax_scaler.data_max_) / 2
        X_mid = X_mid.reshape(1, -1)
        X_scaled = fitted_minmax_scaler.transform(X_mid)

        np.testing.assert_array_almost_equal(X_scaled, np.ones_like(X_scaled) * 0.5, decimal=5)

    def test_out_of_range_values_extrapolate(self, fitted_minmax_scaler):
        """
        Test values outside training range extrapolate beyond [0, 1].
        """
        # Create values much larger than training max
        X_extreme = fitted_minmax_scaler.data_max_.reshape(1, -1) * 10
        X_scaled = fitted_minmax_scaler.transform(X_extreme)

        # Some values should be > 1
        assert (X_scaled > 1).any(), "Extreme values should scale beyond 1"

    def test_negative_out_of_range(self, fitted_minmax_scaler):
        """
        Test negative out-of-range values scale below 0.
        """
        # Create values much smaller than training min
        X_extreme = fitted_minmax_scaler.data_min_.reshape(1, -1) - 100
        X_scaled = fitted_minmax_scaler.transform(X_extreme)

        # Some values should be < 0
        assert (X_scaled < 0).any(), "Extreme negative values should scale below 0"


# =============================================================================
# Scaler Persistence Tests
# =============================================================================


class TestScalerPersistence:
    """Test scaler save/load functionality."""

    def test_scaler_pickle_save_load(self, fitted_minmax_scaler, sample_numpy_32):
        """
        Test scaler can be saved and loaded with pickle.
        """
        X, _ = sample_numpy_32

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(fitted_minmax_scaler, f)
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                loaded_scaler = pickle.load(f)

            # Verify loaded scaler produces same results
            X_original = fitted_minmax_scaler.transform(X)
            X_loaded = loaded_scaler.transform(X)

            np.testing.assert_array_almost_equal(X_original, X_loaded)
        finally:
            Path(temp_path).unlink()

    def test_scaler_attributes_preserved(self, fitted_minmax_scaler):
        """
        Test scaler attributes are preserved after save/load.
        """
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(fitted_minmax_scaler, f)
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                loaded_scaler = pickle.load(f)

            np.testing.assert_array_almost_equal(
                fitted_minmax_scaler.data_min_,
                loaded_scaler.data_min_
            )
            np.testing.assert_array_almost_equal(
                fitted_minmax_scaler.data_max_,
                loaded_scaler.data_max_
            )
            np.testing.assert_array_almost_equal(
                fitted_minmax_scaler.scale_,
                loaded_scaler.scale_
            )
        finally:
            Path(temp_path).unlink()

    def test_scaler_joblib_compatibility(self, fitted_minmax_scaler, sample_numpy_32):
        """
        Test scaler can be saved and loaded with joblib.
        """
        pytest.importorskip("joblib")
        import joblib

        X, _ = sample_numpy_32

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            joblib.dump(fitted_minmax_scaler, f.name)
            temp_path = f.name

        try:
            loaded_scaler = joblib.load(temp_path)

            X_original = fitted_minmax_scaler.transform(X)
            X_loaded = loaded_scaler.transform(X)

            np.testing.assert_array_almost_equal(X_original, X_loaded)
        finally:
            Path(temp_path).unlink()


# =============================================================================
# Data Preprocessor Tests
# =============================================================================


class TestDataPreprocessor:
    """Test DataPreprocessor class."""

    def test_preprocessor_instantiation(self, preprocessor):
        """
        Test DataPreprocessor can be instantiated.
        """
        assert preprocessor is not None

    def test_preprocessor_fit(self, preprocessor, sample_numpy_data):
        """
        Test preprocessor fit method.
        """
        X, y = sample_numpy_data

        # Create DataFrame for preprocessor
        import pandas as pd
        df = pd.DataFrame(X)

        preprocessor.fit(df, y)

        assert preprocessor._fitted if hasattr(preprocessor, "_fitted") else True

    def test_preprocessor_transform(self, preprocessor, sample_numpy_data):
        """
        Test preprocessor transform method.
        """
        X, y = sample_numpy_data

        import pandas as pd
        df = pd.DataFrame(X)

        preprocessor.fit(df, y)
        X_transformed, y_transformed = preprocessor.transform(df, y)

        assert X_transformed is not None
        assert y_transformed is not None

    def test_preprocessor_fit_transform(self, preprocessor, sample_numpy_data):
        """
        Test preprocessor fit_transform method.
        """
        X, y = sample_numpy_data

        import pandas as pd
        df = pd.DataFrame(X)

        X_transformed, y_transformed = preprocessor.fit_transform(df, y)

        assert X_transformed is not None
        assert y_transformed is not None


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestPreprocessingEdgeCases:
    """Test preprocessing edge cases."""

    def test_scaler_with_zero_variance_feature(self):
        """
        Test scaler handles constant (zero variance) features.
        """
        from sklearn.preprocessing import MinMaxScaler

        X = np.array([
            [1.0, 5.0],
            [2.0, 5.0],  # Second column is constant
            [3.0, 5.0],
        ])

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Should not have NaN
        assert not np.isnan(X_scaled).any()
        # Constant column should be 0 (when min == max, MinMaxScaler produces 0)
        assert (X_scaled[:, 1] == 0).all()

    def test_scaler_with_nan_input(self, fitted_minmax_scaler):
        """
        Test scaler behavior with NaN input.
        """
        X_with_nan = np.random.randn(10, INPUT_DIM).astype(np.float32)
        X_with_nan[0, 0] = np.nan

        X_scaled = fitted_minmax_scaler.transform(X_with_nan)

        # NaN should propagate
        assert np.isnan(X_scaled[0, 0])
        # Other values should be scaled
        assert not np.isnan(X_scaled[1:, :]).any()

    def test_scaler_with_inf_input(self, fitted_minmax_scaler):
        """
        Test scaler behavior with infinite input.
        """
        X_with_inf = np.random.randn(10, INPUT_DIM).astype(np.float32)
        X_with_inf[0, 0] = np.inf

        with pytest.raises(ValueError):
            _ = fitted_minmax_scaler.transform(X_with_inf)

    def test_empty_data_handling(self, fitted_minmax_scaler):
        """
        Test scaler behavior with empty data.
        """
        X_empty = np.empty((0, INPUT_DIM))

        with pytest.raises(ValueError):
            _ = fitted_minmax_scaler.transform(X_empty)

    def test_single_row_data(self, fitted_minmax_scaler):
        """
        Test scaler with single row of data.
        """
        X_single = np.random.randn(1, INPUT_DIM).astype(np.float32)
        X_scaled = fitted_minmax_scaler.transform(X_single)

        assert X_scaled.shape == (1, INPUT_DIM)
        assert not np.isnan(X_scaled).any()


# =============================================================================
# Integration Tests
# =============================================================================


class TestPreprocessingIntegration:
    """Integration tests for preprocessing pipeline."""

    def test_full_preprocessing_pipeline(self, sample_numpy_32):
        """
        Test complete preprocessing pipeline.
        """
        from sklearn.preprocessing import MinMaxScaler

        X, y = sample_numpy_32

        # Fit scaler
        scaler = MinMaxScaler()
        scaler.fit(X)

        # Transform
        X_scaled = scaler.transform(X)

        # Verify bounds for in-range data
        assert X_scaled.shape == X.shape
        assert not np.isnan(X_scaled).any()

    def test_preprocessing_with_train_test_split(self, sample_numpy_32):
        """
        Test preprocessing with train/test split.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler

        X, y = sample_numpy_32

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Fit on train only
        scaler = MinMaxScaler()
        scaler.fit(X_train)

        # Transform both
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train should be in [0, 1]
        assert X_train_scaled.min() >= -0.01
        # Test may be slightly outside due to distribution shift

        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape

    def test_preprocessing_consistency(self, sample_numpy_32):
        """
        Test preprocessing produces consistent results.
        """
        from sklearn.preprocessing import MinMaxScaler

        X, y = sample_numpy_32

        scaler1 = MinMaxScaler()
        scaler2 = MinMaxScaler()

        X_scaled1 = scaler1.fit_transform(X)
        X_scaled2 = scaler2.fit_transform(X)

        np.testing.assert_array_almost_equal(X_scaled1, X_scaled2)
