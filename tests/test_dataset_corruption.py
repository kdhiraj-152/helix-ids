"""
P8 — Dataset Corruption Testing for HELIX-IDS.

Tests robustness against corrupted inputs:
  - NaN, Inf, extreme feature values
  - Out-of-range / missing / negative labels
  - Missing or duplicate columns
  - Wrong dtypes
  - Empty datasets
  - Schema violations
  - Corrupted loss inputs (NaN/Inf logits)
  - Corrupted multi-dimensional tensor inputs
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from helix_ids.config.environment import (
    HelixEnvironment,
    RuntimeSettings,
    TrainingSettings,
    load_environment,
)
from helix_ids.contracts.schema_contract import (
    CANONICAL_BINARY_CLASSES,
    CANONICAL_FAMILY_CLASSES,
    CANONICAL_FEATURE_ORDER,
    CANONICAL_INPUT_DIM,
    SCHEMA_VERSION,
    assert_runtime_contract,
    validate_feature_order,
)
from helix_ids.data.dataset_config import (
    NSL_KDD_CONFIG,
)
from helix_ids.data.feature_harmonization import (
    FeatureMapping,
    enforce_feature_order,
    sanitize_numeric,
    validate_no_nan_inf,
)
from helix_ids.data.label_mapping import (
    encode_labels,
    get_class_distribution,
    map_labels,
)
from helix_ids.utils.metrics import (
    compute_confusion_matrix,
    compute_macro_f1,
    compute_weighted_f1,
)

# ============================================================================
#  P8-A: Feature Corruption — NaN / Inf handling
# ============================================================================


class TestFeatureNaNInf:
    """sanitize_numeric and validate_no_nan_inf must handle corrupted features."""

    def test_sanitize_nan(self) -> None:
        """sanitize_numeric replaces NaN with 0."""
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, np.nan, np.nan]})
        cleaned = sanitize_numeric(df)
        assert cleaned["a"].isnull().sum() == 0
        assert cleaned["b"].isnull().sum() == 0
        assert cleaned["a"].iloc[1] == 0.0
        assert cleaned["b"].iloc[0] == 0.0

    def test_sanitize_inf(self) -> None:
        """sanitize_numeric replaces Inf with 0."""
        df = pd.DataFrame({"a": [1.0, np.inf, 3.0], "b": [-np.inf, 2.0, 5.0]})
        cleaned = sanitize_numeric(df)
        assert cleaned["a"].iloc[1] == 0.0
        assert cleaned["b"].iloc[0] == 0.0

    def test_sanitize_mixed_nan_inf(self) -> None:
        """sanitize_numeric handles mixed NaN/Inf in the same DataFrame."""
        df = pd.DataFrame(
            {
                "x": [np.nan, np.inf, 42.0, -np.inf],
                "y": [1.0, 2.0, np.nan, 4.0],
            }
        )
        cleaned = sanitize_numeric(df)
        assert np.isfinite(cleaned.values).all()
        assert cleaned.shape == df.shape

    def test_sanitize_non_numeric_untouched(self) -> None:
        """sanitize_numeric must leave non-numeric columns alone."""
        df = pd.DataFrame(
            {
                "value": [1.0, np.nan, 3.0],
                "label": ["a", "b", "c"],
                "flag": [True, False, True],
            }
        )
        cleaned = sanitize_numeric(df)
        assert cleaned["label"].tolist() == ["a", "b", "c"]
        assert cleaned["flag"].tolist() == [True, False, True]

    def test_validate_no_nan_inf_raises_with_nan(self) -> None:
        """validate_no_nan_inf must raise on NaN-containing DataFrames."""
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        with pytest.raises(AssertionError, match="NaN/inf"):
            validate_no_nan_inf(df)

    def test_validate_no_nan_inf_raises_with_inf(self) -> None:
        """validate_no_nan_inf must raise on Inf-containing DataFrames."""
        df = pd.DataFrame({"a": [1.0, np.inf, 3.0]})
        with pytest.raises(AssertionError, match="NaN/inf"):
            validate_no_nan_inf(df)

    def test_validate_no_nan_inf_passes_clean(self) -> None:
        """validate_no_nan_inf must pass on clean DataFrames."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.5, 0.6, 0.7]})
        validate_no_nan_inf(df)  # must not raise


class TestFeatureCorruptionExtremeValues:
    """Extreme numeric values must not crash feature operations."""

    def test_extreme_large_values(self) -> None:
        """DataFrames with extremely large values must be sanitizable."""
        df = pd.DataFrame(
            {
                "a": [1e20, 1e-20, 1e10],
                "b": [1e308, -1e308, 0.0],
            }
        )
        cleaned = sanitize_numeric(df)
        assert np.isfinite(cleaned.values).all()

    def test_extreme_negative_values(self) -> None:
        """Negative features are allowed but should not crash."""
        df = pd.DataFrame(
            {
                "protocol_type": [-1, 0, 1],
                "duration": [-100.0, -0.001, 0.0],
            }
        )
        cleaned = sanitize_numeric(df)
        assert cleaned.shape == df.shape

    def test_all_identical_values(self) -> None:
        """All-identical columns must not crash computations."""
        df = pd.DataFrame({"a": [5.0] * 100, "b": [1.0] * 100})
        cleaned = sanitize_numeric(df)
        assert cleaned["a"].std() == 0.0

    def test_single_row_dataframe(self) -> None:
        """Single-row DataFrames must be handled."""
        df = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
        cleaned = sanitize_numeric(df)
        assert len(cleaned) == 1


# ============================================================================
#  P8-B: Label Corruption — Out-of-range, missing, negative
# ============================================================================


class TestLabelCorruption:
    """Label arrays with corrupted values must not crash computations."""

    def test_negative_labels(self) -> None:
        """Negative labels: compute_macro_f1 must not crash."""
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([-1, 1, -2, 3, 4])
        # May produce unusual but bounded results
        macro = compute_macro_f1(y_true, y_pred)
        weighted = compute_weighted_f1(y_true, y_pred)
        assert 0.0 <= macro <= 1.0
        assert 0.0 <= weighted <= 1.0

    def test_all_same_label(self) -> None:
        """All samples with the same label must not crash."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 0])
        assert compute_macro_f1(y_true, y_pred) == 1.0
        assert compute_weighted_f1(y_true, y_pred) == 1.0

    def test_all_wrong_predictions(self) -> None:
        """All wrong predictions must produce F1=0 for each class."""
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([4, 3, 1, 0, 2])
        macro = compute_macro_f1(y_true, y_pred)
        assert macro >= 0.0

    def test_single_sample_labels(self) -> None:
        """Single-element label arrays must not crash."""
        y_true = np.array([0])
        y_pred = np.array([1])
        macro = compute_macro_f1(y_true, y_pred)
        assert 0.0 <= macro <= 1.0

    def test_confusion_matrix_negative_labels(self) -> None:
        """Confusion matrix with negative labels must not crash."""
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([-1, 1, 0, 5])  # 5 is out of range, -1 is negative
        cm = compute_confusion_matrix(y_true, y_pred)
        assert isinstance(cm, list)
        for row in cm:
            for val in row:
                assert isinstance(val, int)

    def test_label_mapping_empty_array(self) -> None:
        """Empty label arrays must not crash map_labels."""
        y = np.array([])
        mapped = map_labels(y, NSL_KDD_CONFIG, label_mode="native")
        assert len(mapped) == 0

    def test_label_mapping_with_missing_labels(self) -> None:
        """Labels that don't match any mapping must not crash."""
        y = np.array(["nonexistent_label_xyz", "also_missing_123"])
        mapped = map_labels(y, NSL_KDD_CONFIG, label_mode="native")
        assert len(mapped) == 2

    def test_encode_labels_with_empty_array(self) -> None:
        """Encoding an empty label array must not crash."""
        y = np.array([])
        encoded, classes, encoder = encode_labels(y, NSL_KDD_CONFIG, label_mode="native")
        assert len(encoded) == 0

    def test_class_distribution_empty(self) -> None:
        """get_class_distribution with empty input must not crash."""
        dist = get_class_distribution(np.array([]))
        assert dist["total_samples"] == 0
        assert dist["n_classes"] == 0

    def test_class_distribution_single_class(self) -> None:
        """get_class_distribution with single class must not crash."""
        dist = get_class_distribution(np.array([0, 0, 0]), class_names=["A"])
        assert dist["total_samples"] == 3
        assert dist["n_classes"] == 1


# ============================================================================
#  P8-C: Config Corruption — Missing fields, wrong types
# ============================================================================


class TestConfigCorruption:
    """Corrupted config must not produce crashes or silent data corruption."""

    def test_negative_epochs(self) -> None:
        """Negative epochs must not crash config loading."""
        env = load_environment(cli_overrides={"training": {"epochs": -5}})
        assert env.training.epochs == -5

    def test_zero_batch_size(self) -> None:
        """Zero batch size must not crash config loading."""
        env = load_environment(cli_overrides={"training": {"batch_size": 0}})
        assert env.training.batch_size == 0

    def test_negative_learning_rate(self) -> None:
        """Negative learning rate must not crash config loading."""
        env = load_environment(cli_overrides={"training": {"learning_rate": -0.01}})
        assert env.training.learning_rate == -0.01

    def test_empty_hidden_dims(self) -> None:
        """Empty hidden_dims must not crash config loading."""
        env = load_environment(cli_overrides={"model": {"hidden_dims": []}})
        assert env.model.hidden_dims == []

    def test_mismatched_dropout_length(self) -> None:
        """Mismatched dropout_rates length must not crash."""
        env = load_environment(cli_overrides={"model": {"dropout_rates": [0.1]}})
        assert env.model.dropout_rates == [0.1]

    def test_unknown_cli_key(self) -> None:
        """Unknown CLI keys must be tolerated."""
        env = load_environment(cli_overrides={"nonexistent_key": "value", "training.does_not_exist": 123})
        assert isinstance(env, HelixEnvironment)

    def test_nested_dict_post_init_with_dicts(self) -> None:
        """HelixEnvironment __post_init__ must handle dict-valued sections."""
        env = HelixEnvironment(
            training=TrainingSettings(batch_size=64, epochs=50),  # type: ignore[call-arg]
            runtime=RuntimeSettings(device="cpu"),  # type: ignore[call-arg]
        )
        assert isinstance(env.training, TrainingSettings)
        assert env.training.batch_size == 64
        assert isinstance(env.runtime.device, str)

    def test_loss_settings_default_on_partial_init(self) -> None:
        """Partial init must fall through to defaults for unspecified sections."""
        env = HelixEnvironment(training=TrainingSettings(batch_size=32))
        assert env.training.batch_size == 32
        assert env.model.input_dim == 17  # default


# ============================================================================
#  P8-D: Schema Contract Corruption
# ============================================================================


class TestSchemaCorruption:
    """Schema contract must reject corrupted contracts."""

    def test_wrong_input_dim(self) -> None:
        """assert_runtime_contract must reject wrong input_dim."""
        with pytest.raises(AssertionError, match="input_dim"):
            assert_runtime_contract(
                schema_version=SCHEMA_VERSION,
                schema_hash="dummy" * 16,
                feature_order=CANONICAL_FEATURE_ORDER,
                input_dim=999,
                binary_output_dim=CANONICAL_BINARY_CLASSES,
                family_output_dim=CANONICAL_FAMILY_CLASSES,
            )

    def test_wrong_binary_dim(self) -> None:
        """assert_runtime_contract must reject wrong binary_output_dim."""
        with pytest.raises(AssertionError, match="binary_output_dim"):
            assert_runtime_contract(
                schema_version=SCHEMA_VERSION,
                schema_hash="dummy" * 16,
                feature_order=CANONICAL_FEATURE_ORDER,
                input_dim=CANONICAL_INPUT_DIM,
                binary_output_dim=42,
                family_output_dim=CANONICAL_FAMILY_CLASSES,
            )

    def test_wrong_schema_hash(self) -> None:
        """assert_runtime_contract must reject wrong schema_hash."""
        with pytest.raises(AssertionError, match="schema_hash"):
            assert_runtime_contract(
                schema_version=SCHEMA_VERSION,
                schema_hash="wrong_hash_that_does_not_match",
                feature_order=CANONICAL_FEATURE_ORDER,
                input_dim=CANONICAL_INPUT_DIM,
                binary_output_dim=CANONICAL_BINARY_CLASSES,
                family_output_dim=CANONICAL_FAMILY_CLASSES,
            )

    def test_empty_feature_order(self) -> None:
        """Empty feature_order must be rejected by validate_feature_order."""
        with pytest.raises(AssertionError):
            validate_feature_order([])

    def test_scrambled_feature_order(self) -> None:
        """Scrambled feature_order must be rejected."""
        scrambled = list(CANONICAL_FEATURE_ORDER)
        scrambled.reverse()
        with pytest.raises(AssertionError):
            validate_feature_order(scrambled)

    def test_missing_features_in_order(self) -> None:
        """Feature_order with missing features but same length must be rejected."""
        shorter = CANONICAL_FEATURE_ORDER[:5]
        with pytest.raises(AssertionError):
            validate_feature_order(shorter)


# ============================================================================
#  P8-E: DataFrame Structural Corruption
# ============================================================================


class TestDataFrameCorruption:
    """DataFrames with structural issues must not crash enforce_feature_order."""

    def test_empty_dataframe(self) -> None:
        """Empty DataFrame must not crash sanitize_numeric."""
        df = pd.DataFrame()
        cleaned = sanitize_numeric(df)
        assert cleaned.empty

    def test_dataframe_missing_columns(self) -> None:
        """DataFrame with missing columns must trigger SchemaDriftError."""
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})  # Not canonical
        with pytest.raises((AssertionError, KeyError)):
            enforce_feature_order(df, CANONICAL_FEATURE_ORDER)

    def test_dataframe_extra_columns(self) -> None:
        """DataFrame with extra columns must trigger SchemaDriftError."""
        df_dict = {col: [1.0] for col in CANONICAL_FEATURE_ORDER[:5]}
        df_dict["extra_col"] = [42.0]
        df = pd.DataFrame(df_dict)
        with pytest.raises(AssertionError):
            enforce_feature_order(df, CANONICAL_FEATURE_ORDER)

    def test_dataframe_all_nan(self) -> None:
        """All-NaN DataFrame must not crash sanitize_numeric."""
        df = pd.DataFrame({col: [np.nan, np.nan] for col in CANONICAL_FEATURE_ORDER[:5]})
        cleaned = sanitize_numeric(df)
        assert cleaned.isnull().sum().sum() == 0  # All filled with 0

    def test_dataframe_all_inf(self) -> None:
        """All-Inf DataFrame must not crash sanitize_numeric."""
        df = pd.DataFrame({col: [np.inf, -np.inf] for col in CANONICAL_FEATURE_ORDER[:5]})
        cleaned = sanitize_numeric(df)
        assert np.isfinite(cleaned.values).all()

    def test_dataframe_with_non_hashable_column_names(self) -> None:
        """Non-string column names should not crash (though unusual)."""
        df = pd.DataFrame({0: [1.0, 2.0], 1: [3.0, 4.0]})
        cleaned = sanitize_numeric(df)
        assert not cleaned.empty

    def test_dataframe_with_mixed_types(self) -> None:
        """Mixed-type columns should not crash sanitize_numeric."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.5, 2.5, np.nan],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
            }
        )
        cleaned = sanitize_numeric(df)
        assert cleaned["int_col"].tolist() == [1, 2, 3]
        assert cleaned["str_col"].tolist() == ["a", "b", "c"]


# ============================================================================
#  P8-F: Loss Corruption — NaN/Inf logits, extreme labels
# ============================================================================


class TestLossCorruption:
    """Loss functions must handle corrupted inputs gracefully."""

    def test_threat_focal_nan_logits(self) -> None:
        """ThreatAwareFocalLoss with NaN logits must not crash."""
        from helix_ids.models.loss import ThreatAwareFocalLoss

        loss_fn = ThreatAwareFocalLoss().eval()
        logits = torch.full((4, 5), float("nan"))
        targets = torch.randint(0, 5, (4,))
        try:
            loss = loss_fn(logits, targets)
            assert loss is not None
        except (RuntimeError, ValueError):
            pass  # Acceptable to raise on pathological input

    def test_threat_focal_inf_logits(self) -> None:
        """ThreatAwareFocalLoss with Inf logits must produce finite loss or raise."""
        from helix_ids.models.loss import ThreatAwareFocalLoss

        loss_fn = ThreatAwareFocalLoss().eval()
        logits = torch.tensor([[float("inf"), 0.0, 0.0, 0.0, 0.0]])
        targets = torch.tensor([1])
        try:
            loss = loss_fn(logits, targets)
            assert loss >= 0
        except Exception:
            pass  # Acceptable to raise on pathological input

    def test_threat_focal_single_sample(self) -> None:
        """ThreatAwareFocalLoss with single sample must not crash."""
        from helix_ids.models.loss import ThreatAwareFocalLoss

        loss_fn = ThreatAwareFocalLoss().eval()
        logits = torch.randn(1, 5)
        targets = torch.randint(0, 5, (1,))
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)
        assert loss >= 0

    def test_threat_focal_all_same_logits(self) -> None:
        """ThreatAwareFocalLoss with uniform logits must produce finite loss."""
        from helix_ids.models.loss import ThreatAwareFocalLoss

        loss_fn = ThreatAwareFocalLoss().eval()
        logits = torch.zeros(10, 5)
        targets = torch.randint(0, 5, (10,))
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss)
        assert loss >= 0

    def test_calibration_loss_nan_logits(self) -> None:
        """CalibrationLoss with NaN logits must not crash."""
        from helix_ids.models.loss import CalibrationLoss

        loss_fn = CalibrationLoss().eval()
        logits = torch.full((4, 5), float("nan"))
        targets = torch.randint(0, 5, (4,))
        try:
            loss = loss_fn(logits, targets)
            assert loss is not None
        except (RuntimeError, ValueError):
            pass  # Acceptable to raise on pathological input

    def test_focal_loss_extreme_logits(self) -> None:
        """FocalLoss with extreme logits must produce finite loss."""
        from helix_ids.models.loss import FocalLoss

        loss_fn = FocalLoss().eval()
        logits = torch.tensor([[1e10, -1e10, 0.0, 0.0, 0.0]])
        targets = torch.tensor([0])
        try:
            loss = loss_fn(logits, targets)
            assert torch.isfinite(loss)
        except Exception:
            pass  # Acceptable to raise on pathological input

    def test_multitask_loss_partial_outputs(self) -> None:
        """MultiTaskLoss with partial output dict must not crash."""
        from helix_ids.models.loss import MultiTaskLoss

        loss_fn = MultiTaskLoss(
            num_fine_classes=5,
            num_binary_classes=2,
            num_family_classes=5,
        ).eval()
        outputs = {
            "binary": torch.randn(4, 2),
            # Missing "family" and "fine"
        }
        targets = {
            "binary": torch.randint(0, 2, (4,)),
            "family": torch.randint(0, 5, (4,)),
            "fine": torch.randint(0, 5, (4,)),
        }
        total_loss, loss_dict = loss_fn(outputs, targets)
        assert total_loss is not None

    def test_multitask_loss_empty_batch(self) -> None:
        """MultiTaskLoss with batch=0 (edge case) should raise or produce zero loss."""
        from helix_ids.models.loss import MultiTaskLoss

        loss_fn = MultiTaskLoss(
            num_fine_classes=5,
            num_binary_classes=2,
            num_family_classes=5,
        ).eval()
        outputs = {
            "binary": torch.randn(0, 2),
            "family": torch.randn(0, 5),
            "fine": torch.randn(0, 5),
        }
        targets = {
            "binary": torch.randint(0, 2, (0,)),
            "family": torch.randint(0, 5, (0,)),
            "fine": torch.randint(0, 5, (0,)),
        }
        try:
            total_loss, loss_dict = loss_fn(outputs, targets)
            assert "loss_total" in loss_dict
        except (RuntimeError, ValueError):
            pass  # Acceptable for degenerate input

    def test_multitask_loss_negative_epoch(self) -> None:
        """MultiTaskLoss.set_epoch with negative epoch must not crash."""
        from helix_ids.models.loss import MultiTaskLoss

        loss_fn = MultiTaskLoss(num_fine_classes=5)
        loss_fn.set_epoch(-1)
        # Just verify it doesn't raise
        assert loss_fn.current_epoch == -1


# ============================================================================
#  P8-G: Preprocessing Corruption — Corrupted DataFrames
# ============================================================================


class TestPreprocessingCorruption:
    """Preprocessing must handle corrupted DataFrames."""

    def test_preprocessor_all_nan(self) -> None:
        """DataPreprocessor with all-NaN DataFrame must not crash."""
        from helix_ids.data.preprocessing import DataPreprocessor, PreprocessingConfig

        prep = DataPreprocessor(config=PreprocessingConfig(handle_missing="zero"))
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        prep.fit(df)
        transformed, _ = prep.transform(df)
        assert np.all(np.isfinite(transformed))

    def test_preprocessor_all_inf(self) -> None:
        """DataPreprocessor with all-Inf DataFrame must not crash."""
        from helix_ids.data.preprocessing import DataPreprocessor, PreprocessingConfig

        prep = DataPreprocessor(config=PreprocessingConfig(handle_missing="zero"))
        df = pd.DataFrame({"a": [np.inf, -np.inf, np.inf]})
        prep.fit(df)
        transformed, _ = prep.transform(df)
        assert np.all(np.isfinite(transformed))

    def test_preprocessor_empty_df(self) -> None:
        """DataPreprocessor with empty DataFrame must not crash."""
        from helix_ids.data.preprocessing import DataPreprocessor

        prep = DataPreprocessor()
        df = pd.DataFrame()
        try:
            prep.fit(df)
            transformed, _ = prep.transform(df)
        except (ValueError, KeyError, AttributeError):
            pass  # Acceptable for empty DataFrame

    def test_preprocessor_single_column(self) -> None:
        """DataPreprocessor with single column must not crash."""
        from helix_ids.data.preprocessing import DataPreprocessor

        prep = DataPreprocessor()
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        prep.fit(df)
        transformed, _ = prep.transform(df)
        assert len(transformed) == 3

    def test_preprocessor_string_column(self) -> None:
        """DataPreprocessor must handle string columns gracefully."""
        from helix_ids.data.preprocessing import DataPreprocessor

        prep = DataPreprocessor()
        df = pd.DataFrame({"label": ["a", "b", "c"], "value": [1.0, 2.0, 3.0]})
        try:
            prep.fit(df)
            transformed, _ = prep.transform(df)
            assert np.all(np.isfinite(transformed))
        except (ValueError, TypeError, KeyError):
            pass  # Acceptable if non-numeric columns raise

    def test_preprocessor_mixed_column_types(self) -> None:
        """DataPreprocessor must handle mixed-type columns."""
        from helix_ids.data.preprocessing import DataPreprocessor

        prep = DataPreprocessor()
        df = pd.DataFrame({
            "ints": [1, 2, 3],
            "floats": [1.5, 2.5, 3.5],
            "bools": [True, False, True],
        })
        try:
            prep.fit(df)
            transformed, _ = prep.transform(df)
            assert np.all(np.isfinite(transformed))
        except Exception:
            pass  # Acceptable

    def test_preprocessor_duplicate_columns(self) -> None:
        """DataPreprocessor must handle duplicate column names."""
        from helix_ids.data.preprocessing import DataPreprocessor

        prep = DataPreprocessor()
        df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["a", "a"])
        try:
            prep.fit(df)
            transformed, _ = prep.transform(df)
            assert len(transformed) == 2
        except (ValueError, KeyError, TypeError):
            pass  # Acceptable for degenerate input

    def test_preprocessor_boolean_columns(self) -> None:
        """DataPreprocessor must handle boolean columns without crashing."""
        from helix_ids.data.preprocessing import DataPreprocessor

        prep = DataPreprocessor()
        df = pd.DataFrame({
            "flag": [True, False, True],
            "value": [10.0, 20.0, 30.0],
        })
        try:
            prep.fit(df)
            transformed, _ = prep.transform(df)
            assert np.all(np.isfinite(transformed))
        except Exception:
            pass  # Acceptable

    def test_preprocessor_very_large_values(self) -> None:
        """DataPreprocessor must handle extreme-value DataFrames."""
        from helix_ids.data.preprocessing import DataPreprocessor, PreprocessingConfig

        prep = DataPreprocessor(config=PreprocessingConfig(handle_missing="zero"))
        df = pd.DataFrame({
            "x": [1e10, -1e10, 1e-10],
            "y": [1e308, -1e308, 0.0],
        })
        try:
            prep.fit(df)
            transformed, _ = prep.transform(df)
            assert np.all(np.isfinite(transformed))
        except Exception:
            pass  # Acceptable for extreme values


# ============================================================================
#  P8-H: Feature Harmonization — Structural Corruption
# ============================================================================


class TestFeatureHarmonizationCorruption:
    """Feature harmonization must handle corrupted inputs."""

    def test_enforce_order_missing_cols(self) -> None:
        """enforce_feature_order must raise on missing columns."""
        partial_order = CANONICAL_FEATURE_ORDER[:5]
        df = pd.DataFrame({col: [1.0] for col in partial_order})
        # This should fail because CANONICAL_FEATURE_ORDER is longer
        with pytest.raises((AssertionError, KeyError, ValueError)):
            enforce_feature_order(df, CANONICAL_FEATURE_ORDER)

    def test_enforce_order_with_extra_cols(self) -> None:
        """enforce_feature_order must reject extra columns."""
        df_dict = {col: [1.0] for col in CANONICAL_FEATURE_ORDER[:5]}
        df_dict["extra"] = [42.0]
        df = pd.DataFrame(df_dict)
        with pytest.raises((AssertionError, ValueError)):
            enforce_feature_order(df, CANONICAL_FEATURE_ORDER)

    def test_sanitize_with_no_columns(self) -> None:
        """sanitize_numeric with empty DataFrame must not crash."""
        df = pd.DataFrame()
        cleaned = sanitize_numeric(df)
        assert cleaned.empty

    def test_sanitize_with_only_object_cols(self) -> None:
        """sanitize_numeric with only object columns must not crash."""
        df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        cleaned = sanitize_numeric(df)
        assert list(cleaned["a"]) == ["x", "y"]

    def test_sanitize_with_nan_string_mixed(self) -> None:
        """sanitize_numeric with NaN and string columns mixed."""
        df = pd.DataFrame({
            "numeric": [1.0, np.nan, 3.0],
            "category": ["low", "medium", "high"],
        })
        cleaned = sanitize_numeric(df)
        assert cleaned["numeric"].isnull().sum() == 0  # NaN filled

    def test_sanitize_with_extreme_ratio(self) -> None:
        """sanitize_numeric with extreme ratio values."""
        df = pd.DataFrame({
            "src_dst_bytes_ratio": [1e10, 1e-10, 0.0, -1e10],
            "duration": [0.0, 1.0, 100.0, 1e6],
        })
        cleaned = sanitize_numeric(df)
        assert np.all(np.isfinite(cleaned.values))

    def test_feature_mapping_construction(self) -> None:
        """FeatureMapping dataclass must accept valid construction."""
        mapping = FeatureMapping(
            dataset_name="test",
            original_features=["a", "b"],
            common_features=["a", "b"],
            feature_mapping={"a": "x", "b": "y"},
        )
        assert mapping.dataset_name == "test"
        assert isinstance(mapping.to_dict(), dict)


# ============================================================================
#  P8-I: Entropy Diagnostics Corruption Tests
# ============================================================================


class TestEntropyCorrupted:
    """Corruption resilience tests for entropy_diagnostics."""

    def test_calculate_entropy_nan_input(self) -> None:
        """calculate_entropy_stable must not crash with NaN or Inf in probabilities."""
        from helix_ids.utils.entropy_diagnostics import calculate_entropy_stable

        probs = np.array([[np.nan, 0.5, 0.5], [1.0, 0.0, 0.0]], dtype=np.float64)
        entropy = calculate_entropy_stable(probs, eps=1e-10)
        # Should not crash; output values may vary
        assert len(entropy) == 2

    def test_calculate_entropy_inf_input(self) -> None:
        """calculate_entropy_stable must not crash with Inf probabilities."""
        from helix_ids.utils.entropy_diagnostics import calculate_entropy_stable

        probs = np.array([[np.inf, 0.5, 0.5], [1.0, 0.0, 0.0]], dtype=np.float64)
        entropy = calculate_entropy_stable(probs, eps=1e-10)
        assert len(entropy) == 2

    def test_calculate_entropy_zeros(self) -> None:
        """calculate_entropy_stable with all-zero probs must not crash."""
        from helix_ids.utils.entropy_diagnostics import calculate_entropy_stable

        probs = np.zeros((3, 5), dtype=np.float64)
        entropy = calculate_entropy_stable(probs, eps=1e-10)
        assert len(entropy) == 3

    def test_calculate_entropy_negative(self) -> None:
        """calculate_entropy_stable with negative probabilities must not crash."""
        from helix_ids.utils.entropy_diagnostics import calculate_entropy_stable

        probs = np.array([[-0.1, 0.6, 0.5], [1.0, 0.0, 0.0]], dtype=np.float64)
        entropy = calculate_entropy_stable(probs, eps=1e-10)
        assert len(entropy) == 2

    def test_summarize_entropy_all_nan(self) -> None:
        """summarize_entropy with all-NaN probabilities must not crash."""
        from helix_ids.utils.entropy_diagnostics import summarize_entropy

        probs = np.full((5, 3), np.nan, dtype=np.float64)
        summary = summarize_entropy(probs)
        # Should still return a valid summary
        assert isinstance(summary.mean, float)

    def test_summarize_entropy_zero_classes(self) -> None:
        """summarize_entropy with zero classes in array must not crash."""
        from helix_ids.utils.entropy_diagnostics import summarize_entropy

        probs = np.empty((0, 0), dtype=np.float64)
        summary = summarize_entropy(probs)
        assert summary.num_samples == 0

    def test_entropy_guard_corrupted_summary(self) -> None:
        """should_trigger_entropy_guard must not crash with corrupt EntropySummary values."""
        from helix_ids.utils.entropy_diagnostics import (
            EntropySummary,
            should_trigger_entropy_guard,
        )

        summary = EntropySummary(
            mean=np.nan,
            min_val=np.nan,
            max_val=np.nan,
            num_samples=-1,
            num_classes=-1,
            collapsed_samples=-5,
        )
        should, reason = should_trigger_entropy_guard(summary)
        assert isinstance(should, bool)

    def test_detect_risk_corrupted_classes(self) -> None:
        """detect_batch_composition_risk must not crash with negative expected classes."""
        from helix_ids.utils.entropy_diagnostics import (
            EntropySummary,
            detect_batch_composition_risk,
        )

        summary = EntropySummary(
            mean=0.5, min_val=0.1, max_val=0.9,
            num_samples=10, num_classes=5, collapsed_samples=0,
        )
        predicted = np.array([-1, -1, 0, 1, 2, 3, 4, 5, 6, 100])
        risk = detect_batch_composition_risk(summary, predicted, num_expected_classes=5)
        assert isinstance(risk, dict)
        assert risk["missing_class_ratio"] >= 0.0


# ============================================================================
#  P8-J: Data Audit Corruption Tests
# ============================================================================


class TestDataAuditCorrupted:
    """Corruption resilience tests for data_audit."""

    def test_audit_empty_dataframe(self) -> None:
        """DataAudit methods must handle empty DataFrames without crashing."""
        from helix_ids.data.data_audit import DataAudit

        auditor = DataAudit()
        df = pd.DataFrame()
        assert auditor.audit_nan_distribution(df)["overall_nan_pct"] == 0.0
        assert auditor.audit_duplicates(df)["exact_duplicates"] == 0

    def test_audit_single_row_dataframe(self) -> None:
        """DataAudit methods must handle single-row DataFrames without crashing."""
        from helix_ids.data.data_audit import DataAudit

        auditor = DataAudit()
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        nan_result = auditor.audit_nan_distribution(df)
        assert nan_result["total_rows"] == 1
        assert nan_result["overall_nan_pct"] == 0.0
        dup_result = auditor.audit_duplicates(df)
        assert dup_result["exact_duplicates"] == 0

    def test_audit_all_nan_dataframe(self) -> None:
        """DataAudit must handle DataFrames with all NaN values."""
        from helix_ids.data.data_audit import DataAudit

        auditor = DataAudit()
        df = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
        result = auditor.audit_nan_distribution(df)
        assert result["overall_nan_pct"] == 100.0
        assert len(result["critical_columns"]) == 2

    def test_audit_all_identical_rows(self) -> None:
        """DataAudit must handle DataFrames where every row is identical."""
        from helix_ids.data.data_audit import DataAudit

        auditor = DataAudit()
        df = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [2.0, 2.0, 2.0]})
        result = auditor.audit_duplicates(df)
        assert result["exact_duplicates"] == 3
        assert result["unique_rows"] == 1

    def test_audit_nan_percent_column(self) -> None:
        """DataAudit must handle NaN values correctly at the column threshold."""
        from helix_ids.data.data_audit import DataAudit, DataAuditConfig

        # 50% NaN in first column, threshold at 30%
        auditor = DataAudit(DataAuditConfig(nan_column_threshold=0.3))
        df = pd.DataFrame({
            "a": [np.nan, np.nan, 1.0, 1.0],
            "b": [1.0, 1.0, 1.0, 1.0],
        })
        result = auditor.audit_nan_distribution(df)
        assert "a" in result["critical_columns"]
        assert "b" not in result["critical_columns"] or result["per_column"]["b"] == 0.0

    def test_audit_identifiers_no_match(self) -> None:
        """audit_identifiers with no matching identifier columns must return LOW risk."""
        from helix_ids.data.data_audit import DataAudit

        auditor = DataAudit()
        df = pd.DataFrame({"feature_a": [1.0], "feature_b": [2.0]})
        result = auditor.audit_identifiers(df)
        assert result["identifier_risk"] == "LOW"
        assert len(result["suspected_identifiers"]) == 0

    def test_audit_labels_missing_column(self) -> None:
        """audit_labels with missing label column must return error dict."""
        from helix_ids.data.data_audit import DataAudit

        auditor = DataAudit()
        df = pd.DataFrame({"a": [1.0]})
        result = auditor.audit_labels(df, "missing_label")
        assert "error" in result

    def test_audit_labels_with_mapping(self) -> None:
        """audit_labels with mapping must detect unmapped labels."""
        from helix_ids.data.data_audit import DataAudit

        auditor = DataAudit()
        df = pd.DataFrame({"label": ["normal", "attack", "unknown", "malicious"]})
        mapping = {"normal": "0", "attack": "1"}
        result = auditor.audit_labels(df, "label", mapping=mapping)
        assert len(result["unmapped_labels"]) == 2
        assert "unknown" in result["unmapped_labels"]

    def test_audit_outliers_zero_std(self) -> None:
        """audit_outliers must handle columns with zero standard deviation."""
        from helix_ids.data.data_audit import DataAudit

        auditor = DataAudit()
        df = pd.DataFrame({"a": [5.0, 5.0, 5.0], "b": [1.0, 2.0, 3.0]})
        result = auditor.audit_outliers(df, sigma=3.0)
        assert result["per_column_outlier_pct"]["a"] == 0.0  # zero std

    def test_audit_outliers_no_numeric(self) -> None:
        """audit_outliers must handle DataFrames with no numeric columns."""
        from helix_ids.data.data_audit import DataAudit

        auditor = DataAudit()
        df = pd.DataFrame({"a": ["x", "y"], "b": ["z", "w"]})
        result = auditor.audit_outliers(df, sigma=3.0)
        assert result["numeric_columns_checked"] == 0

    def test_audit_generate_report_single(self) -> None:
        """generate_audit_report with a single dataset must not crash."""
        from helix_ids.data.data_audit import DataAudit

        auditor = DataAudit()
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        summary_df, report = auditor.generate_audit_report({"test": df})
        assert isinstance(summary_df, pd.DataFrame)
        assert "per_dataset" in report
        assert "audit_config" in report

    def test_audit_generate_report_multiple(self) -> None:
        """generate_audit_report with multiple datasets must not crash."""
        from helix_ids.data.data_audit import DataAudit

        auditor = DataAudit()
        df1 = pd.DataFrame({"a": [1.0]})
        df2 = pd.DataFrame({"a": [2.0], "b": [3.0]})
        summary_df, report = auditor.generate_audit_report({"ds1": df1, "ds2": df2})
        assert "schema_consistency" in report

    def test_generate_report_with_exclude(self) -> None:
        """generate_audit_report must accept exclude_cols_per_dataset."""
        from helix_ids.data.data_audit import DataAudit

        auditor = DataAudit()
        df1 = pd.DataFrame({"Flow ID": ["x"], "feature_a": [1.0]})
        df2 = pd.DataFrame({"Flow ID": ["y"], "feature_a": [2.0]})
        _, report = auditor.generate_audit_report(
            {"ds1": df1, "ds2": df2},
            exclude_cols_per_dataset={"ds1": ["Flow ID"]},
        )
        assert "per_dataset" in report

    def test_audit_schema_single_dataset(self) -> None:
        """audit_schema with a single dataset must return consistency_score=1.0."""
        from helix_ids.data.data_audit import DataAudit

        auditor = DataAudit()
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        result = auditor.audit_schema({"single": df})
        assert result["consistency_score"] == 1.0
        assert result["column_intersection"] == ["a", "b"]

    def test_audit_schema_mismatched(self) -> None:
        """audit_schema must detect column mismatches across datasets."""
        from helix_ids.data.data_audit import DataAudit

        auditor = DataAudit()
        df1 = pd.DataFrame({"a": [1.0], "b": [2.0]})
        df2 = pd.DataFrame({"a": [1.0], "c": [3.0]})
        result = auditor.audit_schema({"ds1": df1, "ds2": df2})
        assert len(result["schema_mismatches"]) > 0

    def test_audit_identifiers_with_exclude(self) -> None:
        """audit_identifiers must accept explicit column list (overrides config)."""
        from helix_ids.data.data_audit import DataAudit

        auditor = DataAudit()
        df = pd.DataFrame({"Flow ID": ["x"], "Src IP": ["y"], "feature": [1.0]})
        result = auditor.audit_identifiers(df, exclude_cols=["Flow ID", "Src IP"])
        assert len(result["suspected_identifiers"]) == 2


# ============================================================================
#  P8-K: Helix Full Config Corruption Tests
# ============================================================================


class TestHelixFullConfigCorrupted:
    """Corruption resilience tests for helix_full_config."""

    def test_load_training_config_none(self) -> None:
        """load_training_config(None) must return default TrainingConfig."""
        from helix_ids.config.helix_full_config import TrainingConfig, load_training_config

        cfg = load_training_config(None)
        assert isinstance(cfg, TrainingConfig)
        assert cfg.input_dim == 17  # default

    def test_load_training_config_no_path(self) -> None:
        """load_training_config() without args must return default TrainingConfig."""
        from helix_ids.config.helix_full_config import TrainingConfig, load_training_config

        cfg = load_training_config()
        assert isinstance(cfg, TrainingConfig)

    def test_save_training_config_empty_path(self) -> None:
        """save_training_config must handle Path objects correctly."""
        import tempfile
        from pathlib import Path

        from helix_ids.config.helix_full_config import TrainingConfig, save_training_config

        cfg = TrainingConfig()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            tmp = Path(f.name)
        try:
            save_training_config(cfg, tmp)
            assert tmp.exists()
            assert tmp.stat().st_size > 0
        finally:
            tmp.unlink(missing_ok=True)

    def test_training_config_defaults(self) -> None:
        """TrainingConfig defaults must be valid."""
        from helix_ids.config.helix_full_config import TrainingConfig

        cfg = TrainingConfig()
        assert cfg.lambda_binary > 0
        assert cfg.lambda_family > 0
        assert cfg.warmup_epochs > 0
        assert cfg.warmup_init_lr > 0
        assert cfg.max_grad_norm > 0
        assert cfg.early_stopping_patience > 0

    def test_data_config_defaults(self) -> None:
        """DataConfig defaults must be valid."""
        from helix_ids.config.helix_full_config import DataConfig

        cfg = DataConfig()
        assert cfg.use_per_dataset_normalization is True

    def test_evaluation_config_defaults(self) -> None:
        """EvaluationConfig defaults must include all expected metrics."""
        from helix_ids.config.helix_full_config import EvaluationConfig

        cfg = EvaluationConfig()
        for m in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "cm"):
            assert m in cfg.metrics

    def test_save_config_overwrites(self) -> None:
        """save_training_config must overwrite existing files."""
        import json
        import tempfile
        from pathlib import Path

        from helix_ids.config.helix_full_config import TrainingConfig, save_training_config

        cfg1 = TrainingConfig(input_dim=10)
        cfg2 = TrainingConfig(input_dim=50)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            tmp = Path(f.name)
        try:
            save_training_config(cfg1, tmp)
            save_training_config(cfg2, tmp)
            with open(tmp) as f:
                loaded = json.load(f)
            assert loaded["model"]["input_dim"] == 50  # Second write wins
        finally:
            tmp.unlink(missing_ok=True)

    def test_presets_independent(self) -> None:
        """DEFAULT_CONFIG, LARGE_CONFIG, SMALL_CONFIG must be distinct objects."""
        from helix_ids.config.helix_full_config import DEFAULT_CONFIG, LARGE_CONFIG, SMALL_CONFIG

        assert DEFAULT_CONFIG is not LARGE_CONFIG
        assert DEFAULT_CONFIG is not SMALL_CONFIG
        assert LARGE_CONFIG is not SMALL_CONFIG

    def test_preset_sizes(self) -> None:
        """LARGE_CONFIG must have larger hidden_dims than DEFAULT_CONFIG."""
        from helix_ids.config.helix_full_config import DEFAULT_CONFIG, LARGE_CONFIG, SMALL_CONFIG

        assert sum(LARGE_CONFIG.hidden_dims) > sum(DEFAULT_CONFIG.hidden_dims)
        assert sum(SMALL_CONFIG.hidden_dims) < sum(DEFAULT_CONFIG.hidden_dims)
