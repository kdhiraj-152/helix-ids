"""Regression tests for ValidationOrchestrator (Phase 16 extraction).

Covers:
    - ValidationOrchestrator construction (stateless)
    - check_family_class_coverage (all paths)
    - check_step_coverage (all trigger conditions, hard/soft failures)
    - post_training_macro_floor (epoch thresholds)
    - detect_coverage_collapse (threshold boundary)
    - check_zero_prediction_classes (raise / warn / energy paths)
    - check_per_dataset_macro_floor (pass / fail)
    - log_per_dataset_results (logging output)
    - Empty dataset, single-class, threshold edge cases
"""

from __future__ import annotations

import logging

import pytest
import torch

from scripts.training.validation import ValidationOrchestrator

# ======================================================================
#  Fixtures
# ======================================================================


@pytest.fixture
def logger() -> logging.Logger:
    return logging.getLogger("test_val_orch")


@pytest.fixture
def orch() -> ValidationOrchestrator:
    return ValidationOrchestrator()


# ======================================================================
#  check_family_class_coverage
# ======================================================================


class TestCheckFamilyClassCoverage:
    """Batch-level family class coverage validation."""

    def test_no_active_ids_returns_early(self, orch: ValidationOrchestrator) -> None:
        """When active_family_class_ids is None/empty, no-op."""
        y_family = torch.randint(0, 7, (16,))
        # Should not raise
        orch.check_family_class_coverage(y_family, active_family_class_ids=None)
        orch.check_family_class_coverage(y_family, active_family_class_ids=set())

    def test_single_class_batch_without_enforce_raises(
        self, orch: ValidationOrchestrator,
    ) -> None:
        """A batch with all same class raises RuntimeError."""
        y_family = torch.zeros(16, dtype=torch.long)
        with pytest.raises(RuntimeError, match="batch_single_family_class"):
            orch.check_family_class_coverage(
                y_family,
                active_family_class_ids={0, 1, 2, 3},
                enforce_all_classes_per_batch=False,
            )

    def test_missing_classes_with_enforce_raises(
        self, orch: ValidationOrchestrator,
    ) -> None:
        """When enforce_all_classes_per_batch and classes missing, raises."""
        y_family = torch.zeros(16, dtype=torch.long)
        with pytest.raises(RuntimeError, match="batch_missing_family_classes"):
            orch.check_family_class_coverage(
                y_family,
                active_family_class_ids={0, 1, 2, 3},
                enforce_all_classes_per_batch=True,
            )

    def test_valid_batch_with_enforce_passes(
        self, orch: ValidationOrchestrator,
    ) -> None:
        """Every active class present in batch — no error."""
        y_family = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
        # Should not raise
        orch.check_family_class_coverage(
            y_family,
            active_family_class_ids={0, 1, 2, 3},
            enforce_all_classes_per_batch=True,
        )

    def test_valid_batch_without_enforce_multi_class(
        self, orch: ValidationOrchestrator,
    ) -> None:
        """Multiple classes present, not all required — no error."""
        y_family = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
        orch.check_family_class_coverage(
            y_family,
            active_family_class_ids={0, 1, 2, 3},
            enforce_all_classes_per_batch=False,
        )

    def test_empty_batch_noop(self, orch: ValidationOrchestrator) -> None:
        """Empty tensor is a valid edge case — no unique classes means it raises."""
        y_family = torch.empty((0,), dtype=torch.long)
        with pytest.raises(RuntimeError, match="batch_missing_family_classes"):
            orch.check_family_class_coverage(
                y_family,
                active_family_class_ids={0, 1},
                enforce_all_classes_per_batch=True,
            )


# ======================================================================
#  check_step_coverage
# ======================================================================


class TestCheckStepCoverage:
    """Step-level coverage check with complex trigger conditions."""

    def test_already_checked_returns_early(
        self, orch: ValidationOrchestrator,
    ) -> None:
        """When step_coverage_checked is True, returns immediately."""
        checked, missing = orch.check_step_coverage(
            batch_idx=10,
            family_pred_counts=torch.ones(7),
            step_coverage_checked=True,
            representation_diagnostic_mode=False,
            head_phase_start_step=0,
            representation_phase_active=False,
            global_step=100,
            coverage_check_after_head_steps=50,
            step_coverage_check_step=5,
            active_family_class_ids={0, 1, 2, 3},
        )
        assert checked is True
        assert missing is None

    def test_diagnostic_mode_trigger(
        self, orch: ValidationOrchestrator,
    ) -> None:
        """Check fires in diagnostic mode after head phase threshold."""
        counts = torch.ones(7, dtype=torch.int64)
        checked, missing = orch.check_step_coverage(
            batch_idx=0,
            family_pred_counts=counts,
            step_coverage_checked=False,
            representation_diagnostic_mode=True,
            head_phase_start_step=50,
            representation_phase_active=False,
            global_step=120,
            coverage_check_after_head_steps=60,
            step_coverage_check_step=99,
            active_family_class_ids={0, 1, 2, 3},
        )
        assert checked is True
        assert missing is None

    def test_diagnostic_mode_not_triggered_during_phase(
        self, orch: ValidationOrchestrator,
    ) -> None:
        """No check during active representation phase."""
        checked, missing = orch.check_step_coverage(
            batch_idx=0,
            family_pred_counts=torch.ones(7),
            step_coverage_checked=False,
            representation_diagnostic_mode=True,
            head_phase_start_step=50,
            representation_phase_active=True,
            global_step=120,
            coverage_check_after_head_steps=60,
            step_coverage_check_step=99,
            active_family_class_ids={0, 1, 2, 3},
        )
        assert checked is False  # not modified
        assert missing is None

    def test_normal_mode_trigger(self, orch: ValidationOrchestrator) -> None:
        """Check fires in normal mode after batch_idx threshold."""
        counts = torch.ones(7, dtype=torch.int64)
        checked, missing = orch.check_step_coverage(
            batch_idx=10,
            family_pred_counts=counts,
            step_coverage_checked=False,
            representation_diagnostic_mode=False,
            head_phase_start_step=0,
            representation_phase_active=False,
            global_step=100,
            coverage_check_after_head_steps=50,
            step_coverage_check_step=5,
            active_family_class_ids={0, 1, 2, 3},
        )
        assert checked is True
        assert missing is None

    def test_missing_classes_raises(self, orch: ValidationOrchestrator) -> None:
        """Missing class predictions raise RuntimeError."""
        # Class 2 has zero predictions
        counts = torch.tensor([5, 3, 0, 4, 1, 0, 0], dtype=torch.int64)
        with pytest.raises(RuntimeError, match="step_coverage_missing_predictions"):
            orch.check_step_coverage(
                batch_idx=10,
                family_pred_counts=counts,
                step_coverage_checked=False,
                representation_diagnostic_mode=False,
                head_phase_start_step=0,
                representation_phase_active=False,
                global_step=100,
                coverage_check_after_head_steps=50,
                step_coverage_check_step=5,
                active_family_class_ids={0, 1, 2, 3},
            )

    def test_missing_classes_with_hard_stops_disabled(
        self, orch: ValidationOrchestrator, logger: logging.Logger,
    ) -> None:
        """Missing predictions log warning instead of raising when disabled."""
        counts = torch.tensor([5, 3, 0, 4], dtype=torch.int64)
        # Should not raise
        checked, missing = orch.check_step_coverage(
            batch_idx=10,
            family_pred_counts=counts,
            step_coverage_checked=False,
            representation_diagnostic_mode=False,
            head_phase_start_step=0,
            representation_phase_active=False,
            global_step=100,
            coverage_check_after_head_steps=50,
            step_coverage_check_step=5,
            active_family_class_ids={0, 1, 2, 3},
            disable_integrity_hard_stops=True,
            logger=logger,
        )
        assert checked is True
        assert missing is not None
        assert 2 in missing  # class 2 had zero predictions

    def test_missing_classes_energy_mode(
        self, orch: ValidationOrchestrator, logger: logging.Logger,
    ) -> None:
        """Energy mode logs warning instead of raising."""
        counts = torch.tensor([5, 3, 0, 4], dtype=torch.int64)
        checked, missing = orch.check_step_coverage(
            batch_idx=10,
            family_pred_counts=counts,
            step_coverage_checked=False,
            representation_diagnostic_mode=False,
            head_phase_start_step=0,
            representation_phase_active=False,
            global_step=100,
            coverage_check_after_head_steps=50,
            step_coverage_check_step=5,
            active_family_class_ids={0, 1, 2, 3},
            use_energy_based_family_objective=True,
            logger=logger,
        )
        assert checked is True
        assert missing is not None

    def test_family_pred_counts_none(
        self, orch: ValidationOrchestrator,
    ) -> None:
        """None counts means check does not run."""
        checked, missing = orch.check_step_coverage(
            batch_idx=10,
            family_pred_counts=None,
            step_coverage_checked=False,
            representation_diagnostic_mode=False,
            head_phase_start_step=0,
            representation_phase_active=False,
            global_step=100,
            coverage_check_after_head_steps=50,
            step_coverage_check_step=5,
            active_family_class_ids={0, 1, 2, 3},
        )
        assert checked is False
        assert missing is None

    def test_active_ids_none(
        self, orch: ValidationOrchestrator,
    ) -> None:
        """No active ids means check does not run."""
        checked, missing = orch.check_step_coverage(
            batch_idx=10,
            family_pred_counts=torch.ones(7),
            step_coverage_checked=False,
            representation_diagnostic_mode=False,
            head_phase_start_step=0,
            representation_phase_active=False,
            global_step=100,
            coverage_check_after_head_steps=50,
            step_coverage_check_step=5,
            active_family_class_ids=None,
        )
        assert checked is False
        assert missing is None


# ======================================================================
#  post_training_macro_floor
# ======================================================================


class TestPostTrainingMacroFloor:
    """Epoch-calibrated macro-F1 floor."""

    def test_smoke_floor_2_epochs(self, orch: ValidationOrchestrator) -> None:
        assert orch.post_training_macro_floor(epochs=1) == 0.10
        assert orch.post_training_macro_floor(epochs=2) == 0.10

    def test_short_floor_up_to_10(self, orch: ValidationOrchestrator) -> None:
        assert orch.post_training_macro_floor(epochs=3) == 0.15
        assert orch.post_training_macro_floor(epochs=5) == 0.15
        assert orch.post_training_macro_floor(epochs=10) == 0.15

    def test_full_floor_above_10(self, orch: ValidationOrchestrator) -> None:
        assert orch.post_training_macro_floor(epochs=11) == 0.25
        assert orch.post_training_macro_floor(epochs=50) == 0.25
        assert orch.post_training_macro_floor(epochs=1000) == 0.25

    def test_zero_epochs(self, orch: ValidationOrchestrator) -> None:
        assert orch.post_training_macro_floor(epochs=0) == 0.10


# ======================================================================
#  detect_coverage_collapse
# ======================================================================


class TestDetectCoverageCollapse:
    """Collapse detection threshold logic."""

    def test_no_collapse_normal_coverage(
        self, orch: ValidationOrchestrator,
    ) -> None:
        assert (
            orch.detect_coverage_collapse(
                {"val_family_predicted_class_count": 4.0},
                train_family_class_count=6,
            )
            is False
        )

    def test_collapse_too_few_classes(
        self, orch: ValidationOrchestrator,
    ) -> None:
        assert (
            orch.detect_coverage_collapse(
                {"val_family_predicted_class_count": 1.0},
                train_family_class_count=6,
            )
            is True
        )

    def test_zero_class_count_no_collapse(
        self, orch: ValidationOrchestrator,
    ) -> None:
        assert (
            orch.detect_coverage_collapse(
                {"val_family_predicted_class_count": 0.0},
                train_family_class_count=0,
            )
            is False
        )

    def test_missing_metric_key(self, orch: ValidationOrchestrator) -> None:
        assert (
            orch.detect_coverage_collapse(
                {},
                train_family_class_count=6,
            )
            is True
        )

    def test_boundary_50_percent(
        self, orch: ValidationOrchestrator,
    ) -> None:
        """Exactly 50% of classes should be enough."""
        assert (
            orch.detect_coverage_collapse(
                {"val_family_predicted_class_count": 3.0},
                train_family_class_count=6,
            )
            is False
        )


# ======================================================================
#  check_zero_prediction_classes
# ======================================================================


class TestCheckZeroPredictionClasses:
    """Zero-prediction class integrity guard."""

    def test_no_zero_classes_passes(
        self, orch: ValidationOrchestrator,
    ) -> None:
        orch.check_zero_prediction_classes(
            {"val_family_zero_prediction_classes": 0.0},
        )

    def test_zero_classes_raises(
        self, orch: ValidationOrchestrator,
    ) -> None:
        with pytest.raises(RuntimeError, match="validation_zero_prediction_classes_nonzero"):
            orch.check_zero_prediction_classes(
                {"val_family_zero_prediction_classes": 2.0},
            )

    def test_disabled_hard_stops(
        self, orch: ValidationOrchestrator, logger: logging.Logger,
    ) -> None:
        # Should not raise
        orch.check_zero_prediction_classes(
            {"val_family_zero_prediction_classes": 2.0},
            disable_integrity_hard_stops=True,
            logger=logger,
        )

    def test_energy_mode(
        self, orch: ValidationOrchestrator, logger: logging.Logger,
    ) -> None:
        # Should not raise
        orch.check_zero_prediction_classes(
            {"val_family_zero_prediction_classes": 2.0},
            use_energy_based_family_objective=True,
            logger=logger,
        )

    def test_missing_metric_key_noop(
        self, orch: ValidationOrchestrator,
    ) -> None:
        orch.check_zero_prediction_classes({})


# ======================================================================
#  check_per_dataset_macro_floor
# ======================================================================


class TestCheckPerDatasetMacroFloor:
    """Post-training macro-F1 guard."""

    def test_all_above_floor_passes(
        self, orch: ValidationOrchestrator,
    ) -> None:
        orch.check_per_dataset_macro_floor(
            {"dataset_a": {"family_macro_f1": 0.5}, "dataset_b": {"family_macro_f1": 0.6}},
            macro_floor=0.25,
        )

    def test_below_floor_raises(
        self, orch: ValidationOrchestrator,
    ) -> None:
        with pytest.raises(RuntimeError, match="Post-training macro_f1 guard failed"):
            orch.check_per_dataset_macro_floor(
                {"dataset_a": {"family_macro_f1": 0.05}},
                macro_floor=0.25,
            )

    def test_empty_results_passes(
        self, orch: ValidationOrchestrator,
    ) -> None:
        orch.check_per_dataset_macro_floor({}, macro_floor=0.25)

    def test_fallback_to_family_f1(
        self, orch: ValidationOrchestrator,
    ) -> None:
        """Uses family_f1 key when family_macro_f1 is absent."""
        orch.check_per_dataset_macro_floor(
            {"dataset_a": {"family_f1": 0.5}},
            macro_floor=0.25,
        )

    def test_fallback_below_floor_raises(
        self, orch: ValidationOrchestrator,
    ) -> None:
        with pytest.raises(RuntimeError, match="Post-training macro_f1 guard failed"):
            orch.check_per_dataset_macro_floor(
                {"dataset_a": {"family_f1": 0.05}},
                macro_floor=0.25,
            )


# ======================================================================
#  log_per_dataset_results
# ======================================================================


class TestLogPerDatasetResults:
    """Logging output formatting."""

    def test_logs_each_dataset(
        self, orch: ValidationOrchestrator, caplog: pytest.LogCaptureFixture,
    ) -> None:
        logger = logging.getLogger("test_log")
        logger.setLevel(logging.INFO)
        caplog.set_level(logging.INFO, logger="test_log")

        orch.log_per_dataset_results(
            logger,
            {"ds1": {"acc": 0.9, "f1": 0.85}, "ds2": {"acc": 0.8, "f1": 0.75}},
        )
        assert "ds1:" in caplog.text
        assert "ds2:" in caplog.text
        assert "acc" in caplog.text
        assert "f1" in caplog.text

    def test_empty_results(
        self, orch: ValidationOrchestrator, caplog: pytest.LogCaptureFixture,
    ) -> None:
        logger = logging.getLogger("test_log_empty")
        logger.setLevel(logging.INFO)
        caplog.set_level(logging.INFO, logger="test_log_empty")

        orch.log_per_dataset_results(logger, {})
        assert caplog.text == ""


# ======================================================================
#  Single-class behavior
# ======================================================================


class TestSingleClassBehavior:
    """Edge cases where the model has only 1-2 classes."""

    def test_single_class_coverage_no_enforce(
        self, orch: ValidationOrchestrator,
    ) -> None:
        """With 1 class and not enforcing all, single-class batch still fails (<=1 unique class)."""
        y_family = torch.zeros(16, dtype=torch.long)
        with pytest.raises(RuntimeError, match="batch_single_family_class"):
            orch.check_family_class_coverage(
                y_family,
                active_family_class_ids={0},
                enforce_all_classes_per_batch=False,
            )

    def test_single_class_coverage_enforce(
        self, orch: ValidationOrchestrator,
    ) -> None:
        """With 1 class and enforcing, a single-class batch passes."""
        y_family = torch.zeros(16, dtype=torch.long)
        orch.check_family_class_coverage(
            y_family,
            active_family_class_ids={0},
            enforce_all_classes_per_batch=True,
        )

    def test_single_class_step_coverage(
        self, orch: ValidationOrchestrator,
    ) -> None:
        """Step coverage with a single active class."""
        counts = torch.tensor([10], dtype=torch.int64)
        checked, missing = orch.check_step_coverage(
            batch_idx=10,
            family_pred_counts=counts,
            step_coverage_checked=False,
            representation_diagnostic_mode=False,
            head_phase_start_step=0,
            representation_phase_active=False,
            global_step=100,
            coverage_check_after_head_steps=50,
            step_coverage_check_step=5,
            active_family_class_ids={0},
        )
        assert checked is True
        assert missing is None


# ======================================================================
#  Threshold edge cases
# ======================================================================


class TestThresholdEdgeCases:
    """Boundary and numeric edge cases."""

    def test_coverage_collapse_boundary_half_above(
        self, orch: ValidationOrchestrator,
    ) -> None:
        """Exactly 50% of classes — above collapse threshold."""
        assert (
            orch.detect_coverage_collapse(
                {"val_family_predicted_class_count": 2.0},
                train_family_class_count=4,
            )
            is False
        )

    def test_coverage_collapse_boundary_just_below(
        self, orch: ValidationOrchestrator,
    ) -> None:
        """49% of classes — collapses."""
        assert (
            orch.detect_coverage_collapse(
                {"val_family_predicted_class_count": 1.0},
                train_family_class_count=4,
            )
            is True
        )

    def test_post_training_floor_boundary(
        self, orch: ValidationOrchestrator,
    ) -> None:
        """Exactly at floor value — passes."""
        orch.check_per_dataset_macro_floor(
            {"ds": {"family_macro_f1": 0.25}},
            macro_floor=0.25,
        )

    def test_zero_prediction_float_value(
        self, orch: ValidationOrchestrator,
    ) -> None:
        """Float 0.001 truncates to 0 via int() — does not trigger."""
        orch.check_zero_prediction_classes(
            {"val_family_zero_prediction_classes": 0.001},
        )

    def test_macro_floor_no_metric_keys(
        self, orch: ValidationOrchestrator,
    ) -> None:
        """When neither family_macro_f1 nor family_f1 exists, value is 0.0."""
        with pytest.raises(RuntimeError, match="Post-training macro_f1 guard failed"):
            orch.check_per_dataset_macro_floor(
                {"ds": {"other_metric": 0.9}},
                macro_floor=0.25,
            )
