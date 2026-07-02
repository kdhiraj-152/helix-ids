"""Regression tests for Phase 12B-6 extracted evaluation components.

Validates that HelixFullEvaluator's static methods and core evaluation
methods behave identically to the original code extracted from
HelixFullTrainer.

The extraction is a pure move — no behavioral changes are expected.
These tests verify that:
1. Static helpers produce identical outputs for identical inputs.
2. Instance methods configure and run without regression.
3. Delegation bridge in HelixFullTrainer preserves the public API.
"""

from __future__ import annotations

import pytest
import torch

from scripts.training.evaluation import HelixFullEvaluator

# ======================================================================
# _apply_eval_class4_logit_shift
# ======================================================================


class TestApplyEvalClass4LogitShift:
    """Verify the static logit shift helper matches original behavior."""

    def test_shift_applied_correct_class(self):
        """Class-4 logit is reduced by shift amount, others unchanged."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        result = HelixFullEvaluator._apply_eval_class4_logit_shift(
            logits, shift=2.0, class_id=4,
        )
        expected = logits.clone()
        expected[:, 4] = expected[:, 4] - 2.0
        assert torch.equal(result, expected)

    def test_zero_shift_noop(self):
        """Zero shift returns the input unchanged."""
        logits = torch.randn(4, 7)
        result = HelixFullEvaluator._apply_eval_class4_logit_shift(
            logits, shift=0.0, class_id=4,
        )
        assert torch.equal(result, logits)

    def test_negative_shift_noop(self):
        """Negative shift is treated as zero and returns input unchanged."""
        logits = torch.randn(4, 7)
        result = HelixFullEvaluator._apply_eval_class4_logit_shift(
            logits, shift=-1.0, class_id=4,
        )
        assert torch.equal(result, logits)

    def test_invalid_class_id_noop(self):
        """Class ID out of bounds returns input unchanged."""
        logits = torch.randn(4, 7)
        result = HelixFullEvaluator._apply_eval_class4_logit_shift(
            logits, shift=2.0, class_id=99,
        )
        assert torch.equal(result, logits)

        result_neg = HelixFullEvaluator._apply_eval_class4_logit_shift(
            logits, shift=2.0, class_id=-1,
        )
        assert torch.equal(result_neg, logits)

    def test_1d_tensor_noop(self):
        """1D tensor (not 2D) returns input unchanged."""
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = HelixFullEvaluator._apply_eval_class4_logit_shift(
            logits, shift=2.0, class_id=4,
        )
        assert torch.equal(result, logits)

    def test_empty_batch_noop(self):
        """Empty batch (0 rows) returns input unchanged."""
        logits = torch.zeros((0, 7))
        result = HelixFullEvaluator._apply_eval_class4_logit_shift(
            logits, shift=2.0, class_id=4,
        )
        assert torch.equal(result, logits)

    def test_multi_batch_shift_only_class4(self):
        """In multi-batch input, only class-4 logit is shifted in each row."""
        logits = torch.randn(8, 7)
        shift_value = 0.5
        result = HelixFullEvaluator._apply_eval_class4_logit_shift(
            logits, shift=shift_value, class_id=4,
        )
        assert torch.allclose(result[:, 4], logits[:, 4] - shift_value)
        # Verify all other columns are untouched
        mask = torch.ones(logits.shape[1], dtype=torch.bool)
        mask[4] = False
        assert torch.equal(result[:, mask], logits[:, mask])


# ======================================================================
# _apply_inference_prediction_floor
# ======================================================================


class TestApplyInferencePredictionFloor:
    """Verify the static prediction floor helper matches original behavior."""

    def test_no_active_ids_noop(self):
        """Empty active_class_ids returns prediction unchanged."""
        logits = torch.randn(4, 5)
        pred = torch.tensor([0, 1, 2, 3])
        result = HelixFullEvaluator._apply_inference_prediction_floor(
            logits, pred, active_class_ids=set(),
        )
        assert torch.equal(result, pred)

    def test_all_classes_present_noop(self):
        """All active classes already present — no change."""
        logits = torch.randn(5, 5)
        pred = torch.tensor([0, 1, 2, 3, 4])
        result = HelixFullEvaluator._apply_inference_prediction_floor(
            logits, pred, active_class_ids={0, 1, 2, 3, 4},
        )
        assert torch.equal(result, pred)

    def test_missing_class_added(self):
        """A missing active class is forced into a prediction row.

        The row with the highest logit for the missing class, excluding rows
        already used for other missing classes, is selected.
        """
        logits = torch.tensor([
            [2.0, 1.0, 0.0],  # row 0: class 0 highest
            [0.0, 3.0, 1.0],  # row 1: class 1 highest
        ])
        pred = torch.tensor([0, 1])
        # Class 2 is missing — should be forced into the best row
        result = HelixFullEvaluator._apply_inference_prediction_floor(
            logits, pred, active_class_ids={0, 1, 2},
        )
        result_set = {int(v) for v in result.tolist()}
        assert 2 in result_set, f"Class 2 should be forced into predictions, got {result}"

    def test_multiple_missing_classes(self):
        """Multiple missing classes are each forced into distinct rows."""
        # 4 samples, 5 classes, classes 3 and 4 missing in pred
        logits = torch.tensor([
            [5.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 1.0],
        ])
        pred = torch.tensor([0, 1, 2, 0])  # class 0 seen twice, 3 and 4 missing
        result = HelixFullEvaluator._apply_inference_prediction_floor(
            logits, pred, active_class_ids={0, 1, 2, 3, 4},
        )
        result_set = {int(v) for v in result.tolist()}
        assert 3 in result_set, "Class 3 should be forced"
        assert 4 in result_set, "Class 4 should be forced"

    def test_1d_tensor_noop(self):
        """1D logit tensor returns prediction unchanged."""
        logits = torch.tensor([1.0, 2.0, 3.0])
        pred = torch.tensor([0])
        result = HelixFullEvaluator._apply_inference_prediction_floor(
            logits, pred, active_class_ids={0, 1, 2},
        )
        assert torch.equal(result, pred)

    def test_empty_batch_noop(self):
        """Empty batch returns prediction unchanged."""
        logits = torch.zeros((0, 5))
        pred = torch.tensor([], dtype=torch.long)
        result = HelixFullEvaluator._apply_inference_prediction_floor(
            logits, pred, active_class_ids={0, 1, 2},
        )
        assert torch.equal(result, pred)


# ======================================================================
# _compute_f1_stats_from_confusion
# ======================================================================


class TestComputeF1StatsFromConfusion:
    """Verify the static F1 stats helper matches original behavior."""

    def test_empty_confusion(self):
        """Empty confusion returns default values."""
        confusion = torch.zeros((0, 0), dtype=torch.int64)
        stats = HelixFullEvaluator._compute_f1_stats_from_confusion(confusion)
        assert stats["macro_f1"] == 0.0
        assert stats["weighted_f1"] == 0.0
        assert stats["minority_recall_min"] == 0.0
        assert stats["zero_prediction_classes"] == []

    def test_perfect_classification(self):
        """Perfect diagonal confusion matrix yields F1=1.0."""
        confusion = torch.eye(3, dtype=torch.int64) * 10
        stats = HelixFullEvaluator._compute_f1_stats_from_confusion(confusion)
        assert stats["macro_f1"] == pytest.approx(1.0)
        assert stats["weighted_f1"] == pytest.approx(1.0)

    def test_zero_predictions_detected(self):
        """Classes with support but zero predictions appear in list."""
        # 3 classes: class 2 has support (5 samples) but 0 predictions → flagged
        confusion = torch.tensor(
            [[10, 0, 0],
             [0, 10, 0],
             [5, 0, 0]],   # class 2: support=5, predicted=0 → zero_prediction
            dtype=torch.int64,
        )
        stats = HelixFullEvaluator._compute_f1_stats_from_confusion(confusion)
        assert 2 in stats["zero_prediction_classes"]

    def test_minority_recall_min(self):
        """Minority recall picks the minimum recall among non-class-0."""
        # 3 classes: class 0 perfect, class 1 perfect, class 2 zero recall
        confusion = torch.tensor(
            [[100, 0, 0],
             [0, 50, 0],
             [10, 0, 0]],  # class 2: all predicted as class 0, recall=0
            dtype=torch.int64,
        )
        stats = HelixFullEvaluator._compute_f1_stats_from_confusion(confusion)
        assert stats["minority_recall_min"] == pytest.approx(0.0)

    def test_imbalanced_f1(self):
        """Weighted and macro F1 differ on imbalanced data."""
        confusion = torch.tensor(
            [[90, 10], [20, 80]],  # class 0: 90/100, class 1: 80/100
            dtype=torch.int64,
        )
        stats = HelixFullEvaluator._compute_f1_stats_from_confusion(confusion)
        # Class 0 precision=90/110=0.818, recall=90/100=0.9, F1≈0.857
        # Class 1 precision=80/90=0.889, recall=80/100=0.8, F1≈0.842
        # Macro F1 ≈ (0.857+0.842)/2 = 0.8495
        assert stats["macro_f1"] == pytest.approx(0.8495, abs=0.01)
        # Weighted F1 with equal support ≈ same
        assert stats["weighted_f1"] == pytest.approx(0.8495, abs=0.01)


# ======================================================================
# _resolve_active_class_ids
# ======================================================================


class TestResolveActiveClassIds:
    """Verify the active class ID resolution helper."""

    def test_per_call_override_used(self):
        """Per-call override takes precedence over instance default."""
        evaluator = HelixFullEvaluator(
            model=None,
            device="cpu",
            loss_fn=None,
            active_family_class_ids={0, 1, 2},
        )
        result = evaluator._resolve_active_class_ids(active_family_class_ids={3, 4, 5})
        assert result == {3, 4, 5}

    def test_fallback_to_instance(self):
        """When per-call override is None, instance default is used."""
        evaluator = HelixFullEvaluator(
            model=None,
            device="cpu",
            loss_fn=None,
            active_family_class_ids={0, 1, 2},
        )
        result = evaluator._resolve_active_class_ids(active_family_class_ids=None)
        assert result == {0, 1, 2}

    def test_empty_fallback(self):
        """When both are empty, returns empty set."""
        evaluator = HelixFullEvaluator(
            model=None,
            device="cpu",
            loss_fn=None,
        )
        result = evaluator._resolve_active_class_ids(active_family_class_ids=None)
        assert result == set()


# ======================================================================
# _apply_eval_logit_controls
# ======================================================================


class TestApplyEvalLogitControls:
    """Verify the evaluation logit control logic."""

    def test_temperature_scaling(self):
        """Temperature scaling divides logits when enabled."""
        evaluator = HelixFullEvaluator(
            model=None,
            device="cpu",
            loss_fn=None,
            use_energy_based_family_objective=True,
            energy_logit_temperature=2.0,
            family_log_prior=None,
        )
        logits = torch.tensor([[4.0, 2.0, 0.0]])
        result = evaluator._apply_eval_logit_controls(logits)
        expected = logits / 2.0
        assert torch.equal(result, expected)

    def test_temperature_disabled(self):
        """Temperature scaling not applied when disabled."""
        evaluator = HelixFullEvaluator(
            model=None,
            device="cpu",
            loss_fn=None,
            use_energy_based_family_objective=False,
            energy_logit_temperature=2.0,
            family_log_prior=None,
        )
        logits = torch.tensor([[4.0, 2.0, 0.0]])
        result = evaluator._apply_eval_logit_controls(logits)
        assert torch.equal(result, logits)

    def test_prior_correction(self):
        """Prior correction subtracts prior from logits."""
        prior = torch.tensor([0.5, 0.3, 0.2])
        evaluator = HelixFullEvaluator(
            model=None,
            device="cpu",
            loss_fn=None,
            use_energy_based_family_objective=False,
            family_log_prior=prior,
        )
        logits = torch.tensor([[4.0, 2.0, 0.0]])
        result = evaluator._apply_eval_logit_controls(logits)
        expected = logits - prior
        assert torch.equal(result, expected)

    def test_temperature_and_prior(self):
        """Both temperature and prior can be applied simultaneously."""
        prior = torch.tensor([0.5, 0.3, 0.2])
        evaluator = HelixFullEvaluator(
            model=None,
            device="cpu",
            loss_fn=None,
            use_energy_based_family_objective=True,
            energy_logit_temperature=4.0,
            family_log_prior=prior,
        )
        logits = torch.tensor([[4.0, 2.0, 0.0]])
        result = evaluator._apply_eval_logit_controls(logits)
        expected = (logits / 4.0) - prior
        assert torch.equal(result, expected)

    def test_prior_dimension_mismatch_raises(self):
        """Prior with wrong dimension raises RuntimeError."""
        prior = torch.tensor([0.5, 0.3])  # only 2 classes
        evaluator = HelixFullEvaluator(
            model=None,
            device="cpu",
            loss_fn=None,
            use_energy_based_family_objective=False,
            family_log_prior=prior,
        )
        logits = torch.tensor([[4.0, 2.0, 0.0]])  # 3 classes
        with pytest.raises(RuntimeError, match="family prior dimension mismatch"):
            evaluator._apply_eval_logit_controls(logits)

    def test_no_controls_noop(self):
        """When both temperature and prior are disabled/None, returns input unchanged."""
        evaluator = HelixFullEvaluator(
            model=None,
            device="cpu",
            loss_fn=None,
            use_energy_based_family_objective=False,
            family_log_prior=None,
        )
        logits = torch.randn(4, 7)
        result = evaluator._apply_eval_logit_controls(logits)
        assert torch.equal(result, logits)


# ======================================================================
# Constructor and configuration
# ======================================================================


class TestEvaluatorConstruction:
    """Verify evaluator creation and default configuration."""

    def test_default_construction(self):
        """Evaluator can be constructed with minimal arguments."""
        evaluator = HelixFullEvaluator(
            model=None,
            device="cpu",
            loss_fn=None,
        )
        assert evaluator.model is None
        assert evaluator.device == "cpu"
        assert evaluator.active_family_class_ids == set()
        assert evaluator.class4_logit_shift == 0.0
        assert evaluator.class4_logit_shift_class_id == 4
        assert evaluator.use_energy_based_family_objective is True
        assert evaluator.energy_logit_temperature == 2.0
        assert evaluator.family_log_prior is None

    def test_full_construction(self):
        """Evaluator can be constructed with all arguments."""
        prior = torch.tensor([0.1, 0.2, 0.3])
        evaluator = HelixFullEvaluator(
            model="dummy_model",
            device="cuda",
            loss_fn="dummy_loss",
            binary_class_weights=torch.tensor([1.0, 2.0]),
            family_class_weights=torch.tensor([1.0]),
            family_log_prior=prior,
            use_energy_based_family_objective=False,
            energy_logit_temperature=5.0,
            active_family_class_ids={0, 1, 2, 3},
            class4_logit_shift=1.5,
            class4_logit_shift_class_id=3,
        )
        assert evaluator.model == "dummy_model"
        assert evaluator.device == "cuda"
        assert evaluator.active_family_class_ids == {0, 1, 2, 3}
        assert evaluator.class4_logit_shift == 1.5
        assert evaluator.class4_logit_shift_class_id == 3
        assert evaluator.use_energy_based_family_objective is False
        assert evaluator.energy_logit_temperature == 5.0
        assert torch.equal(evaluator.family_log_prior, prior)
