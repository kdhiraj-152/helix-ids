"""Comprehensive regression tests for the extracted validation calibrator module.

Phase 12B-4: covers all functions exported from
scripts/training/validation/calibrator.py.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from scripts.training.validation.calibrator import (
    _apply_class4_logit_shift,
    _calibrate_family_predictions,
    _fit_temperature_nll,
    _predict_with_class4_threshold,
    _softmax_with_temperature,
)

# ============================================================================
# _softmax_with_temperature
# ============================================================================


class TestSoftmaxWithTemperature:
    def test_basic_softmax(self) -> None:
        """Verify basic probability normalization with T=1.0."""
        logits = np.array([[1.0, 0.0, -1.0]], dtype=np.float64)
        probs = _softmax_with_temperature(logits, t=1.0)
        assert probs.shape == logits.shape
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert probs[0, 0] > probs[0, 1] > probs[0, 2]
        assert np.all(probs >= 0.0)

    def test_temperature_scaling_reduces_peakiness(self) -> None:
        """Higher temperature should produce more uniform probabilities."""
        logits = np.array([[5.0, 0.0, -5.0]], dtype=np.float64)
        probs_cold = _softmax_with_temperature(logits, t=0.5)
        probs_hot = _softmax_with_temperature(logits, t=5.0)
        max_cold = float(np.max(probs_cold))
        max_hot = float(np.max(probs_hot))
        assert max_cold > max_hot

    def test_low_temperature_sharpens(self) -> None:
        """Very low temperature should approach argmax behaviour."""
        logits = np.array([[1.0, 0.9, 0.8]], dtype=np.float64)
        probs = _softmax_with_temperature(logits, t=0.01)
        assert probs[0, 0] > 0.99

    def test_temperature_clipped_positive(self) -> None:
        """Temperature of 0 or negative should be clipped to 1e-6."""
        logits = np.array([[1.0, 0.0]], dtype=np.float64)
        probs_zero = _softmax_with_temperature(logits, t=0.0)
        probs_neg = _softmax_with_temperature(logits, t=-1.0)
        # Both should behave as if T≈1e-6 (near-argmax)
        assert np.allclose(probs_zero, probs_neg, atol=1e-10)

    def test_all_equal_logits(self) -> None:
        """Equal logits should produce uniform probabilities."""
        logits = np.array([[2.0, 2.0, 2.0]], dtype=np.float64)
        probs = _softmax_with_temperature(logits, t=1.0)
        assert np.allclose(probs, 1.0 / 3.0)

    def test_single_class_batch(self) -> None:
        """Single-class logits should produce probability 1.0."""
        logits = np.array([[5.0], [0.0], [-3.0]], dtype=np.float64)
        probs = _softmax_with_temperature(logits, t=1.0)
        assert probs.shape == (3, 1)
        assert np.allclose(probs, 1.0)

    def test_extreme_logits_overflow_safe(self) -> None:
        """Very large logits should not produce NaN due to numerical overflow."""
        logits = np.array([[1e5, 0.0, -1e5]], dtype=np.float64)
        probs = _softmax_with_temperature(logits, t=1.0)
        assert not np.any(np.isnan(probs))
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert probs[0, 0] > 0.99

    def test_multiple_batches(self) -> None:
        """Multiple samples should each have properly normalized probabilities."""
        rng = np.random.RandomState(42)
        logits = rng.randn(10, 5).astype(np.float64)
        probs = _softmax_with_temperature(logits, t=1.0)
        assert np.allclose(probs.sum(axis=1), np.ones(10))
        assert np.all(probs >= 0.0)

    def test_empty_input(self) -> None:
        """Empty logits should return an empty array."""
        logits = np.empty((0, 3), dtype=np.float64)
        probs = _softmax_with_temperature(logits, t=1.0)
        assert probs.shape == (0, 3)

    def test_deterministic(self) -> None:
        """Identical inputs should produce identical outputs."""
        logits = np.array([[0.5, 0.3, 0.2]], dtype=np.float64)
        p1 = _softmax_with_temperature(logits, t=2.0)
        p2 = _softmax_with_temperature(logits, t=2.0)
        assert np.array_equal(p1, p2)


# ============================================================================
# _fit_temperature_nll
# ============================================================================


class TestFitTemperatureNll:
    def test_temperature_improves_nll(self) -> None:
        """Temperature scaling should reduce NLL compared to T=1 where appropriate."""
        rng = np.random.RandomState(42)
        n_classes = 5
        n_samples = 100
        logits = rng.randn(n_samples, n_classes).astype(np.float64)
        labels = rng.randint(0, n_classes, size=n_samples).astype(np.int64)
        t_opt, nll_opt = _fit_temperature_nll(logits, labels, max_temperature=10.0)
        assert t_opt >= 1.0
        assert nll_opt > 0.0
        assert np.isfinite(t_opt)
        assert np.isfinite(nll_opt)

    def test_temperature_finite_range(self) -> None:
        """Fitted temperature must be finite and within expected bounds."""
        rng = np.random.RandomState(123)
        logits = rng.randn(50, 4).astype(np.float64)
        labels = rng.randint(0, 4, size=50).astype(np.int64)
        t_opt, nll_opt = _fit_temperature_nll(logits, labels, max_temperature=5.0)
        assert 1.0 <= t_opt <= 5.0
        assert np.isfinite(nll_opt)

    def test_overconfident_logits_yield_higher_temperature(self) -> None:
        """Overconfident (sharply peaked but wrong) logits may yield T > 1."""
        rng = np.random.RandomState(777)
        n_classes = 3
        n_samples = 50
        labels = rng.randint(0, n_classes, size=n_samples).astype(np.int64)
        # Make model overconfident about WRONG predictions: set a different class to 10.0
        logits = np.zeros((n_samples, n_classes), dtype=np.float64)
        for i in range(n_samples):
            wrong_class = (int(labels[i]) + 1) % n_classes
            logits[i, :] = -10.0
            logits[i, wrong_class] = 10.0
        t_opt, nll_opt = _fit_temperature_nll(logits, labels, max_temperature=10.0)
        assert t_opt > 1.0
        assert np.isfinite(nll_opt)

    def test_empty_logits(self) -> None:
        """Empty logits should return default T=1.0 and NLL=0.0."""
        logits = np.empty((0, 3), dtype=np.float64)
        labels = np.array([], dtype=np.int64)
        t, nll = _fit_temperature_nll(logits, labels, max_temperature=5.0)
        assert t == pytest.approx(1.0)
        assert nll == pytest.approx(0.0)

    def test_single_sample(self) -> None:
        """Single sample should still produce valid temperature."""
        logits = np.array([[1.0, 0.0, -1.0]], dtype=np.float64)
        labels = np.array([0], dtype=np.int64)
        t, nll = _fit_temperature_nll(logits, labels, max_temperature=5.0)
        assert np.isfinite(t)
        assert np.isfinite(nll)

    def test_deterministic_fit(self) -> None:
        """Identical inputs should produce identical temperature."""
        logits = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
        labels = np.array([0, 1], dtype=np.int64)
        t1, _ = _fit_temperature_nll(logits, labels, max_temperature=5.0)
        t2, _ = _fit_temperature_nll(logits, labels, max_temperature=5.0)
        assert t1 == t2


# ============================================================================
# _apply_class4_logit_shift
# ============================================================================


class TestApplyClass4LogitShift:
    def test_shift_applied_only_to_class4(self) -> None:
        """Shift should be applied only to the class-4 column."""
        logits = np.array(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
            dtype=np.float64,
        )
        shifted = _apply_class4_logit_shift(logits, class4_id=4, delta=0.5)
        assert shifted.shape == logits.shape
        assert shifted[0, 4] == pytest.approx(4.5)
        assert shifted[1, 4] == pytest.approx(0.0)
        assert np.allclose(shifted[:, :4], logits[:, :4])
        assert np.allclose(shifted[:, 5:], logits[:, 5:])

    def test_zero_shift_no_change(self) -> None:
        """Delta of 0.0 should produce identical output."""
        logits = np.random.RandomState(42).randn(5, 7).astype(np.float64)
        shifted = _apply_class4_logit_shift(logits, class4_id=4, delta=0.0)
        assert np.array_equal(shifted, logits)

    def test_negative_shift(self) -> None:
        """Negative delta should be subtracted (making class4 larger)."""
        logits = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=np.float64)
        shifted = _apply_class4_logit_shift(logits, class4_id=4, delta=-0.2)
        assert shifted[0, 4] == pytest.approx(0.7)

    def test_class4_id_out_of_range_returns_copy(self) -> None:
        """Invalid class4_id should return an unmodified copy."""
        logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        shifted_neg = _apply_class4_logit_shift(logits, class4_id=-1, delta=0.5)
        assert np.array_equal(shifted_neg, logits)
        shifted_high = _apply_class4_logit_shift(logits, class4_id=5, delta=0.5)
        assert np.array_equal(shifted_high, logits)

    def test_does_not_mutate_input(self) -> None:
        """Function must not modify the input array."""
        logits = np.array([[1.0, 0.0, 4.0, 0.0, 2.0]], dtype=np.float64)
        original = logits.copy()
        _apply_class4_logit_shift(logits, class4_id=4, delta=0.3)
        assert np.array_equal(logits, original)

    def test_empty_input(self) -> None:
        """Empty logits should return an empty copy."""
        logits = np.empty((0, 5), dtype=np.float64)
        shifted = _apply_class4_logit_shift(logits, class4_id=4, delta=0.5)
        assert shifted.shape == (0, 5)

    def test_single_row(self) -> None:
        """Single-row input should work correctly."""
        logits = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
        shifted = _apply_class4_logit_shift(logits, class4_id=4, delta=0.5)
        assert shifted[0, 4] == pytest.approx(0.5)

    def test_deterministic(self) -> None:
        """Identical inputs should produce identical outputs."""
        logits = np.array([[0.5, 0.5, 0.5, 0.5, 0.5]], dtype=np.float64)
        s1 = _apply_class4_logit_shift(logits, class4_id=4, delta=0.25)
        s2 = _apply_class4_logit_shift(logits, class4_id=4, delta=0.25)
        assert np.array_equal(s1, s2)


# ============================================================================
# _predict_with_class4_threshold
# ============================================================================


class TestPredictWithClass4Threshold:
    def test_high_class4_probability_predicts_class4(self) -> None:
        """When P(class4) >= threshold, prediction should be class4_id."""
        probs = np.array(
            [[0.1, 0.1, 0.1, 0.1, 0.6, 0.0], [0.8, 0.05, 0.05, 0.05, 0.05, 0.0]],
            dtype=np.float64,
        )
        preds = _predict_with_class4_threshold(probs, class4_id=4, threshold=0.5)
        assert preds[0] == 4
        assert preds[1] == 0  # class4 prob < threshold

    def test_threshold_gating(self) -> None:
        """Class-4 predictions should be gated by threshold."""
        probs = np.array(
            [[0.1, 0.1, 0.1, 0.1, 0.45, 0.15], [0.1, 0.1, 0.1, 0.1, 0.55, 0.05]],
            dtype=np.float64,
        )
        preds = _predict_with_class4_threshold(probs, class4_id=4, threshold=0.5)
        assert preds[0] != 4  # 0.45 < 0.5
        assert preds[1] == 4  # 0.55 >= 0.5

    def test_class4_id_out_of_range_falls_back_to_argmax(self) -> None:
        """Invalid class4_id should fall back to standard argmax."""
        probs = np.array(
            [[0.2, 0.5, 0.3], [0.6, 0.2, 0.2]], dtype=np.float64
        )
        preds = _predict_with_class4_threshold(probs, class4_id=5, threshold=0.5)
        assert preds[0] == 1
        assert preds[1] == 0

    def test_empty_input(self) -> None:
        """Empty probs should return an empty array."""
        probs = np.empty((0, 5), dtype=np.float64)
        preds = _predict_with_class4_threshold(probs, class4_id=4, threshold=0.5)
        assert preds.shape == (0,)

    def test_all_class4_above_threshold(self) -> None:
        """When all samples exceed threshold, all should be class4."""
        probs = np.array(
            [[0.1, 0.1, 0.1, 0.1, 0.6], [0.05, 0.05, 0.05, 0.05, 0.8]],
            dtype=np.float64,
        )
        preds = _predict_with_class4_threshold(probs, class4_id=4, threshold=0.5)
        assert list(preds) == [4, 4]

    def test_no_class4_above_threshold(self) -> None:
        """When no samples exceed threshold, none should be class4."""
        probs = np.array(
            [[0.3, 0.3, 0.3, 0.0, 0.1], [0.4, 0.4, 0.1, 0.0, 0.1]],
            dtype=np.float64,
        )
        preds = _predict_with_class4_threshold(probs, class4_id=4, threshold=0.5)
        assert preds[0] != 4
        assert preds[1] != 4

    def test_probability_normalization_not_required(self) -> None:
        """Function should work even if rows are not perfectly normalized."""
        probs = np.array(
            [[0.2, 0.2, 0.2, 0.2, 0.6]], dtype=np.float64
        )
        preds = _predict_with_class4_threshold(probs, class4_id=4, threshold=0.5)
        assert preds[0] == 4  # 0.6 >= 0.5

    def test_deterministic(self) -> None:
        """Identical inputs should produce identical outputs."""
        probs = np.array(
            [[0.3, 0.1, 0.1, 0.0, 0.5], [0.1, 0.1, 0.2, 0.0, 0.6]],
            dtype=np.float64,
        )
        p1 = _predict_with_class4_threshold(probs, class4_id=4, threshold=0.5)
        p2 = _predict_with_class4_threshold(probs, class4_id=4, threshold=0.5)
        assert np.array_equal(p1, p2)


# ============================================================================
# _calibrate_family_predictions (integration-level)
# ============================================================================


class _MockModelWithEval:
    """Mock model that provides .eval() for the calibrator contract."""

    def eval(self) -> None:
        return None

    def __call__(self, x: Any) -> tuple[Any, Any]:
        raise NotImplementedError("Should not be called when _collect_eval_family_outputs is monkeypatched.")


class TestCalibrateFamilyPredictions:
    """Integration-level tests using monkeypatched _collect_eval_family_outputs."""

    def test_class4_logit_shift_changes_uncalibrated_ranking(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verifies that applying class-4 logit shift changes uncalibrated precision."""
        from scripts.training.validation import calibrator as val_calibrator

        labels = np.array([4, 0, 0, 0], dtype=np.int64)
        logits = np.array(
            [
                [0.10, 0.20, 0.30, 0.40, 1.20, 0.60, 0.70],
                [0.95, 0.10, 0.10, 0.10, 1.00, 0.20, 0.10],
                [0.95, 0.10, 0.10, 0.10, 1.00, 0.20, 0.10],
                [0.95, 0.10, 0.10, 0.10, 1.00, 0.20, 0.10],
            ],
            dtype=np.float64,
        )

        def _fake_collect_eval_family_outputs(**kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            _ = kwargs
            probs = np.zeros_like(logits)
            return labels.copy(), logits.copy(), probs

        monkeypatch.setattr(
            val_calibrator, "_collect_eval_family_outputs",
            _fake_collect_eval_family_outputs,
        )

        payload_no_shift = _calibrate_family_predictions(
            model=_MockModelWithEval(),
            val_loader=cast(Any, object()),
            test_loader=cast(Any, object()),
            device="cpu",
            class4_id=4,
            threshold_grid=np.array([0.5], dtype=np.float64),
            min_class4_recall=0.0,
            class4_logit_shift=0.0,
        )
        payload_shift = _calibrate_family_predictions(
            model=_MockModelWithEval(),
            val_loader=cast(Any, object()),
            test_loader=cast(Any, object()),
            device="cpu",
            class4_id=4,
            threshold_grid=np.array([0.5], dtype=np.float64),
            min_class4_recall=0.0,
            class4_logit_shift=0.1,
        )

        base_precision = float(payload_no_shift["uncalibrated"]["test_argmax"]["class4_precision"])
        shifted_precision = float(payload_shift["uncalibrated"]["test_argmax"]["class4_precision"])
        assert shifted_precision > base_precision
        assert float(payload_shift["class4_logit_shift"]) == pytest.approx(0.1)

    def test_empty_data_returns_default_payload(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Empty data should return default calibration payload with T=1, tau=0.5."""
        from scripts.training.validation import calibrator as val_calibrator

        def _fake_empty_outputs(**kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return (
                np.array([], dtype=np.int64),
                np.empty((0, 0), dtype=np.float64),
                np.empty((0, 0), dtype=np.float64),
            )

        monkeypatch.setattr(
            val_calibrator, "_collect_eval_family_outputs",
            _fake_empty_outputs,
        )

        payload = _calibrate_family_predictions(
            model=_MockModelWithEval(),
            val_loader=cast(Any, object()),
            test_loader=cast(Any, object()),
            device="cpu",
            class4_id=4,
        )
        assert payload["class4_logit_shift"] == pytest.approx(0.0)
        assert payload["temperature"] == pytest.approx(1.0)
        assert payload["tau_4"] == pytest.approx(0.5)
        assert payload["uncalibrated"]["val_argmax"]["class4_precision"] == pytest.approx(0.0)
        assert payload["uncalibrated"]["test_argmax"]["class4_precision"] == pytest.approx(0.0)
        assert payload["threshold_sweep"]["num_points"] > 0

    def test_payload_structure_is_complete(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify the returned payload has all expected top-level keys."""
        from scripts.training.validation import calibrator as val_calibrator

        labels = np.array([0, 1, 2, 3], dtype=np.int64)
        logits = np.random.RandomState(42).randn(4, 5).astype(np.float64)

        def _fake_outputs(**kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            probs = np.zeros_like(logits)
            return labels.copy(), logits.copy(), probs

        monkeypatch.setattr(
            val_calibrator, "_collect_eval_family_outputs",
            _fake_outputs,
        )

        payload = _calibrate_family_predictions(
            model=_MockModelWithEval(),
            val_loader=cast(Any, object()),
            test_loader=cast(Any, object()),
            device="cpu",
            class4_id=4,
        )

        expected_keys = {
            "class4_logit_shift", "temperature", "tau_4",
            "uncalibrated", "val", "test", "ablation",
            "pr_curve_class4", "threshold_sweep",
        }
        assert set(payload.keys()) == expected_keys
        assert "val_argmax" in payload["uncalibrated"]
        assert "test_argmax" in payload["uncalibrated"]
        assert "without_thresholding" in payload["ablation"]
        assert "without_temperature_scaling" in payload["ablation"]
        assert "precision" in payload["pr_curve_class4"]
        assert "points" in payload["threshold_sweep"]

    def test_threshold_sweep_tau_min_max(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Threshold sweep metadata should reflect input grid."""
        from scripts.training.validation import calibrator as val_calibrator

        labels = np.array([0, 1, 2, 3], dtype=np.int64)
        logits = np.random.RandomState(42).randn(4, 5).astype(np.float64)

        def _fake_outputs(**kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            probs = np.zeros_like(logits)
            return labels.copy(), logits.copy(), probs

        monkeypatch.setattr(
            val_calibrator, "_collect_eval_family_outputs",
            _fake_outputs,
        )

        grid = np.linspace(0.3, 0.95, 10)
        payload = _calibrate_family_predictions(
            model=_MockModelWithEval(),
            val_loader=cast(Any, object()),
            test_loader=cast(Any, object()),
            device="cpu",
            class4_id=4,
            threshold_grid=grid,
        )
        assert payload["threshold_sweep"]["tau_min"] == pytest.approx(0.3)
        assert payload["threshold_sweep"]["tau_max"] == pytest.approx(0.95)
        assert payload["threshold_sweep"]["num_points"] == 10

    def test_deterministic_with_fixed_seed(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Calibration should be deterministic for identical inputs."""
        from scripts.training.validation import calibrator as val_calibrator

        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        logits = np.array(
            [
                [2.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 2.0, 0.0],
            ],
            dtype=np.float64,
        )

        def _fake_outputs(**kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            probs = np.zeros_like(logits)
            return labels.copy(), logits.copy(), probs

        monkeypatch.setattr(
            val_calibrator, "_collect_eval_family_outputs",
            _fake_outputs,
        )

        p1 = _calibrate_family_predictions(
            model=_MockModelWithEval(),
            val_loader=cast(Any, object()),
            test_loader=cast(Any, object()),
            device="cpu",
            class4_id=4,
            threshold_grid=np.array([0.5], dtype=np.float64),
            min_class4_recall=0.0,
        )
        p2 = _calibrate_family_predictions(
            model=_MockModelWithEval(),
            val_loader=cast(Any, object()),
            test_loader=cast(Any, object()),
            device="cpu",
            class4_id=4,
            threshold_grid=np.array([0.5], dtype=np.float64),
            min_class4_recall=0.0,
        )
        assert p1["temperature"] == p2["temperature"]
        assert p1["tau_4"] == p2["tau_4"]
        assert p1["class4_logit_shift"] == p2["class4_logit_shift"]
