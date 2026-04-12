"""
Tests for metrics module.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from helix_ids.utils.metrics import (
    THREAT_WEIGHTS,
    ModelMetrics,
    calculate_per_class_f1,
    calculate_pri_score,
    calculate_threat_weighted_f1,
)


class TestModelMetrics:
    """Test ModelMetrics dataclass."""

    def test_metrics_default_values(self):
        """Test default values are initialized."""
        metrics = ModelMetrics()

        assert metrics.accuracy == pytest.approx(0.0)
        assert metrics.macro_f1 == pytest.approx(0.0)
        assert metrics.r2l_f1 == pytest.approx(0.0)
        assert metrics.u2r_f1 == pytest.approx(0.0)

    def test_metrics_to_dict(self):
        """Test conversion to dictionary."""
        metrics = ModelMetrics(accuracy=0.95, macro_f1=0.85)
        d = metrics.to_dict()

        assert isinstance(d, dict)
        assert d["accuracy"] == pytest.approx(0.95)
        assert d["macro_f1"] == pytest.approx(0.85)

    def test_metrics_with_values(self):
        """Test metrics with set values."""
        metrics = ModelMetrics(
            accuracy=0.95,
            macro_f1=0.85,
            r2l_f1=0.45,
            u2r_f1=0.30,
            model_size_kb=25.6,
        )

        assert metrics.accuracy == pytest.approx(0.95)
        assert metrics.r2l_f1 == pytest.approx(0.45)
        assert metrics.model_size_kb == pytest.approx(25.6)


class TestCalculatePerClassF1:
    """Test calculate_per_class_f1 function."""

    def test_per_class_f1_perfect(self):
        """Test F1 with perfect predictions."""
        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])

        f1_scores = calculate_per_class_f1(y_true, y_pred)

        # All classes should have F1=1.0
        for score in f1_scores.values():
            assert score == pytest.approx(1.0)

    def test_per_class_f1_with_names(self):
        """Test F1 with class names."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        f1_scores = calculate_per_class_f1(y_true, y_pred, class_names=["Normal", "DoS", "Probe"])

        assert "Normal" in f1_scores
        assert "DoS" in f1_scores
        assert "Probe" in f1_scores

    def test_per_class_f1_zero_for_missed(self):
        """Test F1 is 0 for classes never predicted."""
        y_true = np.array([0, 0, 0, 1, 2])
        y_pred = np.array([0, 0, 0, 0, 0])  # Never predicts 1 or 2

        f1_scores = calculate_per_class_f1(y_true, y_pred)

        assert f1_scores["1"] == pytest.approx(0.0)
        assert f1_scores["2"] == pytest.approx(0.0)


class TestThreatWeightedF1:
    """Test threat-weighted F1 calculation."""

    def test_threat_weighted_f1_perfect(self):
        """Test threat-weighted F1 with perfect per-class F1."""
        per_class_f1 = {
            "Normal": 1.0,
            "DoS": 1.0,
            "Probe": 1.0,
            "R2L": 1.0,
            "U2R": 1.0,
        }

        tw_f1 = calculate_threat_weighted_f1(per_class_f1)

        assert tw_f1 == pytest.approx(1.0)

    def test_threat_weighted_f1_minority_matters(self):
        """Test that minority class F1 affects TW-F1 more."""
        # Good at majority, bad at U2R
        f1_miss_u2r = {
            "Normal": 1.0,
            "DoS": 1.0,
            "Probe": 1.0,
            "R2L": 1.0,
            "U2R": 0.0,  # Miss U2R
        }

        # Bad at majority, good at U2R
        f1_miss_normal = {
            "Normal": 0.0,  # Miss Normal
            "DoS": 1.0,
            "Probe": 1.0,
            "R2L": 1.0,
            "U2R": 1.0,
        }

        tw_f1_miss_u2r = calculate_threat_weighted_f1(f1_miss_u2r)
        tw_f1_miss_normal = calculate_threat_weighted_f1(f1_miss_normal)

        # Missing U2R (weight=10) should hurt more than missing Normal (weight=1)
        assert tw_f1_miss_u2r < tw_f1_miss_normal


class TestPRIScore:
    """Test PRI (Production Readiness Index) calculation."""

    def test_pri_score_range(self):
        """Test PRI score is in valid range."""
        _, pri = calculate_pri_score(
            has_hardware_spec=True,
            has_cross_dataset=True,
            has_per_class_metrics=True,
        )

        assert 0 <= pri <= 1

    def test_pri_score_full_marks(self):
        """Test PRI with all criteria met."""
        _, pri = calculate_pri_score(
            has_hardware_spec=True,
            has_cross_dataset=True,
            has_power_measurement=True,
            has_per_class_metrics=True,
            has_drift_protocol=True,
            has_xai_quantified=True,
        )

        assert pri == pytest.approx(1.0)

    def test_pri_score_zero(self):
        """Test PRI with no criteria met."""
        _, pri = calculate_pri_score()

        assert pri == pytest.approx(0.0)

    def test_pri_score_partial_credit(self):
        """Test PRI gives partial credit for some criteria."""
        _, pri_partial = calculate_pri_score(
            has_cross_dataset=False,
            cross_dataset_partial=True,
        )

        _, pri_full = calculate_pri_score(
            has_cross_dataset=True,
        )

        _, pri_none = calculate_pri_score(
            has_cross_dataset=False,
            cross_dataset_partial=False,
        )

        assert pri_none < pri_partial < pri_full

    def test_pri_production_threshold(self):
        """Test PRI threshold for production readiness (0.70)."""
        # Meeting major criteria should exceed threshold
        _, pri = calculate_pri_score(
            has_hardware_spec=True,
            has_cross_dataset=True,
            has_per_class_metrics=True,
            has_power_measurement=True,
        )

        assert pri >= 0.70


class TestThreatWeights:
    """Test threat weight constants."""

    def test_threat_weights_exist(self):
        """Test all threat weights are defined."""
        assert "Normal" in THREAT_WEIGHTS
        assert "DoS" in THREAT_WEIGHTS
        assert "Probe" in THREAT_WEIGHTS
        assert "R2L" in THREAT_WEIGHTS
        assert "U2R" in THREAT_WEIGHTS

    def test_threat_weights_ordering(self):
        """Test threat weights reflect severity."""
        assert THREAT_WEIGHTS["Normal"] < THREAT_WEIGHTS["DoS"]
        assert THREAT_WEIGHTS["DoS"] < THREAT_WEIGHTS["R2L"]
        assert THREAT_WEIGHTS["R2L"] < THREAT_WEIGHTS["U2R"]
