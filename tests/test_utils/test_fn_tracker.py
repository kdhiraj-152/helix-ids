"""Tests for FalseNegativeTracker"""

import numpy as np
import pytest

from helix_ids.metrics.fn_tracker import FalseNegativeTracker, FNThresholds


class TestFNThresholds:
    """Tests for FNThresholds configuration."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = FNThresholds()

        assert thresholds.normal == pytest.approx(0.01)
        assert thresholds.dos == pytest.approx(0.02)
        assert thresholds.probe == pytest.approx(0.02)
        assert thresholds.r2l == pytest.approx(0.05)
        assert thresholds.u2r == pytest.approx(0.05)

    def test_get_threshold_by_class(self):
        """Test getting threshold for specific class."""
        thresholds = FNThresholds()

        assert thresholds.get_threshold("Normal") == pytest.approx(0.01)
        assert thresholds.get_threshold("DoS") == pytest.approx(0.02)
        assert thresholds.get_threshold("Probe") == pytest.approx(0.02)
        assert thresholds.get_threshold("R2L") == pytest.approx(0.05)
        assert thresholds.get_threshold("U2R") == pytest.approx(0.05)
        assert thresholds.get_threshold("Unknown") == pytest.approx(0.02)  # Default

    def test_is_critical_class(self):
        """Test critical class detection."""
        thresholds = FNThresholds()

        assert thresholds.is_critical_class("R2L")
        assert thresholds.is_critical_class("U2R")
        assert not thresholds.is_critical_class("DoS")
        assert not thresholds.is_critical_class("Probe")
        assert not thresholds.is_critical_class("Normal")

    def test_custom_thresholds(self):
        """Test custom threshold values."""
        custom = FNThresholds(normal=0.02, dos=0.03, r2l=0.04)

        assert custom.normal == pytest.approx(0.02)
        assert custom.dos == pytest.approx(0.03)
        assert custom.probe == pytest.approx(0.02)  # Unchanged
        assert custom.r2l == pytest.approx(0.04)
        assert custom.u2r == pytest.approx(0.05)  # Unchanged


class TestFalseNegativeTracker:
    """Tests for FalseNegativeTracker."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = FalseNegativeTracker()

        assert tracker.current_epoch == 0
        assert len(tracker.class_names) == 5
        assert "R2L" in tracker.class_names
        assert all(tracker.fn_rates[cls] == pytest.approx(0.0, abs=1e-9) for cls in tracker.class_names)
        assert len(tracker.history) == 0

    def test_custom_classes(self):
        """Test tracker with custom class names."""
        custom_classes = ["Attack1", "Attack2", "Attack3"]
        tracker = FalseNegativeTracker(class_names=custom_classes)

        assert tracker.class_names == custom_classes
        assert len(tracker.fn_rates) == 3

    def test_perfect_predictions(self):
        """Test when all predictions are correct (no FN)."""
        tracker = FalseNegativeTracker()

        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        y_pred = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

        tracker.update(y_true, y_pred)

        for cls, rate in tracker.fn_rates.items():
            assert rate == pytest.approx(0.0, abs=1e-9), f"Expected FN rate 0.0 for {cls}, got {rate}"

    def test_all_false_negatives(self):
        """Test when all predictions are false negatives for a class."""
        tracker = FalseNegativeTracker()

        # All class 0 instances mislabeled
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
        y_pred = np.array([1, 1, 2, 2, 1, 1, 1, 1, 2, 2])

        tracker.update(y_true, y_pred)

        # Class 0: 4 instances, 4 FN, FN rate = 1.0
        assert tracker.fn_rates["Normal"] == pytest.approx(1.0)
        assert tracker.fn_counts["Normal"] == 4
        assert tracker.class_totals["Normal"] == 4

    def test_partial_false_negatives(self):
        """Test with some false negatives."""
        tracker = FalseNegativeTracker()

        # Class 0: 4 instances, 1 correct, 3 FN → FN rate = 0.75
        # Class 1: 4 instances, 3 correct, 1 FN → FN rate = 0.25
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 2, 1, 1, 1, 0, 2, 2])

        tracker.update(y_true, y_pred)

        assert tracker.fn_rates["Normal"] == pytest.approx(0.75)  # 3/4
        assert tracker.fn_rates["DoS"] == pytest.approx(0.25)  # 1/4
        assert tracker.fn_counts["Normal"] == 3
        assert tracker.fn_counts["DoS"] == 1

    def test_string_labels(self):
        """Test with string class labels."""
        tracker = FalseNegativeTracker()

        y_true = np.array(["Normal", "Normal", "DoS", "DoS", "Probe", "Probe"])
        y_pred = np.array(["Normal", "DoS", "DoS", "Probe", "Probe", "DoS"])

        tracker.update(y_true, y_pred)

        # Normal: 2 instances, 1 correct, 1 FN → FN rate = 0.5
        # DoS: 2 instances, 1 correct, 1 FN → FN rate = 0.5
        # Probe: 2 instances, 1 correct, 1 FN → FN rate = 0.5
        assert tracker.fn_rates["Normal"] == pytest.approx(0.5)
        assert tracker.fn_rates["DoS"] == pytest.approx(0.5)
        assert tracker.fn_rates["Probe"] == pytest.approx(0.5)

    def test_get_fn_rates(self):
        """Test getting FN rates."""
        tracker = FalseNegativeTracker()

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])

        tracker.update(y_true, y_pred)

        fn_rates = tracker.get_fn_rates()

        assert isinstance(fn_rates, dict)
        assert fn_rates["Normal"] == pytest.approx(0.5)  # 1/2
        assert fn_rates["DoS"] == pytest.approx(0.5)  # 1/2

    def test_get_fn_counts(self):
        """Test getting FN counts."""
        tracker = FalseNegativeTracker()

        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])

        tracker.update(y_true, y_pred)

        fn_counts = tracker.get_fn_counts()

        assert fn_counts["Normal"] == 2
        assert fn_counts["DoS"] == 1

    def test_check_thresholds_pass(self):
        """Test check_thresholds when all thresholds pass."""
        tracker = FalseNegativeTracker()

        # FN rates well below thresholds
        y_true = np.array([0] * 100 + [1] * 100 + [3] * 100 + [4] * 100)
        y_pred = y_true.copy()
        y_pred[0] = 1  # One FN for Normal: 0.01

        tracker.update(y_true, y_pred)

        violations = tracker.check_thresholds()

        assert len(violations) == 0

    def test_check_thresholds_violations(self):
        """Test check_thresholds when thresholds are violated."""
        tracker = FalseNegativeTracker()

        # High FN rate for R2L (exceeds 5% threshold)
        y_true = np.array([0, 0, 3, 3, 3, 3, 3, 3, 3, 3])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 3, 3, 3, 3])

        tracker.update(y_true, y_pred)

        violations = tracker.check_thresholds()

        assert len(violations) > 0
        assert any("R2L" in v for v in violations)

    def test_alert_critical_pass(self):
        """Test alert_critical when critical classes are within threshold."""
        tracker = FalseNegativeTracker()

        # Perfect predictions for critical classes
        y_true = np.array([0, 0, 1, 1, 3, 3, 4, 4])
        y_pred = y_true.copy()

        tracker.update(y_true, y_pred)

        assert not tracker.alert_critical()

    def test_alert_critical_fail(self):
        """Test alert_critical when critical class exceeds threshold."""
        tracker = FalseNegativeTracker()

        # High FN for U2R (exceeds 5% threshold)
        y_true = np.array([0, 0, 1, 1, 3, 3, 4, 4, 4, 4])
        y_pred = np.array([0, 0, 1, 1, 3, 3, 0, 0, 4, 4])

        tracker.update(y_true, y_pred)

        assert tracker.alert_critical()

    def test_new_epoch(self):
        """Test epoch tracking."""
        tracker = FalseNegativeTracker()

        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 2, 2, 0])

        # Epoch 0
        tracker.update(y_true, y_pred)
        tracker.new_epoch(0)

        assert len(tracker.history) == 1
        assert tracker.history[0].epoch == 0
        assert tracker.history[0].fn_rates["Normal"] == pytest.approx(0.5)

        # Epoch 1
        y_true_2 = np.array([0, 0, 0, 1, 1])
        y_pred_2 = np.array([0, 0, 1, 1, 0])
        tracker.update(y_true_2, y_pred_2)
        tracker.new_epoch(1)

        assert len(tracker.history) == 2
        assert tracker.history[1].epoch == 1

    def test_get_epoch_history(self):
        """Test retrieving epoch history."""
        tracker = FalseNegativeTracker()

        for epoch in range(3):
            y_true = np.array([0, 0, 1, 1])
            y_pred = np.array([0, 1, 1, 0])
            tracker.update(y_true, y_pred)
            tracker.new_epoch(epoch)

        history = tracker.get_epoch_history()

        assert len(history) == 3
        assert history[0].epoch == 0
        assert history[1].epoch == 1
        assert history[2].epoch == 2

    def test_should_stop_early_no_history(self):
        """Test early stopping with no history."""
        tracker = FalseNegativeTracker()

        assert not tracker.should_stop_early()

    def test_should_stop_early_critical_alert(self):
        """Test early stopping triggered by critical alert."""
        tracker = FalseNegativeTracker()

        # Create pattern: critical alert for 3 consecutive epochs
        for epoch in range(3):
            # High FN for U2R
            y_true = np.array([4] * 10)
            y_pred = np.array([0, 0, 0, 0, 0, 0, 4, 4, 4, 4])
            tracker.update(y_true, y_pred)
            tracker.new_epoch(epoch)

        assert tracker.should_stop_early(patience=3)

    def test_should_stop_early_degradation(self):
        """Test early stopping triggered by FN rate degradation."""
        tracker = FalseNegativeTracker()

        # Simulate degrading FN rate for R2L
        fn_rates_pattern = [0.02, 0.03, 0.04, 0.05]  # Increasing trend

        for epoch, fn_rate in enumerate(fn_rates_pattern):
            # Create predictions with specific FN rate
            n_instances = 100
            n_fn = int(n_instances * fn_rate)
            y_true = np.array([3] * n_instances)
            y_pred = np.array([0] * n_fn + [3] * (n_instances - n_fn))
            tracker.update(y_true, y_pred)
            tracker.new_epoch(epoch)

        assert tracker.should_stop_early(patience=3, degradation_threshold=0.1)

    def test_get_summary_stats(self):
        """Test getting summary statistics."""
        tracker = FalseNegativeTracker()

        for epoch in range(3):
            y_true = np.array([0, 0, 1, 1])
            y_pred = np.array([0, 1, 1, 0])
            tracker.update(y_true, y_pred)
            tracker.new_epoch(epoch)

        stats = tracker.get_summary_stats()

        assert "Normal" in stats
        assert "min" in stats["Normal"]
        assert "max" in stats["Normal"]
        assert "mean" in stats["Normal"]
        assert "std" in stats["Normal"]
        assert stats["Normal"]["min"] == pytest.approx(0.5)
        assert stats["Normal"]["max"] == pytest.approx(0.5)
        assert stats["Normal"]["mean"] == pytest.approx(0.5)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        tracker = FalseNegativeTracker()

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        tracker.update(y_true, y_pred)

        state_dict = tracker.to_dict()

        assert "current_epoch" in state_dict
        assert "fn_rates" in state_dict
        assert "fn_counts" in state_dict
        assert "violations" in state_dict
        assert "critical_alert" in state_dict
        assert state_dict["fn_rates"]["Normal"] == pytest.approx(0.5)

    def test_empty_class(self):
        """Test handling of class with no instances."""
        tracker = FalseNegativeTracker()

        # No instances of classes 2 and 3
        y_true = np.array([0, 0, 1, 1, 4, 4])
        y_pred = np.array([0, 1, 1, 0, 4, 4])

        tracker.update(y_true, y_pred)

        assert tracker.fn_rates["Probe"] == pytest.approx(0.0, abs=1e-9)  # No instances
        assert tracker.fn_rates["R2L"] == pytest.approx(0.0, abs=1e-9)  # No instances
        assert tracker.class_totals["Probe"] == 0
        assert tracker.class_totals["R2L"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
