"""
Tests for per-class metrics tracking.
"""

import numpy as np
import pytest

from src.helix_ids.metrics import (
    DEFAULT_THRESHOLDS,
    ClassMetrics,
    PerClassMetrics,
    PerClassMetricsResult,
)


class TestClassMetrics:
    """Test ClassMetrics dataclass."""

    def test_to_dict_without_auc(self):
        """Test conversion to dict without AUC."""
        metrics = ClassMetrics(
            precision=0.95,
            recall=0.92,
            f1=0.93,
            support=100,
        )
        result = metrics.to_dict()

        assert result["precision"] == pytest.approx(0.95)
        assert result["recall"] == pytest.approx(0.92)
        assert result["f1"] == pytest.approx(0.93)
        assert result["support"] == 100
        assert "auc_roc" not in result

    def test_to_dict_with_auc(self):
        """Test conversion to dict with AUC."""
        metrics = ClassMetrics(
            precision=0.95,
            recall=0.92,
            f1=0.93,
            support=100,
            auc_roc=0.98,
        )
        result = metrics.to_dict()

        assert result["auc_roc"] == pytest.approx(0.98)
        assert "auc_roc" in result


class TestPerClassMetrics:
    """Test PerClassMetrics computation."""

    @pytest.fixture
    def class_names(self):
        """Default class names for IDS."""
        return ["Normal", "DoS", "Probe", "R2L", "U2R"]

    @pytest.fixture
    def pcm(self, class_names):
        """Create PerClassMetrics instance."""
        return PerClassMetrics(class_names)

    def test_initialization(self, class_names):
        """Test initialization with default thresholds."""
        pcm = PerClassMetrics(class_names)

        assert pcm.class_names == class_names
        assert pcm.num_classes == 5
        assert pcm.thresholds == DEFAULT_THRESHOLDS

    def test_initialization_custom_thresholds(self, class_names):
        """Test initialization with custom thresholds."""
        custom_thresholds = {"Normal": 0.99, "DoS": 0.90}
        pcm = PerClassMetrics(class_names, thresholds=custom_thresholds)

        assert pcm.thresholds == custom_thresholds

    def test_compute_perfect_predictions(self, pcm):
        """Test with perfect predictions."""
        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2])
        y_pred = y_true.copy()

        result = pcm.compute(y_true, y_pred)

        assert result.macro_f1 == pytest.approx(1.0)
        assert result.weighted_f1 == pytest.approx(1.0)

        for class_name in pcm.class_names:
            assert result.per_class[class_name].f1 == pytest.approx(1.0)
            assert result.per_class[class_name].precision == pytest.approx(1.0)
            assert result.per_class[class_name].recall == pytest.approx(1.0)

    def test_compute_with_some_errors(self, pcm):
        """Test with some prediction errors."""
        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        y_pred = np.array([0, 0, 1, 0, 2, 2, 3, 3, 4, 3])  # One error in class 1, one in class 4

        result = pcm.compute(y_true, y_pred)

        assert 0 <= result.macro_f1 <= 1.0
        assert 0 <= result.weighted_f1 <= 1.0

        for class_name in pcm.class_names:
            assert 0 <= result.per_class[class_name].f1 <= 1.0
            assert 0 <= result.per_class[class_name].precision <= 1.0
            assert 0 <= result.per_class[class_name].recall <= 1.0

    def test_support_counting(self, pcm):
        """Test that support (class counts) are computed correctly."""
        y_true = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
        y_pred = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])

        result = pcm.compute(y_true, y_pred)

        assert result.per_class["Normal"].support == 3
        assert result.per_class["DoS"].support == 2
        assert result.per_class["Probe"].support == 4

    def test_confusion_matrix_shape(self, pcm):
        """Test confusion matrix generation."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])

        result = pcm.compute(y_true, y_pred)

        assert result.confusion_matrix is not None
        assert result.confusion_matrix.shape == (5, 5)

    def test_violations_detection(self, pcm):
        """Test threshold violation detection."""
        # Create predictions that fail for minority classes
        y_true = np.array([0] * 100 + [4] * 5)  # Mostly class 0, few class 4
        y_pred = np.array([0] * 100 + [0] * 5)  # Misclassify all U2R (class 4)

        result = pcm.compute(y_true, y_pred)

        # U2R should have F1 = 0, which is below threshold of 0.60
        assert "U2R" in str(result.violations) or len(result.violations) > 0

    def test_with_probability_predictions(self, pcm):
        """Test AUC-ROC computation with probability predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 1, 0, 2, 2, 1])

        # Create fake probability predictions
        rng = np.random.default_rng(seed=42)
        y_proba = rng.dirichlet([1, 1, 1, 1, 1], size=len(y_true))

        result = pcm.compute(y_true, y_pred, y_proba=y_proba)

        # Check that AUC-ROC values are computed
        for class_name in pcm.class_names[:3]:
            # Only check first 3 classes that have samples
            metrics = result.per_class[class_name]
            if metrics.support > 0:
                # AUC-ROC should be computed if available
                assert metrics.auc_roc is None or 0 <= metrics.auc_roc <= 1.0

    def test_check_violations(self, pcm):
        """Test violation checking logic."""
        per_class = {
            "Normal": ClassMetrics(f1=0.99),  # Above threshold 0.98
            "DoS": ClassMetrics(f1=0.93),  # Below threshold 0.95
            "Probe": ClassMetrics(f1=0.91),  # Above threshold 0.90
            "R2L": ClassMetrics(f1=0.70),  # Below threshold 0.80
            "U2R": ClassMetrics(f1=0.55),  # Below threshold 0.60
        }

        violations = pcm._check_violations(per_class)

        assert len(violations) == 3  # DoS, R2L, U2R
        assert any("DoS" in v for v in violations)
        assert any("R2L" in v for v in violations)
        assert any("U2R" in v for v in violations)


class TestPerClassMetricsResult:
    """Test PerClassMetricsResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        per_class = {
            "Normal": ClassMetrics(precision=0.99, recall=0.98, f1=0.985, support=1000),
            "DoS": ClassMetrics(precision=0.95, recall=0.94, f1=0.945, support=500),
        }

        result = PerClassMetricsResult(
            per_class=per_class,
            macro_f1=0.965,
            weighted_f1=0.970,
        )

        data = result.to_dict()

        assert data["macro_f1"] == pytest.approx(0.965)
        assert data["weighted_f1"] == pytest.approx(0.970)
        assert "Normal" in data["per_class"]
        assert "DoS" in data["per_class"]
        assert data["per_class"]["Normal"]["f1"] == pytest.approx(0.985)


class TestReporting:
    """Test reporting functionality."""

    @pytest.fixture
    def pcm_with_results(self):
        """Create metrics with sample results."""
        class_names = ["Normal", "DoS", "Probe", "R2L", "U2R"]
        pcm = PerClassMetrics(class_names)

        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4] * 10)  # 100 samples
        y_pred = np.array([0, 0, 1, 0, 2, 2, 3, 3, 4, 3] * 10)

        result = pcm.compute(y_true, y_pred)
        return pcm, result

    def test_get_summary(self, pcm_with_results):
        """Test summary generation."""
        pcm, result = pcm_with_results
        summary = pcm.get_summary(result)

        assert "Macro-F1" in summary
        assert "Weighted-F1" in summary
        assert "Violations" in summary

    def test_print_report_no_cm(self, pcm_with_results, capsys):
        """Test report printing without confusion matrix."""
        pcm, result = pcm_with_results
        pcm.print_report(result, show_cm=False)

        captured = capsys.readouterr()
        assert "HELIX-IDS PER-CLASS METRICS REPORT" in captured.out
        assert "Per-Class Performance" in captured.out
        assert "Normal" in captured.out
        assert "DoS" in captured.out

    def test_print_report_with_cm(self, pcm_with_results, capsys):
        """Test report printing with confusion matrix."""
        pcm, result = pcm_with_results
        pcm.print_report(result, show_cm=True)

        captured = capsys.readouterr()
        assert "Confusion Matrix" in captured.out


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_predictions(self):
        """Test with empty arrays."""
        pcm = PerClassMetrics(["Normal", "DoS"])
        y_true = np.array([])
        y_pred = np.array([])

        # Should not raise, but handle gracefully
        result = pcm.compute(y_true, y_pred)
        assert result is not None

    def test_single_class(self):
        """Test with only one class present."""
        pcm = PerClassMetrics(["Normal", "DoS", "Probe"])
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])

        result = pcm.compute(y_true, y_pred)

        # Class 0 should have perfect score
        assert result.per_class["Normal"].f1 == pytest.approx(1.0)
        # Other classes should have 0 support
        assert result.per_class["DoS"].support == 0

    def test_list_input_conversion(self):
        """Test that list inputs are converted to numpy arrays."""
        pcm = PerClassMetrics(["Normal", "DoS"])
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]

        result = pcm.compute(y_true, y_pred)

        assert result.macro_f1 == pytest.approx(1.0)
