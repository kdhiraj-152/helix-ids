"""
Per-Class Metrics Tracking for HELIX-IDS

Provides comprehensive per-class evaluation metrics including:
- Precision, recall, F1 score, and support per class
- ROC-AUC scores (if probability predictions available)
- Macro and weighted F1 averages
- Confusion matrix analysis
- Threshold-based alerting for quality violations
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

# Target F1 thresholds per class (from research)
DEFAULT_THRESHOLDS = {
    "Normal": 0.98,
    "DoS": 0.95,
    "Probe": 0.90,
    "R2L": 0.80,
    "U2R": 0.60,
}


@dataclass
class ClassMetrics:
    """Metrics for a single class."""

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    support: int = 0
    auc_roc: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
            "support": int(self.support),
        }
        if self.auc_roc is not None:
            result["auc_roc"] = float(self.auc_roc)
        return result


@dataclass
class PerClassMetricsResult:
    """Complete per-class metrics result."""

    per_class: dict[str, ClassMetrics] = field(default_factory=dict)
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "per_class": {cls: metrics.to_dict() for cls, metrics in self.per_class.items()},
            "macro_f1": float(self.macro_f1),
            "weighted_f1": float(self.weighted_f1),
            "violations": self.violations,
        }


class PerClassMetrics:
    """Per-class metrics computation and analysis."""

    def __init__(
        self,
        class_names: list[str],
        thresholds: Optional[dict[str, float]] = None,
    ):
        """
        Initialize per-class metrics tracker.

        Args:
            class_names: List of class names (e.g., ["Normal", "DoS", "Probe", "R2L", "U2R"])
            thresholds: Optional dict of class -> target F1 threshold.
                       Defaults to DEFAULT_THRESHOLDS if not provided.
        """
        self.class_names = class_names
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.num_classes = len(class_names)

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> PerClassMetricsResult:
        """
        Compute comprehensive per-class metrics.

        Args:
            y_true: True labels (numpy array or list)
            y_pred: Predicted labels (numpy array or list)
            y_proba: Probability predictions (optional, for AUC-ROC computation).
                    Shape: (n_samples, n_classes)

        Returns:
            PerClassMetricsResult with per-class metrics, macro/weighted F1, and violations
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        result = PerClassMetricsResult()

        # Handle empty input
        if len(y_true) == 0:
            for class_name in self.class_names:
                result.per_class[class_name] = ClassMetrics()
            result.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
            return result

        # Compute per-class metrics using labels parameter to ensure all classes are included
        labels = list(range(self.num_classes))
        precisions = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
        recalls = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
        f1_scores = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

        # Compute support (count of samples per class)
        unique, counts = np.unique(y_true, return_counts=True)
        support_map = dict(zip(unique, counts))

        for idx, class_name in enumerate(self.class_names):
            metrics = ClassMetrics(
                precision=float(precisions[idx]),
                recall=float(recalls[idx]),
                f1=float(f1_scores[idx]),
                support=int(support_map.get(idx, 0)),
            )

            # Compute AUC-ROC if probability predictions available
            if y_proba is not None and y_proba.shape[1] >= idx + 1:
                try:
                    # One-vs-rest binary classification for this class
                    y_binary = (y_true == idx).astype(int)
                    fpr, tpr, _ = roc_curve(y_binary, y_proba[:, idx])
                    metrics.auc_roc = float(auc(fpr, tpr))
                except Exception:
                    metrics.auc_roc = None

            result.per_class[class_name] = metrics

        # Compute macro and weighted F1
        result.macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        result.weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        # Confusion matrix with all classes
        result.confusion_matrix = confusion_matrix(y_true, y_pred, labels=labels)

        # Check threshold violations
        result.violations = self._check_violations(result.per_class)

        return result

    def _check_violations(self, per_class: dict[str, ClassMetrics]) -> list[str]:
        """
        Check for threshold violations.

        Args:
            per_class: Dictionary of class -> ClassMetrics

        Returns:
            List of violation messages
        """
        violations = []

        for class_name in self.class_names:
            if class_name not in per_class:
                continue

            metrics = per_class[class_name]
            threshold = self.thresholds.get(class_name, 0.50)

            if metrics.f1 < threshold:
                violations.append(
                    f"⚠️  {class_name}: F1 {metrics.f1:.4f} below threshold ({threshold:.4f})"
                )

        return violations

    def _get_status_marker(self, f1: float, threshold: float) -> str:
        """Get status marker based on F1 score and threshold."""
        if f1 < threshold:
            return " ← BELOW THRESHOLD"
        elif f1 < 0.60:
            return " ← POOR"
        elif f1 < 0.80:
            return " ← FAIR"
        elif f1 < 0.95:
            return " ← GOOD"
        else:
            return " ← EXCELLENT"

    def _print_class_metrics_table(self, result: PerClassMetricsResult) -> None:
        """Print the per-class performance table."""
        print("\nPer-Class Performance:")
        print("-" * 90)
        print(
            f"{'Class':<15} {'Precision':>12} {'Recall':>12} {'F1':>12} "
            f"{'Support':>10} {'AUC-ROC':>12}"
        )
        print("-" * 90)

        for class_name in self.class_names:
            if class_name not in result.per_class:
                continue

            metrics = result.per_class[class_name]
            threshold = self.thresholds.get(class_name, 0.50)
            auc_str = f"{metrics.auc_roc:.4f}" if metrics.auc_roc is not None else "N/A"
            status_marker = self._get_status_marker(metrics.f1, threshold)

            print(
                f"{class_name:<15} {metrics.precision:>12.4f} {metrics.recall:>12.4f} "
                f"{metrics.f1:>12.4f} {metrics.support:>10} {auc_str:>12}{status_marker}"
            )

    def _print_thresholds(self, result: PerClassMetricsResult) -> None:
        """Print target F1 thresholds."""
        print("\nTarget F1 Thresholds:")
        print("-" * 90)
        for class_name in self.class_names:
            threshold = self.thresholds.get(class_name, 0.50)
            actual = result.per_class[class_name].f1 if class_name in result.per_class else 0.0
            status = "✓ PASS" if actual >= threshold else "✗ FAIL"
            print(f"  {class_name:<15}: {threshold:.2f} (actual: {actual:.4f}) {status}")

    def _print_violations(self, result: PerClassMetricsResult) -> None:
        """Print violations if any."""
        if result.violations:
            print("\n⚠️  ALERTS:")
            print("-" * 90)
            for violation in result.violations:
                print(f"  {violation}")

    def _print_confusion_matrix(self, result: PerClassMetricsResult) -> None:
        """Print confusion matrix."""
        if result.confusion_matrix is None:
            return
        print("\nConfusion Matrix:")
        print("-" * 90)
        cm = result.confusion_matrix
        header = "Predicted →"
        for _, name in enumerate(self.class_names):
            header += f" {name[:6]:>8}"
        print("  " + header)

        for i, true_label in enumerate(self.class_names):
            row = f"  {true_label[:6]:<6}"
            for j in range(len(self.class_names)):
                row += f" {cm[i, j]:>8}"
            print(row)

    def print_report(
        self,
        result: PerClassMetricsResult,
        show_cm: bool = False,
    ) -> None:
        """
        Print formatted per-class metrics report.

        Args:
            result: PerClassMetricsResult from compute()
            show_cm: Whether to show confusion matrix (default: False)
        """
        print("\n" + "=" * 90)
        print("HELIX-IDS PER-CLASS METRICS REPORT")
        print("=" * 90)

        # Overall metrics
        print("\nOverall Metrics:")
        print(f"  Macro-F1:   {result.macro_f1:.4f}")
        print(f"  Weighted-F1: {result.weighted_f1:.4f}")

        # Print sections
        self._print_class_metrics_table(result)
        self._print_thresholds(result)
        self._print_violations(result)

        if show_cm:
            self._print_confusion_matrix(result)

        print("=" * 90)


    def get_summary(self, result: PerClassMetricsResult) -> str:
        """
        Get a concise single-line summary.

        Args:
            result: PerClassMetricsResult from compute()

        Returns:
            Summary string
        """
        violation_count = len(result.violations)
        return (
            f"Macro-F1: {result.macro_f1:.4f}, Weighted-F1: {result.weighted_f1:.4f}, "
            f"Violations: {violation_count}"
        )
