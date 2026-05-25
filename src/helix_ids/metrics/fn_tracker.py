"""False Negative Rate Tracker for HELIX-IDS

Monitors false negative rates per attack class with configurable thresholds,
early stopping, and visualization capabilities.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.metrics import confusion_matrix


@dataclass
class FNThresholds:
    """False negative rate thresholds per attack class."""

    normal: float = 0.01  # Normal: 1% FN threshold
    dos: float = 0.02  # DoS (standard): 2% FN threshold
    probe: float = 0.02  # Probe (standard): 2% FN threshold
    r2l: float = 0.05  # R2L (critical): 5% FN threshold
    u2r: float = 0.05  # U2R (critical): 5% FN threshold

    def get_threshold(self, class_name: str) -> float:
        """Get threshold for a specific class."""
        thresholds = {
            "Normal": self.normal,
            "DoS": self.dos,
            "Probe": self.probe,
            "R2L": self.r2l,
            "U2R": self.u2r,
        }
        return thresholds.get(class_name, 0.02)

    def is_critical_class(self, class_name: str) -> bool:
        """Check if a class is critical (R2L, U2R)."""
        return class_name in ["R2L", "U2R"]


@dataclass
class FNTrackerState:
    """State snapshot for a single epoch."""

    epoch: int
    fn_rates: dict[str, float] = field(default_factory=dict)
    fn_counts: dict[str, int] = field(default_factory=dict)
    class_totals: dict[str, int] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    critical_alert: bool = False


class FalseNegativeTracker:
    """
    Track false negative rates per attack class with alerting and visualization.

    False negative rate is the fraction of positive instances incorrectly
    classified as negative. For IDS, this means actual attacks not detected.

    Thresholds:
    - Critical attacks (R2L, U2R): <5% FN rate
    - Standard attacks (DoS, Probe): <2% FN rate
    - Normal traffic: <1% FN rate (false alarms)
    """

    def __init__(
        self,
        class_names: Optional[list[str]] = None,
        thresholds: Optional[FNThresholds] = None,
    ):
        """
        Initialize FalseNegativeTracker.

        Args:
            class_names: List of attack class names
            thresholds: Custom FN rate thresholds
        """
        self.class_names = class_names or ["Normal", "DoS", "Probe", "R2L", "U2R"]
        self.thresholds = thresholds or FNThresholds()

        # Current epoch state
        self.fn_counts: dict[str, int] = dict.fromkeys(self.class_names, 0)
        self.class_totals: dict[str, int] = dict.fromkeys(self.class_names, 0)
        self.fn_rates: dict[str, float] = dict.fromkeys(self.class_names, 0.0)

        # Historical tracking (per epoch)
        self.history: list[FNTrackerState] = []
        self.current_epoch = 0

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Update FN counts based on predictions.

        Args:
            y_true: True labels (class indices or names)
            y_pred: Predicted labels (class indices or names)
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Handle string labels (convert to indices)
        if y_true.dtype.kind in ("U", "O", "S"):  # Unicode, Object, or Bytes
            y_true_idx = np.array([self.class_names.index(str(y)) for y in y_true])
            y_pred_idx = np.array([self.class_names.index(str(y)) for y in y_pred])
        else:
            y_true_idx = y_true.astype(int)
            y_pred_idx = y_pred.astype(int)

        # Compute confusion matrix
        cm = confusion_matrix(y_true_idx, y_pred_idx, labels=range(len(self.class_names)))

        # Reset current counts
        self.fn_counts = dict.fromkeys(self.class_names, 0)
        self.class_totals = dict.fromkeys(self.class_names, 0)

        # Extract FN counts from confusion matrix
        # FN = actual instances of class i that were NOT predicted as class i
        for i, cls in enumerate(self.class_names):
            actual_count = cm[i].sum()  # Total instances of class i
            tp_count = cm[i, i]  # True positives for class i
            fn_count = actual_count - tp_count  # False negatives

            self.class_totals[cls] = actual_count
            self.fn_counts[cls] = fn_count

            # Calculate FN rate
            if actual_count > 0:
                self.fn_rates[cls] = fn_count / actual_count
            else:
                self.fn_rates[cls] = 0.0

    def get_fn_rates(self) -> dict[str, float]:
        """
        Get current false negative rates per class.

        Returns:
            Dictionary mapping class names to FN rates
        """
        return self.fn_rates.copy()

    def get_fn_counts(self) -> dict[str, int]:
        """
        Get current false negative counts per class.

        Returns:
            Dictionary mapping class names to FN counts
        """
        return self.fn_counts.copy()

    def get_class_totals(self) -> dict[str, int]:
        """
        Get total instances per class in current epoch.

        Returns:
            Dictionary mapping class names to total instance counts
        """
        return self.class_totals.copy()

    def check_thresholds(self) -> list[str]:
        """
        Check if any FN rate violates its threshold.

        Returns:
            List of violation messages
        """
        violations = []

        for cls, fn_rate in self.fn_rates.items():
            threshold = self.thresholds.get_threshold(cls)
            if fn_rate > threshold:
                violation = (
                    f"{cls}: FN rate {fn_rate:.4f} ({fn_rate * 100:.2f}%) "
                    f"exceeds threshold {threshold:.4f} ({threshold * 100:.2f}%)"
                )
                violations.append(violation)

        return violations

    def alert_critical(self) -> bool:
        """
        Check if any critical attack class (R2L, U2R) exceeds FN threshold.

        Returns:
            True if critical threshold violated
        """
        for cls in ["R2L", "U2R"]:
            fn_rate = self.fn_rates.get(cls, 0.0)
            threshold = self.thresholds.get_threshold(cls)
            if fn_rate > threshold:
                return True
        return False

    def new_epoch(self, epoch: int) -> None:
        """
        Mark the start of a new epoch and save current state.

        Args:
            epoch: Epoch number
        """
        # Save current state to history with the CURRENT epoch (before incrementing)
        state = FNTrackerState(
            epoch=epoch,
            fn_rates=self.fn_rates.copy(),
            fn_counts=self.fn_counts.copy(),
            class_totals=self.class_totals.copy(),
            violations=self.check_thresholds(),
            critical_alert=self.alert_critical(),
        )
        self.history.append(state)

        # Update current epoch
        self.current_epoch = epoch + 1
        self.fn_counts = dict.fromkeys(self.class_names, 0)
        self.class_totals = dict.fromkeys(self.class_names, 0)
        self.fn_rates = dict.fromkeys(self.class_names, 0.0)

    def get_epoch_history(self) -> list[FNTrackerState]:
        """
        Get historical FN tracking data per epoch.

        Returns:
            List of FNTrackerState objects
        """
        return self.history.copy()

    def get_summary_stats(self) -> dict:
        """
        Get summary statistics across all epochs.

        Returns:
            Dictionary with min/max/mean FN rates per class
        """
        if not self.history:
            return {}

        summary = {}

        for cls in self.class_names:
            fn_rates_for_cls = [state.fn_rates.get(cls, 0.0) for state in self.history]

            summary[cls] = {
                "min": min(fn_rates_for_cls),
                "max": max(fn_rates_for_cls),
                "mean": np.mean(fn_rates_for_cls),
                "std": np.std(fn_rates_for_cls),
            }

        return summary

    def should_stop_early(self, patience: int = 3, degradation_threshold: float = 0.1) -> bool:  # NOSONAR
        """
        Determine if training should stop early based on FN rate degradation.

        Early stopping triggers if:
        1. Critical attack FN rate increases by >10% for 3 consecutive epochs
        2. Critical alert raised (FN rate > threshold) for 3 consecutive epochs

        Args:
            patience: Number of bad epochs before stopping
            degradation_threshold: Relative degradation allowed (e.g., 0.1 = 10%)

        Returns:
            True if early stopping should be triggered
        """
        if len(self.history) < patience:
            return False

        recent_states = self.history[-patience:]

        # Check for consecutive critical alerts
        critical_alerts = sum(1 for state in recent_states if state.critical_alert)
        if critical_alerts >= patience:
            return True

        # Check for degradation in R2L/U2R FN rates
        for cls in ["R2L", "U2R"]:
            fn_rates = [state.fn_rates.get(cls, 0.0) for state in recent_states]

            # Check if rates are increasing consistently (ignoring 0 rates)
            valid_rates = [r for r in fn_rates if r > 0]
            if len(valid_rates) >= patience - 1:
                # Check if latest rates are increasing
                if len(valid_rates) >= 2:
                    increasing_count = sum(
                        1
                        for i in range(len(valid_rates) - 1)
                        if valid_rates[i + 1] > valid_rates[i]
                    )
                    if increasing_count >= patience - 2:
                        # Check if degradation is significant
                        if valid_rates[-1] > valid_rates[0] * (1 + degradation_threshold):
                            return True

        return False

    def plot_fn_over_time(self, output_path: Optional[str] = None) -> Optional[object]:
        """
        Plot FN rate over epochs for all classes.

        Args:
            output_path: Path to save plot (optional)

        Returns:
            Matplotlib figure object or None if history is empty
        """
        if not self.history:
            print("No epoch history to plot")
            return None

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed, skipping plot")
            return None

        fig, ax = plt.subplots(figsize=(12, 6))

        epochs = [state.epoch for state in self.history]

        # Plot FN rate for each class
        for cls in self.class_names:
            fn_rates = [state.fn_rates.get(cls, 0.0) for state in self.history]
            ax.plot(epochs, fn_rates, marker="o", label=cls, linewidth=2)

        # Add threshold lines
        for cls in self.class_names:
            threshold = self.thresholds.get_threshold(cls)
            ax.axhline(
                y=threshold,
                linestyle="--",
                alpha=0.5,
                label=f"{cls} Threshold ({threshold * 100:.1f}%)",
            )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("False Negative Rate", fontsize=12)
        ax.set_title("False Negative Rate Over Training Epochs", fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim((0.0, max(0.15, max(self.fn_rates.values()) * 1.2)))

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {output_path}")

        return fig

    def plot_fn_heatmap(self, output_path: Optional[str] = None) -> Optional[object]:
        """
        Plot FN rate heatmap per class over epochs.

        Args:
            output_path: Path to save heatmap (optional)

        Returns:
            Matplotlib figure object or None if history is empty
        """
        if not self.history:
            print("No epoch history to plot")
            return None

        try:
            import matplotlib.pyplot as plt
            import importlib
            sns = importlib.import_module("seaborn")
        except ImportError:
            print("matplotlib or seaborn not installed, skipping heatmap")
            return None

        # Prepare data for heatmap
        fn_data = []
        for cls in self.class_names:
            fn_rates = [state.fn_rates.get(cls, 0.0) for state in self.history]
            fn_data.append(fn_rates)

        epochs = [str(state.epoch) for state in self.history]

        fig, ax = plt.subplots(figsize=(14, 5))

        # Create heatmap
        sns.heatmap(
            fn_data,
            xticklabels=epochs,
            yticklabels=self.class_names,
            cmap="RdYlGn_r",
            annot=True,
            fmt=".3f",
            cbar_kws={"label": "False Negative Rate"},
            ax=ax,
            vmin=0,
            vmax=0.15,
        )

        ax.set_title(
            "False Negative Rate Heatmap (Per Class, Per Epoch)", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Attack Class", fontsize=12)

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Heatmap saved to {output_path}")

        return fig

    def to_dict(self) -> dict:
        """
        Serialize current state to dictionary.

        Returns:
            Dictionary representation of tracker state
        """
        return {
            "current_epoch": self.current_epoch,
            "fn_rates": self.fn_rates.copy(),
            "fn_counts": self.fn_counts.copy(),
            "class_totals": self.class_totals.copy(),
            "violations": self.check_thresholds(),
            "critical_alert": self.alert_critical(),
            "history_size": len(self.history),
            "summary_stats": self.get_summary_stats(),
        }

    def print_report(self) -> None:
        """Print current false negative tracking report."""
        print("\n" + "=" * 70)
        print(f"FALSE NEGATIVE RATE REPORT (Epoch {self.current_epoch})")
        print("=" * 70)

        print("\nFalse Negative Rates:")
        print("-" * 70)
        print(f"{'Class':<15} {'FN Rate':>12} {'Threshold':>12} {'Count/Total':>15} {'Status':>15}")
        print("-" * 70)

        for cls in self.class_names:
            fn_rate = self.fn_rates.get(cls, 0.0)
            threshold = self.thresholds.get_threshold(cls)
            fn_count = self.fn_counts.get(cls, 0)
            total = self.class_totals.get(cls, 0)
            status = "✓ PASS" if fn_rate <= threshold else "✗ FAIL"

            count_str = f"{fn_count}/{total}" if total > 0 else "N/A"

            print(f"{cls:<15} {fn_rate:>11.4f} {threshold:>12.4f} {count_str:>15} {status:>15}")

        print("-" * 70)

        violations = self.check_thresholds()
        if violations:
            print("\n⚠️  VIOLATIONS:")
            for violation in violations:
                print(f"  - {violation}")
        else:
            print("\n✓ All thresholds passed")

        if self.alert_critical():
            print("\n🚨 CRITICAL ALERT: R2L or U2R FN rate exceeded threshold!")

        print("=" * 70)
