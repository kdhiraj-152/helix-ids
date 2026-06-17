"""early_stopping: Early-stopping and hard-stop detection.

Phase 13A-2 extraction from HelixFullTrainer.

EarlyStoppingManager provides:
    - Hard-stop reason detection (val-gap collapse, high-accuracy-high-loss, entropy collapse)
    - Early stopping update (patience-based stop decision)
    - Smoke-mode detection

No trainer-internal imports — pure computation with config injection.
"""

from __future__ import annotations

from typing import Any


class EarlyStoppingManager:
    """Manages early stopping state and hard-stop integrity checks.

    The manager owns all streak counters and best-loss tracking.  Side effects
    (saving best model state, logging) are handled by the trainer via the
    delegation wrapper.
    """

    def __init__(
        self,
        *,
        early_stopping_patience: int,
        early_stopping_threshold: float,
        min_family_minority_recall_for_best: float,
        disable_integrity_hard_stops: bool = False,
        smoke_mode_entropy_threshold: float = 0.10,
        full_mode_entropy_threshold: float = 0.12,
        smoke_mode_entropy_streak: int = 3,
        full_mode_entropy_streak: int = 2,
        smoke_mode_min_epoch: int = 4,
        full_mode_min_epoch: int = 2,
        val_gap_threshold: float = 0.12,
        val_gap_streak: int = 2,
        high_loss_threshold: float = 0.5,
        high_accuracy_threshold: float = 0.95,
        high_loss_streak: int = 2,
        high_loss_min_epoch: int = 1,
        critical_entropy_threshold: float = 0.08,
        critical_entropy_streak: int = 3,
        quality_gate_minority_recall: float | None = None,
        quality_gate_entropy: float = 0.3,
    ) -> None:
        self._patience = int(early_stopping_patience)
        self._threshold = float(early_stopping_threshold)
        self._min_minority_recall = float(min_family_minority_recall_for_best)
        self._disable_hard_stops = bool(disable_integrity_hard_stops)

        # Hard-stop config
        self._smoke_entropy_threshold = float(smoke_mode_entropy_threshold)
        self._full_entropy_threshold = float(full_mode_entropy_threshold)
        self._smoke_entropy_streak = int(smoke_mode_entropy_streak)
        self._full_entropy_streak = int(full_mode_entropy_streak)
        self._smoke_min_epoch = int(smoke_mode_min_epoch)
        self._full_min_epoch = int(full_mode_min_epoch)
        self._val_gap_threshold = float(val_gap_threshold)
        self._val_gap_streak = int(val_gap_streak)
        self._val_gap_collapse_signals = {
            "max_val_macro_f1": 0.25,
            "max_val_minority_recall": 0.10,
            "max_val_entropy": 0.15,
        }
        self._high_loss_threshold = float(high_loss_threshold)
        self._high_accuracy_threshold = float(high_accuracy_threshold)
        self._high_loss_streak = int(high_loss_streak)
        self._high_loss_min_epoch = int(high_loss_min_epoch)
        self._critical_entropy_threshold = float(critical_entropy_threshold)
        self._critical_entropy_streak = int(critical_entropy_streak)

        # Quality gate config
        self._quality_gate_minority_recall = (
            float(quality_gate_minority_recall) if quality_gate_minority_recall is not None else None
        )
        self._quality_gate_entropy = float(quality_gate_entropy)

        # Runtime state
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.val_gap_collapse_streak = 0
        self.high_accuracy_high_loss_streak = 0
        self.entropy_missing_class_streak = 0
        self.entropy_collapse_streak = 0

    # ------------------------------------------------------------------ #
    # Smoke mode detection
    # ------------------------------------------------------------------ #

    @staticmethod
    def is_smoke_mode(*, epochs: int | None = None, gov_profile: str = "") -> bool:
        """Return True when trainer is running a smoke-governance profile."""
        profile = str(gov_profile).strip().lower()
        if profile == "smoke":
            return True
        if epochs is not None:
            return int(epochs) <= 10
        return False

    # ------------------------------------------------------------------ #
    # Hard-stop detection methods
    # ------------------------------------------------------------------ #

    def hard_stop_reason(
        self,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        *,
        is_smoke: bool,
        epoch: int,
    ) -> str | None:
        """Return hard-stop reason string when integrity constraints are violated, else None."""
        if self._disable_hard_stops:
            return None

        reason = self._hard_stop_val_gap_collapse(train_metrics, val_metrics)
        if reason is not None:
            return reason

        reason = self._hard_stop_high_accuracy_high_loss(
            train_metrics, val_metrics, epoch=epoch
        )
        if reason is not None:
            return reason

        return self._hard_stop_entropy_collapse(val_metrics, is_smoke=is_smoke, epoch=epoch)

    def _hard_stop_val_gap_collapse(
        self,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> str | None:
        """Detect persistent val-vs-train loss gap with collapse symptoms."""
        train_loss = train_metrics.get("train_calibrated_loss", 0.0)
        val_loss = val_metrics.get("val_calibrated_loss", 0.0)
        val_gap = train_loss - val_loss

        collapse_signals = (
            val_metrics.get("val_family_macro_f1", 1.0) < self._val_gap_collapse_signals["max_val_macro_f1"]
            or val_metrics.get("val_family_minority_recall_min", 1.0)
            < self._val_gap_collapse_signals["max_val_minority_recall"]
            or val_metrics.get("val_family_entropy", 1.0)
            < self._val_gap_collapse_signals["max_val_entropy"]
        )

        if val_gap > self._val_gap_threshold and collapse_signals:
            self.val_gap_collapse_streak += 1
            if self.val_gap_collapse_streak >= self._val_gap_streak:
                return "val_loss_below_train_loss_with_collapse"
        else:
            self.val_gap_collapse_streak = 0

        return None

    def _hard_stop_high_accuracy_high_loss(
        self,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        *,
        epoch: int,
    ) -> str | None:
        """Detect suspiciously high accuracy paired with persistently high loss."""
        peak_accuracy = max(
            train_metrics.get("train_binary_acc", 0.0),
            train_metrics.get("train_family_acc", 0.0),
            val_metrics.get("val_binary_acc", 0.0),
            val_metrics.get("val_family_acc", 0.0),
        )
        train_loss = train_metrics.get("train_calibrated_loss", 0.0)

        if train_loss > self._high_loss_threshold and peak_accuracy > self._high_accuracy_threshold:
            self.high_accuracy_high_loss_streak += 1
            if int(epoch) >= self._high_loss_min_epoch and self.high_accuracy_high_loss_streak >= self._high_loss_streak:
                return "high_accuracy_with_high_loss"
        else:
            self.high_accuracy_high_loss_streak = 0

        return None

    def _hard_stop_entropy_collapse(
        self,
        val_metrics: dict[str, float],
        *,
        is_smoke: bool,
        epoch: int,
    ) -> str | None:
        """Detect class-collapse using entropy and missing-class evidence."""
        entropy_val = val_metrics.get("val_family_entropy", 0.0)
        same_dataset_entropy_collapse = val_metrics.get("val_entropy_missing_same_dataset", 0.0) > 0

        entropy_threshold = self._smoke_entropy_threshold if is_smoke else self._full_entropy_threshold
        required_streak = self._smoke_entropy_streak if is_smoke else self._full_entropy_streak
        min_epoch = self._smoke_min_epoch if is_smoke else self._full_min_epoch

        if entropy_val < entropy_threshold and same_dataset_entropy_collapse:
            self.entropy_missing_class_streak += 1
            if int(epoch) >= min_epoch and self.entropy_missing_class_streak >= required_streak:
                return "prediction_entropy_collapse_with_missing_classes"
        else:
            self.entropy_missing_class_streak = 0

        # Very strict threshold for extreme cases
        if entropy_val < self._critical_entropy_threshold:
            self.entropy_collapse_streak += 1
            if self.entropy_collapse_streak >= self._critical_entropy_streak:
                return "prediction_entropy_critical_collapse"
            return None

        self.entropy_collapse_streak = 0
        return None

    # ------------------------------------------------------------------ #
    # Early stopping update
    # ------------------------------------------------------------------ #

    def update_early_stopping(
        self,
        val_metrics: dict[str, float],
        *,
        quality_gate_minority_recall: float,
        quality_gate_entropy: float,
    ) -> dict[str, Any]:
        """Update early stopping state and return decision.

        Returns dict:
            should_stop: bool — True when training should stop
            is_best: bool — True if this is a new best model
            best_val_loss: float — current best loss
            patience_counter: int — current patience count
        """
        val_loss = val_metrics["val_loss"]

        quality_gate_pass = (
            val_metrics.get("val_family_minority_recall_min", 0.0)
            >= quality_gate_minority_recall
            and val_metrics.get("val_family_entropy", 0.0) >= quality_gate_entropy
        )

        is_best = False
        if val_loss < self.best_val_loss - self._threshold:
            if quality_gate_pass:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                is_best = True
            # If quality gate fails, we still increment patience below

        if not is_best:
            self.patience_counter += 1

        should_stop = self.patience_counter >= self._patience

        return {
            "should_stop": should_stop,
            "is_best": is_best,
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
        }

    # ------------------------------------------------------------------ #
    # State reset
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset runtime state for fresh training (keeps config)."""
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.val_gap_collapse_streak = 0
        self.high_accuracy_high_loss_streak = 0
        self.entropy_missing_class_streak = 0
        self.entropy_collapse_streak = 0
