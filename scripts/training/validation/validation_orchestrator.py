"""Validation orchestrator for HelixIDS-Full training.

Phase 16 extraction. Encapsulates all validation-time decision logic formerly
embedded in HelixFullTrainer: coverage validation, integrity checks, collapse
detection, and metric post-processing.

All methods are stateless or accept explicit state, enabling unit testing
without a trainer instance.
"""

from __future__ import annotations

import logging

import torch


class ValidationOrchestrator:
    """Validates model predictions, coverage, and integrity constraints.

    Stateless orchestrator — all inputs are passed explicitly, all outputs
    are returned. No trainer state mutation inside this class.
    """

    @staticmethod
    def check_family_class_coverage(
        y_family: torch.Tensor,
        *,
        active_family_class_ids: set[int] | None = None,
        enforce_all_classes_per_batch: bool = False,
    ) -> None:
        """Validate that batch contains all expected family classes.

        Equivalent to HelixFullTrainer._check_family_class_coverage.
        Raises RuntimeError on violation.
        """
        if not active_family_class_ids:
            return
        batch_class_ids = {int(v) for v in torch.unique(y_family, dim=0).tolist()}
        if not enforce_all_classes_per_batch:
            if len(batch_class_ids) <= 1:
                raise RuntimeError(
                    "Hard-stop integrity guard triggered: batch_single_family_class"
                )
            return
        if not active_family_class_ids.issubset(batch_class_ids):
            missing = sorted(active_family_class_ids - batch_class_ids)
            raise RuntimeError(
                "Hard-stop integrity guard triggered: "
                f"batch_missing_family_classes_{missing}"
            )

    @staticmethod
    def check_step_coverage(
        batch_idx: int,
        family_pred_counts: torch.Tensor | None,
        *,
        step_coverage_checked: bool,
        representation_diagnostic_mode: bool,
        head_phase_start_step: int,
        representation_phase_active: bool,
        global_step: int,
        coverage_check_after_head_steps: int,
        step_coverage_check_step: int,
        active_family_class_ids: set[int] | None,
        disable_integrity_hard_stops: bool = False,
        use_energy_based_family_objective: bool = False,
        logger: logging.Logger | None = None,
    ) -> tuple[bool, list[int] | None]:
        """Check step coverage for family class predictions.

        Returns (step_coverage_checked, missing_classes) where
        missing_classes is None if the check did not run.

        Equivalent to HelixFullTrainer._check_step_coverage.
        """
        log = logger or logging.getLogger(__name__)

        if (
            (not step_coverage_checked)
            and (
                (
                    representation_diagnostic_mode
                    and head_phase_start_step >= 0
                    and (not representation_phase_active)
                    and (global_step - head_phase_start_step)
                    >= coverage_check_after_head_steps
                )
                or (
                    (not representation_diagnostic_mode)
                    and batch_idx >= step_coverage_check_step
                )
            )
            and family_pred_counts is not None
            and active_family_class_ids
        ):
            missing_classes = [
                cls
                for cls in sorted(active_family_class_ids)
                if int(family_pred_counts[cls].item()) <= 0
            ]
            step_coverage_checked_new = True

            if missing_classes:
                if disable_integrity_hard_stops:
                    log.warning(
                        "Integrity hard-stops disabled: allowing "
                        "step_coverage_missing_predictions "
                        "(missing_classes=%s by_step=%d)",
                        missing_classes,
                        int(coverage_check_after_head_steps),
                    )
                    return step_coverage_checked_new, missing_classes
                if use_energy_based_family_objective:
                    log.warning(
                        "Energy mode: skipping step_coverage hard-stop "
                        "(missing_classes=%s by_step=%d)",
                        missing_classes,
                        int(coverage_check_after_head_steps),
                    )
                    return step_coverage_checked_new, missing_classes
                raise RuntimeError(
                    "Hard-stop integrity guard triggered: "
                    "step_coverage_missing_predictions_by_step"
                    f"{coverage_check_after_head_steps}_classes_{missing_classes}"
                )

            return step_coverage_checked_new, None

        return step_coverage_checked, None

    @staticmethod
    def post_training_macro_floor(epochs: int) -> float:
        """Return macro-F1 floor calibrated to training budget.

        Short smoke runs use a more permissive floor.
        Equivalent to HelixFullTrainer._post_training_macro_floor.
        """
        if epochs <= 2:
            return 0.10
        if epochs <= 10:
            return 0.15
        return 0.25

    @staticmethod
    def detect_coverage_collapse(
        val_metrics: dict[str, float],
        train_family_class_count: int,
    ) -> bool:
        """Detect insufficient class coverage in validation metrics.

        Returns True when predicted class count < 50% of total classes.
        Equivalent to the collapse-detection in HelixFullTrainer.fit().
        """
        predicted_class_count = int(
            val_metrics.get("val_family_predicted_class_count", 0.0)
        )
        collapse_threshold = max(1, int(train_family_class_count * 0.5))
        if train_family_class_count > 0 and predicted_class_count < collapse_threshold:
            return True
        return False

    @staticmethod
    def check_zero_prediction_classes(
        val_metrics: dict[str, float],
        *,
        disable_integrity_hard_stops: bool = False,
        use_energy_based_family_objective: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        """Check for zero-prediction classes and raise / warn accordingly.

        Raises RuntimeError when hard-stops are enabled and not in energy mode.
        Equivalent to the validation integrity check in HelixFullTrainer.fit().
        """
        log = logger or logging.getLogger(__name__)
        zero_prediction_classes = int(
            val_metrics.get("val_family_zero_prediction_classes", 0.0)
        )
        if zero_prediction_classes > 0:
            if disable_integrity_hard_stops:
                log.warning(
                    "Integrity hard-stops disabled: allowing "
                    "validation_zero_prediction_classes_nonzero "
                    "(missing_classes=%d)",
                    zero_prediction_classes,
                )
            elif use_energy_based_family_objective:
                log.warning(
                    "Energy mode: skipping validation_zero_prediction_classes "
                    "hard-stop (missing_classes=%d)",
                    zero_prediction_classes,
                )
            else:
                raise RuntimeError(
                    "Hard-stop integrity guard triggered: "
                    "validation_zero_prediction_classes_nonzero"
                )

    @staticmethod
    def check_per_dataset_macro_floor(
        per_dataset_results: dict[str, dict[str, float]],
        macro_floor: float,
    ) -> None:
        """Check that all per-dataset macro F1 values meet the minimum floor.

        Raises RuntimeError when any dataset's macro-F1 is below the floor.
        Equivalent to the post-training guard in HelixFullTrainer.fit().
        """
        macro_values = [
            float(metrics.get("family_macro_f1", metrics.get("family_f1", 0.0)))
            for metrics in per_dataset_results.values()
        ]
        if macro_values and min(macro_values) < macro_floor:
            raise RuntimeError(
                "Post-training macro_f1 guard failed: "
                f"min_family_macro_f1={min(macro_values):.4f} < {macro_floor:.2f}"
            )

    @staticmethod
    def log_per_dataset_results(
        logger: logging.Logger,
        per_dataset_results: dict[str, dict[str, float]],
    ) -> None:
        """Log formatted per-dataset metrics.

        Equivalent to HelixFullTrainer._log_per_dataset_results.
        """
        for dataset_name, metrics in per_dataset_results.items():
            logger.info(f"\n{dataset_name}:")
            for key, val in metrics.items():
                logger.info(f"  {key}: {val:.4f}")
