"""Evaluation orchestrator for HelixIDS-Full training.

Phase 16 extraction. Provides the combined evaluation workflows: validation
runs, dataset evaluation, calibration, and metric aggregation.

Coordinates the HelixFullEvaluator with higher-level orchestration patterns
used by the training loop. All evaluation ownership resides in this package
— the trainer delegates entirely through this orchestrator.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.utils.data import DataLoader

from helix_ids.models.full import HelixIDSFull, MultiTaskLoss
from scripts.training.evaluation.evaluator import HelixFullEvaluator


class EvaluationOrchestrator:
    """Orchestrates model evaluation: validation, test evaluation, calibration.

    Owns the HelixFullEvaluator lifecycle and provides higher-level
    orchestration methods used by the training loop.
    """

    def __init__(
        self,
        *,
        model: HelixIDSFull,
        device: str,
        loss_fn: MultiTaskLoss,
        binary_class_weights: torch.Tensor | None = None,
        family_class_weights: torch.Tensor | None = None,
        logger: logging.Logger | None = None,
        family_log_prior: torch.Tensor | None = None,
        use_energy_based_family_objective: bool = True,
        energy_logit_temperature: float = 2.0,
        active_family_class_ids: set[int] | None = None,
        class4_logit_shift: float = 0.0,
        class4_logit_shift_class_id: int = 4,
        disable_integrity_hard_stops: bool = False,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self._evaluator = HelixFullEvaluator(
            model=model,
            device=device,
            loss_fn=loss_fn,
            binary_class_weights=binary_class_weights,
            family_class_weights=family_class_weights,
            logger=self.logger,
            family_log_prior=family_log_prior,
            use_energy_based_family_objective=use_energy_based_family_objective,
            energy_logit_temperature=energy_logit_temperature,
            active_family_class_ids=active_family_class_ids,
            class4_logit_shift=class4_logit_shift,
            class4_logit_shift_class_id=class4_logit_shift_class_id,
            disable_integrity_hard_stops=disable_integrity_hard_stops,
        )

    # --- Full evaluation methods ---

    def validate(
        self,
        val_loaders: dict[str, DataLoader],
        *,
        active_family_class_ids: set[int] | None = None,
        class4_logit_shift: float | None = None,
        class4_logit_shift_class_id: int | None = None,
    ) -> dict[str, float]:
        """Validate per dataset with strict isolation (worst-case aggregation).

        Equivalent to HelixFullTrainer.validate.
        """
        return self._evaluator.validate(
            val_loaders,
            active_family_class_ids=active_family_class_ids,
            class4_logit_shift=class4_logit_shift,
            class4_logit_shift_class_id=class4_logit_shift_class_id,
        )

    def evaluate_per_dataset(
        self,
        test_loaders: dict[str, DataLoader],
        *,
        active_family_class_ids: set[int] | None = None,
        class4_logit_shift: float | None = None,
        class4_logit_shift_class_id: int | None = None,
    ) -> dict[str, dict[str, float]]:
        """Evaluate on per-dataset test sets.

        Equivalent to HelixFullTrainer.evaluate_per_dataset.
        """
        return self._evaluator.evaluate_per_dataset(
            test_loaders,
            active_family_class_ids=active_family_class_ids,
            class4_logit_shift=class4_logit_shift,
            class4_logit_shift_class_id=class4_logit_shift_class_id,
        )

    def evaluate_loader(
        self,
        loader: DataLoader,
        dataset_name: str = "unknown",
        *,
        active_family_class_ids: set[int] | None = None,
        class4_logit_shift: float | None = None,
        class4_logit_shift_class_id: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate metrics on a single dataset loader.

        Equivalent to HelixFullTrainer._evaluate_loader.
        """
        return self._evaluator._evaluate_loader(
            loader,
            dataset_name=dataset_name,
            active_family_class_ids=active_family_class_ids,
            class4_logit_shift=class4_logit_shift,
            class4_logit_shift_class_id=class4_logit_shift_class_id,
        )

    # --- Static helpers re-exported from the evaluator ---

    @staticmethod
    def apply_eval_class4_logit_shift(
        family_logits: torch.Tensor,
        *,
        shift: float,
        class_id: int,
    ) -> torch.Tensor:
        """Apply inference-time class N logit shift."""
        return HelixFullEvaluator._apply_eval_class4_logit_shift(
            family_logits, shift=shift, class_id=class_id,
        )

    @staticmethod
    def apply_inference_prediction_floor(
        family_logits: torch.Tensor,
        family_pred: torch.Tensor,
        *,
        active_class_ids: set[int],
    ) -> torch.Tensor:
        """Inference-only prediction floor to guarantee per-batch class presence."""
        return HelixFullEvaluator._apply_inference_prediction_floor(
            family_logits, family_pred,
            active_class_ids=active_class_ids,
        )

    @staticmethod
    def compute_f1_stats_from_confusion(confusion: torch.Tensor) -> dict[str, Any]:
        """Compute F1-related statistics from confusion matrix counts."""
        return HelixFullEvaluator._compute_f1_stats_from_confusion(confusion)

    def process_test_batch(
        self,
        x: torch.Tensor,
        y_binary: torch.Tensor,
        y_family: torch.Tensor,
        binary_confusion: torch.Tensor,
        family_confusion: torch.Tensor | None,
        *,
        active_family_class_ids: set[int] | None = None,
        class4_logit_shift: float | None = None,
        class4_logit_shift_class_id: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, int, float]:
        """Process one test batch and accumulate metrics."""
        return self._evaluator._process_test_batch(
            x, y_binary, y_family, binary_confusion, family_confusion,
            active_family_class_ids=active_family_class_ids,
            class4_logit_shift=class4_logit_shift,
            class4_logit_shift_class_id=class4_logit_shift_class_id,
        )

    def evaluate_test_loader(
        self,
        test_loader: DataLoader,
        *,
        active_family_class_ids: set[int] | None = None,
        class4_logit_shift: float | None = None,
        class4_logit_shift_class_id: int | None = None,
    ) -> dict[str, float]:
        """Evaluate one test loader with tensor-first aggregation."""
        return self._evaluator._evaluate_test_loader(
            test_loader,
            active_family_class_ids=active_family_class_ids,
            class4_logit_shift=class4_logit_shift,
            class4_logit_shift_class_id=class4_logit_shift_class_id,
        )

    # --- Combined orchestration ---

    def validate_with_logging(
        self,
        val_loaders: dict[str, DataLoader],
        *,
        active_family_class_ids: set[int] | None = None,
        class4_logit_shift: float | None = None,
        class4_logit_shift_class_id: int | None = None,
    ) -> dict[str, float]:
        """Run validation and log per-dataset results."""
        val_metrics = self.validate(
            val_loaders,
            active_family_class_ids=active_family_class_ids,
            class4_logit_shift=class4_logit_shift,
            class4_logit_shift_class_id=class4_logit_shift_class_id,
        )
        return val_metrics

    def evaluate_with_logging(
        self,
        test_loaders: dict[str, DataLoader],
        *,
        active_family_class_ids: set[int] | None = None,
        class4_logit_shift: float | None = None,
        class4_logit_shift_class_id: int | None = None,
    ) -> dict[str, dict[str, float]]:
        """Run per-dataset evaluation and return results."""
        return self.evaluate_per_dataset(
            test_loaders,
            active_family_class_ids=active_family_class_ids,
            class4_logit_shift=class4_logit_shift,
            class4_logit_shift_class_id=class4_logit_shift_class_id,
        )
