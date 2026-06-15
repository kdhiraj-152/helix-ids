"""Extracted evaluation pipeline for HelixIDS-Full (Phase 12B-6).

Replaces the evaluation methods formerly embedded in HelixFullTrainer:
  - _evaluate_loader
  - _process_test_batch
  - _evaluate_test_loader
  - evaluate_per_dataset
  - validate
  - _apply_eval_class4_logit_shift
  - _apply_inference_prediction_floor
  - _compute_f1_stats_from_confusion

Usage:
    evaluator = HelixFullEvaluator(
        model=model,
        device=device,
        loss_fn=loss_fn,
        binary_class_weights=binary_weights,
        family_class_weights=family_weights,
        logger=logger,
        family_log_prior=family_log_prior,
        use_energy_based_family_objective=True,
        active_family_class_ids=active_ids,
        class4_logit_shift=0.0,
    )
    val_metrics = evaluator.validate(val_loaders)
    test_metrics = evaluator.evaluate_per_dataset(test_loaders)
"""

from __future__ import annotations

from scripts.training.evaluation.evaluator import HelixFullEvaluator

__all__ = [
    "HelixFullEvaluator",
]
