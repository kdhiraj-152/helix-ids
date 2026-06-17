"""Extracted evaluation pipeline for HelixIDS-Full (Phase 16).

Full evaluation ownership: inference-time evaluation, validation runs,
and dataset-level reporting. All methods are FULLY delegated from
HelixFullTrainer — zero evaluation decision logic in the trainer.

Components:
  - HelixFullEvaluator: core inference-time evaluation methods
  - EvaluationOrchestrator: higher-level orchestration wrapping the evaluator

Usage:
    orchestrator = EvaluationOrchestrator(
        model=model,
        device=device,
        loss_fn=loss_fn,
        ...
    )
    val_metrics = orchestrator.validate(val_loaders)
    test_metrics = orchestrator.evaluate_per_dataset(test_loaders)
"""

from __future__ import annotations

from scripts.training.evaluation.evaluation_orchestrator import EvaluationOrchestrator
from scripts.training.evaluation.evaluator import HelixFullEvaluator

__all__ = [
    "EvaluationOrchestrator",
    "HelixFullEvaluator",
]
