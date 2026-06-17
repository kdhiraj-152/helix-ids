"""losses: loss-function registry extracted from HelixFullTrainer.

Phase 14 consolidation — LossRegistry is now the single authoritative
loss dispatcher (instance-based). Individual loss functions live in
categorized sibling modules:

    contrastive_losses.py   — SupCon, repulsion, centroid, variance losses
    energy_losses.py        — energy gap, balance, min-winner, EMA state
    regularization_losses.py— entropy warmup/floor, KL-uniform, logit floor,
                              tail focal/CE regularizations

Public API:
    LossRegistry              — single authoritative loss dispatcher

Dependency rules:
    losses -> torch (allowed)
    losses -> numpy (allowed)
    losses -> trainer internals (forbidden)
    losses -> diagnostics (forbidden)
    losses -> scheduler (forbidden)
    losses -> governance (forbidden)
"""

from __future__ import annotations

from scripts.training.losses.loss_registry import LossRegistry

__all__ = [
    "LossRegistry",
]
