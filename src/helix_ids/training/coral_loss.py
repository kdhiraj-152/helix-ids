"""CORAL loss re-export from models/adaptation.

This file makes CORAL loss importable from the training package
while keeping the canonical implementation in models/adaptation/.
"""

from helix_ids.models.adaptation.coral_loss import (
    CORALLoss,
    CombinedAlignmentLoss,
    DeepCORALLoss,
    compute_coral,
    compute_covariance,
)

__all__ = [
    "CORALLoss",
    "DeepCORALLoss",
    "CombinedAlignmentLoss",
    "compute_coral",
    "compute_covariance",
]
