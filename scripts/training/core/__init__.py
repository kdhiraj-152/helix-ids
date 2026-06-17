"""
core: Trainer core package for HelixFullTrainer composition root.

Phase 18 extraction from HelixFullTrainer.

Public API:
    TrainerState       — canonical state holder for init-time configuration
    RecoveryManager    — structure recovery and phase settings configuration
    TrainerFacade      — thin composition root wrapping all delegates

Dependency rules:
    core → scheduler (allowed)
    core → diagnostics (allowed)
    core → execution (allowed)
    core → evaluation (allowed)
    core → validation (allowed)
    core → losses (allowed)
    core → representation (allowed)
    core → trainer internals (forbidden)
"""

from scripts.training.core.recovery_manager import RecoveryManager
from scripts.training.core.trainer_facade import TrainerFacade
from scripts.training.core.trainer_factory import TrainerFactory
from scripts.training.core.trainer_state import TrainerState

__all__ = [
    "TrainerFactory",
    "TrainerState",
    "RecoveryManager",
    "TrainerFacade",
]
