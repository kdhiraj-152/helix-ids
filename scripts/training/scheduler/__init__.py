"""scheduler: scheduling/lifecycle subsystem for HelixFullTrainer.

Phase 13A-2 extraction from HelixFullTrainer.

Public API:
    TrainingPhase         — 5-state enum (IDLE → REPRESENTATION_ONLY → HEAD_ONLY → JOINT_FINETUNE → COMPLETE)
    PhaseManager          — phase state machine, transition validation, inspection helpers
    EarlyStoppingManager  — early stopping + hard-stop integrity detection
    FreezeManager         — backbone freeze/unfreeze state management
    LRScheduler           — cosine-decay LR schedule with warmup

Dependency rules:
    scheduler → diagnostics (allowed)
    scheduler → standard library (allowed)
    scheduler → numpy (allowed)
    scheduler → torch (allowed)
    scheduler → trainer internals (forbidden)
    scheduler → governance (forbidden)
    scheduler → orchestration (forbidden)
"""

from scripts.training.scheduler.early_stopping import EarlyStoppingManager
from scripts.training.scheduler.freeze_manager import FreezeManager
from scripts.training.scheduler.lr_scheduler import LRScheduler
from scripts.training.scheduler.phase_manager import (
    PhaseManager,
    TrainingPhase,
    can_transition,
    validate_transition,
)
from scripts.training.scheduler.phase_orchestrator import PhaseOrchestrator

__all__ = [
    "TrainingPhase",
    "PhaseManager",
    "validate_transition",
    "can_transition",
    "EarlyStoppingManager",
    "FreezeManager",
    "LRScheduler",
    "PhaseOrchestrator",
]
