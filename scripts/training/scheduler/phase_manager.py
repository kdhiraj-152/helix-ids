"""phase_manager: TrainingPhase enum + PhaseManager for phase state machine.

Phase 13A-2 extraction from HelixFullTrainer.

TrainingPhase defines the 5-state lifecycle:
    IDLE → REPRESENTATION_ONLY → HEAD_ONLY → JOINT_FINETUNE → COMPLETE

PhaseManager provides:
    - Phase state storage (current_phase enum)
    - Transition evaluation and validation
    - Phase inspection helpers
    - Representation window-step logic

No behavioral changes — preserves exact runtime state-machine semantics.
"""

from __future__ import annotations

import enum


class TrainingPhase(enum.Enum):
    """Five-state lifecycle for HelixFullTrainer phase machine."""

    IDLE = "idle"
    REPRESENTATION_ONLY = "representation_only"
    HEAD_ONLY = "head_only"
    JOINT_FINETUNE = "joint_finetune"
    COMPLETE = "complete"

    def __str__(self) -> str:
        return str(self.value)

    @property
    def is_terminal(self) -> bool:
        return self in (TrainingPhase.JOINT_FINETUNE, TrainingPhase.COMPLETE)

    @property
    def is_representation(self) -> bool:
        return self == TrainingPhase.REPRESENTATION_ONLY

    @property
    def is_head_only(self) -> bool:
        return self == TrainingPhase.HEAD_ONLY

    @property
    def is_joint_finetune(self) -> bool:
        return self == TrainingPhase.JOINT_FINETUNE


# ---------------------------------------------------------------------------
# Transition validation table
# ---------------------------------------------------------------------------
# Maps (current_phase, target_phase) → allowed?
_VALID_TRANSITIONS: dict[tuple[TrainingPhase, TrainingPhase], bool] = {
    (TrainingPhase.IDLE, TrainingPhase.REPRESENTATION_ONLY): True,
    (TrainingPhase.IDLE, TrainingPhase.HEAD_ONLY): True,
    (TrainingPhase.IDLE, TrainingPhase.COMPLETE): True,
    (TrainingPhase.REPRESENTATION_ONLY, TrainingPhase.REPRESENTATION_ONLY): True,  # stay
    (TrainingPhase.REPRESENTATION_ONLY, TrainingPhase.HEAD_ONLY): True,
    (TrainingPhase.REPRESENTATION_ONLY, TrainingPhase.COMPLETE): True,
    (TrainingPhase.HEAD_ONLY, TrainingPhase.HEAD_ONLY): True,  # stay
    (TrainingPhase.HEAD_ONLY, TrainingPhase.JOINT_FINETUNE): True,
    (TrainingPhase.HEAD_ONLY, TrainingPhase.COMPLETE): True,
    (TrainingPhase.JOINT_FINETUNE, TrainingPhase.JOINT_FINETUNE): True,  # stay
    (TrainingPhase.JOINT_FINETUNE, TrainingPhase.COMPLETE): True,
    (TrainingPhase.COMPLETE, TrainingPhase.COMPLETE): True,  # stay
}

# Illegal transitions (explicitly documented)
_ILLEGAL_TRANSITIONS: dict[tuple[TrainingPhase, TrainingPhase], str] = {
    (TrainingPhase.REPRESENTATION_ONLY, TrainingPhase.JOINT_FINETUNE): "must pass through HEAD_ONLY",
    (TrainingPhase.HEAD_ONLY, TrainingPhase.REPRESENTATION_ONLY): "irreversible",
    (TrainingPhase.JOINT_FINETUNE, TrainingPhase.REPRESENTATION_ONLY): "irreversible",
    (TrainingPhase.JOINT_FINETUNE, TrainingPhase.HEAD_ONLY): "irreversible",
    (TrainingPhase.COMPLETE, TrainingPhase.REPRESENTATION_ONLY): "terminal",
    (TrainingPhase.COMPLETE, TrainingPhase.HEAD_ONLY): "terminal",
    (TrainingPhase.COMPLETE, TrainingPhase.JOINT_FINETUNE): "terminal",
}


def validate_transition(current: TrainingPhase, target: TrainingPhase) -> None:
    """Raise ValueError if the transition is illegal; no-op otherwise."""
    if current == target:
        return
    reason = _ILLEGAL_TRANSITIONS.get((current, target))
    if reason is not None:
        raise ValueError(
            f"Illegal phase transition {current.value!r} → {target.value!r}: {reason}"
        )
    if not _VALID_TRANSITIONS.get((current, target), False):
        # Unknown transition — also illegal
        raise ValueError(
            f"Unknown/illegal phase transition {current.value!r} → {target.value!r}"
        )


def can_transition(current: TrainingPhase, target: TrainingPhase) -> bool:
    """Return True iff the transition is allowed (no exception)."""
    try:
        validate_transition(current, target)
        return True
    except ValueError:
        return False


# Sentinel for "no step configured"
_UNSET = -1


class PhaseManager:
    """Phase state machine for HelixFullTrainer lifecycle.

    This manager owns the phase enum and provides decision/validation helpers.
    Side effects (model parameter mutation, optimizer LR, logger output) are
    handled by the trainer via delegation wrappers — the manager does NOT hold
    references to model, optimizer, or logger.
    """

    def __init__(
        self,
        *,
        representation_only_steps: int = 0,
        head_only_steps: int = 0,
        representation_diagnostic_mode: bool = False,
        use_energy_based_family_objective: bool = True,
        rep_adaptive_exit_ratio_threshold: float = 1.6,
        rep_adaptive_exit_min_inter_threshold: float = 0.30,
        representation_window_pattern: list[tuple[bool, int]] | None = None,
        joint_finetune_backbone_lr_multiplier: float = 0.25,
        joint_finetune_head_lr_multiplier: float = 0.15,
    ) -> None:
        self._current_phase = TrainingPhase.IDLE

        # Phase config (snapshotted at construction)
        self._representation_only_steps = int(representation_only_steps)
        self._head_only_steps = int(head_only_steps)
        self._representation_diagnostic_mode = bool(representation_diagnostic_mode)
        self._use_energy_based_family_objective = bool(use_energy_based_family_objective)
        self._rep_adaptive_exit_ratio_threshold = float(rep_adaptive_exit_ratio_threshold)
        self._rep_adaptive_exit_min_inter_threshold = float(rep_adaptive_exit_min_inter_threshold)
        self._representation_window_pattern: list[tuple[bool, int]] = (
            list(representation_window_pattern) if representation_window_pattern else []
        )
        self._joint_finetune_backbone_lr_multiplier = float(joint_finetune_backbone_lr_multiplier)
        self._joint_finetune_head_lr_multiplier = float(joint_finetune_head_lr_multiplier)

        # Runtime phase tracking (mirrors trainer flag attributes)
        self._representation_phase_active = False
        self._representation_curriculum_complete = False
        self._in_representation_window = False
        self._head_phase_start_step = _UNSET
        self._joint_finetune_start_step = _UNSET
        self._joint_finetune_active = False

    # ------------------------------------------------------------------ #
    # Public property accessors
    # ------------------------------------------------------------------ #

    @property
    def current_phase(self) -> TrainingPhase:
        return self._current_phase

    @property
    def representation_phase_active(self) -> bool:
        return self._representation_phase_active

    @representation_phase_active.setter
    def representation_phase_active(self, value: bool) -> None:
        self._representation_phase_active = bool(value)

    @property
    def representation_curriculum_complete(self) -> bool:
        return self._representation_curriculum_complete

    @representation_curriculum_complete.setter
    def representation_curriculum_complete(self, value: bool) -> None:
        self._representation_curriculum_complete = bool(value)

    @property
    def in_representation_window(self) -> bool:
        return self._in_representation_window

    @in_representation_window.setter
    def in_representation_window(self, value: bool) -> None:
        self._in_representation_window = bool(value)

    @property
    def head_phase_start_step(self) -> int:
        return self._head_phase_start_step

    @head_phase_start_step.setter
    def head_phase_start_step(self, value: int) -> None:
        self._head_phase_start_step = int(value)

    @property
    def joint_finetune_start_step(self) -> int:
        return self._joint_finetune_start_step

    @joint_finetune_start_step.setter
    def joint_finetune_start_step(self, value: int) -> None:
        self._joint_finetune_start_step = int(value)

    @property
    def joint_finetune_active(self) -> bool:
        return self._joint_finetune_active

    @joint_finetune_active.setter
    def joint_finetune_active(self, value: bool) -> None:
        self._joint_finetune_active = bool(value)

    @property
    def representation_only_steps(self) -> int:
        return self._representation_only_steps

    @property
    def head_only_steps(self) -> int:
        return self._head_only_steps

    @property
    def representation_diagnostic_mode(self) -> bool:
        return self._representation_diagnostic_mode

    @property
    def use_energy_based_family_objective(self) -> bool:
        return self._use_energy_based_family_objective

    @property
    def representation_window_pattern(self) -> list[tuple[bool, int]]:
        return list(self._representation_window_pattern)

    @property
    def joint_finetune_backbone_lr_multiplier(self) -> float:
        return self._joint_finetune_backbone_lr_multiplier

    @property
    def joint_finetune_head_lr_multiplier(self) -> float:
        return self._joint_finetune_head_lr_multiplier

    # ------------------------------------------------------------------ #
    # Phase state synchronization helpers
    # ------------------------------------------------------------------ #

    def _sync_from_flags(self) -> None:
        """Recompute _current_phase from flag-based state (for backward compat)."""
        if self._joint_finetune_active:
            self._current_phase = TrainingPhase.JOINT_FINETUNE
        elif self._head_phase_start_step >= 0:
            self._current_phase = TrainingPhase.HEAD_ONLY
        elif self._representation_phase_active:
            self._current_phase = TrainingPhase.REPRESENTATION_ONLY
        else:
            self._current_phase = TrainingPhase.IDLE

    def transition_to(self, target: TrainingPhase) -> None:
        """Validate and apply a phase transition.

        Raises ValueError on illegal transitions.
        """
        validate_transition(self._current_phase, target)
        self._current_phase = target

    # ------------------------------------------------------------------ #
    # Transition evaluation — condition checks (used by trainer wrappers)
    # ------------------------------------------------------------------ #

    def can_start_representation_phase(self, global_step: int) -> bool:
        """Return True if representation phase should be activated."""
        if self._representation_phase_active:
            return False
        if int(global_step) >= self._representation_only_steps:
            return False
        return True

    def is_curriculum_complete(self, global_step: int) -> bool:
        """Return True when representation curriculum should be marked complete."""
        if int(global_step) >= self._representation_only_steps:
            return True
        return False

    def should_exit_curriculum_by_targets(
        self,
        *,
        intra_inter_ratio: float,
        min_inter_center_distance: float,
    ) -> bool:
        """Return True when adaptive geometry targets are met."""
        if self._use_energy_based_family_objective:
            return False
        return (
            intra_inter_ratio < self._rep_adaptive_exit_ratio_threshold
            and min_inter_center_distance > self._rep_adaptive_exit_min_inter_threshold
        )

    def can_activate_joint_finetune(self, global_step: int) -> bool:
        """Return True when joint-finetune transition conditions are met."""
        if not self._representation_diagnostic_mode:
            return False
        if self._representation_phase_active or self._head_phase_start_step < 0:
            return False
        if self._joint_finetune_active:
            return False
        if int(global_step) < self._joint_finetune_start_step:
            return False
        return True

    def can_transition_to_head_phase(self, in_representation_phase: bool) -> bool:
        """Return True when transition from representation to head phase should occur."""
        if in_representation_phase:
            return False
        if not self._representation_phase_active:
            return False
        if not self._representation_curriculum_complete:
            return False
        return True

    # ------------------------------------------------------------------ #
    # Representation window-step logic
    # ------------------------------------------------------------------ #

    def is_representation_window_step(self, step: int) -> bool:
        """Return whether *step* falls in a representation micro-window."""
        if self._representation_curriculum_complete:
            return False
        if not self._representation_window_pattern:
            return int(step) < int(self._representation_only_steps)

        remaining = int(step)
        for is_rep_window, window_steps in self._representation_window_pattern:
            if remaining < int(window_steps):
                return bool(is_rep_window)
            remaining -= int(window_steps)
        return False

    # ------------------------------------------------------------------ #
    # Phase target computation
    # ------------------------------------------------------------------ #

    def compute_joint_finetune_params(self) -> dict[str, float]:
        """Return LR multiplier targets for joint-finetune phase."""
        return {
            "backbone_multiplier": self._joint_finetune_backbone_lr_multiplier,
            "head_multiplier": self._joint_finetune_head_lr_multiplier,
        }

    @staticmethod
    def compute_trainability_targets(
        train_backbone: bool,
        train_family_head: bool,
        *,
        train_family_projection: bool | None = None,
    ) -> dict[str, bool]:
        """Compute trainability target flags.

        Returns dict with keys: train_backbone, train_family_head, train_family_projection.
        """
        return {
            "train_backbone": bool(train_backbone),
            "train_family_head": bool(train_family_head),
            "train_family_projection": bool(
                train_family_head if train_family_projection is None else train_family_projection
            ),
        }

    @staticmethod
    def compute_lr_scale_targets(
        backbone_multiplier: float,
        head_multiplier: float,
    ) -> dict[str, float]:
        """Return clamped LR multiplier targets."""
        return {
            "backbone_multiplier": max(1e-4, float(backbone_multiplier)),
            "head_multiplier": max(1e-4, float(head_multiplier)),
        }

    # ------------------------------------------------------------------ #
    # Phase lifecycle helpers — called step-by-step from trainer wrappers
    # ------------------------------------------------------------------ #

    def apply_representation_phase_start(self) -> None:
        """Apply state changes for representation phase start."""
        self._representation_phase_active = True
        self._representation_curriculum_complete = False
        self._in_representation_window = False
        self._sync_from_flags()

    def apply_curriculum_complete(self) -> None:
        """Mark curriculum complete."""
        self._representation_curriculum_complete = True

    def apply_head_phase_start(self, global_step: int) -> int:
        """Apply state changes when entering head-only phase.

        Returns the joint_finetune_start_step.
        """
        self._representation_phase_active = False
        self._head_phase_start_step = int(global_step)
        self._joint_finetune_start_step = self._head_phase_start_step + self._head_only_steps
        self._sync_from_flags()
        return self._joint_finetune_start_step

    def apply_joint_finetune_activation(self) -> None:
        """Apply state changes for joint-finetune activation."""
        self._joint_finetune_active = True
        self._sync_from_flags()

    def apply_window_enter(self) -> None:
        """Apply state when entering a representation micro-window."""
        self._in_representation_window = True

    def apply_window_exit(self) -> None:
        """Apply state when exiting a representation micro-window."""
        self._in_representation_window = False
        self._sync_from_flags()
