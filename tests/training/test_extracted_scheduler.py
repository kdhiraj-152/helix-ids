"""Regression tests for extracted scheduler package (Phase 13A-2).

Covers:
    - TrainingPhase enum (transition validation, inspection helpers)
    - PhaseManager (transitions, state machine, curriculum helpers, window logic)
    - FreezeManager (freeze/unfreeze state, should_unfreeze logic)
    - LRScheduler (warmup, cosine decay, apply_lr_scales)
    - EarlyStoppingManager (hard-stop detection, early stopping update, smoke mode)
    - Trainer delegation wrappers (integration tests against HelixFullTrainer)
"""


import pytest

from scripts.training.scheduler import (
    EarlyStoppingManager,
    FreezeManager,
    LRScheduler,
    PhaseManager,
    TrainingPhase,
    can_transition,
    validate_transition,
)

# =========================================================================== #
# TrainingPhase enum tests
# =========================================================================== #

class TestTrainingPhaseEnum:
    """TrainingPhase enum smoke tests."""

    def test_enum_values(self) -> None:
        assert TrainingPhase.IDLE.value == "idle"
        assert TrainingPhase.REPRESENTATION_ONLY.value == "representation_only"
        assert TrainingPhase.HEAD_ONLY.value == "head_only"
        assert TrainingPhase.JOINT_FINETUNE.value == "joint_finetune"
        assert TrainingPhase.COMPLETE.value == "complete"

    def test_is_terminal(self) -> None:
        assert not TrainingPhase.IDLE.is_terminal
        assert not TrainingPhase.REPRESENTATION_ONLY.is_terminal
        assert not TrainingPhase.HEAD_ONLY.is_terminal
        assert TrainingPhase.JOINT_FINETUNE.is_terminal
        assert TrainingPhase.COMPLETE.is_terminal

    def test_inspection_properties(self) -> None:
        assert TrainingPhase.REPRESENTATION_ONLY.is_representation
        assert not TrainingPhase.HEAD_ONLY.is_representation
        assert TrainingPhase.HEAD_ONLY.is_head_only
        assert not TrainingPhase.REPRESENTATION_ONLY.is_head_only
        assert TrainingPhase.JOINT_FINETUNE.is_joint_finetune
        assert not TrainingPhase.HEAD_ONLY.is_joint_finetune

    def test_str_representation(self) -> None:
        assert str(TrainingPhase.IDLE) == "idle"
        assert str(TrainingPhase.COMPLETE) == "complete"


# =========================================================================== #
# Transition validation tests
# =========================================================================== #

class TestTransitionValidation:
    """State-machine transition validation and helper functions."""

    def test_valid_idle_to_representation(self) -> None:
        assert can_transition(TrainingPhase.IDLE, TrainingPhase.REPRESENTATION_ONLY)
        validate_transition(TrainingPhase.IDLE, TrainingPhase.REPRESENTATION_ONLY)

    def test_valid_idle_to_head(self) -> None:
        assert can_transition(TrainingPhase.IDLE, TrainingPhase.HEAD_ONLY)
        validate_transition(TrainingPhase.IDLE, TrainingPhase.HEAD_ONLY)

    def test_valid_idle_to_complete(self) -> None:
        assert can_transition(TrainingPhase.IDLE, TrainingPhase.COMPLETE)
        validate_transition(TrainingPhase.IDLE, TrainingPhase.COMPLETE)

    def test_valid_rep_to_head(self) -> None:
        assert can_transition(TrainingPhase.REPRESENTATION_ONLY, TrainingPhase.HEAD_ONLY)
        validate_transition(TrainingPhase.REPRESENTATION_ONLY, TrainingPhase.HEAD_ONLY)

    def test_valid_head_to_joint(self) -> None:
        assert can_transition(TrainingPhase.HEAD_ONLY, TrainingPhase.JOINT_FINETUNE)
        validate_transition(TrainingPhase.HEAD_ONLY, TrainingPhase.JOINT_FINETUNE)

    def test_valid_head_to_complete(self) -> None:
        assert can_transition(TrainingPhase.HEAD_ONLY, TrainingPhase.COMPLETE)
        validate_transition(TrainingPhase.HEAD_ONLY, TrainingPhase.COMPLETE)

    def test_stay_in_phase(self) -> None:
        for phase in TrainingPhase:
            assert can_transition(phase, phase)

    # --- Illegal transitions ---

    def test_illegal_representation_to_joint(self) -> None:
        assert not can_transition(TrainingPhase.REPRESENTATION_ONLY, TrainingPhase.JOINT_FINETUNE)
        msg = "must pass through HEAD_ONLY"
        with pytest.raises(ValueError, match=msg):
            validate_transition(TrainingPhase.REPRESENTATION_ONLY, TrainingPhase.JOINT_FINETUNE)

    def test_illegal_head_to_representation(self) -> None:
        assert not can_transition(TrainingPhase.HEAD_ONLY, TrainingPhase.REPRESENTATION_ONLY)
        msg = "irreversible"
        with pytest.raises(ValueError, match=msg):
            validate_transition(TrainingPhase.HEAD_ONLY, TrainingPhase.REPRESENTATION_ONLY)

    def test_illegal_joint_to_representation(self) -> None:
        assert not can_transition(TrainingPhase.JOINT_FINETUNE, TrainingPhase.REPRESENTATION_ONLY)

    def test_illegal_joint_to_head(self) -> None:
        assert not can_transition(TrainingPhase.JOINT_FINETUNE, TrainingPhase.HEAD_ONLY)

    def test_illegal_complete_back_to_representation(self) -> None:
        assert not can_transition(TrainingPhase.COMPLETE, TrainingPhase.REPRESENTATION_ONLY)
        msg = "terminal"
        with pytest.raises(ValueError, match=msg):
            validate_transition(TrainingPhase.COMPLETE, TrainingPhase.REPRESENTATION_ONLY)

    def test_illegal_complete_back_to_head(self) -> None:
        assert not can_transition(TrainingPhase.COMPLETE, TrainingPhase.HEAD_ONLY)

    def test_illegal_complete_back_to_joint(self) -> None:
        assert not can_transition(TrainingPhase.COMPLETE, TrainingPhase.JOINT_FINETUNE)


# =========================================================================== #
# PhaseManager tests
# =========================================================================== #

class TestPhaseManager:
    """PhaseManager state machine behavior."""

    @pytest.fixture
    def manager(self) -> PhaseManager:
        return PhaseManager(
            representation_only_steps=100,
            head_only_steps=200,
            representation_diagnostic_mode=True,
            use_energy_based_family_objective=True,
            rep_adaptive_exit_ratio_threshold=1.5,
            rep_adaptive_exit_min_inter_threshold=0.25,
            representation_window_pattern=None,
        )

    def test_initial_phase_is_idle(self, manager: PhaseManager) -> None:
        assert manager.current_phase == TrainingPhase.IDLE

    def test_initial_flags_false(self, manager: PhaseManager) -> None:
        assert not manager.representation_phase_active
        assert not manager.representation_curriculum_complete
        assert not manager.in_representation_window
        assert not manager.joint_finetune_active
        assert manager.head_phase_start_step < 0
        assert manager.joint_finetune_start_step < 0

    def test_can_start_representation_phase_early_step(self, manager: PhaseManager) -> None:
        """Before representation_only_steps, can start representation phase."""
        assert manager.can_start_representation_phase(global_step=50)

    def test_cannot_start_representation_phase_past_threshold(self, manager: PhaseManager) -> None:
        """After representation_only_steps, cannot start representation phase."""
        assert not manager.can_start_representation_phase(global_step=150)

    def test_cannot_start_representation_phase_if_already_active(self, manager: PhaseManager) -> None:
        """If representation_phase_active is True, returns False."""
        manager.representation_phase_active = True
        assert not manager.can_start_representation_phase(global_step=50)

    def test_is_curriculum_complete_after_threshold(self, manager: PhaseManager) -> None:
        assert manager.is_curriculum_complete(global_step=150)

    def test_is_curriculum_not_complete_before_threshold(self, manager: PhaseManager) -> None:
        assert not manager.is_curriculum_complete(global_step=50)

    def test_apply_representation_phase_start(self, manager: PhaseManager) -> None:
        manager.apply_representation_phase_start()
        assert manager.representation_phase_active
        assert not manager.representation_curriculum_complete
        assert not manager.in_representation_window
        assert manager.current_phase == TrainingPhase.REPRESENTATION_ONLY

    def test_apply_curriculum_complete(self, manager: PhaseManager) -> None:
        manager.representation_curriculum_complete = True
        assert manager.representation_curriculum_complete

    def test_apply_head_phase_start(self, manager: PhaseManager) -> None:
        manager.apply_representation_phase_start()
        result = manager.apply_head_phase_start(global_step=300)
        assert not manager.representation_phase_active
        assert manager.head_phase_start_step == 300
        assert manager.joint_finetune_start_step == 500  # 300 + 200
        assert result == 500
        assert manager.current_phase == TrainingPhase.HEAD_ONLY

    def test_can_transition_to_head_phase_when_ready(self, manager: PhaseManager) -> None:
        manager.apply_representation_phase_start()
        manager.apply_curriculum_complete()
        assert manager.can_transition_to_head_phase(in_representation_phase=False)

    def test_cannot_transition_to_head_phase_if_in_rep_phase(self, manager: PhaseManager) -> None:
        assert not manager.can_transition_to_head_phase(in_representation_phase=True)

    def test_cannot_transition_to_head_phase_if_not_active(self, manager: PhaseManager) -> None:
        assert not manager.can_transition_to_head_phase(in_representation_phase=False)

    def test_cannot_transition_to_head_phase_if_curriculum_not_complete(self, manager: PhaseManager) -> None:
        manager.representation_phase_active = True
        assert not manager.can_transition_to_head_phase(in_representation_phase=False)

    def test_apply_joint_finetune_activation(self, manager: PhaseManager) -> None:
        manager.apply_joint_finetune_activation()
        assert manager.joint_finetune_active
        assert manager.current_phase == TrainingPhase.JOINT_FINETUNE

    def test_can_activate_joint_finetune(self, manager: PhaseManager) -> None:
        # Must be in head phase first
        manager.apply_representation_phase_start()
        manager.apply_head_phase_start(global_step=50)
        manager.joint_finetune_start_step = 50
        assert manager.can_activate_joint_finetune(global_step=50)

    def test_cannot_activate_joint_finetune_if_not_diagnostic_mode(self) -> None:
        m = PhaseManager(representation_only_steps=100, head_only_steps=200, representation_diagnostic_mode=False)
        assert not m.can_activate_joint_finetune(global_step=50)

    def test_cannot_activate_joint_finetune_if_already_active(self, manager: PhaseManager) -> None:
        manager.joint_finetune_active = True
        manager.head_phase_start_step = 50
        assert not manager.can_activate_joint_finetune(global_step=50)

    def test_window_enter_exit(self, manager: PhaseManager) -> None:
        manager.apply_window_enter()
        assert manager.in_representation_window
        manager.apply_window_exit()
        assert not manager.in_representation_window

    def test_is_representation_window_step_before_threshold(self, manager: PhaseManager) -> None:
        assert manager.is_representation_window_step(step=50)

    def test_is_representation_window_step_after_threshold(self, manager: PhaseManager) -> None:
        manager.representation_curriculum_complete = True
        assert not manager.is_representation_window_step(step=50)

    def test_compute_joint_finetune_params(self, manager: PhaseManager) -> None:
        params = manager.compute_joint_finetune_params()
        assert params["backbone_multiplier"] == 0.25
        assert params["head_multiplier"] == 0.15

    def test_compute_trainability_targets_default(self) -> None:
        targets = PhaseManager.compute_trainability_targets(
            train_backbone=True, train_family_head=True
        )
        assert targets["train_backbone"]
        assert targets["train_family_head"]
        assert targets["train_family_projection"]  # mirrors head

    def test_compute_trainability_targets_projection_override(self) -> None:
        targets = PhaseManager.compute_trainability_targets(
            train_backbone=False, train_family_head=True, train_family_projection=False
        )
        assert not targets["train_backbone"]
        assert targets["train_family_head"]
        assert not targets["train_family_projection"]

    def test_compute_lr_scale_targets_clamps(self) -> None:
        targets = PhaseManager.compute_lr_scale_targets(
            backbone_multiplier=0.0, head_multiplier=-1.0
        )
        assert targets["backbone_multiplier"] >= 1e-4
        assert targets["head_multiplier"] >= 1e-4


# =========================================================================== #
# FreezeManager tests
# =========================================================================== #

class TestFreezeManager:
    """FreezeManager freeze/unfreeze state and decision logic."""

    @pytest.fixture
    def fm(self) -> FreezeManager:
        return FreezeManager()

    def test_initial_state_unfrozen(self, fm: FreezeManager) -> None:
        assert not fm.backbone_frozen

    def test_freeze_state_setter(self, fm: FreezeManager) -> None:
        fm.backbone_frozen = True
        assert fm.backbone_frozen
        fm.backbone_frozen = False
        assert not fm.backbone_frozen

    def test_should_unfreeze_when_not_frozen(self, fm: FreezeManager) -> None:
        assert not fm.should_unfreeze(global_step=100, unfreeze_backbone_step=50)

    def test_should_unfreeze_when_frozen_and_step_met(self, fm: FreezeManager) -> None:
        fm.backbone_frozen = True
        assert fm.should_unfreeze(global_step=100, unfreeze_backbone_step=50)

    def test_should_not_unfreeze_when_frozen_and_step_not_met(self, fm: FreezeManager) -> None:
        fm.backbone_frozen = True
        assert not fm.should_unfreeze(global_step=30, unfreeze_backbone_step=50)

    def test_should_not_unfreeze_when_frozen_and_unfreeze_step_zero(self, fm: FreezeManager) -> None:
        fm.backbone_frozen = True
        assert not fm.should_unfreeze(global_step=100, unfreeze_backbone_step=0)

    def test_should_not_unfreeze_when_frozen_and_unfreeze_step_negative(self, fm: FreezeManager) -> None:
        fm.backbone_frozen = True
        assert not fm.should_unfreeze(global_step=100, unfreeze_backbone_step=-1)

    def test_edge_boundary_exact_match(self, fm: FreezeManager) -> None:
        fm.backbone_frozen = True
        assert fm.should_unfreeze(global_step=50, unfreeze_backbone_step=50)


# =========================================================================== #
# LRScheduler tests
# =========================================================================== #

class TestLRScheduler:
    """LRScheduler warmup, cosine decay, and apply_lr_scales logic."""

    @pytest.fixture
    def lr(self) -> LRScheduler:
        return LRScheduler(
            learning_rate=0.001,
            warmup_epochs=5,
            warmup_init_lr=0.0,
            epochs=50,
            min_lr_ratio=0.05,
        )

    def test_properties(self, lr: LRScheduler) -> None:
        assert lr.learning_rate == 0.001
        assert lr.warmup_epochs == 5
        assert lr.epochs == 50

    def test_get_learning_rate_during_warmup(self, lr: LRScheduler) -> None:
        """Warmup: epoch 0 should give lr = init_lr + (lr - init_lr) * (1/5)."""
        rate = lr.get_learning_rate(epoch=0)
        expected = 0.0 + (0.001 - 0.0) * (1 / 5)
        assert rate == pytest.approx(expected)

    def test_get_learning_rate_at_start_of_warmup(self, lr: LRScheduler) -> None:
        rate = lr.get_learning_rate(epoch=4)
        expected = 0.0 + (0.001 - 0.0) * (5 / 5)
        assert rate == pytest.approx(expected)

    def test_get_learning_rate_after_warmup(self, lr: LRScheduler) -> None:
        """Epoch 5 should be cosine decay start."""
        rate = lr.get_learning_rate(epoch=5)
        min_lr = 0.001 * 0.05
        # cosine decay at step 0 of 45 decay steps
        assert rate == pytest.approx(min_lr + (0.001 - min_lr) * 1.0)

    def test_get_learning_rate_mid_decay(self, lr: LRScheduler) -> None:
        rate = lr.get_learning_rate(epoch=27)
        min_lr = 0.001 * 0.05
        # Should be between min_lr and initial LR
        assert min_lr < rate < 0.001

    def test_get_learning_rate_final_epoch(self, lr: LRScheduler) -> None:
        # At the end of training, LR should be near min_lr (cosine never reaches it exactly)
        rate = lr.get_learning_rate(epoch=49)
        min_lr = 0.001 * 0.05
        assert rate == pytest.approx(min_lr, abs=1e-4)

    def test_apply_lr_scales_basic(self) -> None:
        param_groups = [
            {"group_name": "backbone", "lr_scale": 1.0},
            {"group_name": "family_head", "lr_scale": 1.0},
            {"group_name": "binary_head", "lr_scale": 1.0},
        ]
        base_scales: dict[str, float] = {"backbone": 1.0, "family_head": 1.0, "binary_head": 1.0}
        result = LRScheduler.apply_lr_scales(
            param_groups, base_scales,
            backbone_multiplier=0.25, head_multiplier=0.15,
        )
        assert result["backbone"] == 0.25
        assert result["family_head"] == 0.15
        assert result["binary_head"] == 1.0

    def test_apply_lr_scales_clamps_minimum(self) -> None:
        param_groups = [{"group_name": "backbone", "lr_scale": 1.0}]
        base_scales = {"backbone": 1.0}
        result = LRScheduler.apply_lr_scales(
            param_groups, base_scales,
            backbone_multiplier=0.0, head_multiplier=0.0,
        )
        assert result["backbone"] >= 1e-4

    def test_apply_lr_scales_default_scale_for_missing_groups(self) -> None:
        param_groups = [{"group_name": "unknown_group"}]
        base_scales = {}
        result = LRScheduler.apply_lr_scales(
            param_groups, base_scales,
            backbone_multiplier=1.0, head_multiplier=1.0,
        )
        assert result["unknown_group"] == 1.0


# =========================================================================== #
# EarlyStoppingManager tests
# =========================================================================== #

class TestEarlyStoppingManager:
    """Early-stopping state, hard-stop detection, and smoke mode."""

    @pytest.fixture
    def esm(self) -> EarlyStoppingManager:
        return EarlyStoppingManager(
            early_stopping_patience=3,
            early_stopping_threshold=0.02,
            min_family_minority_recall_for_best=0.15,
            disable_integrity_hard_stops=False,
        )

    def test_initial_state(self, esm: EarlyStoppingManager) -> None:
        assert esm.best_val_loss == float("inf")
        assert esm.patience_counter == 0
        assert esm.val_gap_collapse_streak == 0
        assert esm.high_accuracy_high_loss_streak == 0
        assert esm.entropy_missing_class_streak == 0
        assert esm.entropy_collapse_streak == 0

    def test_smoke_mode_from_gov_profile(self) -> None:
        assert EarlyStoppingManager.is_smoke_mode(gov_profile="smoke")

    def test_smoke_mode_from_epochs(self) -> None:
        assert EarlyStoppingManager.is_smoke_mode(epochs=5)
        assert not EarlyStoppingManager.is_smoke_mode(epochs=50)

    def test_smoke_mode_default_not_smoke(self) -> None:
        assert not EarlyStoppingManager.is_smoke_mode()

    def test_hard_stop_reason_disabled(self) -> None:
        esm = EarlyStoppingManager(
            early_stopping_patience=3,
            early_stopping_threshold=0.02,
            min_family_minority_recall_for_best=0.15,
            disable_integrity_hard_stops=True,
        )
        result = esm.hard_stop_reason({}, {}, is_smoke=False, epoch=0)
        assert result is None

    def test_hard_stop_val_gap_collapse(self, esm: EarlyStoppingManager) -> None:
        train_metrics = {"train_calibrated_loss": 1.0}
        val_metrics = {
            "val_calibrated_loss": 0.5,
            "val_family_macro_f1": 0.1,
            "val_family_minority_recall_min": 0.05,
            "val_family_entropy": 0.05,
        }
        # First call: streak 1
        result = esm.hard_stop_reason(train_metrics, val_metrics, is_smoke=False, epoch=0)
        assert result is None
        assert esm.val_gap_collapse_streak == 1
        # Second call: streak 2 -> triggers
        result = esm.hard_stop_reason(train_metrics, val_metrics, is_smoke=False, epoch=0)
        assert result == "val_loss_below_train_loss_with_collapse"

    def test_hard_stop_high_accuracy_high_loss(self, esm: EarlyStoppingManager) -> None:
        train_metrics = {"train_calibrated_loss": 0.6, "train_binary_acc": 0.96}
        val_metrics = {"val_binary_acc": 0.97}
        # First call: epoch 0 < min_epoch(1) -> streak = 1
        result = esm.hard_stop_reason(train_metrics, val_metrics, is_smoke=False, epoch=0)
        assert result is None
        assert esm.high_accuracy_high_loss_streak == 1
        # Second call: epoch >= min_epoch and streak >= streak -> triggers
        result = esm.hard_stop_reason(train_metrics, val_metrics, is_smoke=False, epoch=2)
        assert result == "high_accuracy_with_high_loss"

    def test_hard_stop_high_accuracy_streak_resets(self, esm: EarlyStoppingManager) -> None:
        esm.high_accuracy_high_loss_streak = 1
        train_metrics = {"train_calibrated_loss": 0.3, "train_binary_acc": 0.96}  # low loss
        val_metrics = {"val_binary_acc": 0.97}
        result = esm.hard_stop_reason(train_metrics, val_metrics, is_smoke=False, epoch=0)
        assert result is None
        assert esm.high_accuracy_high_loss_streak == 0  # reset

    def test_hard_stop_entropy_missing_class(self, esm: EarlyStoppingManager) -> None:
        val_metrics = {"val_family_entropy": 0.05, "val_entropy_missing_same_dataset": 1.0}
        # Full mode: entropy_threshold=0.12, streak=2, min_epoch=2
        # First call (epoch=2): streak 1
        result = esm.hard_stop_reason({}, val_metrics, is_smoke=False, epoch=2)
        assert result is None
        assert esm.entropy_missing_class_streak == 1
        # Second call: streak 2 -> triggers
        result = esm.hard_stop_reason({}, val_metrics, is_smoke=False, epoch=3)
        assert result == "prediction_entropy_collapse_with_missing_classes"

    def test_hard_stop_entropy_critical(self, esm: EarlyStoppingManager) -> None:
        val_metrics = {"val_family_entropy": 0.05, "val_entropy_missing_same_dataset": 0.0}
        # entropy < 0.08 and no missing-class signal -> critical collapse
        result = esm.hard_stop_reason({}, val_metrics, is_smoke=False, epoch=2)
        assert result is None  # streak 1, need 3
        assert esm.entropy_collapse_streak == 1

    def test_early_stopping_update_new_best(self, esm: EarlyStoppingManager) -> None:
        result = esm.update_early_stopping(
            {"val_loss": 0.5, "val_family_minority_recall_min": 0.3, "val_family_entropy": 0.4},
            quality_gate_minority_recall=0.15,
            quality_gate_entropy=0.3,
        )
        assert result["is_best"]
        assert result["best_val_loss"] == 0.5
        assert result["patience_counter"] == 0

    def test_early_stopping_update_not_best(self, esm: EarlyStoppingManager) -> None:
        esm.best_val_loss = 0.3
        result = esm.update_early_stopping(
            {"val_loss": 0.5, "val_family_minority_recall_min": 0.3, "val_family_entropy": 0.4},
            quality_gate_minority_recall=0.15,
            quality_gate_entropy=0.3,
        )
        assert not result["is_best"]
        assert result["patience_counter"] == 1

    def test_early_stopping_patience_triggered(self, esm: EarlyStoppingManager) -> None:
        esm.best_val_loss = 0.1  # Set a baseline so 0.5 is NOT a new best
        result = esm.update_early_stopping(
            {"val_loss": 0.5, "val_family_minority_recall_min": 0.3, "val_family_entropy": 0.4},
            quality_gate_minority_recall=0.15,
            quality_gate_entropy=0.3,
        )
        # patience_counter starts at 0, goes to 1
        # Set to 2 and call again
        esm.patience_counter = 2
        result = esm.update_early_stopping(
            {"val_loss": 0.5, "val_family_minority_recall_min": 0.3, "val_family_entropy": 0.4},
            quality_gate_minority_recall=0.15,
            quality_gate_entropy=0.3,
        )
        assert result["should_stop"]
        assert result["patience_counter"] == 3

    def test_early_stopping_update_best_but_gate_fails(self, esm: EarlyStoppingManager) -> None:
        esm.best_val_loss = 0.5
        result = esm.update_early_stopping(
            {"val_loss": 0.3, "val_family_minority_recall_min": 0.05, "val_family_entropy": 0.4},
            quality_gate_minority_recall=0.15,
            quality_gate_entropy=0.3,
        )
        assert not result["is_best"]  # gate fails on minority_recall < 0.15
        assert result["patience_counter"] == 1

    def test_reset(self, esm: EarlyStoppingManager) -> None:
        esm.best_val_loss = 0.5
        esm.patience_counter = 2
        esm.high_accuracy_high_loss_streak = 3
        esm.reset()
        assert esm.best_val_loss == float("inf")
        assert esm.patience_counter == 0
        assert esm.high_accuracy_high_loss_streak == 0
