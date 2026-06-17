"""
Regression tests for TrainerState construction, delegate wiring, and state lifecycle.

Phase 18: Verifies that TrainerState correctly initializes all state variables,
constructs delegates from its state, and maintains state integrity.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
import torch

from scripts.training.core.trainer_state import TrainerState


@pytest.fixture
def mock_state_kwargs():
    """Minimal mock kwargs for TrainerState construction."""
    model = MagicMock()
    model.backbone.parameters.return_value = []
    model.to.return_value = model

    train_loader = MagicMock()
    train_loader.__len__.return_value = 100

    config = MagicMock()
    config.epochs = 10
    config.learning_rate = 1e-3
    config.freeze_backbone_epochs = 2
    config.unfreeze_backbone_step = 0
    config.entropy_warmup_steps = 50
    config.entropy_warmup_weight = 0.1
    config.warmup_init_lr = 1e-6
    config.warmup_epochs = 0

    optimizer = MagicMock()
    optimizer.param_groups = [{"group_name": "backbone", "lr_scale": 1.0}]

    loss_fn = MagicMock()
    logger = logging.getLogger("test_trainer_state")

    return {
        "model": model,
        "train_loader": train_loader,
        "val_loaders": {"val": MagicMock()},
        "test_loaders": {"test": MagicMock()},
        "optimizer": optimizer,
        "loss_fn": loss_fn,
        "config": config,
        "run_seed": 42,
        "device": "cpu",
        "logger": logger,
    }


class TestTrainerStateConstruction:
    """TrainerState initializes correctly."""

    def test_creates_without_error(self, mock_state_kwargs):
        state = TrainerState(**mock_state_kwargs)
        assert state.model is not None
        assert state.train_loader is not None
        assert state.logger is not None

    def test_core_references_stored(self, mock_state_kwargs):
        state = TrainerState(**mock_state_kwargs)
        assert state.model is mock_state_kwargs["model"]
        assert state.train_loader is mock_state_kwargs["train_loader"]
        assert state.val_loaders is mock_state_kwargs["val_loaders"]
        assert state.test_loaders is mock_state_kwargs["test_loaders"]
        assert state.optimizer is mock_state_kwargs["optimizer"]
        assert state.loss_fn is mock_state_kwargs["loss_fn"]
        assert state.config is mock_state_kwargs["config"]
        assert state.device == "cpu"
        assert state.run_seed == 42

    def test_state_defaults(self, mock_state_kwargs):
        state = TrainerState(**mock_state_kwargs)
        assert state.epoch == 0
        assert state.global_step == 0
        assert state.best_val_loss == float("inf")
        assert state.best_model_state is None
        assert state.patience_counter == 0
        assert state.backbone_frozen is False
        assert state.training_history is not None
        assert "train_loss" in state.training_history

    def test_representation_defaults(self, mock_state_kwargs):
        state = TrainerState(**mock_state_kwargs)
        assert state.representation_phase_active is False
        assert state.representation_curriculum_complete is False
        assert state.representation_diagnostic_mode is False
        assert state.representation_only_steps == 0
        assert state.head_only_steps == 0
        assert state.cluster_centers is None
        assert state.joint_finetune_active is False

    def test_energy_defaults(self, mock_state_kwargs):
        state = TrainerState(**mock_state_kwargs)
        assert state.use_energy_based_family_objective is True
        assert state.energy_gap_margin == 1.0
        assert state.energy_gap_weight == 1.0
        assert state.energy_logit_temperature == 2.0
        assert state.energy_balance_weight == 0.1

    def test_geometry_defaults(self, mock_state_kwargs):
        state = TrainerState(**mock_state_kwargs)
        assert state.geometry_min_inter_threshold == 0.2
        assert state.geometry_max_intra_inter_ratio_warmup == 2.5
        assert state.geometry_max_intra_inter_ratio_post_phase == 1.2
        assert state.geometry_min_cluster_size == 100

    def test_runtime_flags_defaults(self, mock_state_kwargs):
        state = TrainerState(**mock_state_kwargs)
        assert state.disable_integrity_hard_stops is False
        assert state.disable_tail_focal_regularizer is False
        assert state.train_temperature == 1.0
        assert state.base_balance_strategy == "weighted_ce"

    def test_total_train_steps_computed(self, mock_state_kwargs):
        state = TrainerState(**mock_state_kwargs)
        expected = 100 * 10  # len(train_loader) * config.epochs
        assert state.total_train_steps == expected

    def test_base_lr_scales_from_optimizer(self, mock_state_kwargs):
        state = TrainerState(**mock_state_kwargs)
        assert "backbone" in state._base_lr_scales
        assert state._base_lr_scales["backbone"] == 1.0

    def test_reset_phase_state(self, mock_state_kwargs):
        state = TrainerState(**mock_state_kwargs)
        state.representation_phase_active = True
        state.representation_curriculum_complete = True
        state.head_phase_start_step = 100
        state.joint_finetune_active = True
        state.cluster_centers = torch.zeros(5, 8)
        state.phase1_class_centroids = torch.zeros(5, 8)
        state.phase1_centroid_class_ids = [0, 1, 2]
        state.rep_epoch_feature_chunks = [torch.zeros(10, 8)]
        state.rep_epoch_label_chunks = [torch.zeros(10)]
        state.representation_snapshot_id = "test_snapshot"

        state.reset_phase_state()

        assert state.representation_phase_active is False
        assert state.representation_curriculum_complete is False
        assert state.head_phase_start_step == -1
        assert state.joint_finetune_active is False
        assert state.cluster_centers is None
        assert state.phase1_class_centroids is None
        assert state.phase1_centroid_class_ids == []
        assert state.rep_epoch_feature_chunks == []
        assert state.rep_epoch_label_chunks == []
        assert state.representation_snapshot_id is None
        assert state.step_coverage_checked is False


class TestTrainerStateDelegateConstruction:
    """TrainerState factory methods create delegates correctly."""

    def test_build_phase_manager(self, mock_state_kwargs):
        state = TrainerState(**mock_state_kwargs)
        pm = state.build_phase_manager()
        assert pm is not None
        assert hasattr(pm, "should_exit_curriculum_by_targets")
        assert hasattr(pm, "can_transition_to_head_phase")

    def test_build_early_stopping_manager(self, mock_state_kwargs):
        state = TrainerState(**mock_state_kwargs)
        esm = state.build_early_stopping_manager()
        assert esm is not None
        assert hasattr(esm, "update_early_stopping")

    def test_build_freeze_manager(self, mock_state_kwargs):
        state = TrainerState(**mock_state_kwargs)
        fm = state.build_freeze_manager()
        assert fm is not None
        assert hasattr(fm, "should_unfreeze")

    def test_build_evaluation_orchestrator(self, mock_state_kwargs):
        state = TrainerState(**mock_state_kwargs)
        eo = state.build_evaluation_orchestrator()
        assert eo is not None
        assert hasattr(eo, "evaluate_loader")

    def test_build_geometry_analyzer(self, mock_state_kwargs):
        state = TrainerState(**mock_state_kwargs)
        ga = state.build_geometry_analyzer()
        assert ga is not None

    def test_build_centroid_manager(self, mock_state_kwargs):
        state = TrainerState(**mock_state_kwargs)
        cm = state.build_centroid_manager()
        assert cm is not None

    def test_build_loss_registry(self, mock_state_kwargs):
        state = TrainerState(**mock_state_kwargs)
        lr = state.build_loss_registry()
        assert lr is not None
        assert hasattr(lr, "supervised_contrastive_loss")

    def test_build_phase_orchestrator(self, mock_state_kwargs):
        state = TrainerState(**mock_state_kwargs)
        pm = state.build_phase_manager()
        esm = state.build_early_stopping_manager()
        ga = state.build_geometry_analyzer()
        ca = state.build_cluster_analyzer()
        cm = state.build_centroid_manager()
        rd = state.build_rep_diagnostics()
        po = state.build_phase_orchestrator(
            phase_manager=pm,
            early_stopping_manager=esm,
            geometry_analyzer=ga,
            cluster_analyzer=ca,
            centroid_manager=cm,
            rep_diagnostics=rd,
        )
        assert po is not None
        assert hasattr(po, "should_exit_representation_curriculum")

    def test_build_batch_processor(self, mock_state_kwargs):
        state = TrainerState(**mock_state_kwargs)
        lr = state.build_loss_registry()
        bp = state.build_batch_processor(loss_registry=lr)
        assert bp is not None

    def test_build_warmup_manager(self, mock_state_kwargs):
        state = TrainerState(**mock_state_kwargs)
        wm = state.build_warmup_manager()
        assert wm is not None

    def test_build_epoch_runner(self, mock_state_kwargs):
        state = TrainerState(**mock_state_kwargs)
        lr = state.build_loss_registry()
        bp = state.build_batch_processor(loss_registry=lr)
        wm = state.build_warmup_manager()
        er = state.build_epoch_runner(batch_processor=bp, warmup_manager=wm)
        assert er is not None

    def test_build_training_orchestrator(self, mock_state_kwargs):
        state = TrainerState(**mock_state_kwargs)
        lr = state.build_loss_registry()
        bp = state.build_batch_processor(loss_registry=lr)
        wm = state.build_warmup_manager()
        er = state.build_epoch_runner(batch_processor=bp, warmup_manager=wm)
        to = state.build_training_orchestrator(epoch_runner=er)
        assert to is not None
