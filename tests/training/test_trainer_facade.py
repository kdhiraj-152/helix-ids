"""
Regression tests for TrainerFacade: delegate construction, wiring,
and property accessor integrity.

Phase 18: Verifies TrainerFacade.build() creates all delegates correctly
and property accessors return the expected instances.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from scripts.training.core.trainer_facade import TrainerFacade
from scripts.training.core.trainer_state import TrainerState


@pytest.fixture
def mock_state():
    """Minimal TrainerState for facade testing."""
    config = MagicMock()
    config.epochs = 10
    config.learning_rate = 1e-3
    config.freeze_backbone_epochs = 0
    config.unfreeze_backbone_step = 0

    optimizer = MagicMock()
    optimizer.param_groups = [{"group_name": "backbone", "lr_scale": 1.0}]

    model = MagicMock()
    model.backbone.parameters.return_value = []

    train_loader = MagicMock()
    train_loader.__len__.return_value = 100

    state = MagicMock(spec=TrainerState)
    state.model = model
    state.train_loader = train_loader
    state.val_loaders = {"val": MagicMock()}
    state.test_loaders = {"test": MagicMock()}
    state.optimizer = optimizer
    state.loss_fn = MagicMock()
    state.config = config
    state.device = "cpu"
    state.logger = logging.getLogger("test_facade")

    # State defaults
    state.representation_only_steps = 0
    state.head_only_steps = 0
    state.representation_diagnostic_mode = False
    state.use_energy_based_family_objective = True
    state.rep_adaptive_exit_ratio_threshold = 1.6
    state.rep_adaptive_exit_min_inter_threshold = 0.30
    state.representation_window_pattern = []
    state.joint_finetune_backbone_lr_multiplier = 0.25
    state.joint_finetune_head_lr_multiplier = 0.15
    state.entropy_warmup_steps = 0
    state.entropy_warmup_weight = 0.0
    state.kl_uniform_weight = 0.0
    state.warmup_kl_uniform_weight = 0.0
    state.logit_floor = -2.0
    state.logit_floor_weight = 0.0
    state.tail_ce_weight = 0.0
    state.tail_class_mask = None
    state.loss_fn = MagicMock()
    state.energy_gap_weight = 1.0
    state.energy_multi_negative_alpha = 1.0
    state.energy_balance_weight = 0.1
    state.energy_winner_weight = 0.5
    state.energy_winner_min_count = 1
    state.energy_logit_temperature = 2.0
    state.energy_win_rate_ema_momentum = 0.9
    state.energy_emergence_bias_eps = 1e-3
    state.energy_emergence_bias_beta = 0.5
    state.energy_emergence_bias_ratio_max = 0.30
    state.geometry_min_cluster_size = 100
    state.geometry_min_inter_threshold = 0.2
    state.geometry_max_intra_inter_ratio_warmup = 2.5
    state.geometry_max_intra_inter_ratio_post_phase = 1.2
    state.geometry_max_intra_inter_ratio = 1.2
    state.critical_collision_pairs = {(0, 3), (0, 4), (3, 4)}
    state.emergency_label_merge_map = {3: 0, 4: 0}
    state.cluster_relabeling_enabled = False
    state.cluster_relabel_k = None
    state.cluster_relabel_seed = 42
    state.cluster_relabel_objective = "kmeans"
    state.cluster_relabel_spectral_affinity = "nearest_neighbors"
    state.disable_integrity_hard_stops = False
    state.centroid_ema_momentum = 0.9
    state.sampler_mode = "interleaved_rr"
    state._base_lr_scales = {"backbone": 1.0}
    state.run_seed = 42
    state.total_train_steps = 1000
    state.warmup_steps = 0
    state.train_temperature = 1.0
    state.logger = logging.getLogger("test_facade")
    state.binary_class_weights = None
    state.family_class_weights = None
    state.family_log_prior = None
    state.active_family_class_ids = set()
    state.class4_logit_shift = 0.0
    state.class4_logit_shift_class_id = 4
    state.backbone_frozen = False
    state.coverage_check_after_head_steps = 50
    state.rep_backbone_grad_scale = 2.0
    state.rep_var_lower_bound = 0.08
    state.rep_var_upper_bound = 0.12
    state.rep_var_clamp_weight = 0.05
    state.rep_pair_margin_distance = 1.2
    state.rep_pair_margin_weight = 0.15
    state.rep_hard_negative_weight = 3.0
    state.rep_supcon_weight = 0.2
    state.rep_supcon_temperature = 0.03
    state.rep_supcon_negative_weight = 1.5
    state.rep_supcon_min_negatives = 10
    state.rep_centroid_barrier_min_distance = 0.4
    state.rep_centroid_barrier_weight = 0.5
    state.rep_centroid_repulsion_margin = 0.6
    state.rep_centroid_repulsion_weight = 0.6
    state.rep_critical_pair_weight = 0.0
    state.rep_barrier_activation_fraction = 0.30
    state.rep_expansion_target_min_inter = 0.45
    state.rep_compression_supcon_scale = 0.3
    state.rep_topk_nearest_negatives = 3
    state.rep_min_displacement_eps = 0.05
    state.min_family_minority_recall_for_best = 0.3

    return state


class TestTrainerFacadeBuild:
    """TrainerFacade.build() constructs all delegates."""

    def test_build_returns_self(self, mock_state):
        facade = TrainerFacade(mock_state)
        result = facade.build()
        assert result is facade

    def test_build_creates_phase_manager(self, mock_state):
        facade = TrainerFacade(mock_state).build()
        assert facade.phase_manager is not None
        assert hasattr(facade.phase_manager, "should_exit_curriculum_by_targets")

    def test_build_creates_early_stopping_manager(self, mock_state):
        facade = TrainerFacade(mock_state).build()
        assert facade.early_stopping_manager is not None
        assert hasattr(facade.early_stopping_manager, "update_early_stopping")

    def test_build_creates_evaluation_orchestrator(self, mock_state):
        facade = TrainerFacade(mock_state).build()
        assert facade.evaluation_orchestrator is not None
        assert hasattr(facade.evaluation_orchestrator, "evaluate_loader")

    def test_build_creates_validation_orchestrator(self, mock_state):
        facade = TrainerFacade(mock_state).build()
        assert facade.validation_orchestrator is not None

    def test_build_creates_geometry_analyzer(self, mock_state):
        facade = TrainerFacade(mock_state).build()
        assert facade.geometry_analyzer is not None

    def test_build_creates_cluster_analyzer(self, mock_state):
        facade = TrainerFacade(mock_state).build()
        assert facade.cluster_analyzer is not None

    def test_build_creates_centroid_manager(self, mock_state):
        facade = TrainerFacade(mock_state).build()
        assert facade.centroid_manager is not None

    def test_build_creates_phase_orchestrator(self, mock_state):
        facade = TrainerFacade(mock_state).build()
        assert facade.phase_orchestrator is not None
        assert hasattr(facade.phase_orchestrator, "should_exit_representation_curriculum")

    def test_build_creates_loss_registry(self, mock_state):
        facade = TrainerFacade(mock_state).build()
        assert facade.loss_registry is not None
        assert hasattr(facade.loss_registry, "supervised_contrastive_loss")

    def test_build_creates_batch_processor(self, mock_state):
        facade = TrainerFacade(mock_state).build()
        assert facade.batch_processor is not None

    def test_build_creates_epoch_runner(self, mock_state):
        facade = TrainerFacade(mock_state).build()
        assert facade.epoch_runner is not None

    def test_build_creates_training_orchestrator(self, mock_state):
        facade = TrainerFacade(mock_state).build()
        assert facade.training_orchestrator is not None

    def test_build_creates_recovery_manager(self, mock_state):
        facade = TrainerFacade(mock_state).build()
        assert facade.recovery_manager is not None
        assert hasattr(facade.recovery_manager, "configure_structure_recovery")

    def test_build_all_delegates_unique(self, mock_state):
        facade = TrainerFacade(mock_state).build()
        delegates = [
            facade.phase_manager,
            facade.early_stopping_manager,
            facade.freeze_manager,
            facade.lr_scheduler,
            facade.evaluation_orchestrator,
            facade.validation_orchestrator,
            facade.geometry_analyzer,
            facade.cluster_analyzer,
            facade.rep_diagnostics,
            facade.centroid_manager,
            facade.phase_orchestrator,
            facade.representation_coordinator,
            facade.loss_registry,
            facade.batch_processor,
            facade.warmup_manager,
            facade.epoch_runner,
            facade.training_orchestrator,
            facade.recovery_manager,
        ]
        assert len({id(d) for d in delegates}) == len(delegates)

    def test_build_twice_raises(self, mock_state):
        """Note: build() does not guard against double-call; verify it works."""
        facade = TrainerFacade(mock_state).build()
        # Calling build() again overwrites delegates — this is current behavior.
        facade.build()
        assert facade.recovery_manager is not None


class TestTrainerFacadePropertyAccessors:
    """Property accessors throw helpful errors before build()."""

    def test_access_before_build_raises(self, mock_state):
        facade = TrainerFacade(mock_state)
        with pytest.raises(AssertionError):
            _ = facade.phase_manager

    def test_access_after_build_succeeds(self, mock_state):
        facade = TrainerFacade(mock_state).build()
        assert facade.phase_manager is not None

    def test_all_properties_accessible(self, mock_state):
        facade = TrainerFacade(mock_state).build()
        props = [
            "phase_manager",
            "early_stopping_manager",
            "freeze_manager",
            "lr_scheduler",
            "evaluation_orchestrator",
            "validation_orchestrator",
            "geometry_analyzer",
            "cluster_analyzer",
            "rep_diagnostics",
            "centroid_manager",
            "phase_orchestrator",
            "representation_coordinator",
            "loss_registry",
            "batch_processor",
            "warmup_manager",
            "epoch_runner",
            "training_orchestrator",
            "recovery_manager",
        ]
        for prop in props:
            assert getattr(facade, prop) is not None, f"Property {prop} returned None"
