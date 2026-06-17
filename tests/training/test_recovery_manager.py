"""
Regression tests for RecoveryManager lifecycle, settings application,
and edge cases.

Phase 18: Verifies configure_structure_recovery correctly applies settings
to TrainerState, validates inputs, resets delegate state, and handles
edge cases.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from scripts.training.core.recovery_manager import RecoveryManager
from scripts.training.core.trainer_state import TrainerState


@pytest.fixture
def mock_state():
    """State with sensible defaults for recovery testing."""
    config = MagicMock()
    config.epochs = 50

    state = MagicMock(spec=TrainerState)
    state.active_family_class_ids = set()
    state.supcon_weight = 0.0
    state.supcon_temperature = 0.2
    state.rep_supcon_weight = 0.0
    state.rep_supcon_temperature = 0.03
    state.step_coverage_check_step = 50
    state.representation_only_steps = 0
    state.head_only_steps = 0
    state.representation_diagnostic_mode = False
    state.joint_finetune_backbone_lr_multiplier = 0.25
    state.joint_finetune_head_lr_multiplier = 0.15
    state.coverage_check_after_head_steps = 50
    state.cluster_relabeling_enabled = False
    state.cluster_relabel_k = None
    state.cluster_relabel_seed = 42
    state.cluster_relabel_objective = "kmeans"
    state.cluster_relabel_spectral_affinity = "nearest_neighbors"
    state.geometry_max_intra_inter_ratio_warmup = 2.5
    state.geometry_max_intra_inter_ratio_post_phase = 1.2
    state.geometry_max_intra_inter_ratio = 1.2
    state.enforce_all_classes_per_batch = False
    state.sampler_mode = "interleaved_rr"
    state.train_temperature = 1.0
    state.config = config

    # Energy defaults
    state.use_energy_based_family_objective = True
    state.energy_gap_margin = 1.0
    state.energy_gap_weight = 1.0
    state.energy_multi_negative_alpha = 1.0
    state.energy_logit_temperature = 2.0
    state.energy_balance_weight = 0.1
    state.energy_winner_weight = 0.5
    state.energy_winner_min_count = 1
    state.energy_emergence_bias_beta = 0.5
    state.energy_emergence_bias_eps = 1e-3
    state.energy_win_rate_ema_momentum = 0.9
    state.energy_emergence_bias_ratio_min = 0.10
    state.energy_emergence_bias_ratio_max = 0.30
    state.energy_emergence_bias_target_ratio = 0.20
    state.energy_isolate_short_horizon = True

    # Rep defaults
    state.rep_adaptive_exit_ratio_threshold = 1.6
    state.rep_adaptive_exit_min_inter_threshold = 0.30
    state.rep_backbone_grad_scale = 2.0

    # Collection state
    state.rep_epoch_feature_chunks = []
    state.rep_epoch_label_chunks = []
    state.representation_snapshot_id = None
    state.cluster_centers = None
    state.phase1_class_centroids = None
    state.phase1_centroid_class_ids = []
    state.representation_window_pattern = []
    state.step_coverage_checked = True

    return state


@pytest.fixture
def recovery_manager(mock_state):
    """RecoveryManager with mocked dependencies."""
    loss_registry = MagicMock()
    centroid_manager = MagicMock()
    centroid_manager._centroid_ema_state = {}
    centroid_manager._epoch_frozen_centroids = {}
    logger = logging.getLogger("test_recovery")
    return RecoveryManager(
        state=mock_state,
        loss_registry=loss_registry,
        centroid_manager=centroid_manager,
        logger=logger,
    )


_phase_settings_minimal = {
    "representation_only_steps": 100,
    "head_only_steps": 50,
}


class TestRecoveryManagerConfigureStructureRecovery:
    """configure_structure_recovery applies settings correctly."""

    def test_core_settings_applied(self, recovery_manager, mock_state):
        recovery_manager.configure_structure_recovery(
            active_family_classes={0, 1, 2, 3, 4},
            supcon_weight=0.5,
            supcon_temperature=0.03,
            step_coverage_check_step=100,
            representation_diagnostic_mode=True,
            phase_settings=_phase_settings_minimal,
            cluster_relabeling_enabled=False,
            cluster_relabel_k=None,
            cluster_relabel_seed=42,
            cluster_relabel_objective="kmeans",
            cluster_relabel_spectral_affinity="nearest_neighbors",
        )
        assert mock_state.active_family_class_ids == {0, 1, 2, 3, 4}
        assert mock_state.supcon_weight == 0.5
        assert mock_state.supcon_temperature >= 1e-3
        assert mock_state.step_coverage_check_step == 100
        assert mock_state.step_coverage_checked is False
        assert mock_state.representation_diagnostic_mode is True

    def test_phase_steps_configured(self, recovery_manager, mock_state):
        recovery_manager.configure_structure_recovery(
            active_family_classes={0},
            supcon_weight=0.0,
            supcon_temperature=0.2,
            step_coverage_check_step=50,
            representation_diagnostic_mode=False,
            phase_settings={
                "representation_only_steps": 200,
                "head_only_steps": 100,
                # Window pattern must pass 200 as the only window step
                "representation_micro_cycle_steps": [200],
            },
            cluster_relabeling_enabled=False,
            cluster_relabel_k=None,
            cluster_relabel_seed=42,
            cluster_relabel_objective="kmeans",
            cluster_relabel_spectral_affinity="nearest_neighbors",
        )
        # representation_only_steps is overwritten by sum of window steps
        assert mock_state.representation_only_steps == 200
        assert mock_state.head_only_steps == 100

    def test_joint_finetune_multipliers(self, recovery_manager, mock_state):
        recovery_manager.configure_structure_recovery(
            active_family_classes={0},
            supcon_weight=0.0,
            supcon_temperature=0.2,
            step_coverage_check_step=50,
            representation_diagnostic_mode=False,
            phase_settings={**_phase_settings_minimal,
                           "joint_finetune_backbone_lr_multiplier": 0.5,
                           "joint_finetune_head_lr_multiplier": 0.3},
            cluster_relabeling_enabled=False,
            cluster_relabel_k=None,
            cluster_relabel_seed=42,
            cluster_relabel_objective="kmeans",
            cluster_relabel_spectral_affinity="nearest_neighbors",
        )
        assert mock_state.joint_finetune_backbone_lr_multiplier == 0.5
        assert mock_state.joint_finetune_head_lr_multiplier == 0.3

    def test_multipliers_floor_at_1e3(self, recovery_manager, mock_state):
        recovery_manager.configure_structure_recovery(
            active_family_classes={0},
            supcon_weight=0.0,
            supcon_temperature=0.2,
            step_coverage_check_step=50,
            representation_diagnostic_mode=False,
            phase_settings={**_phase_settings_minimal,
                           "joint_finetune_backbone_lr_multiplier": 0.0,
                           "joint_finetune_head_lr_multiplier": -1.0},
            cluster_relabeling_enabled=False,
            cluster_relabel_k=None,
            cluster_relabel_seed=42,
            cluster_relabel_objective="kmeans",
            cluster_relabel_spectral_affinity="nearest_neighbors",
        )
        assert mock_state.joint_finetune_backbone_lr_multiplier >= 1e-3
        assert mock_state.joint_finetune_head_lr_multiplier >= 1e-3

    def test_phase_state_reset_on_configure(self, recovery_manager, mock_state):
        mock_state.reset_phase_state = MagicMock()
        recovery_manager.configure_structure_recovery(
            active_family_classes={0},
            supcon_weight=0.0,
            supcon_temperature=0.2,
            step_coverage_check_step=50,
            representation_diagnostic_mode=False,
            phase_settings=_phase_settings_minimal,
            cluster_relabeling_enabled=False,
            cluster_relabel_k=None,
            cluster_relabel_seed=42,
            cluster_relabel_objective="kmeans",
            cluster_relabel_spectral_affinity="nearest_neighbors",
        )
        mock_state.reset_phase_state.assert_called_once()

    def test_energy_settings(self, recovery_manager, mock_state):
        recovery_manager.configure_structure_recovery(
            active_family_classes={0},
            supcon_weight=0.0,
            supcon_temperature=0.2,
            step_coverage_check_step=50,
            representation_diagnostic_mode=False,
            phase_settings={**_phase_settings_minimal,
                           "energy_gap_margin": 2.0,
                           "energy_gap_weight": 1.5,
                           "energy_balance_weight": 0.3},
            cluster_relabeling_enabled=False,
            cluster_relabel_k=None,
            cluster_relabel_seed=42,
            cluster_relabel_objective="kmeans",
            cluster_relabel_spectral_affinity="nearest_neighbors",
        )
        assert mock_state.energy_gap_margin == 2.0
        assert mock_state.energy_gap_weight == 1.5
        assert mock_state.energy_balance_weight == 0.3

    def test_window_pattern_built(self, recovery_manager, mock_state):
        recovery_manager.configure_structure_recovery(
            active_family_classes={0},
            supcon_weight=0.0,
            supcon_temperature=0.2,
            step_coverage_check_step=50,
            representation_diagnostic_mode=False,
            phase_settings={
                "representation_micro_cycle_steps": [30, 15, 30, 15, 30],
                **_phase_settings_minimal,
            },
            cluster_relabeling_enabled=False,
            cluster_relabel_k=None,
            cluster_relabel_seed=42,
            cluster_relabel_objective="kmeans",
            cluster_relabel_spectral_affinity="nearest_neighbors",
        )
        assert len(mock_state.representation_window_pattern) == 5
        assert mock_state.representation_window_pattern[0] == (True, 30)
        assert mock_state.representation_window_pattern[1] == (False, 15)

    def test_sampler_mode_applied(self, recovery_manager, mock_state):
        recovery_manager.configure_structure_recovery(
            active_family_classes={0},
            supcon_weight=0.0,
            supcon_temperature=0.2,
            step_coverage_check_step=50,
            representation_diagnostic_mode=False,
            phase_settings={**_phase_settings_minimal,
                           "sampler_mode": "balanced"},
            cluster_relabeling_enabled=False,
            cluster_relabel_k=None,
            cluster_relabel_seed=42,
            cluster_relabel_objective="kmeans",
            cluster_relabel_spectral_affinity="nearest_neighbors",
        )
        assert mock_state.sampler_mode == "balanced"


class TestRecoveryManagerClusterRelabel:
    """Cluster relabel settings validation."""

    def test_valid_objective(self, recovery_manager, mock_state):
        for obj in ["kmeans", "gmm", "spectral"]:
            recovery_manager.configure_structure_recovery(
                active_family_classes={0},
                supcon_weight=0.0,
                supcon_temperature=0.2,
                step_coverage_check_step=50,
                representation_diagnostic_mode=False,
                phase_settings=_phase_settings_minimal,
                cluster_relabeling_enabled=True,
                cluster_relabel_k=5,
                cluster_relabel_seed=42,
                cluster_relabel_objective=obj,
                cluster_relabel_spectral_affinity="nearest_neighbors",
            )
            assert mock_state.cluster_relabel_objective == obj

    def test_invalid_objective_raises(self, recovery_manager, mock_state):
        with pytest.raises(ValueError, match="Unsupported cluster relabel objective"):
            recovery_manager.configure_structure_recovery(
                active_family_classes={0},
                supcon_weight=0.0,
                supcon_temperature=0.2,
                step_coverage_check_step=50,
                representation_diagnostic_mode=False,
                phase_settings=_phase_settings_minimal,
                cluster_relabeling_enabled=True,
                cluster_relabel_k=5,
                cluster_relabel_seed=42,
                cluster_relabel_objective="invalid_obj",
                cluster_relabel_spectral_affinity="nearest_neighbors",
            )

    def test_invalid_spectral_affinity_raises(self, recovery_manager, mock_state):
        with pytest.raises(ValueError, match="Unsupported spectral affinity"):
            recovery_manager.configure_structure_recovery(
                active_family_classes={0},
                supcon_weight=0.0,
                supcon_temperature=0.2,
                step_coverage_check_step=50,
                representation_diagnostic_mode=False,
                phase_settings=_phase_settings_minimal,
                cluster_relabeling_enabled=True,
                cluster_relabel_k=5,
                cluster_relabel_seed=42,
                cluster_relabel_objective="kmeans",
                cluster_relabel_spectral_affinity="invalid_aff",
            )

    def test_cluster_k_minimum(self, recovery_manager, mock_state):
        recovery_manager.configure_structure_recovery(
            active_family_classes={0},
            supcon_weight=0.0,
            supcon_temperature=0.2,
            step_coverage_check_step=50,
            representation_diagnostic_mode=False,
            phase_settings=_phase_settings_minimal,
            cluster_relabeling_enabled=True,
            cluster_relabel_k=1,  # below minimum of 2
            cluster_relabel_seed=42,
            cluster_relabel_objective="kmeans",
            cluster_relabel_spectral_affinity="nearest_neighbors",
        )
        assert mock_state.cluster_relabel_k >= 2


class TestRecoveryManagerEdgeCases:
    """Edge cases in recovery configuration."""

    def test_empty_active_family_classes(self, recovery_manager, mock_state):
        recovery_manager.configure_structure_recovery(
            active_family_classes=set(),
            supcon_weight=0.0,
            supcon_temperature=0.2,
            step_coverage_check_step=50,
            representation_diagnostic_mode=False,
            phase_settings=_phase_settings_minimal,
            cluster_relabeling_enabled=False,
            cluster_relabel_k=None,
            cluster_relabel_seed=42,
            cluster_relabel_objective="kmeans",
            cluster_relabel_spectral_affinity="nearest_neighbors",
        )
        assert mock_state.active_family_class_ids == set()

    def test_negative_supcon_weight_clamped(self, recovery_manager, mock_state):
        recovery_manager.configure_structure_recovery(
            active_family_classes={0},
            supcon_weight=-1.0,
            supcon_temperature=0.2,
            step_coverage_check_step=50,
            representation_diagnostic_mode=False,
            phase_settings=_phase_settings_minimal,
            cluster_relabeling_enabled=False,
            cluster_relabel_k=None,
            cluster_relabel_seed=42,
            cluster_relabel_objective="kmeans",
            cluster_relabel_spectral_affinity="nearest_neighbors",
        )
        assert mock_state.supcon_weight == 0.0

    def test_zero_temperature_clamped(self, recovery_manager, mock_state):
        recovery_manager.configure_structure_recovery(
            active_family_classes={0},
            supcon_weight=0.5,
            supcon_temperature=0.0,
            step_coverage_check_step=50,
            representation_diagnostic_mode=False,
            phase_settings=_phase_settings_minimal,
            cluster_relabeling_enabled=False,
            cluster_relabel_k=None,
            cluster_relabel_seed=42,
            cluster_relabel_objective="kmeans",
            cluster_relabel_spectral_affinity="nearest_neighbors",
        )
        assert mock_state.supcon_temperature >= 1e-3
