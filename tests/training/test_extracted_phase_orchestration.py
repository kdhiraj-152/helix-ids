"""Regression tests for PhaseOrchestrator (Phase 15 extraction).

Covers:
    - PhaseOrchestrator construction and property exposure
    - handle_representation_phase_logic (full lifecycle)
    - should_exit_representation_curriculum (adaptive geometry exit)
    - maybe_activate_joint_finetune_phase (activation conditions)
    - update_early_stopping (best model and stop decisions)
    - hard_stop_reason delegation (decision tree routing)
    - hard_stop_val_gap_collapse delegation
    - hard_stop_high_accuracy_high_loss delegation
    - hard_stop_entropy_collapse delegation
    - State synchronization contract
    - Smoke mode interactions
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from scripts.training.scheduler import (
    EarlyStoppingManager,
    PhaseManager,
)
from scripts.training.scheduler.phase_orchestrator import PhaseOrchestrator

# =========================================================================== #
#  Fixtures
# =========================================================================== #


@pytest.fixture
def mock_model() -> MagicMock:
    """A mock model whose state_dict returns a small dict."""
    model = MagicMock()
    model.state_dict.return_value = {
        "backbone.weight": torch.zeros(4, 4),
        "head.bias": torch.ones(4),
    }
    return model


@pytest.fixture
def mock_optimizer() -> MagicMock:
    """A mock optimizer with param_groups."""
    opt = MagicMock()
    opt.param_groups = [
        {"group_name": "backbone", "lr_scale": 1.0},
        {"group_name": "family_head", "lr_scale": 1.0},
        {"group_name": "binary_head", "lr_scale": 1.0},
    ]
    return opt


@pytest.fixture
def mock_logger() -> logging.Logger:
    return logging.getLogger("test_orchestrator")


@pytest.fixture
def mock_geometry_analyzer() -> MagicMock:
    """A minimal mock for geometry_analyzer."""
    g = MagicMock()
    g.enforce_geometry_integrity.return_value = {}
    g.has_critical_collision_pairs.return_value = set()
    return g


@pytest.fixture
def mock_cluster_analyzer() -> MagicMock:
    c = MagicMock()
    c.compute_class_centroids.return_value = (torch.randn(3, 64), [0, 1, 2])
    c.fit_embedding_clusters.return_value = {"cluster_labels": torch.zeros(10, dtype=torch.int64)}
    return c


@pytest.fixture
def mock_centroid_manager() -> MagicMock:
    cm = MagicMock()
    cm.stabilize_centroids.side_effect = lambda features, ids: features
    return cm


@pytest.fixture
def mock_rep_diagnostics() -> MagicMock:
    rd = MagicMock()
    rd.run_representation_diagnostics.return_value = {
        "intra_inter_ratio": 2.0,
        "min_inter_center_distance": 0.5,
        "class_centroids": torch.randn(3, 64),
        "centroid_class_ids": [0, 1, 2],
    }
    rd.build_representation_snapshot_id.return_value = "snapshot_test_001"
    rd.prepare_representation_features.return_value = (torch.randn(10, 64), torch.randint(0, 3, (10,)))
    return rd


def _make_orchestrator(
    mock_model: MagicMock,
    mock_optimizer: MagicMock,
    mock_logger: logging.Logger,
    mock_geometry_analyzer: MagicMock,
    mock_cluster_analyzer: MagicMock,
    mock_centroid_manager: MagicMock,
    mock_rep_diagnostics: MagicMock,
    **overrides: Any,
) -> PhaseOrchestrator:
    """Helper to build a PhaseOrchestrator with test defaults."""

    phase_manager = PhaseManager(
        representation_only_steps=100,
        head_only_steps=200,
        representation_diagnostic_mode=True,
        use_energy_based_family_objective=False,
        rep_adaptive_exit_ratio_threshold=1.5,
        rep_adaptive_exit_min_inter_threshold=0.25,
        representation_window_pattern=None,
    )

    early_stopping_manager = EarlyStoppingManager(
        early_stopping_patience=3,
        early_stopping_threshold=0.02,
        min_family_minority_recall_for_best=0.15,
        disable_integrity_hard_stops=False,
    )

    defaults: dict[str, Any] = {
        "model": mock_model,
        "optimizer": mock_optimizer,
        "logger": mock_logger,
        "base_lr_scales": {"backbone": 1.0, "family_head": 1.0, "binary_head": 1.0},
        "phase_manager": phase_manager,
        "early_stopping_manager": early_stopping_manager,
        "geometry_analyzer": mock_geometry_analyzer,
        "cluster_analyzer": mock_cluster_analyzer,
        "centroid_manager": mock_centroid_manager,
        "rep_diagnostics": mock_rep_diagnostics,
        # Config values
        "representation_only_steps": 100,
        "head_only_steps": 200,
        "representation_diagnostic_mode": True,
        "use_energy_based_family_objective": False,
        "rep_adaptive_exit_ratio_threshold": 1.5,
        "rep_adaptive_exit_min_inter_threshold": 0.25,
        "joint_finetune_backbone_lr_multiplier": 0.25,
        "joint_finetune_head_lr_multiplier": 0.15,
        "cluster_relabeling_enabled": True,
        "cluster_relabel_k": 10,
        "cluster_relabel_seed": 42,
        "cluster_relabel_objective": "kmeans",
        "cluster_relabel_spectral_affinity": "nearest_neighbors",
        "critical_collision_pairs": set(),
        "emergency_label_merge_map": {},
        "disable_integrity_hard_stops": False,
        "min_family_minority_recall_for_best": 0.15,
        "quality_gate_entropy": 0.3,
    }
    defaults.update(overrides)
    return PhaseOrchestrator(**defaults)


# =========================================================================== #
#  Construction and property exposure
# =========================================================================== #


class TestPhaseOrchestratorConstruction:
    """PhaseOrchestrator construction and property delegation."""

    def test_constructs_with_defaults(
        self,
        mock_model: MagicMock,
        mock_optimizer: MagicMock,
        mock_logger: logging.Logger,
        mock_geometry_analyzer: MagicMock,
        mock_cluster_analyzer: MagicMock,
        mock_centroid_manager: MagicMock,
        mock_rep_diagnostics: MagicMock,
    ) -> None:
        orch = _make_orchestrator(
            mock_model, mock_optimizer, mock_logger,
            mock_geometry_analyzer, mock_cluster_analyzer,
            mock_centroid_manager, mock_rep_diagnostics,
        )
        assert isinstance(orch.phase_manager, PhaseManager)
        assert isinstance(orch.early_stopping_manager, EarlyStoppingManager)

    def test_phase_manager_property(
        self,
        mock_model: MagicMock, mock_optimizer: MagicMock, mock_logger: logging.Logger,
        mock_geometry_analyzer: MagicMock, mock_cluster_analyzer: MagicMock,
        mock_centroid_manager: MagicMock, mock_rep_diagnostics: MagicMock,
    ) -> None:
        orch = _make_orchestrator(
            mock_model, mock_optimizer, mock_logger,
            mock_geometry_analyzer, mock_cluster_analyzer,
            mock_centroid_manager, mock_rep_diagnostics,
        )
        assert orch.phase_manager is orch._phase_manager

    def test_early_stopping_property(
        self,
        mock_model: MagicMock, mock_optimizer: MagicMock, mock_logger: logging.Logger,
        mock_geometry_analyzer: MagicMock, mock_cluster_analyzer: MagicMock,
        mock_centroid_manager: MagicMock, mock_rep_diagnostics: MagicMock,
    ) -> None:
        orch = _make_orchestrator(
            mock_model, mock_optimizer, mock_logger,
            mock_geometry_analyzer, mock_cluster_analyzer,
            mock_centroid_manager, mock_rep_diagnostics,
        )
        assert orch.early_stopping_manager is orch._early_stopping_manager

    def test_is_smoke_mode_from_gov_profile(
        self,
        mock_model: MagicMock, mock_optimizer: MagicMock, mock_logger: logging.Logger,
        mock_geometry_analyzer: MagicMock, mock_cluster_analyzer: MagicMock,
        mock_centroid_manager: MagicMock, mock_rep_diagnostics: MagicMock,
    ) -> None:
        orch = _make_orchestrator(
            mock_model, mock_optimizer, mock_logger,
            mock_geometry_analyzer, mock_cluster_analyzer,
            mock_centroid_manager, mock_rep_diagnostics,
        )
        assert orch.is_smoke_mode(gov_profile="smoke")

    def test_is_smoke_mode_from_epochs(
        self,
        mock_model: MagicMock, mock_optimizer: MagicMock, mock_logger: logging.Logger,
        mock_geometry_analyzer: MagicMock, mock_cluster_analyzer: MagicMock,
        mock_centroid_manager: MagicMock, mock_rep_diagnostics: MagicMock,
    ) -> None:
        orch = _make_orchestrator(
            mock_model, mock_optimizer, mock_logger,
            mock_geometry_analyzer, mock_cluster_analyzer,
            mock_centroid_manager, mock_rep_diagnostics,
        )
        assert orch.is_smoke_mode(epochs=5)
        assert not orch.is_smoke_mode(epochs=50)

    def test_is_smoke_mode_default(
        self,
        mock_model: MagicMock, mock_optimizer: MagicMock, mock_logger: logging.Logger,
        mock_geometry_analyzer: MagicMock, mock_cluster_analyzer: MagicMock,
        mock_centroid_manager: MagicMock, mock_rep_diagnostics: MagicMock,
    ) -> None:
        orch = _make_orchestrator(
            mock_model, mock_optimizer, mock_logger,
            mock_geometry_analyzer, mock_cluster_analyzer,
            mock_centroid_manager, mock_rep_diagnostics,
        )
        assert not orch.is_smoke_mode()


# =========================================================================== #
#  handle_representation_phase_logic  — state dict contract
# =========================================================================== #


class TestHandleRepresentationPhaseLogic:
    """Orchestrator's main representation-phase coordinator."""

    @pytest.fixture
    def orch(
        self,
        mock_model: MagicMock, mock_optimizer: MagicMock, mock_logger: logging.Logger,
        mock_geometry_analyzer: MagicMock, mock_cluster_analyzer: MagicMock,
        mock_centroid_manager: MagicMock, mock_rep_diagnostics: MagicMock,
    ) -> PhaseOrchestrator:
        return _make_orchestrator(
            mock_model, mock_optimizer, mock_logger,
            mock_geometry_analyzer, mock_cluster_analyzer,
            mock_centroid_manager, mock_rep_diagnostics,
        )

    def test_returns_state_dict_with_default_keys(self, orch: PhaseOrchestrator) -> None:
        """The returned state dict always carries the standard keys."""
        result = orch.handle_representation_phase_logic(
            in_representation_phase=False,
            global_step=10,
            rep_phase_feature_chunks=[],
            rep_phase_label_chunks=[],
            val_loaders={},
            active_family_class_ids=set(),
            run_seed=42,
        )
        expected_keys = {
            "transition_executed",
            "phase1_class_centroids",
            "phase1_centroid_class_ids",
            "representation_snapshot_id",
            "representation_diagnostics",
            "rep_phase_feature_chunks",
            "rep_phase_label_chunks",
            "representation_phase_active",
            "representation_curriculum_complete",
            "in_representation_window",
        }
        assert expected_keys.issubset(result.keys())

    def test_returns_false_flags_when_not_diagnostic_mode(self, orch: PhaseOrchestrator) -> None:
        """Without diagnostic mode, the state dict indicates no activity."""
        orch._representation_diagnostic_mode = False
        result = orch.handle_representation_phase_logic(
            in_representation_phase=False,
            global_step=10,
            rep_phase_feature_chunks=[],
            rep_phase_label_chunks=[],
            val_loaders={},
            active_family_class_ids=set(),
            run_seed=42,
        )
        assert not result["transition_executed"]
        assert not result["representation_phase_active"]
        assert not result["representation_curriculum_complete"]

    def test_starts_representation_phase_on_first_call(self, orch: PhaseOrchestrator) -> None:
        """Before representation_only_steps, can_start_representation_phase returns True."""
        result = orch.handle_representation_phase_logic(
            in_representation_phase=False,
            global_step=50,
            rep_phase_feature_chunks=[],
            rep_phase_label_chunks=[],
            val_loaders={},
            active_family_class_ids=set(),
            run_seed=42,
        )
        assert result["representation_phase_active"]
        assert len(result["rep_phase_feature_chunks"]) == 0

    def test_skips_start_when_past_threshold(self, orch: PhaseOrchestrator) -> None:
        """After representation_only_steps, can_start_representation_phase returns False."""
        result = orch.handle_representation_phase_logic(
            in_representation_phase=False,
            global_step=150,
            rep_phase_feature_chunks=[],
            rep_phase_label_chunks=[],
            val_loaders={},
            active_family_class_ids=set(),
            run_seed=42,
        )
        assert not result["representation_phase_active"]

    def test_transition_not_executed_when_cannot_transition(self, orch: PhaseOrchestrator) -> None:
        """Without curriculum completion, the orchestrator does not execute a transition."""
        orch._phase_manager.apply_representation_phase_start()
        result = orch.handle_representation_phase_logic(
            in_representation_phase=True,
            global_step=70,
            rep_phase_feature_chunks=[torch.randn(4, 64), torch.randn(4, 64)],
            rep_phase_label_chunks=[torch.randint(0, 3, (4,)), torch.randint(0, 3, (4,))],
            val_loaders={},
            active_family_class_ids={0, 1, 2},
            run_seed=42,
        )
        assert not result["transition_executed"]

    def test_transition_fires_when_ready(self, orch: PhaseOrchestrator) -> None:
        """With active representation phase + curriculum complete, transition executes."""
        orch._phase_manager.apply_representation_phase_start()
        orch._phase_manager.apply_curriculum_complete()
        orch._phase_manager.representation_curriculum_complete = True
        # Patch the internal transition to avoid heavy deps
        transition_original = orch._execute_phase_transition
        try:
            orch._execute_phase_transition = MagicMock(  # type: ignore[method-assign]
                return_value={"executed": True}
            )
            result = orch.handle_representation_phase_logic(
                in_representation_phase=False,
                global_step=300,
                rep_phase_feature_chunks=[torch.randn(4, 64)],
                rep_phase_label_chunks=[torch.randint(0, 3, (4,))],
                val_loaders={},
                active_family_class_ids={0, 1, 2},
                run_seed=42,
            )
            assert len(result["rep_phase_feature_chunks"]) == 0
        finally:
            orch._execute_phase_transition = transition_original

    def test_curriculum_complete_forced_after_representation_only_steps(
        self,
        mock_model: MagicMock, mock_optimizer: MagicMock, mock_logger: logging.Logger,
        mock_geometry_analyzer: MagicMock, mock_cluster_analyzer: MagicMock,
        mock_centroid_manager: MagicMock, mock_rep_diagnostics: MagicMock,
    ) -> None:
        """When global_step >= representation_only_steps, curriculum is forced complete."""
        orch = _make_orchestrator(
            mock_model, mock_optimizer, mock_logger,
            mock_geometry_analyzer, mock_cluster_analyzer,
            mock_centroid_manager, mock_rep_diagnostics,
            representation_only_steps=50,
        )
        orch._phase_manager.apply_representation_phase_start()
        result = orch.handle_representation_phase_logic(
            in_representation_phase=True,
            global_step=60,
            rep_phase_feature_chunks=[],
            rep_phase_label_chunks=[],
            val_loaders={},
            active_family_class_ids=set(),
            run_seed=42,
        )
        assert result["representation_curriculum_complete"]

    def test_state_sync_integrity(self, orch: PhaseOrchestrator) -> None:
        """State dict flags mirror PhaseManager state after method call."""
        result = orch.handle_representation_phase_logic(
            in_representation_phase=False,
            global_step=50,
            rep_phase_feature_chunks=[],
            rep_phase_label_chunks=[],
            val_loaders={},
            active_family_class_ids=set(),
            run_seed=42,
        )
        assert result["representation_phase_active"] == orch._phase_manager.representation_phase_active


# =========================================================================== #
#  should_exit_representation_curriculum
# =========================================================================== #


class TestShouldExitRepresentationCurriculum:
    """Adaptive geometry curriculum-exit logic via orchestrator."""

    @pytest.fixture
    def orch(
        self,
        mock_model: MagicMock, mock_optimizer: MagicMock, mock_logger: logging.Logger,
        mock_geometry_analyzer: MagicMock, mock_cluster_analyzer: MagicMock,
        mock_centroid_manager: MagicMock, mock_rep_diagnostics: MagicMock,
    ) -> PhaseOrchestrator:
        return _make_orchestrator(
            mock_model, mock_optimizer, mock_logger,
            mock_geometry_analyzer, mock_cluster_analyzer,
            mock_centroid_manager, mock_rep_diagnostics,
        )

    def test_returns_no_exit_when_energy_objective(self, orch: PhaseOrchestrator) -> None:
        """Energy-based family objective prevents adaptive exit."""
        orch._use_energy_based_family_objective = True
        result = orch.should_exit_representation_curriculum(
            rep_phase_feature_chunks=[torch.randn(4, 64)],
            rep_phase_label_chunks=[torch.randint(0, 3, (4,))],
            global_step=50,
            val_loaders={},
            active_family_class_ids={0, 1, 2},
            run_seed=42,
        )
        assert not result["should_exit"]

    def test_no_exit_when_chunks_empty(self, orch: PhaseOrchestrator) -> None:
        """No feature chunks means no exit."""
        result = orch.should_exit_representation_curriculum(
            rep_phase_feature_chunks=[],
            rep_phase_label_chunks=[],
            global_step=50,
            val_loaders={},
            active_family_class_ids=set(),
            run_seed=42,
        )
        assert not result["should_exit"]

    def test_no_exit_when_single_class(self, orch: PhaseOrchestrator) -> None:
        """Single unique class cannot trigger curriculum exit."""
        labels = torch.zeros(8, dtype=torch.int64)
        result = orch.should_exit_representation_curriculum(
            rep_phase_feature_chunks=[torch.randn(8, 64)],
            rep_phase_label_chunks=[labels],
            global_step=50,
            val_loaders={},
            active_family_class_ids={0},
            run_seed=42,
        )
        assert not result["should_exit"]

    def test_diagnostics_update_in_result(self, orch: PhaseOrchestrator) -> None:
        """Even when exit is not triggered, the diagnostics update key is present."""
        result = orch.should_exit_representation_curriculum(
            rep_phase_feature_chunks=[],
            rep_phase_label_chunks=[],
            global_step=50,
            val_loaders={},
            active_family_class_ids=set(),
            run_seed=42,
        )
        assert "representation_diagnostics_update" in result
        assert isinstance(result["representation_diagnostics_update"], dict)


# =========================================================================== #
#  maybe_activate_joint_finetune_phase
# =========================================================================== #


class TestMaybeActivateJointFinetunePhase:
    """Joint finetune activation through the orchestrator."""

    @pytest.fixture
    def orch(
        self,
        mock_model: MagicMock, mock_optimizer: MagicMock, mock_logger: logging.Logger,
        mock_geometry_analyzer: MagicMock, mock_cluster_analyzer: MagicMock,
        mock_centroid_manager: MagicMock, mock_rep_diagnostics: MagicMock,
    ) -> PhaseOrchestrator:
        return _make_orchestrator(
            mock_model, mock_optimizer, mock_logger,
            mock_geometry_analyzer, mock_cluster_analyzer,
            mock_centroid_manager, mock_rep_diagnostics,
        )

    def test_cannot_activate_if_not_in_head_phase(self, orch: PhaseOrchestrator) -> None:
        """PhaseManager rejects activation if head phase hasn't started."""
        assert not orch.maybe_activate_joint_finetune_phase(
            global_step=50,
            representation_diagnostics={},
        )

    def test_cannot_activate_if_representation_diagnostic_mode_disabled(
        self,
        mock_model: MagicMock, mock_optimizer: MagicMock, mock_logger: logging.Logger,
        mock_geometry_analyzer: MagicMock, mock_cluster_analyzer: MagicMock,
        mock_centroid_manager: MagicMock, mock_rep_diagnostics: MagicMock,
    ) -> None:
        """Diagnostic mode must be enabled."""
        orch = _make_orchestrator(
            mock_model, mock_optimizer, mock_logger,
            mock_geometry_analyzer, mock_cluster_analyzer,
            mock_centroid_manager, mock_rep_diagnostics,
            representation_diagnostic_mode=False,
        )
        orch._phase_manager.apply_representation_phase_start()
        orch._phase_manager.apply_head_phase_start(global_step=50)
        assert not orch.maybe_activate_joint_finetune_phase(
            global_step=50,
            representation_diagnostics={},
        )

    def test_activates_when_eligible(self, orch: PhaseOrchestrator) -> None:
        """When head phase active and step >= joint_finetune_start_step, activation succeeds."""
        orch._phase_manager.apply_representation_phase_start()
        orch._phase_manager.apply_head_phase_start(global_step=50)
        # joint_finetune_start_step was set to 50 + 200 = 250 by apply_head_phase_start
        assert orch.maybe_activate_joint_finetune_phase(
            global_step=250,
            representation_diagnostics={},
        )
        assert orch._phase_manager.joint_finetune_active

    def test_cannot_activate_twice(self, orch: PhaseOrchestrator) -> None:
        """Once joint_finetune_active is True, further activation attempts return False."""
        orch._phase_manager.apply_representation_phase_start()
        orch._phase_manager.apply_head_phase_start(global_step=50)
        orch._phase_manager.joint_finetune_start_step = 50
        assert orch.maybe_activate_joint_finetune_phase(
            global_step=250,
            representation_diagnostics={},
        )
        assert not orch.maybe_activate_joint_finetune_phase(
            global_step=300,
            representation_diagnostics={},
        )


# =========================================================================== #
#  update_early_stopping  —  via orchestrator
# =========================================================================== #


class TestUpdateEarlyStopping:
    """Early-stopping state updates and best-model tracking through the orchestrator."""

    @pytest.fixture
    def orch(
        self,
        mock_model: MagicMock, mock_optimizer: MagicMock, mock_logger: logging.Logger,
        mock_geometry_analyzer: MagicMock, mock_cluster_analyzer: MagicMock,
        mock_centroid_manager: MagicMock, mock_rep_diagnostics: MagicMock,
    ) -> PhaseOrchestrator:
        return _make_orchestrator(
            mock_model, mock_optimizer, mock_logger,
            mock_geometry_analyzer, mock_cluster_analyzer,
            mock_centroid_manager, mock_rep_diagnostics,
        )

    def test_initial_call_marks_as_best(self, orch: PhaseOrchestrator) -> None:
        """First call with low loss should mark best and NOT stop."""
        val_metrics = {
            "val_loss": 0.5,
            "val_family_minority_recall_min": 0.20,
            "val_family_entropy": 0.50,
        }
        result = orch.update_early_stopping(val_metrics=val_metrics)
        assert result["is_best"]
        assert not result["should_stop"]
        assert result["best_val_loss"] == 0.5
        assert result["best_model_state"] is not None

    def test_worse_loss_does_not_trigger_best(self, orch: PhaseOrchestrator) -> None:
        """Second call with higher loss should not mark best."""
        orch._early_stopping_manager.best_val_loss = 0.5
        result = orch.update_early_stopping(val_metrics={"val_loss": 0.8})
        assert not result["is_best"]
        assert not result["should_stop"]

    def test_early_stopping_triggers_after_patience(self, orch: PhaseOrchestrator) -> None:
        """When patience is exhausted, should_stop returns True."""
        orch._early_stopping_manager.best_val_loss = 0.5
        for _ in range(2):
            orch._early_stopping_manager.patience_counter += 1
        result = orch.update_early_stopping(val_metrics={"val_loss": 0.8})
        assert result["should_stop"]

    def test_best_model_state_cloned(self, orch: PhaseOrchestrator) -> None:
        """best_model_state is a deep-detached clone, not a reference to model state."""
        val_metrics = {
            "val_loss": 0.3,
            "val_family_minority_recall_min": 0.20,
            "val_family_entropy": 0.50,
        }
        result = orch.update_early_stopping(val_metrics=val_metrics)
        best_state = result["best_model_state"]
        assert best_state is not None
        # Verify it's a cloned copy (not the same tensor object)
        orig_state = orch._model.state_dict()
        for key in best_state:
            assert best_state[key].data_ptr() != orig_state[key].data_ptr()


# =========================================================================== #
#  Hard-stop decision tree routing
# =========================================================================== #


class TestHardStopReason:
    """Hard-stop decision tree delegation through the orchestrator."""

    @pytest.fixture
    def orch(
        self,
        mock_model: MagicMock, mock_optimizer: MagicMock, mock_logger: logging.Logger,
        mock_geometry_analyzer: MagicMock, mock_cluster_analyzer: MagicMock,
        mock_centroid_manager: MagicMock, mock_rep_diagnostics: MagicMock,
    ) -> PhaseOrchestrator:
        return _make_orchestrator(
            mock_model, mock_optimizer, mock_logger,
            mock_geometry_analyzer, mock_cluster_analyzer,
            mock_centroid_manager, mock_rep_diagnostics,
        )

    def test_disabled_returns_none(self, orch: PhaseOrchestrator) -> None:
        """When hard stops are disabled, returns None."""
        orch._disable_integrity_hard_stops = True
        result = orch.hard_stop_reason(
            train_metrics={},
            val_metrics={},
            is_smoke=False,
            epoch=0,
        )
        assert result is None

    def test_val_gap_collapse(self, orch: PhaseOrchestrator) -> None:
        """Val-gap collapse detection is routed to ESM and returns reason."""
        train_metrics = {"train_calibrated_loss": 1.0}
        val_metrics = {
            "val_calibrated_loss": 0.5,
            "val_family_macro_f1": 0.1,
            "val_family_minority_recall_min": 0.05,
            "val_family_entropy": 0.05,
        }
        # First call: streak 1 (no trigger)
        result1 = orch.hard_stop_reason(
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            is_smoke=False,
            epoch=2,
        )
        assert result1 is None
        # Second call: streak 2 -> triggers
        result2 = orch.hard_stop_reason(
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            is_smoke=False,
            epoch=2,
        )
        assert result2 == "val_loss_below_train_loss_with_collapse"

    def test_high_accuracy_high_loss(self, orch: PhaseOrchestrator) -> None:
        """High-accuracy + high-loss detection routed through orchestrator."""
        train_metrics = {"train_calibrated_loss": 0.6, "train_binary_acc": 0.96}
        val_metrics = {"val_binary_acc": 0.97}
        # First call: streak 1 (no trigger, epoch 0 < min_epoch=1)
        r1 = orch.hard_stop_reason(
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            is_smoke=False,
            epoch=0,
        )
        assert r1 is None
        # Second call: epoch >= min_epoch, streak >= streak_req -> triggers
        r2 = orch.hard_stop_reason(
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            is_smoke=False,
            epoch=2,
        )
        assert r2 == "high_accuracy_with_high_loss"

    def test_entropy_collapse_with_missing_classes(self, orch: PhaseOrchestrator) -> None:
        """Entropy collapse that includes missing-class signal routes correctly."""
        val_metrics = {"val_family_entropy": 0.05, "val_entropy_missing_same_dataset": 1.0}
        # First call (epoch=2): streak 1
        r1 = orch.hard_stop_reason(
            train_metrics={},
            val_metrics=val_metrics,
            is_smoke=False,
            epoch=2,
        )
        assert r1 is None
        # Second call (epoch=3): streak 2 -> triggers missing-class collapse
        r2 = orch.hard_stop_reason(
            train_metrics={},
            val_metrics=val_metrics,
            is_smoke=False,
            epoch=3,
        )
        assert r2 == "prediction_entropy_collapse_with_missing_classes"

    def test_entropy_critical_collapse(self, orch: PhaseOrchestrator) -> None:
        """Critical entropy collapse (no missing-class signal) routes correctly."""
        val_metrics = {"val_family_entropy": 0.05, "val_entropy_missing_same_dataset": 0.0}
        r1 = orch.hard_stop_reason(
            train_metrics={},
            val_metrics=val_metrics,
            is_smoke=False,
            epoch=2,
        )
        assert r1 is None
        assert orch._early_stopping_manager.entropy_collapse_streak == 1


# =========================================================================== #
#  Smoke mode interactions
# =========================================================================== #


class TestSmokeModeInteractions:
    """Orchestrator behavior under smoke governance profile."""

    @pytest.fixture
    def orch(
        self,
        mock_model: MagicMock, mock_optimizer: MagicMock, mock_logger: logging.Logger,
        mock_geometry_analyzer: MagicMock, mock_cluster_analyzer: MagicMock,
        mock_centroid_manager: MagicMock, mock_rep_diagnostics: MagicMock,
    ) -> PhaseOrchestrator:
        return _make_orchestrator(
            mock_model, mock_optimizer, mock_logger,
            mock_geometry_analyzer, mock_cluster_analyzer,
            mock_centroid_manager, mock_rep_diagnostics,
        )

    def test_hard_stop_does_not_trigger_in_smoke(self, orch: PhaseOrchestrator) -> None:
        """In smoke mode, hard-stop after a single epoch should not trigger."""
        train_metrics = {"train_calibrated_loss": 1.0}
        val_metrics = {
            "val_calibrated_loss": 0.5,
            "val_family_macro_f1": 0.1,
            "val_family_minority_recall_min": 0.05,
            "val_family_entropy": 0.05,
        }
        r1 = orch.hard_stop_reason(
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            is_smoke=True,
            epoch=0,
        )
        assert r1 is None

    def test_early_stopping_behavior_in_smoke(self, orch: PhaseOrchestrator) -> None:
        """Update_early_stopping returns same keys regardless of smoke mode."""
        result = orch.update_early_stopping(
            val_metrics={
                "val_loss": 0.3,
                "val_family_minority_recall_min": 0.20,
                "val_family_entropy": 0.50,
            }
        )
        assert "is_best" in result
        assert "should_stop" in result
        assert "best_val_loss" in result
        assert "patience_counter" in result


# =========================================================================== #
#  Recovery mode interactions
# =========================================================================== #


class TestRecoveryModeInteractions:
    """Orchestrator interactions with recovery mode scenarios."""

    @pytest.fixture
    def orch(
        self,
        mock_model: MagicMock, mock_optimizer: MagicMock, mock_logger: logging.Logger,
        mock_geometry_analyzer: MagicMock, mock_cluster_analyzer: MagicMock,
        mock_centroid_manager: MagicMock, mock_rep_diagnostics: MagicMock,
    ) -> PhaseOrchestrator:
        return _make_orchestrator(
            mock_model, mock_optimizer, mock_logger,
            mock_geometry_analyzer, mock_cluster_analyzer,
            mock_centroid_manager, mock_rep_diagnostics,
        )

    def test_handle_phase_logic_clears_chunks_on_phase_start(self, orch: PhaseOrchestrator) -> None:
        """When representation phase starts, accumulated chunks are cleared."""
        existing_chunks = [torch.randn(4, 64)]
        existing_labels = [torch.randint(0, 3, (4,))]
        result = orch.handle_representation_phase_logic(
            in_representation_phase=False,
            global_step=50,
            rep_phase_feature_chunks=existing_chunks,
            rep_phase_label_chunks=existing_labels,
            val_loaders={},
            active_family_class_ids=set(),
            run_seed=42,
        )
        assert len(result["rep_phase_feature_chunks"]) == 0

    def test_wrong_phase_transition_blocked_by_manager(self, orch: PhaseOrchestrator) -> None:
        """Handles cases where phase transition is attempted from wrong state."""
        assert not orch._phase_manager.can_transition_to_head_phase(
            in_representation_phase=False
        )

    def test_recovery_state_sync(self, orch: PhaseOrchestrator) -> None:
        """After a failed transition, state dict reflects consistent flags."""
        orch._phase_manager.apply_representation_phase_start()
        result = orch.handle_representation_phase_logic(
            in_representation_phase=True,
            global_step=70,
            rep_phase_feature_chunks=[],
            rep_phase_label_chunks=[],
            val_loaders={},
            active_family_class_ids=set(),
            run_seed=42,
        )
        assert not result["transition_executed"]
        assert result["representation_phase_active"] is True

