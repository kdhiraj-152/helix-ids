"""phase_orchestrator: Phase transition orchestration for HelixFullTrainer.

Phase 15 extraction from HelixFullTrainer.

PhaseOrchestrator owns:
    - Phase transitions (representation -> head -> joint finetune)
    - Representation curriculum exits
    - Representation window activation
    - Joint finetune activation
    - Early stopping orchestration
    - Hard-stop decision tree
    - State synchronization (PhaseManager <-> execution)

Dependency rules:
    phase_orchestrator -> scheduler (PhaseManager, EarlyStoppingManager)
    phase_orchestrator -> diagnostics (GeometryAnalyzer, ClusterAnalyzer, RepresentationDiagnostics)
    phase_orchestrator -> torch (allowed)
    phase_orchestrator -> numpy (allowed)
    phase_orchestrator -> trainer internals (forbidden)
"""

from __future__ import annotations

import logging
from typing import Any, Optional, cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scripts.training.scheduler.phase_manager import PhaseManager
from scripts.training.scheduler.early_stopping import EarlyStoppingManager


# =========================================================================== #
# PhaseOrchestrator -- owns all phase-transition logic
# =========================================================================== #


class PhaseOrchestrator:
    """Coordinates phase transitions, curriculum exits, window states,
    and early stopping for HelixFullTrainer.

    The orchestrator holds references to model, optimizer, and all lower-level
    delegates (PhaseManager, EarlyStoppingManager, geometry/cluster/centroid
    managers).  Each public method is a self-contained orchestration action
    that the trainer calls via a 1-3 line wrapper.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        logger: logging.Logger,
        base_lr_scales: dict[str, float],
        phase_manager: PhaseManager,
        early_stopping_manager: EarlyStoppingManager,
        geometry_analyzer: Any,
        cluster_analyzer: Any,
        centroid_manager: Any,
        rep_diagnostics: Any,
        # Config values snapshotted at construction
        representation_only_steps: int,
        head_only_steps: int,
        representation_diagnostic_mode: bool,
        use_energy_based_family_objective: bool,
        rep_adaptive_exit_ratio_threshold: float,
        rep_adaptive_exit_min_inter_threshold: float,
        joint_finetune_backbone_lr_multiplier: float,
        joint_finetune_head_lr_multiplier: float,
        cluster_relabeling_enabled: bool,
        cluster_relabel_k: int | None,
        cluster_relabel_seed: int,
        cluster_relabel_objective: str,
        cluster_relabel_spectral_affinity: str,
        critical_collision_pairs: set[tuple[int, int]],
        emergency_label_merge_map: dict[int, int],
        disable_integrity_hard_stops: bool,
        # Config values for early stopping
        min_family_minority_recall_for_best: float,
        quality_gate_entropy: float = 0.3,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._logger = logger
        self._base_lr_scales = dict(base_lr_scales)

        # Lower-level delegates
        self._phase_manager = phase_manager
        self._early_stopping_manager = early_stopping_manager
        self._geometry_analyzer = geometry_analyzer
        self._cluster_analyzer = cluster_analyzer
        self._centroid_manager = centroid_manager
        self._rep_diagnostics = rep_diagnostics

        # Config params
        self._representation_only_steps = int(representation_only_steps)
        self._head_only_steps = int(head_only_steps)
        self._representation_diagnostic_mode = bool(representation_diagnostic_mode)
        self._use_energy_based_family_objective = bool(use_energy_based_family_objective)
        self._rep_adaptive_exit_ratio_threshold = float(rep_adaptive_exit_ratio_threshold)
        self._rep_adaptive_exit_min_inter_threshold = float(rep_adaptive_exit_min_inter_threshold)
        self._joint_finetune_backbone_lr_multiplier = float(joint_finetune_backbone_lr_multiplier)
        self._joint_finetune_head_lr_multiplier = float(joint_finetune_head_lr_multiplier)
        self._cluster_relabeling_enabled = bool(cluster_relabeling_enabled)
        self._cluster_relabel_k = cluster_relabel_k
        self._cluster_relabel_seed = int(cluster_relabel_seed)
        self._cluster_relabel_objective = str(cluster_relabel_objective)
        self._cluster_relabel_spectral_affinity = str(cluster_relabel_spectral_affinity)
        self._critical_collision_pairs = critical_collision_pairs
        self._emergency_label_merge_map = dict(emergency_label_merge_map)
        self._disable_integrity_hard_stops = bool(disable_integrity_hard_stops)
        self._min_family_minority_recall_for_best = float(min_family_minority_recall_for_best)
        self._quality_gate_entropy = float(quality_gate_entropy)

    # ======================================================================= #
    # Public accessors
    # ======================================================================= #

    @property
    def phase_manager(self) -> PhaseManager:
        return self._phase_manager

    @property
    def early_stopping_manager(self) -> EarlyStoppingManager:
        return self._early_stopping_manager

    def is_smoke_mode(self, *, epochs: int | None = None, gov_profile: str = "") -> bool:
        """Return True when running a smoke-governance profile.

        Delegates to EarlyStoppingManager.
        """
        return self._early_stopping_manager.is_smoke_mode(
            epochs=epochs,
            gov_profile=gov_profile,
        )

    def is_representation_window_step(self, global_step: int) -> bool:
        """Return True when *global_step* falls in a representation micro-window.

        Delegates to PhaseManager.
        """
        return self._phase_manager.is_representation_window_step(global_step)

    # ======================================================================= #
    # Phase lifecycle -- called step-by-step from trainer wrappers
    # ======================================================================= #

    def handle_representation_phase_logic(
        self,
        *,
        in_representation_phase: bool,
        global_step: int,
        rep_phase_feature_chunks: list[torch.Tensor],
        rep_phase_label_chunks: list[torch.Tensor],
        val_loaders: dict[str, DataLoader],
        active_family_class_ids: set[int],
        run_seed: int,
    ) -> dict[str, Any]:
        """Orchestrate representation phase initiation, window state, and
        transition logic.  Returns a state-update dict that the trainer applies.

        This coalesces the full _handle_representation_phase_logic sub-tree:
            _maybe_start_representation_phase
            _update_representation_window_state  (with curriculum exit check)
            _finalize_representation_phase_if_ready
        """
        state: dict[str, Any] = {
            "transition_executed": False,
            "phase1_class_centroids": None,
            "phase1_centroid_class_ids": [],
            "representation_snapshot_id": None,
            "representation_diagnostics": {},
            "rep_phase_feature_chunks": list(rep_phase_feature_chunks),
            "rep_phase_label_chunks": list(rep_phase_label_chunks),
            "representation_phase_active": self._phase_manager.representation_phase_active,
            "representation_curriculum_complete": self._phase_manager.representation_curriculum_complete,
            "in_representation_window": self._phase_manager.in_representation_window,
        }

        if not self._representation_diagnostic_mode:
            return state

        # -- _maybe_start_representation_phase --
        if self._phase_manager.can_start_representation_phase(global_step):
            self._phase_manager.apply_representation_phase_start()
            state["rep_phase_feature_chunks"] = []
            state["rep_phase_label_chunks"] = []
            state["representation_phase_active"] = True
            state["representation_curriculum_complete"] = False
            state["in_representation_window"] = False
            self._logger.info(
                "Representation phase started at step %d",
                int(global_step),
            )

        if not self._phase_manager.representation_phase_active:
            return state

        # -- _update_representation_window_state (with curriculum exit check) --
        self._apply_window_state_update(
            in_representation_phase=in_representation_phase,
            global_step=global_step,
            rep_feature_chunks=state["rep_phase_feature_chunks"],
            rep_label_chunks=state["rep_phase_label_chunks"],
            val_loaders=val_loaders,
            active_family_class_ids=active_family_class_ids,
            run_seed=run_seed,
        )
        state["representation_curriculum_complete"] = self._phase_manager.representation_curriculum_complete
        state["in_representation_window"] = self._phase_manager.in_representation_window

        # -- curriculum step check --
        if int(global_step) >= self._representation_only_steps:
            self._phase_manager.apply_curriculum_complete()
            state["representation_curriculum_complete"] = True

        # -- _finalize_representation_phase_if_ready --
        if not self._phase_manager.can_transition_to_head_phase(in_representation_phase):
            return state

        rep_features = (
            torch.cat(state["rep_phase_feature_chunks"], dim=0)
            if state["rep_phase_feature_chunks"]
            else torch.zeros((0, 0), dtype=torch.float32)
        )
        rep_labels = (
            torch.cat(state["rep_phase_label_chunks"], dim=0)
            if state["rep_phase_label_chunks"]
            else torch.zeros((0,), dtype=torch.int64)
        )

        transition_state = self._execute_phase_transition(
            rep_features=rep_features,
            rep_labels=rep_labels,
            global_step=global_step,
            val_loaders=val_loaders,
            active_family_class_ids=active_family_class_ids,
            run_seed=run_seed,
        )
        state.update(transition_state)
        state["transition_executed"] = transition_state.get("executed", False)

        # Clean up chunks after transition
        state["rep_phase_feature_chunks"] = []
        state["rep_phase_label_chunks"] = []

        # Sync PhaseManager flags back to state dict
        state["representation_phase_active"] = self._phase_manager.representation_phase_active
        state["head_phase_start_step"] = self._phase_manager.head_phase_start_step
        state["joint_finetune_start_step"] = self._phase_manager.joint_finetune_start_step

        return state

    def _apply_window_state_update(
        self,
        *,
        in_representation_phase: bool,
        global_step: int,
        rep_feature_chunks: list[torch.Tensor],
        rep_label_chunks: list[torch.Tensor],
        val_loaders: dict[str, DataLoader],
        active_family_class_ids: set[int],
        run_seed: int,
    ) -> None:
        """Update trainability and LR settings on window transitions."""
        in_window = self._phase_manager.is_representation_window_step(global_step)
        if in_window == self._phase_manager.in_representation_window:
            return  # no transition

        if in_representation_phase:
            self._set_phase_trainability(
                train_backbone=True,
                train_family_head=False,
                train_family_projection=False,
            )
            self._set_phase_lr_scales(
                backbone_multiplier=1.0,
                head_multiplier=1.0,
            )
            self._phase_manager.apply_window_enter()
            return

        # Exiting window -- check curriculum exit
        if self._evaluate_curriculum_exit(
            rep_feature_chunks=rep_feature_chunks,
            rep_label_chunks=rep_label_chunks,
            global_step=global_step,
            val_loaders=val_loaders,
            active_family_class_ids=active_family_class_ids,
            run_seed=run_seed,
        ):
            self._phase_manager.apply_curriculum_complete()

        self._set_phase_trainability(
            train_backbone=False,
            train_family_head=True,
            train_family_projection=True,
        )
        self._set_phase_lr_scales(
            backbone_multiplier=1.0,
            head_multiplier=1.0,
        )
        self._phase_manager.apply_window_exit()

    # ======================================================================= #
    # Representation curriculum exit check
    # ======================================================================= #

    def should_exit_representation_curriculum(
        self,
        *,
        rep_phase_feature_chunks: list[torch.Tensor],
        rep_phase_label_chunks: list[torch.Tensor],
        global_step: int,
        val_loaders: dict[str, DataLoader],
        active_family_class_ids: set[int],
        run_seed: int,
    ) -> dict[str, Any]:
        """Evaluate adaptive geometry exit condition.

        Returns dict:
            should_exit: bool
            representation_diagnostics_update: dict  (probe diagnostics)
        """
        result: dict[str, Any] = {
            "should_exit": False,
            "representation_diagnostics_update": {},
        }

        if self._use_energy_based_family_objective:
            return result
        if not rep_phase_feature_chunks or not rep_phase_label_chunks:
            return result

        rep_features = torch.cat(list(rep_phase_feature_chunks), dim=0)
        rep_labels = torch.cat(list(rep_phase_label_chunks), dim=0)
        if int(torch.unique(rep_labels, dim=0).numel()) <= 1:
            return result

        diagnostics = self._run_diagnostics(
            train_features=rep_features,
            train_labels=rep_labels,
            label_space="phase1_probe",
            val_loaders=val_loaders,
            active_family_class_ids=active_family_class_ids,
            run_seed=run_seed,
        )
        ratio = float(
            diagnostics.get(
                "intra_inter_ratio",
                diagnostics.get("intra_to_inter_ratio", diagnostics.get("ratio", float("inf"))),
            )
        )
        min_inter = float(diagnostics.get("min_inter_center_distance", 0.0))
        pass_exit = self._phase_manager.should_exit_curriculum_by_targets(
            intra_inter_ratio=ratio,
            min_inter_center_distance=min_inter,
        )

        probe = {
            "ratio": ratio,
            "min_inter": min_inter,
            "ratio_threshold": float(self._rep_adaptive_exit_ratio_threshold),
            "min_inter_threshold": float(self._rep_adaptive_exit_min_inter_threshold),
            "pass": bool(pass_exit),
            "step": int(global_step),
        }
        result["should_exit"] = bool(pass_exit)
        result["representation_diagnostics_update"] = {"adaptive_exit_probe": probe}

        self._logger.info(
            "RepDiag[phase1_probe] adaptive_exit_check ratio=%.4f min_inter=%.4f "
            "thresholds=(%.4f, %.4f) pass=%s",
            ratio,
            min_inter,
            float(self._rep_adaptive_exit_ratio_threshold),
            float(self._rep_adaptive_exit_min_inter_threshold),
            str(bool(pass_exit)).lower(),
        )
        return result

    # ======================================================================= #
    # Joint finetune activation
    # ======================================================================= #

    def maybe_activate_joint_finetune_phase(
        self,
        *,
        global_step: int,
        representation_diagnostics: dict[str, Any],
    ) -> bool:
        """Enable low-LR joint tuning after head-only stage completes.

        Delegates condition check to PhaseManager.
        Returns True when joint finetune was activated.
        """
        if not self._phase_manager.can_activate_joint_finetune(global_step=global_step):
            return False

        strict_diag = cast(
            dict[str, Any],
            representation_diagnostics.get(
                "cluster_relabel",
                representation_diagnostics.get("original", {}),
            ),
        )
        if strict_diag:
            self._enforce_geometry_integrity(strict_diag, label_space="joint_finetune")

        self._set_phase_trainability(
            train_backbone=True,
            train_family_head=True,
            train_family_projection=True,
        )
        self._set_phase_lr_scales(
            backbone_multiplier=self._joint_finetune_backbone_lr_multiplier,
            head_multiplier=self._joint_finetune_head_lr_multiplier,
        )
        self._phase_manager.apply_joint_finetune_activation()
        self._logger.info(
            "PhaseTrainability backbone=train family_head=train (joint_low_lr step=%d)",
            int(global_step),
        )
        return True

    # ======================================================================= #
    # Early stopping orchestration
    # ======================================================================= #

    def update_early_stopping(
        self,
        *,
        val_metrics: dict[str, float],
    ) -> dict[str, Any]:
        """Update early stopping state and return decision + best model state.

        Returns dict:
            should_stop: bool
            is_best: bool
            best_val_loss: float
            patience_counter: int
            best_model_state: OrderedDict | None
        """
        result = self._early_stopping_manager.update_early_stopping(
            val_metrics,
            quality_gate_minority_recall=self._min_family_minority_recall_for_best,
            quality_gate_entropy=self._quality_gate_entropy,
        )

        if result["is_best"]:
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in self._model.state_dict().items()
            }
            result["best_model_state"] = best_state
            self._logger.info(
                "✅ Best model update (loss: %.4f)",
                result["best_val_loss"],
            )
        else:
            result["best_model_state"] = None

        if result["should_stop"]:
            self._logger.info(
                "Early stopping triggered (patience %d >= %d)",
                result["patience_counter"],
                self._early_stopping_manager._patience,
            )

        return result

    # ======================================================================= #
    # Hard-stop decision tree
    # ======================================================================= #

    def hard_stop_reason(
        self,
        *,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        is_smoke: bool,
        epoch: int,
    ) -> str | None:
        """Return hard-stop reason when integrity constraints are violated.

        Delegates actual detection to EarlyStoppingManager.
        """
        if self._disable_integrity_hard_stops:
            return None
        return self._early_stopping_manager.hard_stop_reason(
            train_metrics,
            val_metrics,
            is_smoke=is_smoke,
            epoch=epoch,
        )

    # ======================================================================= #
    # Internal: phase transition execution
    # ======================================================================= #

    def _execute_phase_transition(
        self,
        *,
        rep_features: torch.Tensor,
        rep_labels: torch.Tensor,
        global_step: int,
        val_loaders: dict[str, DataLoader],
        active_family_class_ids: set[int],
        run_seed: int,
    ) -> dict[str, Any]:
        """Execute the full representation-to-head-phase transition.

        Called once when the curriculum exit conditions are met.

        Returns state-update dict.
        """
        if self._use_energy_based_family_objective:
            return self._execute_energy_transition(global_step=global_step)

        return self._execute_regular_transition(
            rep_features=rep_features,
            rep_labels=rep_labels,
            global_step=global_step,
            val_loaders=val_loaders,
            active_family_class_ids=active_family_class_ids,
            run_seed=run_seed,
        )

    def _execute_energy_transition(self, *, global_step: int) -> dict[str, Any]:
        """Transition in energy-based mode -- simpler path."""
        snapshot_id = f"energy_phase_v1_step_{int(global_step)}"
        diag_update = {
            "energy_transition": {
                "mode": "class_conditional_energy",
                "global_step": int(global_step),
                "snapshot_id": snapshot_id,
            },
            "representation_snapshot_id": snapshot_id,
        }

        self._set_phase_trainability(
            train_backbone=False,
            train_family_head=True,
            train_family_projection=True,
        )
        self._set_phase_lr_scales(backbone_multiplier=1.0, head_multiplier=1.0)

        self._phase_manager.apply_head_phase_start(global_step=global_step)

        self._logger.info(
            "Representation transition completed in energy mode: snapshot_id=%s",
            snapshot_id,
        )

        return {
            "executed": True,
            "phase1_class_centroids": None,
            "phase1_centroid_class_ids": [],
            "representation_snapshot_id": snapshot_id,
            "representation_diagnostics": diag_update,
            "step_coverage_checked": False,
        }

    def _execute_regular_transition(
        self,
        *,
        rep_features: torch.Tensor,
        rep_labels: torch.Tensor,
        global_step: int,
        val_loaders: dict[str, DataLoader],
        active_family_class_ids: set[int],
        run_seed: int,
    ) -> dict[str, Any]:
        """Full regular transition with diagnostics, centroids, relabeling."""
        rep_features_prepared = self._cluster_analyzer.prepare_representation_features(
            rep_features
        )

        diagnostics = self._run_diagnostics(
            train_features=rep_features_prepared,
            train_labels=rep_labels,
            label_space="original",
            val_loaders=val_loaders,
            active_family_class_ids=active_family_class_ids,
            run_seed=run_seed,
        )

        features_for_next = rep_features_prepared
        labels_for_next = rep_labels
        if self._has_critical_collision_pairs(diagnostics):
            self._logger.warning(
                "RepDiag[original] critical collision pairs unresolved; applying emergency merge map=%s",
                self._emergency_label_merge_map,
            )
            labels_for_next = self._apply_emergency_label_merge(
                labels=rep_labels,
                merge_map=self._emergency_label_merge_map,
            )
            features_for_next = self._cluster_analyzer.prepare_representation_features(
                rep_features
            )
            diagnostics = self._run_diagnostics(
                train_features=features_for_next,
                train_labels=labels_for_next,
                label_space="collision_merge",
                val_loaders=val_loaders,
                active_family_class_ids=active_family_class_ids,
                run_seed=run_seed,
            )

        self._enforce_geometry_integrity(diagnostics, label_space="original")

        phase1_centroids, phase1_class_ids = self._cluster_analyzer.compute_class_centroids(
            features_for_next, labels_for_next
        )
        stabilized_centroids = self._centroid_manager.stabilize_centroids(
            phase1_centroids, list(phase1_class_ids)
        )
        stable_centroids = stabilized_centroids.detach().clone()
        stable_class_ids = list(phase1_class_ids)

        diag_update: dict[str, Any] = {
            "phase1_class_centroids_shape": [int(v) for v in stable_centroids.shape],
            "phase1_centroid_class_ids": [int(v) for v in stable_class_ids],
        }

        self._logger.info(
            "Phase1 centroids frozen: classes=%s shape=%s",
            stable_class_ids,
            tuple(int(v) for v in stable_centroids.shape),
        )

        if self._cluster_relabeling_enabled:
            cluster_diag = self._execute_cluster_relabeling(
                features_for_next,
                labels_for_next,
                active_count=len(active_family_class_ids),
                global_step=global_step,
            )
            diag_update.update(cluster_diag)
            phase_diag = cast(
                dict[str, Any],
                diag_update.get(
                    "cluster_relabel",
                    diag_update.get("original", diagnostics),
                ),
            )
            snapshot_label_space = "cluster_relabel"
        else:
            phase_diag = diagnostics
            snapshot_label_space = "original"

        snapshot_id = self._rep_diagnostics.build_representation_snapshot_id(
            phase_diag,
            label_space=snapshot_label_space,
            representation_only_steps=self._representation_only_steps,
            head_only_steps=self._head_only_steps,
            sampler_mode="",
        )
        diag_update["representation_snapshot_id"] = snapshot_id

        self._logger.info(
            "Representation snapshot locked: id=%s",
            snapshot_id,
        )

        self._set_phase_trainability(
            train_backbone=False,
            train_family_head=True,
            train_family_projection=True,
        )
        self._set_phase_lr_scales(backbone_multiplier=1.0, head_multiplier=1.0)

        self._phase_manager.apply_head_phase_start(global_step=global_step)

        return {
            "executed": True,
            "phase1_class_centroids": stable_centroids,
            "phase1_centroid_class_ids": stable_class_ids,
            "representation_snapshot_id": snapshot_id,
            "representation_diagnostics": diag_update,
            "step_coverage_checked": False,
        }

    # ======================================================================= #
    # Internal helpers
    # ======================================================================= #

    def _evaluate_curriculum_exit(
        self,
        *,
        rep_feature_chunks: list[torch.Tensor],
        rep_label_chunks: list[torch.Tensor],
        global_step: int,
        val_loaders: dict[str, DataLoader],
        active_family_class_ids: set[int],
        run_seed: int,
    ) -> bool:
        """Evaluate whether curriculum exit conditions are met.

        Used by window state logic when exiting representation window.
        """
        if self._use_energy_based_family_objective:
            return False
        if not rep_feature_chunks or not rep_label_chunks:
            return False

        rep_features = torch.cat(list(rep_feature_chunks), dim=0)
        rep_labels = torch.cat(list(rep_label_chunks), dim=0)
        if int(rep_features.shape[0]) == 0 or int(rep_labels.shape[0]) == 0:
            return False
        if int(torch.unique(rep_labels, dim=0).numel()) <= 1:
            return False

        diagnostics = self._run_diagnostics(
            train_features=rep_features,
            train_labels=rep_labels,
            label_space="phase1_probe",
            val_loaders=val_loaders,
            active_family_class_ids=active_family_class_ids,
            run_seed=run_seed,
        )
        ratio = float(
            diagnostics.get(
                "intra_inter_ratio",
                diagnostics.get("intra_to_inter_ratio", diagnostics.get("ratio", float("inf"))),
            )
        )
        min_inter = float(diagnostics.get("min_inter_center_distance", 0.0))
        return bool(
            self._phase_manager.should_exit_curriculum_by_targets(
                intra_inter_ratio=ratio,
                min_inter_center_distance=min_inter,
            )
        )

    def _has_critical_collision_pairs(self, diagnostics: dict[str, Any]) -> bool:
        """Check for unresolved critical collision pairs."""
        return self._geometry_analyzer.has_critical_collision_pairs(diagnostics)

    @staticmethod
    def _apply_emergency_label_merge(
        labels: torch.Tensor,
        *,
        merge_map: dict[int, int],
    ) -> torch.Tensor:
        """Merge critically colliding classes before classifier-head phase."""
        if not merge_map:
            return labels
        out = labels.detach().clone().to(dtype=torch.int64)
        for src_class, dst_class in merge_map.items():
            out[out == int(src_class)] = int(dst_class)
        return out

    def _enforce_geometry_integrity(
        self,
        diagnostics: dict[str, Any],
        *,
        label_space: str,
    ) -> None:
        """Enforce geometry integrity thresholds."""
        self._geometry_analyzer.enforce_geometry_integrity(
            diagnostics,
            label_space=label_space,
        )

    def _run_diagnostics(
        self,
        *,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        label_space: str,
        val_loaders: dict[str, DataLoader],
        active_family_class_ids: set[int],
        run_seed: int,
    ) -> dict[str, Any]:
        """Run representation diagnostics -- delegates to RepresentationDiagnostics."""
        return self._rep_diagnostics.run_representation_diagnostics(
            train_features=train_features,
            train_labels=train_labels,
            label_space=label_space,
            active_family_class_ids=active_family_class_ids,
            val_loaders=val_loaders,
            geometry_analyzer=self._geometry_analyzer,
            run_seed=run_seed,
        )

    def _execute_cluster_relabeling(
        self,
        rep_features: torch.Tensor,
        rep_labels: torch.Tensor,
        *,
        active_count: int,
        global_step: int,
    ) -> dict[str, Any]:
        """Execute cluster relabeling and return diagnostics updates."""
        active_count = max(2, active_count)
        auto_k = max(2, int(np.ceil(float(active_count) / 2.0)))
        k = int(self._cluster_relabel_k or auto_k)

        cluster_labels, cluster_centers = self._cluster_analyzer.fit_embedding_clusters(
            rep_features, n_clusters=k
        )
        cluster_size_counts = np.bincount(
            np.asarray(cluster_labels.to(device="cpu").numpy(), dtype=np.int64),
            minlength=int(cluster_centers.shape[0]),
        ).astype(np.int64)

        diag: dict[str, Any] = {
            "cluster_size_counts": [int(v) for v in cluster_size_counts.tolist()],
            "cluster_size_entropy": float(
                _normalized_entropy_from_counts(
                    [int(v) for v in cluster_size_counts.tolist()]
                )
            ),
            "cluster_relabel_config": {
                "algorithm": str(self._cluster_relabel_objective),
                "k": int(k),
                "seed": int(self._cluster_relabel_seed),
                "spectral_affinity": str(self._cluster_relabel_spectral_affinity),
            },
        }
        self._logger.info(
            "Cluster relabeling: k=%d active_classes=%d cluster_size_entropy=%.4f",
            k,
            active_count,
            diag.get("cluster_size_entropy", 0.0),
        )
        return diag

    # ======================================================================= #
    # Side-effect helpers (model + optimizer mutation)
    # ======================================================================= #

    def _set_phase_trainability(
        self,
        *,
        train_backbone: bool,
        train_family_head: bool,
        train_family_projection: bool | None = None,
    ) -> None:
        """Toggle trainability for backbone/family head."""
        targets = PhaseManager.compute_trainability_targets(
            train_backbone,
            train_family_head,
            train_family_projection=train_family_projection,
        )
        for param in self._model.backbone.parameters():
            param.requires_grad = targets["train_backbone"]
        if hasattr(self._model, "family_projection"):
            for param in self._model.family_projection.parameters():
                param.requires_grad = targets["train_family_projection"]
        for param in self._model.family_head.parameters():
            param.requires_grad = targets["train_family_head"]

        self._logger.info(
            "PhaseTrainability backbone=%s family_projection=%s family_head=%s",
            "train" if targets["train_backbone"] else "frozen",
            "train" if targets["train_family_projection"] else "frozen",
            "train" if train_family_head else "frozen",
        )

    def _set_phase_lr_scales(
        self,
        *,
        backbone_multiplier: float,
        head_multiplier: float,
    ) -> None:
        """Apply phase-specific LR multipliers on top of base group scales."""
        from scripts.training.scheduler.lr_scheduler import LRScheduler

        scales = LRScheduler.apply_lr_scales(
            self._optimizer.param_groups,
            self._base_lr_scales,
            backbone_multiplier,
            head_multiplier,
        )
        for idx, param_group in enumerate(self._optimizer.param_groups):
            group_name = str(param_group.get("group_name", f"group_{idx}"))
            if group_name in scales:
                param_group["lr_scale"] = scales[group_name]


# =========================================================================== #
# Module-level helpers
# =========================================================================== #


def _normalized_entropy_from_counts(counts: list[int]) -> float:
    """Compute normalized entropy from a list of counts."""
    import math

    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log(p)
    max_entropy = math.log(len(counts)) if len(counts) > 1 else 1.0
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0
