"""
recovery_manager: Structure recovery and phase-settings configuration.

Phase 18: Extracts configure_structure_recovery from HelixFullTrainer
into a focused recovery-configuration manager.

Dependencies:
    trainer_state (for applying settings to state)
    Core delegate managers (for lifecycle resets)
"""

from __future__ import annotations

import logging
from typing import Any

from scripts.training.core.trainer_state import TrainerState
from scripts.training.losses import LossRegistry
from scripts.training.representation import CentroidManager


class RecoveryManager:
    """Configures structural anti-collapse constraints and phase settings.

    Owns the configure_structure_recovery lifecycle: validates inputs,
    applies phase settings to TrainerState, and resets delegate state.
    """

    def __init__(
        self,
        state: TrainerState,
        loss_registry: LossRegistry,
        centroid_manager: CentroidManager,
        logger: logging.Logger | None = None,
    ) -> None:
        self._state = state
        self._loss_registry = loss_registry
        self._centroid_manager = centroid_manager
        self._logger = logger or logging.getLogger(__name__)

    # ── Public API ──────────────────────────────────────────────────────

    def configure_structure_recovery(
        self,
        *,
        active_family_classes: set[int],
        supcon_weight: float,
        supcon_temperature: float,
        step_coverage_check_step: int,
        representation_diagnostic_mode: bool,
        phase_settings: dict[str, Any],
        cluster_relabeling_enabled: bool,
        cluster_relabel_k: int | None,
        cluster_relabel_seed: int,
        cluster_relabel_objective: str,
        cluster_relabel_spectral_affinity: str,
    ) -> None:
        """Configure structural anti-collapse constraints for family prediction coverage.

        Validates inputs, applies to TrainerState, resets delegate state.
        """
        # ── Core settings ────────────────────────────────────────────────
        self._state.active_family_class_ids = {int(cls) for cls in active_family_classes}
        self._state.supcon_weight = max(0.0, float(supcon_weight))
        self._state.supcon_temperature = max(1e-3, float(supcon_temperature))
        self._state.rep_supcon_weight = max(0.0, float(supcon_weight))
        self._state.rep_supcon_temperature = max(1e-3, float(supcon_temperature))
        self._state.step_coverage_check_step = max(1, int(step_coverage_check_step))
        self._state.step_coverage_checked = False
        self._state.representation_diagnostic_mode = bool(representation_diagnostic_mode)

        # ── Phase step counts ────────────────────────────────────────────
        self._state.representation_only_steps = max(
            0, int(phase_settings.get("representation_only_steps", 0))
        )
        self._state.head_only_steps = max(
            0, int(phase_settings.get("head_only_steps", 0))
        )

        # ── Phase state reset ────────────────────────────────────────────
        self._state.reset_phase_state()

        # ── Joint finetune multipliers ───────────────────────────────────
        self._state.joint_finetune_backbone_lr_multiplier = max(
            1e-3,
            float(phase_settings.get("joint_finetune_backbone_lr_multiplier", 0.25)),
        )
        self._state.joint_finetune_head_lr_multiplier = max(
            1e-3,
            float(phase_settings.get("joint_finetune_head_lr_multiplier", 0.15)),
        )
        self._state.coverage_check_after_head_steps = max(1, int(step_coverage_check_step))

        # ── Cluster relabeling ───────────────────────────────────────────
        self._apply_cluster_relabel_settings(
            enabled=cluster_relabeling_enabled,
            k=cluster_relabel_k,
            seed=cluster_relabel_seed,
            objective=cluster_relabel_objective,
            spectral_affinity=cluster_relabel_spectral_affinity,
        )

        # ── Geometry thresholds ──────────────────────────────────────────
        self._state.geometry_max_intra_inter_ratio_warmup = max(
            0.1, float(phase_settings.get("geometry_ratio_warmup_threshold", 2.5))
        )
        self._state.geometry_max_intra_inter_ratio_post_phase = max(
            0.1,
            float(phase_settings.get("geometry_ratio_post_phase_threshold", 1.2)),
        )
        self._state.geometry_max_intra_inter_ratio = (
            self._state.geometry_max_intra_inter_ratio_post_phase
        )
        self._state.enforce_all_classes_per_batch = bool(
            phase_settings.get("enforce_all_classes_per_batch", False)
        )

        # ── Representation window pattern ────────────────────────────────
        self._apply_window_pattern(phase_settings)

        # ── Exit thresholds ──────────────────────────────────────────────
        self._state.rep_adaptive_exit_ratio_threshold = float(
            phase_settings.get("adaptive_exit_ratio_threshold", 1.6)
        )
        self._state.rep_adaptive_exit_min_inter_threshold = float(
            phase_settings.get("adaptive_exit_min_inter_threshold", 0.30)
        )

        # ── Sampler mode ─────────────────────────────────────────────────
        self._state.sampler_mode = (
            str(phase_settings.get("sampler_mode", "interleaved_rr")).strip().lower()
            or "interleaved_rr"
        )

        # ── Reset centroids ──────────────────────────────────────────────
        self._state.cluster_centers = None
        self._state.phase1_class_centroids = None
        self._state.phase1_centroid_class_ids = []
        self._centroid_manager._centroid_ema_state.clear()
        self._centroid_manager._epoch_frozen_centroids.clear()
        self._state.rep_epoch_feature_chunks = []
        self._state.rep_epoch_label_chunks = []
        self._state.representation_snapshot_id = None
        self._state.rep_backbone_grad_scale = 2.0

        # ── Energy-based objective settings ──────────────────────────────
        self._apply_energy_settings(phase_settings)

        # ── Disable train-time family temperature scaling ────────────────
        self._state.train_temperature = 1.0

    # ── Internal helpers ─────────────────────────────────────────────────

    def _apply_cluster_relabel_settings(
        self,
        *,
        enabled: bool,
        k: int | None,
        seed: int,
        objective: str,
        spectral_affinity: str,
    ) -> None:
        """Validate and apply cluster relabeling settings."""
        self._state.cluster_relabeling_enabled = bool(enabled)
        self._state.cluster_relabel_k = (
            None if k is None else max(2, int(k))
        )
        self._state.cluster_relabel_seed = int(seed)

        obj = str(objective).strip().lower()
        if obj not in {"kmeans", "gmm", "spectral"}:
            raise ValueError(f"Unsupported cluster relabel objective: {objective!r}")
        self._state.cluster_relabel_objective = obj

        aff = str(spectral_affinity).strip().lower()
        if aff not in {"nearest_neighbors", "rbf"}:
            raise ValueError(f"Unsupported spectral affinity: {spectral_affinity!r}")
        self._state.cluster_relabel_spectral_affinity = aff

    def _apply_window_pattern(self, phase_settings: dict[str, Any]) -> None:
        """Build representation window pattern from phase_settings."""
        cycle_steps_raw = phase_settings.get(
            "representation_micro_cycle_steps",
            [40, 20, 40, 20, 40],
        )
        cycle_steps: list[int] = []
        if isinstance(cycle_steps_raw, (list, tuple)):
            for value in cycle_steps_raw:
                try:
                    cycle_steps.append(max(1, int(value)))
                except (TypeError, ValueError):
                    continue
        if not cycle_steps:
            cycle_steps = [max(1, int(self._state.representation_only_steps))]
        if len(cycle_steps) % 2 == 0:
            cycle_steps.append(cycle_steps[-1])
        self._state.representation_window_pattern = [
            (idx % 2 == 0, int(window_steps))
            for idx, window_steps in enumerate(cycle_steps)
        ]
        self._state.representation_only_steps = int(
            sum(window_steps for _, window_steps in self._state.representation_window_pattern)
        )

    def _apply_energy_settings(self, phase_settings: dict[str, Any]) -> None:
        """Apply energy-based objective settings to state."""
        s = self._state
        s.use_energy_based_family_objective = bool(
            phase_settings.get("use_energy_based_family_objective", True)
        )
        s.energy_gap_margin = max(0.0, float(phase_settings.get("energy_gap_margin", 1.0)))
        s.energy_gap_weight = max(0.0, float(phase_settings.get("energy_gap_weight", 1.0)))
        s.energy_multi_negative_alpha = max(
            0.0, float(phase_settings.get("energy_multi_negative_alpha", 1.0))
        )
        s.energy_logit_temperature = max(
            1.0, float(phase_settings.get("energy_logit_temperature", 2.0))
        )
        s.energy_balance_weight = max(
            0.0, float(phase_settings.get("energy_balance_weight", 0.1))
        )
        s.energy_winner_weight = max(
            0.0, float(phase_settings.get("energy_winner_weight", 0.5))
        )
        s.energy_winner_min_count = max(
            0, int(phase_settings.get("energy_winner_min_count", 1))
        )
        s.energy_emergence_bias_beta = max(
            0.0,
            float(phase_settings.get("energy_emergence_bias_beta", s.energy_emergence_bias_beta)),
        )
        s.energy_emergence_bias_eps = max(
            1e-6,
            float(phase_settings.get("energy_emergence_bias_eps", s.energy_emergence_bias_eps)),
        )
        win_rate_mom = float(
            phase_settings.get("energy_win_rate_ema_momentum", s.energy_win_rate_ema_momentum)
        )
        s.energy_win_rate_ema_momentum = min(max(win_rate_mom, 0.80), 0.95)
        target_ratio = float(
            phase_settings.get(
                "energy_emergence_bias_target_ratio",
                s.energy_emergence_bias_target_ratio,
            )
        )
        s.energy_emergence_bias_target_ratio = min(
            max(target_ratio, s.energy_emergence_bias_ratio_min),
            s.energy_emergence_bias_ratio_max,
        )
        s.energy_isolate_short_horizon = bool(
            phase_settings.get("energy_isolate_short_horizon", s.energy_isolate_short_horizon)
        )

        # Short-run isolation
        if s.use_energy_based_family_objective and s.energy_isolate_short_horizon and int(s.config.epochs) <= 1:
            s.energy_balance_weight = 0.0
            s.energy_winner_weight = 0.0
            self._logger.info(
                "Energy isolation enabled for <=1 epoch run: forcing balance_w=0.0 winner_w=0.0"
            )

        self._loss_registry.reset_energy_win_rate_ema()
