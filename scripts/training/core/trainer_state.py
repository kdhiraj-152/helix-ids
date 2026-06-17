"""
trainer_state: Canonical state holder for HelixFullTrainer init-time configuration.

Phase 18: Extracts state initialization, runtime flags, and delegate construction
from HelixFullTrainer.__init__ into a focused state object.

Public API:
    TrainerState — holds all configuration state and constructs delegates

Dependencies:
    All delegate classes from training.scheduler, training.losses,
    training.representation, training.diagnostics, training.execution,
    training.evaluation, training.validation.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
from torch import optim
from torch.utils.data import DataLoader

from helix_ids.config.helix_full_config import TrainingConfig
from helix_ids.models.full import HelixIDSFull, MultiTaskLoss
from scripts.training.diagnostics import (
    ClusterAnalyzer,
    GeometryAnalyzer,
    RepresentationDiagnostics,
)
from scripts.training.evaluation import EvaluationOrchestrator
from scripts.training.execution import (
    BatchProcessor,
    EpochRunner,
    TrainingOrchestrator,
    WarmupManager,
)
from scripts.training.losses import LossRegistry
from scripts.training.representation import CentroidManager, RepresentationCoordinator
from scripts.training.scheduler import (
    EarlyStoppingManager,
    FreezeManager,
    LRScheduler,
    PhaseManager,
    PhaseOrchestrator,
)
from scripts.training.validation import ValidationOrchestrator


class TrainerState:
    """Canonical holder of all trainer configuration state and delegate construction.

    Holds everything that HelixFullTrainer.__init__ currently owns:
    - Model/optimizer/dataloader references
    - Runtime flags and configuration values
    - Delegate instances (orchestrators, analyzers, managers)

    The HelixFullTrainer accesses state via attribute delegation to reduce
    boilerplate (e.g., ``self.model`` proxies to ``self._trainer_state.model``).
    """

    def __init__(
        self,
        model: HelixIDSFull,
        train_loader: DataLoader,
        val_loaders: dict[str, DataLoader],
        test_loaders: dict[str, DataLoader],
        optimizer: optim.Optimizer,
        loss_fn: MultiTaskLoss,
        config: TrainingConfig,
        binary_class_weights: torch.Tensor | None = None,
        family_class_weights: torch.Tensor | None = None,
        train_family_class_count: int | None = None,
        run_seed: int = 42,
        device: str = "mps",
        logger: logging.Logger | None = None,
    ) -> None:
        # ── Core references ──────────────────────────────────────────────
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loaders = val_loaders
        self.test_loaders = test_loaders
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

        # ── Weight tensors ───────────────────────────────────────────────
        self.binary_class_weights = (
            binary_class_weights.to(device) if binary_class_weights is not None else None
        )
        self.family_class_weights = (
            family_class_weights.to(device) if family_class_weights is not None else None
        )
        self.train_family_class_count = int(train_family_class_count or 0)

        # ── Config-derived defaults ──────────────────────────────────────
        self.base_balance_strategy = "weighted_ce"
        self.focal_warmup_epochs = 0
        self.family_log_prior: torch.Tensor | None = None
        self.tail_class_mask: torch.Tensor | None = None
        self.run_seed = int(run_seed)
        self.train_temperature = 1.0
        self.warmup_kl_uniform_weight = 0.0
        self.kl_uniform_weight = 0.0
        self.logit_floor = -2.0
        self.logit_floor_weight = 0.0
        self.tail_ce_weight = 0.0
        self.warmup_ratio = 0.0
        self.total_train_steps = max(1, int(len(self.train_loader)) * max(1, int(self.config.epochs)))
        self.warmup_steps = max(1, int(math.ceil(self.total_train_steps * self.warmup_ratio)))

        # ── Training step/epoch state ────────────────────────────────────
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_model_state: dict[str, torch.Tensor] | None = None
        self.patience_counter = 0
        self.val_gap_collapse_streak = 0
        self.entropy_missing_class_streak = 0
        self.entropy_collapse_streak = 0
        self.high_accuracy_high_loss_streak = 0

        # ── Freeze / warmup state ────────────────────────────────────────
        self.freeze_backbone_epochs = max(0, int(getattr(self.config, "freeze_backbone_epochs", 0)))
        self.unfreeze_backbone_step = max(0, int(getattr(self.config, "unfreeze_backbone_step", 0)))
        self.entropy_warmup_steps = max(0, int(getattr(self.config, "entropy_warmup_steps", 0)))
        self.entropy_warmup_weight = max(0.0, float(getattr(self.config, "entropy_warmup_weight", 0.0)))
        self.backbone_frozen = False
        self.step10_symmetry_logged = False

        # ── Backbone / feature state ─────────────────────────────────────
        self.backbone_params = [
            param for param in self.model.backbone.parameters() if param.requires_grad
        ]
        self.feature_order: list[str] = []
        self.schema_hash = "unknown"

        # ── Training configuration (set by configure_structure_recovery) ─
        self.supcon_weight = 0.0
        self.supcon_temperature = 0.2
        self.step_coverage_check_step = 50
        self.step_coverage_checked = False
        self.active_family_class_ids: set[int] = set()
        self.class4_logit_shift = 0.0
        self.class4_logit_shift_class_id = 4
        self.representation_diagnostic_mode = False
        self.representation_only_steps = 0
        self.head_only_steps = 0
        self.representation_phase_active = False
        self.representation_curriculum_complete = False
        self.in_representation_window = False
        self.representation_window_pattern: list[tuple[bool, int]] = []
        self.head_phase_start_step = -1
        self.joint_finetune_start_step = -1
        self.joint_finetune_active = False
        self.joint_finetune_backbone_lr_multiplier = 0.25
        self.joint_finetune_head_lr_multiplier = 0.15
        self.coverage_check_after_head_steps = 50
        self.representation_diagnostics: dict[str, Any] = {}
        self.rep_phase_feature_chunks: list[torch.Tensor] = []
        self.rep_phase_label_chunks: list[torch.Tensor] = []
        self.representation_snapshot_id: str | None = None
        self.cluster_relabeling_enabled = False
        self.cluster_relabel_k: int | None = None
        self.cluster_relabel_seed = self.run_seed
        self.cluster_relabel_objective = "kmeans"
        self.cluster_relabel_spectral_affinity = "nearest_neighbors"
        self.cluster_centers: torch.Tensor | None = None
        self.phase1_class_centroids: torch.Tensor | None = None
        self.phase1_centroid_class_ids: list[int] = []

        # ── Geometry / rep parameters ────────────────────────────────────
        self.geometry_min_inter_threshold = 0.2
        self.geometry_max_intra_inter_ratio_warmup = 2.5
        self.geometry_max_intra_inter_ratio_post_phase = 1.2
        self.geometry_max_intra_inter_ratio = self.geometry_max_intra_inter_ratio_post_phase
        self.geometry_min_cluster_size = 100
        self.geometry_min_nearest_center_acc = 0.6
        self.rep_supcon_weight = 0.2
        self.rep_supcon_temperature = 0.03
        self.rep_supcon_negative_weight = 1.5
        self.rep_supcon_min_negatives = 10
        self.rep_var_lower_bound = 0.08
        self.rep_var_upper_bound = 0.12
        self.rep_var_clamp_weight = 0.05
        self.rep_pair_margin_distance = 1.2
        self.rep_pair_margin_weight = 0.15
        self.rep_hard_negative_weight = 3.0
        self.rep_adaptive_exit_ratio_threshold = 1.6
        self.rep_adaptive_exit_min_inter_threshold = 0.30
        self.rep_centroid_barrier_min_distance = 0.4
        self.rep_centroid_barrier_weight = 0.5
        self.rep_centroid_repulsion_margin = 0.6
        self.rep_centroid_repulsion_weight = 0.6
        self.rep_critical_pair_weight = 0.0
        self.rep_barrier_activation_fraction = 0.30
        self.rep_expansion_target_min_inter = 0.45
        self.rep_compression_supcon_scale = 0.3
        self.rep_topk_nearest_negatives = 3
        self.rep_min_displacement_eps = 0.05

        # ── Energy parameters ────────────────────────────────────────────
        self.use_energy_based_family_objective = True
        self.energy_gap_margin = 1.0
        self.energy_gap_weight = 1.0
        self.energy_multi_negative_alpha = 1.0
        self.energy_logit_temperature = 2.0
        self.energy_balance_weight = 0.1
        self.energy_winner_weight = 0.5
        self.energy_winner_min_count = 1
        self.energy_emergence_bias_beta = 0.5
        self.energy_emergence_bias_eps = 1e-3
        self.energy_win_rate_ema_momentum = 0.9
        self.energy_emergence_bias_ratio_min = 0.10
        self.energy_emergence_bias_ratio_max = 0.30
        self.energy_emergence_bias_target_ratio = 0.20
        self.energy_isolate_short_horizon = True

        # ── Misc runtime ─────────────────────────────────────────────────
        self._logit_temp = 1.0
        self._temperature_calibration = 1.0
        self._temperature_calibration_lr = 1e-3
        self.rep_epoch_feature_chunks: list[torch.Tensor] = []
        self.rep_epoch_label_chunks: list[torch.Tensor] = []
        self.rep_backbone_grad_scale = 2.0
        self.centroid_ema_momentum = 0.9
        self.class_starvation_streak = 0
        self.critical_collision_pairs: set[tuple[int, int]] = {(0, 3), (0, 4), (3, 4)}
        self.emergency_label_merge_map: dict[int, int] = {3: 0, 4: 0}
        self.representation_balance_target_per_class = 64
        self.enforce_all_classes_per_batch = False
        self.sampler_mode = "interleaved_rr"

        # ── Base LR scales from optimizer param groups ───────────────────
        self._base_lr_scales: dict[str, float] = {
            str(param_group.get("group_name", f"group_{idx}")):
            float(param_group.get("lr_scale", 1.0))
            for idx, param_group in enumerate(self.optimizer.param_groups)
        }

        # ── Training history ─────────────────────────────────────────────
        self.training_history: dict[str, list[float]] = {
            "train_loss": [],
            "train_binary_acc": [],
            "train_family_acc": [],
            "train_family_logit_max": [],
            "train_family_logit_min": [],
            "val_loss": [],
            "val_binary_acc": [],
            "val_family_acc": [],
            "val_binary_auroc": [],
            "val_binary_auprc": [],
            "val_family_macro_f1": [],
            "val_family_minority_recall_min": [],
            "val_family_entropy": [],
        }

        # ── Disable-integrity flag ───────────────────────────────────────
        self.disable_integrity_hard_stops = False
        self.disable_tail_focal_regularizer = False

    # ── Delegate construction ──────────────────────────────────────────

    def build_phase_manager(self) -> PhaseManager:
        """Construct PhaseManager from current state."""
        return PhaseManager(
            representation_only_steps=int(self.representation_only_steps),
            head_only_steps=int(self.head_only_steps),
            representation_diagnostic_mode=bool(self.representation_diagnostic_mode),
            use_energy_based_family_objective=bool(self.use_energy_based_family_objective),
            rep_adaptive_exit_ratio_threshold=float(self.rep_adaptive_exit_ratio_threshold),
            rep_adaptive_exit_min_inter_threshold=float(
                self.rep_adaptive_exit_min_inter_threshold
            ),
            representation_window_pattern=(
                list(self.representation_window_pattern)
                if self.representation_window_pattern
                else []
            ),
            joint_finetune_backbone_lr_multiplier=float(
                self.joint_finetune_backbone_lr_multiplier
            ),
            joint_finetune_head_lr_multiplier=float(self.joint_finetune_head_lr_multiplier),
        )

    def build_early_stopping_manager(self) -> EarlyStoppingManager:
        """Construct EarlyStoppingManager from current state."""
        return EarlyStoppingManager(
            early_stopping_patience=int(getattr(self.config, "early_stopping_patience", 5)),
            early_stopping_threshold=float(getattr(self.config, "early_stopping_threshold", 0.01)),
            min_family_minority_recall_for_best=float(
                getattr(self.config, "min_family_minority_recall_for_best", 0.3)
            ),
            disable_integrity_hard_stops=bool(
                getattr(self, "disable_integrity_hard_stops", False)
            ),
        )

    def build_freeze_manager(self) -> FreezeManager:
        """Construct FreezeManager from current state."""
        mgr = FreezeManager()
        mgr.backbone_frozen = bool(self.backbone_frozen)
        return mgr

    def build_lr_scheduler(self) -> LRScheduler:
        """Construct LRScheduler from current state."""
        return LRScheduler(
            learning_rate=float(self.config.learning_rate),
            warmup_epochs=int(getattr(self.config, "warmup_epochs", 0)),
            warmup_init_lr=float(getattr(self.config, "warmup_init_lr", 1e-6)),
            epochs=int(self.config.epochs),
        )

    def build_evaluation_orchestrator(self) -> EvaluationOrchestrator:
        """Construct EvaluationOrchestrator from current state."""
        return EvaluationOrchestrator(
            model=self.model,
            device=self.device,
            loss_fn=self.loss_fn,
            binary_class_weights=self.binary_class_weights,
            family_class_weights=self.family_class_weights,
            logger=self.logger,
            family_log_prior=self.family_log_prior,
            use_energy_based_family_objective=self.use_energy_based_family_objective,
            energy_logit_temperature=self.energy_logit_temperature,
            active_family_class_ids=self.active_family_class_ids,
            class4_logit_shift=self.class4_logit_shift,
            class4_logit_shift_class_id=self.class4_logit_shift_class_id,
            disable_integrity_hard_stops=bool(
                getattr(self, "disable_integrity_hard_stops", False)
            ),
        )

    def build_validation_orchestrator(self) -> ValidationOrchestrator:
        """Construct ValidationOrchestrator."""
        return ValidationOrchestrator()

    def build_geometry_analyzer(self) -> GeometryAnalyzer:
        """Construct GeometryAnalyzer from current state."""
        return GeometryAnalyzer(
            use_energy_based_family_objective=self.use_energy_based_family_objective,
            geometry_min_cluster_size=self.geometry_min_cluster_size,
            critical_collision_pairs=self.critical_collision_pairs,
            geometry_min_inter_threshold=self.geometry_min_inter_threshold,
            geometry_max_intra_inter_ratio_warmup=self.geometry_max_intra_inter_ratio_warmup,
            geometry_max_intra_inter_ratio_post_phase=self.geometry_max_intra_inter_ratio_post_phase,
            logger=self.logger,
        )

    def build_cluster_analyzer(self) -> ClusterAnalyzer:
        """Construct ClusterAnalyzer from current state."""
        return ClusterAnalyzer(
            model=self.model,
            device=self.device,
            logger=self.logger,
            cluster_relabel_objective=self.cluster_relabel_objective,
            cluster_relabel_seed=self.cluster_relabel_seed,
            cluster_relabel_spectral_affinity=self.cluster_relabel_spectral_affinity,
        )

    def build_rep_diagnostics(self) -> RepresentationDiagnostics:
        """Construct RepresentationDiagnostics from current state."""
        return RepresentationDiagnostics(
            model=self.model,
            device=self.device,
            logger=self.logger,
            representation_only_steps=self.representation_only_steps,
            head_only_steps=self.head_only_steps,
            sampler_mode=self.sampler_mode,
        )

    def build_centroid_manager(self) -> CentroidManager:
        """Construct CentroidManager from current state."""
        return CentroidManager(centroid_ema_momentum=float(self.centroid_ema_momentum))

    def build_loss_registry(self) -> LossRegistry:
        """Construct LossRegistry from current state."""
        return LossRegistry(
            entropy_warmup_steps=self.entropy_warmup_steps,
            entropy_warmup_weight=self.entropy_warmup_weight,
            kl_uniform_weight=self.kl_uniform_weight,
            warmup_kl_uniform_weight=self.warmup_kl_uniform_weight,
            logit_floor=self.logit_floor,
            logit_floor_weight=self.logit_floor_weight,
            tail_ce_weight=self.tail_ce_weight,
            tail_class_mask=self.tail_class_mask,
            loss_fn=self.loss_fn,
            energy_gap_weight=self.energy_gap_weight,
            energy_multi_negative_alpha=self.energy_multi_negative_alpha,
            energy_balance_weight=self.energy_balance_weight,
            energy_winner_weight=self.energy_winner_weight,
            energy_winner_min_count=self.energy_winner_min_count,
            energy_logit_temperature=self.energy_logit_temperature,
            energy_win_rate_ema_momentum=self.energy_win_rate_ema_momentum,
            energy_emergence_bias_eps=self.energy_emergence_bias_eps,
            energy_emergence_bias_beta=self.energy_emergence_bias_beta,
            energy_emergence_bias_ratio_max=self.energy_emergence_bias_ratio_max,
        )

    def build_phase_orchestrator(
        self,
        phase_manager: PhaseManager,
        early_stopping_manager: EarlyStoppingManager,
        geometry_analyzer: GeometryAnalyzer,
        cluster_analyzer: ClusterAnalyzer,
        centroid_manager: CentroidManager,
        rep_diagnostics: RepresentationDiagnostics,
        *,
        quality_gate_entropy: float = 0.3,
    ) -> PhaseOrchestrator:
        """Construct PhaseOrchestrator wiring all scheduler delegates."""
        return PhaseOrchestrator(
            model=self.model,
            optimizer=self.optimizer,
            logger=self.logger,
            base_lr_scales=self._base_lr_scales,
            phase_manager=phase_manager,
            early_stopping_manager=early_stopping_manager,
            geometry_analyzer=geometry_analyzer,
            cluster_analyzer=cluster_analyzer,
            centroid_manager=centroid_manager,
            rep_diagnostics=rep_diagnostics,
            representation_only_steps=int(self.representation_only_steps),
            head_only_steps=int(self.head_only_steps),
            representation_diagnostic_mode=bool(self.representation_diagnostic_mode),
            use_energy_based_family_objective=bool(self.use_energy_based_family_objective),
            rep_adaptive_exit_ratio_threshold=float(self.rep_adaptive_exit_ratio_threshold),
            rep_adaptive_exit_min_inter_threshold=float(
                self.rep_adaptive_exit_min_inter_threshold
            ),
            joint_finetune_backbone_lr_multiplier=float(
                self.joint_finetune_backbone_lr_multiplier
            ),
            joint_finetune_head_lr_multiplier=float(self.joint_finetune_head_lr_multiplier),
            cluster_relabeling_enabled=bool(self.cluster_relabeling_enabled),
            cluster_relabel_k=self.cluster_relabel_k,
            cluster_relabel_seed=int(self.cluster_relabel_seed),
            cluster_relabel_objective=str(self.cluster_relabel_objective),
            cluster_relabel_spectral_affinity=str(self.cluster_relabel_spectral_affinity),
            critical_collision_pairs=self.critical_collision_pairs,
            emergency_label_merge_map=self.emergency_label_merge_map,
            disable_integrity_hard_stops=bool(
                getattr(self, "disable_integrity_hard_stops", False)
            ),
            min_family_minority_recall_for_best=float(
                getattr(self.config, "min_family_minority_recall_for_best", 0.3)
            ),
            quality_gate_entropy=quality_gate_entropy,
        )

    def build_batch_processor(self, loss_registry: LossRegistry) -> BatchProcessor:
        """Construct BatchProcessor."""
        return BatchProcessor(
            model=self.model, loss_fn=self.loss_fn,
            loss_registry=loss_registry, config=self.config,
        )

    def build_warmup_manager(self) -> WarmupManager:
        """Construct WarmupManager."""
        return WarmupManager(
            model=self.model, loss_fn=self.loss_fn,
            optimizer=self.optimizer, device=self.device, config=self.config,
        )

    def build_epoch_runner(
        self,
        batch_processor: BatchProcessor,
        warmup_manager: WarmupManager,
    ) -> EpochRunner:
        """Construct EpochRunner."""
        return EpochRunner(
            model=self.model,
            train_loader=self.train_loader,
            config=self.config,
            device=self.device,
            logger=self.logger,
            batch_processor=batch_processor,
            warmup_manager=warmup_manager,
        )

    def build_training_orchestrator(self, epoch_runner: EpochRunner) -> TrainingOrchestrator:
        """Construct TrainingOrchestrator."""
        return TrainingOrchestrator(
            config=self.config, logger=self.logger, epoch_runner=epoch_runner,
        )

    def build_representation_coordinator(self) -> RepresentationCoordinator:
        """Construct RepresentationCoordinator."""
        return RepresentationCoordinator()

    def reset_phase_state(self) -> None:
        """Reset representation-phase state for fresh recovery configuration."""
        self.representation_phase_active = False
        self.representation_curriculum_complete = False
        self.in_representation_window = False
        self.head_phase_start_step = -1
        self.joint_finetune_start_step = -1
        self.joint_finetune_active = False
        self.cluster_centers = None
        self.phase1_class_centroids = None
        self.phase1_centroid_class_ids = []
        self.rep_epoch_feature_chunks = []
        self.rep_epoch_label_chunks = []
        self.representation_snapshot_id = None
        self.step_coverage_checked = False
