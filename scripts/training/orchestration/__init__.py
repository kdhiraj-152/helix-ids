"""orchestration: top-level orchestration for HelixIDS-Full training pipeline.

Phase 13A-4 extraction from train_helix_ids_full.py main().

Extraction targets:
  config_parser.py      — argparse construction, validation, normalization,
                          TrainingConfig/DataConfig creation, config payload
  run_orchestrator.py   — dataset loop, trainer construction, model creation,
                          dataset initialization, trainer.fit(), calibration,
                          result aggregation
  governance_pipeline.py — governance execution, promotion workflow, registry
                          updates, A/B evaluation, artifact publication

Public API:
    parse_config()           — parse CLI args, validate, return ParsedConfig
    run_orchestration()      — execute per-dataset training loop
    run_governance_pipeline()— run governance, promotion, A/B, publish results

Dependency rules:
    orchestration → trainer (allowed — lazy imports inside functions)
    orchestration → governance (allowed)
    orchestration → evaluation (allowed)
    orchestration → diagnostics (allowed)
    orchestration → scheduler (allowed)
    orchestration → representation (allowed)
    trainer → orchestration (forbidden)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from helix_ids.config.helix_full_config import TrainingConfig


@dataclass
class ParsedConfig:
    """Consolidated result from config_parser.parse_config()."""

    args: Any  # argparse.Namespace
    train_config: TrainingConfig
    config_payload: dict[str, Any]
    phase_regime: str
    calibration_enabled: bool
    governance_only_mode: bool
    multi_seed_result: dict[str, Any] | None = None
    forced_class_balance_strategy: str = "focal"
    forced_use_class_weights: bool = False
    forced_focal_gamma: float = 1.2
    forced_label_smoothing: float = 0.0
    forced_use_logit_prior_correction: bool = False
    forced_train_temperature: float = 1.0
    forced_min_class_prob_eps: float = 0.0
    forced_entropy_regularization: float = 0.02
    forced_warmup_ratio: float = 0.0
    forced_lambda_family: float = 2.0
    forced_freeze_backbone_epochs: int = 0
    forced_unfreeze_backbone_step: int = 0
    forced_entropy_warmup_steps: int = 200
    forced_entropy_warmup_weight: float = 0.01
    forced_head_lr_multiplier: float = 10.0
    forced_lambda_binary: float = 0.0
    forced_supcon_weight: float = 1.0
    forced_supcon_temperature: float = 0.03
    forced_step_coverage_check_step: int = 50
    forced_representation_diagnostic_mode: bool = False
    forced_representation_only_steps: int = 140
    forced_representation_micro_cycle_steps: list[int] = field(
        default_factory=lambda: [40, 20, 40, 20, 40]
    )
    forced_use_energy_based_family_objective: bool = False
    forced_adaptive_exit_ratio_threshold: float = 1.6
    forced_adaptive_exit_min_inter_threshold: float = 0.30
    forced_representation_only_ratio: float = 0.25
    forced_head_only_ratio: float = 0.20
    forced_joint_finetune_backbone_lr_multiplier: float = 0.25
    forced_joint_finetune_head_lr_multiplier: float = 0.15
    forced_cluster_relabeling_enabled: bool = True
    forced_cluster_relabel_k: int = 3
    forced_cluster_relabel_seed: int = 42
    forced_cluster_relabel_objective: str = "kmeans"
    forced_cluster_relabel_spectral_affinity: str = "nearest_neighbors"
    forced_sampler_mode: str = "interleaved_rr"
    forced_family_margin_loss_weight: float = 0.1
    forced_family_class4_logit_penalty_weight: float = 0.0
    forced_family_feature_separation_weight: float = 0.0
    forced_family_class4_target_scale: float = 1.0
    forced_enforce_all_classes_per_batch: bool = False
    forced_geometry_ratio_warmup_threshold: float = 2.5
    forced_geometry_ratio_post_phase_threshold: float = 1.2
    forced_nsl_kdd_label_merges: list[tuple[int, int]] = field(
        default_factory=list
    )
    forced_num_workers: int = 0


@dataclass
class OrchestrationResult:
    """Result from run_orchestrator.run_orchestration()."""

    per_dataset_results: dict[str, Any] = field(default_factory=dict)
    all_results: dict[str, Any] = field(default_factory=dict)
    ab_raw_current_by_dataset: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    dataset_snapshot_ids: dict[str, str] = field(default_factory=dict)
    dataset_representation_snapshot_ids: dict[str, str] = field(
        default_factory=dict
    )
    training_elapsed_total: float = 0.0
    feature_order: list[str] = field(default_factory=list)
    schema_hash: str = ""
    feature_signature: str = ""
    pretrain_elapsed: float = 0.0
    guard_failure: str | None = None
    governance_dataset_id: str = "helix_full_decoupled"
    results: dict[str, Any] = field(default_factory=dict)
    determinism_state: Any = None


@dataclass
class GovernanceResult:
    """Result from governance_pipeline.run_governance_pipeline()."""

    governance_stages: dict[str, Any] = field(default_factory=dict)
    governance_context: dict[str, Any] = field(default_factory=dict)
    governance_run_record: dict[str, Any] = field(default_factory=dict)
    determinism: dict[str, Any] = field(default_factory=dict)
    return_payload: dict[str, Any] = field(default_factory=dict)
    success: bool = False


from scripts.training.orchestration.config_parser import parse_config  # noqa: E402
from scripts.training.orchestration.governance_pipeline import (  # noqa: E402
    run_governance_pipeline,
)
from scripts.training.orchestration.run_orchestrator import (  # noqa: E402
    run_orchestration,
)

__all__ = [
    "ParsedConfig",
    "OrchestrationResult",
    "GovernanceResult",
    "parse_config",
    "run_orchestration",
    "run_governance_pipeline",
]
