"""Config parsing, validation, and normalization for HelixIDS-Full training.

Phase 13A-4 extraction from train_helix_ids_full.py main().

Moved:
  - argparse construction and argument definitions
  - Argument validation and normalization
  - CLI defaults resolution
  - TrainingConfig creation
  - DataConfig creation
  - Config payload construction
  - Phase regime string construction
  - _resolve_class_balance_strategy()
  - _apply_disable_early_stopping()
"""

from __future__ import annotations

import argparse
import os
import sys

from helix_ids.config.helix_full_config import TrainingConfig
from scripts.training.orchestration import ParsedConfig

# ============================================================================
# Config helpers (extracted from train_helix_ids_full.py)
# ============================================================================


def _resolve_class_balance_strategy(balance_strategy_arg: str) -> tuple[str, bool]:
    """Resolve CLI balance strategy aliases into model strategy + class-weight usage."""
    strategy_raw = str(balance_strategy_arg).strip().lower()
    if strategy_raw == "none":
        return "weighted_ce", False
    if strategy_raw in {"weighted_ce", "focal"}:
        return strategy_raw, True
    if strategy_raw == "sqrt_weighted_ce":
        return "weighted_ce", True
    raise ValueError(
        "--class-balance-strategy must be one of {'none', 'weighted_ce', 'sqrt_weighted_ce', 'focal'}"
    )


def _apply_disable_early_stopping(train_config: TrainingConfig, *, disable_early_stopping: bool) -> None:
    """Disable early stopping by extending patience beyond the full epoch budget."""
    if not disable_early_stopping:
        return
    target_patience = max(int(getattr(train_config, "epochs", 0)) + 1, 10_000)
    train_config.early_stopping_patience = int(target_patience)


# ============================================================================
# Argparse construction
# ============================================================================


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for HelixIDS-Full training."""
    parser = argparse.ArgumentParser(description="Train HelixIDS-Full model")

    parser.add_argument(
        "--config",
        type=str,
        default="config/helix_config.yaml",
        help="Path to training config (YAML)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/helix_full",
        help="Output directory for model/logs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device (mps, cpu, cuda)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Batch size for validation/test evaluation (defaults to --batch-size)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Number of epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.5e-4,
        help="Optimizer learning rate (defaults to stable value 1.5e-4)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=2,
        help="Warmup epochs (also used as stage-1 CE epochs when focal is selected)",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Max gradient norm clipping (0 disables clipping)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Emit intra-epoch progress logs every N steps",
    )
    parser.add_argument(
        "--class-balance-strategy",
        type=str,
        default="focal",
        choices=["none", "weighted_ce", "sqrt_weighted_ce", "focal"],
        help=(
            "Class-balance mode: none=unweighted CE, "
            "weighted_ce=inverse-frequency weighted CE, "
            "sqrt_weighted_ce=sqrt-inverse weighted CE, "
            "focal=focal loss"
        ),
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=1.2,
        help="Focal gamma (used when --class-balance-strategy focal)",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing applied to family loss",
    )
    parser.add_argument(
        "--sampler-mode",
        type=str,
        default="interleaved_rr",
        choices=["interleaved_rr", "weighted_random_sampler"],
        help=(
            "Training sampler mode for train loader: "
            "interleaved_rr=enforced class-interleaved round-robin (default), "
            "weighted_random_sampler=legacy weighted random sampler"
        ),
    )
    parser.add_argument(
        "--min-class4-samples",
        type=int,
        default=2000,
        help="Minimum class-4 samples enforced via deterministic train-set upsampling",
    )
    parser.add_argument(
        "--class4-per-batch-min",
        type=int,
        default=2,
        help="Hard minimum class-4 samples per training batch for interleaved sampler",
    )
    parser.add_argument(
        "--family-margin-loss-weight",
        type=float,
        default=None,
        help=(
            "Override family margin loss weight (defaults: 0.15 for NSL-KDD, 0.1 otherwise)"
        ),
    )
    parser.add_argument(
        "--family-class4-logit-penalty-weight",
        type=float,
        default=0.0,
        help=(
            "Class-4 dominance ranking penalty weight added to family objective as "
            "lambda * mean(relu(logit_class4 - max_other_logits))"
        ),
    )
    parser.add_argument(
        "--family-feature-separation-weight",
        type=float,
        default=0.0,
        help=(
            "Feature-space centroid separation weight for class-4 vs non-class-4 as "
            "lambda_sep * ( - ||mean(z4)-mean(z_not4)||^2 )"
        ),
    )
    parser.add_argument(
        "--family-class4-target-scale",
        type=float,
        default=1.0,
        help=(
            "Per-sample target-pressure scale for family class-4 labels (0..1). "
            "Applied multiplicatively to family CE terms where label==4."
        ),
    )
    parser.add_argument(
        "--enable-logit-adjustment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable family-head logit adjustment with train priors and stabilization controls",
    )
    parser.add_argument(
        "--logit-temp",
        type=float,
        default=1.0,
        help="Family-head logit temperature / tau (T>1 smooths confidence)",
    )
    parser.add_argument(
        "--logit-adjustment-tau",
        type=float,
        default=None,
        help="Alias for logit adjustment temperature; overrides --logit-temp when set",
    )
    parser.add_argument(
        "--min-class-prob-eps",
        type=float,
        default=0.0,
        help="Probability-floor epsilon for family-head softmax when logit adjustment is enabled",
    )
    parser.add_argument(
        "--entropy-regularization",
        type=float,
        default=0.02,
        help="Entropy regularization strength for family-head predictions",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("HELIX_SEED", "42")),
        help="Global seed for deterministic execution",
    )
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Run full epoch budget without early stopping termination",
    )
    parser.add_argument(
        "--calibration-mode",
        type=str,
        default="internal_on",
        choices=["internal_on", "internal_off"],
        help=(
            "Post-training calibration mode: internal_on fits temperature scaling + class-4 threshold "
            "and emits calibration artifacts; internal_off disables calibration pipeline"
        ),
    )
    parser.add_argument(
        "--max-temperature",
        type=float,
        default=5.0,
        help="Maximum temperature allowed during post-training calibration fit",
    )
    parser.add_argument(
        "--class4-logit-shift",
        type=float,
        default=0.0,
        help=(
            "Inference-only class-4 logit shift applied before softmax during evaluation/calibration: "
            "logit_4 <- logit_4 - delta"
        ),
    )
    parser.add_argument(
        "--multi-seed-governance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run fixed 50-epoch multi-seed calibration governance batch instead of single-seed training",
    )
    parser.add_argument(
        "--multi-seeds",
        type=str,
        default="42,1337,2026",
        help="Comma-separated seeds for multi-seed governance mode",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["nsl_kdd", "unsw_nb15", "cicids"],
        help="Run isolated training only for the specified dataset",
    )
    parser.add_argument(
        "--holdout-dataset",
        type=str,
        default="cicids",
        choices=["nsl_kdd", "unsw_nb15", "cicids"],
        help="Dataset to keep fully held out when entity keys are unavailable",
    )
    parser.add_argument(
        "--precomputed-splits-dir",
        type=str,
        default="data/processed/multi_dataset_v1",
        help="Path to precomputed split .npy files",
    )
    parser.add_argument(
        "--force-recompute-splits",
        action="store_true",
        help="Ignore precomputed splits and recompute from raw datasets",
    )
    parser.add_argument(
        "--snapshot-mode",
        type=str,
        default="strict",
        choices=["strict", "research_override"],
        help="Governance mode: strict requires frozen contract snapshot; research_override allows unfrozen validation",
    )
    parser.add_argument(
        "--allow-unfrozen-snapshot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Alias for --snapshot-mode research_override",
    )
    parser.add_argument(
        "--ab-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable strict A/B contract gates for geometry-first promotion",
    )
    parser.add_argument(
        "--ab-track",
        type=str,
        default="objective",
        choices=["feature", "objective"],
        help="A/B change track; only this axis may change versus baseline",
    )
    parser.add_argument(
        "--ab-change-id",
        type=str,
        default="baseline",
        help="Identifier for the single feature/objective change under test",
    )
    parser.add_argument(
        "--ab-baseline",
        type=str,
        default=None,
        help="Path to baseline raw A/B metrics JSON (optional; auto-discovers latest if omitted)",
    )
    parser.add_argument(
        "--ab-require-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if no baseline raw A/B metrics are available",
    )
    parser.add_argument(
        "--cluster-objective",
        type=str,
        default="kmeans",
        choices=["kmeans", "gmm", "spectral"],
        help="Clustering objective used for manifold relabeling",
    )
    parser.add_argument(
        "--cluster-spectral-affinity",
        type=str,
        default="nearest_neighbors",
        choices=["nearest_neighbors", "rbf"],
        help="Fixed affinity for spectral objective",
    )

    return parser


# ============================================================================
# Config normalization
# ============================================================================


def _compute_phase_regime(
    forced_representation_only_ratio: float,
    forced_head_only_ratio: float,
    epochs: int,
    forced_sampler_mode: str,
    objective_regime_tag: str,
    forced_geometry_ratio_warmup_threshold: float,
    forced_geometry_ratio_post_phase_threshold: float,
) -> str:
    """Build deterministic phase regime identifier string."""
    return (
        "helix_full_phase_v2"
        f":rep{forced_representation_only_ratio:.2f}"
        f":head{forced_head_only_ratio:.2f}"
        f":epochs_{int(epochs)}"
        f":sampler_{forced_sampler_mode}"
        f":{objective_regime_tag}"
        f":geom{forced_geometry_ratio_warmup_threshold:.1f}->{forced_geometry_ratio_post_phase_threshold:.1f}"
        ":proj_deep_mlp"
    )


# ============================================================================
# Public API
# ============================================================================


def parse_config(args_list: list[str] | None = None) -> ParsedConfig:
    """Parse, validate, and normalize CLI arguments into a ParsedConfig.

    Handles:
      - argparse construction and argument parsing
      - All argument validation, normalization, and defaults resolution
      - TrainingConfig and DataConfig creation
      - Config payload dict construction
      - Phase regime string generation
      - Multi-seed governance detection (but NOT execution)

    Returns
    -------
    ParsedConfig
        Consolidated config with all parsed, validated, and normalized fields.
    """
    parser = _build_parser()
    if args_list is None:
        args_list = sys.argv[1:]
    args = parser.parse_args(args_list)

    # ------------------------------------------------------------------
    # Snapshot mode alias
    # ------------------------------------------------------------------
    if bool(args.allow_unfrozen_snapshot):
        args.snapshot_mode = "research_override"

    # ------------------------------------------------------------------
    # Forced anti-collapse controls & validation
    # ------------------------------------------------------------------
    forced_batch_size = int(args.batch_size)
    if forced_batch_size <= 0:
        raise ValueError("--batch-size must be >= 1")

    forced_class_balance_strategy, forced_use_class_weights = _resolve_class_balance_strategy(
        args.class_balance_strategy
    )
    forced_use_class_weights = bool(forced_use_class_weights)
    if forced_class_balance_strategy == "focal":
        forced_use_class_weights = False

    if float(args.max_temperature) <= 0.0:
        raise ValueError("--max-temperature must be > 0")
    if float(args.class4_logit_shift) < 0.0:
        raise ValueError("--class4-logit-shift must be >= 0")
    if float(args.class4_logit_shift) > 5.0:
        raise ValueError("--class4-logit-shift must be <= 5.0")

    calibration_enabled = str(args.calibration_mode).strip().lower() == "internal_on"
    forced_focal_gamma = float(args.focal_gamma)
    forced_label_smoothing = float(args.label_smoothing)
    forced_use_logit_prior_correction = bool(args.enable_logit_adjustment)
    forced_train_temperature = float(
        args.logit_adjustment_tau if args.logit_adjustment_tau is not None else args.logit_temp
    )
    forced_min_class_prob_eps = float(args.min_class_prob_eps)
    forced_entropy_regularization = float(args.entropy_regularization)

    forced_warmup_ratio = 0.0
    forced_lambda_family = 2.0
    forced_freeze_backbone_epochs = 0
    forced_unfreeze_backbone_step = 0
    forced_entropy_warmup_steps = 200
    forced_entropy_warmup_weight = 0.01
    forced_head_lr_multiplier = 10.0
    forced_lambda_binary = 0.0
    forced_supcon_weight = 1.0
    forced_supcon_temperature = 0.03
    forced_step_coverage_check_step = 50
    forced_representation_diagnostic_mode = False
    forced_representation_only_steps = 140
    forced_representation_micro_cycle_steps = [40, 20, 40, 20, 40]
    forced_use_energy_based_family_objective = False
    forced_adaptive_exit_ratio_threshold = 1.6
    forced_adaptive_exit_min_inter_threshold = 0.30
    forced_representation_only_ratio = 0.25
    forced_head_only_ratio = 0.20
    forced_joint_finetune_backbone_lr_multiplier = 0.25
    forced_joint_finetune_head_lr_multiplier = 0.15
    forced_cluster_relabeling_enabled = True
    forced_cluster_relabel_k: int = 3
    forced_cluster_relabel_seed = 42
    forced_cluster_relabel_objective = str(args.cluster_objective).strip().lower()
    forced_cluster_relabel_spectral_affinity = (
        str(args.cluster_spectral_affinity).strip().lower()
    )
    forced_sampler_mode = str(args.sampler_mode).strip().lower()
    dataset_key = str(args.dataset).strip().lower() if args.dataset is not None else ""

    default_family_margin_loss_weight = 0.15 if dataset_key == "nsl_kdd" else 0.1
    forced_family_margin_loss_weight = (
        float(args.family_margin_loss_weight)
        if args.family_margin_loss_weight is not None
        else default_family_margin_loss_weight
    )
    forced_family_class4_logit_penalty_weight = (
        float(args.family_class4_logit_penalty_weight)
        if dataset_key == "unsw_nb15"
        else 0.0
    )
    forced_family_feature_separation_weight = (
        float(args.family_feature_separation_weight)
        if dataset_key == "unsw_nb15"
        else 0.0
    )
    forced_family_class4_target_scale = (
        float(args.family_class4_target_scale)
        if dataset_key == "unsw_nb15"
        else 1.0
    )

    forced_enforce_all_classes_per_batch = False
    forced_geometry_ratio_warmup_threshold = 2.5
    forced_geometry_ratio_post_phase_threshold = 1.2
    objective_regime_tag = "energyobj_cebin_l1_1p0_l2_0p1_l3_0p5_t2p0"

    phase_regime = _compute_phase_regime(
        forced_representation_only_ratio=forced_representation_only_ratio,
        forced_head_only_ratio=forced_head_only_ratio,
        epochs=int(args.epochs),
        forced_sampler_mode=forced_sampler_mode,
        objective_regime_tag=objective_regime_tag,
        forced_geometry_ratio_warmup_threshold=forced_geometry_ratio_warmup_threshold,
        forced_geometry_ratio_post_phase_threshold=forced_geometry_ratio_post_phase_threshold,
    )

    forced_nsl_kdd_label_merges: list[tuple[int, int]] = []
    forced_num_workers = 0

    if bool(args.ab_mode) and args.dataset is None:
        raise ValueError("--ab-mode requires --dataset for single-manifold A/B comparability")

    # ------------------------------------------------------------------
    # Multi-seed governance check
    # ------------------------------------------------------------------
    governance_only_mode = bool(args.multi_seed_governance)

    # ------------------------------------------------------------------
    # TrainingConfig and config payload
    # ------------------------------------------------------------------
    train_config = TrainingConfig(
        batch_size=forced_batch_size,
        epochs=args.epochs,
        device=args.device,
    )

    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            raise ValueError("--learning-rate must be > 0")
        train_config.learning_rate = float(args.learning_rate)
    train_config.learning_rate = float(train_config.learning_rate)

    if args.warmup_epochs is not None:
        if int(args.warmup_epochs) < 0:
            raise ValueError("--warmup-epochs must be >= 0")
        train_config.warmup_epochs = int(args.warmup_epochs)

    if args.grad_clip is not None:
        if float(args.grad_clip) < 0.0:
            raise ValueError("--grad-clip must be >= 0")
        train_config.max_grad_norm = float(args.grad_clip)

    if forced_focal_gamma < 0.0:
        raise ValueError("--focal-gamma must be >= 0")
    if forced_train_temperature <= 0.0:
        raise ValueError("--logit-temp/--logit-adjustment-tau must be > 0")
    if forced_min_class_prob_eps < 0.0:
        raise ValueError("--min-class-prob-eps must be >= 0")
    if int(args.log_interval) < 1:
        raise ValueError("--log-interval must be >= 1")
    if forced_entropy_regularization < 0.0:
        raise ValueError("--entropy-regularization must be >= 0")
    if not 0.0 <= forced_label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if forced_sampler_mode not in {"interleaved_rr", "weighted_random_sampler"}:
        raise ValueError(
            "--sampler-mode must be one of {'interleaved_rr', 'weighted_random_sampler'}"
        )
    if int(args.min_class4_samples) < 0:
        raise ValueError("--min-class4-samples must be >= 0")
    if int(args.class4_per_batch_min) < 0:
        raise ValueError("--class4-per-batch-min must be >= 0")
    if forced_family_margin_loss_weight < 0.0:
        raise ValueError("--family-margin-loss-weight must be >= 0")
    if forced_family_class4_logit_penalty_weight < 0.0:
        raise ValueError("--family-class4-logit-penalty-weight must be >= 0")
    if forced_family_feature_separation_weight < 0.0:
        raise ValueError("--family-feature-separation-weight must be >= 0")
    if forced_family_class4_target_scale < 0.0 or forced_family_class4_target_scale > 1.0:
        raise ValueError("--family-class4-target-scale must be in [0, 1]")

    if args.dataset == "unsw_nb15" and args.epochs < 10:
        raise ValueError("--epochs must be >= 10 for UNSW-stable training signal")

    train_config.log_interval = int(args.log_interval)
    train_config.lambda_family = float(forced_lambda_family)
    train_config.class_balance_strategy = forced_class_balance_strategy
    train_config.use_class_weights = bool(forced_use_class_weights)
    train_config.focal_gamma = forced_focal_gamma
    train_config.enable_logit_adjustment = forced_use_logit_prior_correction
    train_config.logit_temp = forced_train_temperature
    train_config.min_class_prob_eps = forced_min_class_prob_eps

    config_payload = {
        "batch_size": train_config.batch_size,
        "epochs": train_config.epochs,
        "learning_rate": train_config.learning_rate,
        "warmup_epochs": train_config.warmup_epochs,
        "grad_clip": train_config.max_grad_norm,
        "lambda_binary": forced_lambda_binary,
        "lambda_family": train_config.lambda_family,
        "class_balance_strategy": forced_class_balance_strategy,
        "use_class_weights": bool(forced_use_class_weights),
        "focal_gamma": forced_focal_gamma,
        "label_smoothing": forced_label_smoothing,
        "use_logit_prior_correction": forced_use_logit_prior_correction,
        "train_temperature": forced_train_temperature,
        "warmup_ratio": forced_warmup_ratio,
        "freeze_backbone_epochs": forced_freeze_backbone_epochs,
        "unfreeze_backbone_step": forced_unfreeze_backbone_step,
        "entropy_warmup_steps": forced_entropy_warmup_steps,
        "entropy_warmup_weight": forced_entropy_warmup_weight,
        "head_lr_multiplier": forced_head_lr_multiplier,
        "supcon_weight": forced_supcon_weight,
        "supcon_temperature": forced_supcon_temperature,
        "step_coverage_check_step": forced_step_coverage_check_step,
        "representation_diagnostic_mode": forced_representation_diagnostic_mode,
        "use_energy_based_family_objective": bool(
            forced_use_energy_based_family_objective
        ),
        "representation_only_steps": forced_representation_only_steps,
        "representation_micro_cycle_steps": [
            int(v) for v in forced_representation_micro_cycle_steps
        ],
        "adaptive_exit_ratio_threshold": forced_adaptive_exit_ratio_threshold,
        "adaptive_exit_min_inter_threshold": forced_adaptive_exit_min_inter_threshold,
        "representation_only_ratio": forced_representation_only_ratio,
        "head_only_ratio": forced_head_only_ratio,
        "joint_finetune_backbone_lr_multiplier": forced_joint_finetune_backbone_lr_multiplier,
        "joint_finetune_head_lr_multiplier": forced_joint_finetune_head_lr_multiplier,
        "cluster_relabeling_enabled": forced_cluster_relabeling_enabled,
        "cluster_relabel_k": forced_cluster_relabel_k,
        "cluster_relabel_seed": forced_cluster_relabel_seed,
        "cluster_relabel_objective": forced_cluster_relabel_objective,
        "cluster_relabel_spectral_affinity": forced_cluster_relabel_spectral_affinity,
        "sampler_mode": forced_sampler_mode,
        "enforce_all_classes_per_batch": forced_enforce_all_classes_per_batch,
        "geometry_ratio_warmup_threshold": forced_geometry_ratio_warmup_threshold,
        "geometry_ratio_post_phase_threshold": forced_geometry_ratio_post_phase_threshold,
        "num_workers": forced_num_workers,
        "ab_mode": bool(args.ab_mode),
        "ab_track": str(args.ab_track),
        "ab_change_id": str(args.ab_change_id),
        "ab_require_baseline": bool(args.ab_require_baseline),
        "nsl_kdd_label_merges": [
            {"src": int(src), "dst": int(dst)}
            for src, dst in forced_nsl_kdd_label_merges
        ],
        "kl_weight_warmup": 0.0,
        "kl_weight_post_warmup": 0.0,
        "logit_floor_weight": 0.0,
        "tail_ce_weight": 0.0,
        "snapshot_mode": str(args.snapshot_mode),
        "allow_unfrozen_snapshot": bool(args.allow_unfrozen_snapshot),
        "min_class_prob_eps": forced_min_class_prob_eps,
        "entropy_regularization": forced_entropy_regularization,
        "family_margin_loss_weight": float(forced_family_margin_loss_weight),
        "family_class4_logit_penalty_weight": float(forced_family_class4_logit_penalty_weight),
        "family_feature_separation_weight": float(forced_family_feature_separation_weight),
        "family_class4_target_scale": float(forced_family_class4_target_scale),
        "family_logit_margin": 1.0,
        "class4_logit_shift": float(args.class4_logit_shift),
        "class4_per_batch_min": int(args.class4_per_batch_min),
        "device": args.device,
        "phase_regime": phase_regime,
        "training_mode": "head_isolation_ce_warmstart",
    }

    train_config.lambda_binary = float(forced_lambda_binary)
    train_config.num_workers = int(forced_num_workers)
    train_config.freeze_backbone_epochs = int(forced_freeze_backbone_epochs)  # type: ignore[attr-defined]
    train_config.unfreeze_backbone_step = int(forced_unfreeze_backbone_step)  # type: ignore[attr-defined]
    train_config.entropy_warmup_steps = int(forced_entropy_warmup_steps)  # type: ignore[attr-defined]
    train_config.entropy_warmup_weight = float(forced_entropy_warmup_weight)  # type: ignore[attr-defined]
    _apply_disable_early_stopping(
        train_config,
        disable_early_stopping=bool(args.disable_early_stopping),
    )

    eval_batch_size = int(args.eval_batch_size or train_config.batch_size)
    if eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be >= 1")

    return ParsedConfig(
        args=args,
        train_config=train_config,
        config_payload=config_payload,
        phase_regime=phase_regime,
        calibration_enabled=calibration_enabled,
        governance_only_mode=governance_only_mode,
        forced_class_balance_strategy=forced_class_balance_strategy,
        forced_use_class_weights=forced_use_class_weights,
        forced_focal_gamma=forced_focal_gamma,
        forced_label_smoothing=forced_label_smoothing,
        forced_use_logit_prior_correction=forced_use_logit_prior_correction,
        forced_train_temperature=forced_train_temperature,
        forced_min_class_prob_eps=forced_min_class_prob_eps,
        forced_entropy_regularization=forced_entropy_regularization,
        forced_warmup_ratio=forced_warmup_ratio,
        forced_lambda_family=forced_lambda_family,
        forced_freeze_backbone_epochs=forced_freeze_backbone_epochs,
        forced_unfreeze_backbone_step=forced_unfreeze_backbone_step,
        forced_entropy_warmup_steps=forced_entropy_warmup_steps,
        forced_entropy_warmup_weight=forced_entropy_warmup_weight,
        forced_head_lr_multiplier=forced_head_lr_multiplier,
        forced_lambda_binary=forced_lambda_binary,
        forced_supcon_weight=forced_supcon_weight,
        forced_supcon_temperature=forced_supcon_temperature,
        forced_step_coverage_check_step=forced_step_coverage_check_step,
        forced_representation_diagnostic_mode=forced_representation_diagnostic_mode,
        forced_representation_only_steps=forced_representation_only_steps,
        forced_representation_micro_cycle_steps=forced_representation_micro_cycle_steps,
        forced_use_energy_based_family_objective=forced_use_energy_based_family_objective,
        forced_adaptive_exit_ratio_threshold=forced_adaptive_exit_ratio_threshold,
        forced_adaptive_exit_min_inter_threshold=forced_adaptive_exit_min_inter_threshold,
        forced_representation_only_ratio=forced_representation_only_ratio,
        forced_head_only_ratio=forced_head_only_ratio,
        forced_joint_finetune_backbone_lr_multiplier=forced_joint_finetune_backbone_lr_multiplier,
        forced_joint_finetune_head_lr_multiplier=forced_joint_finetune_head_lr_multiplier,
        forced_cluster_relabeling_enabled=forced_cluster_relabeling_enabled,
        forced_cluster_relabel_k=forced_cluster_relabel_k,
        forced_cluster_relabel_seed=forced_cluster_relabel_seed,
        forced_cluster_relabel_objective=forced_cluster_relabel_objective,
        forced_cluster_relabel_spectral_affinity=forced_cluster_relabel_spectral_affinity,
        forced_sampler_mode=forced_sampler_mode,
        forced_family_margin_loss_weight=forced_family_margin_loss_weight,
        forced_family_class4_logit_penalty_weight=forced_family_class4_logit_penalty_weight,
        forced_family_feature_separation_weight=forced_family_feature_separation_weight,
        forced_family_class4_target_scale=forced_family_class4_target_scale,
        forced_enforce_all_classes_per_batch=forced_enforce_all_classes_per_batch,
        forced_geometry_ratio_warmup_threshold=forced_geometry_ratio_warmup_threshold,
        forced_geometry_ratio_post_phase_threshold=forced_geometry_ratio_post_phase_threshold,
        forced_nsl_kdd_label_merges=forced_nsl_kdd_label_merges,
        forced_num_workers=forced_num_workers,
    )
