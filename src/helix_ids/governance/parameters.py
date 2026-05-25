"""Centralized governance parameters for deterministic gate execution."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StageTimeouts:
    """Hard timeout contract by orchestrator stage."""

    preload_seconds: int = 15 * 60
    presplit_seconds: int = 20 * 60
    pretrain_seconds: int = 20 * 60
    intrain_seconds: int = 6 * 60 * 60
    posteval_seconds: int = 30 * 60
    prepromote_seconds: int = 10 * 60


@dataclass(frozen=True)
class TrainingAbortPolicy:
    """Fixed abort thresholds to avoid ambiguous early-stop behavior."""

    low_entropy_threshold: float = 0.30
    low_entropy_consecutive_batches: int = 100
    gradient_dominance_threshold: float = 0.60
    gradient_dominance_window_epochs: int = 2
    epochs_without_improvement: int = 12


@dataclass(frozen=True)
class BootstrapPolicy:
    """Deterministic bootstrap CI parameters for macro-F1."""

    n_replicates: int = 2000
    lower_percentile: float = 2.5
    upper_percentile: float = 97.5
    max_ci_width: float = 0.05
    min_ci95_lower_bound: float = 0.50
    seed_offset: int = 7919


@dataclass(frozen=True)
class DriftPolicy:
    """Cross-run drift and anomaly thresholds."""

    baseline_window_runs: int = 20
    min_runs_for_promotion: int = 10
    max_abs_macro_f1_drift: float = 0.05
    max_abs_z_score: float = 2.5


@dataclass(frozen=True)
class DatasetIdentityLeakagePolicy:
    """Tier-0 dataset identity leakage classifier specification."""

    model_name: str = "multinomial_logistic_regression"
    test_size: float = 0.20
    random_state: int = 42
    max_balanced_accuracy: float = 0.90


@dataclass(frozen=True)
class PromotionPolicy:
    """Final promotion gate requirements."""

    min_seed_runs: int = 3
    max_inter_seed_macro_f1_variance: float = 0.01
    reproducibility_tolerance: float = 0.01


@dataclass(frozen=True)
class GovernancePolicy:
    """Top-level immutable policy object consumed by GateOrchestrator."""

    stage_timeouts: StageTimeouts = field(default_factory=StageTimeouts)
    training_abort: TrainingAbortPolicy = field(default_factory=TrainingAbortPolicy)
    bootstrap: BootstrapPolicy = field(default_factory=BootstrapPolicy)
    drift: DriftPolicy = field(default_factory=DriftPolicy)
    dataset_identity: DatasetIdentityLeakagePolicy = field(
        default_factory=DatasetIdentityLeakagePolicy
    )
    promotion: PromotionPolicy = field(default_factory=PromotionPolicy)


DEFAULT_GOVERNANCE_POLICY = GovernancePolicy()
