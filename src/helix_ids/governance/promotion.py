"""Promotion consensus coordinator and multi-seed aggregation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Callable, Iterable


@dataclass(frozen=True)
class SeedRunSummary:
    """Per-seed summary required for promotion consensus."""

    seed: int
    macro_f1: float
    macro_f1_ci_lower: float
    macro_f1_ci_width: float
    tier2_pass: bool


@dataclass(frozen=True)
class PromotionConsensus:
    """Aggregated promotion telemetry across seed runs."""

    seed_run_count: int
    mean_macro_f1: float
    std_dev: float
    inter_seed_macro_f1_variance: float
    reproducibility_delta: float
    consensus_pass: bool
    invalid_reason: str | None

    def to_stage_metrics(self) -> dict[str, float | int | bool]:
        return {
            "seed_run_count": self.seed_run_count,
            "mean_macro_f1": self.mean_macro_f1,
            "inter_seed_macro_f1_std": self.std_dev,
            "inter_seed_macro_f1_variance": self.inter_seed_macro_f1_variance,
            "reproducibility_delta": self.reproducibility_delta,
            "consensus_pass": self.consensus_pass,
        }


def _validate_seed_runs(seed_runs: Iterable[SeedRunSummary]) -> list[SeedRunSummary]:
    runs = list(seed_runs)
    for run in runs:
        metrics = (run.macro_f1, run.macro_f1_ci_lower, run.macro_f1_ci_width)
        if not all(math.isfinite(float(value)) for value in metrics):
            raise ValueError("E-T3-NONFINITE-SEED-METRIC-INVALID")
    return runs


def aggregate_seed_runs(
    seed_runs: Iterable[SeedRunSummary],
    *,
    min_seed_runs: int,
    max_inter_seed_macro_f1_variance: float,
    reproducibility_tolerance: float,
    min_ci95_lower_bound: float,
    max_ci_width: float,
) -> PromotionConsensus:
    """Aggregate multi-seed runs and derive non-bypassable promotion consensus fields."""
    runs = _validate_seed_runs(seed_runs)
    seed_run_count = len(runs)

    if seed_run_count < min_seed_runs:
        return PromotionConsensus(
            seed_run_count=seed_run_count,
            mean_macro_f1=0.0,
            std_dev=0.0,
            inter_seed_macro_f1_variance=0.0,
            reproducibility_delta=0.0,
            consensus_pass=False,
            invalid_reason="E-T3-SINGLE-SEED-INVALID",
        )

    macro_values = [float(run.macro_f1) for run in runs]
    mean_macro_f1 = mean(macro_values)
    std_dev = pstdev(macro_values)
    variance = float(std_dev**2)
    reproducibility_delta = float(max(macro_values) - min(macro_values))

    tier2_all_pass = all(
        run.tier2_pass
        and float(run.macro_f1_ci_lower) >= min_ci95_lower_bound
        and float(run.macro_f1_ci_width) <= max_ci_width
        for run in runs
    )
    variance_ok = variance <= max_inter_seed_macro_f1_variance
    reproducibility_ok = reproducibility_delta <= reproducibility_tolerance

    invalid_reason = None
    if not tier2_all_pass:
        invalid_reason = "E-T3-TIER2-CONSENSUS-INVALID"
    elif not variance_ok:
        invalid_reason = "E-T3-SEED-VARIANCE-INVALID"
    elif not reproducibility_ok:
        invalid_reason = "E-T3-REPRODUCIBILITY-INVALID"

    return PromotionConsensus(
        seed_run_count=seed_run_count,
        mean_macro_f1=mean_macro_f1,
        std_dev=std_dev,
        inter_seed_macro_f1_variance=variance,
        reproducibility_delta=reproducibility_delta,
        consensus_pass=invalid_reason is None,
        invalid_reason=invalid_reason,
    )


def execute_multi_seed_consensus(
    *,
    seeds: Iterable[int],
    run_seed_fn: Callable[[int], SeedRunSummary],
    min_seed_runs: int,
    max_inter_seed_macro_f1_variance: float,
    reproducibility_tolerance: float,
    min_ci95_lower_bound: float,
    max_ci_width: float,
) -> tuple[list[SeedRunSummary], PromotionConsensus]:
    """Execute configured seeds and aggregate non-bypassable promotion consensus."""
    seed_values = list(seeds)
    if len(seed_values) < min_seed_runs:
        raise ValueError("E-T3-SEED-PLAN-TOO-SMALL-INVALID")

    runs = [run_seed_fn(seed) for seed in seed_values]
    consensus = aggregate_seed_runs(
        runs,
        min_seed_runs=min_seed_runs,
        max_inter_seed_macro_f1_variance=max_inter_seed_macro_f1_variance,
        reproducibility_tolerance=reproducibility_tolerance,
        min_ci95_lower_bound=min_ci95_lower_bound,
        max_ci_width=max_ci_width,
    )
    return runs, consensus
