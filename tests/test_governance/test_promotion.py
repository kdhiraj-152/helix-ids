"""Tests for multi-seed promotion consensus aggregation."""

from __future__ import annotations

from helix_ids.governance.promotion import SeedRunSummary, aggregate_seed_runs


def test_aggregate_seed_runs_invalid_for_single_seed():
    consensus = aggregate_seed_runs(
        [
            SeedRunSummary(
                seed=42,
                macro_f1=0.8,
                macro_f1_ci_lower=0.7,
                macro_f1_ci_width=0.02,
                tier2_pass=True,
            )
        ],
        min_seed_runs=3,
        max_inter_seed_macro_f1_variance=0.01,
        reproducibility_tolerance=0.01,
        min_ci95_lower_bound=0.5,
        max_ci_width=0.05,
    )

    assert consensus.consensus_pass is False
    assert consensus.invalid_reason == "E-T3-SINGLE-SEED-INVALID"


def test_aggregate_seed_runs_passes_with_three_consistent_seeds():
    consensus = aggregate_seed_runs(
        [
            SeedRunSummary(seed=41, macro_f1=0.81, macro_f1_ci_lower=0.7, macro_f1_ci_width=0.02, tier2_pass=True),
            SeedRunSummary(seed=42, macro_f1=0.815, macro_f1_ci_lower=0.71, macro_f1_ci_width=0.02, tier2_pass=True),
            SeedRunSummary(seed=43, macro_f1=0.812, macro_f1_ci_lower=0.705, macro_f1_ci_width=0.02, tier2_pass=True),
        ],
        min_seed_runs=3,
        max_inter_seed_macro_f1_variance=0.01,
        reproducibility_tolerance=0.01,
        min_ci95_lower_bound=0.5,
        max_ci_width=0.05,
    )

    assert consensus.consensus_pass is True
    assert consensus.seed_run_count == 3
    assert consensus.invalid_reason is None
