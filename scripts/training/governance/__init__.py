"""Extracted governance orchestration for HelixIDS-Full (Phase 12B-7).

Replaces governance logic formerly embedded in train_helix_ids_full.py:
  - _load_seed_run_artifacts
  - _normalize_metrics_payload
  - _summarize_governance
  - _materialize_phase8_artifacts
  - _normalize_calibration_block
  - evaluate_ab_candidate
  - _build_ab_raw_metrics
  - _ab_rejection
  - _validate_ab_contract
  - _detect_feature_and_objective_changes
  - _validate_track
  - _detect_cluster_mode_collapse
  - _normalized_entropy_from_counts
  - _coerce_finite_float
  - _load_json_dict

Usage (delegation from train script -> governance package):
    from scripts.training.governance import (
        load_seed_run_artifacts,
        normalize_metrics_payload,
        summarize_governance,
        evaluate_ab_candidate,
        ...
    )
"""

from __future__ import annotations

from scripts.training.governance.ab_testing import (
    ABEvaluationInput,
    ABEvaluationResult,
    ab_rejection,
    build_ab_raw_metrics,
    detect_cluster_mode_collapse,
    detect_feature_and_objective_changes,
    evaluate_ab_candidate,
    normalized_entropy_from_counts,
    validate_ab_contract,
    validate_track,
)
from scripts.training.governance.orchestrator import (
    CoerceFloatError,
    coerce_finite_float,
    load_json_dict,
    load_seed_run_artifacts,
    materialize_phase8_artifacts,
    normalize_calibration_block,
    normalize_metrics_payload,
)
from scripts.training.governance.promotion import (
    PromotionInput,
    PromotionResult,
    build_promotion_result,
)
from scripts.training.governance.reporting import (
    GovernanceSummary,
    GovernanceSummaryInput,
    summarize_governance,
)

__all__ = [
    "CoerceFloatError",
    "coerce_finite_float",
    "load_json_dict",
    "load_seed_run_artifacts",
    "materialize_phase8_artifacts",
    "normalize_calibration_block",
    "normalize_metrics_payload",
    "PromotionInput",
    "PromotionResult",
    "build_promotion_result",
    "ABEvaluationInput",
    "ABEvaluationResult",
    "ab_rejection",
    "build_ab_raw_metrics",
    "detect_cluster_mode_collapse",
    "detect_feature_and_objective_changes",
    "evaluate_ab_candidate",
    "normalized_entropy_from_counts",
    "validate_ab_contract",
    "validate_track",
    "GovernanceSummary",
    "GovernanceSummaryInput",
    "summarize_governance",
]
