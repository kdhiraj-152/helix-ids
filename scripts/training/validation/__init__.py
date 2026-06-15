"""Evaluation and calibration utilities for HELIX-IDS full training pipeline.

Phase 12B-4 extraction: evaluation-only logic extracted from
train_helix_ids_full.py into well-separated validation/calibration boundary.
"""

from .artifacts import (
    _atomic_write_json,
    _emit_calibration_artifacts,
    _materialize_phase8_artifacts,
    _normalize_calibration_block,
)
from .calibrator import (
    _apply_class4_logit_shift,
    _calibrate_family_predictions,
    _fit_temperature_nll,
    _predict_with_class4_threshold,
    _softmax_with_temperature,
)
from .evaluator import (
    _ab_rejection,
    _build_ab_raw_metrics,
    _collect_eval_family_outputs,
    _detect_feature_and_objective_changes,
    _normalized_entropy_from_probs,
    _validate_ab_contract,
    _validate_track,
    evaluate_ab_candidate,
)

__all__ = [
    # evaluator
    "_ab_rejection",
    "_build_ab_raw_metrics",
    "_collect_eval_family_outputs",
    "_detect_feature_and_objective_changes",
    "_normalized_entropy_from_probs",
    "_validate_ab_contract",
    "_validate_track",
    "evaluate_ab_candidate",
    # calibrator
    "_apply_class4_logit_shift",
    "_calibrate_family_predictions",
    "_fit_temperature_nll",
    "_predict_with_class4_threshold",
    "_softmax_with_temperature",
    # artifacts
    "_atomic_write_json",
    "_emit_calibration_artifacts",
    "_materialize_phase8_artifacts",
    "_normalize_calibration_block",
]
