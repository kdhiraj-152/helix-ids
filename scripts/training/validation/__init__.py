"""Evaluation and calibration utilities for HELIX-IDS full training pipeline.

Phase 16: validation/coverage integrity checks, calibration, artifact generation.
All validation decision logic resides here — fully delegated from HelixFullTrainer.
"""

# Canonical implementations live in governance/ (Phase 13A-5 deduplication).
from scripts.training.governance.ab_testing import (  # noqa: F401  -- re-export alias
    ab_rejection as _ab_rejection,
)
from scripts.training.governance.ab_testing import (
    build_ab_raw_metrics as _build_ab_raw_metrics,
)
from scripts.training.governance.ab_testing import (
    detect_feature_and_objective_changes as _detect_feature_and_objective_changes,
)
from scripts.training.governance.ab_testing import (
    evaluate_ab_candidate,
)
from scripts.training.governance.ab_testing import (
    validate_ab_contract as _validate_ab_contract,
)
from scripts.training.governance.ab_testing import (
    validate_track as _validate_track,
)
from scripts.training.governance.orchestrator import (  # noqa: F401  -- re-export alias
    materialize_phase8_artifacts as _materialize_phase8_artifacts,
)
from scripts.training.governance.orchestrator import (
    normalize_calibration_block as _normalize_calibration_block,
)

from .artifacts import (
    _atomic_write_json,
    _emit_calibration_artifacts,
)
from .calibrator import (
    _apply_class4_logit_shift,
    _calibrate_family_predictions,
    _fit_temperature_nll,
    _predict_with_class4_threshold,
    _softmax_with_temperature,
)
from .evaluator import (
    _collect_eval_family_outputs,
    _normalized_entropy_from_probs,
)
from .validation_orchestrator import (
    ValidationOrchestrator,
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
    # orchestrator
    "ValidationOrchestrator",
]
