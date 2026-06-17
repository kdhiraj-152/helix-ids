"""Data-building utilities for HELIX-IDS full training pipeline."""

# Canonical implementations live in governance/ (Phase 13A-5 deduplication).
from scripts.training.governance.ab_testing import (  # noqa: F401  -- re-export alias
    detect_cluster_mode_collapse as _detect_cluster_mode_collapse,
)
from scripts.training.governance.ab_testing import (
    normalized_entropy_from_counts as _normalized_entropy_from_counts,
)
from scripts.training.governance.orchestrator import (  # noqa: F401  -- re-export alias
    coerce_finite_float as _coerce_finite_float,
)
from scripts.training.governance.orchestrator import (
    normalize_metrics_payload as _normalize_metrics_payload,
)

from .dataset_builder import (
    MultiTaskNumpyDataset,
    _build_stratified_subset_indices,
    _build_stratified_val_subset,
    _chunk_finite_check,
    _sample_rows,
    build_class_index,
)
from .samplers import (
    ClassBalancedIndexSampler,
    FrozenIndexSampler,
    _build_frozen_class_balanced_indices,
    _build_frozen_tempered_indices,
    _default_tail_multiplier,
    _inverse_frequency_weights,
    _sqrt_inverse_frequency_weights,
)
from .validators import (
    _apply_label_merges,
    _assert_categorical_encoding_sanity,
    _assert_feature_dimensions,
    _assert_feature_sanity_for_dataset,
    _assert_numeric_finite_and_variance,
    _compute_class4_metrics,
    _compute_multiclass_confusion,
    _normalize_engineered_feature_block,
    _summarize_prediction_coverage,
)

__all__ = [
    "MultiTaskNumpyDataset",
    "ClassBalancedIndexSampler",
    "FrozenIndexSampler",
    "build_class_index",
    "_apply_label_merges",
    "_assert_categorical_encoding_sanity",
    "_assert_feature_dimensions",
    "_assert_feature_sanity_for_dataset",
    "_assert_numeric_finite_and_variance",
    "_build_frozen_class_balanced_indices",
    "_build_frozen_tempered_indices",
    "_build_stratified_subset_indices",
    "_build_stratified_val_subset",
    "_chunk_finite_check",
    "_coerce_finite_float",
    "_compute_class4_metrics",
    "_compute_multiclass_confusion",
    "_default_tail_multiplier",
    "_detect_cluster_mode_collapse",
    "_inverse_frequency_weights",
    "_normalize_engineered_feature_block",
    "_normalize_metrics_payload",
    "_normalized_entropy_from_counts",
    "_sample_rows",
    "_sqrt_inverse_frequency_weights",
    "_summarize_prediction_coverage",
]
