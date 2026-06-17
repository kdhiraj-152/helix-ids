"""Pure validation and helper functions for HELIX-IDS full training pipeline.

Extracted from scripts/training/train_helix_ids_full.py — Phase 12B-3.
No behavioral changes. These functions have no model state, optimizer,
or trainer lifecycle dependencies.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

# Canonical implementations live in scripts/training/governance/ (Phase 13A-5).
# Compatibility aliases preserve the private-name public API.
from scripts.training.governance import (  # noqa: F401
    coerce_finite_float as _coerce_finite_float,
)
from scripts.training.governance.ab_testing import (  # noqa: F401
    detect_cluster_mode_collapse as _detect_cluster_mode_collapse,
)
from scripts.training.governance.orchestrator import (  # noqa: F401
    normalize_metrics_payload as _normalize_metrics_payload,
)

from .dataset_builder import _chunk_finite_check, _sample_rows

# ============================================================================
# Label manipulation helpers
# ============================================================================


def _apply_label_merges(
    y: np.ndarray,
    *,
    merges: list[tuple[int, int]],
) -> np.ndarray:
    """Apply deterministic label merges (src -> dst) to family labels."""
    y_int: np.ndarray = np.asarray(y, dtype=np.int64).copy()
    if not merges:
        return y_int

    merge_map = {int(src): int(dst) for src, dst in merges}

    def _resolve(label: int) -> int:
        seen: set[int] = set()
        cur = int(label)
        while cur in merge_map and cur not in seen:
            seen.add(cur)
            cur = int(merge_map[cur])
        return cur

    if merge_map:
        remap = {src: _resolve(src) for src in merge_map}
        for src, dst in remap.items():
            y_int[y_int == int(src)] = int(dst)

    return y_int


# ============================================================================
# Classification metrics (pure numpy, no model dependency)
# ============================================================================


def _compute_multiclass_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_count: int,
) -> np.ndarray:
    """Compute multiclass confusion matrix with fixed class space."""
    conf = np.zeros((int(class_count), int(class_count)), dtype=np.int64)
    if y_true.size == 0:
        return conf
    y_t = np.clip(np.asarray(y_true, dtype=np.int64), 0, int(class_count) - 1)
    y_p = np.clip(np.asarray(y_pred, dtype=np.int64), 0, int(class_count) - 1)
    np.add.at(conf, (y_t, y_p), 1)
    return conf


def _compute_class4_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class4_id: int,
) -> dict[str, float]:
    """Compute class-4 precision/recall from predicted labels."""
    if y_true.size == 0:
        return {"class4_precision": 0.0, "class4_recall": 0.0}
    pos_true = np.asarray(y_true, dtype=np.int64) == int(class4_id)
    pos_pred = np.asarray(y_pred, dtype=np.int64) == int(class4_id)
    tp = float(np.sum(pos_true & pos_pred))
    fp = float(np.sum((~pos_true) & pos_pred))
    fn = float(np.sum(pos_true & (~pos_pred)))
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    return {"class4_precision": float(precision), "class4_recall": float(recall)}


def _summarize_prediction_coverage(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """Count classes present in truth but never predicted."""
    if y_true.size == 0:
        return 0
    present = {int(v) for v in np.unique(y_true).tolist()}
    predicted = {int(v) for v in np.unique(y_pred).tolist()}
    return int(len(sorted(present - predicted)))


# ============================================================================
# Feature validation guards
# ============================================================================


def _normalize_engineered_feature_block(
    *,
    dataset_name: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    feature_names: list[str],
    engineered_feature_names: set[str],
    min_feature_std: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    """Apply train-fit z-score normalization to engineered geometry features only."""
    engineered_indices = [
        idx
        for idx, name in enumerate(feature_names)
        if str(name).strip().lower() in engineered_feature_names
    ]
    if not engineered_indices:
        return (
            np.asarray(x_train, dtype=np.float32),
            np.asarray(x_val, dtype=np.float32),
            np.asarray(x_test, dtype=np.float32),
            {},
        )

    train = np.asarray(x_train, dtype=np.float32).copy()
    val = np.asarray(x_val, dtype=np.float32).copy()
    test = np.asarray(x_test, dtype=np.float32).copy()

    stats: dict[str, dict[str, float]] = {}
    for idx in engineered_indices:
        feature_name = str(feature_names[idx])
        train_col = np.asarray(train[:, idx], dtype=np.float64)
        if not np.isfinite(train_col).all():
            raise RuntimeError(
                "Hard-stop integrity guard triggered: engineered_feature_non_finite_train_"
                f"{dataset_name}:{feature_name}"
            )

        mean = float(np.mean(train_col))
        std = float(np.std(train_col))
        scale = max(min_feature_std, std)

        train[:, idx] = ((train[:, idx] - mean) / scale).astype(np.float32, copy=False)
        val[:, idx] = ((val[:, idx] - mean) / scale).astype(np.float32, copy=False)
        test[:, idx] = ((test[:, idx] - mean) / scale).astype(np.float32, copy=False)

        stats[feature_name] = {
            "train_mean": float(np.mean(train[:, idx])),
            "train_std": float(np.std(train[:, idx])),
            "train_p99_abs": float(np.percentile(np.abs(train[:, idx]), 99.0)),
        }

    return train, val, test, stats


def _assert_categorical_encoding_sanity(
    *,
    dataset_name: str,
    feature_names: list[str],
    train_sample: np.ndarray,
    val_sample: np.ndarray,
) -> None:
    categorical_feature_names = {
        "protocol",
        "protocol_type",
        "service",
        "flag",
        "state",
    }
    categorical_indices = [
        idx
        for idx, name in enumerate(feature_names)
        if str(name).strip().lower() in categorical_feature_names
    ]
    for idx in categorical_indices:
        feature_name = str(feature_names[idx])
        train_col = np.asarray(train_sample[:, idx], dtype=np.float64)
        val_col = np.asarray(val_sample[:, idx], dtype=np.float64)

        if not np.isfinite(train_col).all() or not np.isfinite(val_col).all():
            raise RuntimeError(
                "Hard-stop integrity guard triggered: categorical_non_finite_values_"
                f"{dataset_name}:{feature_name}"
            )

        train_integer_like = float(np.mean(np.abs(train_col - np.rint(train_col)) < 1e-6))
        val_integer_like = float(np.mean(np.abs(val_col - np.rint(val_col)) < 1e-6))
        if train_integer_like < 0.999 or val_integer_like < 0.999:
            raise RuntimeError(
                "Hard-stop integrity guard triggered: categorical_not_integer_encoded_"
                f"{dataset_name}:{feature_name}"
            )

        train_codes = np.rint(train_col).astype(np.int64, copy=False)
        if train_codes.size == 0:
            continue
        unique_codes = np.unique(train_codes)
        if unique_codes.size > 4096:
            raise RuntimeError(
                "Hard-stop integrity guard triggered: categorical_code_cardinality_too_high_"
                f"{dataset_name}:{feature_name}:{unique_codes.size}"
            )


def _assert_feature_dimensions(
    *,
    dataset_name: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    feature_names: list[str],
    expected_feature_dim: Optional[int],
) -> None:
    """Validate rank and feature-dimension alignment for train/val matrices."""
    if x_train.ndim != 2:
        raise RuntimeError(
            f"Hard-stop integrity guard triggered: invalid_train_feature_rank_{dataset_name}"
        )
    if x_val.ndim != 2:
        raise RuntimeError(
            f"Hard-stop integrity guard triggered: invalid_val_feature_rank_{dataset_name}"
        )
    if expected_feature_dim is not None:
        if int(x_train.shape[1]) != int(expected_feature_dim):
            raise RuntimeError(
                "Hard-stop integrity guard triggered: feature_dim_not_expected_"
                f"{dataset_name}:expected_{int(expected_feature_dim)}_got_{int(x_train.shape[1])}"
            )
        if int(x_val.shape[1]) != int(expected_feature_dim):
            raise RuntimeError(
                "Hard-stop integrity guard triggered: val_feature_dim_not_expected_"
                f"{dataset_name}:expected_{int(expected_feature_dim)}_got_{int(x_val.shape[1])}"
            )
    if int(x_train.shape[1]) != len(feature_names):
        raise RuntimeError(
            "Hard-stop integrity guard triggered: feature_name_count_mismatch_"
            f"{dataset_name}"
        )
    if not np.issubdtype(np.asarray(x_train).dtype, np.number):
        raise RuntimeError(
            f"Hard-stop integrity guard triggered: non_numeric_train_features_{dataset_name}"
        )


def _assert_numeric_finite_and_variance(
    *,
    dataset_name: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    feature_names: list[str],
    min_feature_std: float,
) -> None:
    """Validate full-array finiteness and variance floor before sampling checks."""

    if not _chunk_finite_check(np.asarray(x_train, dtype=np.float32)) or not _chunk_finite_check(
        np.asarray(x_val, dtype=np.float32)
    ):
        raise RuntimeError(
            f"Hard-stop integrity guard triggered: non_finite_feature_values_{dataset_name}"
        )

    full_feature_std = np.nanstd(np.asarray(x_train, dtype=np.float32), axis=0)
    low_std_idx = np.nonzero(full_feature_std < float(min_feature_std))[0]
    if low_std_idx.size > 0:
        low_std_features = [feature_names[int(idx)] for idx in low_std_idx[:10].tolist()]
        raise RuntimeError(
            "Hard-stop integrity guard triggered: low_variance_features_present_"
            f"{dataset_name}: threshold={float(min_feature_std):.2e} features={low_std_features}"
        )


def _assert_feature_sanity_for_dataset(
    *,
    dataset_name: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    feature_names: list[str],
    expected_feature_dim: Optional[int],
    min_feature_std: float,
    seed: int,
    logger: Any,
) -> None:
    """Validate feature integrity: no constants, sane scaling, and encoded categoricals."""
    _assert_feature_dimensions(
        dataset_name=dataset_name,
        x_train=x_train,
        x_val=x_val,
        feature_names=feature_names,
        expected_feature_dim=expected_feature_dim,
    )
    _assert_numeric_finite_and_variance(
        dataset_name=dataset_name,
        x_train=x_train,
        x_val=x_val,
        feature_names=feature_names,
        min_feature_std=min_feature_std,
    )

    train_sample = _sample_rows(x_train, seed=seed)
    val_sample = _sample_rows(x_val, seed=seed + 1)

    if not np.isfinite(train_sample).all() or not np.isfinite(val_sample).all():
        raise RuntimeError(
            f"Hard-stop integrity guard triggered: non_finite_feature_values_{dataset_name}"
        )

    feature_std = np.nanstd(train_sample, axis=0)
    constant_idx = np.nonzero(feature_std <= 1e-12)[0]
    if constant_idx.size > 0:
        constant_features = [feature_names[int(idx)] for idx in constant_idx[:10].tolist()]
        raise RuntimeError(
            "Hard-stop integrity guard triggered: constant_features_present_"
            f"{dataset_name}: {constant_features}"
        )

    _assert_categorical_encoding_sanity(
        dataset_name=dataset_name,
        feature_names=feature_names,
        train_sample=train_sample,
        val_sample=val_sample,
    )

    train_scale = float(np.nanpercentile(np.abs(train_sample), 99))
    val_scale = float(np.nanpercentile(np.abs(val_sample), 99))
    if not np.isfinite(train_scale) or not np.isfinite(val_scale):
        raise RuntimeError(
            f"Hard-stop integrity guard triggered: invalid_scale_statistics_{dataset_name}"
        )
    if train_scale > 1e4 or val_scale > 1e4:
        raise RuntimeError(
            f"Hard-stop integrity guard triggered: scale_normalization_not_applied_{dataset_name}"
        )

    min_scale = max(1e-8, float(min(train_scale, val_scale)))
    scale_ratio = max(train_scale, val_scale) / min_scale
    if scale_ratio > 20.0:
        raise RuntimeError(
            "Hard-stop integrity guard triggered: cross_split_scale_mismatch_"
            f"{dataset_name}"
        )

    integer_like_fraction = np.mean(np.abs(train_sample - np.rint(train_sample)) < 1e-6, axis=0)
    integer_like_count = int(np.sum(integer_like_fraction >= 0.999))

    logger.info(
        "FeatureSanity[%s] constant_features=0 scale_p99(train)=%.4f scale_p99(val)=%.4f "
        "scale_ratio=%.3f integer_like_features=%d",
        dataset_name,
        train_scale,
        val_scale,
        scale_ratio,
        integer_like_count,
    )
