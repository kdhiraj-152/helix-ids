"""UNSW learnability contract utilities for processed dataset artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


@dataclass(frozen=True)
class LearnabilityThresholds:
    macro_f1_min: float = 0.40
    min_recall_min: float = 0.20
    unique_pred_coverage_min: float = 0.80
    centroid_min_distance_min: float = 0.01
    random_ratio_min: float = 2.5
    zero_variance_fraction_max: float = 0.10
    near_constant_fraction_max: float = 0.20
    identical_columns_fraction_max: float = 0.05
    centroid_shrinkage_ratio_min: float = 0.30


DEFAULT_THRESHOLDS = LearnabilityThresholds()
META_FILENAME = "meta.json"
FROZEN_SNAPSHOT_FILENAME = "frozen_snapshot_id.txt"
REFERENCE_PROFILE_FILENAME = "reference_profile.json"
REFERENCE_STD_EPSILON = 1e-6
REFERENCE_MIN_SAMPLES = 20
REFERENCE_SKEW_THRESHOLD = 2.5
DIAGNOSIS_DRIFT_INVALIDATION_THRESHOLD = 2.5
REFERENCE_METRIC_KEYS = (
    "centroid_min_distance",
    "unique_pred_coverage",
    "zero_variance_fraction",
    "min_centroid_shrinkage_ratio",
    "label_entropy",
    "signal_to_random_ratio",
)

CAUSE_METRIC_MAP = {
    "feature_space_collapse": "centroid_min_distance",
    "class_prediction_collapse": "unique_pred_coverage",
    "feature_degeneracy": "zero_variance_fraction",
    "scaling_destruction": "min_centroid_shrinkage_ratio",
    "label_distribution_issue": "label_entropy",
    "weak_signal": "signal_to_random_ratio",
}


def _fit_linear_probe(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    clf = LogisticRegression(max_iter=1000)
    clf.fit(x, y)
    return np.asarray(clf.predict(x), dtype=np.int64)


def compute_linear_probe_macro_f1(x: np.ndarray, y: np.ndarray) -> float:
    pred = _fit_linear_probe(np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64))
    return float(f1_score(y, pred, average="macro", zero_division=0))


def _label_entropy(y: np.ndarray) -> tuple[float, float]:
    classes, counts = np.unique(y, return_counts=True)
    probs = counts.astype(np.float64) / max(1, int(counts.sum()))
    entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
    effective_classes = float(np.exp(entropy))
    _ = classes
    return entropy, effective_classes


def _prediction_distribution(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    classes = sorted(int(c) for c in np.unique(y_true).tolist())
    out: dict[str, dict[str, float]] = {}
    for cls in classes:
        mask = y_true == cls
        total = int(np.sum(mask))
        if total == 0:
            out[str(cls)] = {}
            continue
        vals, counts = np.unique(y_pred[mask], return_counts=True)
        out[str(cls)] = {str(int(v)): float(c / total) for v, c in zip(vals, counts)}
    return out


def _feature_degeneracy_metrics(x: np.ndarray) -> dict[str, Any]:
    if x.ndim != 2 or x.shape[1] == 0:
        return {
            "zero_variance_fraction": 1.0,
            "near_constant_fraction": 1.0,
            "identical_columns_fraction": 1.0,
            "zero_variance_features": [],
            "near_constant_features": [],
        }

    variances = np.var(x, axis=0)
    stds = np.std(x, axis=0)
    zero_var_idx = np.flatnonzero(variances <= 1e-12)
    near_constant_idx = np.flatnonzero(stds < 1e-6)

    n_features = int(x.shape[1])
    identical_pairs = 0
    total_pairs = max(1, (n_features * (n_features - 1)) // 2)
    rounded = np.round(x, decimals=12)
    col_hashes = [hash(rounded[:, i].tobytes()) for i in range(n_features)]
    seen: dict[int, int] = {}
    for h in col_hashes:
        seen[h] = seen.get(h, 0) + 1
    for count in seen.values():
        if count > 1:
            identical_pairs += (count * (count - 1)) // 2

    return {
        "zero_variance_fraction": float(len(zero_var_idx) / n_features),
        "near_constant_fraction": float(len(near_constant_idx) / n_features),
        "identical_columns_fraction": float(identical_pairs / total_pairs),
        "zero_variance_features": [int(i) for i in zero_var_idx.tolist()],
        "near_constant_features": [int(i) for i in near_constant_idx.tolist()],
    }


def _mean_centroid_pair_distance(x: np.ndarray, y: np.ndarray) -> float:
    classes = sorted(int(c) for c in np.unique(y).tolist())
    if len(classes) < 2:
        return 0.0
    centroids = {c: np.mean(x[y == c], axis=0) for c in classes}
    dists = [float(np.linalg.norm(centroids[a] - centroids[b])) for a, b in combinations(classes, 2)]
    return float(np.mean(dists)) if dists else 0.0


def compute_stage_diagnostics(
    *,
    stage_snapshots: dict[str, np.ndarray],
    y_train: np.ndarray,
    feature_names: list[str],
    random_seed: int = 42,
) -> dict[str, Any]:
    y = np.asarray(y_train, dtype=np.int64)
    stage_names = list(stage_snapshots.keys())
    diagnostics: dict[str, Any] = {}
    previous_var: np.ndarray | None = None
    previous_mi: np.ndarray | None = None

    for stage_name in stage_names:
        x = np.asarray(stage_snapshots[stage_name], dtype=np.float32)
        pred = _fit_linear_probe(x, y)
        macro_f1 = float(f1_score(y, pred, average="macro", zero_division=0))
        var = np.var(x, axis=0).astype(np.float64)
        mi = mutual_info_classif(x, y, random_state=random_seed).astype(np.float64)

        zero_variance = [feature_names[i] for i in np.flatnonzero(var <= 1e-12).tolist()]
        dropped_features: list[str] = []
        variance_delta = [0.0 for _ in feature_names]
        mutual_info_delta = [0.0 for _ in feature_names]
        if previous_var is not None and previous_mi is not None:
            variance_delta = (var - previous_var).tolist()
            mutual_info_delta = (mi - previous_mi).tolist()
            dropped_idx = np.flatnonzero((previous_var > 1e-12) & (var <= 1e-12))
            dropped_features = [feature_names[i] for i in dropped_idx.tolist()]

        diagnostics[stage_name] = {
            "macro_f1": macro_f1,
            "feature_variance": var.tolist(),
            "mutual_info": mi.tolist(),
            "variance_delta": variance_delta,
            "mutual_info_delta": mutual_info_delta,
            "dropped_features": dropped_features,
            "zero_variance_features": zero_variance,
        }

        previous_var = var
        previous_mi = mi

    transitions: dict[str, Any] = {}
    for prev_name, curr_name in zip(stage_names, stage_names[1:]):
        prev_x = np.asarray(stage_snapshots[prev_name], dtype=np.float32)
        curr_x = np.asarray(stage_snapshots[curr_name], dtype=np.float32)
        prev_f1 = float(diagnostics[prev_name]["macro_f1"])
        curr_f1 = float(diagnostics[curr_name]["macro_f1"])
        f1_ratio = float(curr_f1 / max(1e-12, prev_f1))
        centroid_prev = _mean_centroid_pair_distance(prev_x, y)
        centroid_curr = _mean_centroid_pair_distance(curr_x, y)
        shrink_ratio = float(centroid_curr / max(1e-12, centroid_prev))
        transitions[f"{prev_name}->{curr_name}"] = {
            "f1_ratio": f1_ratio,
            "centroid_shrinkage_ratio": shrink_ratio,
        }

    return {
        "stages": diagnostics,
        "transitions": transitions,
    }


def _centroid_min_distance(x: np.ndarray, y: np.ndarray) -> float:
    classes = sorted(int(c) for c in np.unique(y).tolist())
    if len(classes) < 2:
        return 0.0

    centroids: dict[int, np.ndarray] = {
        c: np.mean(x[y == c], axis=0) for c in classes
    }

    distances = [
        float(np.linalg.norm(centroids[a] - centroids[b]))
        for a, b in combinations(classes, 2)
    ]
    return float(min(distances)) if distances else 0.0


def compute_schema_hash(*, feature_columns: list[str], transformations: list[str]) -> str:
    payload = {
        "feature_columns": feature_columns,
        "transformations": transformations,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def compute_contract_metrics(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    dataset: str,
    schema_hash: str,
    feature_names: list[str] | None = None,
    feature_lineage: dict[str, str] | None = None,
    stage_snapshots: dict[str, np.ndarray] | None = None,
    random_seed: int = 42,
) -> dict[str, Any]:
    y = np.asarray(y_train, dtype=np.int64)
    x = np.asarray(x_train, dtype=np.float32)

    pred = _fit_linear_probe(x, y)
    macro_f1 = float(f1_score(y, pred, average="macro", zero_division=0))

    classes = sorted(int(c) for c in np.unique(y).tolist())
    class_distribution = {str(c): int(np.sum(y == c)) for c in classes}

    per_class_recall: dict[str, float] = {}
    for c in classes:
        class_mask = y == c
        recall = float(np.mean(pred[class_mask] == c)) if np.any(class_mask) else 0.0
        per_class_recall[str(c)] = recall

    unique_preds = np.unique(pred).astype(np.int64)
    unique_pred_coverage = float(len(unique_preds) / max(1, len(classes)))

    centroid_min_distance = _centroid_min_distance(x, y)
    confusion = confusion_matrix(y, pred, labels=classes).astype(np.int64)

    label_entropy, effective_num_classes = _label_entropy(y)
    feature_degeneracy = _feature_degeneracy_metrics(x)

    rng = np.random.default_rng(random_seed)
    x_rand = rng.standard_normal(size=x.shape)
    pred_rand = _fit_linear_probe(x_rand, y)
    random_macro_f1 = float(f1_score(y, pred_rand, average="macro", zero_division=0))

    if feature_names is None:
        feature_names = [f"f_{idx}" for idx in range(int(x.shape[1]))]
    stage_data = stage_snapshots or {"final": x}
    stage_diag_bundle = compute_stage_diagnostics(
        stage_snapshots=stage_data,
        y_train=y,
        feature_names=feature_names,
        random_seed=random_seed,
    )

    return {
        "dataset": dataset,
        "num_classes": int(len(classes)),
        "class_distribution": class_distribution,
        "linear_probe_macro_f1": macro_f1,
        "per_class_recall": per_class_recall,
        "unique_pred_coverage": unique_pred_coverage,
        "unique_preds": [int(v) for v in unique_preds.tolist()],
        "centroid_min_distance": float(centroid_min_distance),
        "random_macro_f1": random_macro_f1,
        "schema_hash": schema_hash,
        "label_entropy": label_entropy,
        "effective_num_classes": effective_num_classes,
        "feature_degeneracy": feature_degeneracy,
        "class_collapse": {
            "confusion_matrix": confusion.tolist(),
            "pred_distribution": _prediction_distribution(y, pred),
        },
        "stage_diagnostics": stage_diag_bundle["stages"],
        "stage_transitions": stage_diag_bundle["transitions"],
        "feature_lineage": feature_lineage or {},
    }


def rank_failure_stages(
    stage_transitions: dict[str, Any],
) -> dict[str, Any]:
    """Rank stages by F1 drop to identify primary failure stage.
    
    Returns:
        dict with 'primary_failure_stage', 'name', 'f1_drop', and 'stages_ranked'
    """
    if not stage_transitions:
        return {"primary_failure_stage": None, "name": None, "f1_drop": 0.0, "stages_ranked": []}
    
    stage_impacts: list[tuple[str, float]] = []
    for transition_name, transition_data in stage_transitions.items():
        f1_ratio = float(transition_data.get("f1_ratio", 1.0))
        f1_drop = 1.0 - f1_ratio
        stage_impacts.append((transition_name, f1_drop))
    
    # Sort by magnitude of drop (descending)
    stage_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    
    if not stage_impacts:
        return {"primary_failure_stage": None, "name": None, "f1_drop": 0.0, "stages_ranked": []}
    
    primary_transition, primary_drop = stage_impacts[0]
    return {
        "primary_failure_stage": primary_transition,
        "name": primary_transition,
        "f1_drop": float(primary_drop),
        "stages_ranked": [{"stage": name, "f1_drop": float(drop)} for name, drop in stage_impacts],
    }


def extract_feature_kill_list(
    stage_diagnostics: dict[str, Any],
    top_n: int = 5,
    epsilon: float = 1e-3,
) -> list[str]:
    """Extract top-N features with highest negative mutual_info_delta.
    
    These features are responsible for collapse.
    """
    feature_impacts: dict[int, float] = {}
    
    for stage_name, stage_data in stage_diagnostics.items():
        mi_delta = stage_data.get("mutual_info_delta", [])
        for feat_idx, delta in enumerate(mi_delta):
            # Keep only meaningful MI loss, not tiny numerical jitter.
            if float(delta) < -abs(float(epsilon)):
                feature_impacts[feat_idx] = feature_impacts.get(feat_idx, 0) + abs(float(delta))
    
    # Sort by absolute MI loss (descending)
    sorted_features = sorted(feature_impacts.items(), key=lambda x: x[1], reverse=True)
    
    capped_top_n = max(0, min(int(top_n), 10))
    # Extract feature names (using f_N format)
    kill_list = [f"f_{feat_idx}" for feat_idx, _ in sorted_features[:capped_top_n]]
    return kill_list


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def _normalize_against_reference(metric_value: float, ref_mean: float, ref_std: float) -> float:
    z = (float(metric_value) - float(ref_mean)) / (float(ref_std) + 1e-8)
    return _sigmoid(z)


def _get_reference_stats(reference_profile: dict[str, Any], metric_name: str) -> tuple[float, float]:
    metric_profile = reference_profile.get(metric_name)
    if not isinstance(metric_profile, dict):
        return 0.0, 1.0
    mean = float(metric_profile.get("mean", 0.0))
    std = float(metric_profile.get("std", 1.0))
    return mean, max(abs(std), REFERENCE_STD_EPSILON)


def _require_reference_profile(meta: dict[str, Any]) -> dict[str, Any]:
    reference_profile = meta.get("reference_profile")
    if not isinstance(reference_profile, dict) or not reference_profile:
        raise RuntimeError(
            "Reference profile is required for calibrated diagnosis. "
            "Provide meta['reference_profile'] derived from successful historical runs."
        )
    return reference_profile


def _compute_observational_scores(meta: dict[str, Any]) -> dict[str, float]:
    current = _extract_current_reference_metrics(meta)
    num_classes = max(2, int(meta.get("num_classes", 2)))
    expected_entropy = float(np.log(num_classes))

    # Uncalibrated fallback: use contract thresholds and observable metrics only.
    return {
        "feature_space_collapse": _sigmoid(
            (DEFAULT_THRESHOLDS.centroid_min_distance_min - current["centroid_min_distance"])
            / max(DEFAULT_THRESHOLDS.centroid_min_distance_min, 1e-6)
        ),
        "class_prediction_collapse": _sigmoid(
            (DEFAULT_THRESHOLDS.unique_pred_coverage_min - current["unique_pred_coverage"]) / 0.1
        ),
        "feature_degeneracy": _sigmoid(
            (current["zero_variance_fraction"] - DEFAULT_THRESHOLDS.zero_variance_fraction_max) / 0.05
        ),
        "scaling_destruction": _sigmoid(
            (DEFAULT_THRESHOLDS.centroid_shrinkage_ratio_min - current["min_centroid_shrinkage_ratio"]) / 0.1
        ),
        "label_distribution_issue": _sigmoid(
            ((0.6 * expected_entropy) - current["label_entropy"]) / max(0.2 * expected_entropy, 1e-6)
        ),
        "weak_signal": _sigmoid(
            (DEFAULT_THRESHOLDS.random_ratio_min - current["signal_to_random_ratio"]) / 0.5
        ),
    }


def _extract_current_reference_metrics(meta: dict[str, Any]) -> dict[str, float]:
    transitions = meta.get("stage_transitions", {})
    min_shrink_ratio = 1.0
    for transition_data in transitions.values():
        min_shrink_ratio = min(min_shrink_ratio, float(transition_data.get("centroid_shrinkage_ratio", 1.0)))
    random_f1 = max(1e-8, float(meta.get("random_macro_f1", 0.0)))
    signal_ratio = float(meta.get("linear_probe_macro_f1", 0.0)) / random_f1
    degeneracy = meta.get("feature_degeneracy", {})
    return {
        "centroid_min_distance": float(meta.get("centroid_min_distance", 0.0)),
        "unique_pred_coverage": float(meta.get("unique_pred_coverage", 0.0)),
        "zero_variance_fraction": float(degeneracy.get("zero_variance_fraction", 0.0)),
        "min_centroid_shrinkage_ratio": float(min_shrink_ratio),
        "label_entropy": float(meta.get("label_entropy", 0.0)),
        "signal_to_random_ratio": float(signal_ratio),
    }


def _validate_reference_profile(reference_profile: dict[str, Any]) -> tuple[list[str], set[str]]:
    flags: list[str] = []
    valid_metrics: set[str] = set()
    for metric_name in REFERENCE_METRIC_KEYS:
        metric_profile = reference_profile.get(metric_name)
        if not isinstance(metric_profile, dict):
            flags.append(f"invalid_reference_profile:missing_metric:{metric_name}")
            continue

        std = float(metric_profile.get("std", 0.0))
        sample_count = int(metric_profile.get("sample_count", 0))
        skew = float(metric_profile.get("distribution_skew", 0.0))

        if std < REFERENCE_STD_EPSILON:
            flags.append(f"invalid_reference_profile:zero_variance_metric:{metric_name}")
            continue
        if sample_count < REFERENCE_MIN_SAMPLES:
            flags.append(f"invalid_reference_profile:insufficient_samples:{metric_name}")
            continue
        if abs(skew) > REFERENCE_SKEW_THRESHOLD:
            flags.append("non_stationary_reference")

        valid_metrics.add(metric_name)

    return sorted(set(flags)), valid_metrics


def _compute_metric_drifts(
    meta: dict[str, Any],
    reference_profile: dict[str, Any],
    available_metrics: set[str] | None = None,
) -> tuple[dict[str, float], float]:
    current = _extract_current_reference_metrics(meta)
    drifts: dict[str, float] = {}
    for metric_name, value in current.items():
        if available_metrics is not None and metric_name not in available_metrics:
            continue
        ref_mean, ref_std = _get_reference_stats(reference_profile, metric_name)
        drifts[metric_name] = float(abs(value - ref_mean) / (ref_std + 1e-8))
    drift_score = float(np.mean(list(drifts.values()))) if drifts else 0.0
    return drifts, drift_score


def _collect_correlation_data(metric_history: list[Any]) -> tuple[list[str], np.ndarray | None]:
    vectors: dict[str, list[float]] = {k: [] for k in REFERENCE_METRIC_KEYS}
    for row in metric_history:
        if not isinstance(row, dict):
            continue
        for key in REFERENCE_METRIC_KEYS:
            if key in row:
                vectors[key].append(float(row[key]))

    keys = [k for k in REFERENCE_METRIC_KEYS if len(vectors[k]) >= 5]
    if len(keys) < 2:
        return keys, None

    data = np.asarray([vectors[k] for k in keys], dtype=np.float64)
    corr = np.corrcoef(data)
    return keys, np.atleast_2d(np.asarray(corr, dtype=np.float64))


def _adjust_scores_for_redundancy(
    scores: dict[str, float],
    keys: list[str],
    corr: np.ndarray,
) -> tuple[dict[str, float], list[str]]:
    adjusted = dict(scores)
    redundancy_flags: list[str] = []

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if abs(float(corr[i, j])) <= 0.9:
                continue
            redundant_metric = keys[j]
            for cause, metric in CAUSE_METRIC_MAP.items():
                if metric == redundant_metric and cause in adjusted:
                    adjusted[cause] = float(adjusted[cause] * 0.8)
                    redundancy_flags.append(f"redundant_metric:{redundant_metric}")

    return adjusted, sorted(set(redundancy_flags))


def _apply_metric_correlation_weighting(scores: dict[str, float], meta: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    metric_history = meta.get("metric_history", [])
    if not isinstance(metric_history, list) or len(metric_history) < 5:
        return dict(scores), []

    keys, corr = _collect_correlation_data(metric_history)
    if corr is None:
        return dict(scores), []

    return _adjust_scores_for_redundancy(scores, keys, corr)


def _causal_sanity_factor(scores: dict[str, float], primary_cause: str) -> tuple[float, bool]:
    if primary_cause not in scores:
        return 1.0, False
    simulated = dict(scores)
    simulated[primary_cause] = 0.0
    before = float(max(scores.values())) if scores else 0.0
    after = float(max(simulated.values())) if simulated else 0.0
    recovery = before - after
    if recovery < 0.1:
        return 0.7, True
    return 1.0, False


def _apply_mode_override_hierarchy(
    *,
    metric_inconsistency: bool,
    distribution_shift: bool,
    weak_signal_regime: bool,
    composite_candidate: bool,
    default_mode: str,
) -> str:
    if metric_inconsistency:
        return "inconsistent"
    if distribution_shift:
        return "distribution_shift"
    if weak_signal_regime:
        return "weak_signal"
    if composite_candidate:
        return "composite"
    return default_mode


def _severity_feature_space(meta: dict[str, Any], reference_profile: dict[str, Any]) -> float:
    d = float(meta.get("centroid_min_distance", 0.0))
    ref_mean, ref_std = _get_reference_stats(reference_profile, "centroid_min_distance")
    return _normalize_against_reference(ref_mean - d, 0.0, ref_std)


def _severity_class_collapse(meta: dict[str, Any], reference_profile: dict[str, Any]) -> float:
    c = float(meta.get("unique_pred_coverage", 0.0))
    ref_mean, ref_std = _get_reference_stats(reference_profile, "unique_pred_coverage")
    return _normalize_against_reference(ref_mean - c, 0.0, ref_std)


def _severity_degeneracy(meta: dict[str, Any], reference_profile: dict[str, Any]) -> float:
    degeneracy = meta.get("feature_degeneracy", {})
    z = float(degeneracy.get("zero_variance_fraction", 0.0))
    ref_mean, ref_std = _get_reference_stats(reference_profile, "zero_variance_fraction")
    return _normalize_against_reference(z - ref_mean, 0.0, ref_std)


def _severity_scaling(meta: dict[str, Any], reference_profile: dict[str, Any]) -> float:
    transitions = meta.get("stage_transitions", {})
    min_shrink_ratio = 1.0
    for transition_data in transitions.values():
        shrink_ratio = float(transition_data.get("centroid_shrinkage_ratio", 1.0))
        min_shrink_ratio = min(min_shrink_ratio, shrink_ratio)
    ref_mean, ref_std = _get_reference_stats(reference_profile, "min_centroid_shrinkage_ratio")
    return _normalize_against_reference(ref_mean - min_shrink_ratio, 0.0, ref_std)


def _severity_label_distribution(meta: dict[str, Any], reference_profile: dict[str, Any]) -> float:
    label_entropy = float(meta.get("label_entropy", 0.0))
    ref_mean, ref_std = _get_reference_stats(reference_profile, "label_entropy")
    return _normalize_against_reference(ref_mean - label_entropy, 0.0, ref_std)


def _severity_weak_signal(meta: dict[str, Any], reference_profile: dict[str, Any]) -> float:
    macro_f1 = float(meta.get("linear_probe_macro_f1", 0.0))
    random_f1 = max(1e-8, float(meta.get("random_macro_f1", 0.0)))
    signal_ratio = macro_f1 / random_f1
    ref_mean, ref_std = _get_reference_stats(reference_profile, "signal_to_random_ratio")
    return _normalize_against_reference(ref_mean - signal_ratio, 0.0, ref_std)


def _compute_temporal_instability(meta: dict[str, Any], primary_cause: str, current_score: float) -> tuple[float, bool]:
    history = meta.get("diagnosis_history", [])
    if not isinstance(history, list):
        return 1.0, False

    score_samples: list[float] = [float(current_score)]
    for item in history:
        if not isinstance(item, dict):
            continue
        scores = item.get("scores", {})
        if not isinstance(scores, dict) or primary_cause not in scores:
            continue
        score_samples.append(float(scores[primary_cause]))

    if len(score_samples) < 3:
        return 1.0, False

    variance = float(np.var(np.asarray(score_samples, dtype=np.float64)))
    if variance > 0.03:
        return 0.7, True
    return 1.0, False


def _annotate_systemic_features(meta: dict[str, Any], kill_list: list[str]) -> list[str]:
    past_failure_kill_lists = meta.get("past_failure_kill_lists", [])
    if not isinstance(past_failure_kill_lists, list) or not past_failure_kill_lists:
        return []

    total = 0
    counts: dict[str, int] = {}
    for entry in past_failure_kill_lists:
        if not isinstance(entry, list):
            continue
        total += 1
        for feature in entry:
            if not isinstance(feature, str):
                continue
            counts[feature] = counts.get(feature, 0) + 1

    if total == 0:
        return []

    return [
        feature for feature in kill_list
        if (counts.get(feature, 0) / float(total)) > 0.60
    ]


def _infer_trace(stage_diagnostics: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    trace: list[dict[str, Any]] = []
    f1_values: list[float] = []
    for stage_name, stage_data in stage_diagnostics.items():
        f1 = float(stage_data.get("macro_f1", 0.0))
        trace.append({"stage": stage_name, "f1": round(f1, 6)})
        f1_values.append(f1)

    if len(f1_values) < 2:
        return trace, "gradual"

    drops = [max(0.0, f1_values[i] - f1_values[i + 1]) for i in range(len(f1_values) - 1)]
    total_drop = float(sum(drops))
    max_drop = float(max(drops)) if drops else 0.0
    if total_drop <= 1e-9:
        return trace, "gradual"

    collapse_gradient = "sudden" if (max_drop >= 0.12 and (max_drop / total_drop) >= 0.6) else "gradual"
    return trace, collapse_gradient


def _cause_stage_mismatch(cause: str, stage: str | None) -> bool:
    if not stage:
        return False
    stage_l = stage.lower()
    cause_expected_tokens: dict[str, tuple[str, ...]] = {
        "scaling_destruction": ("scale", "scaler", "normaliz"),
        "label_distribution_issue": ("raw", "split", "sampling", "label"),
    }
    expected_tokens = cause_expected_tokens.get(cause)
    if expected_tokens is None:
        return False
    return not any(tok in stage_l for tok in expected_tokens)


def _has_metric_inconsistency(stage_transitions: dict[str, Any]) -> bool:
    for transition in stage_transitions.values():
        centroid_ratio = float(transition.get("centroid_shrinkage_ratio", 1.0))
        f1_ratio = float(transition.get("f1_ratio", 1.0))
        # Metric gaming signal: separation appears better while predictive value does not.
        if centroid_ratio > 1.0 and f1_ratio <= 1.0:
            return True
    return False


def _resolve_reference_profile(meta: dict[str, Any]) -> tuple[bool, dict[str, Any], list[str], float]:
    flags: list[str] = []
    confidence_multiplier = 1.0
    reference_profile_obj = meta.get("reference_profile")
    reference_profile: dict[str, Any] = (
        dict(reference_profile_obj)
        if isinstance(reference_profile_obj, dict) and bool(reference_profile_obj)
        else {}
    )
    has_reference_profile = bool(reference_profile)

    if not has_reference_profile:
        flags.append("missing_reference_profile")
        confidence_multiplier *= 0.3

    return has_reference_profile, reference_profile, flags, confidence_multiplier


def _compute_reference_coverage(
    meta: dict[str, Any],
    has_reference_profile: bool,
    reference_profile: dict[str, Any],
) -> tuple[float, set[str]]:
    current_reference_metrics = _extract_current_reference_metrics(meta)
    profile_metrics = set(reference_profile.keys()) if has_reference_profile else set()
    available_metrics = set(current_reference_metrics.keys()).intersection(profile_metrics)
    coverage = len(available_metrics) / float(len(REFERENCE_METRIC_KEYS))
    return coverage, available_metrics


def _collect_reference_validity(
    has_reference_profile: bool,
    reference_profile: dict[str, Any],
    available_metrics: set[str],
) -> tuple[list[str], set[str], set[str]]:
    if not has_reference_profile:
        return [], set(), set()

    reference_validity_flags, valid_reference_metrics = _validate_reference_profile(reference_profile)
    available_valid_metrics = available_metrics.intersection(valid_reference_metrics)
    return reference_validity_flags, valid_reference_metrics, available_valid_metrics


def _build_raw_scores(
    *,
    meta: dict[str, Any],
    reference_profile: dict[str, Any],
    has_reference_profile: bool,
    coverage: float,
    available_valid_metrics: set[str],
) -> dict[str, float]:
    use_calibrated_scores = has_reference_profile and coverage >= 0.4 and len(available_valid_metrics) >= 4
    if not use_calibrated_scores:
        return _compute_observational_scores(meta)

    observational_scores = _compute_observational_scores(meta)
    return {
        "feature_space_collapse": (
            _severity_feature_space(meta, reference_profile)
            if "centroid_min_distance" in available_valid_metrics
            else observational_scores["feature_space_collapse"]
        ),
        "class_prediction_collapse": (
            _severity_class_collapse(meta, reference_profile)
            if "unique_pred_coverage" in available_valid_metrics
            else observational_scores["class_prediction_collapse"]
        ),
        "feature_degeneracy": (
            _severity_degeneracy(meta, reference_profile)
            if "zero_variance_fraction" in available_valid_metrics
            else observational_scores["feature_degeneracy"]
        ),
        "scaling_destruction": (
            _severity_scaling(meta, reference_profile)
            if "min_centroid_shrinkage_ratio" in available_valid_metrics
            else observational_scores["scaling_destruction"]
        ),
        "label_distribution_issue": (
            _severity_label_distribution(meta, reference_profile)
            if "label_entropy" in available_valid_metrics
            else observational_scores["label_distribution_issue"]
        ),
        "weak_signal": (
            _severity_weak_signal(meta, reference_profile)
            if "signal_to_random_ratio" in available_valid_metrics
            else observational_scores["weak_signal"]
        ),
    }


def _derive_base_mode(
    *,
    coverage: float,
    has_reference_profile: bool,
    weak_signal_regime: bool,
    composite_candidate: bool,
) -> tuple[str, list[str]]:
    mode = "single"
    flags: list[str] = []

    if weak_signal_regime:
        mode = "weak_signal"
    elif composite_candidate:
        mode = "composite"

    if coverage < 0.6 and mode == "single":
        mode = "weak_signal"
        flags.append("low_profile_coverage")

    if coverage < 0.4 or not has_reference_profile:
        mode = "uncalibrated"
        flags.append("uncalibrated_mode")

    return mode, flags


def _resolve_output_for_mode(
    *,
    mode: str,
    primary: str,
    secondary_cause: str,
    composite_scores: dict[str, float],
    metric_drifts: dict[str, float],
    confidence: float,
) -> tuple[str, list[str], float]:
    if mode == "inconsistent":
        secondary = list(composite_scores.keys()) or [primary, secondary_cause]
        return "metric_inconsistency", secondary, max(0.80, confidence)

    if mode == "distribution_shift":
        return "distribution_shift", list(metric_drifts.keys()), 0.0

    if mode == "weak_signal":
        secondary = [cause for cause in composite_scores if cause != "weak_signal"]
        return "weak_signal", secondary, _clamp01(confidence * 0.5)

    if mode == "uncalibrated":
        secondary = [cause for cause in composite_scores if cause != primary] or [secondary_cause]
        return primary, secondary, confidence

    if mode == "composite":
        secondary = list(composite_scores.keys()) or [primary, secondary_cause]
        return "composite_failure", secondary, confidence

    return primary, [secondary_cause], confidence


def derive_root_cause(meta: dict[str, Any]) -> dict[str, Any]:
    """Continuous diagnostic inference with conflict resolution and ambiguity support."""
    has_reference_profile, reference_profile, flags, confidence_multiplier = _resolve_reference_profile(meta)

    stage_diagnostics = meta.get("stage_diagnostics", {})
    stage_transitions = meta.get("stage_transitions", {})

    coverage, available_metrics = _compute_reference_coverage(meta, has_reference_profile, reference_profile)
    reference_validity_flags, _, available_valid_metrics = _collect_reference_validity(
        has_reference_profile,
        reference_profile,
        available_metrics,
    )
    flags.extend(reference_validity_flags)

    has_invalid_reference = any(flag.startswith("invalid_reference_profile") for flag in flags)
    if has_invalid_reference:
        confidence_multiplier *= 0.5

    metric_drifts, drift_score = _compute_metric_drifts(meta, reference_profile, available_valid_metrics)

    failure_stage_info = rank_failure_stages(stage_transitions)
    primary_failure_stage = failure_stage_info.get("primary_failure_stage")

    raw_scores = _build_raw_scores(
        meta=meta,
        reference_profile=reference_profile,
        has_reference_profile=has_reference_profile,
        coverage=coverage,
        available_valid_metrics=available_valid_metrics,
    )

    scores, correlation_flags = _apply_metric_correlation_weighting(raw_scores, meta)
    flags.extend(correlation_flags)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary, p_score = sorted_scores[0]
    secondary_cause, s_score = sorted_scores[1]
    global_intensity = float(max(scores.values())) if scores else 0.0

    trace, collapse_gradient = _infer_trace(stage_diagnostics)
    kill_list = extract_feature_kill_list(stage_diagnostics, top_n=10, epsilon=1e-3)
    systemic_features = _annotate_systemic_features(meta, kill_list)

    confidence = _clamp01((p_score - s_score) * p_score)
    cause_stage_mismatch = _cause_stage_mismatch(primary, primary_failure_stage)
    if cause_stage_mismatch:
        confidence = _clamp01(confidence * 0.5)

    temporal_factor, temporal_instability = _compute_temporal_instability(meta, primary, p_score)
    confidence = _clamp01(confidence * temporal_factor)

    composite_scores = {
        cause: float(score)
        for cause, score in sorted_scores
        if score > (0.5 * p_score)
    }

    metric_inconsistency = _has_metric_inconsistency(stage_transitions)
    weak_signal_regime = global_intensity < 0.3
    composite_candidate = confidence < 0.15

    mode, mode_flags = _derive_base_mode(
        coverage=coverage,
        has_reference_profile=has_reference_profile,
        weak_signal_regime=weak_signal_regime,
        composite_candidate=composite_candidate,
    )
    flags.extend(mode_flags)

    distribution_shift = drift_score > DIAGNOSIS_DRIFT_INVALIDATION_THRESHOLD and p_score < 0.6
    if drift_score > DIAGNOSIS_DRIFT_INVALIDATION_THRESHOLD and not distribution_shift:
        flags.append("high_drift_detected")

    mode = _apply_mode_override_hierarchy(
        metric_inconsistency=metric_inconsistency,
        distribution_shift=distribution_shift,
        weak_signal_regime=False,
        composite_candidate=False,
        default_mode=mode,
    )

    primary_output, secondary, confidence = _resolve_output_for_mode(
        mode=mode,
        primary=primary,
        secondary_cause=secondary_cause,
        composite_scores=composite_scores,
        metric_drifts=metric_drifts,
        confidence=confidence,
    )

    expected_profile_version = meta.get("expected_reference_profile_version")
    profile_version = reference_profile.get("version")
    profile_version_mismatch = bool(expected_profile_version) and str(profile_version) != str(expected_profile_version)
    if profile_version_mismatch:
        flags.append("reference_profile_version_mismatch")
        confidence = _clamp01(confidence * 0.8)

    causal_factor, failed_causal_sanity = _causal_sanity_factor(scores, primary if mode == "single" else primary_output)
    confidence = _clamp01(confidence * causal_factor)
    confidence = _clamp01(confidence * confidence_multiplier)
    confidence = max(float(confidence), 0.05)

    if (not has_reference_profile) or coverage < 0.4:
        regime = "uncalibrated"
    elif has_invalid_reference or coverage < 1.0:
        regime = "degraded"
    else:
        regime = "calibrated"

    return {
        "primary": primary_output,
        "secondary": secondary,
        "scores": {k: float(v) for k, v in scores.items()},
        "confidence": float(confidence),
        "mode": mode,
        "regime": regime,
        "drift_score": float(drift_score),
        "metric_drifts": metric_drifts,
        "coverage": float(coverage),
        "available_metrics": sorted(available_valid_metrics),
        "reference_validity_flags": reference_validity_flags,
        "correlation_flags": correlation_flags,
        "flags": sorted(set(flags)),
        "profile_version": profile_version,
        "profile_version_mismatch": bool(profile_version_mismatch),
        "stage": primary_failure_stage,
        "offending_stage": primary_failure_stage,
        "kill_list": kill_list,
        "systemic_features": systemic_features,
        "global_intensity": float(global_intensity),
        "failure_regime": "hard_failure" if global_intensity > 0.8 else "normal",
        "trace": trace,
        "collapse_gradient": collapse_gradient,
        "cause_stage_mismatch": bool(cause_stage_mismatch),
        "metric_inconsistency": bool(metric_inconsistency),
        "distribution_shift": bool(distribution_shift),
        "temporal_instability": bool(temporal_instability),
        "failed_causal_sanity": bool(failed_causal_sanity),
        "composite": composite_scores,
    }
def get_action_directive(diagnosis: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convert diagnosis to a mechanism-tied action directive.
    
    Maps root cause → action type + target stage/features
    """
    primary = diagnosis.get("primary", "unknown")
    mode = diagnosis.get("mode", "single")
    stage = diagnosis.get("stage", None)
    kill_list = diagnosis.get("kill_list", [])
    confidence = float(diagnosis.get("confidence", 0.5))

    if mode == "uncalibrated" or confidence < 0.2:
        return {
            "type": "NO_OP",
            "target_stage": stage,
            "expected_effect": "suppress intervention under insufficient diagnostic confidence",
            "confidence": confidence,
            "target_features": kill_list,
            "rationale": "insufficient_diagnostic_confidence",
            "reason": "insufficient_diagnostic_confidence",
            "feasibility_warning": None,
        }
    
    action_mapping = {
        "scaling_destruction": "REMOVE_SCALING",
        "feature_degeneracy": "DROP_FEATURES",
        "class_prediction_collapse": "REBUILD_LABELS",
        "feature_space_collapse": "FIX_ENCODING",
        "label_distribution_issue": "REBALANCE_CLASSES",
        "weak_signal": "FEATURE_ENGINEERING",
        "composite_failure": "MULTI_STAGE_REPAIR",
        "metric_inconsistency": "VALIDATE_METRICS",
        "distribution_shift": "REFRESH_BASELINE",
        "unknown": "INVESTIGATE",
    }

    expected_effect_mapping = {
        "REMOVE_SCALING": "increase centroid distance",
        "DROP_FEATURES": "reduce zero-variance feature load",
        "REBUILD_LABELS": "restore prediction coverage across classes",
        "FIX_ENCODING": "increase class centroid separation",
        "REBALANCE_CLASSES": "increase label entropy and minority recall",
        "FEATURE_ENGINEERING": "increase separability signal above random floor",
        "MULTI_STAGE_REPAIR": "address overlapping failures across stages",
        "VALIDATE_METRICS": "eliminate metric inconsistency between separation and F1",
        "REFRESH_BASELINE": "re-align scoring baseline to current data regime",
        "INVESTIGATE": "collect additional diagnostics",
    }
    
    action_type = action_mapping.get(primary, "INVESTIGATE")
    feasibility_warning = None
    past_successful_config = (context or {}).get("past_successful_config", {})
    forbidden_actions = set(past_successful_config.get("forbidden_actions", [])) if isinstance(past_successful_config, dict) else set()
    if action_type in forbidden_actions:
        confidence = _clamp01(confidence * 0.7)
        feasibility_warning = "action_contradicts_past_successful_config"
    
    return {
        "type": action_type,
        "target_stage": stage,
        "expected_effect": expected_effect_mapping.get(action_type, "collect additional diagnostics"),
        "confidence": confidence,
        "target_features": kill_list,
        "rationale": f"Diagnosis primary: {primary}",
        "feasibility_warning": feasibility_warning,
    }


def create_summary(meta: dict[str, Any]) -> dict[str, Any]:
    """Create compressed summary for CI output and error messages.
    
    Extracts:
    - status (PASS/FAIL)
    - primary_issue (root cause)
    - stage (failure stage)
    - action (what to fix)
    - confidence (0-1)
    """
    validated = bool(meta.get("validated", False))
    status = "PASS" if validated else "FAIL"
    
    diagnosis_obj = meta.get("diagnosis")
    diagnosis: dict[str, Any] = diagnosis_obj if isinstance(diagnosis_obj, dict) else derive_root_cause(meta)

    action_obj = meta.get("action")
    action: dict[str, Any] = (
        action_obj
        if isinstance(action_obj, dict)
        else get_action_directive(diagnosis, context=meta)
    )
    confidence = float(min(diagnosis.get("confidence", 0.5), action.get("confidence", diagnosis.get("confidence", 0.5))))
    
    return {
        "status": status,
        "primary_issue": diagnosis.get("primary", "unknown"),
        "stage": diagnosis.get("stage", None),
        "action": action.get("type", "INVESTIGATE"),
        "confidence": confidence,
        "kill_list": diagnosis.get("kill_list", []),
        "mode": diagnosis.get("mode", "single"),
        "global_intensity": float(diagnosis.get("global_intensity", 0.0)),
    }


def format_failure_message(summary: dict[str, Any]) -> str:
    """Format deterministic failure message for training errors.
    
    Replaces vague RuntimeError with actionable diagnosis.
    """
    return (
        f"UNSW CONTRACT FAILURE\n"
        f"Primary: {summary['primary_issue']}\n"
        f"Stage: {summary['stage'] or 'unknown'}\n"
        f"Action: {summary['action']}\n"
        f"Confidence: {summary['confidence']:.2f}"
    )


def evaluate_contract(
    metrics: dict[str, Any],
    *,
    thresholds: LearnabilityThresholds = DEFAULT_THRESHOLDS,
) -> tuple[bool, dict[str, list[str]]]:
    violations: dict[str, list[str]] = {
        "BLOCKER": [],
        "DEGRADATION": [],
        "WARNING": [],
    }

    if float(metrics["linear_probe_macro_f1"]) < thresholds.macro_f1_min:
        violations["BLOCKER"].append("linear_probe_macro_f1_below_min")

    min_recall = min(float(v) for v in metrics["per_class_recall"].values())
    if min_recall < thresholds.min_recall_min:
        violations["BLOCKER"].append("per_class_recall_below_min")

    if float(metrics["unique_pred_coverage"]) < thresholds.unique_pred_coverage_min:
        violations["BLOCKER"].append("unique_pred_coverage_below_min")

    if float(metrics["centroid_min_distance"]) <= thresholds.centroid_min_distance_min:
        violations["BLOCKER"].append("centroid_min_distance_below_min")

    random_macro = float(metrics["random_macro_f1"])
    real_macro = float(metrics["linear_probe_macro_f1"])
    if real_macro < (thresholds.random_ratio_min * random_macro):
        violations["BLOCKER"].append("real_macro_not_above_random_floor")

    degeneracy = metrics.get("feature_degeneracy", {})
    if float(degeneracy.get("zero_variance_fraction", 0.0)) > thresholds.zero_variance_fraction_max:
        violations["BLOCKER"].append("zero_variance_fraction_above_max")
    if float(degeneracy.get("near_constant_fraction", 0.0)) > thresholds.near_constant_fraction_max:
        violations["BLOCKER"].append("near_constant_fraction_above_max")
    if float(degeneracy.get("identical_columns_fraction", 0.0)) > thresholds.identical_columns_fraction_max:
        violations["BLOCKER"].append("identical_columns_fraction_above_max")

    transitions = metrics.get("stage_transitions", {})
    for transition_name, transition in transitions.items():
        if float(transition.get("f1_ratio", 1.0)) < 0.90:
            violations["BLOCKER"].append(f"stage_f1_drop_gt_10pct:{transition_name}")
        if float(transition.get("centroid_shrinkage_ratio", 1.0)) < thresholds.centroid_shrinkage_ratio_min:
            violations["BLOCKER"].append(f"centroid_shrinkage_ratio_below_min:{transition_name}")

    if float(metrics.get("effective_num_classes", 0.0)) < 0.60 * max(1.0, float(metrics.get("num_classes", 1.0))):
        violations["DEGRADATION"].append("low_effective_class_count")
    if float(metrics.get("label_entropy", 0.0)) < 0.60 * np.log(max(2, int(metrics.get("num_classes", 2)))):
        violations["WARNING"].append("low_label_entropy")

    return (len(violations["BLOCKER"]) == 0), violations


def build_meta(
    metrics: dict[str, Any],
    *,
    thresholds: LearnabilityThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    if "reference_profile" not in metrics:
        raise RuntimeError(
            "Missing reference_profile in metrics. Diagnosis requires calibrated baseline from successful runs."
        )

    validated, violations = evaluate_contract(metrics, thresholds=thresholds)
    min_recall = min(float(v) for v in metrics["per_class_recall"].values())
    
    # Build meta with all diagnostics
    meta = {
        **metrics,
        "validated": bool(validated),
        "violations": violations,
        "min_per_class_recall": min_recall,
        "thresholds": asdict(thresholds),
        "snapshot_id": None,
        "frozen": False,
    }
    
    # Add deterministic diagnosis analysis
    meta["diagnosis"] = derive_root_cause(meta)
    meta["action"] = get_action_directive(meta["diagnosis"], context=meta)
    meta["summary"] = create_summary(meta)

    should_update_baseline = bool(
        meta.get("validated", False)
        and float(meta.get("summary", {}).get("confidence", 0.0)) > 0.8
    )
    if should_update_baseline:
        meta["reference_profile"] = update_reference_profile(meta["reference_profile"], meta)
        meta["reference_profile_updated"] = True
    else:
        meta["reference_profile_updated"] = False
    
    return meta


def write_meta(meta: dict[str, Any], *, artifact_dir: Path) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / META_FILENAME
    out_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def load_reference_profile(*, artifact_dir: Path) -> dict[str, Any]:
    bundle = load_reference_profile_bundle(artifact_dir=artifact_dir, dataset_signature="default")
    return dict(bundle["profile"])


def load_reference_profile_bundle(
    *,
    artifact_dir: Path,
    dataset_signature: str,
) -> dict[str, Any]:
    reference_path = artifact_dir / REFERENCE_PROFILE_FILENAME
    if not reference_path.exists():
        raise RuntimeError(
            "Missing reference profile. Calibrated diagnosis is blocked until "
            f"{REFERENCE_PROFILE_FILENAME} exists in {artifact_dir}."
        )
    payload = dict(json.loads(reference_path.read_text(encoding="utf-8")))
    if not payload:
        raise RuntimeError("Reference profile is empty; diagnosis is blocked")

    profiles = payload.get("reference_profiles")
    if not isinstance(profiles, dict):
        # Backward-compatible single profile payload.
        return {
            "profile_key": str(dataset_signature),
            "profile": payload,
            "payload": {"reference_profiles": {str(dataset_signature): payload}},
        }

    if dataset_signature in profiles:
        profile_key = str(dataset_signature)
    else:
        signature_l = str(dataset_signature).lower()
        profile_key = ""
        for candidate in profiles:
            if candidate.lower() in signature_l or signature_l in candidate.lower():
                profile_key = str(candidate)
                break
        if not profile_key:
            raise RuntimeError(f"invalid_reference_profile: no_matching_profile_for:{dataset_signature}")

    selected = profiles.get(profile_key)
    if not isinstance(selected, dict):
        raise RuntimeError(f"invalid_reference_profile: malformed_profile:{profile_key}")

    return {
        "profile_key": profile_key,
        "profile": selected,
        "payload": payload,
    }


def update_reference_profile(
    reference_profile: dict[str, Any],
    meta: dict[str, Any],
    *,
    rolling_window: int = 200,
    max_influence: float = 0.05,
) -> dict[str, Any]:
    updated = dict(reference_profile)
    current_metrics = _extract_current_reference_metrics(meta)

    for metric_name in REFERENCE_METRIC_KEYS:
        profile = dict(updated.get(metric_name, {}))
        if "mean" not in profile or "std" not in profile:
            continue
        mean = float(profile.get("mean", 0.0))
        std = max(REFERENCE_STD_EPSILON, float(profile.get("std", 0.0)))
        sample_count = max(1, int(profile.get("sample_count", REFERENCE_MIN_SAMPLES)))
        current_value = float(current_metrics.get(metric_name, mean))

        z = abs(current_value - mean) / (std + 1e-8)
        if z > 3.0:
            # Outlier rejection: do not absorb this point.
            updated[metric_name] = profile
            continue

        influence = min(float(max_influence), 1.0 / float(sample_count))
        new_mean = (1.0 - influence) * mean + influence * current_value
        variance = std**2
        new_variance = (1.0 - influence) * variance + influence * (current_value - new_mean) ** 2
        new_std = float(max(REFERENCE_STD_EPSILON, np.sqrt(max(0.0, new_variance))))

        profile["mean"] = float(new_mean)
        profile["std"] = new_std
        profile["sample_count"] = int(min(rolling_window, sample_count + 1))
        updated[metric_name] = profile

    updated["source_runs"] = int(updated.get("source_runs", 0)) + 1
    return updated


def write_reference_profile(reference_profile: dict[str, Any], *, artifact_dir: Path) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / REFERENCE_PROFILE_FILENAME
    out_path.write_text(json.dumps(reference_profile, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def compute_snapshot_id(*, artifact_dir: Path, dataset: str) -> str:
    sha = hashlib.sha256()
    file_names = [
        f"X_train_{dataset}.npy",
        f"y_train_{dataset}.npy",
        "feature_columns.npy",
        META_FILENAME,
    ]
    for name in file_names:
        path = artifact_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact for snapshot hashing: {path}")
        sha.update(name.encode("utf-8"))
        with path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                sha.update(chunk)
    return sha.hexdigest()


def freeze_snapshot_if_valid(*, artifact_dir: Path) -> dict[str, Any]:
    meta = load_meta(artifact_dir=artifact_dir)
    dataset = str(meta.get("dataset", "unsw_nb15"))
    snapshot_id = compute_snapshot_id(artifact_dir=artifact_dir, dataset=dataset)
    meta["snapshot_id"] = snapshot_id
    if bool(meta.get("validated", False)):
        (artifact_dir / FROZEN_SNAPSHOT_FILENAME).write_text(snapshot_id + "\n", encoding="utf-8")
        meta["frozen"] = True
    else:
        meta["frozen"] = False
    write_meta(meta, artifact_dir=artifact_dir)
    return meta


def load_meta(*, artifact_dir: Path) -> dict[str, Any]:
    meta_path = artifact_dir / META_FILENAME
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing learnability contract file: {meta_path}")
    meta = dict(json.loads(meta_path.read_text(encoding="utf-8")))

    # Backward compatibility for already-produced artifacts.
    if "diagnosis" not in meta and "root_cause" in meta:
        legacy = dict(meta.get("root_cause", {}))
        meta["diagnosis"] = {
            "primary": legacy.get("primary", "unknown"),
            "secondary": list(legacy.get("secondary", [])),
            "scores": {},
            "confidence": float(legacy.get("confidence", 0.5)),
            "mode": "single",
            "stage": legacy.get("stage"),
            "offending_stage": legacy.get("offending_stage"),
            "kill_list": list(legacy.get("kill_list", [])),
            "trace": [],
            "collapse_gradient": "gradual",
            "cause_stage_mismatch": False,
            "metric_inconsistency": False,
        }

    if "summary" in meta and "mode" not in meta["summary"]:
        meta["summary"]["mode"] = str(meta.get("diagnosis", {}).get("mode", "single"))

    if "action" in meta and "expected_effect" not in meta["action"]:
        refreshed_action = get_action_directive(meta.get("diagnosis", {}))
        meta["action"] = {**meta["action"], **refreshed_action}

    return meta


def assert_contract(
    *,
    artifact_dir: Path,
    expected_schema_hash: str | None = None,
    require_frozen: bool = True,
) -> dict[str, Any]:
    meta = load_meta(artifact_dir=artifact_dir)

    if not bool(meta.get("validated", False)):
        # Use deterministic root-cause diagnosis for error message
        summary = meta.get("summary", {})
        if summary:
            failure_msg = format_failure_message(summary)
        else:
            # Fallback to original format if summary not present
            violation_obj = meta.get("violations", {})
            blockers = violation_obj.get("BLOCKER", violation_obj)
            failure_msg = (
                "Dataset learnability contract invalid: validated=false "
                f"(violations={blockers})"
            )
        raise RuntimeError(failure_msg)

    if expected_schema_hash is not None and str(meta.get("schema_hash")) != expected_schema_hash:
        raise RuntimeError(
            "Dataset schema hash mismatch; re-run preprocessing and validation. "
            f"expected={expected_schema_hash} got={meta.get('schema_hash')}"
        )

    if require_frozen:
        freeze_marker = artifact_dir / FROZEN_SNAPSHOT_FILENAME
        if not freeze_marker.exists():
            raise RuntimeError("Dataset snapshot is not frozen; validation must produce frozen snapshot ID")
        frozen_id = freeze_marker.read_text(encoding="utf-8").strip()
        if frozen_id != str(meta.get("snapshot_id", "")):
            raise RuntimeError("Frozen snapshot ID mismatch; artifact must be regenerated and revalidated")

    return meta
