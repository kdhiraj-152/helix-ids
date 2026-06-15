"""Evaluation helpers for HELIX-IDS: family output collection, metrics, A/B evaluation."""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader

from helix_ids.models.full import HelixIDSFull
from scripts.training.data import (
    _detect_cluster_mode_collapse,
    _normalized_entropy_from_counts,
)


def _collect_eval_family_outputs(
    *,
    model: HelixIDSFull,
    loader: DataLoader,
    device: str,
    active_class_ids: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect family-label evaluation outputs (labels, logits, probs) for calibration."""
    model.eval()
    labels_chunks: list[np.ndarray] = []
    logits_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for x, _y_binary, y_family in loader:
            x_dev = x.to(device, non_blocking=True)
            _binary_logits, family_logits_dev = model(x_dev)
            family_logits = family_logits_dev.detach().to(device="cpu")

            if active_class_ids:
                allowed = [
                    int(cls)
                    for cls in sorted(active_class_ids)
                    if 0 <= int(cls) < int(family_logits.shape[1])
                ]
                if allowed:
                    mask = torch.full_like(family_logits, float("-inf"))
                    mask[:, allowed] = family_logits[:, allowed]
                    family_logits = mask

            logits_chunks.append(family_logits.numpy().astype(np.float64, copy=False))
            labels_chunks.append(
                y_family.to(device="cpu", dtype=torch.long, non_blocking=True)
                .numpy()
                .astype(np.int64, copy=False)
            )

    if not logits_chunks:
        return (
            np.array([], dtype=np.int64),
            np.empty((0, 0), dtype=np.float64),
            np.empty((0, 0), dtype=np.float64),
        )

    labels = np.concatenate(labels_chunks, axis=0).astype(np.int64, copy=False)
    logits = np.concatenate(logits_chunks, axis=0).astype(np.float64, copy=False)
    logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)

    logits_shift = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits_shift)
    probs = exp_logits / np.clip(np.sum(exp_logits, axis=1, keepdims=True), 1e-12, None)
    return labels, logits, probs


def _normalized_entropy_from_probs(probs: np.ndarray) -> float:
    """Compute mean normalized entropy in [0, 1] from class probabilities."""
    if probs.ndim != 2 or probs.shape[0] == 0:
        return 0.0
    safe = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    ent = -np.sum(safe * np.log(safe), axis=1)
    class_count = int(safe.shape[1])
    if class_count <= 1:
        return 0.0
    return float(np.mean(ent / math.log(float(class_count))))


def _ab_rejection(reason: str) -> dict[str, Any]:
    """Return standard A/B rejection response."""
    return {
        "accepted": False,
        "reason": reason,
        "tier_1_geometry_pass": False,
        "tier_2_cluster_quality_pass": False,
        "tier_3_classifier_pass": False,
        "tier_4_governance_pass": False,
        "tier_3_evaluated": False,
    }


def _validate_ab_contract(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any] | None:
    """Validate A/B contract fields. Returns error dict if invalid, None if valid."""
    required_contract_fields = [
        "dataset_id",
        "split_snapshot_id",
        "batch_size",
        "eval_label_path",
        "k",
        "seed",
    ]
    for field in required_contract_fields:
        if current.get(field) != baseline.get(field):
            return _ab_rejection(f"ab_contract_mismatch:{field}")
    return None


def _detect_feature_and_objective_changes(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> tuple[bool, bool]:
    """Detect if feature or objective changed."""
    feature_changed = str(current.get("feature_signature", "")) != str(
        baseline.get("feature_signature", "")
    )
    objective_changed = (
        str(current.get("cluster_objective", "")) != str(baseline.get("cluster_objective", ""))
        or str(current.get("cluster_spectral_affinity", ""))
        != str(baseline.get("cluster_spectral_affinity", ""))
    )
    return feature_changed, objective_changed


def _validate_track(track: str, feature_changed: bool, objective_changed: bool) -> dict[str, Any] | None:
    """Validate track against changes. Returns error dict if invalid, None if valid."""
    track_lower = str(track).strip().lower()
    if track_lower == "feature":
        if (not feature_changed) or objective_changed:
            return _ab_rejection("ab_contract_invalid_feature_track")
    elif track_lower == "objective":
        if feature_changed or (not objective_changed):
            return _ab_rejection("ab_contract_invalid_objective_track")
    else:
        return _ab_rejection(f"ab_contract_invalid_track:{track}")
    return None


def evaluate_ab_candidate(
    *,
    current: dict[str, Any],
    baseline: dict[str, Any],
    ab_track: str,
    governance_z_score: float,
    governance_z_tolerance: float,
) -> dict[str, Any]:
    """Evaluate strict tiered A/B acceptance gates and promotion rule."""
    # Contract validation
    contract_error = _validate_ab_contract(current, baseline)
    if contract_error:
        return contract_error

    # Feature/objective detection and validation
    feature_changed, objective_changed = _detect_feature_and_objective_changes(current, baseline)
    if feature_changed and objective_changed:
        return _ab_rejection("ab_anti_pattern_mixed_feature_and_objective_change")

    track_error = _validate_track(ab_track, feature_changed, objective_changed)
    if track_error:
        return track_error

    # Tier 1: Geometry
    tier_1_geometry_pass = (
        float(current.get("ratio", 0.0)) < float(baseline.get("ratio", 0.0))
        and float(current.get("min_inter", 0.0)) > float(baseline.get("min_inter", 0.0))
    )
    if not tier_1_geometry_pass:
        return {
            "accepted": False,
            "reason": "tier1_geometry_regression",
            "tier_1_geometry_pass": False,
            "tier_2_cluster_quality_pass": False,
            "tier_3_classifier_pass": False,
            "tier_4_governance_pass": False,
            "tier_3_evaluated": False,
            "delta_ratio": float(current.get("ratio", 0.0)) - float(baseline.get("ratio", 0.0)),
            "delta_min_inter": float(current.get("min_inter", 0.0)) - float(baseline.get("min_inter", 0.0)),
        }

    # Tier 2: Cluster quality
    cluster_sizes = [int(v) for v in cast(list[Any], current.get("cluster_sizes", []))]
    collapse, collapse_metrics = _detect_cluster_mode_collapse(cluster_sizes)
    tier_2_cluster_quality_pass = not collapse
    if not tier_2_cluster_quality_pass:
        return {
            "accepted": False,
            "reason": "tier2_cluster_mode_collapse",
            "tier_1_geometry_pass": True,
            "tier_2_cluster_quality_pass": False,
            "tier_3_classifier_pass": False,
            "tier_4_governance_pass": False,
            "tier_3_evaluated": False,
            **collapse_metrics,
        }

    # Tier 3: Classifier quality
    current_zero_prediction_classes = float(current.get("zero_prediction_classes", 0.0))
    eps = 1e-9
    no_zero_prediction_classes = current_zero_prediction_classes < eps
    baseline_macro_f1 = float(baseline.get("macro_f1", 0.0))
    baseline_zero_prediction_classes = float(baseline.get("zero_prediction_classes", 0.0))
    current_macro_f1 = float(current.get("macro_f1", 0.0))
    tier_3_classifier_pass = (
        no_zero_prediction_classes
        and current_macro_f1 > (baseline_macro_f1 - eps)
        and current_zero_prediction_classes < (baseline_zero_prediction_classes + eps)
    )
    if not tier_3_classifier_pass:
        return {
            "accepted": False,
            "reason": "tier3_classifier_surface_regression",
            "tier_1_geometry_pass": True,
            "tier_2_cluster_quality_pass": True,
            "tier_3_classifier_pass": False,
            "tier_4_governance_pass": False,
            "tier_3_evaluated": True,
            "delta_macro_f1": float(current.get("macro_f1", 0.0))
            - float(baseline.get("macro_f1", 0.0)),
        }

    # Tier 4: Governance
    tier_4_governance_pass = abs(float(governance_z_score)) <= float(governance_z_tolerance)
    accepted = bool(
        tier_1_geometry_pass and tier_2_cluster_quality_pass and tier_3_classifier_pass and tier_4_governance_pass
    )
    return {
        "accepted": accepted,
        "reason": "ok" if accepted else "tier4_governance_drift_out_of_tolerance",
        "tier_1_geometry_pass": bool(tier_1_geometry_pass),
        "tier_2_cluster_quality_pass": bool(tier_2_cluster_quality_pass),
        "tier_3_classifier_pass": bool(tier_3_classifier_pass),
        "tier_4_governance_pass": bool(tier_4_governance_pass),
        "tier_3_evaluated": True,
        "delta_ratio": float(current.get("ratio", 0.0)) - float(baseline.get("ratio", 0.0)),
        "delta_min_inter": float(current.get("min_inter", 0.0)) - float(baseline.get("min_inter", 0.0)),
        "delta_macro_f1": float(current.get("macro_f1", 0.0)) - float(baseline.get("macro_f1", 0.0)),
        "governance_z_score": float(governance_z_score),
        "governance_z_tolerance": float(governance_z_tolerance),
        **collapse_metrics,
    }


def _build_ab_raw_metrics(
    *,
    dataset_name: str,
    dataset_id: str,
    split_snapshot_id: str,
    ab_track: str,
    ab_change_id: str,
    k: int,
    seed: int,
    batch_size: int,
    feature_signature: str,
    cluster_objective: str,
    cluster_spectral_affinity: str,
    representation_diagnostics: dict[str, Any],
    dataset_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build raw per-run A/B metrics payload for a single dataset."""
    cluster_diag = cast(
        dict[str, Any],
        representation_diagnostics.get("cluster_relabel", representation_diagnostics.get("original", {})),
    )
    bridge = cast(dict[str, Any], representation_diagnostics.get("cluster_label_bridge", {}))

    ratio = float(cluster_diag.get("intra_inter_ratio", 0.0))
    min_inter = float(cluster_diag.get("min_inter_center_distance", 0.0))
    nearest_center_acc_val = float(cluster_diag.get("nearest_center_accuracy_val", 0.0))
    collision_count = int(len(cast(list[Any], cluster_diag.get("collision_pairs", []))))
    nearest_cluster_pairs_top5 = cast(list[Any], cluster_diag.get("nearest_cluster_pairs_top5", []))
    density_variance = float(cluster_diag.get("density_variance", 0.0))

    cluster_sizes = [
        int(v)
        for v in cast(list[Any], representation_diagnostics.get("cluster_size_counts", []))
    ]
    if not cluster_sizes:
        cluster_to_old = cast(dict[str, Any], bridge.get("cluster_to_old_counts", {}))
        if cluster_to_old:
            ordered_cluster_ids = sorted(cluster_to_old.keys(), key=lambda value: int(value))
            cluster_sizes = [
                int(sum(int(count) for count in cast(dict[str, Any], cluster_to_old[c_id]).values()))
                for c_id in ordered_cluster_ids
            ]
    cluster_size_entropy = float(
        representation_diagnostics.get(
            "cluster_size_entropy",
            _normalized_entropy_from_counts(cluster_sizes),
        )
    )

    purity_map = cast(dict[str, Any], bridge.get("old_to_cluster_purity", {}))
    cluster_purity = [
        float(purity_map[label])
        for label in sorted(purity_map.keys(), key=lambda value: int(value))
    ]
    macro_f1 = float(dataset_metrics.get("family_macro_f1", dataset_metrics.get("family_f1", 0.0)))
    zero_prediction_classes = float(dataset_metrics.get("family_zero_prediction_classes", 0.0))

    return {
        "timestamp": datetime.now().isoformat(),
        "dataset": str(dataset_name),
        "dataset_id": str(dataset_id),
        "ab_track": str(ab_track),
        "ab_change_id": str(ab_change_id),
        "k": int(k),
        "seed": int(seed),
        "ratio": float(ratio),
        "min_inter": float(min_inter),
        "cluster_sizes": cluster_sizes,
        "cluster_purity": cluster_purity,
        "macro_f1": float(macro_f1),
        "nearest_center_acc_val": float(nearest_center_acc_val),
        "nearest_center_acc": float(nearest_center_acc_val),
        "cluster_size_entropy": float(cluster_size_entropy),
        "collision_count": int(collision_count),
        "top5_pairs": nearest_cluster_pairs_top5,
        "density_variance": float(density_variance),
        "zero_prediction_classes": float(zero_prediction_classes),
        "split_snapshot_id": str(split_snapshot_id),
        "batch_size": int(batch_size),
        "eval_label_path": "cpu",
        "feature_signature": str(feature_signature),
        "cluster_objective": str(cluster_objective),
        "cluster_spectral_affinity": str(cluster_spectral_affinity),
    }
