"""A/B testing evaluation and contract validation.

Extracted from train_helix_ids_full.py:
  - evaluate_ab_candidate
  - _build_ab_raw_metrics
  - _ab_rejection
  - _validate_ab_contract
  - _detect_feature_and_objective_changes
  - _validate_track
  - _detect_cluster_mode_collapse
  - _normalized_entropy_from_counts
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

import numpy as np

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ABEvaluationInput:
    """Typed input for evaluate_ab_candidate."""

    current: dict[str, Any]
    baseline: dict[str, Any]
    ab_track: str
    governance_z_score: float
    governance_z_tolerance: float


@dataclass
class ABEvaluationResult:
    """Structured A/B evaluation decision."""

    accepted: bool
    reason: str
    tier_1_geometry_pass: bool
    tier_2_cluster_quality_pass: bool
    tier_3_classifier_pass: bool
    tier_4_governance_pass: bool
    tier_3_evaluated: bool
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "accepted": self.accepted,
            "reason": self.reason,
            "tier_1_geometry_pass": self.tier_1_geometry_pass,
            "tier_2_cluster_quality_pass": self.tier_2_cluster_quality_pass,
            "tier_3_classifier_pass": self.tier_3_classifier_pass,
            "tier_4_governance_pass": self.tier_4_governance_pass,
            "tier_3_evaluated": self.tier_3_evaluated,
        }
        d.update(self.extra)
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalized_entropy_from_counts(counts: list[int]) -> float:
    """Compute normalized entropy in [0, 1] for cluster-size distribution."""
    arr = np.asarray(counts, dtype=np.float64)
    total = float(arr.sum())
    if total <= 0.0 or int(arr.shape[0]) <= 1:
        return 0.0
    probs = np.clip(arr / total, 1e-12, 1.0)
    entropy = float(-np.sum(probs * np.log(probs)) / math.log(float(arr.shape[0])))
    return float(np.clip(entropy, 0.0, 1.0))


def detect_cluster_mode_collapse(
    cluster_sizes: list[int],
    *,
    min_entropy: float = 0.30,
    max_dominance: float = 0.85,
) -> tuple[bool, dict[str, float]]:
    """Detect cluster mode collapse using entropy and dominant-cluster share."""
    counts = [max(0, int(v)) for v in cluster_sizes]
    total = int(sum(counts))
    if total <= 0:
        return True, {
            "cluster_size_entropy": 0.0,
            "dominant_cluster_fraction": 1.0,
            "active_cluster_count": 0.0,
        }

    entropy = normalized_entropy_from_counts(counts)
    dominant_fraction = float(max(counts) / max(1, total))
    active_cluster_count = int(sum(1 for count in counts if count > 0))
    collapse = (
        active_cluster_count < 2
        or dominant_fraction >= float(max_dominance)
        or entropy < float(min_entropy)
    )
    return collapse, {
        "cluster_size_entropy": float(entropy),
        "dominant_cluster_fraction": float(dominant_fraction),
        "active_cluster_count": float(active_cluster_count),
    }


def ab_rejection(reason: str) -> dict[str, Any]:
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


def validate_ab_contract(
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
    for field_name in required_contract_fields:
        if current.get(field_name) != baseline.get(field_name):
            return ab_rejection(f"ab_contract_mismatch:{field_name}")
    return None


def detect_feature_and_objective_changes(
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


def validate_track(track: str, feature_changed: bool, objective_changed: bool) -> dict[str, Any] | None:
    """Validate track against changes. Returns error dict if invalid, None if valid."""
    track_lower = str(track).strip().lower()
    if track_lower == "feature":
        if (not feature_changed) or objective_changed:
            return ab_rejection("ab_contract_invalid_feature_track")
    elif track_lower == "objective":
        if feature_changed or (not objective_changed):
            return ab_rejection("ab_contract_invalid_objective_track")
    else:
        return ab_rejection(f"ab_contract_invalid_track:{track}")
    return None


# ---------------------------------------------------------------------------
# Main A/B evaluation
# ---------------------------------------------------------------------------


def evaluate_ab_candidate(
    *,
    current: dict[str, Any],
    baseline: dict[str, Any],
    ab_track: str,
    governance_z_score: float,
    governance_z_tolerance: float,
) -> dict[str, Any]:
    """Evaluate strict tiered A/B acceptance gates and promotion rule.

    Contract validation -> feature/objective change detection -> track
    validation -> Tier 1 geometry -> Tier 2 cluster quality -> Tier 3
    classifier surface -> Tier 4 governance drift.

    Returns a dict shaped exactly as the original inline version did.
    """
    # Contract validation
    contract_error = validate_ab_contract(current, baseline)
    if contract_error:
        return contract_error

    # Feature/objective detection and validation
    feature_changed, objective_changed = detect_feature_and_objective_changes(current, baseline)
    if feature_changed and objective_changed:
        return ab_rejection("ab_anti_pattern_mixed_feature_and_objective_change")

    track_error = validate_track(ab_track, feature_changed, objective_changed)
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
    collapse, collapse_metrics = detect_cluster_mode_collapse(cluster_sizes)
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


def build_ab_raw_metrics(
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
    """Build raw per-run A/B metrics payload for a single dataset.

    Collects cluster geometry, purity, classifier surface, and run
    metadata into a single dict that serves as the "current" side of
    an A/B comparison against a baseline.
    """
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
            normalized_entropy_from_counts(cluster_sizes),
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
