"""Unit tests for continuous diagnosis reducer in learnability contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from helix_ids.data.learnability_contract import (
    create_summary,
    derive_root_cause,
    extract_feature_kill_list,
    format_failure_message,
    get_action_directive,
    load_meta,
    rank_failure_stages,
)


def _reference_profile() -> dict[str, Any]:
    return {
        "centroid_min_distance": {"mean": 0.03, "std": 0.01, "sample_count": 50, "distribution_skew": 0.1},
        "unique_pred_coverage": {"mean": 0.95, "std": 0.08, "sample_count": 50, "distribution_skew": 0.1},
        "zero_variance_fraction": {"mean": 0.01, "std": 0.02, "sample_count": 50, "distribution_skew": 0.1},
        "min_centroid_shrinkage_ratio": {"mean": 0.9, "std": 0.2, "sample_count": 50, "distribution_skew": 0.1},
        "label_entropy": {"mean": 1.5, "std": 0.3, "sample_count": 50, "distribution_skew": 0.1},
        "signal_to_random_ratio": {"mean": 4.0, "std": 1.0, "sample_count": 50, "distribution_skew": 0.1},
        "linear_probe_minus_random": {"mean": 0.45, "std": 0.2, "sample_count": 50, "distribution_skew": 0.1},
        "version": "v3.2",
        "source_runs": 50,
    }


def test_derive_root_cause_feature_space_collapse() -> None:
    """Feature-space collapse should dominate under extreme centroid overlap."""
    meta = {
        "reference_profile": _reference_profile(),
        "centroid_min_distance": 0.0005,  # Near-complete overlap
        "unique_pred_coverage": 0.9,
        "linear_probe_macro_f1": 0.5,
        "random_macro_f1": 0.15,
        "label_entropy": 1.5,
        "num_classes": 3,
        "feature_degeneracy": {"zero_variance_fraction": 0.01},
        "stage_diagnostics": {},
        "stage_transitions": {},
    }
    
    root_cause = derive_root_cause(meta)
    
    assert root_cause["primary"] == "feature_space_collapse"
    assert root_cause["mode"] == "single"
    assert 0.0 <= root_cause["confidence"] <= 1.0
    assert "scores" in root_cause
    assert root_cause["scores"]["feature_space_collapse"] > 0.9


def test_derive_root_cause_class_prediction_collapse() -> None:
    """Class collapse should dominate when prediction coverage is low."""
    meta = {
        "reference_profile": _reference_profile(),
        "centroid_min_distance": 0.05,  # Passes threshold
        "unique_pred_coverage": 0.6,  # Below threshold of 0.80
        "linear_probe_macro_f1": 0.5,
        "random_macro_f1": 0.15,
        "label_entropy": 1.5,
        "num_classes": 3,
        "feature_degeneracy": {"zero_variance_fraction": 0.01},
        "stage_diagnostics": {},
        "stage_transitions": {},
    }
    
    root_cause = derive_root_cause(meta)
    
    assert root_cause["primary"] == "class_prediction_collapse"
    assert root_cause["mode"] == "single"
    assert root_cause["scores"]["class_prediction_collapse"] > 0.2


def test_derive_root_cause_feature_degeneracy() -> None:
    """Feature degeneracy should dominate when zero-variance fraction is high."""
    meta = {
        "reference_profile": _reference_profile(),
        "centroid_min_distance": 0.05,  # Passes threshold
        "unique_pred_coverage": 0.85,  # Passes threshold
        "linear_probe_macro_f1": 0.5,
        "random_macro_f1": 0.15,
        "label_entropy": 1.5,
        "num_classes": 3,
        "feature_degeneracy": {"zero_variance_fraction": 0.15},  # Above threshold of 0.10
        "stage_diagnostics": {},
        "stage_transitions": {},
    }
    
    root_cause = derive_root_cause(meta)
    
    assert root_cause["primary"] == "feature_degeneracy"
    assert root_cause["mode"] == "single"
    assert root_cause["scores"]["feature_degeneracy"] > 0.9


def test_derive_root_cause_ambiguous_composite_failure() -> None:
    """Small score margin should yield composite failure mode."""
    meta = {
        "reference_profile": _reference_profile(),
        "centroid_min_distance": 0.009,
        "unique_pred_coverage": 0.55,
        "linear_probe_macro_f1": 0.5,
        "random_macro_f1": 0.15,
        "label_entropy": 1.2,
        "num_classes": 3,
        "feature_degeneracy": {"zero_variance_fraction": 0.002},
        "stage_diagnostics": {
            "raw": {"macro_f1": 0.41},
            "encoded": {"macro_f1": 0.39},
            "scaled": {"macro_f1": 0.37},
        },
        "stage_transitions": {
            "raw->encoded": {"f1_ratio": 0.95, "centroid_shrinkage_ratio": 0.95},
            "encoded->scaled": {"f1_ratio": 0.95, "centroid_shrinkage_ratio": 0.95},
        },
    }

    diagnosis = derive_root_cause(meta)

    assert diagnosis["primary"] == "composite_failure"
    assert diagnosis["mode"] == "composite"
    assert len(diagnosis["secondary"]) >= 2
    assert diagnosis["confidence"] < 0.15


def test_metric_inconsistency_overrides_primary_cause() -> None:
    """If centroid distance improves but F1 does not, metric inconsistency overrides."""
    meta = {
        "reference_profile": _reference_profile(),
        "centroid_min_distance": 0.02,
        "unique_pred_coverage": 0.78,
        "linear_probe_macro_f1": 0.35,
        "random_macro_f1": 0.15,
        "label_entropy": 1.3,
        "num_classes": 4,
        "feature_degeneracy": {"zero_variance_fraction": 0.02},
        "stage_diagnostics": {
            "raw": {"macro_f1": 0.41},
            "scaled": {"macro_f1": 0.39},
        },
        "stage_transitions": {
            "raw->scaled": {"f1_ratio": 0.95, "centroid_shrinkage_ratio": 1.10},
        },
    }

    diagnosis = derive_root_cause(meta)
    assert diagnosis["primary"] == "metric_inconsistency"
    assert diagnosis["mode"] == "inconsistent"
    assert diagnosis["metric_inconsistency"] is True
    assert diagnosis["confidence"] >= 0.8


def test_derive_root_cause_degrades_when_reference_profile_missing() -> None:
    meta = {
        "centroid_min_distance": 0.01,
        "unique_pred_coverage": 0.9,
        "linear_probe_macro_f1": 0.4,
        "random_macro_f1": 0.15,
        "label_entropy": 1.2,
        "num_classes": 3,
        "feature_degeneracy": {"zero_variance_fraction": 0.02},
        "stage_diagnostics": {},
        "stage_transitions": {},
    }
    diagnosis = derive_root_cause(meta)
    assert isinstance(diagnosis, dict)
    assert "primary" in diagnosis
    assert "mode" in diagnosis
    assert "confidence" in diagnosis
    assert "flags" in diagnosis
    assert diagnosis["mode"] == "uncalibrated"
    assert "missing_reference_profile" in diagnosis["flags"]
    assert diagnosis["confidence"] >= 0.05


def test_derive_root_cause_soft_flags_invalid_reference_profile() -> None:
    bad_profile = _reference_profile()
    bad_profile["unique_pred_coverage"] = {
        "mean": 0.95,
        "std": 0.0,
        "sample_count": 3,
        "distribution_skew": 0.1,
    }
    meta = {
        "reference_profile": bad_profile,
        "centroid_min_distance": 0.02,
        "unique_pred_coverage": 0.7,
        "linear_probe_macro_f1": 0.35,
        "random_macro_f1": 0.15,
        "label_entropy": 1.3,
        "num_classes": 4,
        "feature_degeneracy": {"zero_variance_fraction": 0.02},
        "stage_diagnostics": {},
        "stage_transitions": {},
    }

    diagnosis = derive_root_cause(meta)
    assert isinstance(diagnosis, dict)
    assert any(flag.startswith("invalid_reference_profile") for flag in diagnosis["flags"])
    assert diagnosis["confidence"] >= 0.05


def test_derive_root_cause_weak_signal_mode_when_global_intensity_low() -> None:
    reference_profile = {
        "centroid_min_distance": {"mean": 0.5, "std": 0.4, "sample_count": 50, "distribution_skew": 0.1},
        "unique_pred_coverage": {"mean": 0.9, "std": 0.4, "sample_count": 50, "distribution_skew": 0.1},
        "zero_variance_fraction": {"mean": 0.5, "std": 0.4, "sample_count": 50, "distribution_skew": 0.1},
        "min_centroid_shrinkage_ratio": {"mean": 0.5, "std": 0.4, "sample_count": 50, "distribution_skew": 0.1},
        "label_entropy": {"mean": 1.0, "std": 0.4, "sample_count": 50, "distribution_skew": 0.1},
        "signal_to_random_ratio": {"mean": 2.5, "std": 0.4, "sample_count": 50, "distribution_skew": 0.1},
        "linear_probe_minus_random": {"mean": 0.7, "std": 0.4, "sample_count": 50, "distribution_skew": 0.1},
    }
    meta = {
        "reference_profile": reference_profile,
        "centroid_min_distance": 0.9,
        "unique_pred_coverage": 1.4,
        "linear_probe_macro_f1": 0.95,
        "random_macro_f1": 0.3,
        "label_entropy": 2.0,
        "num_classes": 3,
        "feature_degeneracy": {"zero_variance_fraction": 0.0},
        "stage_diagnostics": {},
        "stage_transitions": {
            "raw->scaled": {"f1_ratio": 1.02, "centroid_shrinkage_ratio": 1.6},
        },
    }
    diagnosis = derive_root_cause(meta)
    assert diagnosis["mode"] == "weak_signal"
    assert diagnosis["primary"] == "weak_signal"
    assert diagnosis["global_intensity"] < 0.3


def test_distribution_shift_is_conditional_override() -> None:
    meta = {
        "reference_profile": _reference_profile(),
        "centroid_min_distance": 0.03,
        "unique_pred_coverage": 0.95,
        "linear_probe_macro_f1": 0.60,
        "random_macro_f1": 0.15,
        "label_entropy": 8.0,
        "num_classes": 3,
        "feature_degeneracy": {"zero_variance_fraction": 0.01},
        "stage_diagnostics": {},
        "stage_transitions": {
            "raw->scaled": {"f1_ratio": 1.0, "centroid_shrinkage_ratio": 1.0},
        },
    }
    diagnosis = derive_root_cause(meta)
    assert diagnosis["mode"] == "distribution_shift"
    assert diagnosis["primary"] == "distribution_shift"
    assert diagnosis["drift_score"] > 2.5


def test_high_drift_with_strong_primary_keeps_primary_and_sets_flag() -> None:
    meta = {
        "reference_profile": _reference_profile(),
        "centroid_min_distance": 0.0001,
        "unique_pred_coverage": 0.85,
        "linear_probe_macro_f1": 0.25,
        "random_macro_f1": 0.15,
        "label_entropy": 8.0,
        "num_classes": 3,
        "feature_degeneracy": {"zero_variance_fraction": 0.01},
        "stage_diagnostics": {},
        "stage_transitions": {
            "raw->scaled": {"f1_ratio": 0.8, "centroid_shrinkage_ratio": 0.2},
        },
    }

    diagnosis = derive_root_cause(meta)
    assert diagnosis["drift_score"] > 2.5
    assert diagnosis["primary"] != "distribution_shift"
    assert "high_drift_detected" in diagnosis["flags"]


def test_rank_failure_stages() -> None:
    """Test stage ranking by F1 drop."""
    stage_transitions = {
        "stage1->stage2": {"f1_ratio": 0.6, "centroid_shrinkage_ratio": 0.8},
    }
    
    result = rank_failure_stages(stage_transitions)
    
    assert result["primary_failure_stage"] == "stage1->stage2"
    assert result["f1_drop"] == pytest.approx(0.4, abs=0.001)
    assert len(result["stages_ranked"]) == 1


def test_extract_feature_kill_list() -> None:
    """Kill list should filter noise and cap size."""
    stage_diagnostics = {
        "stage1": {"mutual_info_delta": [-0.5, -0.3, -0.0002, -0.2, -0.1, 0.0]},
        "stage2": {"mutual_info_delta": [-0.8, -0.01, -0.0005, -0.4, -0.2, -0.6]},
    }
    
    kill_list = extract_feature_kill_list(stage_diagnostics, top_n=25, epsilon=1e-3)
    
    assert len(kill_list) <= 10
    assert all(isinstance(f, str) and f.startswith("f_") for f in kill_list)
    assert "f_2" not in kill_list  # filtered as epsilon-level noise


def test_get_action_directive_scaling_destruction() -> None:
    """Test action directive mapping for scaling_destruction."""
    diagnosis = {
        "primary": "scaling_destruction",
        "stage": "standard_scaling",
        "kill_list": ["f_0", "f_1"],
        "confidence": 0.87,
    }
    
    action = get_action_directive(diagnosis)
    
    assert action["type"] == "REMOVE_SCALING"
    assert action["target_stage"] == "standard_scaling"
    assert action["expected_effect"] == "increase centroid distance"
    assert action["confidence"] == pytest.approx(0.87, abs=1e-6)
    assert action["target_features"] == ["f_0", "f_1"]


def test_get_action_directive_feature_degeneracy() -> None:
    """Test action directive mapping for feature_degeneracy."""
    diagnosis = {
        "primary": "feature_degeneracy",
        "stage": None,
        "kill_list": ["f_5", "f_6", "f_7"],
        "confidence": 0.7,
    }
    
    action = get_action_directive(diagnosis)
    
    assert action["type"] == "DROP_FEATURES"
    assert action["target_features"] == ["f_5", "f_6", "f_7"]


def test_get_action_directive_class_prediction_collapse() -> None:
    """Test action directive mapping for class_prediction_collapse."""
    diagnosis: dict[str, Any] = {
        "primary": "class_prediction_collapse",
        "stage": None,
        "kill_list": [],
        "confidence": 0.6,
    }
    
    action = get_action_directive(diagnosis)
    
    assert action["type"] == "REBUILD_LABELS"


def test_get_action_directive_composite_failure() -> None:
    """Composite diagnosis should route to multi-stage repair."""
    diagnosis: dict[str, Any] = {
        "primary": "composite_failure",
        "stage": "encoded->scaled",
        "kill_list": ["f_1"],
        "confidence": 0.3,
    }

    action = get_action_directive(diagnosis)
    assert action["type"] == "PROBE"
    assert action["objective"] == "increase_diagnostic_confidence"


def test_get_action_directive_distribution_shift() -> None:
    diagnosis: dict[str, Any] = {
        "primary": "distribution_shift",
        "mode": "distribution_shift",
        "stage": None,
        "kill_list": [],
        "confidence": 0.7,
    }
    action = get_action_directive(diagnosis)
    assert action["type"] == "REFRESH_BASELINE"


def test_get_action_directive_suppresses_low_confidence_actions() -> None:
    diagnosis: dict[str, Any] = {
        "primary": "feature_space_collapse",
        "mode": "single",
        "stage": "raw->scaled",
        "kill_list": ["f_1"],
        "confidence": 0.1,
    }
    action = get_action_directive(diagnosis)
    assert action["type"] == "PROBE"
    assert action["reason"] == "insufficient_diagnostic_confidence"


def test_get_action_directive_suppresses_uncalibrated_actions() -> None:
    diagnosis: dict[str, Any] = {
        "primary": "class_prediction_collapse",
        "mode": "uncalibrated",
        "stage": None,
        "kill_list": [],
        "confidence": 0.6,
    }
    action = get_action_directive(diagnosis)
    assert action["type"] == "PROBE"
    assert action["reason"] == "insufficient_diagnostic_confidence"


def test_create_summary() -> None:
    """Test summary compression layer."""
    meta = {
        "validated": False,
        "reference_profile": _reference_profile(),
        "centroid_min_distance": 0.005,
        "unique_pred_coverage": 0.9,
        "linear_probe_macro_f1": 0.5,
        "random_macro_f1": 0.15,
        "label_entropy": 1.5,
        "num_classes": 3,
        "feature_degeneracy": {"zero_variance_fraction": 0.05},
        "stage_diagnostics": {},
        "stage_transitions": {},
    }
    
    summary = create_summary(meta)
    
    assert summary["status"] == "FAIL"
    assert summary["primary_issue"] in {
        "feature_space_collapse",
        "composite_failure",
        "metric_inconsistency",
    }
    assert summary["action"] in {
        "FIX_ENCODING",
        "MULTI_STAGE_REPAIR",
        "VALIDATE_METRICS",
        "FEATURE_ENGINEERING",
        "PROBE",
    }


def test_derive_root_cause_marks_non_identifiable_when_probes_exhausted() -> None:
    meta = {
        "reference_profile": _reference_profile(),
        "centroid_min_distance": 0.02,
        "unique_pred_coverage": 0.79,
        "linear_probe_macro_f1": 0.33,
        "random_macro_f1": 0.15,
        "label_entropy": 1.2,
        "num_classes": 4,
        "feature_degeneracy": {"zero_variance_fraction": 0.02},
        "stage_diagnostics": {},
        "stage_transitions": {},
        "diagnostic_cycle": {
            "iteration": 2,
            "probes_exhausted": True,
            "probe_results": [
                {"hypothesis": "class_prediction_collapse", "confirms": False},
                {"hypothesis": "label_distribution_issue", "confirms": False},
            ],
        },
    }

    diagnosis = derive_root_cause(meta)
    assert diagnosis["mode"] == "non_identifiable"
    assert diagnosis["primary"] == "irreducible_uncertainty"
    assert "irreducible_uncertainty" in diagnosis["flags"]
    assert isinstance(diagnosis["diagnostic_cycle"]["confidence_trajectory"], list)
    assert isinstance(diagnosis["confidence"], float)
    assert 0 <= diagnosis["confidence"] <= 1
    assert diagnosis["mode"] in {
        "single",
        "composite",
        "weak_signal",
        "inconsistent",
        "distribution_shift",
        "uncalibrated",
        "non_identifiable",
    }


def test_conflicting_probe_signals() -> None:
    meta = {
        "reference_profile": _reference_profile(),
        "centroid_min_distance": 0.015,
        "unique_pred_coverage": 0.82,
        "linear_probe_macro_f1": 0.31,
        "random_macro_f1": 0.15,
        "label_entropy": 1.2,
        "num_classes": 4,
        "feature_degeneracy": {"zero_variance_fraction": 0.02},
        "stage_diagnostics": {},
        "stage_transitions": {},
        "diagnostic_cycle": {
            "probe_results": [
                {"probe": "inspect_label_distribution", "hypothesis": "class_prediction_collapse", "confirms": True},
                {"probe": "inspect_label_distribution", "hypothesis": "class_prediction_collapse", "confirms": False},
            ]
        },
    }
    diagnosis = derive_root_cause(meta)
    assert any(flag.startswith("conflicting_probe_signals") for flag in diagnosis["flags"])
    traj = diagnosis["diagnostic_cycle"]["confidence_trajectory"]
    assert max(traj) - traj[0] <= 0.15 + 1e-9


def test_false_positive_probe() -> None:
    base_meta = {
        "reference_profile": _reference_profile(),
        "centroid_min_distance": 0.02,
        "unique_pred_coverage": 0.78,
        "linear_probe_macro_f1": 0.34,
        "random_macro_f1": 0.15,
        "label_entropy": 1.2,
        "num_classes": 4,
        "feature_degeneracy": {"zero_variance_fraction": 0.02},
        "stage_diagnostics": {},
        "stage_transitions": {},
    }
    with_probe = {
        **base_meta,
        "diagnostic_cycle": {
            "probe_results": [
                {
                    "probe": "inspect_label_distribution",
                    "hypothesis": "class_prediction_collapse",
                    "confirms": True,
                    "noisy": True,
                }
            ]
        },
    }
    d0 = derive_root_cause(base_meta)
    d1 = derive_root_cause(with_probe)
    assert "false_positive_probe_signal" in d1["flags"]
    assert d1["confidence"] <= d0["confidence"] + 1e-9


def test_probe_loop_stagnation() -> None:
    meta = {
        "reference_profile": _reference_profile(),
        "centroid_min_distance": 0.02,
        "unique_pred_coverage": 0.79,
        "linear_probe_macro_f1": 0.33,
        "random_macro_f1": 0.15,
        "label_entropy": 1.2,
        "num_classes": 4,
        "feature_degeneracy": {"zero_variance_fraction": 0.02},
        "stage_diagnostics": {},
        "stage_transitions": {},
        "diagnostic_cycle": {
            "probe_results": [
                {"probe": "variance_floor_scan", "hypothesis": "feature_degeneracy", "confirms": False, "noisy": True},
                {"probe": "variance_floor_scan", "hypothesis": "feature_degeneracy", "confirms": False, "noisy": True},
                {"probe": "variance_floor_scan", "hypothesis": "feature_degeneracy", "confirms": False, "noisy": True},
            ]
        },
    }
    diagnosis = derive_root_cause(meta)
    assert "probe_loop_stagnation" in diagnosis["flags"]
    assert diagnosis["mode"] == "non_identifiable"


def test_monotonic_information_gain_bound() -> None:
    meta = {
        "reference_profile": _reference_profile(),
        "centroid_min_distance": 0.02,
        "unique_pred_coverage": 0.78,
        "linear_probe_macro_f1": 0.33,
        "random_macro_f1": 0.15,
        "label_entropy": 1.2,
        "num_classes": 4,
        "feature_degeneracy": {"zero_variance_fraction": 0.02},
        "stage_diagnostics": {},
        "stage_transitions": {},
        "diagnostic_cycle": {
            "probe_results": [
                {
                    "probe": "inspect_label_distribution",
                    "hypothesis": "class_prediction_collapse",
                    "confirms": True,
                    "information_gain_bound": 0.05,
                }
            ]
        },
    }
    diagnosis = derive_root_cause(meta)
    step = diagnosis["diagnostic_cycle"]["probe_steps"][0]
    assert abs(step["new_confidence"] - step["old_confidence"]) <= step["max_allowed_delta"] + 1e-9


def test_probe_redundancy_filter_discards_similar_probe() -> None:
    meta = {
        "reference_profile": _reference_profile(),
        "centroid_min_distance": 0.0005,
        "unique_pred_coverage": 0.9,
        "linear_probe_macro_f1": 0.5,
        "random_macro_f1": 0.15,
        "label_entropy": 1.5,
        "num_classes": 3,
        "feature_degeneracy": {"zero_variance_fraction": 0.01},
        "stage_diagnostics": {},
        "stage_transitions": {},
        "diagnostic_cycle": {
            "probe_results": [
                {
                    "probe": "run_without_scaling",
                    "type": "ablation",
                    "target": "scaler",
                    "disambiguates": ["feature_space_collapse", "scaling_destruction"],
                    "hypothesis": "feature_space_collapse",
                    "confirms": False,
                }
            ]
        },
    }
    diagnosis = derive_root_cause(meta)
    assert "redundant_probe_filtered:run_without_scaling" in diagnosis["flags"]
    assert all(p.get("probe") != "run_without_scaling" for p in diagnosis["probe_candidates"])


def test_probe_budget_exhaustion_forces_non_identifiable() -> None:
    meta = {
        "reference_profile": _reference_profile(),
        "centroid_min_distance": 0.02,
        "unique_pred_coverage": 0.79,
        "linear_probe_macro_f1": 0.33,
        "random_macro_f1": 0.15,
        "label_entropy": 1.2,
        "num_classes": 4,
        "feature_degeneracy": {"zero_variance_fraction": 0.02},
        "stage_diagnostics": {},
        "stage_transitions": {},
        "probe_cost_budget": 0.1,
        "diagnostic_cycle": {
            "probe_results": [
                {"probe": "inspect_label_distribution", "hypothesis": "class_prediction_collapse", "confirms": False, "execution_cost": 0.2}
            ]
        },
    }
    diagnosis = derive_root_cause(meta)
    assert diagnosis["mode"] == "non_identifiable"
    assert diagnosis["diagnostic_cycle"]["terminal_reason"] == "budget_exhausted"


def test_probe_bias_detected_discounts_update() -> None:
    meta = {
        "reference_profile": _reference_profile(),
        "centroid_min_distance": 0.02,
        "unique_pred_coverage": 0.79,
        "linear_probe_macro_f1": 0.33,
        "random_macro_f1": 0.15,
        "label_entropy": 1.2,
        "num_classes": 4,
        "feature_degeneracy": {"zero_variance_fraction": 0.02},
        "stage_diagnostics": {},
        "stage_transitions": {},
        "diagnostic_cycle": {
            "probe_results": [
                {
                    "probe": "inspect_label_distribution",
                    "hypothesis": "class_prediction_collapse",
                    "confirms": True,
                    "probe_changes_distribution": True,
                }
            ]
        },
    }
    diagnosis = derive_root_cause(meta)
    assert "probe_bias_detected" in diagnosis["flags"]
    step = diagnosis["diagnostic_cycle"]["probe_steps"][0]
    assert abs(step["new_confidence"] - step["old_confidence"]) <= 0.075 + 1e-9


def test_probe_isolation() -> None:
    meta = {
        "reference_profile": _reference_profile(),
        "centroid_min_distance": 0.02,
        "unique_pred_coverage": 0.79,
        "linear_probe_macro_f1": 0.33,
        "random_macro_f1": 0.15,
        "label_entropy": 1.2,
        "num_classes": 4,
        "feature_degeneracy": {"zero_variance_fraction": 0.02},
        "stage_diagnostics": {},
        "stage_transitions": {},
        "diagnostic_cycle": {
            "probe_results": [
                {
                    "probe": "inspect_label_distribution",
                    "hypothesis": "class_prediction_collapse",
                    "confirms": True,
                    "affected_hypotheses": ["class_prediction_collapse", "label_distribution_issue"],
                }
            ]
        },
    }
    diagnosis = derive_root_cause(meta)
    assert "probe_not_isolated" in diagnosis["flags"]


def test_confidence_collapse_all_hypotheses_contradicted() -> None:
    meta = {
        "reference_profile": _reference_profile(),
        "centroid_min_distance": 0.015,
        "unique_pred_coverage": 0.79,
        "linear_probe_macro_f1": 0.30,
        "random_macro_f1": 0.15,
        "label_entropy": 1.1,
        "num_classes": 4,
        "feature_degeneracy": {"zero_variance_fraction": 0.02},
        "stage_diagnostics": {},
        "stage_transitions": {},
        "diagnostic_cycle": {
            "probe_results": [
                {"probe": "inspect_label_distribution", "hypothesis": "class_prediction_collapse", "confirms": False},
                {"probe": "run_without_scaling", "hypothesis": "feature_space_collapse", "confirms": False},
                {"probe": "inject_synthetic_signal_test", "hypothesis": "weak_signal", "confirms": False},
            ]
        },
    }
    diagnosis = derive_root_cause(meta)
    assert diagnosis["mode"] == "non_identifiable"
    assert diagnosis["confidence"] >= 0.05
    assert diagnosis["confidence"] <= 0.3


def test_diagnostic_cycle_log_integrity() -> None:
    meta = {
        "reference_profile": _reference_profile(),
        "centroid_min_distance": 0.02,
        "unique_pred_coverage": 0.79,
        "linear_probe_macro_f1": 0.33,
        "random_macro_f1": 0.15,
        "label_entropy": 1.2,
        "num_classes": 4,
        "feature_degeneracy": {"zero_variance_fraction": 0.02},
        "stage_diagnostics": {},
        "stage_transitions": {},
        "diagnostic_cycle": {
            "probe_results": [
                {"probe": "inspect_label_distribution", "hypothesis": "class_prediction_collapse", "confirms": True},
                {"probe": "variance_floor_scan", "hypothesis": "feature_degeneracy", "confirms": False},
            ]
        },
    }
    diagnosis = derive_root_cause(meta)
    cycle = diagnosis["diagnostic_cycle"]
    assert len(cycle["confidence_trajectory"]) == cycle["iteration"]
    assert len(cycle["probe_steps"]) == len(cycle["probes_run"])
    assert len(cycle["confidence_trajectory"]) == len(cycle["probes_run"]) + 1


def test_action_layer_frozen_until_confidence_verified() -> None:
    diagnosis = {
        "primary": "feature_space_collapse",
        "mode": "single",
        "confidence": 0.59,
        "stage": None,
        "kill_list": [],
        "probe_candidates": [],
    }
    action = get_action_directive(diagnosis)
    assert action["type"] == "PROBE"


def test_format_failure_message() -> None:
    """Test failure message formatting."""
    summary = {
        "status": "FAIL",
        "primary_issue": "scaling_destruction",
        "stage": "standard_scaling",
        "action": "REMOVE_SCALING",
        "confidence": 0.90,
    }
    
    message = format_failure_message(summary)
    
    assert "UNSW CONTRACT FAILURE" in message
    assert "scaling_destruction" in message
    assert "REMOVE_SCALING" in message
    assert "0.90" in message


def test_actual_artifact_contract_diagnosis() -> None:
    """Integration test with actual processed artifact."""
    artifact_dir = Path("data/processed/multi_dataset_v1")
    
    if not artifact_dir.exists():
        # Skip if artifact doesn't exist
        return
    
    meta_path = artifact_dir / "meta.json"
    if not meta_path.exists():
        return

    meta = load_meta(artifact_dir=artifact_dir)
    
    # Verify diagnosis structure
    assert "diagnosis" in meta
    assert "primary" in meta["diagnosis"]
    assert "secondary" in meta["diagnosis"]
    assert "confidence" in meta["diagnosis"]
    assert "scores" in meta["diagnosis"]
    assert "mode" in meta["diagnosis"]
    
    # Verify action directive
    assert "action" in meta
    assert "type" in meta["action"]
    
    # Verify summary
    assert "summary" in meta
    assert meta["summary"]["status"] in ["PASS", "FAIL"]
    assert "action" in meta["summary"]


# Skip pytest if not installed for manual running
try:
    import pytest
except ImportError:
    pytest = None  # type: ignore
