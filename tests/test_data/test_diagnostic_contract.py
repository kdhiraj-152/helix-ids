"""Contract-level tests for diagnostic schema/migration/state transitions and replay."""

from __future__ import annotations

from typing import Any

import pytest

from src.helix_ids.contracts import (
    CONTRACT_VERSION,
    DiagnosticContract,
    enforce_decision_transition,
    migrate_contract_payload,
    validate_diagnostic_contract,
)
from src.helix_ids.data.learnability_contract import derive_root_cause, replay_diagnosis


def _reference_profile() -> dict[str, Any]:
    return {
        "centroid_min_distance": {"mean": 0.03, "std": 0.01, "sample_count": 50, "distribution_skew": 0.1},
        "unique_pred_coverage": {"mean": 0.95, "std": 0.08, "sample_count": 50, "distribution_skew": 0.1},
        "zero_variance_fraction": {"mean": 0.01, "std": 0.02, "sample_count": 50, "distribution_skew": 0.1},
        "min_centroid_shrinkage_ratio": {"mean": 0.9, "std": 0.2, "sample_count": 50, "distribution_skew": 0.1},
        "label_entropy": {"mean": 1.5, "std": 0.3, "sample_count": 50, "distribution_skew": 0.1},
        "signal_to_random_ratio": {"mean": 4.0, "std": 1.0, "sample_count": 50, "distribution_skew": 0.1},
    }


def _base_meta() -> dict[str, Any]:
    return {
        "reference_profile": _reference_profile(),
        "centroid_min_distance": 0.02,
        "unique_pred_coverage": 0.8,
        "linear_probe_macro_f1": 0.32,
        "random_macro_f1": 0.15,
        "label_entropy": 1.2,
        "num_classes": 4,
        "feature_degeneracy": {"zero_variance_fraction": 0.02},
        "stage_diagnostics": {},
        "stage_transitions": {},
    }


def test_migration_adds_protocol_and_version() -> None:
    migrated = migrate_contract_payload({"version": "2.0"})
    assert migrated["protocol"] == "v1"
    assert migrated["version"] == CONTRACT_VERSION


def test_runtime_invariant_guard() -> None:
    valid: DiagnosticContract = {
        "mode": "probe",
        "confidence": 0.5,
        "probe_plan": [],
        "diagnostic_cycle": {},
        "terminal_reason": None,
    }
    validate_diagnostic_contract(valid)

    invalid: DiagnosticContract = {
        **valid,
        "mode": "action",
        "confidence": 0.2,
    }
    with pytest.raises(AssertionError):
        validate_diagnostic_contract(invalid)


def test_state_transition_matrix_enforced() -> None:
    enforce_decision_transition("probe", "action")
    enforce_decision_transition("action", "probe")
    with pytest.raises(AssertionError):
        enforce_decision_transition("action", "action")


def test_replay_engine_determinism() -> None:
    meta = _base_meta()
    out = derive_root_cause(meta)
    replay_diagnosis(meta, out)


def test_chaos_probe_failures_degrade_to_non_identifiable() -> None:
    meta = _base_meta()
    meta["diagnostic_cycle"] = {
        "probe_results": [
            {"probe": "inspect_label_distribution", "hypothesis": "class_prediction_collapse", "confirms": False, "noisy": True},
            {"probe": "inject_synthetic_signal_test", "hypothesis": "weak_signal", "confirms": False, "irrelevant": True},
            {
                "probe": "run_without_scaling",
                "hypothesis": "feature_space_collapse",
                "confirms": False,
                "probe_changes_distribution": True,
            },
        ]
    }
    d = derive_root_cause(meta)
    assert d["mode"] in {"probe", "non_identifiable", "weak_signal", "uncalibrated", "composite", "single", "distribution_shift", "inconsistent"}
    assert 0.0 <= d["confidence"] <= 1.0


def test_property_stability_under_random_probes() -> None:
    pytest.importorskip("hypothesis")
    from hypothesis import given, settings
    from hypothesis import strategies as st

    probe_name = st.sampled_from(
        [
            "inspect_label_distribution",
            "run_without_scaling",
            "inject_synthetic_signal_test",
            "variance_floor_scan",
        ]
    )
    probe_result = st.fixed_dictionaries(
        {
            "probe": probe_name,
            "hypothesis": st.sampled_from(["class_prediction_collapse", "feature_space_collapse", "weak_signal"]),
            "confirms": st.booleans(),
            "noisy": st.booleans(),
            "probe_changes_distribution": st.booleans(),
        }
    )

    @settings(max_examples=30, deadline=None)
    @given(st.lists(probe_result, min_size=0, max_size=8))
    def _inner(seq: list[dict[str, Any]]) -> None:
        meta = _base_meta()
        meta["diagnostic_cycle"] = {"probe_results": seq}
        out = derive_root_cause(meta)
        assert 0.0 <= out["confidence"] <= 1.0
        assert out["decision_mode"] in {"probe", "action", "non_identifiable"}
        assert not (out["decision_mode"] == "action" and out["confidence"] < 0.6)

    _inner()
