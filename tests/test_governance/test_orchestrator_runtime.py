"""Runtime governance gate orchestration tests."""

from __future__ import annotations

import json

import pytest

from helix_ids.governance.entrypoint import _resolve_policy_from_env, governed_entrypoint
from helix_ids.governance.orchestrator import GateOrchestrator


def test_run_stage_sequence_allows_missing_metrics_in_non_strict_mode(tmp_path):
    event_log = tmp_path / "events.jsonl"
    orchestrator = GateOrchestrator(
        event_log_path=event_log,
        strict_missing_metrics=False,
    )

    decisions = orchestrator.run_stage_sequence(
        {
            "run_id": "run-1",
            "entrypoint": "tests.dummy",
        },
        stages=("preload",),
    )

    assert decisions
    assert all(decision.status == "PASS" for decision in decisions)


def test_posteval_ci_gate_fails_when_threshold_exceeded_in_strict_mode(tmp_path):
    orchestrator = GateOrchestrator(
        event_log_path=tmp_path / "events.jsonl",
        strict_missing_metrics=True,
    )

    context = {
        "run_id": "run-2",
        "entrypoint": "tests.dummy",
        "posteval_elapsed_seconds": 1,
        "macro_f1_ci_width": 0.20,
        "macro_f1_ci_lower": 0.9,
        "abs_macro_f1_drift": 0.01,
        "abs_macro_f1_zscore": 0.5,
    }

    with pytest.raises(RuntimeError, match="E-T2-CI-WIDTH"):
        orchestrator.run("posteval", context)


def test_entrypoint_decorator_executes_stage_payloads(monkeypatch, tmp_path):
    event_log = tmp_path / "events.jsonl"
    monkeypatch.setenv("HELIX_GATE_EVENTS", str(event_log))
    monkeypatch.setenv("HELIX_GOV_STAGE_SEQUENCE", "preload,posteval")

    @governed_entrypoint(entrypoint_id="tests.governed")
    def dummy_main():
        return {
            "governance_stages": {
                "posteval": {
                    "posteval_elapsed_seconds": 2,
                    "macro_f1_ci_width": 0.01,
                    "macro_f1_ci_lower": 0.8,
                    "abs_macro_f1_drift": 0.01,
                    "abs_macro_f1_zscore": 1.0,
                }
            }
        }

    dummy_main()

    lines = event_log.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line) for line in lines]

    stages = {event["stage"] for event in events}
    assert "preload" in stages
    assert "posteval" in stages
    assert all(event["status"] == "PASS" for event in events)


def test_entrypoint_smoke_policy_relaxes_ci_lower_bound(monkeypatch, tmp_path):
    event_log = tmp_path / "events.jsonl"
    monkeypatch.setenv("HELIX_GATE_EVENTS", str(event_log))
    monkeypatch.setenv("HELIX_GOV_STAGE_SEQUENCE", "preload,posteval")
    monkeypatch.setenv("HELIX_GOV_POLICY_PROFILE", "smoke")

    @governed_entrypoint(entrypoint_id="tests.governed.smoke")
    def dummy_main():
        return {
            "governance_stages": {
                "posteval": {
                    "posteval_elapsed_seconds": 2,
                    "macro_f1_ci_width": 0.01,
                    "macro_f1_ci_lower": 0.20,
                    "abs_macro_f1_drift": 0.01,
                    "abs_macro_f1_zscore": 3.0,
                }
            }
        }

    dummy_main()

    lines = event_log.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line) for line in lines]
    assert any(event["stage"] == "posteval" for event in events)
    assert all(event["status"] == "PASS" for event in events)


def test_smoke_policy_relaxes_drift_zscore(monkeypatch):
    monkeypatch.setenv("HELIX_GOV_POLICY_PROFILE", "smoke")
    policy = _resolve_policy_from_env()
    assert policy.drift.max_abs_z_score == pytest.approx(1_000_000.0)


def test_entrypoint_smoke_profile_relaxes_run_registry_strictness(monkeypatch, tmp_path):
    event_log = tmp_path / "events.jsonl"
    registry_log = tmp_path / "run_registry.jsonl"
    monkeypatch.setenv("HELIX_GATE_EVENTS", str(event_log))
    monkeypatch.setenv("HELIX_RUN_REGISTRY", str(registry_log))
    monkeypatch.setenv("HELIX_GOV_STAGE_SEQUENCE", "preload")
    monkeypatch.setenv("HELIX_GOV_POLICY_PROFILE", "smoke")

    @governed_entrypoint(entrypoint_id="tests.governed.smoke.registry")
    def dummy_main():
        return {
            "governance_run_record": {
                "dataset_id": "helix_full_decoupled",
                "macro_f1": 0.2,
                "seed": 42,
                # fingerprint intentionally omitted to exercise smoke non-strict path
                "lineage": {},
            }
        }

    dummy_main()

    assert registry_log.exists()
    records = [json.loads(line) for line in registry_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert records
    assert records[-1]["state"] == "accepted"


def test_stage_schema_rejects_nonfinite_metric(tmp_path):
    orchestrator = GateOrchestrator(
        event_log_path=tmp_path / "events.jsonl",
        strict_missing_metrics=True,
    )

    context = {
        "run_id": "run-3",
        "entrypoint": "tests.dummy",
        "posteval_elapsed_seconds": 1,
        "macro_f1_ci_width": float("nan"),
        "macro_f1_ci_lower": 0.8,
        "abs_macro_f1_drift": 0.01,
        "abs_macro_f1_zscore": 0.5,
    }

    with pytest.raises(RuntimeError, match="E-GATE-SCHEMA-NONFINITE:macro_f1_ci_width"):
        orchestrator.run("posteval", context)
