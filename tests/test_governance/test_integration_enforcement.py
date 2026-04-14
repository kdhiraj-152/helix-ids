"""Integration enforcement tests for non-bypassable governance pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from helix_ids.governance.entrypoint import governed_entrypoint


def _base_stages() -> dict[str, dict[str, float | int | bool]]:
    return {
        "presplit": {
            "presplit_elapsed_seconds": 1.0,
            "split_train_rows": 1000,
            "split_binary_class_count": 2,
        },
        "pretrain": {
            "pretrain_elapsed_seconds": 1.0,
            "family_class_weight_min": 0.1,
            "binary_class_weight_min": 0.1,
        },
        "intrain": {
            "intrain_elapsed_seconds": 5.0,
            "low_entropy_consecutive_batches": 0,
            "gradient_dominance": 0.1,
            "epochs_without_improvement": 0,
        },
        "posteval": {
            "posteval_elapsed_seconds": 1.0,
            "dataset_identity_balanced_accuracy": 0.6,
            "macro_f1_ci_width": 0.02,
            "macro_f1_ci_lower": 0.7,
            "abs_macro_f1_drift": 0.01,
            "abs_macro_f1_zscore": 0.5,
        },
        "prepromote": {
            "prepromote_elapsed_seconds": 1.0,
            "seed_run_count": 3,
            "inter_seed_macro_f1_variance": 0.0001,
            "reproducibility_delta": 0.005,
            "consensus_pass": True,
            "macro_f1_ci_width": 0.02,
            "macro_f1_ci_lower": 0.7,
        },
    }


def _lineage(model_artifact: Path, metrics_artifact: Path, dataset_hashes: str = "h1") -> dict[str, str]:
    return {
        "dataset_hashes": dataset_hashes,
        "schema_hash": "s1",
        "mapping_version": "m1",
        "model_artifact": str(model_artifact),
        "metrics_artifact": str(metrics_artifact),
    }


def _make_pipeline(run_record: dict[str, object], stages: dict[str, dict[str, float | int | bool]]):
    @governed_entrypoint(entrypoint_id="tests.integration.pipeline")
    def pipeline():
        return {
            "governance_context": {"seed": 42},
            "governance_stages": stages,
            "governance_run_record": run_record,
        }

    return pipeline


def _setup_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path, Path]:
    events_path = tmp_path / "events.jsonl"
    registry_path = tmp_path / "run_registry.jsonl"
    failure_path = tmp_path / "failure_memory.jsonl"

    monkeypatch.setenv("HELIX_RUN_ID", "run-1")
    monkeypatch.setenv("HELIX_GATE_EVENTS", str(events_path))
    monkeypatch.setenv("HELIX_RUN_REGISTRY", str(registry_path))
    monkeypatch.setenv("HELIX_FAILURE_MEMORY", str(failure_path))
    monkeypatch.setenv("HELIX_PARENT_RUN_ID", "")

    return events_path, registry_path, failure_path


def test_full_pipeline_valid_passes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    events_path, _, _ = _setup_env(monkeypatch, tmp_path)
    model_artifact = tmp_path / "model.pt"
    metrics_artifact = tmp_path / "metrics.json"
    model_artifact.write_text("model", encoding="utf-8")
    metrics_artifact.write_text("{}", encoding="utf-8")

    pipeline = _make_pipeline(
        {
            "dataset_id": "dataset-a",
            "macro_f1": 0.80,
            "fingerprint": "fp-a",
            "lineage": _lineage(model_artifact, metrics_artifact),
        },
        _base_stages(),
    )

    pipeline()

    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line) for line in lines]
    assert any(event["gate"] == "promotion_contract" and event["status"] == "PASS" for event in events)


def test_missing_lineage_is_immediately_invalid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    events_path, _, _ = _setup_env(monkeypatch, tmp_path)

    pipeline = _make_pipeline(
        {
            "dataset_id": "dataset-a",
            "macro_f1": 0.80,
            "fingerprint": "fp-a",
        },
        _base_stages(),
    )

    with pytest.raises(RuntimeError, match="E-LINEAGE-MISSING-CHAIN-INVALID"):
        pipeline()

    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line) for line in lines]
    assert any(event["gate"] == "reproducibility_check" and event["status"] == "INVALID" for event in events)


def test_lineage_hash_mismatch_is_immediately_invalid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _, _, _ = _setup_env(monkeypatch, tmp_path)
    model_artifact = tmp_path / "model.pt"
    metrics_artifact = tmp_path / "metrics.json"
    model_artifact.write_text("model", encoding="utf-8")
    metrics_artifact.write_text("{}", encoding="utf-8")

    root_pipeline = _make_pipeline(
        {
            "dataset_id": "dataset-a",
            "macro_f1": 0.80,
            "fingerprint": "fp-a",
            "lineage": _lineage(model_artifact, metrics_artifact, dataset_hashes="h1"),
        },
        _base_stages(),
    )
    root_pipeline()

    monkeypatch.setenv("HELIX_PARENT_RUN_ID", "run-1")
    monkeypatch.setenv("HELIX_RUN_ID", "run-2")

    child_pipeline = _make_pipeline(
        {
            "dataset_id": "dataset-a",
            "macro_f1": 0.80,
            "fingerprint": "fp-a",
            "lineage": _lineage(model_artifact, metrics_artifact, dataset_hashes="h2"),
        },
        _base_stages(),
    )

    with pytest.raises(RuntimeError, match="E-LINEAGE-HASH-MISMATCH-INVALID"):
        child_pipeline()


def test_metric_tampering_is_immediately_invalid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _, _, _ = _setup_env(monkeypatch, tmp_path)
    model_artifact = tmp_path / "model.pt"
    metrics_artifact = tmp_path / "metrics.json"
    model_artifact.write_text("model", encoding="utf-8")
    metrics_artifact.write_text("{}", encoding="utf-8")

    pipeline = _make_pipeline(
        {
            "dataset_id": "dataset-a",
            "macro_f1": 1.5,
            "fingerprint": "fp-a",
            "lineage": _lineage(model_artifact, metrics_artifact),
        },
        _base_stages(),
    )

    with pytest.raises(RuntimeError, match="E-METRIC-TAMPERING-INVALID"):
        pipeline()


def test_single_seed_promotion_path_is_invalid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    events_path, _, _ = _setup_env(monkeypatch, tmp_path)
    model_artifact = tmp_path / "model.pt"
    metrics_artifact = tmp_path / "metrics.json"
    model_artifact.write_text("model", encoding="utf-8")
    metrics_artifact.write_text("{}", encoding="utf-8")

    stages = _base_stages()
    stages["prepromote"]["seed_run_count"] = 1

    pipeline = _make_pipeline(
        {
            "dataset_id": "dataset-a",
            "macro_f1": 0.80,
            "fingerprint": "fp-a",
            "lineage": _lineage(model_artifact, metrics_artifact),
        },
        stages,
    )

    with pytest.raises(RuntimeError, match="E-T3-SINGLE-SEED-INVALID"):
        pipeline()

    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line) for line in lines]
    assert any(event["gate"] == "promotion_contract" and event["status"] == "INVALID" for event in events)


def test_same_seed_reproducibility_delta_invalid_logged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    events_path, _, _ = _setup_env(monkeypatch, tmp_path)
    model_artifact = tmp_path / "model.pt"
    metrics_artifact = tmp_path / "metrics.json"
    model_artifact.write_text("model", encoding="utf-8")
    metrics_artifact.write_text("{}", encoding="utf-8")

    first = _make_pipeline(
        {
            "dataset_id": "dataset-a",
            "macro_f1": 0.80,
            "fingerprint": "fp-a",
            "lineage": _lineage(model_artifact, metrics_artifact),
        },
        _base_stages(),
    )
    first()

    monkeypatch.setenv("HELIX_RUN_ID", "run-2")
    second = _make_pipeline(
        {
            "dataset_id": "dataset-a",
            "macro_f1": 0.83,
            "fingerprint": "fp-a",
            "lineage": _lineage(model_artifact, metrics_artifact),
        },
        _base_stages(),
    )

    with pytest.raises(RuntimeError, match="E-REPRODUCIBILITY-DELTA-INVALID"):
        second()

    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line) for line in lines]
    assert any(event["gate"] == "reproducibility_check" and event["status"] == "INVALID" for event in events)
