"""Run registry lineage and failure-memory tests."""

from __future__ import annotations

import json

from helix_ids.governance.run_registry import RunRegistry


def _read_records(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_validate_and_register_writes_rejected_record_for_missing_parent(tmp_path):
    registry_path = tmp_path / "run_registry.jsonl"
    registry = RunRegistry(registry_path)

    decision = registry.validate_and_register(
        run_id="run-child",
        dataset_id="dataset-a",
        macro_f1=0.81,
        fingerprint="fp-a",
        parent_run_id="missing-parent",
        seed=42,
        lineage={
            "dataset_hashes": "h1",
            "schema_hash": "s1",
            "mapping_version": "m1",
            "model_artifact": "model.pt",
            "metrics_artifact": "metrics.json",
        },
        tolerance=1e-6,
        strict_lineage=True,
        strict_orphan_artifacts=False,
    )

    assert decision.accepted is False
    assert decision.reason_code == "E-LINEAGE-MISSING-PARENT-INVALID"

    records = _read_records(registry_path)
    assert len(records) == 1
    assert records[0]["state"] == "rejected"
    assert records[0]["reason_code"] == "E-LINEAGE-MISSING-PARENT-INVALID"


def test_validate_and_register_accepts_and_persists_record(tmp_path):
    registry_path = tmp_path / "run_registry.jsonl"
    registry = RunRegistry(registry_path)

    decision = registry.validate_and_register(
        run_id="run-root",
        dataset_id="dataset-a",
        macro_f1=0.82,
        fingerprint="fp-a",
        parent_run_id=None,
        seed=42,
        lineage={
            "dataset_hashes": "h1",
            "schema_hash": "s1",
            "mapping_version": "m1",
            "model_artifact": "model.pt",
            "metrics_artifact": "metrics.json",
        },
        tolerance=1e-6,
        strict_lineage=True,
        strict_orphan_artifacts=False,
    )

    assert decision.accepted is True
    assert decision.reason_code == "OK"

    records = _read_records(registry_path)
    assert len(records) == 1
    assert records[0]["state"] == "accepted"
    assert records[0]["run_id"] == "run-root"
