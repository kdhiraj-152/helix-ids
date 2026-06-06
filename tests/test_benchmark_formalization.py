from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.evaluation.benchmarks as benchmarks

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_manifest_expansion_is_deterministic() -> None:
    manifests = [
        PROJECT_ROOT / "config" / "experiments" / "smoke.yaml",
        PROJECT_ROOT / "config" / "experiments" / "governance_ablation.yaml",
        PROJECT_ROOT / "config" / "experiments" / "edge_latency.yaml",
        PROJECT_ROOT / "config" / "experiments" / "drift_robustness.yaml",
    ]

    first_pass = [spec.run_id() for spec in benchmarks._load_all_manifests(manifests)]
    second_pass = [spec.run_id() for spec in benchmarks._load_all_manifests(list(reversed(manifests)))]

    assert first_pass == second_pass


def test_dry_run_selects_one_run_per_experiment() -> None:
    manifests = [
        PROJECT_ROOT / "config" / "experiments" / "smoke.yaml",
        PROJECT_ROOT / "config" / "experiments" / "governance_ablation.yaml",
        PROJECT_ROOT / "config" / "experiments" / "edge_latency.yaml",
        PROJECT_ROOT / "config" / "experiments" / "drift_robustness.yaml",
    ]

    specs = benchmarks._load_all_manifests(manifests)
    dry_run_specs = benchmarks._select_dry_run_specs(specs)
    experiment_ids = [spec.experiment_id for spec in dry_run_specs]

    assert len(dry_run_specs) == 4
    assert experiment_ids == sorted(
        [
            "phase1_smoke",
            "phase1_governance_ablation",
            "phase1_edge_latency",
            "phase1_drift_robustness",
        ]
    )
    assert len(experiment_ids) == len(set(experiment_ids))


def test_dry_run_manifest_and_result_schema_are_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = benchmarks.RunSpec(
        experiment_id="unit_benchmark",
        benchmark_id="dry_run",
        variant_id="base",
        seed=7,
        entrypoint="benchmark_e2e_v2_fixed",
        config={
            "evaluation_mode": "smoke",
            "platforms": ["production"],
            "platform_targets": ["production"],
        },
        dataset={
            "dataset_id": "unit_dataset",
            "dataset_roots": ["data"],
            "hashes": {
                "raw": "raw-hash",
                "processed": "processed-hash",
                "split": "split-hash",
                "primary": "processed-hash",
            },
        },
        governance={},
        manifest_path=tmp_path / "unit.yaml",
    )

    monkeypatch.setattr(benchmarks, "_resolve_artifact_paths", lambda _run: [])
    monkeypatch.setattr(
        benchmarks,
        "_resolve_git_context",
        lambda: {"commit": "abc123", "branch": "main"},
    )
    monkeypatch.setattr(
        benchmarks,
        "_runtime_environment",
        lambda: {
            "python_version": "3.11",
            "platform": "test",
            "machine": "test",
            "processor": "test",
            "torch_version": "test",
            "numpy_version": "test",
            "sklearn_version": "test",
        },
    )

    context = benchmarks._build_run_context(run, dry_run=True)
    assert context["model_architecture"] == benchmarks.DRY_RUN_MODEL_ARCHITECTURE
    assert context["model_architecture_source"] == benchmarks.DRY_RUN_MODEL_ARCHITECTURE_SOURCE
    assert context["dataset_hashes"]["primary"] == context["dataset_hash_primary"]
    context["determinism"] = {"seed": run.seed, "mode": "dry_run"}
    context["runtime"] = benchmarks._runtime_environment()

    resolved_manifest = {
        **context["manifest_base"],
        "manifest_hash": context["manifest_hash"],
        "config_hash": context["config_hash"],
        "config_hashes": context["config_hashes"],
        "dataset_hashes": context["dataset_hashes"],
        "dataset_details": context["dataset_details"],
        "fingerprint": context["fingerprint"],
        "model_architecture": context["model_architecture"],
        "model_architecture_source": context["model_architecture_source"],
        "evaluation_mode": context["evaluation_mode"],
        "platform_targets": context["platform_targets"],
        "mapping_version": context["mapping_version"],
        "mapping_hash": context["mapping_hash"],
        "git_commit": context["git_context"]["commit"],
        "git_branch": context["git_context"]["branch"],
    }

    with pytest.raises(ValueError, match="placeholder only allowed"):
        benchmarks._validate_manifest_payload(resolved_manifest)

    benchmarks._validate_manifest_payload(resolved_manifest, allow_dry_run_placeholder=True)

    result_payload = benchmarks._build_result_schema(
        run=run,
        context=context,
        metrics_payload={"datasets": [], "note": "dry_run"},
        dispatch_status={"status": "skipped", "reason": "dry_run"},
        lifecycle_status={"status": "skipped", "reason": "dry_run"},
        ast_status={"status": "pass", "violations": []},
        artifact_status={"status": "skipped"},
        timing={"total_seconds": 0.0},
        outputs={"manifest": str(tmp_path / "unit.manifest.json"), "artifact_paths": [], "metrics": {}},
    )

    benchmarks._validate_result_schema(result_payload)
    assert result_payload["lineage"]["dataset_hashes"]["primary"] == result_payload["lineage"]["dataset_hash_primary"]
    assert result_payload["lineage"]["model_architecture_source"] == benchmarks.DRY_RUN_MODEL_ARCHITECTURE_SOURCE
    assert result_payload["artifact_metadata"]["model_architecture_source"] == benchmarks.DRY_RUN_MODEL_ARCHITECTURE_SOURCE

    schema_text = benchmarks.RESULT_SCHEMA_PATH.read_text(encoding="utf-8")
    schema = json.loads(schema_text)
    assert "lineage" in schema["properties"]
    assert "artifact_metadata" in schema["required"]
