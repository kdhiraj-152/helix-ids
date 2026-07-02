import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_dirs(repo_root: Path):
    (repo_root / "results" / "manifests").mkdir(parents=True, exist_ok=True)
    (repo_root / "results" / "metrics").mkdir(parents=True, exist_ok=True)
    (repo_root / "results" / "gates").mkdir(parents=True, exist_ok=True)


def _write_files_for_run(
    repo_root: Path,
    run_id: str,
    *,
    corrupt_result=False,
    omit_csv=False,
    omit_tex=False,
    result_schema_extra=None,
    dataset_primary=None,
    model_arch=None,
    manifest_experiment_id=None,
):
    manifests_dir = repo_root / "results" / "manifests"
    metrics_dir = repo_root / "results" / "metrics"
    _ensure_dirs(repo_root)

    manifest_experiment_id = manifest_experiment_id or run_id

    metrics_json = metrics_dir / f"{run_id}.metrics.json"
    metrics_csv = metrics_dir / f"{run_id}.metrics.csv"
    metrics_tex = metrics_dir / f"{run_id}.metrics.tex"
    result_json = metrics_dir / f"{run_id}.result.json"

    # minimal metrics
    metrics_payload: dict[str, list] = {"datasets": []}
    metrics_json.write_text(json.dumps(metrics_payload), encoding="utf-8")
    if not omit_csv:
        metrics_csv.write_text("dataset_id,accuracy\n", encoding="utf-8")
    if not omit_tex:
        metrics_tex.write_text("\\begin{tabular}{}\n\\end{tabular}\n", encoding="utf-8")

    # Build minimal valid result payload
    from scripts.evaluation import benchmarks

    result_payload = {
        "schema_version": benchmarks.RESULT_SCHEMA_VERSION,
        "experiment_id": "unit",
        "lineage": {
            "run_id": run_id,
            "benchmark_id": "unit-bench",
            "variant_id": "v1",
            "manifest_hash": "mh",
            "manifest_path": str((manifests_dir / f"{manifest_experiment_id}.manifest.json").as_posix()),
            "dataset_hashes": {"raw": "r", "processed": "p1", "split": "s", "primary": "p1"},
            "config_hashes": {
                "helix_config": "h",
                "training_config": "t",
                "platform_configs": "pconf",
                "attack_params": "a",
                "schema_contract_hash": "s",
                "mapping_version": "mver",
                "mapping_hash": "mh",
                "governance_policy_hash": "gph",
                "run_config_hash": "rconf",
            },
            "schema_hash": "s",
            "mapping_version": "mver",
            "mapping_hash": "mh",
            "dataset_hash_primary": dataset_primary or "p1",
            "dataset_id": "unit-dataset",
            "model_architecture": model_arch or "unit-model",
            "model_architecture_source": "artifact_manifest",
            "git_commit": "abc123",
            "git_branch": "main",
            "fingerprint": "fp",
        },
        # lifecycle_metadata required by result schema validator
        "lifecycle_metadata": {"status": "completed"},
        "governance_metadata": {"state": "candidate", "status": "ok"},
        "reproducibility_metadata": {"seed": 42, "determinism": True, "run_id": run_id},
        "metrics": {},
        "runtime_metrics": {"dispatch": {}, "timing": {}, "runtime": {}},
        "platform_metrics": {"latency": {}},
        "artifact_metadata": {"model_architecture": model_arch or "unit-model", "model_architecture_source": "artifact_manifest", "artifact_paths": [str(result_json.as_posix())], "outputs": {}},
    }
    if result_schema_extra:
        result_payload.update(result_schema_extra)

    if corrupt_result:
        result_json.write_text("not a json", encoding="utf-8")
    else:
        result_json.write_text(json.dumps(result_payload), encoding="utf-8")

    # manifest payload
    manifest_payload = {
        "manifest_hash": "mh",
        "dataset_hash_primary": dataset_primary or "p1",
        "dataset_hashes": {"raw": "r", "processed": "p1", "split": "s", "primary": dataset_primary or "p1"},
        "model_architecture": model_arch or "unit-model",
        "model_architecture_source": "artifact_manifest",
        "schema_version": benchmarks.MANIFEST_SCHEMA_VERSION,
        # config_hashes values must match top-level cross-hash fields for Rules 3 and 4
        "config_hashes": {
            "helix_config": "h",
            "training_config": "t",
            "platform_configs": "pconf",
            "attack_params": "a",
            "schema_contract_hash": "s",  # must equal schema_hash (Rule 3)
            "mapping_version": "mver",
            "mapping_hash": "mh",  # must equal mapping_hash (Rule 4)
            "governance_policy_hash": "gph",
            "run_config_hash": "rconf",
        },
        "dataset_id": "unit-dataset",
        "dataset_roots": {"raw": "/data/raw"},
        "evaluation_mode": "dry-run",
        "experiment_id": manifest_experiment_id,
        "governance_state": "candidate",
        "mapping_version": "mver",
        "platform_targets": ["cpu"],
        "processed_hash": "p1",
        "raw_hash": "r",
        "schema_hash": "s",
        "seed": 42,
        "split_hash": "s",
        "outputs": {
            "manifest": str((manifests_dir / f"{manifest_experiment_id}.manifest.json").as_posix()),
            "metrics": {"json": str(metrics_json.as_posix()), "csv": str(metrics_csv.as_posix()), "latex": str(metrics_tex.as_posix())},
            "result_schema": str(result_json.as_posix()),
        },
    }

    mf = manifests_dir / f"{manifest_experiment_id}.manifest.json"
    mf.write_text(json.dumps(manifest_payload), encoding="utf-8")
    return mf, metrics_json, metrics_csv, metrics_tex, result_json


def _run_validator():
    # Import the validator and run main()
    import scripts.ci.validate_benchmark_outputs as validator

    return validator.main()


def _load_report(repo_root: Path):
    rp = repo_root / "results" / "gates" / "benchmark_output_validation.json"
    assert rp.exists(), "Report missing"
    return json.loads(rp.read_text(encoding="utf-8"))


def cleanup_results(repo_root: Path):
    # remove created results to keep workspace clean
    for sub in ("manifests", "metrics", "gates"):
        d = repo_root / "results" / sub
        if d.exists():
            for p in d.glob("*"):
                try:
                    p.unlink()
                except Exception:
                    pass


@pytest.fixture(autouse=True)
def clean_repo_results():
    try:
        yield
    finally:
        cleanup_results(REPO_ROOT)


def test_validator_happy_path():
    run_id = "happy-run"
    _write_files_for_run(REPO_ROOT, run_id)
    rc = _run_validator()
    assert rc == 0
    report = _load_report(REPO_ROOT)
    assert report["summary"]["result"] == "pass"
    assert len(report["manifests"]) >= 1


def test_validator_expected_manifest_gate_counts(monkeypatch: pytest.MonkeyPatch):
    expected_experiment_ids = [
        "phase1_smoke",
        "phase1_governance_ablation",
        "phase1_edge_latency",
        "phase1_drift_robustness",
    ]
    monkeypatch.setenv("BENCHMARK_EXPECTED_MANIFEST_IDS", ",".join(expected_experiment_ids))

    for experiment_id in expected_experiment_ids:
        _write_files_for_run(REPO_ROOT, f"{experiment_id}-run", manifest_experiment_id=experiment_id)

    rc = _run_validator()
    assert rc == 0
    report = _load_report(REPO_ROOT)
    assert report["summary"]["manifests_expected"] == 4
    assert report["summary"]["manifests_found"] == 4
    assert report["summary"]["manifest_validation_passed"] == 4
    assert report["summary"]["manifest_validation_failed"] == 0
    assert report["summary"]["result"] == "pass"


def test_missing_manifest_fails():
    # ensure no manifests exist
    cleanup_results(REPO_ROOT)
    rc = _run_validator()
    assert rc != 0
    report = _load_report(REPO_ROOT)
    assert report["summary"]["result"] == "fail"


def test_missing_result_json_fails():
    run_id = "missing-result"
    _, _, _, _, rj = _write_files_for_run(REPO_ROOT, run_id)
    # remove result json
    rj.unlink()
    rc = _run_validator()
    assert rc != 0
    report = _load_report(REPO_ROOT)
    assert report["summary"]["result"] == "fail"


def test_schema_violation_fails():
    # create result missing schema_version
    run_id = "bad-schema"
    _write_files_for_run(REPO_ROOT, run_id, result_schema_extra={"schema_version": "wrong"})
    rc = _run_validator()
    assert rc != 0
    report = _load_report(REPO_ROOT)
    assert report["summary"]["result"] == "fail"


def test_missing_dataset_hash_primary_fails():
    run_id = "no-primary"
    # create manifest with missing dataset_hash_primary
    manifests_dir = REPO_ROOT / "results" / "manifests"
    _ensure_dirs(REPO_ROOT)
    mf = manifests_dir / f"{run_id}.manifest.json"
    # write minimal manifest missing dataset_hash_primary
    from scripts.evaluation import benchmarks
    payload = {"manifest_hash": "mh", "dataset_hashes": {"raw": "r", "processed": "p", "split": "s", "primary": "p"}, "model_architecture": "m", "model_architecture_source": "artifact_manifest", "schema_version": benchmarks.MANIFEST_SCHEMA_VERSION, "outputs": {"metrics": {}}}
    mf.write_text(json.dumps(payload), encoding="utf-8")
    rc = _run_validator()
    assert rc != 0
    report = _load_report(REPO_ROOT)
    assert report["summary"]["result"] == "fail"


def test_missing_model_architecture_fails():
    run_id = "no-model"
    # write manifest missing model_architecture
    manifests_dir = REPO_ROOT / "results" / "manifests"
    _ensure_dirs(REPO_ROOT)
    from scripts.evaluation import benchmarks
    mf = manifests_dir / f"{run_id}.manifest.json"
    payload = {"manifest_hash": "mh", "dataset_hash_primary": "p", "dataset_hashes": {"raw": "r", "processed": "p", "split": "s", "primary": "p"}, "model_architecture_source": "artifact_manifest", "schema_version": benchmarks.MANIFEST_SCHEMA_VERSION, "outputs": {"metrics": {}}}
    mf.write_text(json.dumps(payload), encoding="utf-8")
    rc = _run_validator()
    assert rc != 0


def test_hash_mismatch_fails():
    run_id = "hash-mismatch"
    # dataset_hashes.primary differs from dataset_hash_primary
    _write_files_for_run(REPO_ROOT, run_id, dataset_primary="primaryX")
    # Now modify manifest to mismatch
    mf = REPO_ROOT / "results" / "manifests" / f"{run_id}.manifest.json"
    payload = json.loads(mf.read_text(encoding="utf-8"))
    payload["dataset_hashes"]["primary"] = "different"
    mf.write_text(json.dumps(payload), encoding="utf-8")
    rc = _run_validator()
    assert rc != 0


def test_missing_csv_or_latex_fails():
    run_id = "no-csv"
    _write_files_for_run(REPO_ROOT, run_id, omit_csv=True)
    rc = _run_validator()
    assert rc != 0

    run_id2 = "no-tex"
    _write_files_for_run(REPO_ROOT, run_id2, omit_tex=True)
    rc2 = _run_validator()
    assert rc2 != 0


def test_duplicate_experiment_id_fails_without_expected_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("BENCHMARK_EXPECTED_MANIFEST_IDS", raising=False)
    experiment_id = "dup-experiment"
    _write_files_for_run(REPO_ROOT, "dup-a", manifest_experiment_id=experiment_id)
    _write_files_for_run(REPO_ROOT, "dup-b", manifest_experiment_id=experiment_id)

    # Force two distinct files that both claim the same experiment_id payload.
    first_manifest = REPO_ROOT / "results" / "manifests" / f"{experiment_id}.manifest.json"
    second_manifest = REPO_ROOT / "results" / "manifests" / "dup-b-alt.manifest.json"
    payload = json.loads(first_manifest.read_text(encoding="utf-8"))
    payload["experiment_id"] = experiment_id
    second_manifest.write_text(json.dumps(payload), encoding="utf-8")

    rc = _run_validator()
    assert rc != 0
    report = _load_report(REPO_ROOT)
    assert report["summary"]["result"] == "fail"
    duplicate_ids = report["summary"].get("duplicate_experiment_ids", [])
    assert experiment_id in duplicate_ids


def test_empty_manifest_hash_or_schema_version_fails():
    run_id = "empty-critical-fields"
    manifest_path, *_ = _write_files_for_run(REPO_ROOT, run_id)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["manifest_hash"] = ""
    payload["schema_version"] = ""
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    rc = _run_validator()
    assert rc != 0
    report = _load_report(REPO_ROOT)
    assert report["summary"]["result"] == "fail"
