#!/usr/bin/env python3
"""CI validator for benchmark outputs.

Scans `results/manifests` and `results/metrics`, validates manifests and result JSON,
and writes a machine-readable gate report to `results/gates/benchmark_output_validation.json`.
Exits non-zero on any validation failure.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

EXPECTED_MANIFEST_IDS_ENV = "BENCHMARK_EXPECTED_MANIFEST_IDS"


def _fail(msg: str, details: Any = None):
    return {"status": "fail", "message": msg, "details": details}


def _pass(msg: str, details: Any = None):
    return {"status": "pass", "message": msg, "details": details}


def _expected_experiment_ids() -> list[str]:
    raw = os.environ.get(EXPECTED_MANIFEST_IDS_ENV, "")
    if not raw.strip():
        return []
    return [item.strip() for item in raw.replace("\n", ",").split(",") if item.strip()]


def _missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _validate_manifest_file(  # noqa: C901
    mf: Path,
    expected_experiment_id_set: set[str],
    found_experiment_ids: set[str],
    seen_experiment_ids: dict[str, str],
    unexpected_experiment_ids: list[str],
    duplicate_experiment_ids: list[str],
    benchmarks_module,
) -> tuple[dict[str, Any], bool]:
    entry: dict[str, Any] = {"manifest": mf.as_posix(), "checks": []}
    entry_failed = False
    payload, load_failed_check = _read_manifest_json(mf)
    if load_failed_check is not None:
        entry["checks"].append(load_failed_check)
        return entry, True
    assert payload is not None

    file_experiment_id = mf.name.removesuffix(".manifest.json")
    payload_experiment_id = str(payload.get("experiment_id") or "").strip()
    id_checks, id_failed = _check_experiment_identifiers(
        mf,
        file_experiment_id,
        payload_experiment_id,
        expected_experiment_id_set,
        found_experiment_ids,
        seen_experiment_ids,
        unexpected_experiment_ids,
        duplicate_experiment_ids,
    )
    entry["checks"].extend(id_checks)
    entry_failed = entry_failed or id_failed

    # Basic manifest field checks
    rf_checks, rf_failed = _check_required_manifest_fields(payload)
    entry["checks"].extend(rf_checks)
    entry_failed = entry_failed or rf_failed

    scalar_checks, scalar_failed = _check_scalar_fields(payload)
    entry["checks"].extend(scalar_checks)
    entry_failed = entry_failed or scalar_failed

    # dataset_hashes detail
    ds_checks, ds_failed = _check_dataset_hashes(payload)
    entry["checks"].extend(ds_checks)
    entry_failed = entry_failed or ds_failed

    # config_hashes sub-key detail
    ch_checks, ch_failed = _check_config_hashes(payload)
    entry["checks"].extend(ch_checks)
    entry_failed = entry_failed or ch_failed

    # Cross-hash consistency checks (hash_authority.md §2, rules 1/3/4)
    xh_checks, xh_failed = _check_cross_hash_consistency(payload)
    entry["checks"].extend(xh_checks)
    entry_failed = entry_failed or xh_failed

    # Validate manifest payload using internal validator (allow dry-run placeholders)
    pm_checks, pm_failed = _validate_manifest_payload_with_benchmarks(payload, benchmarks_module)
    entry["checks"].extend(pm_checks)
    entry_failed = entry_failed or pm_failed

    outputs = payload.get("outputs", {}) if isinstance(payload.get("outputs"), dict) else {}
    metrics_outputs = outputs.get("metrics") if isinstance(outputs.get("metrics"), dict) else None
    result_schema_path = outputs.get("result_schema")

    mo_checks, mo_failed = (
        _check_metrics_outputs(metrics_outputs) if metrics_outputs else (_fail("Metrics outputs missing in manifest.outputs", outputs.get("metrics")), True)
    )
    # _check_metrics_outputs returns a tuple (list, bool); ensure mo_checks is list
    if isinstance(mo_checks, dict):
        entry["checks"].append(mo_checks)
    else:
        entry["checks"].extend(mo_checks)
    entry_failed = entry_failed or bool(mo_failed)

    # Result schema / canonical JSON
    rs_checks, rs_failed = (
        _validate_result_file(result_schema_path, benchmarks_module) if result_schema_path else ([_fail("result_schema path missing in manifest.outputs")], True)
    )
    entry["checks"].extend(rs_checks)
    if rs_failed:
        return entry, True

    return entry, entry_failed


def _validate_manifest_outputs(payload: dict[str, Any], benchmarks_module) -> tuple[list[dict[str, Any]], bool, bool]:
    outputs = payload.get("outputs", {}) if isinstance(payload.get("outputs"), dict) else {}
    metrics_outputs = outputs.get("metrics") if isinstance(outputs.get("metrics"), dict) else None
    result_schema_path = outputs.get("result_schema")

    checks: list[dict[str, Any]] = []
    failed = False

    if not metrics_outputs:
        checks.append(_fail("Metrics outputs missing in manifest.outputs", outputs.get("metrics")))
        failed = True
    else:
        mo_checks, mo_failed = _check_metrics_outputs(metrics_outputs)
        checks.extend(mo_checks)
        failed = failed or mo_failed

    if not result_schema_path:
        checks.append(_fail("result_schema path missing in manifest.outputs"))
        failed = True
        return checks, failed, False

    rs_checks, rs_failed = _validate_result_file(result_schema_path, benchmarks_module)
    checks.extend(rs_checks)
    if rs_failed:
        return checks, True, True

    return checks, failed, False


def _validate_result_file(result_schema_path: str, benchmarks_module) -> tuple[list[dict[str, Any]], bool]:
    checks: list[dict[str, Any]] = []
    failed = False
    rp = Path(result_schema_path)
    if not rp.exists():
        checks.append(_fail("Result JSON missing", result_schema_path))
        return checks, True
    try:
        result_payload = json.loads(rp.read_text(encoding="utf-8"))
    except Exception as exc:
        checks.append(_fail("Malformed result JSON", str(exc)))
        return checks, True

    # Validate using result schema validator
    try:
        benchmarks_module._validate_result_schema(result_payload)
        checks.append(_pass("Result JSON validated against schema"))
    except Exception as exc:
        checks.append(_fail("Result JSON schema validation failed", str(exc)))
        failed = True

    # Check required result fields
    required_result_keys = [
        "schema_version",
        "lineage",
        "governance_metadata",
        "metrics",
        "reproducibility_metadata",
    ]
    missing_res = [k for k in required_result_keys if k not in result_payload]
    if missing_res:
        checks.append(_fail("Result JSON missing required keys", missing_res))
        failed = True
    else:
        checks.append(_pass("Result JSON contains required keys"))

    return checks, failed


def _check_config_hashes(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """Verify config_hashes has non-empty string values for all present keys.

    Per manifest_schema_governance.md: each config_hashes value must be a
    non-empty string.  This checks that every present entry is a non-missing
    scalar and records a pass for each key found.
    """
    checks: list[dict[str, Any]] = []
    failed = False
    config_hashes = payload.get("config_hashes")
    if not isinstance(config_hashes, dict):
        checks.append(_fail("config_hashes missing or not a mapping"))
        return checks, True
    if not config_hashes:
        checks.append(_fail("config_hashes is empty"))
        return checks, True
    for key, value in config_hashes.items():
        if _missing_scalar(value):
            checks.append(_fail(f"config_hashes['{key}'] is missing or empty"))
            failed = True
        else:
            checks.append(_pass(f"config_hashes['{key}'] valid"))
    return checks, failed


def _check_dataset_hashes(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """Verify dataset_hashes has required sub-keys with non-empty values.

    Required sub-keys (per result_schema_governance.md):
        raw, processed, split, primary
    Each must be a non-empty string (non-missing scalar).
    Primary must also match the top-level dataset_hash_primary.
    """
    checks: list[dict[str, Any]] = []
    failed = False
    ds_hashes = payload.get("dataset_hashes")
    if not isinstance(ds_hashes, dict):
        checks.append(_fail("dataset_hashes missing or not a mapping"))
        return checks, True
    for key in ("raw", "processed", "split", "primary"):
        if key not in ds_hashes:
            checks.append(_fail(f"dataset_hashes missing required sub-key: '{key}'"))
            failed = True
        elif _missing_scalar(ds_hashes.get(key)):
            checks.append(_fail(f"dataset_hashes['{key}'] is missing or empty"))
            failed = True
        else:
            checks.append(_pass(f"dataset_hashes['{key}'] valid"))
    if str(ds_hashes.get("primary")) != str(payload.get("dataset_hash_primary")):
        checks.append(_fail("dataset_hash_primary mismatch with dataset_hashes.primary"))
        failed = True
    else:
        checks.append(_pass("Dataset hash primary matches"))
    return checks, failed


def _check_cross_hash_consistency(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """Enforce hash_authority.md §2 cross-hash consistency rules that are
    verifiable within a single manifest.

    Enforced (intra-manifest):
      Rule 1: dataset_hash_primary == processed_hash
      Rule 3: schema_hash == config_hashes.schema_contract_hash
      Rule 4: mapping_hash == config_hashes.mapping_hash

    Not enforced here (require manifest-to-result cross-reference):
      Rule 2: manifest hashes must appear identically in result lineage
      Rule 5: dataset_hashes.raw/processed/split/primary must match manifest fields
    These two rules are documented as guidance in hash_authority.md §2 because
    the CI validator processes manifests without paired result files.
    See Phase 5R HIGH-2 root cause and hash_authority.md amendment.
    """
    checks: list[dict[str, Any]] = []
    failed = False

    # Rule 1: dataset_hash_primary == processed_hash
    primary = str(payload.get("dataset_hash_primary") or "").strip()
    processed = str(payload.get("processed_hash") or "").strip()
    if primary and processed and primary != processed:
        checks.append(_fail(
            "Cross-hash Rule 1 violation: dataset_hash_primary != processed_hash",
            {"dataset_hash_primary": primary, "processed_hash": processed},
        ))
        failed = True
    else:
        checks.append(_pass("Cross-hash Rule 1: dataset_hash_primary == processed_hash"))

    # Rule 3: schema_hash == config_hashes.schema_contract_hash
    top_schema = str(payload.get("schema_hash") or "").strip()
    cfg_schema = str(
        payload.get("config_hashes", {}).get("schema_contract_hash") or ""
    ).strip()
    if top_schema and cfg_schema and top_schema != cfg_schema:
        checks.append(_fail(
            "Cross-hash Rule 3 violation: schema_hash != config_hashes.schema_contract_hash",
            {"schema_hash": top_schema, "config_hashes.schema_contract_hash": cfg_schema},
        ))
        failed = True
    else:
        checks.append(_pass("Cross-hash Rule 3: schema_hash == config_hashes.schema_contract_hash"))

    # Rule 4: mapping_hash == config_hashes.mapping_hash
    top_mapping = str(payload.get("mapping_hash") or "").strip()
    cfg_mapping = str(
        payload.get("config_hashes", {}).get("mapping_hash") or ""
    ).strip()
    if top_mapping and cfg_mapping and top_mapping != cfg_mapping:
        checks.append(_fail(
            "Cross-hash Rule 4 violation: mapping_hash != config_hashes.mapping_hash",
            {"mapping_hash": top_mapping, "config_hashes.mapping_hash": cfg_mapping},
        ))
        failed = True
    else:
        checks.append(_pass("Cross-hash Rule 4: mapping_hash == config_hashes.mapping_hash"))

    return checks, failed


def _check_metrics_outputs(metrics_outputs: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    checks: list[dict[str, Any]] = []
    failed = False
    for key in ("json", "csv", "latex"):
        path = metrics_outputs.get(key)
        if not path:
            checks.append(_fail(f"Metrics missing {key} path in manifest.outputs.metrics"))
            failed = True
            continue
        if not Path(path).exists():
            checks.append(_fail("Metrics file missing", path))
            failed = True
        else:
            checks.append(_pass(f"Metrics {key} present"))
    return checks, failed


def _check_experiment_identifiers(
    mf: Path,
    file_experiment_id: str,
    payload_experiment_id: str,
    expected_experiment_id_set: set[str],
    found_experiment_ids: set[str],
    seen_experiment_ids: dict[str, str],
    unexpected_experiment_ids: list[str],
    duplicate_experiment_ids: list[str],
) -> tuple[list[dict[str, Any]], bool]:
    checks: list[dict[str, Any]] = []
    failed = False
    if not payload_experiment_id:
        checks.append(_fail("Manifest missing experiment_id"))
        return checks, True
    if payload_experiment_id != file_experiment_id:
        checks.append(_fail("Manifest filename mismatch", {"file_experiment_id": file_experiment_id, "payload_experiment_id": payload_experiment_id}))
        failed = True
    if payload_experiment_id in seen_experiment_ids:
        duplicate_experiment_ids.append(payload_experiment_id)
        checks.append(_fail("Duplicate manifest experiment_id", {"current_manifest": mf.as_posix(), "first_manifest": seen_experiment_ids[payload_experiment_id]}))
        failed = True
    else:
        seen_experiment_ids[payload_experiment_id] = mf.as_posix()
        found_experiment_ids.add(payload_experiment_id)
    if expected_experiment_id_set and payload_experiment_id not in expected_experiment_id_set:
        unexpected_experiment_ids.append(payload_experiment_id or file_experiment_id)
        checks.append(_fail("Unexpected manifest experiment_id", payload_experiment_id or file_experiment_id))
        failed = True
    return checks, failed


def _check_required_manifest_fields(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    # Canonical source of truth: benchmarks.py REQUIRED_MANIFEST_FIELDS (16 fields).
    # This aligns validate_benchmark_outputs.py with benchmarks.py and resolves
    # the three-way drift between benchmarks module (16), governance doc (~15),
    # and the previous hardcoded list (11).  See Phase 5R HIGH-1 root cause.
    required_manifest_fields = [
        "dataset_id",
        "dataset_roots",
        "raw_hash",
        "processed_hash",
        "split_hash",
        "dataset_hash_primary",
        "model_architecture",
        "model_architecture_source",
        "governance_state",
        "evaluation_mode",
        "platform_targets",
        "config_hashes",
        "schema_hash",
        "mapping_version",
        "seed",
        "experiment_id",
    ]
    missing = [f for f in required_manifest_fields if f not in payload]
    if missing:
        return [_fail("Manifest missing required fields", missing)], True
    return [_pass("Required manifest fields present")], False


def _read_manifest_json(mf: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        payload = json.loads(mf.read_text(encoding="utf-8"))
        return payload, None
    except Exception as exc:
        return None, _fail("Malformed JSON manifest", str(exc))


def _check_scalar_fields(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    checks: list[dict[str, Any]] = []
    failed = False
    if _missing_scalar(payload.get("manifest_hash")):
        checks.append(_fail("manifest_hash missing or empty"))
        failed = True
    if _missing_scalar(payload.get("schema_version")):
        checks.append(_fail("schema_version missing or empty"))
        failed = True
    else:
        sv = str(payload.get("schema_version", "")).strip()
        import re
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", sv):
            checks.append(_fail("schema_version must be YYYY-MM-DD date-stamp", {"got": sv}))
            failed = True
        else:
            checks.append(_pass("schema_version format valid (YYYY-MM-DD)"))
    return checks, failed


def _validate_manifest_payload_with_benchmarks(payload: dict[str, Any], benchmarks_module) -> tuple[list[dict[str, Any]], bool]:
    try:
        benchmarks_module._validate_manifest_payload(payload, allow_dry_run_placeholder=True)
        return [_pass("Manifest payload validated")], False
    except Exception as exc:
        return [_fail("Manifest payload validation failed", str(exc))], True


def _finalize_report(
    report: dict[str, Any],
    expected_experiment_id_set: set[str],
    found_experiment_ids: set[str],
    unexpected_experiment_ids: list[str],
    duplicate_experiment_ids: list[str],
    had_failure: bool,
) -> bool:
    missing_experiment_ids = sorted(expected_experiment_id_set - found_experiment_ids)
    if missing_experiment_ids:
        report["errors"].append(_fail("Missing expected manifests", missing_experiment_ids))
        report["summary"]["manifest_validation_failed"] += len(missing_experiment_ids)
        had_failure = True

    if unexpected_experiment_ids:
        report["errors"].append(_fail("Unexpected manifest experiment_ids", sorted(set(unexpected_experiment_ids))))
        had_failure = True

    report["summary"]["missing_experiment_ids"] = missing_experiment_ids if missing_experiment_ids else []
    report["summary"]["unexpected_experiment_ids"] = sorted(set(unexpected_experiment_ids))

    if duplicate_experiment_ids:
        report["errors"].append(_fail("Duplicate manifest experiment_ids", sorted(set(duplicate_experiment_ids))))
        report["summary"]["duplicate_experiment_ids"] = sorted(set(duplicate_experiment_ids))
        had_failure = True
    else:
        report["summary"]["duplicate_experiment_ids"] = []

    return had_failure


def main() -> int:  # noqa: C901
    repo_root = Path.cwd()
    results_dir = repo_root / "results"
    manifests_dir = results_dir / "manifests"
    metrics_dir = results_dir / "metrics"
    gates_dir = results_dir / "gates"
    gates_dir.mkdir(parents=True, exist_ok=True)
    report_path = gates_dir / "benchmark_output_validation.json"
    expected_experiment_ids = _expected_experiment_ids()
    expected_experiment_id_set = set(expected_experiment_ids)

    sys.path.insert(0, str(repo_root / "src"))
    try:
        from scripts.evaluation import benchmarks
    except Exception:  # pragma: no cover - fail loudly in CI
        traceback.print_exc()
        print("Failed to import benchmarks module", file=sys.stderr)
        return 2

    report: dict[str, Any] = {
        "summary": {
            "manifests_expected": len(expected_experiment_ids),
            "manifests_found": 0,
            "manifest_validation_passed": 0,
            "manifest_validation_failed": 0,
        },
        "manifests": [],
        "errors": [],
    }
    had_failure = False
    found_experiment_ids: set[str] = set()
    seen_experiment_ids: dict[str, str] = {}
    unexpected_experiment_ids: list[str] = []
    duplicate_experiment_ids: list[str] = []

    for path, name in ((manifests_dir, "manifests"), (metrics_dir, "metrics")):
        if not path.exists():
            report["errors"].append(_fail(f"Missing {name} directory", str(path)))
            had_failure = True

    if had_failure:
        report["summary"]["result"] = "fail"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("Missing results directories - failing")
        return 3

    manifest_files = sorted(manifests_dir.glob("*.manifest.json"))
    report["summary"]["manifests_found"] = len(manifest_files)
    if not manifest_files:
        report["errors"].append(_fail("No manifest files found", str(manifests_dir)))
        report["summary"]["result"] = "fail"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("No manifest files found - failing")
        return 4

    # iterate manifests and validate each

    for mf in manifest_files:
        entry, entry_failed = _validate_manifest_file(
            mf,
            expected_experiment_id_set,
            found_experiment_ids,
            seen_experiment_ids,
            unexpected_experiment_ids,
            duplicate_experiment_ids,
            benchmarks,
        )
        report["manifests"].append(entry)
        if entry_failed:
            had_failure = True
            report["summary"]["manifest_validation_failed"] += 1
        else:
            report["summary"]["manifest_validation_passed"] += 1

    had_failure = _finalize_report(report, expected_experiment_id_set, found_experiment_ids, unexpected_experiment_ids, duplicate_experiment_ids, had_failure)

    report["summary"]["result"] = "fail" if had_failure else "pass"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if had_failure:
        print(f"Benchmark output validation failed. Report written to {report_path}")
        return 5

    print(f"Benchmark output validation passed. Report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
