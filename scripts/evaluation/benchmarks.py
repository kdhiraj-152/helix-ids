#!/usr/bin/env python3
"""Governed benchmark orchestrator for HELIX-IDS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import chain
from pathlib import Path
from typing import Any, cast

import yaml  # type: ignore[import-untyped]
from jsonschema import Draft202012Validator

from helix_ids.contracts import SCHEMA_HASH, SCHEMA_VERSION, runtime_contract_payload
from helix_ids.data.feature_harmonization import (
    create_cicids_mapping,
    create_nslkdd_mapping,
    create_unsw_mapping,
)
from helix_ids.governance import (
    build_dataset_manifest_hash,
    build_run_fingerprint,
    canonical_json_hash,
    set_global_determinism,
)
from helix_ids.governance.ast_validator import ASTValidator
from helix_ids.governance.lifecycle_verifier import run_lifecycle_verification
from helix_ids.governance.parameters import DEFAULT_GOVERNANCE_POLICY
from helix_ids.governance.provenance import (
    ArtifactManifestError,
    artifact_manifest_path,
    manifest_from_json,
    read_embedded_manifest,
    verify_artifact_manifest,
    verify_artifact_provenance,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_SCHEMA_VERSION = "2026-06-02"
RESULT_SCHEMA_VERSION = "2026-06-02"
RESULT_SCHEMA_PATH = PROJECT_ROOT / "scripts" / "evaluation" / "benchmark_result.schema.json"
DRY_RUN_MODEL_ARCHITECTURE = "dry-run-placeholder"
DRY_RUN_MODEL_ARCHITECTURE_SOURCE = "dry_run_placeholder"
DEFAULT_EXPERIMENTS_DIR = PROJECT_ROOT / "config" / "experiments"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_MANIFESTS_DIR = DEFAULT_RESULTS_DIR / "manifests"
DEFAULT_METRICS_DIR = DEFAULT_RESULTS_DIR / "metrics"

RUNNER_SPECS: dict[str, dict[str, Any]] = {
    "benchmark_e2e_v2_fixed": {
        "script": PROJECT_ROOT / "scripts" / "evaluation" / "benchmark_e2e_v2_fixed.py",
        "results_path": PROJECT_ROOT / "results" / "v2_fixed" / "e2e_benchmark_v2.json",
        "artifact_root": PROJECT_ROOT / "models" / "v2_fixed",
        "artifact_file": "model_v2.pt",
        "artifact_kind": "checkpoint",
        "platforms": ["production", "rpi4", "rpi_zero", "esp32"],
    },
    "holdout_evaluation_v2": {
        "script": PROJECT_ROOT / "scripts" / "evaluation" / "holdout_evaluation_v2.py",
        "results_path": PROJECT_ROOT / "results" / "v2_fixed" / "holdout_evaluation_v2.json",
        "artifact_root": None,
        "artifact_file": None,
        "artifact_kind": None,
        "platforms": [],
    },
    "tamper_future": {
        "script": None,
        "results_path": None,
        "artifact_root": None,
        "artifact_file": None,
        "artifact_kind": None,
        "platforms": [],
        "future": True,
    },
}

REQUIRED_MANIFEST_FIELDS = {
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
}

REQUIRED_CONFIG_HASH_FIELDS = {
    "helix_config",
    "training_config",
    "platform_configs",
    "attack_params",
    "schema_contract_hash",
    "mapping_version",
    "mapping_hash",
    "governance_policy_hash",
}

REQUIRED_DATASET_HASH_FIELDS = {"raw", "processed", "split", "primary"}

REQUIRED_RESULT_FIELDS = {
    "schema_version",
    "experiment_id",
    "lineage",
    "governance_metadata",
    "lifecycle_metadata",
    "reproducibility_metadata",
    "metrics",
    "runtime_metrics",
    "platform_metrics",
    "artifact_metadata",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be a mapping: {path}")
    return payload


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _prepare_hash_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(k): _prepare_hash_payload(v) for k, v in sorted(payload.items(), key=lambda item: str(item[0]))}
    if isinstance(payload, list):
        return [_prepare_hash_payload(v) for v in payload]
    if isinstance(payload, Path):
        return payload.as_posix()
    return payload


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _hash_file_entries(files: list[Path], *, label: str) -> str:
    if not files:
        raise FileNotFoundError(f"{label} hash requires at least one file")
    return str(build_dataset_manifest_hash(files))


def _canonical_hash(payload: dict[str, Any]) -> str:
    prepared = _prepare_hash_payload(payload)
    if not isinstance(prepared, dict):
        raise ValueError("Canonical hash payload must be a mapping")
    return str(canonical_json_hash(prepared))


def _resolve_git_context() -> dict[str, str]:
    commit = os.getenv("GITHUB_SHA") or os.getenv("CI_COMMIT_SHA") or os.getenv("GIT_COMMIT")
    branch = os.getenv("GITHUB_REF_NAME") or os.getenv("CI_COMMIT_REF_NAME") or os.getenv("GIT_BRANCH")

    def _run_git(args: list[str]) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    if not commit:
        commit = _run_git(["rev-parse", "HEAD"]) or "unknown"
    if not branch:
        branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"

    return {"commit": str(commit), "branch": str(branch)}


def _runtime_environment() -> dict[str, Any]:
    try:
        import numpy as np
        import sklearn
        import torch
    except Exception:
        np = None  # type: ignore[assignment]
        torch = None  # type: ignore[assignment]
        sklearn = None  # type: ignore[assignment]

    return {
        "python_version": sys.version.split(" ")[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "numpy_version": getattr(np, "__version__", "unknown"),
        "sklearn_version": getattr(sklearn, "__version__", "unknown"),
    }


def _resolve_files(file_entries: Iterable[str] | None) -> list[Path]:
    files: list[Path] = []
    if not file_entries:
        return files
    for entry in file_entries:
        path = (PROJECT_ROOT / entry).resolve()
        if path.is_dir():
            files.extend(sorted(path.rglob("*")))
        else:
            files.append(path)
    return [path for path in files if path.is_file()]


def _dataset_hashes_from_manifest(manifest_path: str) -> tuple[dict[str, str], dict[str, Any]]:
    path = PROJECT_ROOT / manifest_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset manifest must be a mapping: {path}")
    dataset_hash = str(payload.get("dataset_hash", "unknown"))
    return {"processed": dataset_hash, "raw": dataset_hash, "split": dataset_hash, "primary": dataset_hash}, {
        "strategy": "manifest",
        "manifest_path": path.as_posix(),
        "dataset_hash": dataset_hash,
    }


def _dataset_hashes_from_groups(groups: list[dict[str, Any]], *, label: str) -> tuple[str, dict[str, Any]]:
    group_hashes: dict[str, str] = {}
    details: dict[str, Any] = {"strategy": "grouped", "groups": {}}
    for group in groups:
        name = str(group.get("name", "group"))
        files = _resolve_files(group.get("files", []))
        if not files:
            raise FileNotFoundError(f"Dataset group {name} has no files")
        group_hashes[name] = build_dataset_manifest_hash(files)
        details["groups"][name] = [p.as_posix() for p in files]
    if not group_hashes:
        raise FileNotFoundError(f"{label} hash requires at least one dataset group")
    dataset_files = [Path(path) for path in chain.from_iterable(details["groups"].values())]
    dataset_hash = build_dataset_manifest_hash(dataset_files)
    details["primary"] = dataset_hash
    return dataset_hash, details


def _dataset_hashes_from_files(file_entries: Iterable[str] | None, *, label: str) -> tuple[str, dict[str, Any]]:
    files = _resolve_files(file_entries)
    dataset_hash = _hash_file_entries(files, label=label)
    return dataset_hash, {
        "strategy": "files",
        "files": [p.as_posix() for p in files],
        "primary": dataset_hash,
    }


def _resolve_dataset_hashes(dataset_spec: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:  # noqa: C901
    if not dataset_spec:
        raise ValueError("Dataset specification is required")

    dataset_id = str(dataset_spec.get("dataset_id", "")).strip()
    if not dataset_id:
        raise ValueError("dataset_id is required for dataset specification")

    dataset_roots = dataset_spec.get("dataset_roots")
    if not isinstance(dataset_roots, list) or not dataset_roots:
        raise ValueError("dataset_roots must be a non-empty list")

    if isinstance(dataset_spec.get("hashes"), dict):
        hashes = {str(k): str(v) for k, v in dataset_spec["hashes"].items()}
        missing = REQUIRED_DATASET_HASH_FIELDS - set(hashes)
        if missing:
            raise ValueError(f"Dataset hashes missing required fields: {sorted(missing)}")
        if str(hashes["primary"]) != str(hashes["processed"]):
            raise ValueError("Dataset hashes primary must match processed hash")
        return hashes, {"strategy": "provided", "hashes": hashes}

    manifest_path = dataset_spec.get("manifest_path")
    if manifest_path:
        return _dataset_hashes_from_manifest(str(manifest_path))

    # group/file keys will be read dynamically below

    details: dict[str, Any] = {"strategy": "files"}
    hashes_map: dict[str, str | None] = {"raw": None, "processed": None, "split": None}

    for key, groups_key, files_key in (
        ("raw", "raw_groups", "raw_files"),
        ("processed", "processed_groups", "processed_files"),
        ("split", "split_groups", "split_files"),
    ):
        groups = dataset_spec.get(groups_key)
        if isinstance(groups, list):
            h, details[key] = _dataset_hashes_from_groups(groups, label=key)
        else:
            h, details[key] = _dataset_hashes_from_files(dataset_spec.get(files_key, []), label=key)
        hashes_map[key] = h

    hashes = {k: str(hashes_map[k]) for k in ("raw", "processed", "split")}
    hashes["primary"] = str(hashes_map["processed"])
    return hashes, details


def _mapping_bundle_hash() -> str:
    mappings = {
        "nsl_kdd": create_nslkdd_mapping().to_dict(),
        "unsw_nb15": create_unsw_mapping().to_dict(),
        "cicids2018": create_cicids_mapping().to_dict(),
    }
    return str(canonical_json_hash(mappings))


def _governance_policy_hash() -> str:
    policy_payload = asdict(DEFAULT_GOVERNANCE_POLICY)
    return str(canonical_json_hash(policy_payload))


def _resolve_config_hashes(*, mapping_version: str, mapping_hash: str, run_config_hash: str) -> dict[str, str]:
    config_paths = {
        "helix_config": PROJECT_ROOT / "config" / "helix_config.yaml",
        "training_config": PROJECT_ROOT / "config" / "training.yaml",
        "platform_configs": PROJECT_ROOT / "config" / "platform_configs.yaml",
        "attack_params": PROJECT_ROOT / "config" / "attack_params.yaml",
    }
    hashes = {name: _file_sha256(path) for name, path in config_paths.items()}
    hashes.update(
        {
            "schema_contract_hash": str(SCHEMA_HASH),
            "mapping_version": str(mapping_version),
            "mapping_hash": str(mapping_hash),
            "governance_policy_hash": _governance_policy_hash(),
            "run_config_hash": str(run_config_hash),
        }
    )
    return hashes


def _load_artifact_manifest(path: Path, *, kind: str) -> dict[str, Any] | None:
    embedded = read_embedded_manifest(path, kind=kind)
    if embedded is not None:
        return cast(dict[str, Any], embedded)
    sidecar_path = artifact_manifest_path(path)
    if sidecar_path.exists():
        return cast(dict[str, Any], manifest_from_json(sidecar_path.read_text(encoding="utf-8")))
    return None


def _find_architecture_in_artifacts(paths: list[Path], kind: str | None) -> str | None:
    if not kind:
        return None
    resolved: str | None = None
    for path in paths:
        if not path.exists():
            continue
        manifest = _load_artifact_manifest(path, kind=kind)
        if manifest is None:
            continue
        architecture = manifest.get("model_architecture")
        if not architecture:
            raise RuntimeError(f"Artifact manifest missing model_architecture for {path}")
        if resolved is None:
            resolved = str(architecture)
        elif resolved != str(architecture):
            raise RuntimeError("Model architecture mismatch across artifacts")
    return resolved


def _resolve_model_architecture(  # noqa: C901
    artifact_paths: list[Path],
    *,
    kind: str | None,
    fallback: str | None,
    allow_missing: bool,
) -> tuple[str, str]:
    resolved = _find_architecture_in_artifacts(artifact_paths, kind)
    if resolved:
        return resolved, "artifact_manifest"
    if fallback:
        return str(fallback), "config"
    if allow_missing:
        return DRY_RUN_MODEL_ARCHITECTURE, DRY_RUN_MODEL_ARCHITECTURE_SOURCE
    raise RuntimeError("Unable to resolve model_architecture from artifact manifest")


def _validate_required_fields(payload: dict[str, Any], required: set[str], *, context: str) -> None:
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{context} missing required fields: {sorted(missing)}")


def _validate_hash_value(value: Any, *, label: str) -> None:
    if value is None:
        raise ValueError(f"{label} missing")
    value_str = str(value).strip()
    if not value_str or value_str == "unknown":
        raise ValueError(f"{label} missing or unknown")


@lru_cache(maxsize=1)
def _load_result_schema() -> dict[str, Any]:
    if not RESULT_SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Missing benchmark result schema: {RESULT_SCHEMA_PATH}")
    schema = json.loads(RESULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    if not isinstance(schema, dict):
        raise ValueError(f"Benchmark result schema must be a mapping: {RESULT_SCHEMA_PATH}")

    return schema


@lru_cache(maxsize=1)
def _result_schema_validator() -> Draft202012Validator:
    schema = _load_result_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _validate_manifest_payload(manifest: dict[str, Any], *, allow_dry_run_placeholder: bool = False) -> None:
    _validate_required_fields(manifest, REQUIRED_MANIFEST_FIELDS, context="Resolved manifest")
    model_architecture = str(manifest.get("model_architecture", "")).strip()
    model_architecture_source = str(manifest.get("model_architecture_source", "")).strip()
    if not model_architecture:
        raise ValueError("model_architecture missing")
    if model_architecture == DRY_RUN_MODEL_ARCHITECTURE:
        if not allow_dry_run_placeholder:
            raise ValueError("model_architecture placeholder only allowed for dry-run manifests")
        if model_architecture_source != DRY_RUN_MODEL_ARCHITECTURE_SOURCE:
            raise ValueError("Dry-run model_architecture_source must be explicit")
    elif model_architecture_source not in {"artifact_manifest", "config"}:
        raise ValueError("model_architecture_source missing or invalid")
    dataset_hashes = manifest.get("dataset_hashes")
    if not isinstance(dataset_hashes, dict):
        raise ValueError("Resolved manifest dataset_hashes must be a mapping")
    missing_hashes = REQUIRED_DATASET_HASH_FIELDS - set(dataset_hashes)
    if missing_hashes:
        raise ValueError(f"Resolved manifest dataset_hashes missing: {sorted(missing_hashes)}")
    for key in REQUIRED_DATASET_HASH_FIELDS:
        _validate_hash_value(dataset_hashes.get(key), label=f"dataset_hashes.{key}")
    if str(dataset_hashes.get("primary")) != str(manifest.get("processed_hash")):
        raise ValueError("dataset_hashes.primary must match processed_hash")

    config_hashes = manifest.get("config_hashes")
    if not isinstance(config_hashes, dict):
        raise ValueError("Resolved manifest config_hashes must be a mapping")
    missing_configs = REQUIRED_CONFIG_HASH_FIELDS - set(config_hashes)
    if missing_configs:
        raise ValueError(f"Resolved manifest config_hashes missing: {sorted(missing_configs)}")
    for key in REQUIRED_CONFIG_HASH_FIELDS:
        _validate_hash_value(config_hashes.get(key), label=f"config_hashes.{key}")

    _validate_hash_value(manifest.get("schema_hash"), label="schema_hash")
    _validate_hash_value(manifest.get("mapping_version"), label="mapping_version")
    _validate_hash_value(manifest.get("processed_hash"), label="processed_hash")
    _validate_hash_value(manifest.get("raw_hash"), label="raw_hash")
    _validate_hash_value(manifest.get("split_hash"), label="split_hash")
    _validate_hash_value(manifest.get("dataset_hash_primary"), label="dataset_hash_primary")


def _validate_result_schema(payload: dict[str, Any]) -> None:
    errors = sorted(_result_schema_validator().iter_errors(payload), key=lambda error: list(error.path))
    if errors:
        error = errors[0]
        path = "/".join(str(part) for part in error.path) or "<root>"
        raise ValueError(f"Result schema validation failed at {path}: {error.message}")
    _validate_required_fields(payload, REQUIRED_RESULT_FIELDS, context="Result schema")
    if str(payload.get("schema_version")) != RESULT_SCHEMA_VERSION:
        raise ValueError("Result schema_version mismatch")
    lineage = payload.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("Result lineage must be a mapping")
    for key in (
        "dataset_hashes",
        "config_hashes",
        "schema_hash",
        "mapping_version",
        "mapping_hash",
        "dataset_hash_primary",
        "model_architecture_source",
    ):
        if key not in lineage:
            raise ValueError(f"Result lineage missing {key}")
    if not isinstance(lineage.get("dataset_hashes"), dict):
        raise ValueError("Result lineage dataset_hashes must be a mapping")
    if not isinstance(lineage.get("config_hashes"), dict):
        raise ValueError("Result lineage config_hashes must be a mapping")
    if str(lineage.get("dataset_hashes", {}).get("primary")) != str(lineage.get("dataset_hash_primary")):
        raise ValueError("Result lineage primary dataset hash mismatch")


def _default_governance() -> dict[str, Any]:
    return {
        "ast_enforced": True,
        "lifecycle": "auto",
        "replay_protection": True,
        "manifest_mode": "full",
        "bypass": False,
        "allow_legacy_artifacts": False,
    }


def _preset_governance_variants() -> list[dict[str, Any]]:
    return [
        {"id": "ast_on", "overrides": {"governance": {"ast_enforced": True}}},
        {"id": "ast_off", "overrides": {"governance": {"ast_enforced": False}}},
        {"id": "lifecycle_on", "overrides": {"governance": {"lifecycle": "required"}}},
        {"id": "lifecycle_off", "overrides": {"governance": {"lifecycle": "off"}}},
        {"id": "replay_on", "overrides": {"governance": {"replay_protection": True}}},
        {"id": "replay_off", "overrides": {"governance": {"replay_protection": False}}},
        {"id": "manifest_embedded", "overrides": {"governance": {"manifest_mode": "embedded"}}},
        {"id": "manifest_sidecar", "overrides": {"governance": {"manifest_mode": "sidecar"}}},
        {"id": "manifest_full", "overrides": {"governance": {"manifest_mode": "full"}}},
        {
            "id": "governance_bypass",
            "overrides": {
                "governance": {
                    "bypass": True,
                    "ast_enforced": False,
                    "lifecycle": "off",
                    "replay_protection": False,
                    "manifest_mode": "bypass",
                    "allow_legacy_artifacts": True,
                }
            },
        },
    ]


def _variants_from_list(variants: list[Any]) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for variant in variants:
        if not isinstance(variant, dict) or "id" not in variant:
            continue
        overrides = variant.get("overrides", {}) if isinstance(variant.get("overrides"), dict) else {}
        resolved.append({"id": str(variant["id"]), "overrides": overrides})
    return resolved


def _variants_from_axes(axes: dict[str, Any]) -> list[dict[str, Any]]:
    variants = [{"id": "base", "overrides": {}}]
    for axis, values in sorted(axes.items(), key=lambda item: str(item[0])):
        normalized_values = list(values) if isinstance(values, list) else [values]
        next_variants: list[dict[str, Any]] = []
        for variant in variants:
            for value in normalized_values:
                override = {"governance": {axis: value}}
                base_overrides = variant.get("overrides")
                if not isinstance(base_overrides, dict):
                    base_overrides = {}
                next_variants.append(
                    {
                        "id": f"{variant['id']}-{axis}-{str(value).lower()}",
                        "overrides": _deep_merge(base_overrides, override),
                    }
                )
        variants = next_variants
    return variants


def _governance_variants(ablations: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not ablations:
        return [{"id": "base", "overrides": {}}]

    preset = str(ablations.get("preset", "")).strip().lower()
    if preset == "governance_first":
        return _preset_governance_variants()

    if isinstance(ablations.get("variants"), list):
        variants = _variants_from_list(ablations["variants"])
        return variants or [{"id": "base", "overrides": {}}]

    axes = ablations.get("axes") if isinstance(ablations.get("axes"), dict) else {}
    if not axes:
        return [{"id": "base", "overrides": {}}]

    return _variants_from_axes(axes)


@dataclass(frozen=True)
class RunSpec:
    experiment_id: str
    benchmark_id: str
    variant_id: str
    seed: int
    entrypoint: str
    config: dict[str, Any]
    dataset: dict[str, Any]
    governance: dict[str, Any]
    manifest_path: Path

    def run_id(self) -> str:
        return f"{self.experiment_id}-{self.benchmark_id}-{self.variant_id}-seed{self.seed}"


def _resolve_defaults(manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[int]]:
    defaults_raw = manifest.get("defaults")
    defaults = defaults_raw if isinstance(defaults_raw, dict) else {}
    governance = _deep_merge(_default_governance(), defaults.get("governance", {}))
    config = defaults.get("config", {}) if isinstance(defaults.get("config"), dict) else {}
    dataset = defaults.get("dataset", {}) if isinstance(defaults.get("dataset"), dict) else {}
    seeds = defaults.get("seeds", [42])
    seeds_list = [int(seed) for seed in (seeds if isinstance(seeds, list) else [seeds])]
    return governance, config, dataset, seeds_list


def _expand_benchmark(
    *,
    experiment_id: str,
    benchmark: dict[str, Any],
    variants: list[dict[str, Any]],
    default_governance: dict[str, Any],
    default_config: dict[str, Any],
    default_dataset: dict[str, Any],
    default_seeds: list[int],
    source_path: Path,
) -> list[RunSpec]:
    bench_id = str(benchmark.get("id") or benchmark.get("name") or benchmark.get("entrypoint"))
    entrypoint = str(benchmark.get("entrypoint", "")).strip()
    if entrypoint not in RUNNER_SPECS:
        raise ValueError(f"Unsupported entrypoint {entrypoint} in {source_path}")

    seeds = benchmark.get("seeds", default_seeds)
    seeds_list = [int(seed) for seed in (seeds if isinstance(seeds, list) else [seeds])]
    base_config = _deep_merge(default_config, benchmark.get("config", {}))
    base_dataset = _deep_merge(default_dataset, benchmark.get("dataset", {}))

    specs: list[RunSpec] = []
    for variant in variants:
        variant_id = str(variant["id"])
        merged_governance = _deep_merge(default_governance, variant.get("overrides", {}).get("governance", {}))
        merged_governance = _deep_merge(merged_governance, benchmark.get("governance", {}))
        merged_config = _deep_merge(base_config, variant.get("overrides", {}).get("config", {}))
        merged_dataset = _deep_merge(base_dataset, variant.get("overrides", {}).get("dataset", {}))
        for seed in seeds_list:
            specs.append(
                RunSpec(
                    experiment_id=experiment_id,
                    benchmark_id=bench_id,
                    variant_id=variant_id,
                    seed=int(seed),
                    entrypoint=entrypoint,
                    config=merged_config,
                    dataset=merged_dataset,
                    governance=merged_governance,
                    manifest_path=source_path,
                )
            )
    return specs


def _expand_experiment(manifest: dict[str, Any], *, source_path: Path) -> list[RunSpec]:
    experiment_id = str(manifest.get("experiment_id") or manifest.get("id") or source_path.stem)
    default_governance, default_config, default_dataset, default_seeds = _resolve_defaults(manifest)

    benchmarks = manifest.get("benchmarks") or manifest.get("runs")
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ValueError(f"Manifest {source_path} must define benchmarks")

    variants = _governance_variants(manifest.get("ablations"))
    specs: list[RunSpec] = []
    for bench in benchmarks:
        if not isinstance(bench, dict):
            continue
        specs.extend(
            _expand_benchmark(
                experiment_id=experiment_id,
                benchmark=bench,
                variants=variants,
                default_governance=default_governance,
                default_config=default_config,
                default_dataset=default_dataset,
                default_seeds=default_seeds,
                source_path=source_path,
            )
        )
    return specs


def _ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _resolve_artifact_paths(run: RunSpec) -> list[Path]:
    spec = RUNNER_SPECS[run.entrypoint]
    artifact_root = spec.get("artifact_root")
    artifact_file = spec.get("artifact_file")
    platforms = run.config.get("platforms") or spec.get("platforms", [])
    if not artifact_root or not artifact_file:
        return []

    paths: list[Path] = []
    for platform_name in platforms:
        paths.append(Path(artifact_root) / str(platform_name) / str(artifact_file))
    return paths


def _validate_ast(paths: list[Path]) -> dict[str, Any]:
    validator = ASTValidator()
    resolved_paths = [p if p.is_absolute() else (PROJECT_ROOT / p) for p in paths]
    violations = validator.validate_paths(resolved_paths)
    return {
        "status": "pass" if not violations else "fail",
        "violations": [v.as_dict() for v in violations],
    }


def _validate_lifecycle(workdir: Path, *, mode: str) -> dict[str, Any]:
    mode = mode.lower()
    if mode == "off":
        return {"status": "skipped", "reason": "disabled"}

    try:
        require_onnx = mode == "required"
        result = run_lifecycle_verification(workdir, require_onnx=require_onnx)
        return {"status": "pass", "result": _prepare_hash_payload(result)}
    except ImportError as exc:
        if mode == "required":
            raise
        return {"status": "skipped", "reason": f"onnx_missing: {exc}"}
    except Exception as exc:
        if mode == "required":
            raise
        return {"status": "fail", "reason": str(exc)}


def _verify_artifacts(
    artifact_paths: list[Path],
    *,
    kind: str,
    manifest_mode: str,
) -> dict[str, Any]:
    manifest_mode = manifest_mode.lower()
    if manifest_mode == "bypass":
        return {"status": "skipped", "reason": "bypass"}

    runtime_contract = runtime_contract_payload()
    results: list[dict[str, Any]] = []
    for path in artifact_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact: {path}")
        try:
            if manifest_mode == "embedded":
                embedded = read_embedded_manifest(path, kind=kind)
                verify_artifact_manifest(
                    path,
                    kind=kind,
                    contract=runtime_contract,
                    embedded_manifest=embedded,
                    require_embedded_manifest=True,
                )
            elif manifest_mode == "sidecar":
                verify_artifact_manifest(
                    path,
                    kind=kind,
                    contract=runtime_contract,
                    embedded_manifest=None,
                    require_embedded_manifest=False,
                )
            else:
                sidecars = {
                    "contract": path.with_suffix(path.suffix + ".contract.json"),
                    "feature_order": path.with_suffix(path.suffix + ".feature_order.json"),
                    "schema_hash": path.with_suffix(path.suffix + ".schema_hash.txt"),
                }
                deploy_path = path.parent / "deployment.manifest.json"
                deployment_manifest = deploy_path if deploy_path.exists() else None
                verify_artifact_provenance(
                    path,
                    kind=kind,
                    contract=runtime_contract,
                    embedded_manifest=None,
                    require_embedded_manifest=True,
                    sidecars=sidecars,
                    require_chain=True,
                    deployment_manifest=deployment_manifest,
                )
            results.append({"artifact": path.as_posix(), "status": "pass"})
        except ArtifactManifestError as exc:
            results.append({"artifact": path.as_posix(), "status": "fail", "reason": str(exc)})
    status = "pass" if all(item["status"] == "pass" for item in results) else "fail"
    return {"status": status, "artifacts": results}


def _metrics_from_benchmark_e2e(raw: dict[str, Any]) -> dict[str, Any]:
    datasets: list[dict[str, Any]] = []
    latency: dict[str, Any] = {}
    for platform_name, payload in sorted(raw.items()):
        if not isinstance(payload, dict):
            continue
        for key in ("nsl_kdd", "unsw_nb15"):
            metrics = payload.get(key)
            if not isinstance(metrics, dict):
                continue
            datasets.append(
                {
                    "dataset_id": f"{platform_name}/{key}",
                    "accuracy": metrics.get("accuracy"),
                    "macro_f1": metrics.get("macro_f1"),
                    "weighted_f1": metrics.get("weighted_f1"),
                    "ci95_lower": metrics.get("ci95_lower"),
                    "ci95_upper": metrics.get("ci95_upper"),
                    "ci95_width": metrics.get("ci95_width"),
                }
            )
        if isinstance(payload.get("latency"), dict):
            latency[platform_name] = payload["latency"]
    return {"datasets": datasets, "latency": latency}


def _metrics_from_holdout(raw: dict[str, Any]) -> dict[str, Any]:
    cv = raw.get("cross_validation", {}) if isinstance(raw, dict) else {}
    folds = cv.get("folds", []) if isinstance(cv.get("folds"), list) else []
    datasets: list[dict[str, Any]] = []
    for fold in folds:
        if not isinstance(fold, dict):
            continue
        ci_lower = fold.get("ci95_lower")
        ci_upper = fold.get("ci95_upper")
        ci_width = None
        if isinstance(ci_lower, (float, int)) and isinstance(ci_upper, (float, int)):
            ci_width = ci_upper - ci_lower
        datasets.append(
            {
                "dataset_id": f"cv_fold_{fold.get('fold')}",
                "accuracy": fold.get("accuracy"),
                "macro_f1": fold.get("f1_macro"),
                "ci95_lower": ci_lower,
                "ci95_upper": ci_upper,
                "ci95_width": ci_width,
            }
        )
    summary = {
        "mean_accuracy": cv.get("mean_accuracy"),
        "mean_f1_macro": cv.get("mean_f1_macro"),
        "std_f1_macro": cv.get("std_f1_macro"),
    }
    return {"datasets": datasets, "summary": summary}


def _load_metrics(entrypoint: str, results_path: Path) -> dict[str, Any]:
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results: {results_path}")
    raw = json.loads(results_path.read_text(encoding="utf-8"))
    if entrypoint == "benchmark_e2e_v2_fixed" and isinstance(raw, dict):
        return _metrics_from_benchmark_e2e(raw)
    if entrypoint == "holdout_evaluation_v2" and isinstance(raw, dict):
        return _metrics_from_holdout(raw)
    return {"datasets": []}


def _write_metrics_exports(metrics: dict[str, Any], *, run_id: str, output_dir: Path) -> dict[str, str]:
    _ensure_dirs(output_dir)
    json_path = output_dir / f"{run_id}.metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    csv_path = output_dir / f"{run_id}.metrics.csv"
    rows = [
        "dataset_id,accuracy,macro_f1,weighted_f1,ci95_lower,ci95_upper,ci95_width",
    ]
    for dataset in metrics.get("datasets", []):
        rows.append(
            ",".join(
                str(dataset.get(key, ""))
                for key in (
                    "dataset_id",
                    "accuracy",
                    "macro_f1",
                    "weighted_f1",
                    "ci95_lower",
                    "ci95_upper",
                    "ci95_width",
                )
            )
        )
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    tex_path = output_dir / f"{run_id}.metrics.tex"
    tex_lines = [
        "\\begin{tabular}{lrrrrrr}",
        "Dataset & Acc & Macro F1 & W-F1 & CI95 L & CI95 U & CI95 W \\\\",
        "\\hline",
    ]
    for dataset in metrics.get("datasets", []):
        tex_lines.append(
            "{} & {} & {} & {} & {} & {} & {} \\\\".format(
                dataset.get("dataset_id", ""),
                dataset.get("accuracy", ""),
                dataset.get("macro_f1", ""),
                dataset.get("weighted_f1", ""),
                dataset.get("ci95_lower", ""),
                dataset.get("ci95_upper", ""),
                dataset.get("ci95_width", ""),
            )
        )
    tex_lines.append("\\end{tabular}")
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    return {
        "json": json_path.as_posix(),
        "csv": csv_path.as_posix(),
        "latex": tex_path.as_posix(),
    }


def _build_manifest_base(run: RunSpec, governance: dict[str, Any], *, manifest_fields: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": run.experiment_id,
        **manifest_fields,
        "schema_hash": str(SCHEMA_HASH),
        "benchmark_id": run.benchmark_id,
        "variant_id": run.variant_id,
        "seed": run.seed,
        "entrypoint": run.entrypoint,
        "manifest_source": run.manifest_path.as_posix(),
        "config": run.config,
        "dataset": run.dataset,
        "governance": governance,
    }


def _build_run_context(run: RunSpec, *, dry_run: bool) -> dict[str, Any]:
    git_context = _resolve_git_context()
    dataset_hashes, dataset_details = _resolve_dataset_hashes(run.dataset)
    config_hash = _canonical_hash(run.config) if run.config else "unknown"
    governance = _deep_merge(_default_governance(), run.governance)
    mapping_version = str(run.config.get("mapping_version", SCHEMA_VERSION))
    mapping_hash = _mapping_bundle_hash()
    config_hashes = _resolve_config_hashes(
        mapping_version=mapping_version,
        mapping_hash=mapping_hash,
        run_config_hash=config_hash,
    )
    dataset_id = str(run.dataset.get("dataset_id", ""))
    dataset_roots = [str(root) for root in run.dataset.get("dataset_roots", [])]
    if not dataset_roots:
        raise ValueError("dataset_roots must be provided for benchmark manifests")
    raw_hash = str(dataset_hashes.get("raw", "unknown"))
    processed_hash = str(dataset_hashes.get("processed", "unknown"))
    split_hash = str(dataset_hashes.get("split", "unknown"))
    dataset_hash_primary = str(dataset_hashes.get("primary", processed_hash))
    spec = RUNNER_SPECS[run.entrypoint]
    artifact_paths = _resolve_artifact_paths(run)
    model_architecture, model_architecture_source = _resolve_model_architecture(
        artifact_paths,
        kind=spec.get("artifact_kind"),
        fallback=run.config.get("model_architecture"),
        allow_missing=dry_run,
    )
    evaluation_mode = str(run.config.get("evaluation_mode", run.entrypoint))
    platform_targets = run.config.get("platform_targets") or spec.get("platforms", [])
    governance_state = {
        "ast_enforced": bool(governance.get("ast_enforced", True)),
        "lifecycle": str(governance.get("lifecycle", "auto")),
        "replay_protection": bool(governance.get("replay_protection", True)),
        "manifest_mode": str(governance.get("manifest_mode", "full")),
        "bypass": bool(governance.get("bypass", False)),
        "allow_legacy_artifacts": bool(governance.get("allow_legacy_artifacts", False)),
        "policy_hash": config_hashes.get("governance_policy_hash"),
    }
    fingerprint = build_run_fingerprint(
        dataset_hashes=dataset_hashes,
        mapping_version=mapping_version,
        schema_hash=str(SCHEMA_HASH),
        model_config_hash=config_hash,
        commit_sha=git_context["commit"],
    )
    manifest_base = _build_manifest_base(
        run,
        governance,
        manifest_fields={
            "dataset_id": dataset_id,
            "dataset_roots": dataset_roots,
            "raw_hash": raw_hash,
            "processed_hash": processed_hash,
            "split_hash": split_hash,
            "dataset_hash_primary": dataset_hash_primary,
            "model_architecture": model_architecture,
            "model_architecture_source": model_architecture_source,
            "governance_state": governance_state,
            "evaluation_mode": evaluation_mode,
            "platform_targets": [str(p) for p in platform_targets],
            "config_hashes": config_hashes,
            "mapping_version": str(mapping_version),
            "mapping_hash": str(mapping_hash),
        },
    )
    manifest_hash = _canonical_hash(manifest_base)
    return {
        "git_context": git_context,
        "dataset_hashes": dataset_hashes,
        "dataset_details": dataset_details,
        "dataset_hash_primary": dataset_hash_primary,
        "config_hash": config_hash,
        "config_hashes": config_hashes,
        "governance": governance,
        "governance_state": governance_state,
        "fingerprint": fingerprint,
        "manifest_base": manifest_base,
        "manifest_hash": manifest_hash,
        "run_id": run.run_id(),
        "mapping_version": mapping_version,
        "mapping_hash": mapping_hash,
        "model_architecture": model_architecture,
        "model_architecture_source": model_architecture_source,
        "evaluation_mode": evaluation_mode,
        "platform_targets": [str(p) for p in platform_targets],
    }


def _run_prechecks(  # noqa: C901
    run: RunSpec,
    *,
    governance: dict[str, Any],
    manifests_dir: Path,
    dry_run: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], float]:
    ast_status: dict[str, Any] = {"status": "skipped"}
    artifact_status: dict[str, Any] = {"status": "skipped"}

    if dry_run:
        lifecycle_status: dict[str, Any] = {"status": "skipped", "reason": "dry_run"}
        return ast_status, lifecycle_status, artifact_status, 0.0

    precheck_start = time.perf_counter()
    if governance.get("ast_enforced", True):
        ast_paths = run.governance.get("ast_paths") if isinstance(run.governance.get("ast_paths"), list) else None
        if not ast_paths:
            ast_paths = [RUNNER_SPECS[run.entrypoint]["script"], PROJECT_ROOT / "src" / "helix_ids"]
        ast_status = _validate_ast([Path(path) for path in ast_paths])
        if ast_status["status"] != "pass":
            raise RuntimeError("AST governance violations detected")

    lifecycle_mode = str(governance.get("lifecycle", "auto"))
    lifecycle_status = _validate_lifecycle(manifests_dir / "lifecycle" / run.run_id(), mode=lifecycle_mode)
    if lifecycle_mode == "required" and lifecycle_status["status"] != "pass":
        raise RuntimeError("Lifecycle verification failed")

    artifact_paths = _resolve_artifact_paths(run)
    spec = RUNNER_SPECS[run.entrypoint]
    if artifact_paths and spec.get("artifact_kind"):
        artifact_status = _verify_artifacts(
            artifact_paths,
            kind=str(spec["artifact_kind"]),
            manifest_mode=str(governance.get("manifest_mode", "full")),
        )
        if artifact_status["status"] != "pass":
            raise RuntimeError("Artifact provenance verification failed")

    precheck_elapsed = time.perf_counter() - precheck_start
    return ast_status, lifecycle_status, artifact_status, precheck_elapsed


def _build_env(run: RunSpec, context: dict[str, Any]) -> dict[str, str]:
    dataset_hashes_payload = json.dumps(context["dataset_hashes"], sort_keys=True)
    env = os.environ.copy()
    env.update(
        {
            "HELIX_SEED": str(run.seed),
            "HELIX_RUN_ID": context["run_id"],
            "HELIX_DATASET_HASHES": dataset_hashes_payload,
            "HELIX_SCHEMA_HASH": str(SCHEMA_HASH),
            "HELIX_MAPPING_VERSION": str(run.config.get("mapping_version", SCHEMA_VERSION)),
            "HELIX_FINGERPRINT": context["fingerprint"],
            "HELIX_GOV_CONTEXT": json.dumps(
                {
                    "experiment_id": run.experiment_id,
                    "benchmark_id": run.benchmark_id,
                    "variant_id": run.variant_id,
                    "manifest_hash": context["manifest_hash"],
                },
                sort_keys=True,
            ),
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
        }
    )
    if not context["governance"].get("replay_protection", True):
        env["HELIX_GOV_POLICY_PROFILE"] = "smoke"
    if context["governance"].get("allow_legacy_artifacts"):
        env["HELIX_ALLOW_LEGACY_ARTIFACTS"] = "1"
        env["HELIX_ALLOW_LEGACY_MANIFEST"] = "1"
    return env


def _dispatch_run(
    run: RunSpec,
    *,
    env: dict[str, str],
    manifests_dir: Path,
    metrics_dir: Path,
    dry_run: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    spec = RUNNER_SPECS[run.entrypoint]
    if dry_run:
        dispatch_status: dict[str, Any] = {"status": "skipped", "reason": "dry_run"}
        metrics_payload = {"datasets": [], "note": "dry_run"}
        metrics_outputs = _write_metrics_exports(metrics_payload, run_id=run.run_id(), output_dir=metrics_dir)
        return dispatch_status, metrics_payload, metrics_outputs

    if spec.get("future"):
        raise RuntimeError(f"Entry point {run.entrypoint} is not yet implemented")

    log_path = manifests_dir / f"{run.run_id()}.log"
    with log_path.open("w", encoding="utf-8") as handle:
        dispatch_start = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, spec["script"].as_posix()],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
        dispatch_elapsed = time.perf_counter() - dispatch_start

    dispatch_status = {
        "status": "ok" if completed.returncode == 0 else "error",
        "returncode": completed.returncode,
        "elapsed_seconds": dispatch_elapsed,
        "log_path": log_path.as_posix(),
    }
    metrics_payload = _load_metrics(run.entrypoint, Path(spec["results_path"]))
    metrics_outputs = _write_metrics_exports(metrics_payload, run_id=run.run_id(), output_dir=metrics_dir)
    return dispatch_status, metrics_payload, metrics_outputs


def _build_result_schema(
    *,
    run: RunSpec,
    context: dict[str, Any],
    metrics_payload: dict[str, Any],
    dispatch_status: dict[str, Any],
    lifecycle_status: dict[str, Any],
    ast_status: dict[str, Any],
    artifact_status: dict[str, Any],
    timing: dict[str, Any],
    outputs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment_id": run.experiment_id,
        "lineage": {
            "run_id": context["run_id"],
            "benchmark_id": run.benchmark_id,
            "variant_id": run.variant_id,
            "manifest_hash": context["manifest_hash"],
            "manifest_path": outputs.get("manifest"),
            "schema_hash": str(SCHEMA_HASH),
            "mapping_version": context["mapping_version"],
            "mapping_hash": context["mapping_hash"],
            "dataset_id": context["manifest_base"]["dataset_id"],
            "dataset_hash_primary": context["dataset_hash_primary"],
            "dataset_hashes": context["dataset_hashes"],
            "config_hashes": context["config_hashes"],
            "model_architecture": context["model_architecture"],
            "model_architecture_source": context["model_architecture_source"],
            "git_commit": context["git_context"]["commit"],
            "git_branch": context["git_context"]["branch"],
            "fingerprint": context["fingerprint"],
        },
        "governance_metadata": {
            "state": context["governance_state"],
            "status": {
                "ast": ast_status,
                "lifecycle": lifecycle_status,
                "artifact": artifact_status,
                "replay_protection": bool(context["governance"].get("replay_protection", True)),
                "manifest_mode": context["governance"].get("manifest_mode"),
                "bypass": bool(context["governance"].get("bypass")),
            },
        },
        "lifecycle_metadata": {
            "status": lifecycle_status,
        },
        "reproducibility_metadata": {
            "seed": run.seed,
            "determinism": context["determinism"],
            "run_id": context["run_id"],
        },
        "metrics": metrics_payload,
        "runtime_metrics": {
            "dispatch": dispatch_status,
            "timing": timing,
            "runtime": context["runtime"],
        },
        "platform_metrics": {
            "latency": metrics_payload.get("latency", {}),
        },
        "artifact_metadata": {
            "model_architecture": context["model_architecture"],
            "model_architecture_source": context["model_architecture_source"],
            "artifact_paths": outputs.get("artifact_paths", []),
            "outputs": outputs,
        },
    }


def _run_benchmark(run: RunSpec, *, manifests_dir: Path, metrics_dir: Path, dry_run: bool) -> dict[str, Any]:
    start = time.perf_counter()
    context = _build_run_context(run, dry_run=dry_run)
    governance = context["governance"]
    run_id = context["run_id"]

    determinism_state = set_global_determinism(run.seed)
    context["determinism"] = determinism_state.to_dict()
    context["runtime"] = _runtime_environment()

    ast_status, lifecycle_status, artifact_status, precheck_elapsed = _run_prechecks(
        run,
        governance=governance,
        manifests_dir=manifests_dir,
        dry_run=dry_run,
    )

    env = _build_env(run, context)
    dispatch_status, _, metrics_outputs = _dispatch_run(
        run,
        env=env,
        manifests_dir=manifests_dir,
        metrics_dir=metrics_dir,
        dry_run=dry_run,
    )

    total_elapsed = time.perf_counter() - start
    manifest_path = manifests_dir / f"{run.experiment_id}.manifest.json"
    output_paths = {
        "raw_results": Path(RUNNER_SPECS[run.entrypoint]["results_path"]).as_posix(),
        "manifest": manifest_path.as_posix(),
        "artifact_paths": [path.as_posix() for path in _resolve_artifact_paths(run)],
    }

    resolved_manifest = {
        **context["manifest_base"],
        "manifest_hash": context["manifest_hash"],
        "config_hash": context["config_hash"],
        "config_hashes": context["config_hashes"],
        "dataset_hashes": context["dataset_hashes"],
        "dataset_details": context["dataset_details"],
        "fingerprint": context["fingerprint"],
        "model_architecture": context["model_architecture"],
        "evaluation_mode": context["evaluation_mode"],
        "platform_targets": context["platform_targets"],
        "mapping_version": context["mapping_version"],
        "mapping_hash": context["mapping_hash"],
        "git_commit": context["git_context"]["commit"],
        "git_branch": context["git_context"]["branch"],
        "runtime": context["runtime"],
        "determinism": context["determinism"],
        "governance_status": {
            "ast": ast_status,
            "lifecycle": lifecycle_status,
            "artifact": artifact_status,
            "replay_protection": bool(governance.get("replay_protection", True)),
            "manifest_mode": governance.get("manifest_mode"),
            "bypass": bool(governance.get("bypass")),
        },
        "dispatch": dispatch_status,
        "outputs": {
            **output_paths,
            "metrics": metrics_outputs,
        },
        "timing": {
            "precheck_seconds": precheck_elapsed,
            "total_seconds": total_elapsed,
            "governance_overhead_seconds": precheck_elapsed,
            "governance_overhead_ratio": precheck_elapsed / max(total_elapsed, 1e-6),
        },
    }

    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_hash = existing.get("manifest_hash")
        if not dry_run and existing_hash and existing_hash != resolved_manifest["manifest_hash"]:
            raise RuntimeError("Manifest hash changed between runs")

    _validate_manifest_payload(resolved_manifest, allow_dry_run_placeholder=dry_run)
    manifest_path.write_text(json.dumps(resolved_manifest, indent=2), encoding="utf-8")

    result_payload = _build_result_schema(
        run=run,
        context=context,
        metrics_payload=json.loads(Path(metrics_outputs["json"]).read_text(encoding="utf-8")),
        dispatch_status=dispatch_status,
        lifecycle_status=lifecycle_status,
        ast_status=ast_status,
        artifact_status=artifact_status,
        timing=resolved_manifest["timing"],
        outputs={**output_paths, "metrics": metrics_outputs},
    )
    result_path = metrics_dir / f"{run_id}.result.json"
    result_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    _validate_result_schema(result_payload)

    for required_path in (
        metrics_outputs["json"],
        metrics_outputs["csv"],
        metrics_outputs["latex"],
        result_path.as_posix(),
    ):
        if not Path(required_path).exists():
            raise FileNotFoundError(f"Missing benchmark output: {required_path}")

    resolved_manifest["outputs"]["result_schema"] = result_path.as_posix()
    manifest_path.write_text(json.dumps(resolved_manifest, indent=2), encoding="utf-8")
    return resolved_manifest


def _load_all_manifests(paths: list[Path]) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for path in sorted(paths):
        manifest = _read_yaml(path)
        specs.extend(_expand_experiment(manifest, source_path=path))
    return specs


def _select_dry_run_specs(specs: list[RunSpec]) -> list[RunSpec]:
    """Select one deterministic run per experiment for dry-run manifest emission."""
    selected: dict[str, RunSpec] = {}
    for spec in sorted(specs, key=lambda item: item.run_id()):
        selected.setdefault(spec.experiment_id, spec)
    return [selected[key] for key in sorted(selected)]


def main() -> None:
    parser = argparse.ArgumentParser(description="HELIX-IDS governed benchmark orchestrator")
    parser.add_argument("--manifest", action="append", help="Path to a manifest YAML file")
    parser.add_argument("--experiments-dir", default=str(DEFAULT_EXPERIMENTS_DIR))
    parser.add_argument("--experiment-id", help="Filter by experiment id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true", help="List expanded runs")
    args = parser.parse_args()

    manifests: list[Path] = []
    if args.manifest:
        manifests.extend([Path(path) for path in args.manifest])
    else:
        manifest_dir = Path(args.experiments_dir)
        if not manifest_dir.exists():
            raise FileNotFoundError(f"Missing experiments directory: {manifest_dir}")
        manifests.extend(sorted(manifest_dir.glob("*.yaml")))

    if not manifests:
        raise FileNotFoundError("No experiment manifests found")

    specs = _load_all_manifests(manifests)
    if args.experiment_id:
        specs = [spec for spec in specs if spec.experiment_id == args.experiment_id]

    if args.dry_run:
        specs = _select_dry_run_specs(specs)

    if not specs:
        raise RuntimeError("No benchmark runs selected")

    if args.list:
        for spec in specs:
            print(spec.run_id())
        return

    _ensure_dirs(DEFAULT_MANIFESTS_DIR, DEFAULT_METRICS_DIR)

    for spec in specs:
        resolved = _run_benchmark(
            spec,
            manifests_dir=DEFAULT_MANIFESTS_DIR,
            metrics_dir=DEFAULT_METRICS_DIR,
            dry_run=args.dry_run,
        )
        print(f"Completed {resolved['manifest_hash']} -> {resolved['outputs']['manifest']}")


if __name__ == "__main__":
    main()
