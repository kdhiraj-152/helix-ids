"""Artifact provenance manifests and verification helpers.

Provenance verification is strict by default. Use `verify_ingress_artifact`
to enable ingress-scoped compatibility for legacy artifacts when explicitly
allowed (via the `HELIX_ALLOW_LEGACY_ARTIFACTS` env var or the
`allow_legacy_local_dev` flag).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from helix_ids import __version__ as HELIX_IDS_VERSION
from helix_ids.contracts import (
    CONTRACT_VERSION,
    EXPORTER_API_VERSION,
    FEATURE_ORDER_HASH,
    MANIFEST_VERSION,
    runtime_contract_payload,
)
from helix_ids.contracts.schema_contract import assert_runtime_contract
from helix_ids.governance.fingerprinting import canonical_json_hash
from helix_ids.governance.parameters import allow_legacy_artifacts, is_production_runtime

ARTIFACT_MANIFEST_KEY = "artifact_manifest"
ARTIFACT_MANIFEST_FILENAME = "manifest.json"
LEGACY_ARTIFACT_MANIFEST_FILENAME = "artifact_manifest.json"
DEPLOYMENT_MANIFEST_FILENAME = "deployment.manifest.json"
PROVENANCE_CHAIN_KEY = "provenance_chain"
_VOLATILE_MANIFEST_FIELDS = {
    "artifact_sha256",
    PROVENANCE_CHAIN_KEY,
    "training_timestamp",
    "torch_version",
    "git_dirty",
}
_OPTIONAL_MANIFEST_FIELDS = {
    "manifest_version",
    "dataset_hash",
    "normalized_dataset_hash",
    "config_hash",
    "training_code_hash",
    "training_config_hash",
    "export_config_hash",
    "git_commit",
    "git_branch",
    "exporter_version",
    "exporter_api_version",
    "runtime_version",
    "training_timestamp",
    "onnx_opset",
    "artifact_sha256",
    PROVENANCE_CHAIN_KEY,
    "model_architecture",
    "quantization_config_hash",
}
_UNORDERED_LIST_FIELDS = {
    "classes",
    "input_names",
    "output_names",
}


class ArtifactManifestError(RuntimeError):
    """Raised when an artifact manifest or its embedded provenance is invalid."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def artifact_sha256(path: Path | str) -> str:
    file_path = Path(path)
    return _sha256_bytes(file_path.read_bytes())


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _hash_optional_payload(payload: Mapping[str, Any] | None) -> str | None:
    if payload is None:
        return None
    # canonical_json_hash may be untyped; coerce to str for callers
    return str(canonical_json_hash(dict(payload)))


def _load_sidecar_manifest(path: Path) -> dict[str, Any] | None:
    """Load and validate sidecar manifest if it exists. Returns None when absent."""
    sidecar_path = artifact_manifest_path(path)
    legacy_path = legacy_artifact_manifest_path(path)
    if legacy_path.exists() and not _allow_legacy_manifest():
        raise ArtifactManifestError(
            f"Legacy manifest filename is not permitted: {legacy_path.name}"
        )
    if not sidecar_path.exists():
        if legacy_path.exists() and _allow_legacy_manifest():
            sidecar_path = legacy_path
        else:
            return None
    sidecar_manifest = manifest_from_json(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(sidecar_manifest, dict):
        raise ArtifactManifestError("Artifact manifest sidecar must decode to a mapping")
    return sidecar_manifest


def _environment_value(*names: str, default: str = "unknown") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _environment_value_optional(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _allow_legacy_manifest() -> bool:
    allowed = allow_legacy_artifacts() or os.getenv("HELIX_ALLOW_LEGACY_MANIFEST", "").strip() == "1"
    if allowed:
        assert not is_production_runtime(), (
            "Legacy manifest allowance is forbidden in production runtimes. "
            "Disable legacy flags or switch to a non-production env."
        )
    return allowed


def _telemetry_path() -> Path:
    return Path(os.getenv("HELIX_PROVENANCE_TELEMETRY", "results/provenance/provenance_events.jsonl"))


def _emit_provenance_telemetry(
    *,
    artifact_path: Path,
    kind: str,
    failure: Exception,
) -> None:
    payload = {
        "event": "provenance_verification_failed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_path": str(artifact_path),
        "kind": str(kind),
        "error": str(failure),
    }
    try:
        path = _telemetry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    except Exception:
        # Best-effort telemetry only.
        pass


def _manifest_feature_payload(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "feature_order": [str(feature) for feature in contract["feature_order"]],
        "input_dim": int(contract["input_dim"]),
        "binary_output_dim": int(contract["binary_output_dim"]),
        "family_output_dim": int(contract["family_output_dim"]),
        "schema_hash": str(contract["schema_hash"]),
    }


def build_artifact_manifest(
    *,
    contract: Mapping[str, Any] | None = None,
    model_architecture: str,
    dataset_hash: str | None = None,
    training_config: Mapping[str, Any] | None = None,
    export_config: Mapping[str, Any] | None = None,
    git_commit: str | None = None,
    git_branch: str | None = None,
    exporter_version: str | None = None,
    runtime_version: str | None = None,
    training_timestamp: str | None = None,
    onnx_opset: int | None = None,
    artifact_sha256_value: str | None = None,
    provenance_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical provenance manifest for an artifact."""

    contract_payload = dict(contract or runtime_contract_payload())
    feature_payload = _manifest_feature_payload(contract_payload)
    feature_order_hash = contract_payload.get("feature_order_hash") or FEATURE_ORDER_HASH
    provenance_payload = dict(provenance_fields or {})
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "schema_version": str(contract_payload["schema_version"]),
        "schema_hash": str(contract_payload["schema_hash"]),
        "contract_version": str(contract_payload.get("contract_version", CONTRACT_VERSION)),
        "feature_order_hash": str(feature_order_hash),
        "feature_hash": str(provenance_payload.pop("feature_hash", canonical_json_hash(feature_payload))),
        "dataset_hash": dataset_hash,
        "normalized_dataset_hash": provenance_payload.pop("normalized_dataset_hash", None),
        "config_hash": provenance_payload.pop("config_hash", None),
        "training_code_hash": provenance_payload.pop("training_code_hash", None),
        "training_config_hash": _hash_optional_payload(training_config),
        "export_config_hash": _hash_optional_payload(export_config),
        "git_commit": git_commit or provenance_payload.pop("git_commit", None) or _environment_value(
            "GITHUB_SHA", "CI_COMMIT_SHA", "GIT_COMMIT", default=HELIX_IDS_VERSION
        ),
        "git_branch": git_branch or _environment_value("GITHUB_REF_NAME", "CI_COMMIT_REF_NAME", "GIT_BRANCH"),
        "git_dirty": provenance_payload.pop("git_dirty", None),
        "exporter_version": exporter_version or HELIX_IDS_VERSION,
        "exporter_api_version": EXPORTER_API_VERSION,
        "runtime_version": runtime_version or HELIX_IDS_VERSION,
        "training_timestamp": training_timestamp
        or datetime.now(timezone.utc).isoformat(),
        "torch_version": str(torch.__version__),
        "onnx_opset": int(onnx_opset) if onnx_opset is not None else None,
        "quantization_config_hash": provenance_payload.pop("quantization_config_hash", None),
        "artifact_sha256": artifact_sha256_value,
        PROVENANCE_CHAIN_KEY: None,
        "model_architecture": str(model_architecture),
        "binary_output_dim": int(contract_payload["binary_output_dim"]),
        "family_output_dim": int(contract_payload["family_output_dim"]),
        "input_dim": int(contract_payload["input_dim"]),
    }
    manifest.update(provenance_payload)
    return manifest


def manifest_without_artifact_sha256(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key not in {"artifact_sha256", PROVENANCE_CHAIN_KEY}
    }


def _normalize_timestamp(value: str) -> str:
    raw = value.strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return raw
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _normalize_manifest_value(key: str, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {inner_key: _normalize_manifest_value(inner_key, inner_value) for inner_key, inner_value in value.items()}
    if isinstance(value, list):
        normalized = [_normalize_manifest_value(key, item) for item in value]
        if key in _UNORDERED_LIST_FIELDS:
            return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
        return normalized
    if isinstance(value, str) and (key.endswith("_timestamp") or key in {"timestamp", "timestamp_utc"}):
        return _normalize_timestamp(value)
    return value


def normalize_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(manifest)
    for key in _OPTIONAL_MANIFEST_FIELDS:
        normalized[key] = normalized.get(key)
    for key in _VOLATILE_MANIFEST_FIELDS:
        normalized.pop(key, None)
    for key, value in list(normalized.items()):
        normalized[key] = _normalize_manifest_value(key, value)
    for field in ("binary_output_dim", "family_output_dim", "input_dim"):
        if field in normalized and normalized[field] is not None:
            normalized[field] = int(normalized[field])
    return normalized


def canonical_manifest_hash(manifest: Mapping[str, Any]) -> str:
    return str(canonical_json_hash(normalize_manifest(manifest)))


def manifest_json(manifest: Mapping[str, Any]) -> str:
    return _canonical_json(dict(manifest))


def manifest_from_json(payload: str) -> dict[str, Any]:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ArtifactManifestError("Artifact manifest payload must decode to a mapping")
    return data


def artifact_manifest_path(artifact_path: Path | str) -> Path:
    path = Path(artifact_path)
    return path.with_suffix(path.suffix + f".{ARTIFACT_MANIFEST_FILENAME}")


def legacy_artifact_manifest_path(artifact_path: Path | str) -> Path:
    path = Path(artifact_path)
    return path.with_suffix(path.suffix + f".{LEGACY_ARTIFACT_MANIFEST_FILENAME}")


def deployment_manifest_path(artifact_path: Path | str) -> Path:
    path = Path(artifact_path)
    return path.parent / DEPLOYMENT_MANIFEST_FILENAME


def write_artifact_manifest_sidecar(artifact_path: Path | str, manifest: Mapping[str, Any]) -> Path:
    sidecar_path = artifact_manifest_path(artifact_path)
    sidecar_path.write_text(json.dumps(dict(manifest), indent=2), encoding="utf-8")
    return sidecar_path


def write_deployment_manifest(artifact_path: Path | str, manifest: Mapping[str, Any]) -> Path:
    deploy_path = deployment_manifest_path(artifact_path)
    deploy_path.write_text(json.dumps(dict(manifest), indent=2), encoding="utf-8")
    return deploy_path


def write_contract_sidecars(artifact_path: Path | str, contract: Mapping[str, Any]) -> dict[str, Path]:
    path = Path(artifact_path)
    contract_path = path.with_suffix(path.suffix + ".contract.json")
    feature_order_path = path.with_suffix(path.suffix + ".feature_order.json")
    schema_hash_path = path.with_suffix(path.suffix + ".schema_hash.txt")

    contract_path.write_text(json.dumps(dict(contract), indent=2), encoding="utf-8")
    feature_order_path.write_text(
        json.dumps([str(feature) for feature in contract["feature_order"]], indent=2),
        encoding="utf-8",
    )
    schema_hash_path.write_text(str(contract["schema_hash"]) + "\n", encoding="utf-8")
    return {
        "contract": contract_path,
        "feature_order": feature_order_path,
        "schema_hash": schema_hash_path,
    }


def checkpoint_manifest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return manifest_without_artifact_sha256(manifest)


def torchscript_extra_files_for_manifest(manifest: Mapping[str, Any]) -> dict[str, bytes]:
    return {ARTIFACT_MANIFEST_FILENAME: manifest_json(manifest_without_artifact_sha256(manifest)).encode("utf-8")}


def embed_manifest_in_onnx_metadata(model: Any, manifest: Mapping[str, Any]) -> None:
    serialized = manifest_json(manifest_without_artifact_sha256(manifest))
    metadata = getattr(model, "metadata_props", None)
    if metadata is None:
        raise ArtifactManifestError("ONNX model does not expose metadata properties")
    entry = metadata.add()
    entry.key = ARTIFACT_MANIFEST_KEY
    entry.value = serialized


def _read_checkpoint_embedded_manifest(path: Path) -> dict[str, Any] | None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    embedded = payload.get(ARTIFACT_MANIFEST_KEY)
    if embedded is None:
        return None
    if not isinstance(embedded, Mapping):
        raise ArtifactManifestError("Checkpoint embedded manifest must be a mapping")
    return dict(embedded)


def _read_torchscript_embedded_manifest(path: Path) -> dict[str, Any] | None:
    extra_files = {ARTIFACT_MANIFEST_FILENAME: b""}
    torch.jit.load(str(path), _extra_files=extra_files)
    raw = extra_files[ARTIFACT_MANIFEST_FILENAME]
    if not raw:
        return None
    return manifest_from_json(raw.decode("utf-8"))


def _read_onnx_embedded_manifest(path: Path) -> dict[str, Any] | None:
    import onnx

    model = onnx.load(str(path))
    metadata = {prop.key: prop.value for prop in model.metadata_props}
    raw = metadata.get(ARTIFACT_MANIFEST_KEY)
    if not raw:
        return None
    return manifest_from_json(raw)


def read_embedded_manifest(path: Path | str, *, kind: str) -> dict[str, Any] | None:
    file_path = Path(path)
    if kind == "checkpoint":
        return _read_checkpoint_embedded_manifest(file_path)
    if kind == "torchscript":
        return _read_torchscript_embedded_manifest(file_path)
    if kind == "onnx":
        return _read_onnx_embedded_manifest(file_path)
    raise ArtifactManifestError(f"Unsupported artifact kind for embedded manifest: {kind}")


def _normalized_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return normalize_manifest(manifest)


def _manifest_projection(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "manifest_version": str(manifest["manifest_version"]),
        "schema_version": str(manifest["schema_version"]),
        "schema_hash": str(manifest["schema_hash"]),
        "contract_version": str(manifest["contract_version"]),
        "feature_order_hash": str(manifest["feature_order_hash"]),
        "feature_hash": str(manifest["feature_hash"]),
        "input_dim": int(manifest["input_dim"]),
        "binary_output_dim": int(manifest["binary_output_dim"]),
        "family_output_dim": int(manifest["family_output_dim"]),
    }


def _expected_manifest_projection(contract: Mapping[str, Any]) -> dict[str, Any]:
    feature_payload = _manifest_feature_payload(contract)
    return {
        "manifest_version": MANIFEST_VERSION,
        "schema_version": str(contract["schema_version"]),
        "schema_hash": str(contract["schema_hash"]),
        "contract_version": str(contract.get("contract_version", CONTRACT_VERSION)),
        "feature_order_hash": str(contract.get("feature_order_hash") or FEATURE_ORDER_HASH),
        "feature_hash": canonical_json_hash(feature_payload),
        "input_dim": int(contract["input_dim"]),
        "binary_output_dim": int(contract["binary_output_dim"]),
        "family_output_dim": int(contract["family_output_dim"]),
    }


def build_deployment_manifest(
    *,
    artifact_path: Path | str,
    manifest: Mapping[str, Any],
    config_hash: str,
) -> dict[str, Any]:
    _ = artifact_path
    return {
        "artifact_sha256": str(manifest["artifact_sha256"]),
        "schema_hash": str(manifest["schema_hash"]),
        "config_hash": str(config_hash),
        "dataset_hash": str(manifest["dataset_hash"]),
        "git_commit": str(manifest["git_commit"]),
        "exporter_version": str(manifest["exporter_version"]),
        "torch_version": str(manifest["torch_version"]),
        "onnx_opset": manifest.get("onnx_opset"),
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def read_deployment_manifest(artifact_path: Path | str) -> dict[str, Any]:
    payload = json.loads(deployment_manifest_path(artifact_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArtifactManifestError("Deployment manifest payload must decode to a mapping")
    return dict(payload)


def verify_deployment_manifest(
    artifact_path: Path | str,
    *,
    manifest: Mapping[str, Any],
    deployment_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(artifact_path)
    payload = deployment_manifest or read_deployment_manifest(path)
    expected = {
        "artifact_sha256": str(manifest["artifact_sha256"]),
        "schema_hash": str(manifest["schema_hash"]),
        "config_hash": str(manifest["export_config_hash"]),
        "dataset_hash": str(manifest["dataset_hash"]),
        "git_commit": str(manifest["git_commit"]),
        "exporter_version": str(manifest["exporter_version"]),
        "torch_version": str(manifest["torch_version"]),
        "onnx_opset": manifest.get("onnx_opset"),
    }
    for key, expected_value in expected.items():
        if str(payload.get(key)) != str(expected_value):
            raise ArtifactManifestError(
                f"Deployment manifest {key} mismatch: expected {expected_value}, got {payload.get(key)}"
            )
    if str(payload.get("artifact_sha256")) != artifact_sha256(path):
        raise ArtifactManifestError("Deployment manifest artifact checksum mismatch")
    return dict(payload)


def _validate_manifest_contract_projection(
    *,
    path: Path,
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    verified_contract = verify_contract_integrity(contract, context=f"{path.name} contract")
    projection = _manifest_projection(manifest)
    expected_projection = _expected_manifest_projection(verified_contract)
    if projection != expected_projection:
        raise ArtifactManifestError(f"Manifest contract projection mismatch for {path.name}")


def _validate_manifest_required_fields(*, path: Path, manifest: Mapping[str, Any]) -> None:
    required_fields = ("exporter_version", "git_commit", "runtime_version", "git_dirty")
    for field in required_fields:
        if field not in manifest:
            raise ArtifactManifestError(f"Manifest missing required field: {field}")
    if not str(manifest.get("exporter_version", "")).strip() or str(manifest.get("exporter_version")) == "unknown":
        raise ArtifactManifestError(f"Manifest exporter_version missing for {path.name}")
    if not str(manifest.get("git_commit", "")).strip() or str(manifest.get("git_commit")) == "unknown":
        raise ArtifactManifestError(f"Manifest git_commit missing for {path.name}")
    if str(manifest.get("contract_version")) != CONTRACT_VERSION:
        raise ArtifactManifestError(f"Manifest contract_version mismatch for {path.name}")
    if str(manifest.get("feature_order_hash")) != FEATURE_ORDER_HASH:
        raise ArtifactManifestError(f"Manifest feature_order_hash mismatch for {path.name}")


def _validate_sidecar_manifest(
    *,
    path: Path,
    sidecar_manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    _validate_manifest_contract_projection(path=path, manifest=sidecar_manifest, contract=contract)
    _validate_manifest_required_fields(path=path, manifest=sidecar_manifest)


def _hash_file(path: Path) -> str:
    return artifact_sha256(path)


def build_provenance_chain(
    artifact_path: Path | str,
    *,
    manifest: Mapping[str, Any],
    sidecars: Mapping[str, Path],
    deployment_manifest: Path | None = None,
    exporter_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    chain_payload = {
        "artifact_sha256": artifact_sha256(artifact_path),
        "manifest_sha256": canonical_json_hash(manifest_without_artifact_sha256(manifest)),
        "sidecar_sha256": {key: _hash_file(path) for key, path in sidecars.items()},
        "deployment_manifest_sha256": _hash_file(deployment_manifest) if deployment_manifest else None,
        "exporter_metadata_hash": _hash_optional_payload(exporter_metadata),
    }
    chain_payload["chain_sha256"] = canonical_json_hash(chain_payload)
    return chain_payload


def verify_provenance_chain(
    artifact_path: Path | str,
    *,
    manifest: Mapping[str, Any],
    sidecars: Mapping[str, Path],
    deployment_manifest: Path | None = None,
    exporter_metadata: Mapping[str, Any] | None = None,
    require_chain: bool = True,
) -> None:
    chain = manifest.get(PROVENANCE_CHAIN_KEY)
    if not chain:
        if require_chain:
            raise ArtifactManifestError("Missing provenance chain in artifact manifest")
        return
    computed = build_provenance_chain(
        artifact_path,
        manifest=manifest,
        sidecars=sidecars,
        deployment_manifest=deployment_manifest,
        exporter_metadata=exporter_metadata,
    )
    if str(chain.get("chain_sha256")) != str(computed.get("chain_sha256")):
        raise ArtifactManifestError("Provenance chain checksum mismatch")


def finalize_artifact_manifest(
    artifact_path: Path | str,
    manifest: Mapping[str, Any],
    *,
    provenance_chain: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    finalized = dict(manifest)
    finalized["artifact_sha256"] = artifact_sha256(artifact_path)
    if provenance_chain is not None:
        finalized[PROVENANCE_CHAIN_KEY] = dict(provenance_chain)
    write_artifact_manifest_sidecar(artifact_path, finalized)
    return finalized


def verify_contract_integrity(
    contract: Mapping[str, Any],
    *,
    context: str = "artifact contract",
) -> dict[str, Any]:
    normalized = dict(contract)
    assert_runtime_contract(
        schema_version=str(normalized["schema_version"]),
        schema_hash=str(normalized["schema_hash"]),
        feature_order=[str(feature) for feature in normalized["feature_order"]],
        input_dim=int(normalized["input_dim"]),
        binary_output_dim=int(normalized["binary_output_dim"]),
        family_output_dim=int(normalized["family_output_dim"]),
        context=context,
    )
    return normalized


def verify_artifact_manifest(
    artifact_path: Path | str,
    *,
    kind: str,
    contract: Mapping[str, Any] | None = None,
    embedded_manifest: Mapping[str, Any] | None = None,
    require_embedded_manifest: bool = True,
) -> dict[str, Any] | None:
    path = Path(artifact_path)

    sidecar_manifest = _load_sidecar_manifest(path)
    if sidecar_manifest is not None:
        actual_sha256 = artifact_sha256(path)
        expected_sha256 = str(sidecar_manifest.get("artifact_sha256"))
        if expected_sha256 != actual_sha256:
            raise ArtifactManifestError(
                f"Artifact checksum mismatch for {path.name}: expected {expected_sha256}, got {actual_sha256}"
            )

    if embedded_manifest is None:
        embedded_manifest = read_embedded_manifest(path, kind=kind)

    if sidecar_manifest is None and embedded_manifest is None and require_embedded_manifest:
        raise ArtifactManifestError(f"Missing artifact manifest sidecar or embedded manifest for {path.name}")

    if embedded_manifest is not None and sidecar_manifest is not None:
        if _normalized_manifest(embedded_manifest) != _normalized_manifest(manifest_without_artifact_sha256(sidecar_manifest)):
            raise ArtifactManifestError(f"Embedded manifest mismatch for {path.name}")

    if contract is not None and sidecar_manifest is not None:
        _validate_sidecar_manifest(path=path, sidecar_manifest=sidecar_manifest, contract=contract)

    return sidecar_manifest


def verify_runtime_compatibility(
    artifact_path: Path | str,
    *,
    kind: str,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    return verify_artifact_manifest(artifact_path, kind=kind, contract=contract)


def verify_export_provenance(
    artifact_path: Path | str,
    *,
    kind: str,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    return verify_artifact_manifest(artifact_path, kind=kind, contract=contract)


def verify_ingress_artifact(
    artifact_path: Path | str,
    *,
    kind: str,
    contract: Mapping[str, Any] | None = None,
    embedded_manifest: Mapping[str, Any] | None = None,
    allow_legacy_local_dev: bool = False,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """
    Ingress-level verification wrapper that optionally tolerates legacy artifacts.

    This wrapper is the only place where missing artifact manifests may be tolerated
    for external runtime bootstrap or explicitly gated local-dev migration flows.
    Use the `HELIX_ALLOW_LEGACY_ARTIFACTS=1` environment variable or set
    `allow_legacy_local_dev=True` to enable permissive behavior.
    """
    path = Path(artifact_path)
    # Centralize legacy-artifact gating via `allow_legacy_artifacts()` so
    # this logic is not scattered across the codebase and tests can toggle
    # the behavior by setting `HELIX_ALLOW_LEGACY_ARTIFACTS=1`.
    allow = allow_legacy_artifacts()
    if allow:
        assert not is_production_runtime(), (
            "Legacy artifact allowance is forbidden in production runtimes. "
            "Unset HELIX_ALLOW_LEGACY_ARTIFACTS or switch to a non-production env."
        )
    if allow_legacy_local_dev:
        if is_production_runtime():
            raise AssertionError(
                "Legacy artifact allowance is forbidden in production runtimes. "
                "Unset local-dev flags or switch to a non-production env."
            )
        allow = True
    try:
        # When allowed, do not require an embedded/sidecar manifest — let callers
        # proceed when only contract sidecars exist. Otherwise enforce manifest
        # presence and strict verification.
        return verify_artifact_provenance(
            path,
            kind=kind,
            contract=contract,
            embedded_manifest=embedded_manifest,
            require_embedded_manifest=not allow,
            **kwargs,
        )
    except ArtifactManifestError as exc:
        # If permissive mode is enabled and the only problem is a missing
        # manifest, swallow the error and return None (tolerate legacy).
        msg = str(exc)
        if allow and "Missing artifact manifest sidecar or embedded manifest" in msg:
            return None
        raise


def verify_sidecar_set(
    artifact_path: Path | str,
    *,
    kind: str,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    return verify_artifact_manifest(artifact_path, kind=kind, contract=contract)


def verify_artifact_provenance(
    artifact_path: Path | str,
    *,
    kind: str,
    contract: Mapping[str, Any] | None = None,
    embedded_manifest: Mapping[str, Any] | None = None,
    require_embedded_manifest: bool = True,
    sidecars: Mapping[str, Path] | None = None,
    deployment_manifest: Path | None = None,
    exporter_metadata: Mapping[str, Any] | None = None,
    require_chain: bool = True,
) -> dict[str, Any] | None:
    path = Path(artifact_path)
    try:
        deployment_manifest_payload = None
        if deployment_manifest is not None:
            deployment_manifest_payload = (
                read_deployment_manifest(path)
                if isinstance(deployment_manifest, Path)
                else deployment_manifest
            )
        sidecar_manifest = verify_artifact_manifest(
            path,
            kind=kind,
            contract=contract,
            embedded_manifest=embedded_manifest,
            require_embedded_manifest=require_embedded_manifest,
        )
        manifest = sidecar_manifest or embedded_manifest
        if manifest is None:
            return sidecar_manifest
        if deployment_manifest_payload is not None:
            verify_deployment_manifest(
                path,
                manifest=manifest,
                deployment_manifest=deployment_manifest_payload,
            )
        if sidecars:
            verify_provenance_chain(
                path,
                manifest=manifest,
                sidecars=sidecars,
                deployment_manifest=deployment_manifest,
                exporter_metadata=exporter_metadata,
                require_chain=require_chain,
            )
        if sidecar_manifest is None:
            return None
        return dict(sidecar_manifest)
    except Exception as exc:
        _emit_provenance_telemetry(artifact_path=path, kind=kind, failure=exc)
        raise


__all__ = [
    "ARTIFACT_MANIFEST_FILENAME",
    "ARTIFACT_MANIFEST_KEY",
    "ArtifactManifestError",
    "artifact_sha256",
    "artifact_manifest_path",
    "build_deployment_manifest",
    "build_artifact_manifest",
    "build_provenance_chain",
    "checkpoint_manifest_payload",
    "deployment_manifest_path",
    "embed_manifest_in_onnx_metadata",
    "finalize_artifact_manifest",
    "manifest_from_json",
    "manifest_json",
    "manifest_without_artifact_sha256",
    "normalize_manifest",
    "canonical_manifest_hash",
    "read_embedded_manifest",
    "read_deployment_manifest",
    "verify_deployment_manifest",
    "torchscript_extra_files_for_manifest",
    "verify_artifact_manifest",
    "verify_artifact_provenance",
    "verify_contract_integrity",
    "verify_provenance_chain",
    "verify_sidecar_set",
    "write_artifact_manifest_sidecar",
    "write_deployment_manifest",
    "write_contract_sidecars",
]
