"""Artifact provenance manifests and verification helpers."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from helix_ids import __version__ as HELIX_IDS_VERSION
from helix_ids.contracts import CONTRACT_VERSION, runtime_contract_payload
from helix_ids.contracts.schema_contract import assert_runtime_contract
from helix_ids.governance.fingerprinting import canonical_json_hash


ARTIFACT_MANIFEST_KEY = "artifact_manifest"
ARTIFACT_MANIFEST_FILENAME = "artifact_manifest.json"


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
    if not sidecar_path.exists():
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
    training_timestamp: str | None = None,
    onnx_opset: int | None = None,
    artifact_sha256_value: str | None = None,
) -> dict[str, Any]:
    """Build a canonical provenance manifest for an artifact."""

    contract_payload = dict(contract or runtime_contract_payload())
    feature_payload = _manifest_feature_payload(contract_payload)
    manifest = {
        "schema_version": str(contract_payload["schema_version"]),
        "schema_hash": str(contract_payload["schema_hash"]),
        "contract_version": str(contract_payload.get("contract_version", CONTRACT_VERSION)),
        "feature_order_hash": canonical_json_hash({"feature_order": feature_payload["feature_order"]}),
        "feature_hash": canonical_json_hash(feature_payload),
        "dataset_hash": dataset_hash,
        "training_config_hash": _hash_optional_payload(training_config),
        "export_config_hash": _hash_optional_payload(export_config),
        "git_commit": git_commit or _environment_value("GITHUB_SHA", "CI_COMMIT_SHA", "GIT_COMMIT"),
        "git_branch": git_branch or _environment_value("GITHUB_REF_NAME", "CI_COMMIT_REF_NAME", "GIT_BRANCH"),
        "exporter_version": exporter_version or HELIX_IDS_VERSION,
        "training_timestamp": training_timestamp
        or datetime.now(timezone.utc).isoformat(),
        "torch_version": str(torch.__version__),
        "onnx_opset": int(onnx_opset) if onnx_opset is not None else None,
        "artifact_sha256": artifact_sha256_value,
        "model_architecture": str(model_architecture),
        "binary_output_dim": int(contract_payload["binary_output_dim"]),
        "family_output_dim": int(contract_payload["family_output_dim"]),
        "input_dim": int(contract_payload["input_dim"]),
    }
    return manifest


def manifest_without_artifact_sha256(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "artifact_sha256"}


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


def write_artifact_manifest_sidecar(artifact_path: Path | str, manifest: Mapping[str, Any]) -> Path:
    sidecar_path = artifact_manifest_path(artifact_path)
    sidecar_path.write_text(json.dumps(dict(manifest), indent=2), encoding="utf-8")
    return sidecar_path


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
    normalized = dict(manifest)
    normalized["dataset_hash"] = normalized.get("dataset_hash")
    normalized["training_config_hash"] = normalized.get("training_config_hash")
    normalized["export_config_hash"] = normalized.get("export_config_hash")
    normalized["git_commit"] = normalized.get("git_commit")
    normalized["git_branch"] = normalized.get("git_branch")
    normalized["exporter_version"] = normalized.get("exporter_version")
    normalized["training_timestamp"] = normalized.get("training_timestamp")
    normalized["onnx_opset"] = normalized.get("onnx_opset")
    normalized["artifact_sha256"] = normalized.get("artifact_sha256")
    normalized["model_architecture"] = normalized.get("model_architecture")
    normalized["binary_output_dim"] = int(normalized["binary_output_dim"])
    normalized["family_output_dim"] = int(normalized["family_output_dim"])
    normalized["input_dim"] = int(normalized["input_dim"])
    return normalized


def _manifest_projection(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": str(manifest["schema_version"]),
        "schema_hash": str(manifest["schema_hash"]),
        "feature_order_hash": str(manifest["feature_order_hash"]),
        "feature_hash": str(manifest["feature_hash"]),
        "input_dim": int(manifest["input_dim"]),
        "binary_output_dim": int(manifest["binary_output_dim"]),
        "family_output_dim": int(manifest["family_output_dim"]),
    }


def _expected_manifest_projection(contract: Mapping[str, Any]) -> dict[str, Any]:
    feature_payload = _manifest_feature_payload(contract)
    return {
        "schema_version": str(contract["schema_version"]),
        "schema_hash": str(contract["schema_hash"]),
        "feature_order_hash": canonical_json_hash({"feature_order": feature_payload["feature_order"]}),
        "feature_hash": canonical_json_hash(feature_payload),
        "input_dim": int(contract["input_dim"]),
        "binary_output_dim": int(contract["binary_output_dim"]),
        "family_output_dim": int(contract["family_output_dim"]),
    }


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

    # Load and optionally verify sidecar manifest
    sidecar_manifest = _load_sidecar_manifest(path)
    if sidecar_manifest is not None:
        # verify checksum matches the artifact
        actual_sha256 = artifact_sha256(path)
        expected_sha256 = str(sidecar_manifest.get("artifact_sha256"))
        if expected_sha256 != actual_sha256:
            raise ArtifactManifestError(
                f"Artifact checksum mismatch for {path.name}: expected {expected_sha256}, got {actual_sha256}"
            )

    # Read embedded manifest when not provided explicitly
    if embedded_manifest is None:
        embedded_manifest = read_embedded_manifest(path, kind=kind)

    # Require a provenance manifest when requested
    if sidecar_manifest is None and embedded_manifest is None and require_embedded_manifest:
        raise ArtifactManifestError(f"Missing artifact manifest sidecar or embedded manifest for {path.name}")

    # When both exist, validate normalized equality (ignore artifact_sha256)
    if embedded_manifest is not None and sidecar_manifest is not None:
        if _normalized_manifest(embedded_manifest) != _normalized_manifest(manifest_without_artifact_sha256(sidecar_manifest)):
            raise ArtifactManifestError(f"Embedded manifest mismatch for {path.name}")

    # If a contract is provided, ensure the sidecar (if present) matches
    if contract is not None and sidecar_manifest is not None:
        verified_contract = verify_contract_integrity(contract, context=f"{path.name} contract")
        projection = _manifest_projection(sidecar_manifest)
        expected_projection = _expected_manifest_projection(verified_contract)
        if projection != expected_projection:
            raise ArtifactManifestError(f"Manifest contract projection mismatch for {path.name}")

    return sidecar_manifest


def finalize_artifact_manifest(artifact_path: Path | str, manifest: Mapping[str, Any]) -> dict[str, Any]:
    finalized = dict(manifest)
    finalized["artifact_sha256"] = artifact_sha256(artifact_path)
    write_artifact_manifest_sidecar(artifact_path, finalized)
    return finalized


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


def verify_sidecar_set(
    artifact_path: Path | str,
    *,
    kind: str,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    return verify_artifact_manifest(artifact_path, kind=kind, contract=contract)


__all__ = [
    "ARTIFACT_MANIFEST_FILENAME",
    "ARTIFACT_MANIFEST_KEY",
    "ArtifactManifestError",
    "artifact_sha256",
    "artifact_manifest_path",
    "build_artifact_manifest",
    "checkpoint_manifest_payload",
    "embed_manifest_in_onnx_metadata",
    "finalize_artifact_manifest",
    "manifest_from_json",
    "manifest_json",
    "manifest_without_artifact_sha256",
    "read_embedded_manifest",
    "torchscript_extra_files_for_manifest",
    "verify_artifact_manifest",
    "verify_contract_integrity",
    "verify_export_provenance",
    "verify_runtime_compatibility",
    "verify_sidecar_set",
    "write_artifact_manifest_sidecar",
    "write_contract_sidecars",
]