"""Lifecycle artifact verifier for deterministic contract enforcement."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn

from helix_ids import __version__ as HELIX_IDS_VERSION
from helix_ids.contracts import (
    CANONICAL_BINARY_CLASSES,
    CANONICAL_FAMILY_CLASSES,
    CANONICAL_INPUT_DIM,
    CONTRACT_VERSION,
    FEATURE_ORDER_HASH,
    runtime_contract_payload,
)
from helix_ids.governance.parameters import allow_legacy_artifacts
from helix_ids.governance.provenance import (
    ARTIFACT_MANIFEST_KEY,
    ArtifactManifestError,
    artifact_manifest_path,
    build_artifact_manifest,
    build_provenance_chain,
    checkpoint_manifest_payload,
    finalize_artifact_manifest,
    manifest_without_artifact_sha256,
    normalize_manifest,
    read_embedded_manifest,
    verify_artifact_manifest,
    verify_contract_integrity,
    verify_provenance_chain,
    write_contract_sidecars,
)

# Parity tolerances (float32 drift between formats is expected; keep explicit).
PARITY_ATOL = 1e-4
PARITY_RTOL = 1e-4

# Artifact sidecar file extensions
CONTRACT_SUFFIX = ".contract.json"
FEATURE_ORDER_SUFFIX = ".feature_order.json"
SCHEMA_HASH_SUFFIX = ".schema_hash.txt"
MANIFEST_SUFFIX = "manifest.json"


@dataclass(frozen=True)
class LifecycleArtifacts:
    workdir: Path
    checkpoint: Path
    torchscript: Path
    onnx: Path
    contract: dict[str, Any]
    manifest: dict[str, Any]


class _TinyHelixNet(nn.Module):
    def __init__(self, input_dim: int, binary_dim: int, family_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.ReLU(),
        )
        self.binary_head = nn.Linear(8, binary_dim)
        self.family_head = nn.Linear(8, family_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        return self.binary_head(feats), self.family_head(feats)


def _seed_everything(seed: int = 13) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _assert_no_legacy_flags() -> None:
    if allow_legacy_artifacts():
        raise AssertionError("Legacy artifact allowance is forbidden during lifecycle verification")
    if os.getenv("HELIX_ALLOW_LEGACY_MANIFEST"):
        raise AssertionError("Legacy manifest allowance is forbidden during lifecycle verification")
    if os.getenv("HELIX_ALLOW_LEGACY_ARTIFACTS"):
        raise AssertionError("Legacy artifact allowance is forbidden during lifecycle verification")


def _synthetic_dataset(samples: int = 32, seed: int = 17) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((samples, CANONICAL_INPUT_DIM)).astype(np.float32)
    binary = (x.sum(axis=1) > 0).astype(np.int64)
    family = (np.abs(x[:, 0]) * 3 + np.abs(x[:, 1]) * 5).astype(np.int64) % CANONICAL_FAMILY_CLASSES
    return torch.from_numpy(x), torch.from_numpy(binary), torch.from_numpy(family)


def _train_tiny_model(x: torch.Tensor, y_bin: torch.Tensor, y_family: torch.Tensor) -> _TinyHelixNet:
    model = _TinyHelixNet(CANONICAL_INPUT_DIM, CANONICAL_BINARY_CLASSES, CANONICAL_FAMILY_CLASSES)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        logits_bin, logits_family = model(x)
        loss = loss_fn(logits_bin, y_bin) + loss_fn(logits_family, y_family)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _make_contract() -> dict[str, Any]:
    contract = runtime_contract_payload()
    contract["feature_order"] = list(contract["feature_order"])
    return cast(dict[str, Any], contract)


def _build_manifest(contract: dict[str, Any], *, model_architecture: str, export_config: dict[str, Any]) -> dict[str, Any]:
    result = build_artifact_manifest(
        contract=contract,
        model_architecture=model_architecture,
        export_config=export_config,
        git_commit=_expected_git_commit(),
        exporter_version=HELIX_IDS_VERSION,
        runtime_version=HELIX_IDS_VERSION,
    )
    return cast(dict[str, Any], result)


def _write_checkpoint(
    path: Path,
    model: nn.Module,
    contract: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {"model_state_dict": model.state_dict()}
    payload.update(contract)
    payload[ARTIFACT_MANIFEST_KEY] = checkpoint_manifest_payload(manifest)
    torch.save(payload, path)
    sidecars = write_contract_sidecars(path, contract)
    chain = build_provenance_chain(path, manifest=manifest, sidecars=sidecars)
    result = finalize_artifact_manifest(path, manifest, provenance_chain=chain)
    return cast(dict[str, Any], result)


def _expected_git_commit() -> str:
    return (
        os.getenv("GITHUB_SHA")
        or os.getenv("CI_COMMIT_SHA")
        or os.getenv("GIT_COMMIT")
        or HELIX_IDS_VERSION
    )


def _reload_checkpoint(path: Path, contract: dict[str, Any]) -> _TinyHelixNet:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ArtifactManifestError("Checkpoint payload must be a mapping")
    contract_keys = list(contract.keys())
    payload_contract = {key: payload.get(key) for key in contract_keys}
    if any(value is None for value in payload_contract.values()):
        raise ArtifactManifestError("Checkpoint contract payload is incomplete")
    verify_contract_integrity(payload_contract, context="checkpoint contract")
    if payload_contract != contract:
        raise ArtifactManifestError("Checkpoint contract payload mismatch")
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ArtifactManifestError("Checkpoint missing model_state_dict")
    model = _TinyHelixNet(
        int(contract["input_dim"]),
        int(contract["binary_output_dim"]),
        int(contract["family_output_dim"]),
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _write_torchscript(path: Path, model: nn.Module, contract: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    example = torch.arange(CANONICAL_INPUT_DIM, dtype=torch.float32).reshape(1, -1)
    traced = torch.jit.trace(model, example, strict=False)
    from helix_ids.governance.provenance import torchscript_extra_files_for_manifest

    torch.jit.save(
        traced,
        str(path),
        _extra_files=torchscript_extra_files_for_manifest(manifest),
    )
    sidecars = write_contract_sidecars(path, contract)
    chain = build_provenance_chain(path, manifest=manifest, sidecars=sidecars)
    result = finalize_artifact_manifest(path, manifest, provenance_chain=chain)
    return cast(dict[str, Any], result)


def _write_onnx(path: Path, model: nn.Module, contract: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    import onnx

    dummy = torch.arange(CANONICAL_INPUT_DIM, dtype=torch.float32).reshape(1, -1)
    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        export_params=True,
        opset_version=13,
        input_names=["features"],
        output_names=["binary_logits", "family_logits"],
        dynamic_axes={"features": {0: "batch"}, "binary_logits": {0: "batch"}, "family_logits": {0: "batch"}},
    )
    model_onnx = onnx.load(str(path))
    from helix_ids.governance.provenance import embed_manifest_in_onnx_metadata

    embed_manifest_in_onnx_metadata(model_onnx, manifest)
    onnx.save(model_onnx, str(path))
    sidecars = write_contract_sidecars(path, contract)
    chain = build_provenance_chain(path, manifest=manifest, sidecars=sidecars)
    result = finalize_artifact_manifest(path, manifest, provenance_chain=chain)
    return cast(dict[str, Any], result)


def _parity_check(
    *,
    model: nn.Module,
    torchscript_path: Path,
    onnx_path: Path,
    inputs: torch.Tensor,
) -> None:
    with torch.no_grad():
        ref_bin, ref_family = model(inputs)
        ts_model = torch.jit.load(str(torchscript_path))
        ts_bin, ts_family = ts_model(inputs)
    ref_bin_np = ref_bin.detach().cpu().numpy()
    ref_family_np = ref_family.detach().cpu().numpy()
    ts_bin_np = ts_bin.detach().cpu().numpy()
    ts_family_np = ts_family.detach().cpu().numpy()
    if not np.allclose(ref_bin_np, ts_bin_np, atol=PARITY_ATOL, rtol=PARITY_RTOL):
        raise RuntimeError("TorchScript parity check failed")
    if not np.allclose(ref_family_np, ts_family_np, atol=PARITY_ATOL, rtol=PARITY_RTOL):
        raise RuntimeError("TorchScript parity check failed")

    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: inputs.detach().cpu().numpy()})
    onnx_bin, onnx_family = outputs
    if not np.allclose(ref_bin_np, onnx_bin, atol=PARITY_ATOL, rtol=PARITY_RTOL):
        raise RuntimeError("ONNX parity check failed")
    if not np.allclose(ref_family_np, onnx_family, atol=PARITY_ATOL, rtol=PARITY_RTOL):
        raise RuntimeError("ONNX parity check failed")


def create_lifecycle_artifacts(workdir: Path, *, require_onnx: bool = True) -> LifecycleArtifacts:
    _assert_no_legacy_flags()
    _seed_everything(23)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    if require_onnx:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401

    x, y_bin, y_family = _synthetic_dataset()
    model = _train_tiny_model(x, y_bin, y_family)

    contract = _make_contract()

    checkpoint_path = workdir / "lifecycle.pt"
    torchscript_path = workdir / "lifecycle.torchscript.pt"
    onnx_path = workdir / "lifecycle.onnx"

    manifest_ckpt = _build_manifest(contract, model_architecture=model.__class__.__name__, export_config={"format": "checkpoint"})
    manifest_ts = _build_manifest(contract, model_architecture=model.__class__.__name__, export_config={"format": "torchscript"})
    manifest_onnx = _build_manifest(contract, model_architecture=model.__class__.__name__, export_config={"format": "onnx", "opset": 13})

    manifest_ckpt = _write_checkpoint(checkpoint_path, model, contract, manifest_ckpt)
    manifest_ts = _write_torchscript(torchscript_path, model, contract, manifest_ts)
    manifest_onnx = _write_onnx(onnx_path, model, contract, manifest_onnx)

    verify_lifecycle_artifacts(
        LifecycleArtifacts(
            workdir=workdir,
            checkpoint=checkpoint_path,
            torchscript=torchscript_path,
            onnx=onnx_path,
            contract=contract,
            manifest={"checkpoint": manifest_ckpt, "torchscript": manifest_ts, "onnx": manifest_onnx},
        ),
        require_onnx=require_onnx,
    )

    reloaded_model = _reload_checkpoint(checkpoint_path, contract)
    _parity_check(model=reloaded_model, torchscript_path=torchscript_path, onnx_path=onnx_path, inputs=x[:4])

    return LifecycleArtifacts(
        workdir=workdir,
        checkpoint=checkpoint_path,
        torchscript=torchscript_path,
        onnx=onnx_path,
        contract=contract,
        manifest={"checkpoint": manifest_ckpt, "torchscript": manifest_ts, "onnx": manifest_onnx},
    )


def _verify_manifest_pair(path: Path, *, kind: str, contract: dict[str, Any]) -> dict[str, Any]:
    sidecar = verify_artifact_manifest(path, kind=kind, contract=contract)
    embedded = read_embedded_manifest(path, kind=kind)
    if embedded is None or sidecar is None:
        raise ArtifactManifestError(f"Missing artifact manifest sidecar or embedded manifest for {path.name}")
    if _manifest_payload(sidecar) != _manifest_payload(embedded):
        raise ArtifactManifestError(f"Embedded manifest mismatch for {path.name}")
    return cast(dict[str, Any], sidecar)


def _manifest_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    result = normalize_manifest(manifest_without_artifact_sha256(manifest))
    return cast(dict[str, Any], result)


def verify_lifecycle_artifacts(artifacts: LifecycleArtifacts, *, require_onnx: bool = True) -> dict[str, Any]:
    _assert_no_legacy_flags()
    contract = artifacts.contract
    expected_git_commit = _expected_git_commit()
    checkpoints = {
        "checkpoint": (artifacts.checkpoint, "checkpoint"),
        "torchscript": (artifacts.torchscript, "torchscript"),
    }
    if require_onnx:
        checkpoints["onnx"] = (artifacts.onnx, "onnx")

    manifests: dict[str, Any] = {}
    for key, (path, kind) in checkpoints.items():
        manifest = _verify_manifest_pair(path, kind=kind, contract=contract)
        sidecars = {
            "contract": path.with_suffix(path.suffix + CONTRACT_SUFFIX),
            "feature_order": path.with_suffix(path.suffix + FEATURE_ORDER_SUFFIX),
            "schema_hash": path.with_suffix(path.suffix + SCHEMA_HASH_SUFFIX),
        }
        _verify_contract_sidecars(path, contract)
        verify_provenance_chain(path, manifest=manifest, sidecars=sidecars, require_chain=True)
        manifests[key] = manifest

    for key, manifest in manifests.items():
        if str(manifest.get("exporter_version")) != str(HELIX_IDS_VERSION):
            raise RuntimeError(f"exporter_version mismatch for {key}")
        if str(manifest.get("git_commit")) != str(expected_git_commit):
            raise ArtifactManifestError(f"git_commit mismatch for {key}")
        if str(manifest.get("contract_version")) != CONTRACT_VERSION:
            raise ArtifactManifestError(f"Manifest contract_version mismatch for {key}")
        if str(manifest.get("feature_order_hash")) != FEATURE_ORDER_HASH:
            raise ArtifactManifestError(f"Manifest feature_order_hash mismatch for {key}")
    return manifests


def run_lifecycle_verification(workdir: Path, *, require_onnx: bool = True) -> dict[str, Any]:
    artifacts = create_lifecycle_artifacts(workdir, require_onnx=require_onnx)
    verify_lifecycle_artifacts(artifacts, require_onnx=require_onnx)
    return {
        "artifacts": {
            "checkpoint": artifacts.checkpoint,
            "torchscript": artifacts.torchscript,
            "onnx": artifacts.onnx,
        },
        "parity_ok": True,
    }


def clone_lifecycle_artifacts(artifacts: LifecycleArtifacts, workdir: Path) -> LifecycleArtifacts:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    mapping = {
        artifacts.checkpoint: workdir / artifacts.checkpoint.name,
        artifacts.torchscript: workdir / artifacts.torchscript.name,
        artifacts.onnx: workdir / artifacts.onnx.name,
    }
    for src, dst in mapping.items():
        dst.write_bytes(src.read_bytes())
        for suffix in [CONTRACT_SUFFIX, FEATURE_ORDER_SUFFIX, SCHEMA_HASH_SUFFIX, "." + MANIFEST_SUFFIX]:
            sidecar = src.with_suffix(src.suffix + suffix)
            if sidecar.exists():
                (dst.with_suffix(dst.suffix + suffix)).write_bytes(sidecar.read_bytes())
    return LifecycleArtifacts(
        workdir=workdir,
        checkpoint=mapping[artifacts.checkpoint],
        torchscript=mapping[artifacts.torchscript],
        onnx=mapping[artifacts.onnx],
        contract=dict(artifacts.contract),
        manifest=dict(artifacts.manifest),
    )


def _load_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    return cast(dict[str, Any], result)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _verify_contract_sidecars(path: Path, contract: dict[str, Any]) -> None:
    contract_path = path.with_suffix(path.suffix + CONTRACT_SUFFIX)
    feature_order_path = path.with_suffix(path.suffix + FEATURE_ORDER_SUFFIX)
    schema_hash_path = path.with_suffix(path.suffix + SCHEMA_HASH_SUFFIX)
    if not contract_path.exists():
        raise ArtifactManifestError(f"Missing required sidecar: {contract_path.name}")
    if not feature_order_path.exists():
        raise ArtifactManifestError(f"Missing required sidecar: {feature_order_path.name}")
    if not schema_hash_path.exists():
        raise ArtifactManifestError(f"Missing required sidecar: {schema_hash_path.name}")
    payload = _load_json(contract_path)
    try:
        verify_contract_integrity(payload, context="contract sidecar")
    except AssertionError as exc:
        raise ArtifactManifestError(str(exc)) from exc
    if payload != contract:
        raise ArtifactManifestError(f"Contract sidecar mismatch for {path.name}")
    feature_payload = _load_json(feature_order_path)
    if not isinstance(feature_payload, list):
        raise ArtifactManifestError(f"Feature order sidecar must be a list for {path.name}")
    if [str(feature) for feature in feature_payload] != [str(feature) for feature in contract["feature_order"]]:
        raise ArtifactManifestError(f"Feature order sidecar mismatch for {path.name}")
    schema_hash = schema_hash_path.read_text(encoding="utf-8").strip()
    if schema_hash != str(contract["schema_hash"]):
        raise ArtifactManifestError(f"Schema hash sidecar mismatch for {path.name}")


def tamper_deleted_manifest(artifacts: LifecycleArtifacts, *, kind: str) -> None:
    path = _resolve_artifact_path(artifacts, kind)
    manifest_path = artifact_manifest_path(path)
    if manifest_path.exists():
        manifest_path.unlink()


def tamper_reordered_feature_sidecar(artifacts: LifecycleArtifacts, *, kind: str) -> None:
    path = _resolve_artifact_path(artifacts, kind)
    contract_path = path.with_suffix(path.suffix + CONTRACT_SUFFIX)
    payload = _load_json(contract_path)
    payload["feature_order"] = list(reversed(payload["feature_order"]))
    _write_json(contract_path, payload)


def tamper_missing_feature_sidecar(artifacts: LifecycleArtifacts, *, kind: str) -> None:
    path = _resolve_artifact_path(artifacts, kind)
    contract_path = path.with_suffix(path.suffix + CONTRACT_SUFFIX)
    payload = _load_json(contract_path)
    features = list(payload.get("feature_order", []))
    if features:
        features.pop()
    payload["feature_order"] = features
    _write_json(contract_path, payload)


def tamper_extra_feature_sidecar(artifacts: LifecycleArtifacts, *, kind: str) -> None:
    path = _resolve_artifact_path(artifacts, kind)
    contract_path = path.with_suffix(path.suffix + CONTRACT_SUFFIX)
    payload = _load_json(contract_path)
    features = list(payload.get("feature_order", []))
    features.append("extra_feature")
    payload["feature_order"] = features
    _write_json(contract_path, payload)


def tamper_schema_hash(artifacts: LifecycleArtifacts, *, kind: str) -> None:
    path = _resolve_artifact_path(artifacts, kind)
    contract_path = path.with_suffix(path.suffix + CONTRACT_SUFFIX)
    payload = _load_json(contract_path)
    payload["schema_hash"] = "deadbeef"
    _write_json(contract_path, payload)


def tamper_contract_version(artifacts: LifecycleArtifacts, *, kind: str) -> None:
    path = _resolve_artifact_path(artifacts, kind)
    manifest_path = artifact_manifest_path(path)
    payload = _load_json(manifest_path)
    payload["contract_version"] = "0.0.0"
    _write_json(manifest_path, payload)


def tamper_artifact_hash(artifacts: LifecycleArtifacts, *, kind: str) -> None:
    path = _resolve_artifact_path(artifacts, kind)
    manifest_path = artifact_manifest_path(path)
    payload = _load_json(manifest_path)
    payload["artifact_sha256"] = "bad" + str(payload.get("artifact_sha256"))
    _write_json(manifest_path, payload)


def tamper_exporter_version(artifacts: LifecycleArtifacts, *, kind: str) -> None:
    path = _resolve_artifact_path(artifacts, kind)
    manifest_path = artifact_manifest_path(path)
    payload = _load_json(manifest_path)
    payload["exporter_version"] = "0.0.0"
    _write_json(manifest_path, payload)


def tamper_provenance_chain(artifacts: LifecycleArtifacts, *, kind: str) -> None:
    path = _resolve_artifact_path(artifacts, kind)
    manifest_path = artifact_manifest_path(path)
    payload = _load_json(manifest_path)
    chain = payload.get("provenance_chain")
    if isinstance(chain, dict):
        chain["chain_sha256"] = "00" + str(chain.get("chain_sha256"))
        payload["provenance_chain"] = chain
    _write_json(manifest_path, payload)


def tamper_embedded_sidecar_mismatch(artifacts: LifecycleArtifacts, *, kind: str) -> None:
    path = _resolve_artifact_path(artifacts, kind)
    if kind == "checkpoint":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        embedded = dict(payload.get(ARTIFACT_MANIFEST_KEY, {}))
        embedded["schema_hash"] = "mismatch"
        payload[ARTIFACT_MANIFEST_KEY] = embedded
        torch.save(payload, path)
        return
    if kind == "torchscript":
        extra = {MANIFEST_SUFFIX: b""}
        model = torch.jit.load(str(path), _extra_files=extra)
        embedded = json.loads(extra[MANIFEST_SUFFIX].decode("utf-8"))
        embedded["schema_hash"] = "mismatch"
        extra_files = {MANIFEST_SUFFIX: json.dumps(embedded).encode("utf-8")}
        torch.jit.save(model, str(path), _extra_files=extra_files)
        return
    if kind == "onnx":
        import onnx

        model = onnx.load(str(path))
        for prop in model.metadata_props:
            if prop.key == ARTIFACT_MANIFEST_KEY:
                embedded = json.loads(prop.value)
                embedded["schema_hash"] = "mismatch"
                prop.value = json.dumps(embedded)
        onnx.save(model, str(path))
        return
    raise ValueError(f"Unsupported kind for embedded tamper: {kind}")


def tamper_sidecar_manifest_mismatch(artifacts: LifecycleArtifacts, *, kind: str) -> None:
    path = _resolve_artifact_path(artifacts, kind)
    manifest_path = artifact_manifest_path(path)
    payload = _load_json(manifest_path)
    payload["schema_hash"] = "sidecar-mismatch"
    _write_json(manifest_path, payload)


def tamper_embedded_and_sidecar_mismatch(artifacts: LifecycleArtifacts, *, kind: str) -> None:
    path = _resolve_artifact_path(artifacts, kind)
    manifest_path = artifact_manifest_path(path)
    payload = _load_json(manifest_path)
    payload["schema_hash"] = "sidecar-mismatch"
    _write_json(manifest_path, payload)
    if kind == "checkpoint":
        checkpoint_payload = torch.load(path, map_location="cpu", weights_only=True)
        embedded = dict(checkpoint_payload.get(ARTIFACT_MANIFEST_KEY, {}))
        embedded["schema_hash"] = "embedded-mismatch"
        checkpoint_payload[ARTIFACT_MANIFEST_KEY] = embedded
        torch.save(checkpoint_payload, path)
        return
    if kind == "torchscript":
        extra = {MANIFEST_SUFFIX: b""}
        model = torch.jit.load(str(path), _extra_files=extra)
        embedded = json.loads(extra[MANIFEST_SUFFIX].decode("utf-8"))
        embedded["schema_hash"] = "embedded-mismatch"
        extra_files = {MANIFEST_SUFFIX: json.dumps(embedded).encode("utf-8")}
        torch.jit.save(model, str(path), _extra_files=extra_files)
        return
    if kind == "onnx":
        import onnx

        model = onnx.load(str(path))
        for prop in model.metadata_props:
            if prop.key == ARTIFACT_MANIFEST_KEY:
                embedded = json.loads(prop.value)
                embedded["schema_hash"] = "embedded-mismatch"
                prop.value = json.dumps(embedded)
        onnx.save(model, str(path))
        return
    raise ValueError(f"Unsupported kind for embedded tamper: {kind}")


def tamper_manifest_replay(artifacts: LifecycleArtifacts, *, kind: str) -> None:
    path = _resolve_artifact_path(artifacts, kind)
    manifest_path = artifact_manifest_path(path)
    payload = _load_json(manifest_path)
    payload["git_commit"] = "replay-commit"
    sidecars = {
        "contract": path.with_suffix(path.suffix + CONTRACT_SUFFIX),
        "feature_order": path.with_suffix(path.suffix + FEATURE_ORDER_SUFFIX),
        "schema_hash": path.with_suffix(path.suffix + SCHEMA_HASH_SUFFIX),
    }
    payload["provenance_chain"] = build_provenance_chain(path, manifest=payload, sidecars=sidecars)
    _write_json(manifest_path, payload)

    embedded = manifest_without_artifact_sha256(payload)
    if kind == "checkpoint":
        checkpoint_payload = torch.load(path, map_location="cpu", weights_only=True)
        checkpoint_payload[ARTIFACT_MANIFEST_KEY] = embedded
        torch.save(checkpoint_payload, path)
        return
    if kind == "torchscript":
        extra_files = {MANIFEST_SUFFIX: json.dumps(embedded).encode("utf-8")}
        model = torch.jit.load(str(path))
        torch.jit.save(model, str(path), _extra_files=extra_files)
        return
    if kind == "onnx":
        import onnx

        model = onnx.load(str(path))
        for prop in model.metadata_props:
            if prop.key == ARTIFACT_MANIFEST_KEY:
                prop.value = json.dumps(embedded)
        onnx.save(model, str(path))
        return
    raise ValueError(f"Unsupported kind for manifest replay: {kind}")


def _resolve_artifact_path(artifacts: LifecycleArtifacts, kind: str) -> Path:
    if kind == "checkpoint":
        return artifacts.checkpoint
    if kind == "torchscript":
        return artifacts.torchscript
    if kind == "onnx":
        return artifacts.onnx
    raise ValueError(f"Unknown artifact kind: {kind}")


