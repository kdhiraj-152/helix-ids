import json
import os
from pathlib import Path

import pytest
import torch

from helix_ids.governance import provenance


def _tiny_model():
    return torch.nn.Linear(4, 2)


def test_checkpoint_provenance_roundtrip(tmp_path):
    model = _tiny_model()
    path = tmp_path / "test_ckpt.pt"

    contract = provenance.runtime_contract_payload()
    manifest_base = provenance.build_artifact_manifest(model_architecture="linear", contract=contract)
    # embed manifest (without sha)
    payload = provenance.checkpoint_manifest_payload(manifest_base)
    checkpoint = {"model_state": model.state_dict(), provenance.ARTIFACT_MANIFEST_KEY: payload}
    torch.save(checkpoint, str(path))

    _finalized = provenance.finalize_artifact_manifest(path, manifest_base)
    provenance.write_contract_sidecars(path, contract)

    sidecar = provenance.verify_artifact_manifest(path, kind="checkpoint", contract=contract)
    assert sidecar is not None
    assert str(_finalized.get("artifact_sha256")) == provenance.artifact_sha256(path)


def test_detached_sidecar_tamper_detection(tmp_path):
    model = _tiny_model()
    path = tmp_path / "test_ckpt2.pt"

    contract = provenance.runtime_contract_payload()
    manifest_base = provenance.build_artifact_manifest(model_architecture="linear", contract=contract)
    payload = provenance.checkpoint_manifest_payload(manifest_base)
    checkpoint = {"model_state": model.state_dict(), provenance.ARTIFACT_MANIFEST_KEY: payload}
    torch.save(checkpoint, str(path))

    _finalized = provenance.finalize_artifact_manifest(path, manifest_base)
    provenance.write_contract_sidecars(path, contract)

    # Tamper the sidecar artifact sha
    sidecar_path = provenance.artifact_manifest_path(path)
    data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    data["artifact_sha256"] = "deadbeef"
    sidecar_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(provenance.ArtifactManifestError):
        provenance.verify_artifact_manifest(path, kind="checkpoint", contract=contract)


def test_torchscript_loader_manifest_verification(tmp_path):
    model = _tiny_model()
    scripted = torch.jit.trace(model, torch.randn(1, 4))
    path = tmp_path / "model.ts"

    contract = provenance.runtime_contract_payload()
    manifest_base = provenance.build_artifact_manifest(model_architecture="linear", contract=contract)
    extra = provenance.torchscript_extra_files_for_manifest(manifest_base)

    # Save torchscript with extra files
    torch.jit.save(scripted, str(path), _extra_files=extra)

    provenance.finalize_artifact_manifest(path, manifest_base)
    provenance.write_contract_sidecars(path, contract)

    sidecar = provenance.verify_artifact_manifest(path, kind="torchscript", contract=contract)
    assert sidecar is not None


def test_onnx_metadata_manifest_verification(tmp_path):
    try:
        import onnx  # type: ignore
    except Exception:
        pytest.skip("onnx not available")

    model = _tiny_model()
    dummy_input = torch.randn(1, 4)
    path = tmp_path / "model.onnx"

    # Export to ONNX then embed metadata
    torch.onnx.export(model, dummy_input, str(path), opset_version=12, do_constant_folding=True)

    try:
        import onnx  # type: ignore
    except Exception:
        pytest.skip("onnx not available")

    contract = provenance.runtime_contract_payload()
    manifest_base = provenance.build_artifact_manifest(model_architecture="linear", contract=contract, onnx_opset=12)
    # load the ONNX model, embed metadata, and save back
    onnx_model = onnx.load(str(path))
    provenance.embed_manifest_in_onnx_metadata(onnx_model, manifest_base)
    onnx.save(onnx_model, str(path))

    provenance.finalize_artifact_manifest(path, manifest_base)
    provenance.write_contract_sidecars(path, contract)

    sidecar = provenance.verify_artifact_manifest(path, kind="onnx", contract=contract)
    assert sidecar is not None
