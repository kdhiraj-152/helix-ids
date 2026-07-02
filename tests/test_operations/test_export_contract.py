from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from helix_ids.contracts import (
    CANONICAL_BINARY_CLASSES,
    CANONICAL_FAMILY_CLASSES,
    CANONICAL_INPUT_DIM,
    runtime_contract_payload,
)
from helix_ids.data.feature_harmonization import FEATURE_ORDER
from helix_ids.governance import (
    ARTIFACT_MANIFEST_KEY,
    artifact_manifest_path,
    checkpoint_manifest_payload,
    read_embedded_manifest,
    write_contract_sidecars,
)
from helix_ids.models.full import HelixFullConfig, create_helix_full
from helix_ids.operations.inference_runtime import HelixInferenceRuntime
from helix_ids.utils.export import (
    build_export_manifest,
    finalize_export_artifact,
    verify_export_artifact,
)

EXPECTED_FEATURE_ORDER = list(FEATURE_ORDER)



def _make_runtime(tmp_path: Path) -> HelixInferenceRuntime:
    model = create_helix_full(
        HelixFullConfig(
            input_dim=CANONICAL_INPUT_DIM,
            binary_output_dim=CANONICAL_BINARY_CLASSES,
            family_output_dim=CANONICAL_FAMILY_CLASSES,
        )
    )
    checkpoint = tmp_path / "canonical.pt"
    payload = {"model_state_dict": model.state_dict()}
    contract = runtime_contract_payload()
    contract["feature_order"] = EXPECTED_FEATURE_ORDER
    contract["input_dim"] = CANONICAL_INPUT_DIM
    contract["binary_output_dim"] = CANONICAL_BINARY_CLASSES
    contract["family_output_dim"] = CANONICAL_FAMILY_CLASSES
    payload.update(contract)
    manifest_base = build_export_manifest(
        contract=contract,
        model_architecture=model.__class__.__name__,
        export_config={"format": "checkpoint", "origin": "test_export_contract"},
    )
    payload[ARTIFACT_MANIFEST_KEY] = checkpoint_manifest_payload(manifest_base)
    torch.save(payload, checkpoint)
    sidecars = write_contract_sidecars(checkpoint, contract)
    finalize_export_artifact(checkpoint, manifest_base, sidecars=sidecars)
    verify_export_artifact(
        checkpoint,
        kind="checkpoint",
        contract=contract,
        embedded_manifest=checkpoint_manifest_payload(manifest_base),
    )
    return HelixInferenceRuntime(checkpoint)



def test_torchscript_and_onnx_exports_include_contract_sidecars(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)

    torchscript_path = runtime.export_torchscript(tmp_path / "helix_ids.torchscript.pt")
    torchscript_contract_path = torchscript_path.with_suffix(torchscript_path.suffix + ".contract.json")
    assert torchscript_contract_path.exists()

    torchscript_contract = json.loads(torchscript_contract_path.read_text(encoding="utf-8"))
    assert torchscript_contract["input_dim"] == CANONICAL_INPUT_DIM
    assert torchscript_contract["binary_output_dim"] == CANONICAL_BINARY_CLASSES
    assert torchscript_contract["family_output_dim"] == CANONICAL_FAMILY_CLASSES
    assert torchscript_contract["feature_order"] == EXPECTED_FEATURE_ORDER
    assert torchscript_contract["schema_version"] == runtime._contract_metadata()["schema_version"]

    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError:
        pytest.skip("onnx/onnxruntime not installed")

    onnx_path = runtime.export_onnx(tmp_path / "helix_ids.onnx")
    onnx_contract_path = onnx_path.with_suffix(onnx_path.suffix + ".contract.json")
    assert onnx_contract_path.exists()

    onnx_contract = json.loads(onnx_contract_path.read_text(encoding="utf-8"))
    assert onnx_contract == torchscript_contract

    torchscript_manifest_path = artifact_manifest_path(torchscript_path)
    onnx_manifest_path = artifact_manifest_path(onnx_path)
    assert torchscript_manifest_path.exists()
    assert onnx_manifest_path.exists()



def test_service_contract_matches_runtime_metadata(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    service_contract = {
        "input": {
            "shape": ["batch", runtime.model.input_dim],
            "feature_order": runtime.feature_order,
            "schema_hash": runtime.schema_hash,
            "schema_version": runtime.schema_version,
            "input_dim": CANONICAL_INPUT_DIM,
            "binary_output_dim": CANONICAL_BINARY_CLASSES,
            "family_output_dim": CANONICAL_FAMILY_CLASSES,
        },
        "canonical_contract": runtime._contract_metadata(),
    }

    assert service_contract["input"]["shape"] == ["batch", CANONICAL_INPUT_DIM]
    assert service_contract["input"]["feature_order"] == EXPECTED_FEATURE_ORDER
    assert service_contract["canonical_contract"]["feature_order"] == EXPECTED_FEATURE_ORDER
    assert service_contract["canonical_contract"]["input_dim"] == CANONICAL_INPUT_DIM


def test_checkpoint_manifest_chain_is_embedded(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    manifest_path = artifact_manifest_path(runtime.checkpoint_path)
    assert manifest_path.exists()
    embedded = read_embedded_manifest(runtime.checkpoint_path, kind="checkpoint")
    assert embedded is not None
