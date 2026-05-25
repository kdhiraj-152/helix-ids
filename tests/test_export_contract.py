from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from helix_ids.contracts import (
    CANONICAL_BINARY_CLASSES,
    CANONICAL_FAMILY_CLASSES,
    CANONICAL_INPUT_DIM,
    runtime_contract_payload,
)
from helix_ids.data.feature_harmonization import FEATURE_ORDER
from helix_ids.models.full import HelixFullConfig, create_helix_full
from helix_ids.operations.inference_runtime import HelixInferenceRuntime


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
    torch.save(payload, checkpoint)
    # write sidecars
    import json

    (checkpoint.with_suffix(checkpoint.suffix + ".contract.json")).write_text(json.dumps(contract, indent=2), encoding="utf-8")
    (checkpoint.with_suffix(checkpoint.suffix + ".feature_order.json")).write_text(json.dumps(contract["feature_order"], indent=2), encoding="utf-8")
    (checkpoint.with_suffix(checkpoint.suffix + ".schema_hash.txt")).write_text(str(contract["schema_hash"]) + "\n", encoding="utf-8")
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
