from __future__ import annotations

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
from helix_ids.governance import (
    ARTIFACT_MANIFEST_KEY,
    checkpoint_manifest_payload,
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


class _BadBinaryModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_dim = CANONICAL_INPUT_DIM

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = int(x.shape[0])
        return torch.zeros((batch, 3), dtype=torch.float32, device=x.device), torch.zeros(
            (batch, CANONICAL_FAMILY_CLASSES), dtype=torch.float32, device=x.device
        )


class _BadFamilyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_dim = CANONICAL_INPUT_DIM

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = int(x.shape[0])
        return torch.zeros((batch, CANONICAL_BINARY_CLASSES), dtype=torch.float32, device=x.device), torch.zeros(
            (batch, 6), dtype=torch.float32, device=x.device
        )



def _write_checkpoint(path: Path, *, feature_order: list[str] = EXPECTED_FEATURE_ORDER, input_dim: int = CANONICAL_INPUT_DIM) -> None:
    model = create_helix_full(HelixFullConfig(input_dim=input_dim, binary_output_dim=CANONICAL_BINARY_CLASSES, family_output_dim=CANONICAL_FAMILY_CLASSES))
    payload = {"model_state_dict": model.state_dict()}
    contract = runtime_contract_payload()
    contract["feature_order"] = feature_order
    contract["input_dim"] = input_dim
    contract["binary_output_dim"] = CANONICAL_BINARY_CLASSES
    contract["family_output_dim"] = CANONICAL_FAMILY_CLASSES
    payload.update(contract)
    manifest_base = build_export_manifest(
        contract=contract,
        model_architecture=model.__class__.__name__,
        export_config={"format": "checkpoint", "origin": "test_runtime_invariants"},
    )
    payload[ARTIFACT_MANIFEST_KEY] = checkpoint_manifest_payload(manifest_base)
    torch.save(payload, path)
    sidecars = write_contract_sidecars(path, contract)
    finalize_export_artifact(path, manifest_base, sidecars=sidecars)
    verify_export_artifact(
        path,
        kind="checkpoint",
        contract=contract,
        embedded_manifest=checkpoint_manifest_payload(manifest_base),
    )



def test_runtime_accepts_canonical_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "canonical.pt"
    _write_checkpoint(checkpoint)

    runtime = HelixInferenceRuntime(checkpoint)
    assert runtime.model.input_dim == CANONICAL_INPUT_DIM
    assert runtime.feature_order == EXPECTED_FEATURE_ORDER
    assert runtime.schema_version == runtime._contract_metadata()["schema_version"]



def test_runtime_rejects_reordered_feature_order(tmp_path: Path) -> None:
    checkpoint = tmp_path / "reordered.pt"
    with pytest.raises(AssertionError, match="canonical feature order"):
        _write_checkpoint(checkpoint, feature_order=list(reversed(EXPECTED_FEATURE_ORDER)))



def test_runtime_rejects_invalid_binary_logits_shape(tmp_path: Path) -> None:
    checkpoint = tmp_path / "canonical.pt"
    _write_checkpoint(checkpoint)
    runtime = HelixInferenceRuntime(checkpoint)
    runtime.model = _BadBinaryModel()

    with pytest.raises(RuntimeError, match="binary logits mismatch"):
        runtime.predict(np.zeros((2, CANONICAL_INPUT_DIM), dtype=np.float32))



def test_runtime_rejects_invalid_family_logits_shape(tmp_path: Path) -> None:
    checkpoint = tmp_path / "canonical.pt"
    _write_checkpoint(checkpoint)
    runtime = HelixInferenceRuntime(checkpoint)
    runtime.model = _BadFamilyModel()

    with pytest.raises(RuntimeError, match="family logits mismatch"):
        runtime.predict(np.zeros((2, CANONICAL_INPUT_DIM), dtype=np.float32))
