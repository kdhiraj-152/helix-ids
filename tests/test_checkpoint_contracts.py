from __future__ import annotations

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
from helix_ids.models.full import HelixFullConfig, create_helix_full
from helix_ids.operations.inference_runtime import HelixInferenceRuntime


EXPECTED_FEATURE_ORDER = list(FEATURE_ORDER)



def _make_payload(
    *,
    input_dim: int = CANONICAL_INPUT_DIM,
    feature_order: list[str] = EXPECTED_FEATURE_ORDER,
    include_metadata: bool = True,
) -> dict[str, object]:
    model = create_helix_full(
        HelixFullConfig(
            input_dim=input_dim,
            binary_output_dim=CANONICAL_BINARY_CLASSES,
            family_output_dim=CANONICAL_FAMILY_CLASSES,
        )
    )
    payload: dict[str, object] = {"model_state_dict": model.state_dict()}
    # Merge canonical runtime contract fields
    contract = runtime_contract_payload()
    contract["feature_order"] = feature_order
    contract["input_dim"] = input_dim
    contract["binary_output_dim"] = CANONICAL_BINARY_CLASSES
    contract["family_output_dim"] = CANONICAL_FAMILY_CLASSES
    payload.update(contract)
    if not include_metadata:
        for k in ["schema_version", "schema_hash", "feature_order", "input_dim", "binary_output_dim", "family_output_dim"]:
            payload.pop(k, None)
    return payload


def _write_sidecars(path: Path, payload: dict[str, object]) -> None:
    import json

    contract_path = path.with_suffix(path.suffix + ".contract.json")
    feature_order_path = path.with_suffix(path.suffix + ".feature_order.json")
    schema_hash_path = path.with_suffix(path.suffix + ".schema_hash.txt")
    contract_path.write_text(json.dumps({
        "schema_version": payload.get("schema_version"),
        "schema_hash": payload.get("schema_hash"),
        "feature_order": payload.get("feature_order"),
        "input_dim": payload.get("input_dim"),
        "binary_output_dim": payload.get("binary_output_dim"),
        "family_output_dim": payload.get("family_output_dim"),
    }, indent=2), encoding="utf-8")
    feature_order_path.write_text(json.dumps(payload.get("feature_order", []), indent=2), encoding="utf-8")
    schema_hash_path.write_text(str(payload.get("schema_hash", "")) + "\n", encoding="utf-8")



def test_missing_checkpoint_metadata_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "missing-metadata.pt"
    payload = _make_payload(include_metadata=False)
    torch.save(payload, checkpoint)
    _write_sidecars(checkpoint, payload)

    with pytest.raises(AssertionError, match="Missing required checkpoint contract metadata"):
        HelixInferenceRuntime(checkpoint)



def test_legacy_19_feature_checkpoint_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "legacy-19.pt"
    legacy_order = [f"legacy_{idx}" for idx in range(19)]
    payload = _make_payload(input_dim=19, feature_order=legacy_order)
    torch.save(payload, checkpoint)
    _write_sidecars(checkpoint, payload)

    with pytest.raises(ValueError, match="Checkpoint input_dim mismatch"):
        HelixInferenceRuntime(checkpoint)



def test_contract_version_mismatch_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "bad-version.pt"
    payload = _make_payload()
    # intentionally corrupt schema_version to simulate mismatch
    payload["schema_version"] = "0.0"
    torch.save(payload, checkpoint)
    _write_sidecars(checkpoint, payload)

    with pytest.raises(AssertionError, match="schema_version mismatch"):
        HelixInferenceRuntime(checkpoint)
