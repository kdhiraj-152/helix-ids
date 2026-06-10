from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from helix_ids.models.full import HelixFullConfig, create_helix_full
from helix_ids.operations.baseline_freeze import seal_baseline


def _make_checkpoint(path: Path) -> None:
    model = create_helix_full(HelixFullConfig(input_dim=17, family_output_dim=7))

    from helix_ids.contracts import runtime_contract_payload

    payload = {"model_state_dict": model.state_dict()}
    contract = runtime_contract_payload()
    payload.update(contract)
    from helix_ids.governance import (
        ARTIFACT_MANIFEST_KEY,
        checkpoint_manifest_payload,
        write_contract_sidecars,
    )
    from helix_ids.utils.export import build_export_manifest, finalize_export_artifact
    manifest_base = build_export_manifest(
        contract=contract,
        model_architecture=model.__class__.__name__,
        export_config={"format": "checkpoint", "origin": "test_baseline_freeze"},
    )
    payload[ARTIFACT_MANIFEST_KEY] = checkpoint_manifest_payload(manifest_base)
    torch.save(payload, path)
    sidecars = write_contract_sidecars(path, contract)
    finalize_export_artifact(path, manifest_base, sidecars=sidecars)


def test_seal_baseline_creates_manifest_and_core_artifacts(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir(parents=True)
    np.save(artifact_dir / "X_train.npy", np.zeros((10, 17), dtype=np.float32))
    np.save(artifact_dir / "y_train.npy", np.zeros((10,), dtype=np.int64))
    np.save(artifact_dir / "X_val.npy", np.zeros((5, 17), dtype=np.float32))
    np.save(artifact_dir / "y_val.npy", np.zeros((5,), dtype=np.int64))

    ckpt = tmp_path / "model.pt"
    _make_checkpoint(ckpt)

    training = tmp_path / "training.json"
    eval_report = tmp_path / "eval.json"
    training.write_text(json.dumps({"config": {"seed": 42}}), encoding="utf-8")
    eval_report.write_text(json.dumps({"nsl_kdd": {"family_macro_f1": 0.5}}), encoding="utf-8")

    release_dir = seal_baseline(
        release_id="helix_ids_v1.0-test",
        model_checkpoint=ckpt,
        artifact_dir=artifact_dir,
        training_report=training,
        eval_report=eval_report,
        output_root=tmp_path / "releases",
    )

    assert (release_dir / "manifest.json").exists()
    assert (release_dir / "checkpoint" / ckpt.name).exists()
    assert (release_dir / "config" / "config_snapshot.json").exists()
    assert (release_dir / "dataset" / "dataset_hash_manifest.json").exists()
    assert (release_dir / "splits" / "split_indices_manifest.json").exists()
    assert (release_dir / "metrics" / "metric_report.json").exists()
