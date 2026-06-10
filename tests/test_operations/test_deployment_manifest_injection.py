"""Tests for deployment manifest verification injection in freeze/train/benchmark paths.

Proves that verify_deployment_manifest() is reached when deployment.manifest.json
exists at the artifact parent directory, and that behavior is unchanged when it is absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from helix_ids.contracts import runtime_contract_payload
from helix_ids.governance import (
    ARTIFACT_MANIFEST_KEY,
    ArtifactManifestError,
    checkpoint_manifest_payload,
    write_contract_sidecars,
)
from helix_ids.governance.provenance import (
    deployment_manifest_path,
)
from helix_ids.models.full import HelixFullConfig, create_helix_full
from helix_ids.utils.export import (
    build_export_manifest,
    finalize_export_artifact,
)

# ============================================================================
# ============================================================================
# Helpers
# ============================================================================


def _make_test_checkpoint(path: Path) -> dict[str, object]:
    """Create a valid checkpoint with embedded manifest + sidecars.

    Returns manifest_base dict so callers can build a matching deployment.manifest.json.
    """
    model = create_helix_full(HelixFullConfig(input_dim=17, family_output_dim=7))
    payload: dict[str, object] = {"model_state_dict": model.state_dict()}
    contract = runtime_contract_payload()
    payload.update(contract)  # type: ignore[typeddict-item]
    manifest_base = build_export_manifest(
        contract=contract,
        model_architecture=model.__class__.__name__,
        export_config={"format": "checkpoint", "origin": "test_deployment_manifest"},
    )
    payload[ARTIFACT_MANIFEST_KEY] = checkpoint_manifest_payload(manifest_base)  # type: ignore[typeddict-item]
    torch.save(payload, path)
    sidecars = write_contract_sidecars(path, contract)
    finalize_export_artifact(path, manifest_base, sidecars=sidecars)
    return manifest_base


def _write_mismatched_deployment_manifest(checkpoint_path: Path, manifest_base: dict[str, object]) -> Path:
    """Write a deployment.manifest.json with a deliberately wrong artifact_sha256.

    When this file is found by the call site, verify_deployment_manifest will
    raise ArtifactManifestError, proving the verification path was triggered.
    """
    deploy_path = deployment_manifest_path(checkpoint_path)
    payload = {
        "artifact_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "schema_hash": str(manifest_base["schema_hash"]),
        "config_hash": str(manifest_base["export_config_hash"]),
        "dataset_hash": str(manifest_base["dataset_hash"]),
        "git_commit": str(manifest_base["git_commit"]),
        "exporter_version": str(manifest_base["exporter_version"]),
        "torch_version": str(manifest_base["torch_version"]),
        "onnx_opset": manifest_base.get("onnx_opset"),
    }
    deploy_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return deploy_path


# ============================================================================
# Baseline freeze path  (baseline_freeze.py → seal_baseline → verify_ingress_artifact)
# ============================================================================


def test_baseline_freeze_runs_deployment_manifest_verification_when_present(tmp_path: Path) -> None:
    """Proves seal_baseline triggers verify_deployment_manifest when the file exists.

    A mismatched deployment.manifest.json raises ArtifactManifestError
    from verify_deployment_manifest.
    """
    from helix_ids.operations.baseline_freeze import seal_baseline

    ckpt = tmp_path / "model.pt"
    manifest_base = _make_test_checkpoint(ckpt)
    _write_mismatched_deployment_manifest(ckpt, manifest_base)

    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    for name in ("X_train", "y_train", "X_val", "y_val"):
        np.save(artifact_dir / f"{name}.npy", np.zeros((5, 17), dtype=np.float32))
    training = tmp_path / "training.json"
    training.write_text(json.dumps({"config": {"seed": 42}}), encoding="utf-8")
    eval_report = tmp_path / "eval.json"
    eval_report.write_text(json.dumps({"nsl_kdd": {"family_macro_f1": 0.5}}), encoding="utf-8")

    with pytest.raises(ArtifactManifestError, match="artifact_sha256 mismatch"):
        seal_baseline(
            release_id="test-baseline-present",
            model_checkpoint=ckpt,
            artifact_dir=artifact_dir,
            training_report=training,
            eval_report=eval_report,
            output_root=tmp_path / "releases",
        )


def test_baseline_freeze_skips_deployment_manifest_when_absent(tmp_path: Path) -> None:
    """Proves seal_baseline succeeds when deployment.manifest.json does not exist."""
    from helix_ids.operations.baseline_freeze import seal_baseline

    ckpt = tmp_path / "model.pt"
    _make_test_checkpoint(ckpt)

    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    for name in ("X_train", "y_train", "X_val", "y_val"):
        np.save(artifact_dir / f"{name}.npy", np.zeros((5, 17), dtype=np.float32))
    training = tmp_path / "training.json"
    training.write_text(json.dumps({"config": {"seed": 42}}), encoding="utf-8")
    eval_report = tmp_path / "eval.json"
    eval_report.write_text(json.dumps({"nsl_kdd": {"family_macro_f1": 0.5}}), encoding="utf-8")

    release_dir = seal_baseline(
        release_id="test-baseline-absent",
        model_checkpoint=ckpt,
        artifact_dir=artifact_dir,
        training_report=training,
        eval_report=eval_report,
        output_root=tmp_path / "releases",
    )
    assert release_dir.exists()
    assert (release_dir / "manifest.json").exists()


# ============================================================================
# Transfer learning path  (transfer_learning.py → MultiDatasetPretrainer.load_checkpoint)
#
# MultiDatasetPretrainer.load_checkpoint():
#   1. path = Path(path)
#   2. checkpoint_preview = torch.load(path, weights_only=True)
#   3. Discover deployment.manifest.json
#   4. verify_artifact_provenance(path, ..., deployment_manifest=...)
#   5. checkpoint = torch.load(path, weights_only=False)  # accesses config, source_dims
#
# Our checkpoint lacks required keys, so we mock step 5 via torch.load.
# This lets us test step 4 (the deployment manifest injection) in isolation.
# ============================================================================


def test_transfer_learning_passes_deployment_manifest_when_present(tmp_path: Path) -> None:
    """Proves load_checkpoint forwards deployment_manifest to verify_artifact_provenance.

    We monkeypatch verify_artifact_provenance (imported directly in
    transfer_learning.py) and assert it receives deployment_manifest as a Path
    when deployment.manifest.json exists. We raise ArtifactManifestError to
    prove the mock was reached.
    """
    from helix_ids.models.adaptation.transfer_learning import (
        MultiDatasetPretrainer,
        TransferLearningConfig,
    )

    ckpt = tmp_path / "model.pt"
    manifest_base = _make_test_checkpoint(ckpt)
    _write_mismatched_deployment_manifest(ckpt, manifest_base)

    captured = {"deployment_manifest_sent": False}

    def _check_vap(path, **kwargs):
        dm = kwargs.get("deployment_manifest")
        if dm is not None and isinstance(dm, Path):
            captured["deployment_manifest_sent"] = True
        raise ArtifactManifestError("artifact_sha256 mismatch (test marker)")

    config = TransferLearningConfig()
    trainer = MultiDatasetPretrainer(config)

    with patch(
        "helix_ids.models.adaptation.transfer_learning.verify_artifact_provenance",
        side_effect=_check_vap,
    ):
        with pytest.raises(ArtifactManifestError, match="artifact_sha256 mismatch"):
            trainer.load_checkpoint(ckpt)

    assert captured["deployment_manifest_sent"], (
        "deployment_manifest was not passed to verify_artifact_provenance"
    )


def test_transfer_learning_omits_deployment_manifest_when_absent(tmp_path: Path) -> None:
    """Proves load_checkpoint passes deployment_manifest=None when file absent.

    The verification function is called with deployment_manifest=None (or not set).
    """
    from helix_ids.models.adaptation.transfer_learning import (
        MultiDatasetPretrainer,
        TransferLearningConfig,
    )

    ckpt = tmp_path / "model.pt"
    _make_test_checkpoint(ckpt)

    captured = {"deployment_manifest": "NOT_CALLED"}

    def _check_vap(path, **kwargs):
        captured["deployment_manifest"] = kwargs.get("deployment_manifest")
        return {"artifact_sha256": "", "schema_hash": "", "feature_order_hash": ""}

    def _mock_tl_torch_load(*args, **kwargs):
        return {
            "config": {"seed": 42},
            "source_dims": {"nsl-kdd": 17},
            "pretrain_history": [],
            "finetune_history": [],
            "is_pretrained": False,
            "feature_aligner_state": None,
            "dann_model_state": None,
        }

    config = TransferLearningConfig()
    trainer = MultiDatasetPretrainer(config)

    with (
        patch(
            "helix_ids.models.adaptation.transfer_learning.verify_artifact_provenance",
            side_effect=_check_vap,
        ),
        patch(
            "helix_ids.models.adaptation.transfer_learning.torch.load",
            side_effect=_mock_tl_torch_load,
        ),
    ):
        trainer.load_checkpoint(ckpt)

    assert captured["deployment_manifest"] is None, (
        f"deployment_manifest should be None when file absent, got {captured['deployment_manifest']}"
    )


# ============================================================================
# Benchmark verification path  (benchmarks.py → _verify_artifacts)
#
# _verify_artifacts catches ArtifactManifestError internally and returns a
# {"status": "fail", "artifacts": [...]} dict rather than re-raising. We verify
# that deployment manifest mismatch produces a "fail" status with the proper
# error reason in the artifact entry.
# ============================================================================


def test_benchmark_verification_fails_on_deployment_manifest_mismatch(tmp_path: Path) -> None:
    """Proves _verify_artifacts detects deployment manifest mismatch.

    The function catches ArtifactManifestError internally, so we check that
    it returns status "fail" with the deployment-manifest error embedded.
    """
    from scripts.evaluation.benchmarks import _verify_artifacts

    ckpt = tmp_path / "model.pt"
    manifest_base = _make_test_checkpoint(ckpt)
    _write_mismatched_deployment_manifest(ckpt, manifest_base)

    result = _verify_artifacts([ckpt], kind="checkpoint", manifest_mode="full")

    assert result["status"] == "fail"
    assert len(result["artifacts"]) == 1
    failed = result["artifacts"][0]
    assert failed["status"] == "fail"
    assert "artifact_sha256 mismatch" in failed["reason"]


def test_benchmark_verification_skips_deployment_manifest_when_absent(tmp_path: Path) -> None:
    """Proves _verify_artifacts succeeds when deployment.manifest.json does not exist."""
    from scripts.evaluation.benchmarks import _verify_artifacts

    ckpt = tmp_path / "model.pt"
    _make_test_checkpoint(ckpt)

    result = _verify_artifacts([ckpt], kind="checkpoint", manifest_mode="full")
    assert result["status"] == "pass"
