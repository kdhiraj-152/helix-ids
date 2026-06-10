from __future__ import annotations

import os
from pathlib import Path

import pytest

from helix_ids.governance import ArtifactManifestError
from helix_ids.governance.lifecycle_verifier import (
    clone_lifecycle_artifacts,
    create_lifecycle_artifacts,
    run_lifecycle_verification,
    tamper_artifact_hash,
    tamper_contract_version,
    tamper_deleted_manifest,
    tamper_embedded_and_sidecar_mismatch,
    tamper_embedded_sidecar_mismatch,
    tamper_exporter_version,
    tamper_extra_feature_sidecar,
    tamper_manifest_replay,
    tamper_missing_feature_sidecar,
    tamper_provenance_chain,
    tamper_reordered_feature_sidecar,
    tamper_schema_hash,
    tamper_sidecar_manifest_mismatch,
    verify_lifecycle_artifacts,
)

try:
    import onnx  # noqa: F401
    import onnxruntime  # noqa: F401
except ImportError:
    if os.getenv("CI", "").strip().lower() in {"1", "true", "yes"}:
        pytest.fail("Lifecycle tests skipped in CI")
    pytest.skip("onnx/onnxruntime not installed", allow_module_level=True)


@pytest.fixture(scope="module")
def lifecycle_artifacts(tmp_path_factory: pytest.TempPathFactory):
    workdir = tmp_path_factory.mktemp("lifecycle")
    return create_lifecycle_artifacts(workdir, require_onnx=True)


@pytest.fixture()
def tamper_workspace(tmp_path: Path) -> Path:
    return tmp_path


def _clone(artifacts, workdir: Path):
    return clone_lifecycle_artifacts(artifacts, workdir)


def test_lifecycle_verification_runs(tmp_path: Path) -> None:
    summary = run_lifecycle_verification(tmp_path, require_onnx=True)
    assert summary["parity_ok"] is True
    assert summary["artifacts"]["checkpoint"].exists()
    assert summary["artifacts"]["torchscript"].exists()
    assert summary["artifacts"]["onnx"].exists()


def test_lifecycle_rejects_legacy_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HELIX_ALLOW_LEGACY_ARTIFACTS", "1")
    with pytest.raises(AssertionError, match="Legacy artifact allowance is forbidden"):
        run_lifecycle_verification(tmp_path, require_onnx=True)


def test_lifecycle_rejects_deleted_manifest(lifecycle_artifacts, tamper_workspace: Path) -> None:
    artifacts = _clone(lifecycle_artifacts, tamper_workspace)
    tamper_deleted_manifest(artifacts, kind="checkpoint")
    with pytest.raises(ArtifactManifestError, match="Missing artifact manifest sidecar"):
        verify_lifecycle_artifacts(artifacts, require_onnx=True)


def test_lifecycle_rejects_reordered_features(lifecycle_artifacts, tamper_workspace: Path) -> None:
    artifacts = _clone(lifecycle_artifacts, tamper_workspace)
    tamper_reordered_feature_sidecar(artifacts, kind="checkpoint")
    with pytest.raises(ArtifactManifestError):
        verify_lifecycle_artifacts(artifacts, require_onnx=True)


def test_lifecycle_rejects_modified_schema_hash(lifecycle_artifacts, tamper_workspace: Path) -> None:
    artifacts = _clone(lifecycle_artifacts, tamper_workspace)
    tamper_schema_hash(artifacts, kind="checkpoint")
    with pytest.raises(ArtifactManifestError, match="schema_hash"):
        verify_lifecycle_artifacts(artifacts, require_onnx=True)


def test_lifecycle_rejects_stale_contract_version(lifecycle_artifacts, tamper_workspace: Path) -> None:
    artifacts = _clone(lifecycle_artifacts, tamper_workspace)
    tamper_contract_version(artifacts, kind="checkpoint")
    with pytest.raises(ArtifactManifestError, match="contract_version"):
        verify_lifecycle_artifacts(artifacts, require_onnx=True)


def test_lifecycle_rejects_corrupted_artifact_hash(lifecycle_artifacts, tamper_workspace: Path) -> None:
    artifacts = _clone(lifecycle_artifacts, tamper_workspace)
    tamper_artifact_hash(artifacts, kind="checkpoint")
    with pytest.raises(ArtifactManifestError, match="Artifact checksum mismatch"):
        verify_lifecycle_artifacts(artifacts, require_onnx=True)


def test_lifecycle_rejects_exporter_version_mismatch(lifecycle_artifacts, tamper_workspace: Path) -> None:
    artifacts = _clone(lifecycle_artifacts, tamper_workspace)
    tamper_exporter_version(artifacts, kind="checkpoint")
    with pytest.raises(RuntimeError, match="exporter_version mismatch"):
        verify_lifecycle_artifacts(artifacts, require_onnx=True)


def test_lifecycle_rejects_mutated_provenance_chain(lifecycle_artifacts, tamper_workspace: Path) -> None:
    artifacts = _clone(lifecycle_artifacts, tamper_workspace)
    tamper_provenance_chain(artifacts, kind="checkpoint")
    with pytest.raises(ArtifactManifestError, match="Provenance chain checksum mismatch"):
        verify_lifecycle_artifacts(artifacts, require_onnx=True)


def test_lifecycle_rejects_embedded_sidecar_mismatch(lifecycle_artifacts, tamper_workspace: Path) -> None:
    artifacts = _clone(lifecycle_artifacts, tamper_workspace)
    tamper_embedded_sidecar_mismatch(artifacts, kind="checkpoint")
    with pytest.raises(ArtifactManifestError, match="Embedded manifest mismatch"):
        verify_lifecycle_artifacts(artifacts, require_onnx=True)


def test_lifecycle_rejects_sidecar_only_mismatch(lifecycle_artifacts, tamper_workspace: Path) -> None:
    artifacts = _clone(lifecycle_artifacts, tamper_workspace)
    tamper_sidecar_manifest_mismatch(artifacts, kind="checkpoint")
    with pytest.raises(ArtifactManifestError, match="Embedded manifest mismatch"):
        verify_lifecycle_artifacts(artifacts, require_onnx=True)


def test_lifecycle_rejects_embedded_and_sidecar_divergence(lifecycle_artifacts, tamper_workspace: Path) -> None:
    artifacts = _clone(lifecycle_artifacts, tamper_workspace)
    tamper_embedded_and_sidecar_mismatch(artifacts, kind="checkpoint")
    with pytest.raises(ArtifactManifestError, match="Embedded manifest mismatch"):
        verify_lifecycle_artifacts(artifacts, require_onnx=True)


def test_lifecycle_rejects_missing_feature(lifecycle_artifacts, tamper_workspace: Path) -> None:
    artifacts = _clone(lifecycle_artifacts, tamper_workspace)
    tamper_missing_feature_sidecar(artifacts, kind="checkpoint")
    with pytest.raises(ArtifactManifestError):
        verify_lifecycle_artifacts(artifacts, require_onnx=True)


def test_lifecycle_rejects_extra_feature(lifecycle_artifacts, tamper_workspace: Path) -> None:
    artifacts = _clone(lifecycle_artifacts, tamper_workspace)
    tamper_extra_feature_sidecar(artifacts, kind="checkpoint")
    with pytest.raises(ArtifactManifestError):
        verify_lifecycle_artifacts(artifacts, require_onnx=True)


def test_lifecycle_rejects_manifest_replay(lifecycle_artifacts, tamper_workspace: Path) -> None:
    artifacts = _clone(lifecycle_artifacts, tamper_workspace)
    tamper_manifest_replay(artifacts, kind="checkpoint")
    with pytest.raises(ArtifactManifestError, match="git_commit mismatch"):
        verify_lifecycle_artifacts(artifacts, require_onnx=True)
