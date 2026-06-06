from __future__ import annotations

import pytest

from helix_ids.governance import provenance
from helix_ids.governance.parameters import allow_legacy_artifacts, is_production_runtime


def _clear_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HELIX_RUNTIME_ENV", "HELIX_ENV", "HELIX_DEPLOY_ENV"):
        monkeypatch.delenv(name, raising=False)


def test_is_production_runtime_defaults_false(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_runtime_env(monkeypatch)
    assert is_production_runtime() is False


def test_is_production_runtime_detects_production(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("HELIX_RUNTIME_ENV", "production")
    assert is_production_runtime() is True


def test_allow_legacy_artifacts_defaults_false(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_runtime_env(monkeypatch)
    monkeypatch.delenv("HELIX_ALLOW_LEGACY_ARTIFACTS", raising=False)
    assert allow_legacy_artifacts() is False


def test_allow_legacy_artifacts_allows_in_non_production(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("HELIX_ALLOW_LEGACY_ARTIFACTS", "1")
    monkeypatch.setenv("HELIX_ENV", "dev")
    assert allow_legacy_artifacts() is True


def test_allow_legacy_artifacts_rejects_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("HELIX_ALLOW_LEGACY_ARTIFACTS", "1")
    monkeypatch.setenv("HELIX_ENV", "production")
    with pytest.raises(AssertionError, match="Legacy artifact allowance is forbidden"):
        allow_legacy_artifacts()


def test_allow_legacy_manifest_rejects_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("HELIX_ALLOW_LEGACY_MANIFEST", "1")
    monkeypatch.setenv("HELIX_ENV", "production")
    with pytest.raises(AssertionError, match="Legacy manifest allowance is forbidden"):
        provenance._allow_legacy_manifest()


def test_verify_ingress_artifact_blocks_local_dev_in_production(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("HELIX_ENV", "production")
    with pytest.raises(AssertionError, match="Legacy artifact allowance is forbidden"):
        provenance.verify_ingress_artifact(
            tmp_path / "artifact.pt",
            kind="checkpoint",
            allow_legacy_local_dev=True,
        )
