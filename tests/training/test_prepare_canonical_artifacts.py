"""Tests for canonical artifact preparation preflight checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.training.prepare_canonical_artifacts import (
    REQUIRED_ARTIFACTS,
    REQUIRED_CICIDS_ARTIFACTS,
    _check_raw_inputs,
    _verify_artifacts,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name == "canonical_contract.json":
        import json

        from helix_ids.contracts import runtime_contract_payload

        path.write_text(json.dumps(runtime_contract_payload(), indent=2), encoding="utf-8")
    else:
        path.write_text("ok", encoding="utf-8")


def test_check_raw_inputs_fails_when_required_files_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Raw dataset prerequisite check failed"):
        _check_raw_inputs(tmp_path, require_cicids=True)


def test_check_raw_inputs_passes_with_required_files(tmp_path: Path) -> None:
    _touch(tmp_path / "data" / "nsl_kdd" / "raw" / "KDDTrain+.txt")
    _touch(tmp_path / "data" / "unsw_nb15" / "raw" / "UNSW_NB15_training-set.csv")
    _touch(tmp_path / "data" / "cicids2018" / "raw" / "Wednesday.csv")

    _check_raw_inputs(tmp_path, require_cicids=True)


def test_verify_artifacts_fails_for_missing_files(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "data" / "processed" / "multi_dataset_v1"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="Processed artifact check failed"):
        _verify_artifacts(artifact_dir, require_cicids=True)


def test_verify_artifacts_fails_when_meta_not_validated(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "data" / "processed" / "multi_dataset_v1"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    for name in REQUIRED_ARTIFACTS + REQUIRED_CICIDS_ARTIFACTS:
        if name == "meta.json":
            continue
        _touch(artifact_dir / name)

    (artifact_dir / "meta.json").write_text(json.dumps({"validated": False}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="learnability contract is not validated"):
        _verify_artifacts(artifact_dir, require_cicids=True)


def test_verify_artifacts_passes_with_minimum_required_set(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "data" / "processed" / "multi_dataset_v1"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    for name in REQUIRED_ARTIFACTS + REQUIRED_CICIDS_ARTIFACTS:
        if name == "meta.json":
            continue
        _touch(artifact_dir / name)

    (artifact_dir / "meta.json").write_text(json.dumps({"validated": True}), encoding="utf-8")

    _verify_artifacts(artifact_dir, require_cicids=True)
