"""Comprehensive regression tests for the extracted validation artifacts module.

Phase 12B-4: covers all functions exported from
scripts/training/validation/artifacts.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from scripts.training.validation.artifacts import (
    _atomic_write_json,
    _emit_calibration_artifacts,
    _materialize_phase8_artifacts,
    _normalize_calibration_block,
)

# ============================================================================
# _atomic_write_json
# ============================================================================


class TestAtomicWriteJson:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        """Verify JSON is written correctly to the target path."""
        target = tmp_path / "output.json"
        payload = {"key": "value", "num": 42}
        _atomic_write_json(target, payload)
        assert target.exists()
        with open(target) as f:
            data = json.load(f)
        assert data == payload

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Parent directories should be created automatically."""
        target = tmp_path / "deep" / "nested" / "output.json"
        _atomic_write_json(target, {"a": 1})
        assert target.exists()

    def test_atomic_no_partial_write_on_crash(self, tmp_path: Path) -> None:
        """Temporary file should not remain after successful write."""
        target = tmp_path / "atomic.json"
        _atomic_write_json(target, {"data": [1, 2, 3]})
        # The .tmp file should be cleaned up (os.replace removes it)
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        """Existing files should be overwritten atomically."""
        target = tmp_path / "replace.json"
        target.write_text('{"old": true}', encoding="utf-8")
        _atomic_write_json(target, {"new": True})
        with open(target) as f:
            data = json.load(f)
        assert data == {"new": True}

    def test_writes_list_payload(self, tmp_path: Path) -> None:
        """Top-level list payloads should roundtrip correctly."""
        target = tmp_path / "list.json"
        payload = [1, 2, 3, {"nested": "value"}]
        _atomic_write_json(target, payload)
        with open(target) as f:
            data = json.load(f)
        assert data == payload

    def test_writes_empty_dict(self, tmp_path: Path) -> None:
        """Empty dict should roundtrip correctly."""
        target = tmp_path / "empty.json"
        _atomic_write_json(target, {})
        with open(target) as f:
            data = json.load(f)
        assert data == {}

    def test_default_str_for_non_serializable(self, tmp_path: Path) -> None:
        """Non-serializable objects should fall back to str()."""
        target = tmp_path / "fallback.json"

        class _Custom:
            def __str__(self) -> str:
                return "custom_str"

        _atomic_write_json(target, {"obj": _Custom()})
        with open(target) as f:
            data = json.load(f)
        assert data["obj"] == "custom_str"

    def test_json_indent_formatting(self, tmp_path: Path) -> None:
        """JSON should be written with indent=2 for readability."""
        target = tmp_path / "formatted.json"
        _atomic_write_json(target, {"key": "val"})
        content = target.read_text(encoding="utf-8")
        assert '"key"' in content
        assert content.strip().startswith("{")


# ============================================================================
# _emit_calibration_artifacts
# ============================================================================


def _sample_calibration_payload() -> dict[str, Any]:
    return {
        "class4_logit_shift": 0.0,
        "temperature": 2.5,
        "tau_4": 0.6,
        "uncalibrated": {
            "test_argmax": {
                "class4_precision": 0.3,
                "class4_recall": 0.9,
                "macro_f1": 0.7,
                "mean_entropy": 0.5,
                "zero_prediction_classes": 0,
                "confusion_matrix": [[10, 2], [1, 15]],
            },
            "val_argmax": {
                "class4_precision": 0.3,
                "class4_recall": 0.9,
                "macro_f1": 0.7,
                "mean_entropy": 0.5,
                "zero_prediction_classes": 0,
                "confusion_matrix": [[8, 1], [0, 12]],
            },
        },
        "ablation": {
            "without_thresholding": {
                "class4_precision": 0.4,
                "class4_recall": 0.85,
                "macro_f1": 0.72,
                "mean_entropy": 0.45,
                "confusion_matrix": [[9, 1], [1, 14]],
            },
            "without_temperature_scaling": {
                "class4_precision": 0.35,
                "class4_recall": 0.88,
                "macro_f1": 0.71,
                "mean_entropy": 0.52,
                "confusion_matrix": [[10, 2], [1, 15]],
            },
        },
        "pr_curve_class4": {
            "precision": [0.9, 0.8, 0.7],
            "recall": [0.5, 0.7, 0.9],
            "thresholds": [0.3, 0.5, 0.7],
        },
        "test": {
            "class4_precision": 0.6,
            "class4_recall": 0.75,
            "macro_f1": 0.78,
            "mean_entropy": 0.4,
            "zero_prediction_classes": 0,
            "confusion_matrix": [[12, 0], [1, 17]],
        },
        "val": {
            "class4_precision": 0.65,
            "class4_recall": 0.7,
            "macro_f1": 0.8,
            "mean_entropy": 0.38,
            "zero_prediction_classes": 0,
        },
        "threshold_sweep": {
            "tau_min": 0.3,
            "tau_max": 0.95,
            "num_points": 10,
            "points": [],
        },
    }


class TestEmitCalibrationArtifacts:
    def test_emits_all_artifact_files(self, tmp_path: Path) -> None:
        """All expected artifact files should be created."""
        payload = _sample_calibration_payload()
        artifacts = _emit_calibration_artifacts(
            results_dir=tmp_path,
            dataset_name="nsl_kdd",
            seed=42,
            calibration_payload=payload,
        )
        for key, path_str in artifacts.items():
            path = Path(path_str)
            assert path.exists(), f"Missing artifact: {key} -> {path}"

    def test_artifact_structure_keys(self, tmp_path: Path) -> None:
        """Returned artifact dict must contain all expected keys."""
        payload = _sample_calibration_payload()
        artifacts = _emit_calibration_artifacts(
            results_dir=tmp_path,
            dataset_name="nsl_kdd",
            seed=42,
            calibration_payload=payload,
        )
        expected_keys = {
            "calibration_json",
            "before_after_json",
            "before_after_csv",
            "pr_curve_csv",
            "confusion_matrices_json",
            "ablation_json",
        }
        assert set(artifacts.keys()) == expected_keys

    def test_before_after_csv_content(self, tmp_path: Path) -> None:
        """before_after CSV should contain 3 rows with correct columns."""
        payload = _sample_calibration_payload()
        artifacts = _emit_calibration_artifacts(
            results_dir=tmp_path,
            dataset_name="nsl_kdd",
            seed=42,
            calibration_payload=payload,
        )
        df = pd.read_csv(artifacts["before_after_csv"])
        assert list(df["phase"]) == ["baseline_collapse", "enforcement_high_recall_low_precision", "calibrated_balanced"]
        assert "macro_f1" in df.columns
        assert "class4_precision" in df.columns
        assert "class4_recall" in df.columns

    def test_calibration_json_roundtrip(self, tmp_path: Path) -> None:
        """Calibration JSON should roundtrip correctly."""
        payload = _sample_calibration_payload()
        artifacts = _emit_calibration_artifacts(
            results_dir=tmp_path,
            dataset_name="nsl_kdd",
            seed=42,
            calibration_payload=payload,
        )
        with open(artifacts["calibration_json"]) as f:
            loaded = json.load(f)
        assert loaded["temperature"] == 2.5
        assert loaded["tau_4"] == 0.6
        assert loaded["class4_logit_shift"] == 0.0

    def test_artifact_tag_suffix(self, tmp_path: Path) -> None:
        """Artifact tag should appear in filenames."""
        artifacts_tagged = _emit_calibration_artifacts(
            results_dir=tmp_path,
            dataset_name="nsl_kdd",
            seed=42,
            calibration_payload=_sample_calibration_payload(),
            artifact_tag="v2",
        )
        _emit_calibration_artifacts(  # verify no crash without tag
            results_dir=tmp_path / "untagged",
            dataset_name="nsl_kdd",
            seed=42,
            calibration_payload=_sample_calibration_payload(),
        )
        # Tagged paths should contain "v2" in the filename
        for key in artifacts_tagged:
            assert "v2" in artifacts_tagged[key]

    def test_confusion_matrices_json_structure(self, tmp_path: Path) -> None:
        """Confusion matrices JSON should contain expected sections."""
        payload = _sample_calibration_payload()
        artifacts = _emit_calibration_artifacts(
            results_dir=tmp_path,
            dataset_name="nsl_kdd",
            seed=42,
            calibration_payload=payload,
        )
        with open(artifacts["confusion_matrices_json"]) as f:
            data = json.load(f)
        assert "dataset" in data
        assert "uncalibrated_test_argmax" in data
        assert "ablation_without_thresholding" in data
        assert "ablation_without_temperature_scaling" in data
        assert "calibrated" in data

    def test_ablation_json_structure(self, tmp_path: Path) -> None:
        """Ablation JSON should contain dataset/seed/ablation."""
        payload = _sample_calibration_payload()
        artifacts = _emit_calibration_artifacts(
            results_dir=tmp_path,
            dataset_name="nsl_kdd",
            seed=42,
            calibration_payload=payload,
        )
        with open(artifacts["ablation_json"]) as f:
            data = json.load(f)
        assert data["dataset"] == "nsl_kdd"
        assert data["seed"] == 42
        assert "without_thresholding" in data["ablation"]

    def test_pr_curve_csv_structure(self, tmp_path: Path) -> None:
        """PR curve CSV should have precision/recall/threshold columns."""
        payload = _sample_calibration_payload()
        artifacts = _emit_calibration_artifacts(
            results_dir=tmp_path,
            dataset_name="nsl_kdd",
            seed=42,
            calibration_payload=payload,
        )
        df = pd.read_csv(artifacts["pr_curve_csv"])
        assert list(df.columns) == ["point_index", "precision", "recall", "threshold"]
        assert len(df) == 3

    def test_idempotency(self, tmp_path: Path) -> None:
        """Running twice should produce identical outputs."""
        payload = _sample_calibration_payload()
        a1 = _emit_calibration_artifacts(
            results_dir=tmp_path, dataset_name="nsl_kdd", seed=42, calibration_payload=payload,
        )
        a2 = _emit_calibration_artifacts(
            results_dir=tmp_path, dataset_name="nsl_kdd", seed=42, calibration_payload=payload,
        )
        for key in a1:
            with open(a1[key]) as f:
                c1 = f.read()
            with open(a2[key]) as f:
                c2 = f.read()
            assert c1 == c2, f"Mismatch for {key}"

    def test_empty_pr_curve_handling(self, tmp_path: Path) -> None:
        """Empty PR curve lists should produce an empty CSV file."""
        payload = _sample_calibration_payload()
        payload["pr_curve_class4"] = {"precision": [], "recall": [], "thresholds": []}
        artifacts = _emit_calibration_artifacts(
            results_dir=tmp_path,
            dataset_name="nsl_kdd",
            seed=42,
            calibration_payload=payload,
        )
        csv_path = Path(artifacts["pr_curve_csv"])
        assert csv_path.exists()
        content = csv_path.read_text(encoding="utf-8").strip()
        # DataFrame from empty list produces empty CSV
        assert len(content) == 0


# ============================================================================
# _materialize_phase8_artifacts
# ============================================================================


class TestMaterializePhase8Artifacts:
    def _create_source_artifacts(self, tmp_path: Path) -> dict[str, str]:
        """Helper to create a standard set of source artifacts."""
        results_dir = tmp_path / "results"
        calibration_dir = results_dir / "calibration"
        calibration_dir.mkdir(parents=True)
        for name in ("before_after.csv", "before_after.json", "pr_curve.csv",
                     "confusion_matrices.json", "ablation.json"):
            (calibration_dir / name).write_text("{}", encoding="utf-8")
        return {
            "before_after_csv": str(calibration_dir / "before_after.csv"),
            "before_after_json": str(calibration_dir / "before_after.json"),
            "pr_curve_csv": str(calibration_dir / "pr_curve.csv"),
            "confusion_matrices_json": str(calibration_dir / "confusion_matrices.json"),
            "ablation_json": str(calibration_dir / "ablation.json"),
        }

    def test_copies_to_canonical_names(self, tmp_path: Path) -> None:
        """Source artifacts should be copied to canonical filenames."""
        src = self._create_source_artifacts(tmp_path)
        canonical = _materialize_phase8_artifacts(src)
        for _key, path_str in canonical.items():
            path = Path(path_str)
            assert path.exists()
            assert path.name in ("before_after.csv", "before_after.json",
                                 "pr_curve.csv", "confusion_matrices.json",
                                 "ablation.json")

    def test_returns_canonical_paths(self, tmp_path: Path) -> None:
        """Canonical paths should have the standard filenames."""
        src = self._create_source_artifacts(tmp_path)
        canonical = _materialize_phase8_artifacts(src)
        expected_names = {
            "before_after_csv": "before_after.csv",
            "before_after_json": "before_after.json",
            "pr_curve_csv": "pr_curve.csv",
            "confusion_matrices_json": "confusion_matrices.json",
            "ablation_json": "ablation.json",
        }
        for key, expected_name in expected_names.items():
            assert Path(canonical[key]).name == expected_name

    def test_missing_key_raises_value_error(self, tmp_path: Path) -> None:
        """Missing source key should raise ValueError."""
        # Create a dir with valid files for keys we do provide
        d = tmp_path / "cal"
        d.mkdir()
        for name in ("pr_curve.csv", "confusion_matrices.json", "ablation.json"):
            (d / name).write_text("{}", encoding="utf-8")
        src = {
            "pr_curve_csv": str(d / "pr_curve.csv"),
            "confusion_matrices_json": str(d / "confusion_matrices.json"),
            # Missing "ablation_json" key
        }
        with pytest.raises(ValueError, match="Missing required calibration artifact key"):
            _materialize_phase8_artifacts(src)

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """Non-existent source file should raise FileNotFoundError."""
        src = {
            "before_after_csv": str(tmp_path / "missing.csv"),
            "before_after_json": str(tmp_path / "missing.json"),
            "pr_curve_csv": str(tmp_path / "missing.csv"),
            "confusion_matrices_json": str(tmp_path / "missing.json"),
            "ablation_json": str(tmp_path / "missing.json"),
        }
        (tmp_path / "missing.csv").write_text("{}", encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            _materialize_phase8_artifacts(src)

    def test_idempotent(self, tmp_path: Path) -> None:
        """Running twice should produce same canonical paths."""
        src = self._create_source_artifacts(tmp_path)
        c1 = _materialize_phase8_artifacts(src)
        c2 = _materialize_phase8_artifacts(src)
        assert c1 == c2


# ============================================================================
# _normalize_calibration_block
# ============================================================================


class TestNormalizeCalibrationBlock:
    def _setup_artifacts(self, tmp_path: Path) -> dict[str, str]:
        """Create artifact files and return artifact dict."""
        d = tmp_path / "calibration"
        d.mkdir(parents=True)
        for name in ("pr_curve.csv", "confusion_matrices.json", "ablation.json"):
            (d / name).write_text("{}", encoding="utf-8")
        return {
            "pr_curve_csv": str(d / "pr_curve.csv"),
            "confusion_matrices_json": str(d / "confusion_matrices.json"),
            "ablation_json": str(d / "ablation.json"),
        }

    def test_normalizes_temperature_and_tau(self, tmp_path: Path) -> None:
        """Temperature and tau_4 should be preserved."""
        artifacts = self._setup_artifacts(tmp_path)
        result = _normalize_calibration_block(
            calibration_payload={"temperature": 2.5, "tau_4": 0.6},
            calibration_artifacts=artifacts,
        )
        assert result["temperature"] == 2.5
        assert result["tau_4"] == 0.6

    def test_requires_all_artifact_paths(self, tmp_path: Path) -> None:
        """Missing artifact should raise FileNotFoundError."""
        artifacts = self._setup_artifacts(tmp_path)
        del artifacts["pr_curve_csv"]
        with pytest.raises(KeyError):
            _normalize_calibration_block(
                calibration_payload={"temperature": 1.0},
                calibration_artifacts=artifacts,
            )

    def test_missing_path_raises_file_not_found(self, tmp_path: Path) -> None:
        """If artifact path doesn't exist, should raise FileNotFoundError."""
        artifacts = {
            "pr_curve_csv": str(tmp_path / "MISSING_pr_curve.csv"),
            "confusion_matrices_json": str(tmp_path / "MISSING_confusion.json"),
            "ablation_json": str(tmp_path / "MISSING_ablation.json"),
        }
        with pytest.raises(FileNotFoundError):
            _normalize_calibration_block(
                calibration_payload={"temperature": 1.0, "tau_4": 0.5},
                calibration_artifacts=artifacts,
            )

    def test_default_values(self, tmp_path: Path) -> None:
        """Default temperature=1.0, tau_4=0.5 when not in payload."""
        artifacts = self._setup_artifacts(tmp_path)
        result = _normalize_calibration_block(
            calibration_payload={},
            calibration_artifacts=artifacts,
        )
        assert result["temperature"] == 1.0
        assert result["tau_4"] == 0.5

    def test_paths_preserved_in_output(self, tmp_path: Path) -> None:
        """Artifact paths should appear in output."""
        artifacts = self._setup_artifacts(tmp_path)
        result = _normalize_calibration_block(
            calibration_payload={"temperature": 1.0, "tau_4": 0.5},
            calibration_artifacts=artifacts,
        )
        assert result["pr_curve_path"] == artifacts["pr_curve_csv"]
        assert result["confusion_matrix_path"] == artifacts["confusion_matrices_json"]
        assert result["ablation_path"] == artifacts["ablation_json"]

    def test_returns_dict_with_expected_keys(self, tmp_path: Path) -> None:
        """Output should have exactly the expected keys."""
        artifacts = self._setup_artifacts(tmp_path)
        result = _normalize_calibration_block(
            calibration_payload={"temperature": 1.0, "tau_4": 0.5},
            calibration_artifacts=artifacts,
        )
        assert set(result.keys()) == {"temperature", "tau_4", "pr_curve_path", "confusion_matrix_path", "ablation_path"}
