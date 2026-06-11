"""Tests for HELIX-IDS export infrastructure (non-ONNX-dependent paths)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from helix_ids.utils.export import (
    DEFAULT_THREAT_WEIGHTS,
    HELIX_CLASSES,
    HELIX_VARIANT_SIZES,
    ExportMetadata,
    build_export_manifest,
    check_onnx_dependencies,
)


class TestConstants:
    """Test module-level constants."""

    def test_helix_variant_sizes(self):
        assert HELIX_VARIANT_SIZES["nano"] == 30
        assert HELIX_VARIANT_SIZES["lite"] == 200
        assert HELIX_VARIANT_SIZES["full"] == 2000

    def test_helix_classes(self):
        expected = ["Normal", "DoS", "Probe", "R2L", "U2R", "Generic", "Backdoor"]
        assert HELIX_CLASSES == expected

    def test_default_threat_weights(self):
        assert DEFAULT_THREAT_WEIGHTS["Normal"] == 1.0
        assert DEFAULT_THREAT_WEIGHTS["DoS"] == 2.0
        assert DEFAULT_THREAT_WEIGHTS["Probe"] == 2.5
        assert DEFAULT_THREAT_WEIGHTS["R2L"] == 4.0
        assert DEFAULT_THREAT_WEIGHTS["U2R"] == 5.0
        assert DEFAULT_THREAT_WEIGHTS["Generic"] == 3.5
        assert DEFAULT_THREAT_WEIGHTS["Backdoor"] == 5.5

    def test_default_threat_weights_sum_positive(self):
        total = sum(DEFAULT_THREAT_WEIGHTS.values())
        assert total > 0


class TestExportMetadata:
    """Test ExportMetadata dataclass."""

    def test_default_initialization(self):
        meta = ExportMetadata()
        assert meta.model_name == "HelixIDS-Full"
        assert meta.variant == "lite"
        assert meta.input_names == ["input"]
        assert meta.output_names == ["binary", "family"]
        assert meta.classes == HELIX_CLASSES
        assert meta.threat_weights == DEFAULT_THREAT_WEIGHTS
        assert meta.target_size_kb == 200.0
        assert meta.exported_at != ""

    def test_custom_initialization(self):
        meta = ExportMetadata(
            model_name="CustomModel",
            variant="nano",
            input_dim=16,
            num_classes=3,
        )
        assert meta.model_name == "CustomModel"
        assert meta.variant == "nano"
        assert meta.input_dim == 16
        assert meta.num_classes == 3

    def test_to_dict(self):
        meta = ExportMetadata(model_name="TestModel")
        d = meta.to_dict()
        assert isinstance(d, dict)
        assert d["model_name"] == "TestModel"
        assert "variant" in d
        assert "version" in d
        assert "exported_at" in d

    def test_to_json_and_from_json(self, tmp_path):
        meta = ExportMetadata(model_name="RoundTrip")
        path = tmp_path / "metadata.json"
        meta.to_json(path)
        assert path.exists()
        loaded = ExportMetadata.from_json(path)
        assert loaded.model_name == "RoundTrip"
        assert loaded.variant == meta.variant
        assert loaded.input_names == ["input"]

    def test_from_json_with_data(self, tmp_path):
        data = {
            "model_name": "JSONTest",
            "variant": "full",
            "exported_at": "2025-01-01T00:00:00",
        }
        path = tmp_path / "meta.json"
        with open(path, "w") as f:
            json.dump(data, f)
        meta = ExportMetadata.from_json(path)
        assert meta.model_name == "JSONTest"
        assert meta.variant == "full"

    def test_input_names_default(self):
        meta = ExportMetadata()
        assert meta.input_names == ["input"]

    def test_output_names_default(self):
        meta = ExportMetadata()
        assert meta.output_names == ["binary", "family"]

    def test_threat_weights_copied_not_shared(self):
        meta1 = ExportMetadata()
        meta2 = ExportMetadata()
        assert meta1.threat_weights is not meta2.threat_weights
        assert meta1.threat_weights == meta2.threat_weights


class TestCheckOnnxDependencies:
    """Test ONNX dependency checking."""

    def test_onnx_not_available_returns_false(self):
        with patch("helix_ids.utils.export.ONNX_AVAILABLE", False):
            available, msg = check_onnx_dependencies()
            assert not available
            assert "ONNX is not installed" in msg

    def test_onnx_available_runtime_not_required(self):
        with patch("helix_ids.utils.export.ONNX_AVAILABLE", True):
            available, msg = check_onnx_dependencies(require_runtime=False)
            assert available
            assert "available" in msg

    def test_onnx_available_runtime_required_no_runtime(self):
        with patch("helix_ids.utils.export.ONNX_AVAILABLE", True):
            with patch("helix_ids.utils.export.ONNXRUNTIME_AVAILABLE", False):
                available, msg = check_onnx_dependencies(require_runtime=True)
                assert not available
                assert "ONNX Runtime is not installed" in msg

    def test_both_available(self):
        with patch("helix_ids.utils.export.ONNX_AVAILABLE", True):
            with patch("helix_ids.utils.export.ONNXRUNTIME_AVAILABLE", True):
                available, msg = check_onnx_dependencies(require_runtime=True)
                assert available
                assert "available" in msg


class TestBuildExportManifest:
    """Test build_export_manifest function."""

    def _minimal_contract(self) -> dict:
        return {
            "schema_version": "2.0",
            "schema_hash": "abc123",
            "binary_output_dim": 2,
            "family_output_dim": 7,
            "input_dim": 32,
            "feature_order": ["duration", "protocol_type", "service", "flag"],
            "feature_order_hash": "xyz789",
        }

    def test_basic_manifest(self):
        manifest = build_export_manifest(
            contract=self._minimal_contract(),
            model_architecture="TestModel",
            export_config={"format": "checkpoint"},
        )
        assert isinstance(manifest, dict)
        assert manifest["model_architecture"] == "TestModel"

    def test_manifest_with_provenance(self):
        contract = self._minimal_contract()
        contract["feature_order"] = ["f1", "f2"]
        manifest = build_export_manifest(
            contract=contract,
            model_architecture="HelixFull",
            export_config={"format": "onnx"},
            git_commit="abc123",
            exporter_version="2.0.0",
        )
        assert manifest["model_architecture"] == "HelixFull"
        assert manifest["git_commit"] == "abc123"
        assert manifest["exporter_version"] == "2.0.0"

    def test_manifest_default_git_commit(self):
        manifest = build_export_manifest(
            contract=self._minimal_contract(),
            model_architecture="Custom",
            export_config={},
        )
        assert manifest["model_architecture"] == "Custom"

    def test_manifest_default_exporter_version(self):
        manifest = build_export_manifest(
            contract=self._minimal_contract(),
            model_architecture="Test",
            export_config={},
        )
        assert "exporter_version" in manifest
        assert manifest["exporter_version"] is not None


class TestFinalizeVerifyExportArtifact:
    """Test finalize/verify with non-existent paths (no sidecars)."""

    _minimal_contract = {
        "schema_version": "2.0",
        "schema_hash": "abc123",
        "binary_output_dim": 2,
        "family_output_dim": 7,
        "input_dim": 32,
        "feature_order": ["duration", "protocol_type", "service", "flag"],
        "feature_order_hash": "xyz789",
    }

    def test_finalize_nonexistent_path(self):
        from helix_ids.utils.export import finalize_export_artifact

        none_path = Path("/tmp/nonexistent_checkpoint_12345.pt")
        manifest = build_export_manifest(
            contract=self._minimal_contract,
            model_architecture="TestModel",
            export_config={"format": "test"},
        )
        # Should not crash — handles missing paths gracefully
        with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
            finalize_export_artifact(
                none_path,
                manifest,
                sidecars={},
            )
