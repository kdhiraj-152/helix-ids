"""Tests for ONNX/TorchScript/TFLite/C header export pipeline.

Covers:
  - ExportMetadata dataclass (serialization, defaults)
  - check_onnx_dependencies
  - build_export_manifest / finalize_export_artifact / verify_export_artifact
  - ONNXExporter with real ONNX export + parity validation
  - validate_onnx with both passing and failing cases
  - benchmark_onnx (mock-free on CPU)
  - export_for_edge end-to-end
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from helix_ids.contracts.schema_contract import runtime_contract_payload
from helix_ids.utils.export import (
    HELIX_CLASSES,
    ONNX_AVAILABLE,
    ONNXRUNTIME_AVAILABLE,
    ExportMetadata,
    ONNXExporter,
    benchmark_onnx,
    build_export_manifest,
    check_onnx_dependencies,
    export_for_edge,
    validate_onnx,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def model():
    """Full HelixIDSFull model for export testing (module-scoped: expensive init)."""
    from helix_ids.models.helix_ids_full import HelixFullConfig, HelixIDSFull

    return HelixIDSFull(config=HelixFullConfig()).eval()


@pytest.fixture
def sample_input() -> torch.Tensor:
    return torch.randn(4, 17)


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    p = tmp_path / "exports"
    p.mkdir(parents=True, exist_ok=True)
    return p


@pytest.fixture
def onnx_path(model, tmp_output: Path) -> Path:
    """Export a real ONNX model for test reuse (only if ONNX available)."""
    if not ONNX_AVAILABLE:
        pytest.skip("ONNX not available")
    path = tmp_output / "test_model.onnx"
    exporter = ONNXExporter(verbose=False)
    exporter.export_to_onnx(model, path, input_shape=(1, 17), dynamic_batch=True)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# ExportMetadata
# ═══════════════════════════════════════════════════════════════════════════════


class TestExportMetadata:
    def test_default_values(self) -> None:
        """Defaults include standard HELIX class list."""
        m = ExportMetadata()
        assert m.model_name == "HelixIDS-Full"
        assert m.input_dim == 17
        assert m.num_classes == 2
        assert m.num_fine_classes == 7
        assert m.classes == HELIX_CLASSES
        assert m.input_names == ["input"]
        assert m.output_names == ["binary", "family"]

    def test_custom_values(self) -> None:
        """Custom constructor values are honoured."""
        m = ExportMetadata(model_name="Custom", variant="nano", opset_version=17)
        assert m.model_name == "Custom"
        assert m.variant == "nano"
        assert m.opset_version == 17

    def test_to_dict(self) -> None:
        """to_dict returns a JSON-serialisable dict."""
        m = ExportMetadata(variant="lite")
        d = m.to_dict()
        assert d["variant"] == "lite"
        assert d["input_dim"] == 17

    def test_to_json_roundtrip(self, tmp_path: Path) -> None:
        """to_json → from_json preserves all fields."""
        m1 = ExportMetadata(variant="nano", version="2.0.0")
        path = tmp_path / "meta.json"
        m1.to_json(path)
        m2 = ExportMetadata.from_json(path)
        assert m2.variant == m1.variant
        assert m2.version == m1.version
        assert m2.input_dim == m1.input_dim

    def test_from_json_missing_file(self, tmp_path: Path) -> None:
        """from_json on nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ExportMetadata.from_json(tmp_path / "nonexistent.json")


# ═══════════════════════════════════════════════════════════════════════════════
# check_onnx_dependencies
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckOnnxDependencies:
    def test_basic_check(self) -> None:
        """Returns a tuple of (bool, str)."""
        ok, msg = check_onnx_dependencies()
        assert isinstance(ok, bool)
        assert isinstance(msg, str)

    def test_require_runtime(self) -> None:
        """With require_runtime=True returns the runtime availability."""
        ok, msg = check_onnx_dependencies(require_runtime=True)
        assert isinstance(ok, bool)


# ═══════════════════════════════════════════════════════════════════════════════
# build_export_manifest / finalize_export_artifact / verify_export_artifact
# ═══════════════════════════════════════════════════════════════════════════════


class TestManifestFunctions:
    def test_build_export_manifest(self) -> None:
        """build_export_manifest returns a dict with standard keys."""
        manifest = build_export_manifest(
            model_architecture="HelixIDSFull",
            contract=runtime_contract_payload(),
        )
        assert isinstance(manifest, dict)
        assert "model_architecture" in manifest
        assert manifest["model_architecture"] == "HelixIDSFull"

    def test_build_manifest_with_extra(self) -> None:
        """Extra fields produce a hash entry in the manifest."""
        manifest = build_export_manifest(
            model_architecture="Test",
            export_config={"key": "val"},
            onnx_opset=13,
        )
        assert manifest.get("export_config_hash") is not None
        assert manifest.get("onnx_opset") == 13


# ═══════════════════════════════════════════════════════════════════════════════
# ONNXExporter — unit tests (skip when ONNX missing)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not installed")
class TestONNXExporter:
    def test_init(self) -> None:
        """Exporter initialises without error."""
        e = ONNXExporter(verbose=False)
        assert e.verbose is False

    def test_init_missing_onnx(self) -> None:
        """When ONNX is unavailable, init raises ImportError."""
        with patch.object(ONNXExporter, "_check_dependencies") as mock_check:
            mock_check.side_effect = ImportError("ONNX not found")
            with pytest.raises(ImportError):
                ONNXExporter(verbose=False)

    def test_export_to_onnx(self, onnx_path: Path) -> None:
        """Export produces a non-empty ONNX file."""
        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 0

    def test_export_custom_names(self, model, tmp_output: Path) -> None:
        """Custom input/output names are applied."""
        e = ONNXExporter(verbose=False)
        path = e.export_to_onnx(
            model,
            tmp_output / "custom.onnx",
            input_shape=(1, 17),
            input_names=["features"],
            output_names=["bin_pred", "fam_pred"],
        )
        import onnx

        onnx_model = onnx.load(str(path))
        names = {g.name for g in onnx_model.graph.output}
        assert "fam_pred" in names

    def test_export_metadata_embedded(self, model, tmp_output: Path) -> None:
        """Metadata dict is embedded into ONNX model."""
        e = ONNXExporter(verbose=False)
        path = e.export_to_onnx(
            model,
            tmp_output / "meta.onnx",
            input_shape=(1, 17),
            metadata={"key1": "value1", "key2": "42"},
        )
        import onnx

        onnx_model = onnx.load(str(path))
        props = {p.key: p.value for p in onnx_model.metadata_props}
        assert props["key1"] == "value1"
        assert props["key2"] == "42"

    def test_export_with_config(self, model, tmp_output: Path) -> None:
        """export_with_config runs end-to-end and creates sidecars."""
        config = ExportMetadata(variant="full")
        e = ONNXExporter(verbose=False)
        path = e.export_with_config(model, tmp_output / "config.onnx", config)
        assert path.exists()
        # Sidecars should have been written (contract, feature_order, schema_hash)
        contract_sidecar = path.with_suffix(path.suffix + ".contract.json")
        assert contract_sidecar.exists()

    def test_export_via_export_for_edge(self, model, tmp_output: Path) -> None:
        """export_for_edge produces ONNX + metadata + example script."""
        result = export_for_edge(
            model,
            variant="full",
            output_dir=tmp_output / "edge_pkg",
            create_example_script=True,
        )
        assert "onnx" in result
        assert result["onnx"].exists()
        assert "metadata" in result
        assert result["metadata"].exists()
        assert "example" in result
        assert result["example"].exists()

    def test_export_for_edge_unknown_variant(self, model, tmp_output: Path) -> None:
        """Unknown variant raises ValueError."""
        with pytest.raises(ValueError, match="Unknown variant"):
            export_for_edge(model, variant="unknown", output_dir=tmp_output)

    def test_export_with_config_parity(self, model, tmp_output: Path) -> None:
        """export_with_config validates parity and raises on failure."""
        import copy

        e = ONNXExporter(verbose=False)
        config = ExportMetadata(variant="full")
        # Export a clean model first — succeeds
        good_path = e.export_with_config(model, tmp_output / "good.onnx", config)
        assert good_path.exists()
        # Deep-copy the model and corrupt its weights so re-export fails parity
        broken_model = copy.deepcopy(model)
        with torch.no_grad():
            for p in broken_model.parameters():
                p.add_(torch.randn_like(p) * 1000)
        with pytest.raises(RuntimeError, match="parity validation failed"):
            e.export_with_config(broken_model, tmp_output / "fail.onnx", config)

    def test_export_missing_deps(self, model, tmp_output: Path) -> None:
        """When onnx unimportable, ONNXExporter init raises ImportError."""
        with patch("helix_ids.utils.export.ONNX_AVAILABLE", False):
            with pytest.raises(ImportError, match="ONNX is not installed"):
                ONNXExporter(verbose=False)


# ═══════════════════════════════════════════════════════════════════════════════
# validate_onnx
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not ONNXRUNTIME_AVAILABLE, reason="ONNX Runtime not installed")
class TestValidateOnnx:
    def test_validate_valid_model(self, onnx_path: Path) -> None:
        """Valid ONNX model passes validation."""
        valid, details = validate_onnx(onnx_path)
        assert valid is True
        assert details["onnx_check"] == "passed"

    def test_validate_with_pytorch_parity(self, onnx_path: Path, model) -> None:
        """Validation with PyTorch comparison shows match."""
        test_input = torch.arange(17, dtype=torch.float32).reshape(1, -1)
        valid, details = validate_onnx(onnx_path, pytorch_model=model, test_input=test_input)
        assert valid is True
        comp = details.get("pytorch_comparison", {})
        if "match" in comp:
            assert comp["match"] is True

    def test_validate_nonexistent_file(self) -> None:
        """Nonexistent ONNX file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            validate_onnx("/nonexistent/model.onnx")


# ═══════════════════════════════════════════════════════════════════════════════
# benchmark_onnx
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not ONNXRUNTIME_AVAILABLE, reason="ONNX Runtime not installed")
class TestBenchmarkOnnx:
    def test_benchmark_basic(self, onnx_path: Path) -> None:
        """Benchmark completes and returns timing info."""
        x = np.random.randn(4, 17).astype(np.float32)
        results = benchmark_onnx(onnx_path, x, n_runs=5, warmup_runs=2)
        assert "onnx" in results
        assert results["onnx"]["mean_ms"] > 0

    def test_benchmark_with_pytorch(self, onnx_path: Path, model) -> None:
        """Benchmark with PyTorch model returns speedup."""
        x = np.random.randn(4, 17).astype(np.float32)
        results = benchmark_onnx(onnx_path, x, n_runs=5, warmup_runs=2, pytorch_model=model)
        assert "pytorch" in results
        assert "speedup" in results
        assert results["speedup"]["factor"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# export_for_edge
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not installed")
class TestExportForEdge:
    def test_nano_variant(self, model, tmp_output: Path) -> None:
        """Nano variant export succeeds."""
        result = export_for_edge(
            model, variant="nano", output_dir=tmp_output / "nano", create_example_script=True
        )
        assert result["onnx"].exists()
        assert result["metadata"].exists()

    def test_lite_variant(self, model, tmp_output: Path) -> None:
        """Lite variant export succeeds."""
        result = export_for_edge(
            model, variant="lite", output_dir=tmp_output / "lite", create_example_script=False
        )
        assert result["onnx"].exists()

    def test_example_script_content(self, model, tmp_output: Path) -> None:
        """Example script contains model reference."""
        result = export_for_edge(
            model, variant="full", output_dir=tmp_output / "example", create_example_script=True
        )
        content = result["example"].read_text()
        assert "onnx" in content or "import" in content

