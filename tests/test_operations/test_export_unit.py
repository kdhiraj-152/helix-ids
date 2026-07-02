"""Targeted unit tests for export.py ONNX paths.

Covers ONNXExporter, validate_onnx, benchmark_onnx, export_for_edge,
quick_export, get_onnx_info, and related helpers.
ONNX and onnxruntime are available in the venv so we test real exports.
Governance-heavy paths (finalize/verify) are mocked where they'd fail.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
import pytest
import torch
import torch.nn as nn

from helix_ids.utils.export import (
    ExportMetadata,
    ONNXExporter,
    _benchmark_pytorch,
    _compare_outputs,
    _create_example_script,
    benchmark_onnx,
    check_onnx_dependencies,
    export_for_edge,
    get_onnx_info,
    quick_export,
    validate_onnx,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_model() -> nn.Module:
    """A minimal nn.Linear model."""
    return nn.Linear(17, 9)


@pytest.fixture
def dual_output_model() -> nn.Module:
    """A model that returns a tuple of tensors (binary, family)."""

    class DualOutputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.binary = nn.Linear(17, 2)
            self.family = nn.Linear(17, 7)

        def forward(self, x):
            return self.binary(x), self.family(x)

    return DualOutputModel()


@pytest.fixture
def onnx_exporter() -> ONNXExporter:
    return ONNXExporter(verbose=False)


@pytest.fixture
def tmp_onnx_path(tmp_path: Path) -> Path:
    return tmp_path / "export" / "test_model.onnx"


@pytest.fixture
def onnx_model_path(onnx_exporter, dual_output_model, tmp_onnx_path) -> Path:
    """Export a real ONNX model and return its path."""
    return onnx_exporter.export_to_onnx(
        model=dual_output_model,
        filepath=tmp_onnx_path,
        input_shape=(1, 17),
        opset_version=13,
        dynamic_batch=True,
    )


# =============================================================================
# ONNXExporter — constructor basics
# =============================================================================


class TestONNXExporterInit:
    def test_init_verbose(self):
        exporter = ONNXExporter(verbose=True)
        assert exporter.verbose is True

    def test_init_non_verbose(self):
        exporter = ONNXExporter(verbose=False)
        assert exporter.verbose is False

    def test_check_dependencies_ok(self):
        exporter = ONNXExporter(verbose=False)
        exporter._check_dependencies()

    def test_check_dependencies_fails(self):
        with patch("helix_ids.utils.export.ONNX_AVAILABLE", False):
            with pytest.raises(ImportError, match="ONNX is not installed"):
                ONNXExporter(verbose=False)

    def test_log_verbose(self, capsys):
        exporter = ONNXExporter(verbose=True)
        exporter._log("hello world")
        captured = capsys.readouterr()
        assert "hello world" in captured.out

    def test_log_non_verbose(self, capsys):
        exporter = ONNXExporter(verbose=False)
        exporter._log("should not appear")
        captured = capsys.readouterr()
        assert captured.out == ""


# =============================================================================
# ONNXExporter.export_to_onnx
# =============================================================================


class TestExportToOnnx:
    def test_basic_export(self, onnx_exporter, dual_output_model, tmp_path):
        path = tmp_path / "basic.onnx"
        result = onnx_exporter.export_to_onnx(
            model=dual_output_model, filepath=path, input_shape=(1, 17),
        )
        assert result == path
        assert path.exists()
        assert path.stat().st_size > 0
        onnx_model = onnx.load(str(path))
        onnx.checker.check_model(onnx_model)

    def test_export_with_metadata(self, onnx_exporter, dual_output_model, tmp_path):
        path = tmp_path / "meta.onnx"
        metadata = {"model_name": "test", "version": "1.0"}
        result = onnx_exporter.export_to_onnx(
            model=dual_output_model, filepath=path, input_shape=(1, 17),
            metadata=metadata,
        )
        assert result == path
        onnx_model = onnx.load(str(path))
        props = {p.key: p.value for p in onnx_model.metadata_props}
        assert props["model_name"] == "test"
        assert props["version"] == "1.0"

    def test_export_no_dynamic_batch(self, onnx_exporter, dual_output_model, tmp_path):
        path = tmp_path / "static.onnx"
        result = onnx_exporter.export_to_onnx(
            model=dual_output_model, filepath=path, input_shape=(4, 17),
            dynamic_batch=False,
        )
        assert result.exists()

    def test_export_custom_names(self, onnx_exporter, dual_output_model, tmp_path):
        path = tmp_path / "custom_names.onnx"
        result = onnx_exporter.export_to_onnx(
            model=dual_output_model, filepath=path, input_shape=(1, 17),
            input_names=["features"], output_names=["bin_out", "fam_out"],
        )
        assert result.exists()
        onnx_model = onnx.load(str(path))
        assert onnx_model.graph.input[0].name == "features"

    def test_export_creates_parent_dir(self, onnx_exporter, dual_output_model, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "nested.onnx"
        result = onnx_exporter.export_to_onnx(
            model=dual_output_model, filepath=nested, input_shape=(1, 17),
        )
        assert result.exists()

    def test_export_cpu_device(self, onnx_exporter, dual_output_model, tmp_path):
        model = dual_output_model.cpu()
        path = tmp_path / "cpu.onnx"
        result = onnx_exporter.export_to_onnx(
            model=model, filepath=path, input_shape=(1, 17),
        )
        assert result.exists()


# =============================================================================
# ONNXExporter._embed_metadata
# =============================================================================


class TestEmbedMetadata:
    def test_embed_metadata(self, onnx_exporter, onnx_model_path):
        meta = {"key1": "value1", "key2": "value2"}
        onnx_exporter._embed_metadata(onnx_model_path, meta)
        onnx_model = onnx.load(str(onnx_model_path))
        props = {p.key: p.value for p in onnx_model.metadata_props}
        assert props["key1"] == "value1"
        assert props["key2"] == "value2"

    def test_embed_empty_metadata(self, onnx_exporter, onnx_model_path):
        onnx_exporter._embed_metadata(onnx_model_path, {})
        onnx_model = onnx.load(str(onnx_model_path))
        assert len(list(onnx_model.metadata_props)) == 0


# =============================================================================
# ONNXExporter.export_with_config — requires governance mock
# =============================================================================


class TestExportWithConfig:
    def test_export_with_config_basic(self, onnx_exporter, dual_output_model, tmp_path):
        path = tmp_path / "config_test.onnx"
        config = ExportMetadata(
            model_name="TestModel", variant="nano", input_dim=17, opset_version=13,
        )
        with (
            patch("helix_ids.utils.export.finalize_export_artifact") as mock_fin,
            patch("helix_ids.utils.export.verify_export_artifact") as mock_ver,
        ):
            mock_fin.return_value = {"status": "ok"}
            mock_ver.return_value = {"status": "ok"}
            result = onnx_exporter.export_with_config(
                model=dual_output_model, filepath=path, config=config,
            )
        assert result == path
        assert path.exists()
        onnx_model = onnx.load(str(path))
        props = {p.key: p.value for p in onnx_model.metadata_props}
        assert props["model_name"] == "TestModel"
        assert props["variant"] == "nano"

    def test_export_with_config_parity_failure(self, onnx_exporter, dual_output_model, tmp_path):
        path = tmp_path / "fail.onnx"
        config = ExportMetadata(input_dim=17)
        with (
            patch("helix_ids.utils.export.finalize_export_artifact"),
            patch("helix_ids.utils.export.verify_export_artifact"),
            patch("helix_ids.utils.export.validate_onnx") as mock_val,
        ):
            mock_val.return_value = (False, {"error": "parity mismatch"})
            with pytest.raises(RuntimeError, match="parity validation failed"):
                onnx_exporter.export_with_config(
                    model=dual_output_model, filepath=path, config=config,
                )

    def test_export_with_config_verbose(self, dual_output_model, tmp_path, capsys):
        exporter = ONNXExporter(verbose=True)
        path = tmp_path / "verbose.onnx"
        config = ExportMetadata(input_dim=17)
        with (
            patch("helix_ids.utils.export.finalize_export_artifact"),
            patch("helix_ids.utils.export.verify_export_artifact"),
        ):
            exporter.export_with_config(
                model=dual_output_model, filepath=path, config=config,
            )
        captured = capsys.readouterr()
        assert "Exporting" in captured.out


# =============================================================================
# validate_onnx — mocks verify_export_artifact (no governance manifests)
# =============================================================================


class TestValidateOnnx:
    @staticmethod
    def _mock_verify():
        """Context manager that patches verify_export_artifact for validate_onnx."""
        return patch("helix_ids.utils.export.verify_export_artifact",
                     return_value={"status": "ok"})

    def test_validate_valid_model(self, onnx_model_path):
        with self._mock_verify():
            is_valid, details = validate_onnx(onnx_model_path)
        assert is_valid is True
        assert details["onnx_check"] == "passed"
        assert details["valid"] is True
        assert details["file_size_kb"] > 0

    def test_validate_with_pytorch_comparison(self, onnx_model_path, dual_output_model):
        with self._mock_verify():
            test_input = torch.arange(17, dtype=torch.float32).reshape(1, -1)
            is_valid, details = validate_onnx(
                onnx_model_path,
                pytorch_model=dual_output_model,
                test_input=test_input,
            )
        assert is_valid is True
        assert details["pytorch_comparison"]["match"] is True

    def test_validate_with_tighter_tolerance(self, onnx_model_path, dual_output_model):
        with self._mock_verify():
            test_input = torch.randn(1, 17)
            is_valid, details = validate_onnx(
                onnx_model_path,
                pytorch_model=dual_output_model,
                test_input=test_input,
                rtol=1e-5, atol=1e-6,
            )
        assert is_valid is True

    def test_validate_with_mismatched_model(self, onnx_model_path):
        with self._mock_verify():
            wrong_model = nn.Linear(17, 5)
            test_input = torch.randn(1, 17)
            is_valid, details = validate_onnx(
                onnx_model_path,
                pytorch_model=wrong_model,
                test_input=test_input,
            )
        assert is_valid is False
        assert details["pytorch_comparison"]["match"] is False

    def test_validate_nonexistent_file(self, tmp_path):
        # Create an invalid file so stat passes but onnx.load fails
        fake_path = tmp_path / "invalid.onnx"
        fake_path.write_bytes(b"not an ONNX file")
        with patch("helix_ids.utils.export.verify_export_artifact") as m:
            m.return_value = {"status": "ok"}
            is_valid, details = validate_onnx(fake_path)
        assert is_valid is False

    def test_validate_with_metadata(self, onnx_model_path, onnx_exporter):
        onnx_exporter._embed_metadata(onnx_model_path, {"env": "test"})
        with self._mock_verify():
            is_valid, details = validate_onnx(onnx_model_path)
        assert is_valid is True
        assert details["metadata"]["env"] == "test"

    def test_validate_multibatch(self, onnx_model_path, dual_output_model):
        with self._mock_verify():
            test_input = torch.randn(4, 17)
            is_valid, details = validate_onnx(
                onnx_model_path,
                pytorch_model=dual_output_model,
                test_input=test_input,
            )
        assert is_valid is True

    def test_validate_runtime_not_available(self, onnx_model_path):
        with patch("helix_ids.utils.export.check_onnx_dependencies") as mc:
            mc.return_value = (False, "ONNX Runtime is not installed")
            is_valid, details = validate_onnx(onnx_model_path)
        assert is_valid is False
        assert "error" in details


# =============================================================================
# validate_onnx — governance verification failure (no mock)
# =============================================================================


class TestValidateOnnxGovernance:
    def test_validate_verify_artifact_fails(self, onnx_model_path):
        with patch("helix_ids.utils.export.verify_export_artifact") as m:
            m.side_effect = RuntimeError("verification failed")
            with pytest.raises(RuntimeError, match="verification failed"):
                validate_onnx(onnx_model_path)


# =============================================================================
# validate_onnx edge cases — empty metadata model
# =============================================================================


class TestValidateOnnxEmptyMetadata:
    def test_model_without_metadata(self, onnx_model_path):
        with patch("helix_ids.utils.export.verify_export_artifact") as m:
            m.return_value = {"status": "ok"}
            is_valid, details = validate_onnx(onnx_model_path)
        assert is_valid is True
        assert "metadata" in details


# =============================================================================
# validate_onnx - verify outputs structure
# =============================================================================


class TestValidateOnnxDetails:
    def test_details_has_inputs_outputs(self, onnx_model_path):
        with patch("helix_ids.utils.export.verify_export_artifact") as m:
            m.return_value = {"status": "ok"}
            is_valid, details = validate_onnx(onnx_model_path)
        assert is_valid is True
        assert "inputs" in details
        assert len(details["inputs"]) > 0
        assert "outputs" in details
        assert len(details["outputs"]) > 0

    def test_details_opset_version(self, onnx_model_path):
        with patch("helix_ids.utils.export.verify_export_artifact") as m:
            m.return_value = {"status": "ok"}
            is_valid, details = validate_onnx(onnx_model_path)
        assert details["opset_version"] >= 13


# =============================================================================
# _compare_outputs
# =============================================================================


class TestCompareOutputs:
    def test_compare_matching(self, onnx_model_path, dual_output_model):
        test_input = torch.randn(1, 17)
        comparison = _compare_outputs(
            onnx_model_path, dual_output_model, test_input, rtol=1e-3, atol=1e-5
        )
        assert comparison["match"] is True
        assert len(comparison["outputs"]) > 0

    def test_compare_different_shapes(self, onnx_model_path):
        wrong_model = nn.Linear(17, 3)
        test_input = torch.randn(1, 17)
        comparison = _compare_outputs(
            onnx_model_path, wrong_model, test_input, rtol=1e-3, atol=1e-5
        )
        assert comparison["match"] is False

    def test_compare_multibatch(self, onnx_model_path, dual_output_model):
        test_input = torch.randn(8, 17)
        comparison = _compare_outputs(
            onnx_model_path, dual_output_model, test_input, rtol=1e-3, atol=1e-5
        )
        assert comparison["match"] is True

    def test_compare_single_output_model(self, tmp_path, onnx_exporter):
        """A model with a single tensor output -- _compare_outputs expects
        the pytorch model to return a sequence of tensors (tuple)."""
        # Use a model that returns a tuple with one element
        class SingleOutputModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(17, 5)

            def forward(self, x):
                return (self.linear(x),)

        model = SingleOutputModel()
        path = tmp_path / "single.onnx"
        onnx_exporter.export_to_onnx(
            model=model, filepath=path, input_shape=(1, 17),
            output_names=["output"],
        )
        test_input = torch.randn(1, 17)
        comparison = _compare_outputs(path, model, test_input, rtol=1e-3, atol=1e-5)
        assert comparison["match"] is True


# =============================================================================
# benchmark_onnx — mocks verify_export_artifact
# =============================================================================


class TestBenchmarkOnnx:
    @staticmethod
    def _mock_verify():
        return patch("helix_ids.utils.export.verify_export_artifact",
                     return_value={"status": "ok"})

    def test_benchmark_basic(self, onnx_model_path):
        x = np.random.randn(1, 17).astype(np.float32)
        with self._mock_verify():
            results = benchmark_onnx(onnx_model_path, x, n_runs=5, warmup_runs=2)
        assert "onnx" in results
        assert results["onnx"]["mean_ms"] > 0
        assert results["onnx"]["throughput_samples_per_sec"] > 0

    def test_benchmark_with_pytorch(self, onnx_model_path, dual_output_model):
        x = np.random.randn(1, 17).astype(np.float32)
        with self._mock_verify():
            results = benchmark_onnx(
                onnx_model_path, x, n_runs=5, warmup_runs=2,
                pytorch_model=dual_output_model,
            )
        assert "pytorch" in results
        assert "speedup" in results

    def test_benchmark_with_tensor_input(self, onnx_model_path):
        x = torch.randn(1, 17)
        with self._mock_verify():
            results = benchmark_onnx(onnx_model_path, x, n_runs=3, warmup_runs=1)
        assert "onnx" in results

    def test_benchmark_multibatch(self, onnx_model_path):
        x = np.random.randn(16, 17).astype(np.float32)
        with self._mock_verify():
            results = benchmark_onnx(onnx_model_path, x, n_runs=3)
        assert results["batch_size"] == 16

    def test_benchmark_runtime_not_available(self, onnx_model_path):
        with patch("helix_ids.utils.export.check_onnx_dependencies") as mc:
            mc.return_value = (False, "ONNX Runtime is not installed")
            x = np.random.randn(1, 17).astype(np.float32)
            results = benchmark_onnx(onnx_model_path, x)
        assert "error" in results

    def test_benchmark_few_runs(self, onnx_model_path):
        x = np.random.randn(1, 17).astype(np.float32)
        with self._mock_verify():
            results = benchmark_onnx(onnx_model_path, x, n_runs=1, warmup_runs=0)
        assert results["onnx"]["mean_ms"] > 0


# =============================================================================
# benchmark_onnx — default/custom providers
# =============================================================================


class TestBenchmarkOnnxProviders:
    @staticmethod
    def _mock_verify():
        return patch("helix_ids.utils.export.verify_export_artifact",
                     return_value={"status": "ok"})

    def test_benchmark_default_providers(self, onnx_model_path):
        x = np.random.randn(1, 17).astype(np.float32)
        with self._mock_verify():
            results = benchmark_onnx(onnx_model_path, x, n_runs=2)
        assert "CPUExecutionProvider" in results["execution_providers"]

    def test_benchmark_with_pytorch_speedup(self, onnx_model_path, dual_output_model):
        x = np.random.randn(1, 17).astype(np.float32)
        with self._mock_verify():
            results = benchmark_onnx(
                onnx_model_path, x, n_runs=3, warmup_runs=1,
                pytorch_model=dual_output_model,
            )
        assert "speedup" in results
        assert results["speedup"]["factor"] > 0


# =============================================================================
# benchmark_onnx — verify_export_artifact exception propagation
# =============================================================================


class TestBenchmarkOnnxVerification:
    def test_benchmark_verify_fails_raises(self, onnx_model_path):
        with patch("helix_ids.utils.export.verify_export_artifact") as m:
            m.side_effect = RuntimeError("verification failed")
            x = np.random.randn(1, 17).astype(np.float32)
            with pytest.raises(RuntimeError, match="verification failed"):
                benchmark_onnx(onnx_model_path, x, n_runs=2)


# =============================================================================
# _benchmark_pytorch
# =============================================================================


class TestBenchmarkPytorch:
    def test_benchmark_basic(self, dual_output_model):
        x = np.random.randn(1, 17).astype(np.float32)
        results = _benchmark_pytorch(dual_output_model, x, n_runs=5, warmup_runs=2)
        assert results["mean_ms"] > 0
        assert results["throughput_samples_per_sec"] > 0

    def test_benchmark_multibatch(self, dual_output_model):
        x = np.random.randn(32, 17).astype(np.float32)
        results = _benchmark_pytorch(dual_output_model, x, n_runs=3, warmup_runs=1)
        assert results["mean_ms"] > 0

    def test_benchmark_few_runs(self, dual_output_model):
        x = np.random.randn(1, 17).astype(np.float32)
        results = _benchmark_pytorch(dual_output_model, x, n_runs=1, warmup_runs=0)
        assert results["mean_ms"] > 0


# =============================================================================
# export_for_edge
# =============================================================================


class TestExportForEdge:
    def test_export_nano(self, dual_output_model, tmp_path):
        with (
            patch("helix_ids.utils.export.verify_export_artifact") as m_v,
            patch("helix_ids.utils.export.finalize_export_artifact") as m_f,
        ):
            m_v.return_value = {"status": "ok"}
            m_f.return_value = {"status": "ok"}
            result = export_for_edge(
                model=dual_output_model, variant="nano",
                output_dir=tmp_path / "edge_nano", input_dim=17,
            )
        assert "onnx" in result
        assert "metadata" in result
        assert result["onnx"].exists()
        assert result["metadata"].exists()

    def test_export_lite_with_example(self, dual_output_model, tmp_path):
        with (
            patch("helix_ids.utils.export.verify_export_artifact") as m_v,
            patch("helix_ids.utils.export.finalize_export_artifact") as m_f,
        ):
            m_v.return_value = {"status": "ok"}
            m_f.return_value = {"status": "ok"}
            result = export_for_edge(
                model=dual_output_model, variant="lite",
                output_dir=tmp_path / "edge_lite", input_dim=17,
            )
        assert "example" in result
        assert result["example"].exists()
        assert result["example"].stat().st_mode & 0o111

    def test_export_no_example_script(self, dual_output_model, tmp_path):
        with (
            patch("helix_ids.utils.export.verify_export_artifact") as m_v,
            patch("helix_ids.utils.export.finalize_export_artifact") as m_f,
        ):
            m_v.return_value = {"status": "ok"}
            m_f.return_value = {"status": "ok"}
            result = export_for_edge(
                model=dual_output_model, variant="nano",
                output_dir=tmp_path / "edge_no_example", input_dim=17,
                create_example_script=False,
            )
        assert "example" not in result

    def test_export_invalid_variant(self, dual_output_model, tmp_path):
        with pytest.raises(ValueError, match="Unknown variant"):
            export_for_edge(
                model=dual_output_model, variant="invalid",
                output_dir=tmp_path / "edge_bad", input_dim=17,
            )

    def test_export_onnx_not_available(self, dual_output_model, tmp_path):
        with patch("helix_ids.utils.export.ONNX_AVAILABLE", False):
            with pytest.raises(ImportError, match="ONNX is not installed"):
                export_for_edge(
                    model=dual_output_model, variant="nano",
                    output_dir=tmp_path / "edge_no_onnx", input_dim=17,
                )

    def test_export_custom_version(self, dual_output_model, tmp_path):
        with (
            patch("helix_ids.utils.export.verify_export_artifact") as m_v,
            patch("helix_ids.utils.export.finalize_export_artifact") as m_f,
        ):
            m_v.return_value = {"status": "ok"}
            m_f.return_value = {"status": "ok"}
            result = export_for_edge(
                model=dual_output_model, variant="lite",
                output_dir=tmp_path / "edge_v2", input_dim=17,
                version="2.0.0",
            )
        assert "v2_0_0" in str(result["onnx"])


# =============================================================================
# quick_export
# =============================================================================


class TestQuickExport:
    def test_quick_export_basic(self, dual_output_model, tmp_path):
        path = tmp_path / "quick.onnx"
        with (
            patch("helix_ids.utils.export.finalize_export_artifact") as m_f,
            patch("helix_ids.utils.export.verify_export_artifact") as m_v,
        ):
            m_f.return_value = {"status": "ok"}
            m_v.return_value = {"status": "ok"}
            result = quick_export(
                model=dual_output_model, output_path=path, input_dim=17,
            )
        assert result == path
        assert path.exists()


# =============================================================================
# get_onnx_info
# =============================================================================


class TestGetOnnxInfo:
    def test_get_info_basic(self, onnx_model_path):
        info = get_onnx_info(onnx_model_path)
        assert info["filepath"] == str(onnx_model_path)
        assert info["file_size_kb"] > 0
        assert info["opset_version"] > 0
        assert len(info["inputs"]) > 0
        assert len(info["outputs"]) > 0
        assert info["num_nodes"] > 0

    def test_get_info_nonexistent(self, tmp_path):
        """onnx.load raises FileNotFoundError for nonexistent paths."""
        with pytest.raises(FileNotFoundError):
            get_onnx_info(tmp_path / "nope.onnx")

    def test_get_info_metadata(self, onnx_model_path, onnx_exporter):
        onnx_exporter._embed_metadata(onnx_model_path, {"foo": "bar"})
        info = get_onnx_info(onnx_model_path)
        assert info["metadata"]["foo"] == "bar"

    def test_get_info_onnx_not_available(self, onnx_model_path):
        with patch("helix_ids.utils.export.ONNX_AVAILABLE", False):
            info = get_onnx_info(onnx_model_path)
            assert "error" in info


# =============================================================================
# _create_example_script
# =============================================================================


class TestCreateExampleScript:
    def test_create_script(self, tmp_path):
        path = tmp_path / "inference_example.py"
        _create_example_script(
            filepath=path, model_filename="model.onnx",
            variant="nano", input_dim=17,
        )
        assert path.exists()
        content = path.read_text()
        assert "model.onnx" in content
        assert "HELIX-IDS NANO Inference Example" in content
        assert "17" in content

    def test_create_script_is_executable(self, tmp_path):
        path = tmp_path / "inference_example.py"
        _create_example_script(
            filepath=path, model_filename="test.onnx",
            variant="lite", input_dim=32,
        )
        assert path.stat().st_mode & 0o111

    def test_create_script_full_variant(self, tmp_path):
        path = tmp_path / "full_example.py"
        _create_example_script(
            filepath=path, model_filename="helix_full.onnx",
            variant="full", input_dim=32,
        )
        content = path.read_text()
        assert "full" in content.lower()


# =============================================================================
# ExportMetadata edge cases
# =============================================================================


class TestExportMetadataEdgeCases:
    def test_feature_order_default(self):
        meta = ExportMetadata()
        assert meta.feature_order == []

    def test_feature_order_custom(self):
        meta = ExportMetadata(feature_order=["f1", "f2"])
        assert meta.feature_order == ["f1", "f2"]

    def test_exported_at_custom(self):
        meta = ExportMetadata(exported_at="2025-06-01T00:00:00")
        assert meta.exported_at == "2025-06-01T00:00:00"

    def test_post_init_sets_timestamp(self):
        meta = ExportMetadata(exported_at="")
        assert meta.exported_at != ""


# =============================================================================
# check_onnx_dependencies — additional edge cases
# =============================================================================


class TestCheckOnnxDependenciesEdge:
    def test_onnx_available_runtime_required_and_available(self):
        with (
            patch("helix_ids.utils.export.ONNX_AVAILABLE", True),
            patch("helix_ids.utils.export.ONNXRUNTIME_AVAILABLE", True),
        ):
            available, msg = check_onnx_dependencies(require_runtime=True)
            assert available
            assert "available" in msg

    def test_onnx_not_available_no_runtime_check(self):
        with patch("helix_ids.utils.export.ONNX_AVAILABLE", False):
            available, msg = check_onnx_dependencies(require_runtime=False)
            assert not available
            assert "ONNX is not installed" in msg


# =============================================================================
# Export device handling
# =============================================================================


class TestExportDeviceHandling:
    def test_model_on_cpu(self, onnx_exporter, tmp_path):
        model = nn.Linear(17, 9).cpu()
        path = tmp_path / "cpu_model.onnx"
        result = onnx_exporter.export_to_onnx(
            model=model, filepath=path, input_shape=(2, 17),
        )
        assert result.exists()
        onnx.checker.check_model(onnx.load(str(result)))


# =============================================================================
# export_to_onnx with all parameters explicit
# =============================================================================


class TestExportToOnnxFullParams:
    def test_all_params(self, onnx_exporter, dual_output_model, tmp_path):
        path = tmp_path / "full_params.onnx"
        metadata = {"author": "test"}
        result = onnx_exporter.export_to_onnx(
            model=dual_output_model, filepath=path, input_shape=(2, 17),
            opset_version=13, dynamic_batch=True,
            input_names=["features"], output_names=["binary", "family"],
            metadata=metadata,
        )
        assert result.exists()
        onnx_model = onnx.load(str(result))
        assert onnx_model.graph.input[0].name == "features"
        props = {p.key: p.value for p in onnx_model.metadata_props}
        assert props["author"] == "test"


# =============================================================================
# _compare_outputs with non-finite / mismatched dtype (branch coverage)
# =============================================================================


class TestCompareOutputsEdge:
    def test_compare_shape_mismatch(self, tmp_path, onnx_exporter):
        """Export a model producing different shapes to trigger shape mismatch."""
        model_a = nn.Linear(17, 5)
        path_a = tmp_path / "model_a.onnx"
        onnx_exporter.export_to_onnx(
            model=model_a, filepath=path_a, input_shape=(1, 17),
            output_names=["out"],
        )
        # Compare with a model of different output shape
        model_b = nn.Linear(17, 3)
        x = torch.randn(1, 17)
        comp = _compare_outputs(path_a, model_b, x, rtol=1e-3, atol=1e-5)
        assert comp["match"] is False

    def test_compare_non_finite(self, tmp_path, onnx_exporter, dual_output_model):
        """Forcing comparison outputs to contain non-finite values is hard with
        a linear model; we cover the code path by verifying that finite floats
        do NOT hit the non-finite branch (happy path)."""
        path = tmp_path / "finite.onnx"
        onnx_exporter.export_to_onnx(
            model=dual_output_model, filepath=path, input_shape=(1, 17),
        )
        x = torch.randn(1, 17)
        comp = _compare_outputs(path, dual_output_model, x, rtol=1e-3, atol=1e-5)
        assert comp["match"] is True

    def test_compare_dtype_mismatch(self, tmp_path, onnx_exporter, dual_output_model):
        """Force dtype mismatch by running f32 ONNX model but feeding f64 PyTorch model."""
        import unittest.mock as mock

        import numpy as np

        from helix_ids.utils.export import ort as export_ort

        # Export float32 model to ONNX
        path = tmp_path / "dtype_test.onnx"
        onnx_exporter.export_to_onnx(
            model=dual_output_model, filepath=path, input_shape=(1, 17),
        )
        # Use float64 PyTorch model and float64 input — ONNX model is still f32.
        model_f64 = dual_output_model.double()
        x_f64 = torch.randn(1, 17).double()

        # Get a real session (before patching)
        real_session = export_ort.InferenceSession(str(path))
        real_input_name = real_session.get_inputs()[0].name
        original_run = export_ort.InferenceSession.run

        def mock_run(_self, output_names, input_feed):
            # Convert f64 input to f32 for onnx runtime to avoid crash
            feed = dict(input_feed)
            feed[real_input_name] = feed[real_input_name].astype(np.float32)
            return original_run(_self, output_names, feed)

        with mock.patch.object(
            export_ort.InferenceSession, "run", mock_run
        ):
            comp = _compare_outputs(path, model_f64, x_f64, rtol=1e-3, atol=1e-5)
        # The ONNX model is float32, pytorch is float64 → dtype mismatch
        assert comp["match"] is False

    def test_compare_values_differ(self, tmp_path, onnx_exporter, dual_output_model):
        """Make PyTorch model differ from ONNX by changing weights."""
        path = tmp_path / "values_test.onnx"
        onnx_exporter.export_to_onnx(
            model=dual_output_model, filepath=path, input_shape=(1, 17),
        )
        # Modify the model weights so values differ
        modified_model = dual_output_model
        with torch.no_grad():
            for p in modified_model.parameters():
                p.add_(100.0)  # Large offset ensures outputs differ
        x = torch.randn(1, 17)
        comp = _compare_outputs(path, modified_model, x, rtol=1e-3, atol=1e-5)
        # Shapes and dtypes match but values should differ significantly
        assert comp["match"] is False


# =============================================================================
# finalize_export_artifact — direct test with mocked governance internals
# =============================================================================


class TestFinalizeExportArtifactDirect:
    def test_finalize_with_sidecars_auto_discover(self, tmp_path):
        """Test finalize_export_artifact with empty sidecars (auto-discover)."""
        from helix_ids.utils.export import finalize_export_artifact

        artifact = tmp_path / "model.onnx"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("fake onnx content")

        manifest = {"model_architecture": "TestModel"}

        with (
            patch("helix_ids.utils.export.build_provenance_chain") as mock_bpc,
            patch("helix_ids.utils.export.finalize_artifact_manifest") as mock_fam,
        ):
            mock_bpc.return_value = {"chain": []}
            mock_fam.return_value = {"manifest_version": "1.0", "status": "ok"}

            result = finalize_export_artifact(
                artifact,
                manifest,
                sidecars={},  # empty → triggers auto-discover
            )
        assert result == {"manifest_version": "1.0", "status": "ok"}
        mock_bpc.assert_called_once()
        mock_fam.assert_called_once()

    def test_finalize_with_explicit_sidecars(self, tmp_path):
        from helix_ids.utils.export import finalize_export_artifact

        artifact = tmp_path / "model.onnx"
        artifact.write_text("fake onnx content")

        manifest = {"model_architecture": "TestModel"}
        sidecars = {"contract": tmp_path / "contract.json"}
        sidecars["contract"].write_text("{}")

        with (
            patch("helix_ids.utils.export.build_provenance_chain") as mock_bpc,
            patch("helix_ids.utils.export.finalize_artifact_manifest") as mock_fam,
        ):
            mock_bpc.return_value = {"chain": []}
            mock_fam.return_value = {"status": "ok"}

            result = finalize_export_artifact(
                artifact,
                manifest,
                sidecars=sidecars,
            )
        assert result == {"status": "ok"}

    def test_finalize_with_deployment_manifest(self, tmp_path):
        from helix_ids.utils.export import finalize_export_artifact

        artifact = tmp_path / "model.onnx"
        artifact.write_text("fake onnx content")
        dep_manifest = tmp_path / "deployment.manifest.json"
        dep_manifest.write_text("{}")

        manifest = {"model_architecture": "TestModel"}

        with (
            patch("helix_ids.utils.export.build_provenance_chain") as mock_bpc,
            patch("helix_ids.utils.export.finalize_artifact_manifest") as mock_fam,
        ):
            mock_bpc.return_value = {"chain": []}
            mock_fam.return_value = {"status": "ok"}

            result = finalize_export_artifact(
                artifact,
                manifest,
                sidecars={},
                deployment_manifest=dep_manifest,
                exporter_metadata={"version": "1.0"},
            )
        assert result == {"status": "ok"}
        # build_provenance_chain should have received deployment_manifest
        _, kwargs = mock_bpc.call_args
        assert kwargs["deployment_manifest"] == dep_manifest
        assert kwargs["exporter_metadata"] == {"version": "1.0"}


# =============================================================================
# verify_export_artifact — direct test with mocked governance internals
# =============================================================================


class TestVerifyExportArtifactDirect:
    def test_verify_basic(self, tmp_path):
        from helix_ids.utils.export import verify_export_artifact

        artifact = tmp_path / "model.onnx"
        artifact.write_text("fake content")

        with patch("helix_ids.utils.export.verify_artifact_provenance") as mock_vap:
            mock_vap.return_value = {"verified": True}

            result = verify_export_artifact(
                artifact,
                kind="onnx",
            )
        assert result == {"verified": True}

    def test_verify_returns_none(self, tmp_path):
        from helix_ids.utils.export import verify_export_artifact

        artifact = tmp_path / "model.onnx"
        artifact.write_text("fake content")

        with patch("helix_ids.utils.export.verify_artifact_provenance") as mock_vap:
            mock_vap.return_value = None

            result = verify_export_artifact(
                artifact,
                kind="onnx",
            )
        assert result is None
