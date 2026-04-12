"""
ONNX Export Infrastructure for HELIX-IDS

Provides utilities for exporting PyTorch models to ONNX format
for edge deployment on ESP32, Raspberry Pi, and other edge devices.

Features:
- ONNX export with dynamic batch size
- Model validation and output comparison
- Benchmarking against PyTorch inference
- Edge-optimized export with metadata
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn

# Optional ONNX dependencies - gracefully handle if not installed
try:
    import onnx

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    onnx = None

try:
    import onnxruntime as ort

    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    ort = None


# HELIX-IDS specific constants
HELIX_VARIANT_SIZES = {
    "nano": 30,  # KB
    "lite": 200,  # KB
    "full": 2000,  # KB
}

HELIX_CLASSES = ["Normal", "DoS", "Probe", "R2L", "U2R"]

DEFAULT_THREAT_WEIGHTS = {
    "Normal": 1.0,
    "DoS": 2.0,
    "Probe": 2.5,
    "R2L": 4.0,
    "U2R": 5.0,
}


@dataclass
class ExportMetadata:
    """Metadata for exported ONNX model."""

    model_name: str = "HELIX-IDS"
    variant: str = "lite"
    version: str = "1.0.0"
    input_dim: int = 41
    num_classes: int = 5
    num_fine_classes: int = 23
    opset_version: int = 13
    input_names: Optional[list[str]] = None
    output_names: Optional[list[str]] = None
    classes: Optional[list[str]] = None
    threat_weights: Optional[dict[str, float]] = None
    target_size_kb: float = 200.0
    exported_at: str = ""

    def __post_init__(self):
        if self.input_names is None:
            self.input_names = ["input"]
        if self.output_names is None:
            self.output_names = ["binary", "family"]
        if self.classes is None:
            self.classes = HELIX_CLASSES.copy()
        if self.threat_weights is None:
            self.threat_weights = DEFAULT_THREAT_WEIGHTS.copy()
        if not self.exported_at:
            from datetime import datetime

            self.exported_at = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, filepath: Union[str, Path]) -> None:
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> "ExportMetadata":
        with open(filepath) as f:
            data = json.load(f)
        return cls(**data)


def check_onnx_dependencies(require_runtime: bool = False) -> tuple[bool, str]:
    """
    Check if ONNX dependencies are available.

    Args:
        require_runtime: Whether onnxruntime is required

    Returns:
        Tuple of (available, message)
    """
    if not ONNX_AVAILABLE:
        return False, (
            "ONNX is not installed. Install with:\n"
            "  pip install onnx\n"
            "Or for full functionality:\n"
            "  pip install onnx onnxruntime"
        )

    if require_runtime and not ONNXRUNTIME_AVAILABLE:
        return False, (
            "ONNX Runtime is not installed. Install with:\n"
            "  pip install onnxruntime\n"
            "For GPU support:\n"
            "  pip install onnxruntime-gpu"
        )

    return True, "ONNX dependencies available"


class ONNXExporter:
    """
    ONNX Exporter for HELIX-IDS models.

    Supports exporting PyTorch models to ONNX format with:
    - Dynamic batch size support
    - Custom input/output naming
    - Metadata embedding
    - Model optimization
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the exporter.

        Args:
            verbose: Whether to print progress messages
        """
        self.verbose = verbose
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Verify ONNX is available."""
        available, message = check_onnx_dependencies()
        if not available:
            raise ImportError(message)

    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[ONNXExporter] {message}")

    def export_to_onnx(
        self,
        model: nn.Module,
        filepath: Union[str, Path],
        input_shape: tuple[int, ...],
        opset_version: int = 13,
        dynamic_batch: bool = True,
        input_names: Optional[list[str]] = None,
        output_names: Optional[list[str]] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> Path:
        """
        Export PyTorch model to ONNX format.

        Args:
            model: PyTorch model to export
            filepath: Output ONNX file path
            input_shape: Input tensor shape (batch_size, input_dim)
            opset_version: ONNX opset version (default: 13)
            dynamic_batch: Enable dynamic batch size
            input_names: Custom input tensor names
            output_names: Custom output tensor names
            metadata: Additional metadata to embed

        Returns:
            Path to exported ONNX file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Set model to evaluation mode
        model.eval()

        # Create dummy input
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_shape, device=device)

        # Default names
        if input_names is None:
            input_names = ["input"]
        if output_names is None:
            output_names = ["binary", "family", "features"]

        # Dynamic axes for variable batch size
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {name: {0: "batch_size"} for name in input_names}
            dynamic_axes.update({name: {0: "batch_size"} for name in output_names})

        self._log(f"Exporting model to {filepath}")
        self._log(f"Input shape: {input_shape}, Opset: {opset_version}")

        # Export to ONNX
        torch.onnx.export(
            model,
            (dummy_input,),
            str(filepath),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

        # Add metadata if provided
        if metadata:
            self._embed_metadata(filepath, metadata)

        self._log(f"Export complete: {filepath}")
        self._log(f"File size: {filepath.stat().st_size / 1024:.2f} KB")

        return filepath

    def _embed_metadata(self, filepath: Path, metadata: dict[str, str]) -> None:
        """Embed metadata into ONNX model."""
        onnx_model = onnx.load(str(filepath))

        for key, value in metadata.items():
            meta = onnx_model.metadata_props.add()
            meta.key = key
            meta.value = str(value)

        onnx.save(onnx_model, str(filepath))
        self._log(f"Embedded {len(metadata)} metadata entries")

    def export_with_config(
        self,
        model: nn.Module,
        filepath: Union[str, Path],
        config: "ExportMetadata",
    ) -> Path:
        """
        Export model with full configuration and metadata.

        Args:
            model: PyTorch model to export
            filepath: Output ONNX file path
            config: Export configuration and metadata

        Returns:
            Path to exported ONNX file
        """
        input_shape = (1, config.input_dim)

        metadata = {
            "model_name": config.model_name,
            "variant": config.variant,
            "version": config.version,
            "input_dim": str(config.input_dim),
            "num_classes": str(config.num_classes),
            "exported_at": config.exported_at,
        }

        return self.export_to_onnx(
            model=model,
            filepath=filepath,
            input_shape=input_shape,
            opset_version=config.opset_version,
            input_names=config.input_names,
            output_names=config.output_names,
            metadata=metadata,
        )


def validate_onnx(
    filepath: Union[str, Path],
    pytorch_model: Optional[nn.Module] = None,
    test_input: Optional[torch.Tensor] = None,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> tuple[bool, dict[str, Any]]:
    """
    Validate ONNX model and optionally compare with PyTorch model.

    Args:
        filepath: Path to ONNX model
        pytorch_model: Optional PyTorch model for comparison
        test_input: Optional test input tensor
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        Tuple of (is_valid, details_dict)
    """
    available, message = check_onnx_dependencies(require_runtime=True)
    if not available:
        return False, {"error": message}

    filepath = Path(filepath)
    results = {
        "filepath": str(filepath),
        "valid": False,
        "file_size_kb": filepath.stat().st_size / 1024,
    }

    # Load and check ONNX model
    try:
        onnx_model = onnx.load(str(filepath))
        onnx.checker.check_model(onnx_model)
        results["onnx_check"] = "passed"
        results["valid"] = True
    except Exception as e:
        results["onnx_check"] = f"failed: {str(e)}"
        return False, results

    # Extract model info
    results["opset_version"] = onnx_model.opset_import[0].version
    results["inputs"] = [
        {"name": inp.name, "shape": [d.dim_value for d in inp.type.tensor_type.shape.dim]}
        for inp in onnx_model.graph.input
    ]
    results["outputs"] = [{"name": out.name} for out in onnx_model.graph.output]

    # Extract metadata
    results["metadata"] = {prop.key: prop.value for prop in onnx_model.metadata_props}

    # Compare with PyTorch model if provided
    if pytorch_model is not None and test_input is not None:
        comparison = _compare_outputs(filepath, pytorch_model, test_input, rtol, atol)
        results["pytorch_comparison"] = comparison
        if not comparison["match"]:
            results["valid"] = False

    return bool(results["valid"]), results


def _compare_outputs(
    onnx_path: Path,
    pytorch_model: nn.Module,
    test_input: torch.Tensor,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Compare ONNX and PyTorch model outputs."""
    comparison: dict[str, Any] = {"match": True, "outputs": {}}

    # Get PyTorch outputs
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input)

    # Get ONNX outputs
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: test_input.numpy()})

    # Compare each output
    output_names = [out.name for out in session.get_outputs()]
    pytorch_outputs = [pytorch_output[key].numpy() for key in ["binary", "family"]]

    for _i, (name, pt_out, onnx_out) in enumerate(
        zip(output_names[: len(pytorch_outputs)], pytorch_outputs, onnx_output)
    ):
        matches = np.allclose(pt_out, onnx_out, rtol=rtol, atol=atol)
        max_diff = np.max(np.abs(pt_out - onnx_out))

        comparison["outputs"][name] = {
            "match": bool(matches),
            "max_difference": float(max_diff),
        }

        if not matches:
            comparison["match"] = False

    return comparison


def benchmark_onnx(
    filepath: Union[str, Path],
    x_sample: Union[np.ndarray, torch.Tensor],
    n_runs: int = 100,
    pytorch_model: Optional[nn.Module] = None,
    warmup_runs: int = 10,
    execution_providers: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Benchmark ONNX model inference performance.

    Args:
        filepath: Path to ONNX model
        x_sample: Sample input data for benchmarking
        n_runs: Number of inference runs
        pytorch_model: Optional PyTorch model for comparison
        warmup_runs: Number of warmup runs before timing
        execution_providers: ONNX Runtime execution providers
            (default: ['CPUExecutionProvider'])

    Returns:
        Dictionary with benchmark results including latency and speedup
    """
    available, message = check_onnx_dependencies(require_runtime=True)
    if not available:
        return {"error": message}

    filepath = Path(filepath)

    # Convert input to numpy
    if isinstance(x_sample, torch.Tensor):
        x_sample = x_sample.numpy()
    x_sample = x_sample.astype(np.float32)

    # Default execution providers
    if execution_providers is None:
        execution_providers = ["CPUExecutionProvider"]
        # Try to add CUDA if available
        if "CUDAExecutionProvider" in ort.get_available_providers():
            execution_providers.insert(0, "CUDAExecutionProvider")

    results = {
        "filepath": str(filepath),
        "n_runs": n_runs,
        "batch_size": x_sample.shape[0],
        "execution_providers": execution_providers,
    }

    # Create ONNX Runtime session
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(str(filepath), sess_options, providers=execution_providers)
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(warmup_runs):
        session.run(None, {input_name: x_sample})

    # Benchmark ONNX
    onnx_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run(None, {input_name: x_sample})
        onnx_times.append(time.perf_counter() - start)

    results["onnx"] = {
        "mean_ms": np.mean(onnx_times) * 1000,
        "std_ms": np.std(onnx_times) * 1000,
        "min_ms": np.min(onnx_times) * 1000,
        "max_ms": np.max(onnx_times) * 1000,
        "throughput_samples_per_sec": x_sample.shape[0] / np.mean(onnx_times),
    }

    # Benchmark PyTorch if provided
    if pytorch_model is not None:
        pytorch_times = _benchmark_pytorch(pytorch_model, x_sample, n_runs, warmup_runs)
        results["pytorch"] = pytorch_times

        # Calculate speedup
        speedup = results["pytorch"]["mean_ms"] / results["onnx"]["mean_ms"]
        results["speedup"] = {
            "factor": speedup,
            "description": f"ONNX is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than PyTorch",
        }

    return results


def _benchmark_pytorch(
    model: nn.Module,
    x_sample: np.ndarray,
    n_runs: int,
    warmup_runs: int,
) -> dict[str, float]:
    """Benchmark PyTorch model inference."""
    model.eval()
    device = next(model.parameters()).device
    x_tensor = torch.from_numpy(x_sample).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            model(x_tensor)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(x_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
        "throughput_samples_per_sec": x_sample.shape[0] / np.mean(times),
    }


def export_for_edge(
    model: nn.Module,
    variant: str,
    output_dir: Union[str, Path],
    input_dim: int = 41,
    version: str = "1.0.0",
    create_example_script: bool = True,
) -> dict[str, Path]:
    """
    Export model for edge deployment with full package.

    Creates:
    - ONNX model file with optimizations
    - Metadata JSON file
    - Example inference script

    Args:
        model: PyTorch model to export
        variant: Model variant ('nano', 'lite', 'full')
        output_dir: Output directory for export package
        input_dim: Input feature dimension
        version: Model version string
        create_example_script: Whether to create example inference script

    Returns:
        Dictionary with paths to created files
    """
    available, message = check_onnx_dependencies()
    if not available:
        raise ImportError(message)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate variant
    if variant not in HELIX_VARIANT_SIZES:
        raise ValueError(
            f"Unknown variant: {variant}. Choose from {list(HELIX_VARIANT_SIZES.keys())}"
        )

    # Create metadata
    metadata = ExportMetadata(
        variant=variant,
        version=version,
        input_dim=input_dim,
        target_size_kb=float(HELIX_VARIANT_SIZES[variant]),
    )

    # File paths
    model_filename = f"helix_{variant}_v{version.replace('.', '_')}.onnx"
    created_files = {
        "onnx": output_dir / model_filename,
        "metadata": output_dir / f"helix_{variant}_metadata.json",
    }

    # Export ONNX model
    exporter = ONNXExporter(verbose=True)
    exporter.export_with_config(model, created_files["onnx"], metadata)

    # Save metadata JSON
    metadata.to_json(created_files["metadata"])
    print(f"[export_for_edge] Metadata saved: {created_files['metadata']}")

    # Create example script
    if create_example_script:
        created_files["example"] = output_dir / f"inference_example_{variant}.py"
        _create_example_script(created_files["example"], model_filename, variant, input_dim)
        print(f"[export_for_edge] Example script created: {created_files['example']}")

    # Validate export
    is_valid, _ = validate_onnx(created_files["onnx"])
    print(f"[export_for_edge] Validation: {'PASSED' if is_valid else 'FAILED'}")

    # Size check
    actual_size_kb = created_files["onnx"].stat().st_size / 1024
    target_size_kb = HELIX_VARIANT_SIZES[variant]
    size_ok = actual_size_kb <= target_size_kb
    print(
        f"[export_for_edge] Size check: {actual_size_kb:.2f} KB / {target_size_kb} KB target {'✓' if size_ok else '✗'}"
    )

    return created_files


def _create_example_script(
    filepath: Path,
    model_filename: str,
    variant: str,
    input_dim: int,
) -> None:
    """Create example inference script for edge deployment."""
    script_content = f'''#!/usr/bin/env python3
"""
HELIX-IDS {variant.upper()} Inference Example

This script demonstrates how to run inference with the exported ONNX model.
Designed for edge deployment on devices like Raspberry Pi and ESP32.

Usage:
    python {filepath.name} --input <input_data.npy>
    python {filepath.name} --random  # Test with random input
"""

import argparse
import time
from pathlib import Path

import numpy as np

# Check for ONNX Runtime
try:
    import onnxruntime as ort
except ImportError:
    print("Error: ONNX Runtime not installed.")
    print("Install with: pip install onnxruntime")
    exit(1)


# HELIX-IDS classes
CLASSES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']

# Threat weights for severity scoring
THREAT_WEIGHTS = {{
    'Normal': 1.0,
    'DoS': 2.0,
    'Probe': 2.5,
    'R2L': 4.0,
    'U2R': 5.0,
}}


def load_model(model_path: str) -> ort.InferenceSession:
    """Load ONNX model with optimal providers."""
    providers = ['CPUExecutionProvider']

    # Try GPU providers if available
    available = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available:
        providers.insert(0, 'CUDAExecutionProvider')

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    return ort.InferenceSession(model_path, sess_options, providers=providers)


def predict(session: ort.InferenceSession, features: np.ndarray) -> dict:
    """
    Run inference on input features.

    Args:
        session: ONNX Runtime session
        features: Input features [batch_size, {input_dim}]

    Returns:
        Dictionary with predictions, probabilities, and timing
    """
    # Ensure correct dtype and shape
    if features.ndim == 1:
        features = features.reshape(1, -1)
    features = features.astype(np.float32)

    # Run inference with timing
    input_name = session.get_inputs()[0].name
    start = time.perf_counter()
    outputs = session.run(None, {{input_name: features}})
    inference_time = (time.perf_counter() - start) * 1000  # ms

    # Process outputs
    binary_logits = outputs[0]  # [batch, 2]
    family_logits = outputs[1]  # [batch, 4]

    # Compute probabilities
    binary_probs = softmax(binary_logits)
    family_probs = softmax(family_logits)

    # Compute 5-class probabilities
    # P(Normal) = P(binary=0)
    # P(Attack_k) = P(binary=1) * P(family=k)
    probs = np.zeros((features.shape[0], 5))
    probs[:, 0] = binary_probs[:, 0]
    probs[:, 1:] = binary_probs[:, 1:2] * family_probs

    predictions = probs.argmax(axis=1)

    return {{
        'predictions': predictions,
        'class_names': [CLASSES[p] for p in predictions],
        'probabilities': probs,
        'confidence': probs.max(axis=1),
        'inference_time_ms': inference_time,
    }}


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description='HELIX-IDS Inference')
    parser.add_argument('--model', default='{model_filename}',
                        help='Path to ONNX model')
    parser.add_argument('--input', type=str, help='Path to input .npy file')
    parser.add_argument('--random', action='store_true',
                        help='Use random test input')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark with multiple iterations')
    parser.add_argument('--n-runs', type=int, default=100,
                        help='Number of benchmark runs')
    args = parser.parse_args()

    # Load model
    model_path = Path(__file__).parent / args.model
    print(f"Loading model: {{model_path}}")
    session = load_model(str(model_path))
    print(f"Model loaded. Providers: {{session.get_providers()}}")

    # Prepare input
    if args.input:
        features = np.load(args.input)
    elif args.random:
        print(f"Using random input ({input_dim} features)")
        features = np.random.randn(1, {input_dim}).astype(np.float32)
    else:
        parser.error("Provide --input or --random")

    # Run inference
    result = predict(session, features)

    print(f"\\nPrediction: {{result['class_names'][0]}}")
    print(f"Confidence: {{result['confidence'][0]:.4f}}")
    print(f"Inference time: {{result['inference_time_ms']:.3f}} ms")
    print(f"\\nClass probabilities:")
    for cls, prob in zip(CLASSES, result['probabilities'][0]):
        print(f"  {{cls}}: {{prob:.4f}}")

    # Benchmark if requested
    if args.benchmark:
        print(f"\\nRunning benchmark ({{args.n_runs}} iterations)...")
        times = []
        for _ in range(args.n_runs):
            r = predict(session, features)
            times.append(r['inference_time_ms'])

        print(f"Mean: {{np.mean(times):.3f}} ms")
        print(f"Std:  {{np.std(times):.3f}} ms")
        print(f"Min:  {{np.min(times):.3f}} ms")
        print(f"Max:  {{np.max(times):.3f}} ms")
        print(f"Throughput: {{1000 / np.mean(times):.1f}} samples/sec")


if __name__ == '__main__':
    main()
'''

    filepath.write_text(script_content)
    filepath.chmod(0o755)


# Convenience functions for quick exports
def quick_export(
    model: nn.Module,
    output_path: Union[str, Path],
    input_dim: int = 41,
    opset_version: int = 13,
) -> Path:
    """
    Quick export with sensible defaults.

    Args:
        model: PyTorch model to export
        output_path: Output ONNX file path
        input_dim: Input feature dimension
        opset_version: ONNX opset version

    Returns:
        Path to exported model
    """
    exporter = ONNXExporter(verbose=True)
    return exporter.export_to_onnx(
        model=model,
        filepath=output_path,
        input_shape=(1, input_dim),
        opset_version=opset_version,
    )


def get_onnx_info(filepath: Union[str, Path]) -> dict[str, Any]:
    """
    Get information about an ONNX model.

    Args:
        filepath: Path to ONNX model

    Returns:
        Dictionary with model information
    """
    available, message = check_onnx_dependencies()
    if not available:
        return {"error": message}

    filepath = Path(filepath)
    onnx_model = onnx.load(str(filepath))

    return {
        "filepath": str(filepath),
        "file_size_kb": filepath.stat().st_size / 1024,
        "opset_version": onnx_model.opset_import[0].version,
        "ir_version": onnx_model.ir_version,
        "producer": onnx_model.producer_name,
        "inputs": [
            {
                "name": inp.name,
                "shape": [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim],
                "dtype": inp.type.tensor_type.elem_type,
            }
            for inp in onnx_model.graph.input
        ],
        "outputs": [
            {
                "name": out.name,
                "shape": [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim],
            }
            for out in onnx_model.graph.output
        ],
        "metadata": {prop.key: prop.value for prop in onnx_model.metadata_props},
        "num_nodes": len(onnx_model.graph.node),
    }


__all__ = [
    # Classes
    "ONNXExporter",
    "ExportMetadata",
    # Main functions
    "validate_onnx",
    "benchmark_onnx",
    "export_for_edge",
    # Utilities
    "quick_export",
    "get_onnx_info",
    "check_onnx_dependencies",
    # Constants
    "HELIX_CLASSES",
    "HELIX_VARIANT_SIZES",
    "DEFAULT_THREAT_WEIGHTS",
    "ONNX_AVAILABLE",
    "ONNXRUNTIME_AVAILABLE",
]
