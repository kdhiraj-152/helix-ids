"""
HELIX-IDS: Unified Deployment Script

Single entry point for deploying HELIX-IDS across all platforms.
Automatically selects the right model based on target platform.

Usage:
    # Inference on production model
    python scripts/deploy.py predict --input data.csv --output predictions.csv

    # Export for specific platform
    python scripts/deploy.py export --platform rpi_4 --output-dir ./deploy

    # Benchmark model
    python scripts/deploy.py benchmark --platform production
"""

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class HelixIDS(nn.Module):
    """HELIX-IDS Production Model"""

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dims: list | None = None,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class HelixDeployer:
    """Unified deployer for HELIX-IDS"""

    MODEL_PLATFORM_HELP = "Model platform"

    PLATFORMS = {
        "production": {
            "hidden_dims": [64, 32, 16],
            "model_path": "models/production/helix_ids_v1.pt",
            "scaler_path": "models/production/scaler.pkl",
            "features_path": "models/production/feature_names.json",
        },
        "rpi_4": {
            "hidden_dims": [64, 32, 16],
            "model_path": "models/rpi_4/helix_rpi_4.pt",
            "scaler_path": "models/rpi_4/scaler.pkl",
            "features_path": "models/rpi_4/feature_names.json",
        },
        "rpi_zero": {
            "hidden_dims": [32, 16],
            "model_path": "models/rpi_zero/helix_rpi_zero.pt",
            "scaler_path": "models/rpi_zero/scaler.pkl",
            "features_path": "models/rpi_zero/feature_names.json",
        },
        "esp32": {
            "hidden_dims": [16, 8],
            "model_path": "models/esp32/helix_esp32.pt",
            "scaler_path": "models/esp32/scaler.pkl",
            "features_path": "models/esp32/feature_names.json",
        },
    }

    def __init__(self, platform: str = "production"):
        self.platform = platform
        self.config = self.PLATFORMS.get(platform)

        if not self.config:
            raise ValueError(
                f"Unknown platform: {platform}. Available: {list(self.PLATFORMS.keys())}"
            )

        self.model: HelixIDS | None = None
        self.scaler: Any = None
        self.feature_names: list[str] | None = None
        self._loaded = False

    def load(self):
        """Load model, scaler, and feature names"""
        if self._loaded:
            return

        # Check paths exist
        model_path = Path(self.config["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        for sidecar_path in [
            model_path.with_suffix(model_path.suffix + ".contract.json"),
            model_path.with_suffix(model_path.suffix + ".feature_order.json"),
            model_path.with_suffix(model_path.suffix + ".schema_hash.txt"),
        ]:
            if not sidecar_path.exists():
                raise RuntimeError(f"Missing deployment provenance sidecar: {sidecar_path}")

        # Load model
        self.model = HelixIDS(input_dim=32, hidden_dims=self.config["hidden_dims"])
        self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        self.model.eval()

        # Load scaler
        with open(self.config["scaler_path"], "rb") as f:
            self.scaler = pickle.load(f)

        # Load feature names
        with open(self.config["features_path"]) as f:
            self.feature_names = json.load(f)

        self._loaded = True
        print(
            f"✓ Loaded {self.platform} model ({sum(p.numel() for p in self.model.parameters()):,} params)"
        )

    def predict(self, X: np.ndarray) -> tuple:
        """Predict on input features"""
        self.load()
        assert self.scaler is not None
        assert self.model is not None

        # Scale
        x_scaled = self.scaler.transform(X)

        # Predict
        with torch.no_grad():
            logits = self.model(torch.FloatTensor(x_scaled))
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

        return preds.numpy(), probs.numpy()

    def predict_file(self, input_path: str, output_path: str):
        """Predict on CSV file"""
        print(f"\nLoading {input_path}...")
        df = pd.read_csv(input_path)

        assert self.feature_names is not None
        # Check features
        missing = set(self.feature_names) - set(df.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")

        X = df[self.feature_names].values

        print(f"Predicting on {len(X):,} samples...")
        preds, probs = self.predict(X)

        # Create output
        df["prediction"] = preds
        df["prediction_label"] = np.where(preds == 0, "Normal", "Attack")
        df["confidence"] = probs.max(axis=1)

        df.to_csv(output_path, index=False)
        print(f"✓ Saved predictions to {output_path}")

        # Summary
        attack_count = (preds == 1).sum()
        print("\nSummary:")
        print(f"  • Normal: {(preds == 0).sum():,} ({(preds == 0).mean():.1%})")
        print(f"  • Attack: {attack_count:,} ({(preds == 1).mean():.1%})")

    def benchmark(self, n_samples: int = 10000):
        """Benchmark inference speed"""
        self.load()
        assert self.scaler is not None
        assert self.model is not None

        print(f"\nBenchmarking {self.platform} model...")

        # Generate random input
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n_samples, 32), dtype=np.float32)
        x_scaled = self.scaler.transform(X)
        x_tensor = torch.FloatTensor(x_scaled)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(x_tensor[:100])

        # Benchmark
        start = time.perf_counter()
        with torch.no_grad():
            _ = self.model(x_tensor)
        elapsed = time.perf_counter() - start

        throughput = n_samples / elapsed
        latency_ms = elapsed / n_samples * 1000

        print(f"\nResults ({n_samples:,} samples):")
        print(f"  • Total time:  {elapsed * 1000:.2f} ms")
        print(f"  • Per sample:  {latency_ms:.4f} ms")
        print(f"  • Throughput:  {throughput:,.0f} samples/sec")

        return {"throughput": throughput, "latency_ms": latency_ms}

    def export_onnx(self, output_path: str):
        """Export to ONNX format"""
        self.load()

        assert self.model is not None
        dummy_input = torch.randn(1, 32)
        torch.onnx.export(
            self.model,
            (dummy_input,),
            output_path,
            input_names=["features"],
            output_names=["logits"],
            dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        )
        print(f"✓ Exported ONNX model to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="HELIX-IDS Deployment")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run predictions")
    predict_parser.add_argument("--input", required=True, help="Input CSV file")
    predict_parser.add_argument("--output", required=True, help="Output CSV file")
    predict_parser.add_argument(
        "--platform", default="production", help=HelixDeployer.MODEL_PLATFORM_HELP
    )

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark model")
    bench_parser.add_argument(
        "--platform", default="production", help=HelixDeployer.MODEL_PLATFORM_HELP
    )
    bench_parser.add_argument("--samples", type=int, default=10000, help="Number of samples")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export model")
    export_parser.add_argument(
        "--platform", default="production", help=HelixDeployer.MODEL_PLATFORM_HELP
    )
    export_parser.add_argument("--format", default="onnx", choices=["onnx"], help="Export format")
    export_parser.add_argument("--output", required=True, help="Output path")

    args = parser.parse_args()

    if args.command == "predict":
        deployer = HelixDeployer(args.platform)
        deployer.predict_file(args.input, args.output)

    elif args.command == "benchmark":
        deployer = HelixDeployer(args.platform)
        deployer.benchmark(args.samples)

    elif args.command == "export":
        deployer = HelixDeployer(args.platform)
        deployer.export_onnx(args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
