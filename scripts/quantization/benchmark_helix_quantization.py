"""
Phase 4: Benchmark FP32 vs Lite INT8 vs Micro INT8 artifacts.

Benchmarks latency and prediction agreement on synthetic or optional
saved processed splits.

Usage:
  python scripts/benchmark_helix_quantization.py \
    --full-checkpoint models/helix_full/helix_full_best.pt \
    --lite-checkpoint models/quantized/helix_ids_lite_int8.pt \
    --micro-checkpoint models/quantized/helix_ids_micro_int8.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from helix_ids.models.full import HelixFullConfig, HelixIDSFull


def _load_fp32(path: Path) -> torch.nn.Module:
    model = HelixIDSFull(HelixFullConfig())
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return cast(torch.nn.Module, model)


def _load_quantized(path: Path) -> torch.nn.Module:
    model = cast(torch.nn.Module, torch.load(path, map_location="cpu", weights_only=False))
    model.eval()
    return model


def _time_model(model: torch.nn.Module, x: torch.Tensor, rounds: int = 50) -> dict[str, float]:
    with torch.no_grad():
        _ = model(x)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(rounds):
            _ = model(x)
    elapsed = time.perf_counter() - start

    samples = x.shape[0] * rounds
    return {
        "total_seconds": elapsed,
        "samples_per_second": samples / elapsed,
        "ms_per_batch": (elapsed / rounds) * 1000.0,
    }


def _pred_labels(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        binary_logits, family_logits = model(x)
        binary = torch.argmax(binary_logits, dim=1)
        family = torch.argmax(family_logits, dim=1)
        labels = torch.where(binary == 0, torch.zeros_like(family), family)
    return labels.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Helix quantized variants")
    parser.add_argument("--full-checkpoint", type=Path, required=True)
    parser.add_argument("--lite-checkpoint", type=Path, required=True)
    parser.add_argument("--micro-checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument(
        "--output", type=Path, default=Path("results/benchmarks/helix_quantization_benchmark.json")
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fp32 = _load_fp32(args.full_checkpoint)
    lite = _load_quantized(args.lite_checkpoint)
    micro = _load_quantized(args.micro_checkpoint)

    x = torch.randn(args.batch_size, 31)

    fp32_metrics = _time_model(fp32, x, rounds=args.rounds)
    lite_metrics = _time_model(lite, x, rounds=args.rounds)
    micro_metrics = _time_model(micro, x, rounds=args.rounds)

    p_full = _pred_labels(fp32, x)
    p_lite = _pred_labels(lite, x)
    p_micro = _pred_labels(micro, x)

    report = {
        "batch_size": args.batch_size,
        "rounds": args.rounds,
        "latency": {
            "fp32": fp32_metrics,
            "lite_int8": lite_metrics,
            "micro_int8": micro_metrics,
        },
        "agreement": {
            "lite_vs_fp32": float(np.mean(p_lite == p_full)),
            "micro_vs_fp32": float(np.mean(p_micro == p_full)),
            "micro_vs_lite": float(np.mean(p_micro == p_lite)),
        },
    }

    with open(args.output, "w", encoding="ascii") as f:
        json.dump(report, f, indent=2)

    print(f"Saved benchmark report: {args.output}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
