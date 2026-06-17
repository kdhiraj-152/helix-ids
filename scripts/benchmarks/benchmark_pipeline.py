#!/usr/bin/env python3
"""
Phase 22C — Performance Baseline Capture (C1).

Measures and records baseline performance metrics for the core HELIX-IDS
pipeline stages. Produces ``benchmarks/baseline.json``.

Stages
------
1. Data loading (single-thread)   — pd.read_csv over NSL-KDD train set
2. Data loading (multi-thread)    — concurrent.futures over data chunks
3. Feature engineering            — pipeline transforms on a data batch
4. Training step                  — forward + backward on HelixFull
5. Inference                      — model.eval() forward pass
6. Checkpoint save                — torch.save(state_dict)
7. Checkpoint load                — torch.load(state_dict)

Output schema (baseline.json)
-----------------------------
{
  "stages": { <name>: {"mean": ..., "median": ..., "p95": ..., "p99": ..., "stddev": ...} },
  "environment": {"cpu_model": "...", "ram_gb": ..., "gpu_model": "...", "torch_version": "..."},
  "ci_gates": {"training_step": 0.05, "inference": 0.05, "checkpoint_throughput": 0.10}
}
"""

# ruff: noqa: E402

import argparse
import gc
import json
import math
import platform
import subprocess
import sys
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
DATA_DIR = PROJECT_ROOT / "data"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ── Optional imports (graceful fallback for CI) ──────────────────────────────

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore[assignment]

# Lazily import package modules
_MODEL = None
_CONFIG = None


def _get_model():
    global _MODEL, _CONFIG
    if _MODEL is not None:
        return _MODEL, _CONFIG
    from helix_ids.models.full import HelixFullConfig, create_helix_full

    _CONFIG = HelixFullConfig()
    _MODEL = create_helix_full(_CONFIG)
    return _MODEL, _CONFIG


# ── Helpers ──────────────────────────────────────────────────────────────────


def _timed(fn, *, warmup: int = 2, repeats: int = 20) -> dict[str, float]:
    """Time *fn* with warmup rounds, return stats dict."""
    for _ in range(warmup):
        fn()

    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)

    arr = np.array(samples)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "stddev": float(np.std(arr, ddof=1)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "samples": samples,
    }


def _env_metadata() -> dict[str, Any]:
    """Capture environment metadata."""
    meta: dict[str, Any] = {"torch_version": torch.__version__ if torch else "N/A"}

    # CPU
    if sys.platform == "darwin":
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
        ram = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
        meta["cpu_model"] = cpu or platform.processor()
        meta["ram_gb"] = round(int(ram) / 1e9, 1) if ram.isdigit() else 0.0
    else:
        meta["cpu_model"] = platform.processor()
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        meta["ram_gb"] = round(int(line.split()[1]) / 1e6, 1)
                        break
        except OSError:
            meta["ram_gb"] = 0.0

    # GPU
    if torch and torch.cuda.is_available():
        meta["gpu_model"] = torch.cuda.get_device_name(0)
        meta["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1
        )
    elif torch and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        meta["gpu_model"] = "MPS (Apple Silicon)"
    else:
        meta["gpu_model"] = "N/A"

    meta["python_version"] = platform.python_version()
    meta["platform"] = platform.platform()
    return meta


# ── Stage implementations ────────────────────────────────────────────────────


def _bench_data_loading_single(data_path: str | Path):
    """Single-thread: pd.read_csv on the full NSL-KDD train set."""
    if pd is None:
        raise RuntimeError("pandas not available")
    df = pd.read_csv(data_path)
    # Touch every column to force full materialisation
    df.describe()


def _bench_data_loading_multi(data_path: str | Path):
    """Multi-thread: split file into N chunks, load concurrently."""
    if pd is None:
        raise RuntimeError("pandas not available")
    N = 4
    total_lines = sum(1 for _ in open(data_path)) - 1  # header subtracted
    chunk_size = total_lines // N

    def _load_chunk(skip: int, nrows: int) -> object:
        return pd.read_csv(data_path, skiprows=range(1, skip + 1), nrows=nrows)

    with ThreadPoolExecutor(max_workers=N) as pool:
        futures = []
        for i in range(N):
            skip = 1 + i * chunk_size
            nrows = chunk_size if i < N - 1 else total_lines - i * chunk_size
            futures.append(pool.submit(_load_chunk, skip, nrows))
        for f in as_completed(futures):
            f.result()


def _bench_feature_engineering(batch_size: int = 4096):
    """Run the feature engineering pipeline on a synthetic batch."""
    from helix_ids.data.feature_engineering import FeatureEngineer  # type: ignore[import-untyped]

    engineer = FeatureEngineer()
    columns = [f"feature_{i}" for i in range(41)]
    X = pd.DataFrame(np.random.randn(batch_size, 41).astype(np.float32), columns=columns)
    engineer.engineer_all_features(X)


def _bench_training_step(batch_size: int = 256):
    """One forward + backward pass on HelixFull."""
    model, config = _get_model()
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    x = torch.randn(batch_size, config.input_dim)
    labels_bin = torch.randint(0, 2, (batch_size,))
    labels_fam = torch.randint(0, 7, (batch_size,))

    optimizer.zero_grad()
    bin_out, fam_out = model(x)
    loss = nn.functional.cross_entropy(bin_out, labels_bin) + nn.functional.cross_entropy(fam_out, labels_fam)
    loss.backward()
    optimizer.step()


def _bench_inference(batch_size: int = 256):
    """Model.eval() forward pass on HelixFull."""
    model, config = _get_model()
    model.eval()
    x = torch.randn(batch_size, config.input_dim)
    with torch.no_grad():
        model(x)


def _bench_checkpoint_save(tmpdir: Path, model):
    """torch.save model state_dict."""
    path = tmpdir / "_bench_ckpt.pt"
    torch.save(model.state_dict(), path)


def _bench_checkpoint_load(tmpdir: Path, model):
    """torch.load model state_dict."""
    path = tmpdir / "_bench_ckpt.pt"
    state = torch.load(path, weights_only=True, map_location="cpu")
    model.load_state_dict(state)


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 22C — Performance Baseline Capture (C1).",
    )
    parser.add_argument(
        "--output",
        default=BENCHMARKS_DIR / "baseline.json",
        type=Path,
        help="Output path for baseline JSON (default: benchmarks/baseline.json)",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()

    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    tmpdir = BENCHMARKS_DIR / "_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)

    # Determine data path
    data_path = DATA_DIR / "nsl_kdd" / "train.csv"
    if not data_path.exists():
        print(f"[WARN] NSL-KDD data not found at {data_path}; using synthetic data for load benchmarks")
        data_path = None

    print("=" * 60)
    print("Phase 22C — Performance Baseline Capture")
    print("=" * 60)

    model, _ = _get_model()

    results: dict[str, dict[str, Any]] = {}

    # ── Stage 1: Data loading (single-thread) ────────────────────────────────
    if data_path:
        print("\n[1/7] Data loading (single-thread) …")
        stats = _timed(lambda: _bench_data_loading_single(data_path), warmup=1, repeats=10)
        results["data_loading_single"] = stats
        print(f"      mean={stats['mean']:.4f}s  median={stats['median']:.4f}s  p95={stats['p95']:.4f}s")
    else:
        print("\n[1/7] Data loading (single-thread) … SKIPPED (no data)")

    # ── Stage 2: Data loading (multi-thread) ─────────────────────────────────
    if data_path:
        print("[2/7] Data loading (multi-thread) …")
        stats = _timed(lambda: _bench_data_loading_multi(data_path), warmup=1, repeats=10)
        results["data_loading_multi"] = stats
        print(f"      mean={stats['mean']:.4f}s  median={stats['median']:.4f}s  p95={stats['p95']:.4f}s")
    else:
        print("[2/7] Data loading (multi-thread) … SKIPPED (no data)")

    # ── Stage 3: Feature engineering ─────────────────────────────────────────
    print("[3/7] Feature engineering …")
    stats = _timed(lambda: _bench_feature_engineering(), warmup=2, repeats=15)
    results["feature_engineering"] = stats
    print(f"      mean={stats['mean']:.4f}s  median={stats['median']:.4f}s  p95={stats['p95']:.4f}s")

    # ── Stage 4: Training step ───────────────────────────────────────────────
    print("[4/7] Training step …")
    gc.collect()
    stats = _timed(lambda: _bench_training_step(), warmup=3, repeats=30)
    results["training_step"] = stats
    print(f"      mean={stats['mean']:.4f}s  median={stats['median']:.4f}s  p95={stats['p95']:.4f}s")

    # ── Stage 5: Inference ───────────────────────────────────────────────────
    print("[5/7] Inference …")
    gc.collect()
    stats = _timed(lambda: _bench_inference(), warmup=3, repeats=30)
    results["inference"] = stats
    print(f"      mean={stats['mean']:.4f}s  median={stats['median']:.4f}s  p95={stats['p95']:.4f}s")

    # ── Stage 6: Checkpoint save ─────────────────────────────────────────────
    print("[6/7] Checkpoint save …")
    gc.collect()
    stats = _timed(lambda: _bench_checkpoint_save(tmpdir, model), warmup=2, repeats=20)
    results["checkpoint_save"] = stats
    print(f"      mean={stats['mean']:.4f}s  median={stats['median']:.4f}s  p95={stats['p95']:.4f}s")

    # ── Stage 7: Checkpoint load ─────────────────────────────────────────────
    print("[7/7] Checkpoint load …")
    gc.collect()
    stats = _timed(lambda: _bench_checkpoint_load(tmpdir, model), warmup=2, repeats=20)
    results["checkpoint_load"] = stats
    print(f"      mean={stats['mean']:.4f}s  median={stats['median']:.4f}s  p95={stats['p95']:.4f}s")

    # Compute derived: checkpoint throughput (MB/s)
    if "checkpoint_save" in results:
        ckpt_size = model.state_dict().__len__()
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        ckpt_mb = param_bytes / (1024 * 1024)
        save_mean = results["checkpoint_save"]["mean"]
        results["checkpoint_save"]["throughput_mbps"] = round(ckpt_mb / save_mean, 2) if save_mean > 0 else 0.0
        results["checkpoint_save"]["checkpoint_size_mb"] = round(ckpt_mb, 2)

    if "checkpoint_load" in results:
        ckpt_size = results["checkpoint_save"]["checkpoint_size_mb"]
        load_mean = results["checkpoint_load"]["mean"]
        results["checkpoint_load"]["throughput_mbps"] = round(ckpt_size / load_mean, 2) if load_mean > 0 else 0.0

    # Cleanup tmp
    for f in tmpdir.iterdir():
        f.unlink()
    tmpdir.rmdir()

    # ── Environment ──────────────────────────────────────────────────────────
    env = _env_metadata()
    results["environment"] = env

    # ── CI gates ─────────────────────────────────────────────────────────────
    gates = {
        "training_step_regression_pct": 5.0,
        "inference_regression_pct": 5.0,
        "checkpoint_throughput_regression_pct": 10.0,
    }
    results["ci_gates"] = gates

    # ── Write baseline.json ──────────────────────────────────────────────────
    baseline_path = args.output
    # Strip raw samples from the output to keep it compact
    output = {}
    for stage, data in results.items():
        if stage in ("environment", "ci_gates"):
            output[stage] = data
        else:
            d = {k: v for k, v in data.items() if k != "samples"}
            output[stage] = d

    with open(baseline_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n✓ Baseline written to {baseline_path}")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BASELINE SUMMARY")
    print("=" * 60)
    print(f"{'Stage':<30} {'Mean (s)':<12} {'p95 (s)':<12} {'p99 (s)':<12}")
    print("-" * 66)
    for stage, data in results.items():
        if stage in ("environment", "ci_gates"):
            continue
        print(f"{stage:<30} {data.get('mean', 0):<12.4f} {data.get('p95', 0):<12.4f} {data.get('p99', 0):<12.4f}")

    print("\nEnvironment:")
    for k, v in env.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
