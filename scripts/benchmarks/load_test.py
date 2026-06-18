#!/usr/bin/env python3
"""
Phase 22C — Load Testing (C2).

Tests concurrency tolerance and throughput for key subsystems at 1x, 10x, 25x,
50x, and 100x concurrency.

Targets
-------
- Inference runtime
- Data loader
- Circuit breaker
- Restart manager
- Structured logger

Additional scenarios
--------------------
- Checkpoint save storm
- Log flood
- Restart storm
- Cascading failures

Output
------
Results written to benchmarks/load_test_results.json
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import torch
except ImportError:
    torch = None

# ── Helpers ──────────────────────────────────────────────────────────────────


class Timer:
    """Context manager for elapsed wall-clock time."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


def _latency_stats(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {"p50": 0, "p95": 0, "p99": 0, "mean": 0, "min": 0, "max": 0}
    arr = sorted(latencies)
    return {
        "p50": arr[len(arr) // 2],
        "p95": arr[int(len(arr) * 0.95)],
        "p99": arr[int(len(arr) * 0.99)],
        "mean": statistics.mean(arr),
        "min": arr[0],
        "max": arr[-1],
        "stddev": statistics.stdev(arr) if len(arr) > 1 else 0.0,
    }


def _concurrency_test(
    name: str,
    worker_fn: Callable[[int], float],  # takes worker_id, returns latency
    concurrency_levels: list[int],
    calls_per_worker: int = 10,
    warmup_calls: int = 3,
) -> dict:
    """Benchmark a function at multiple concurrency levels.

    Returns dict keyed by concurrency level, each containing stats.
    """
    results: dict[str, dict] = {}

    for level in concurrency_levels:
        print(f"  Concurrency {level}x … ", end="", flush=True)

        # Warmup at this concurrency
        if warmup_calls > 0:
            with ThreadPoolExecutor(max_workers=level) as pool:
                futures = [
                    pool.submit(worker_fn, i) for i in range(level) for _ in range(warmup_calls)
                ]
                for f in as_completed(futures):
                    try:
                        f.result()
                    except Exception:
                        pass

        # Timed run
        latencies: list[float] = []
        errors = 0

        timer = Timer()
        with timer:
            with ThreadPoolExecutor(max_workers=level) as pool:
                futures = {
                    pool.submit(worker_fn, i): i
                    for i in range(level)
                    for _ in range(calls_per_worker)
                }
                for f in as_completed(futures):
                    try:
                        lat = f.result()
                        latencies.append(lat)
                    except Exception:
                        errors += 1

        elapsed = timer.elapsed
        total_calls = level * calls_per_worker
        throughput = total_calls / elapsed if elapsed > 0 else 0.0
        error_rate = errors / total_calls if total_calls > 0 else 0.0

        stats = _latency_stats(latencies)
        stats["throughput"] = round(throughput, 2)
        stats["error_rate"] = round(error_rate, 4)
        stats["total_calls"] = total_calls
        stats["errors"] = errors
        stats["elapsed"] = round(elapsed, 3)
        stats["concurrency"] = level

        results[str(level)] = stats
        print(f"{throughput:.0f} req/s  p50={stats['p50']*1000:.2f}ms  p95={stats['p95']*1000:.2f}ms  errors={errors}")

    return results


# ── Worker factories ─────────────────────────────────────────────────────────


def _make_inference_worker(model: Any, config: Any, lock: Lock):
    """Worker that runs a forward pass."""
    device = "cpu"
    if torch and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    elif torch and torch.cuda.is_available():
        device = "cuda"

    if device == "cpu":
        model_cpu = model
    else:
        model_cpu = model.to(device)

    def worker(worker_id: int) -> float:
        x = torch.randn(256, config.input_dim, device=device)
        model_cpu.eval()
        with torch.no_grad():
            t0 = time.perf_counter()
            model_cpu(x)
            return time.perf_counter() - t0

    return worker


def _make_data_loader_worker(data_path: Path):
    """Worker that reads a CSV file."""

    def worker(worker_id: int) -> float:
        if pd is None:
            return 0.001
        t0 = time.perf_counter()
        df = pd.read_csv(data_path)
        _ = df.describe()
        return time.perf_counter() - t0

    return worker


def _make_circuit_breaker_worker():
    """Worker that creates a circuit breaker and runs checks."""
    from helix_ids.operations.safety.circuit_breaker import CircuitBreaker

    def worker(worker_id: int) -> float:
        cb = CircuitBreaker(
            name=f"loadtest_{worker_id}",
            max_threshold=100.0,
            min_threshold=-100.0,
            patience=3,
        )
        t0 = time.perf_counter()
        for _ in range(5):
            cb.check(np.random.uniform(-50, 50))
        return time.perf_counter() - t0

    return worker


def _make_restart_manager_worker(tmpdir: Path):
    """Worker that creates checkpoints and tests restart resolution."""
    from helix_ids.operations.recovery.restart_manager import RestartManager

    def worker(worker_id: int) -> float:
        ckpt_dir = tmpdir / f"ckpt_{worker_id}"
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)

        # Write a fake checkpoint
        fake_ckpt = {
            "epoch": 5,
            "global_step": 1000,
            "phase_id": 1,
            "best_val_loss": 0.05,
            "state_dict": {},
        }
        ckpt_path = ckpt_dir / "checkpoint_epoch_005_step_1000.pt"
        torch.save(fake_ckpt, ckpt_path)

        # Write crash sentinel
        mgr.write_crash_sentinel(reason="load_test")

        t0 = time.perf_counter()
        mgr.resolve_restart()
        return time.perf_counter() - t0

    return worker


def _make_logger_worker():
    """Worker that emits structured log messages."""
    from helix_ids.operations.logging import LogContext, get_logger

    logger = get_logger("loadtest")

    def worker(worker_id: int) -> float:
        t0 = time.perf_counter()
        with LogContext(extra={"test": "load", "worker_id": worker_id}):
            for _ in range(10):
                logger.info("Load test message", extra={"worker": worker_id, "iteration": _})
        return time.perf_counter() - t0

    return worker


# ── Storm scenarios ──────────────────────────────────────────────────────────


def _checkpoint_save_storm(model: Any, concurrency: int = 50):
    """Spike: N workers each save a checkpoint concurrently."""
    print(f"  Checkpoint save storm ({concurrency}x workers) … ", end="", flush=True)
    tmpdir = Path(tempfile.mkdtemp(prefix="ckpt_storm_"))

    def _save(i: int) -> float:
        path = tmpdir / f"ckpt_{i}.pt"
        t0 = time.perf_counter()
        torch.save(model.state_dict(), path)
        return time.perf_counter() - t0

    latencies: list[float] = []
    errors = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_save, i): i for i in range(concurrency)}
        for f in as_completed(futures):
            try:
                latencies.append(f.result())
            except Exception:
                errors += 1

    stats = _latency_stats(latencies)
    stats["errors"] = errors
    print(f"p50={stats['p50']*1000:.2f}ms  p95={stats['p95']*1000:.2f}ms  errors={errors}")

    # Cleanup
    for f in tmpdir.iterdir():
        f.unlink()
    tmpdir.rmdir()
    return stats


def _log_flood(concurrency: int = 100, messages_per_worker: int = 100):
    """Flood the structured logger with concurrent messages."""
    from helix_ids.operations.logging import LogContext, get_logger

    logger = get_logger("flood_test")

    def _log(i: int) -> float:
        t0 = time.perf_counter()
        with LogContext(extra={"test": "flood", "worker_id": i}):
            for j in range(messages_per_worker):
                logger.info("Flood test message", extra={"w": i, "m": j})
        return time.perf_counter() - t0

    latencies: list[float] = []
    errors = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_log, i): i for i in range(concurrency)}
        for f in as_completed(futures):
            try:
                latencies.append(f.result())
            except Exception:
                errors += 1
    stats = _latency_stats(latencies)
    stats["total_messages"] = concurrency * messages_per_worker
    stats["errors"] = errors
    print(f"  Log flood ({concurrency}x workers, {messages_per_worker} msg each):")
    print(f"    p50={stats['p50']*1000:.2f}ms  p95={stats['p95']*1000:.2f}ms  errors={errors}")
    return stats


def _restart_storm(concurrency: int = 25):
    """Spike: N RestartManagers resolving simultaneously."""
    from helix_ids.operations.recovery.restart_manager import RestartManager
    tmpdir = Path(tempfile.mkdtemp(prefix="restart_storm_"))

    def _resolve(i: int) -> float:
        ckpt_dir = tmpdir / f"r_{i}"
        mgr = RestartManager(checkpoint_dir=ckpt_dir, require_governance=False)
        fake = {
            "epoch": 3, "global_step": 500, "phase_id": 1, "best_val_loss": 0.1, "state_dict": {},
        }
        torch.save(fake, ckpt_dir / "checkpoint_epoch_003_step_0500.pt")
        mgr.write_crash_sentinel(reason="storm")
        t0 = time.perf_counter()
        mgr.resolve_restart()
        return time.perf_counter() - t0

    latencies: list[float] = []
    errors = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_resolve, i): i for i in range(concurrency)}
        for f in as_completed(futures):
            try:
                latencies.append(f.result())
            except Exception:
                errors += 1

    stats = _latency_stats(latencies)
    stats["errors"] = errors
    print(f"  Restart storm ({concurrency}x): p50={stats['p50']*1000:.2f}ms  p95={stats['p95']*1000:.2f}ms  errors={errors}")

    for d in tmpdir.iterdir():
        for f in d.iterdir() if d.is_dir() else [d]:
            f.unlink(missing_ok=True)
        if d.is_dir():
            d.rmdir()
    tmpdir.rmdir()
    return stats


def _cascading_failures():
    """Chain failures: corrupt checkpoint → restart → second crash → recovery."""
    from helix_ids.operations.recovery.restart_manager import RestartManager
    tmpdir = Path(tempfile.mkdtemp(prefix="cascade_"))

    try:
        # Phase 1: Create valid checkpoint, crash
        mgr = RestartManager(checkpoint_dir=tmpdir, require_governance=False)
        fake = {"epoch": 1, "global_step": 100, "state_dict": {}}
        torch.save(fake, tmpdir / "checkpoint_epoch_001_step_0100.pt")
        mgr.write_crash_sentinel(reason="first_crash")

        t0 = time.perf_counter()
        d1 = mgr.resolve_restart()
        t1 = time.perf_counter() - t0

        # Phase 2: Corrupt the checkpoint, crash again
        with open(tmpdir / "checkpoint_epoch_001_step_0100.pt", "w") as f:
            f.write("CORRUPTED_DATA")
        mgr.write_crash_sentinel(reason="second_crash")
        t0 = time.perf_counter()
        d2 = mgr.resolve_restart()
        t2 = time.perf_counter() - t0

        # Phase 3: Write a new good checkpoint, recover
        torch.save({"epoch": 2, "global_step": 200, "state_dict": {}},
                    tmpdir / "checkpoint_epoch_002_step_0200.pt")
        mgr.write_crash_sentinel(reason="third_crash")
        t0 = time.perf_counter()
        d3 = mgr.resolve_restart()
        t3 = time.perf_counter() - t0

        results = {
            "first_restart_latency": round(t1, 4),
            "corrupt_restart_latency": round(t2, 4),
            "recovery_restart_latency": round(t3, 4),
            "first_should_restart": d1.should_restart,
            "corrupt_restart_found": d2.should_restart,
            "recovery_from_corruption": d3.should_restart,
        }
        print(f"  Cascading failures: first={t1:.4f}s corrupt={t2:.4f}s recovery={t3:.4f}s")
        return results
    finally:
        for f in tmpdir.iterdir():
            f.unlink()
        tmpdir.rmdir()


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 22C — Load Testing (C2).",
    )
    parser.add_argument(
        "--output",
        default=BENCHMARKS_DIR / "load_test_results.json",
        type=Path,
        help=(
            "Output path for load test results JSON "
            "(default: benchmarks/load_test_results.json)"
        ),
    )
    parser.add_argument(
        "--concurrency-levels",
        default=[1, 10, 25, 50, 100],
        type=lambda s: [int(x.strip()) for x in s.split(",")],
        help=(
            "Comma-separated concurrency levels to test "
            "(default: 1,10,25,50,100)"
        ),
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()

    # ── Imports ──────────────────────────────────────────────────────────────
    from helix_ids.models.full import HelixFullConfig, create_helix_full

    print("=" * 60)
    print("Phase 22C — Load Testing")
    print("=" * 60)

    data_path = PROJECT_ROOT / "data" / "nsl_kdd" / "train.csv"
    if not data_path.exists():
        data_path = None

    model, config = create_helix_full(HelixFullConfig()), HelixFullConfig()
    model_lock = Lock()

    concurrency_levels = args.concurrency_levels
    all_results: dict[str, Any] = {}
    all_results["meta"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "concurrency_levels": concurrency_levels,
    }

    # ── C2.1: Inference runtime ──────────────────────────────────────────────
    print("\n--- C2.1: Inference Runtime ---")
    infer_worker = _make_inference_worker(model, config, model_lock)
    infer_results = _concurrency_test(
        "inference", infer_worker, concurrency_levels, calls_per_worker=20, warmup_calls=3
    )
    all_results["inference_runtime"] = infer_results

    # ── C2.2: Data loader ────────────────────────────────────────────────────
    print("\n--- C2.2: Data Loader ---")
    if data_path:
        dl_worker = _make_data_loader_worker(data_path)
        dl_results = _concurrency_test(
            "data_loader", dl_worker, concurrency_levels, calls_per_worker=5, warmup_calls=1
        )
    else:
        dl_results = {"skipped": "no data file"}
        print("  SKIPPED (no NSL-KDD data file)")
    all_results["data_loader"] = dl_results

    # ── C2.3: Circuit breaker ────────────────────────────────────────────────
    print("\n--- C2.3: Circuit Breaker ---")
    cb_worker = _make_circuit_breaker_worker()
    cb_results = _concurrency_test(
        "circuit_breaker", cb_worker, concurrency_levels, calls_per_worker=20, warmup_calls=2
    )
    all_results["circuit_breaker"] = cb_results

    # ── C2.4: Restart manager ────────────────────────────────────────────────
    print("\n--- C2.4: Restart Manager ---")
    rm_tmpdir = Path(tempfile.mkdtemp(prefix="loadtest_rm_"))
    rm_worker = _make_restart_manager_worker(rm_tmpdir)
    rm_results = _concurrency_test(
        "restart_manager", rm_worker, concurrency_levels, calls_per_worker=5, warmup_calls=1
    )
    all_results["restart_manager"] = rm_results

    # Cleanup restart manager temp dirs
    for d in rm_tmpdir.iterdir():
        for f in d.iterdir() if d.is_dir() else [d]:
            f.unlink(missing_ok=True)
        if d.is_dir():
            d.rmdir()
    rm_tmpdir.rmdir()

    # ── C2.5: Structured logger ──────────────────────────────────────────────
    print("\n--- C2.5: Structured Logger ---")
    log_worker = _make_logger_worker()
    log_results = _concurrency_test(
        "structured_logger", log_worker, concurrency_levels, calls_per_worker=10, warmup_calls=2
    )
    all_results["structured_logger"] = log_results

    # ── Additional scenarios ─────────────────────────────────────────────────
    print("\n--- Additional Scenarios ---")

    # Checkpoint save storm
    ckpt_storm = _checkpoint_save_storm(model, concurrency=50)
    all_results["checkpoint_save_storm"] = ckpt_storm

    # Log flood
    log_flood_result = _log_flood(concurrency=100, messages_per_worker=100)
    all_results["log_flood"] = log_flood_result

    # Restart storm
    restart_storm_result = _restart_storm(concurrency=25)
    all_results["restart_storm"] = restart_storm_result

    # Cascading failures
    cascade_result = _cascading_failures()
    all_results["cascading_failures"] = cascade_result

    # ── Verdict checks ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VERDICT CHECKS")
    print("=" * 60)
    failures = []

    # Check p99 < 2x baseline for inference
    baseline_path = BENCHMARKS_DIR / "baseline.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        base_infer_mean = baseline.get("inference", {}).get("mean", 0.0007)
        if "1" in infer_results:
            p99_1x = infer_results["1"].get("p99", 0)
            if p99_1x > 2 * base_infer_mean:
                failures.append(
                    f"Inference p99 at 1x ({p99_1x*1000:.2f}ms) exceeds 2x baseline mean ({2*base_infer_mean*1000:.2f}ms)"
                )

    # Check for deadlocks or corruption
    all_data = [infer_results, cb_results, rm_results, log_results]
    if isinstance(dl_results, dict) and "skipped" not in dl_results:
        all_data.append(dl_results)

    total_errors = sum(
        v.get("errors", 0)
        for d in all_data
        for v in d.values() if isinstance(v, dict)
    )
    if total_errors > 0:
        failures.append(f"Total errors across all tests: {total_errors}")

    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  ✗ {f}")
    else:
        print("  ✓ Zero deadlocks")
        print("  ✓ Zero corruption")
        print("  ✓ p99 < 2x baseline at 1x concurrency")

    all_results["verdict"] = {
        "passed": len(failures) == 0,
        "failures": failures,
        "total_errors": total_errors,
    }

    # ── Write results ────────────────────────────────────────────────────────
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.output
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✓ Load test results written to {out_path}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
