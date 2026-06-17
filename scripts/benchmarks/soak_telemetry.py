#!/usr/bin/env python3
"""
Soak telemetry collector (C3).

Collects hourly resource snapshots during long-running soak tests.

Metrics captured:
  - RSS memory (MB)
  - GPU memory (MB, if available)
  - File handles
  - Threads
  - GC stats (objects by generation)
  - Tensor count (torch tensors, if available)
  - Checkpoint count
  - Log volume (bytes written)
  - Latency (p50, p95, p99)
  - Throughput

Snapshots written to: artifacts/soak/<run_id>/
"""

import gc
import json
import os
import platform
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "soak"


# ── Resource collectors ──────────────────────────────────────────────────────


def _rss_memory_mb() -> float:
    """Return current RSS memory in MB (cross-platform)."""
    try:
        # macOS / Linux
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        try:
            # Linux /proc fallback
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        except Exception:
            pass
        return 0.0


def _gpu_memory_mb() -> dict[str, float]:
    """Return GPU memory usage in MB per device."""
    result: dict[str, float] = {}
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
                reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)
                result[f"cuda:{i}"] = {"allocated_mb": allocated, "reserved_mb": reserved}
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS doesn't expose memory APIs, report placeholder
            result["mps"] = {"allocated_mb": -1, "reserved_mb": -1}
    except ImportError:
        pass
    return result


def _file_handles() -> int:
    """Return open file handle count."""
    try:
        import psutil

        return psutil.Process(os.getpid()).num_fds()
    except (ImportError, AttributeError):
        try:
            return len(os.listdir(f"/proc/{os.getpid()}/fd"))
        except Exception:
            return -1


def _thread_count() -> int:
    """Return active thread count."""
    return threading.active_count()


def _gc_stats() -> dict[str, int]:
    """Return GC object counts per generation."""
    return {
        "gen0": gc.get_count()[0],
        "gen1": gc.get_count()[1],
        "gen2": gc.get_count()[2],
        "threshold": list(gc.get_threshold()),
        "total_collected": sum(gc.get_stats()[g]["collected"] for g in range(3)),
    }


def _tensor_count() -> int:
    """Return number of live torch tensors (if available)."""
    try:
        import torch

        return sum(1 for _ in torch._C._tensor_impls())
    except (ImportError, AttributeError):
        return -1


def _checkpoint_count(checkpoint_dir: str | Path | None) -> int:
    """Count checkpoint .pt files in directory."""
    if checkpoint_dir is None:
        return -1
    p = Path(checkpoint_dir)
    if not p.exists():
        return 0
    return len(list(p.glob("*.pt")))


def _log_volume(log_dir: str | Path | None) -> int:
    """Count total bytes in log files."""
    if log_dir is None:
        return -1
    p = Path(log_dir)
    if not p.exists():
        return 0
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


# ── Telemetry snapshot ───────────────────────────────────────────────────────


def collect_snapshot(
    run_id: str,
    *,
    checkpoint_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
    custom_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Collect a single telemetry snapshot."""
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "uptime_seconds": time.time(),
        "rss_memory_mb": _rss_memory_mb(),
        "gpu_memory": _gpu_memory_mb(),
        "file_handles": _file_handles(),
        "threads": _thread_count(),
        "gc": _gc_stats(),
        "tensor_count": _tensor_count(),
        "checkpoint_count": _checkpoint_count(checkpoint_dir),
        "log_volume_bytes": _log_volume(log_dir),
    }
    if custom_metrics:
        snapshot["custom"] = custom_metrics
    return snapshot


def write_snapshot(snapshot: dict[str, Any], run_id: str) -> Path:
    """Write a telemetry snapshot to artifacts/soak/<run_id>/."""
    out_dir = ARTIFACTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    t = snapshot["timestamp"].replace(":", "-")
    path = out_dir / f"snapshot_{t}.json"
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    return path


# ── Continuous collector ─────────────────────────────────────────────────────


def collect_loop(
    run_id: str,
    *,
    interval_seconds: int = 3600,
    total_hours: int = 24,
    checkpoint_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
    latency_queue: list[float] | None = None,
    throughput_queue: list[float] | None = None,
) -> None:
    """Run the telemetry collection loop for the soak duration.

    Parameters
    ----------
    run_id : str
        Unique run identifier.
    interval_seconds : int
        Seconds between snapshots (default 3600 = 1 hour).
    total_hours : int
        Total run duration in hours (default 24).
    checkpoint_dir : str or Path or None
        Directory to monitor for checkpoint counts.
    log_dir : str or Path or None
        Directory to monitor for log volume.
    latency_queue : list[float] or None
        Shared list of observed latencies since last snapshot.
    throughput_queue : list[float] or None
        Shared list of throughput measurements since last snapshot.
    """
    total_snapshots = int(total_hours * 3600 / interval_seconds)
    print(f"Telemetry collector starting: run={run_id} interval={interval_seconds}s "
          f"snapshots={total_snapshots}")

    for i in range(total_snapshots):
        # Compute custom metrics from queues
        custom: dict[str, float] = {}
        if latency_queue:
            lats = list(latency_queue)
            latency_queue.clear()
            if lats:
                lats.sort()
                custom["latency_p50_ms"] = lats[len(lats) // 2] * 1000
                custom["latency_p95_ms"] = lats[int(len(lats) * 0.95)] * 1000
                custom["latency_p99_ms"] = lats[int(len(lats) * 0.99)] * 1000
        if throughput_queue:
            tputs = list(throughput_queue)
            throughput_queue.clear()
            if tputs:
                custom["throughput_mean"] = sum(tputs) / len(tputs)

        snapshot = collect_snapshot(
            run_id,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            custom_metrics=custom or None,
        )
        path = write_snapshot(snapshot, run_id)
        print(f"  [{i+1}/{total_snapshots}] Snapshot: {path}  "
              f"RSS={snapshot['rss_memory_mb']:.0f}MB "
              f"threads={snapshot['threads']} "
              f"tensors={snapshot['tensor_count']}")

        if interval_seconds < 3600 and i < total_snapshots - 1:
            time.sleep(interval_seconds)

    print(f"Telemetry collector complete: {total_snapshots} snapshots to {ARTIFACTS_DIR / run_id}")


def summarize_run(run_id: str) -> dict[str, Any]:
    """Summarise a completed soak run's telemetry."""
    run_dir = ARTIFACTS_DIR / run_id
    if not run_dir.exists():
        return {"error": f"No data for run {run_id}"}

    snapshots = sorted(run_dir.glob("snapshot_*.json"))
    if not snapshots:
        return {"error": f"No snapshots found in {run_dir}"}

    data: list[dict] = []
    for sp in snapshots:
        with open(sp) as f:
            data.append(json.load(f))

    rss_values = [d.get("rss_memory_mb", 0) for d in data]
    trends = {
        "snapshot_count": len(data),
        "rss_memory": {
            "start_mb": rss_values[0],
            "end_mb": rss_values[-1],
            "min_mb": min(rss_values),
            "max_mb": max(rss_values),
            "mean_mb": sum(rss_values) / len(rss_values),
            "trend": "flat" if abs(rss_values[-1] - rss_values[0]) < 50
            else "rising" if rss_values[-1] > rss_values[0]
            else "falling",
        },
        "threads": {
            "start": data[0].get("threads", 0),
            "end": data[-1].get("threads", 0),
        },
        "checkpoint_count": {
            "start": data[0].get("checkpoint_count", 0),
            "end": data[-1].get("checkpoint_count", 0),
        },
        "tensor_count": {
            "start": data[0].get("tensor_count", 0),
            "end": data[-1].get("tensor_count", 0),
        },
        "gc_total_collected": {
            "start": data[0].get("gc", {}).get("total_collected", 0),
            "end": data[-1].get("gc", {}).get("total_collected", 0),
        },
    }
    return trends


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Soak telemetry collector")
    parser.add_argument("--run-id", required=True, help="Unique run identifier")
    parser.add_argument("--interval", type=int, default=3600, help="Seconds between snapshots")
    parser.add_argument("--hours", type=int, default=24, help="Total run duration in hours")
    parser.add_argument("--checkpoint-dir", help="Checkpoint directory to monitor")
    parser.add_argument("--log-dir", help="Log directory to monitor")
    args = parser.parse_args()

    collect_loop(
        args.run_id,
        interval_seconds=args.interval,
        total_hours=args.hours,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )
    trends = summarize_run(args.run_id)
    print("\n--- Summary ---")
    print(json.dumps(trends, indent=2))
