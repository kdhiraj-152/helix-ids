#!/usr/bin/env python3
"""
Phase 23 Soak Certification Monitor.

Checks all three soak processes and their telemetry.
Generates a status summary. Designed to run as an hourly cron job.
"""
# ruff: noqa: E402

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "soak"
STATUS_DIR = ARTIFACTS_DIR / "_monitor"
STATUS_DIR.mkdir(parents=True, exist_ok=True)

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


SOAKS = {
    "training": {
        "script": "soak_training.py",
        "log_file": ARTIFACTS_DIR / "training.log",
        "pid": None,
    },
    "inference": {
        "script": "soak_inference.py",
        "log_file": ARTIFACTS_DIR / "inference.log",
        "pid": None,
    },
    "logging": {
        "script": "soak_logging.py",
        "log_file": ARTIFACTS_DIR / "logging.log",
        "pid": None,
    },
}


def find_process_pid(script_name: str) -> int | None:
    """Find the PID of a running soak process by script name."""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            if script_name in line and "python" in line.lower() and "grep" not in line:
                parts = line.split()
                # Prevent matching the shell wrapper; want the actual python process
                if "Python.app" in line or line.strip().endswith(script_name):
                    return int(parts[1])
        return None
    except Exception:
        return None


def get_alive_status(script_name: str) -> tuple[bool, int | None]:
    """Check if a soak process is alive."""
    pid = find_process_pid(script_name)
    return (pid is not None, pid)


def get_snapshots(run_prefix: str) -> list[dict]:
    """Load all snapshots for a given run prefix."""
    snapshots = []
    for run_dir in sorted(ARTIFACTS_DIR.glob(f"{run_prefix}_*")):
        for sp in sorted(run_dir.glob("snapshot_*.json")):
            with open(sp) as f:
                snapshots.append(json.load(f))
    return snapshots


def get_latest_snapshot(run_prefix: str) -> dict | None:
    """Get the most recent snapshot for a run prefix."""
    latest = None
    latest_time = 0
    for run_dir in sorted(ARTIFACTS_DIR.glob(f"{run_prefix}_*")):
        for sp in sorted(run_dir.glob("snapshot_*.json")):
            try:
                with open(sp) as f:
                    data = json.load(f)
                t = data.get("uptime_seconds", 0)
                if t > latest_time:
                    latest_time = t
                    latest = data
            except Exception:
                pass
    return latest


def get_run_dir(run_prefix: str) -> Path | None:
    """Get the latest run directory for a prefix."""
    dirs = sorted(ARTIFACTS_DIR.glob(f"{run_prefix}_*"))
    return dirs[-1] if dirs else None


def compute_resource_trend(snapshots: list[dict], key: str) -> str:
    """Compute trend direction for a numeric metric."""
    if len(snapshots) < 2:
        return "insufficient_data"
    values = [s.get(key, 0) for s in snapshots]
    first, last = values[0], values[-1]
    if abs(last - first) < 50:
        return "flat"
    return "rising" if last > first else "falling"


def compute_trend_simple(values: list[float], threshold: float = 10) -> str:
    """Compute trend for a metric with custom threshold."""
    if len(values) < 2:
        return "insufficient_data"
    first, last = values[0], values[-1]
    if abs(last - first) < threshold:
        return "flat"
    return "rising" if last > first else "falling"


def check_stalled_telemetry(run_prefix: str, max_age_minutes: int = 90) -> str | None:
    """Check if telemetry snapshots have stalled."""
    latest = get_latest_snapshot(run_prefix)
    if latest is None:
        return "NO_SNAPSHOTS_EVER"
    try:
        ts = latest["timestamp"]
        snapshot_time = datetime.fromisoformat(ts)
        age = (datetime.now(timezone.utc) - snapshot_time).total_seconds() / 60
        if age > max_age_minutes:
            return f"STALLED (last snapshot {age:.0f} min ago)"
    except Exception:
        return "PARSE_ERROR"
    return None


def get_run_id(run_prefix: str) -> str | None:
    """Get the run ID from the latest run directory."""
    run_dir = get_run_dir(run_prefix)
    if run_dir:
        return run_dir.name
    return None


def compute_elapsed_hours(run_prefix: str, start_time_str: str | None = None) -> float:
    """Compute elapsed hours since the first snapshot."""
    snapshots = get_snapshots(run_prefix)
    if not snapshots:
        return 0.0
    try:
        first_ts = snapshots[0]["timestamp"]
        t0 = datetime.fromisoformat(first_ts)
        elapsed = (datetime.now(timezone.utc) - t0).total_seconds() / 3600
        return elapsed
    except Exception:
        return 0.0


def main() -> int:
    now = datetime.now(timezone.utc)
    print(f"=== Phase 23 Soak Monitor: {now.strftime('%Y-%m-%d %H:%M:%S UTC')} ===\n")

    results = {}
    any_fail = False

    for name, cfg in SOAKS.items():
        print(f"--- {name.upper()} ---")
        alive, pid = get_alive_status(cfg["script"])
        cfg["pid"] = pid

        status = "ALIVE" if alive else "DEAD"
        print(f"  Process: {status}" + (f" (PID={pid})" if pid else ""))

        run_prefix = f"soak_{name}"
        run_id = get_run_id(run_prefix)
        run_dir = get_run_dir(run_prefix)
        snapshots = get_snapshots(run_prefix)

        stalled = check_stalled_telemetry(run_prefix)
        if stalled:
            print(f"  Telemetry: WARNING - {stalled}")
            any_fail = True

        elapsed_h = compute_elapsed_hours(run_prefix)
        print(f"  Elapsed: {elapsed_h:.1f}h / 24h")

        if snapshots:
            n = len(snapshots)
            latest = snapshots[-1]
            first = snapshots[0]

            print(f"  Snapshots: {n} total")
            print(f"  Latest: {latest['timestamp']}")
            print(f"  RSS: {latest.get('rss_memory_mb', 'N/A'):.0f} MB" if isinstance(latest.get('rss_memory_mb'), (int, float)) else f"  RSS: {latest.get('rss_memory_mb', 'N/A')}")
            print(f"  Threads: {latest.get('threads', 'N/A')}")
            print(f"  File handles: {latest.get('file_handles', 'N/A')}")
            print(f"  Tensor count: {latest.get('tensor_count', 'N/A')}")
            print(f"  GC total collected: {latest.get('gc', {}).get('total_collected', 'N/A')}")

            # Memory trend
            rss_values = [s.get("rss_memory_mb", 0) for s in snapshots]
            trend = compute_trend_simple(rss_values, threshold=50)
            print(f"  RSS trend: {trend} ({rss_values[0]:.0f} -> {rss_values[-1]:.0f} MB)")

            # Thread trend
            thread_values = [s.get("threads", 0) for s in snapshots]
            thread_trend = compute_trend_simple(thread_values, threshold=2)
            print(f"  Thread trend: {thread_trend} ({thread_values[0]} -> {thread_values[-1]})")

            # File handle trend
            fh_values = [s.get("file_handles", 0) for s in snapshots]
            fh_trend = compute_trend_simple(fh_values, threshold=5)
            print(f"  FD trend: {fh_trend} ({fh_values[0]} -> {fh_values[-1]})")

            # Check for memory leak (6+ consecutive hours rising)
            if len(rss_values) >= 3:
                # Check last 3 points (if available) for monotonic increase
                last_3 = rss_values[-3:]
                if len(last_3) >= 3 and all(last_3[i] <= last_3[i+1] for i in range(len(last_3)-1)):
                    print(f"  LEAK WARNING: RSS rising in last {len(last_3)} snapshots")

            # Custom metrics (latency, throughput)
            custom = latest.get("custom", {})
            if custom:
                for k, v in custom.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.2f}")
                    else:
                        print(f"  {k}: {v}")

                # Check latency baseline (p50 > 2x first snapshot?)
                if len(snapshots) >= 2:
                    first_custom = snapshots[0].get("custom", {})
                    for key in ["latency_p50_ms", "latency_p95_ms", "latency_p99_ms"]:
                        if key in custom and key in first_custom:
                            try:
                                ratio = custom[key] / max(first_custom[key], 1e-6)
                                if ratio > 2.0:
                                    print(f"  LATENCY WARNING: {key} {ratio:.1f}x baseline ({first_custom[key]:.1f} -> {custom[key]:.1f} ms)")
                            except (TypeError, ZeroDivisionError):
                                pass

            # Checkpoint count
            ckpt = latest.get("checkpoint_count", -1)
            if ckpt >= 0:
                print(f"  Checkpoints: {ckpt}")

        else:
            print("  Snapshots: NONE YET")
            any_fail = True

        results[name] = {
            "alive": alive,
            "pid": pid,
            "snapshots": len(snapshots) if snapshots else 0,
            "elapsed_h": elapsed_h,
            "run_id": run_id,
            "stalled": stalled,
            "rss_trend": trend if snapshots else "no_data",
        }

        print()

    # Overall summary
    print("=== SUMMARY ===")
    all_alive = all(r["alive"] for r in results.values())
    all_snapshots = all(r["snapshots"] > 0 for r in results.values())

    verdict = "PASS" if (all_alive and all_snapshots and not any_fail) else "MONITORING"
    print(f"  All alive: {'YES' if all_alive else 'NO'}")
    print(f"  Has snapshots: {'YES' if all_snapshots else 'NO'}")
    print(f"  Verdict: {verdict}")

    if not all_alive:
        for name, r in results.items():
            if not r["alive"]:
                print(f"  FAILED: {name} process is DEAD")

    # Write status report
    report = {
        "timestamp": now.isoformat(),
        "results": results,
        "verdict": verdict,
    }
    report_path = STATUS_DIR / f"status_{now.strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nStatus written to: {report_path}")

    return 0 if all_alive else 1


if __name__ == "__main__":
    sys.exit(main())
