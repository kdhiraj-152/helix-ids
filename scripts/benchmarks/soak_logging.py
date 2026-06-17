#!/usr/bin/env python3
"""
24-Hour Logging Subsystem Certification (C4).

Runs continuous structured logging for 24 hours with telemetry collection.

Usage:
    python scripts/benchmarks/soak_logging.py [--duration 24] [--interval 3600]
"""

# ruff: noqa: E402

import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import logging

from soak_telemetry import collect_snapshot, write_snapshot, summarize_run, ARTIFACTS_DIR


def certify_24h_logging(
    duration_hours: int = 24,
    snapshot_interval: int = 3600,
):
    """Run 24 hours of structured logging with hourly telemetry."""
    run_id = f"soak_logging_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    print(f"=== 24h Logging Certification: {run_id} ===")

    from helix_ids.operations.logging import get_logger, LogContext
    from helix_ids.operations.logging.console_log_handler import ConsoleLogHandler

    # Create log directory
    log_dir = Path(tempfile.mkdtemp(prefix="soak_logging_"))
    print(f"Log directory: {log_dir}")

    # Set up file handler
    logger = get_logger("soak_logging")
    file_handler = logging.FileHandler(log_dir / "soak.log")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    logger.logger.addHandler(file_handler)

    # Disable console output to avoid stdout flood during soak
    logger.logger.handlers = [h for h in logger.logger.handlers
                              if not isinstance(h, ConsoleLogHandler)]

    end_time = time.time() + duration_hours * 3600
    log_count = 0
    last_snapshot_time = 0.0

    phases = ["training", "validation", "inference", "checkpoint", "startup", "shutdown"]
    levels = [logger.info, logger.warning, logger.error]
    phase_idx = 0

    while time.time() < end_time:
        phase = phases[phase_idx % len(phases)]
        level_fn = levels[phase_idx % len(levels)]
        phase_idx += 1

        with LogContext(phase=phase, epoch=log_count // 1000, step=log_count):
            level_fn(
                f"Logging soak message #{log_count}",
                extra={"soak_run": run_id, "phase": phase, "count": log_count},
            )
        log_count += 1

        # Hourly snapshot
        now = time.time()
        if now - last_snapshot_time >= snapshot_interval or last_snapshot_time == 0:
            custom = {
                "log_count": log_count,
                "log_rate": log_count / max(now - (end_time - duration_hours * 3600), 1),
            }
            snapshot = collect_snapshot(
                run_id,
                log_dir=log_dir,
                custom_metrics=custom,
            )
            path = write_snapshot(snapshot, run_id)
            elapsed_h = (now - (end_time - duration_hours * 3600)) / 3600

            print(
                f"  [{elapsed_h:.1f}h / {duration_hours}h] "
                f"Logs={log_count} "
                f"RSS={snapshot['rss_memory_mb']:.0f}MB "
                f"FileHandles={snapshot['file_handles']} "
                f"LogVolume={snapshot['log_volume_bytes'] / 1024:.0f}KB"
            )
            last_snapshot_time = now

    print(f"\n=== Logging Certification Complete ===")
    trends = summarize_run(run_id)
    print(json.dumps(trends, indent=2))

    passes = True
    reasons = []
    rss_trend = trends.get("rss_memory", {}).get("trend", "flat")
    if rss_trend == "rising":
        passes = False
        reasons.append(f"RSS trending upward ({rss_trend})")

    file_start = trends.get("file_handles", {}).get("start", 0)
    file_end = trends.get("file_handles", {}).get("end", 0)
    if file_start > 0 and file_end > file_start * 2:
        passes = False
        reasons.append(f"File handle leak: {file_start} -> {file_end}")

    verdict = "PASS" if passes else "FAIL"
    status_path = ARTIFACTS_DIR / run_id / "certification_verdict.json"
    with open(status_path, "w") as f:
        json.dump({
            "run_id": run_id,
            "verdict": verdict,
            "reasons": reasons,
            "duration_hours": duration_hours,
            "log_count": log_count,
            "log_volume_kb": trends.get("log_volume_bytes", {}).get("end", 0) / 1024,
            "trends": trends,
        }, f, indent=2)

    print(f"Verdict: {verdict}")
    if reasons:
        print(f"Reasons: {'; '.join(reasons)}")
    return passes


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=24, help="Duration in hours")
    parser.add_argument("--interval", type=int, default=3600, help="Snapshot interval in seconds")
    args = parser.parse_args()
    sys.exit(0 if certify_24h_logging(args.duration, args.interval) else 1)
