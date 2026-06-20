#!/usr/bin/env python3
"""
24-Hour Training Loop Certification (C4).

Runs a continuous training loop for 24 hours with telemetry collection.

Usage:
    python scripts/benchmarks/soak_training.py [--duration 24] [--interval 3600]
"""

# ruff: noqa: E402

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import torch.nn as nn
from soak_telemetry import ARTIFACTS_DIR, collect_snapshot, summarize_run, write_snapshot


def training_step(model: nn.Module, device: str = "cpu", batch_size: int = 256) -> float:
    """Run a single training step against multi-head model and return latency."""
    loss_fn_binary = nn.BCEWithLogitsLoss()
    loss_fn_family = nn.CrossEntropyLoss()

    x = torch.randn(batch_size, 17, device=device)
    y_binary = torch.randint(0, 2, (batch_size, 2), device=device).float()
    y_family = torch.randint(0, 7, (batch_size,), device=device)

    t0 = time.perf_counter()
    model.zero_grad()
    binary_logits, family_logits = model(x)
    loss = loss_fn_binary(binary_logits, y_binary) + loss_fn_family(family_logits, y_family)
    loss.backward()
    # Minimal optimizer step — one param group is enough for soak purposes
    for group in model.parameters():
        if group.requires_grad and group.grad is not None:
            with torch.no_grad():
                group.sub_(1e-4 * group.grad)
            break
    return time.perf_counter() - t0


def certify_24h_training(
    duration_hours: int = 24,
    snapshot_interval: int = 3600,
):
    """Run a 24-hour training loop with hourly telemetry."""
    run_id = f"soak_training_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    print(f"=== 24h Training Certification: {run_id} ===")

    from helix_ids.models.full import HelixFullConfig, create_helix_full
    model = create_helix_full(HelixFullConfig())
    device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
    model = model.to(device)

    end_time = time.time() + duration_hours * 3600
    step_count = 0
    latencies: list[float] = []
    last_snapshot_time = 0.0

    while time.time() < end_time:
        # Run training step
        lat = training_step(model, device=device)
        latencies.append(lat)
        step_count += 1

        # Periodic telemetry snapshot
        now = time.time()
        if now - last_snapshot_time >= snapshot_interval or last_snapshot_time == 0:
            if latencies:
                lats = sorted(latencies)
                custom = {
                    "latency_p50_ms": lats[len(lats) // 2] * 1000,
                    "latency_p95_ms": lats[int(len(lats) * 0.95)] * 1000,
                    "latency_p99_ms": lats[int(len(lats) * 0.99)] * 1000,
                    "throughput_steps_per_sec": len(latencies) / max(now - last_snapshot_time, 1),
                    "total_steps": step_count,
                }
            else:
                custom = {"total_steps": step_count}

            snapshot = collect_snapshot(run_id, custom_metrics=custom)
            write_snapshot(snapshot, run_id)
            elapsed_h = (now - (end_time - duration_hours * 3600)) / 3600

            print(
                f"  [{elapsed_h:.1f}h / {duration_hours}h] "
                f"Steps={step_count} "
                f"RSS={snapshot['rss_memory_mb']:.0f}MB "
                f"Threads={snapshot['threads']} "
                f"Tensors={snapshot['tensor_count']} "
                f"p50={custom.get('latency_p50_ms', 0):.1f}ms "
                f"Throughput={custom.get('throughput_steps_per_sec', 0):.2f}step/s"
            )

            latencies.clear()
            model = model.to(device)
            last_snapshot_time = now

    # Final summary
    print("\n=== Training Certification Complete ===")
    trends = summarize_run(run_id)
    print(json.dumps(trends, indent=2))

    # Success criteria
    passes = True
    reasons = []
    rss_trend = trends.get("rss_memory", {}).get("trend", "flat")
    if rss_trend == "rising":
        passes = False
        reasons.append(f"RSS trending upward: {rss_trend}")

    if trends.get("file_handles", {}).get("end", -1) > 0:
        file_start = trends.get("file_handles", {}).get("start", 0)
        file_end = trends.get("file_handles", {}).get("end", 0)
        if file_end > file_start * 1.5 and file_start > 0:
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
            "step_count": step_count,
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
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: artifacts/soak/)")
    args = parser.parse_args()
    if args.output:
        import soak_telemetry as _st
        _st.ARTIFACTS_DIR = Path(args.output)
        globals()["ARTIFACTS_DIR"] = _st.ARTIFACTS_DIR
    sys.exit(0 if certify_24h_training(args.duration, args.interval) else 1)
