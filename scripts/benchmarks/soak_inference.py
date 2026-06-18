#!/usr/bin/env python3
"""
24-Hour Inference Service Certification (C4).

Runs continuous inference for 24 hours with telemetry collection.

Usage:
    python scripts/benchmarks/soak_inference.py [--duration 24] [--interval 3600]
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
from soak_telemetry import ARTIFACTS_DIR, collect_snapshot, summarize_run, write_snapshot


def certify_24h_inference(
    duration_hours: int = 24,
    snapshot_interval: int = 3600,
):
    """Run 24 hours of continuous inference with hourly telemetry."""
    run_id = f"soak_inference_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    print(f"=== 24h Inference Certification: {run_id} ===")

    from helix_ids.models.full import HelixFullConfig, create_helix_full
    model = create_helix_full(HelixFullConfig())
    device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
    model = model.to(device)
    model.eval()

    end_time = time.time() + duration_hours * 3600
    inference_count = 0
    latencies: list[float] = []
    last_snapshot_time = 0.0

    with torch.no_grad():
        while time.time() < end_time:
            x = torch.randn(256, 41, device=device)
            t0 = time.perf_counter()
            _ = model(x)
            lat = time.perf_counter() - t0
            latencies.append(lat)
            inference_count += 1

            # Periodic telemetry snapshot
            now = time.time()
            if now - last_snapshot_time >= snapshot_interval or last_snapshot_time == 0:
                if latencies:
                    lats = sorted(latencies)
                    custom = {
                        "latency_p50_ms": lats[len(lats) // 2] * 1000,
                        "latency_p95_ms": lats[int(len(lats) * 0.95)] * 1000,
                        "latency_p99_ms": lats[int(len(lats) * 0.99)] * 1000,
                        "throughput_inf_per_sec": len(latencies) / max(now - last_snapshot_time, 1),
                        "total_inferences": inference_count,
                    }
                else:
                    custom = {"total_inferences": inference_count}

                snapshot = collect_snapshot(run_id, custom_metrics=custom)
                write_snapshot(snapshot, run_id)
                elapsed_h = (now - (end_time - duration_hours * 3600)) / 3600

                print(
                    f"  [{elapsed_h:.1f}h / {duration_hours}h] "
                    f"Inferences={inference_count} "
                    f"RSS={snapshot['rss_memory_mb']:.0f}MB "
                    f"Threads={snapshot['threads']} "
                    f"p50={custom.get('latency_p50_ms', 0):.1f}ms "
                    f"Throughput={custom.get('throughput_inf_per_sec', 0):.0f}inf/s"
                )

                latencies.clear()
                last_snapshot_time = now

    print("\n=== Inference Certification Complete ===")
    trends = summarize_run(run_id)
    print(json.dumps(trends, indent=2))

    passes = True
    reasons = []
    rss_trend = trends.get("rss_memory", {}).get("trend", "flat")
    if rss_trend == "rising":
        passes = False
        reasons.append(f"RSS trending upward: {rss_trend}")

    verdict = "PASS" if passes else "FAIL"
    status_path = ARTIFACTS_DIR / run_id / "certification_verdict.json"
    with open(status_path, "w") as f:
        json.dump({
            "run_id": run_id,
            "verdict": verdict,
            "reasons": reasons,
            "duration_hours": duration_hours,
            "inference_count": inference_count,
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
    sys.exit(0 if certify_24h_inference(args.duration, args.interval) else 1)
