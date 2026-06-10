#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HELIX staging promotion gate check")
    p.add_argument("--metrics-endpoint", required=True, help="Prometheus metrics endpoint URL")
    return p.parse_args()


def _parse_prometheus_text(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if " " not in line:
            continue
        key, value = line.split(None, 1)
        out[key] = value.strip()
    return out


def _fetch_metrics(url: str) -> str:
    with urlopen(url, timeout=15) as resp:  # nosec B310 - endpoint is user-specified operational target
        return resp.read().decode("utf-8", errors="replace")


def main() -> int:
    args = parse_args()

    try:
        payload = _fetch_metrics(args.metrics_endpoint)
    except (URLError, HTTPError, TimeoutError, OSError) as exc:
        print("[HELIX GATE] BLOCKED", flush=True)
        print("override_rate=nan", flush=True)
        print("degraded_state=1", flush=True)
        print(f"error={exc}", file=sys.stderr)
        return 1

    metrics = _parse_prometheus_text(payload)

    rate_raw = metrics.get("helix_coverage_override_rate")
    degraded_raw = metrics.get("helix_degraded_state")

    if rate_raw is None or degraded_raw is None:
        print("[HELIX GATE] BLOCKED", flush=True)
        print(f"override_rate={rate_raw if rate_raw is not None else 'nan'}", flush=True)
        print("degraded_state=1", flush=True)
        missing = []
        if rate_raw is None:
            missing.append("helix_coverage_override_rate")
        if degraded_raw is None:
            missing.append("helix_degraded_state")
        print(f"error=missing metrics: {', '.join(missing)}", file=sys.stderr)
        return 1

    try:
        override_rate = float(rate_raw)
        degraded_state = 1 if float(degraded_raw) >= 1.0 else 0
    except ValueError:
        print("[HELIX GATE] BLOCKED", flush=True)
        print(f"override_rate={rate_raw}", flush=True)
        print("degraded_state=1", flush=True)
        print("error=invalid numeric metrics", file=sys.stderr)
        return 1

    blocked = override_rate > 0.02 or degraded_state == 1

    if blocked:
        print("[HELIX GATE] BLOCKED", flush=True)
        print(f"override_rate={rate_raw}", flush=True)
        print(f"degraded_state={degraded_state}", flush=True)
        return 1

    print("[HELIX GATE] OK", flush=True)
    print(f"override_rate={rate_raw}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
