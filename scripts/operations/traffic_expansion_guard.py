#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HELIX real-time traffic expansion guard")
    p.add_argument("--metrics-endpoint", required=True, help="Prometheus metrics endpoint URL")
    p.add_argument("--interval", type=float, default=2.0, help="Polling interval in seconds")
    return p.parse_args()


def _fetch_metrics(url: str) -> str:
    with urlopen(url, timeout=15) as resp:  # nosec B310 - operational endpoint is user-supplied
        return resp.read().decode("utf-8", errors="replace")


def _parse_metric_value(metrics_text: str, metric_name: str) -> str | None:
    for raw in metrics_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if " " not in line:
            continue
        key, value = line.split(None, 1)
        if key == metric_name:
            return value.strip()
    return None


def main() -> int:
    args = parse_args()
    interval = float(args.interval)
    if interval < 0:
        interval = 0.0

    while True:
        try:
            metrics_text = _fetch_metrics(args.metrics_endpoint)
            degraded_raw = _parse_metric_value(metrics_text, "helix_degraded_state")
            if degraded_raw is None:
                print("[HELIX GUARD] HALT", flush=True)
                print("degraded_state=1", flush=True)
                print("error=missing metric: helix_degraded_state", file=sys.stderr)
                return 1

            degraded_state = 1 if float(degraded_raw) >= 1.0 else 0
        except (ValueError, HTTPError, URLError, TimeoutError, OSError) as exc:
            print("[HELIX GUARD] HALT", flush=True)
            print("degraded_state=1", flush=True)
            print(f"error={exc}", file=sys.stderr)
            return 1

        if degraded_state == 1:
            print("[HELIX GUARD] HALT", flush=True)
            print("degraded_state=1", flush=True)
            return 1

        print("[HELIX GUARD] OK", flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
