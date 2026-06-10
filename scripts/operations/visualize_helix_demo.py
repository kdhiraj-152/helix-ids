#!/usr/bin/env python3
"""Visualize HELIX vs no-HELIX demo.

This script launches the local `serve_rest.py` server twice:
- baseline (HELIX controls disabled)
- helix (HELIX controls enabled)

It sends a short stream of synthetic requests, captures the per-request
events written to `artifacts/operations/live_events.jsonl`, and produces a
comparison plot saved under the output directory.

Usage (quick):
  PYTHONPATH=src python3 scripts/operations/visualize_helix_demo.py --requests-per-run 200

The output image will be written to `artifacts/operations/visuals/helix_vs_nohelix.png`.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def start_server(checkpoint: str, host: str, port: int, extra_args: list[str], log_path: str):
    """Start the `serve_rest.py` server from the repository root and wait for its /health endpoint.

    Uses an absolute PYTHONPATH based on the repo layout so callers can run this script
    from any CWD.
    """
    repo_root = Path(__file__).resolve().parents[2]
    src_path = str(repo_root / "src")
    env = os.environ.copy()
    env["PYTHONPATH"] = src_path

    script_path = str(repo_root / "scripts" / "operations" / "serve_rest.py")
    cmd = [sys.executable, script_path, "--checkpoint", checkpoint, "--host", host, "--port", str(port), "--device", "cpu"] + extra_args

    logf = open(log_path, "wb")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=str(repo_root))

    # wait for health
    import urllib.request

    url = f"http://{host}:{port}/health"
    for _ in range(60):
        try:
            with urllib.request.urlopen(url, timeout=2):
                return proc, logf
        except Exception:
            time.sleep(0.5)

    # didn't start
    try:
        logf.close()
    except Exception:
        pass
    try:
        proc.terminate()
    except Exception:
        pass
    raise RuntimeError(f"server failed to become healthy in time; see log: {log_path}")


def stop_server(proc: subprocess.Popen, logf) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        logf.close()
    except Exception:
        pass


def send_requests(host: str, port: int, n: int, features_dim: int = 17) -> None:
    import urllib.request

    url = f"http://{host}:{port}/predict"
    sample = [0.0] * features_dim
    data = json.dumps({"features": sample}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    for _i in range(n):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                _ = resp.read()
        except Exception:
            # ignore per-request failures for the demo
            pass


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def compute_override_series(events: list[dict[str, Any]]) -> list[float]:
    vals = [1.0 if (e.get("override", {}).get("applied", False)) else 0.0 for e in events]
    cum = []
    s = 0.0
    for i, v in enumerate(vals, start=1):
        s += v
        cum.append(s / float(i))
    return cum


def compute_class_counts(events: list[dict[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for e in events:
        pred = e.get("prediction", {}).get("family_class")
        if isinstance(pred, list):
            for c in pred:
                counts[int(c)] = counts.get(int(c), 0) + 1
        elif pred is not None:
            counts[int(pred)] = counts.get(int(pred), 0) + 1
    return counts


def plot_results(
    nohelix_series,
    helix_series,
    nohelix_counts,
    helix_counts,
    no_confidences,
    helix_confidences,
    no_margins,
    helix_margins,
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        print("Please install matplotlib and numpy in your environment: pip install matplotlib numpy")
        raise

    try:
        plt.style.use("seaborn-darkgrid")
    except Exception:
        try:
            plt.style.use("seaborn")
        except Exception:
            pass

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # override rate plot (cumulative)
    ax = axes[0]
    x1 = list(range(1, len(nohelix_series) + 1))
    x2 = list(range(1, len(helix_series) + 1))
    if x1:
        ax.plot(x1, nohelix_series, label="No HELIX (cum. override rate)", color="#d62728")
    if x2:
        ax.plot(x2, helix_series, label="With HELIX (cum. override rate)", color="#1f77b4")
    ax.set_xlabel("Requests")
    ax.set_ylabel("Override rate")
    ax.set_ylim(-0.01, 1.01)
    ax.legend()

    # confidence distribution
    ax2 = axes[1]
    bins = np.linspace(0.0, 1.0, 21)
    if no_confidences:
        ax2.hist(no_confidences, bins=bins, alpha=0.6, label="No HELIX", color="#d62728", density=False)
    if helix_confidences:
        ax2.hist(helix_confidences, bins=bins, alpha=0.6, label="With HELIX", color="#1f77b4", density=False)
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Count")
    ax2.legend()

    # margin distribution (only where margins present)
    ax3 = axes[2]
    all_margins = [m for m in (no_margins + helix_margins) if m is not None]
    if all_margins:
        # choose bins around observed range
        mn = min(all_margins)
        mx = max(all_margins)
        if mn == mx:
            bins = 20
        else:
            bins = np.linspace(mn, mx, 21)
        if no_margins:
            ax3.hist(no_margins, bins=bins, alpha=0.6, label="No HELIX", color="#d62728")
        if helix_margins:
            ax3.hist(helix_margins, bins=bins, alpha=0.6, label="With HELIX", color="#1f77b4")
        ax3.set_xlabel("Class margin")
        ax3.set_ylabel("Count")
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "No margin values captured", ha="center", va="center")
        ax3.set_axis_off()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    print(f"Saved comparison plot: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--requests-per-run", type=int, default=200)
    p.add_argument("--output-dir", default="artifacts/operations/visuals")
    p.add_argument("--checkpoint", default="models/helix_full/helix_full_nsl_kdd_best.pt")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18080)
    p.add_argument("--features-dim", type=int, default=17)
    p.add_argument(
        "--with-helix-args",
        default="",
        help="Extra flags to pass to the HELIX server (quoted string, e.g. '--flag1 val --flag2').",
    )
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_events = out_dir / "live_events_nohelix.jsonl"
    helix_events = out_dir / "live_events_helix.jsonl"

    # Variants
    user_helix_flags = shlex.split(args.with_helix_args) if args.with_helix_args else []
    variants = [
        ("nohelix", ["--no-class-margin-override-enabled", "--no-enable-global-coverage-floor"], base_events),
        ("helix", user_helix_flags, helix_events),
    ]

    for name, flags, dest in variants:
        print(f"Running variant: {name}")
        # remove existing capture file
        capture = Path("artifacts/operations/live_events.jsonl")
        try:
            if capture.exists():
                capture.unlink()
        except Exception:
            pass

        log_path = f"/tmp/helix_demo_{name}.log"
        proc, logf = start_server(args.checkpoint, args.host, args.port, flags, log_path)
        try:
            send_requests(args.host, args.port, args.requests_per_run, features_dim=args.features_dim)
            # let server flush
            time.sleep(0.5)
        finally:
            stop_server(proc, logf)

        # copy capture file
        if capture.exists():
            shutil.copyfile(capture, dest)
            print(f"Saved events to: {dest}")
        else:
            print(f"Warning: capture file not found for variant {name} (expected {capture})")

    # parse and plot
    no_events = parse_jsonl(base_events)
    h_events = parse_jsonl(helix_events)

    no_series = compute_override_series(no_events)
    h_series = compute_override_series(h_events)
    no_counts = compute_class_counts(no_events)
    h_counts = compute_class_counts(h_events)

    def _extract_confidences(events: list[dict[str, Any]]) -> list[float]:
        out: list[float] = []
        for e in events:
            pred = e.get("prediction") or {}
            c = pred.get("confidence") if isinstance(pred, dict) else None
            if c is not None:
                try:
                    out.append(float(c))
                except Exception:
                    continue
        return out


    def _extract_margins(events: list[dict[str, Any]]) -> list[float]:
        out: list[float] = []
        for e in events:
            cm = e.get("class_margin_override") or {}
            m = cm.get("margin") if isinstance(cm, dict) else None
            if m is not None:
                try:
                    out.append(float(m))
                except Exception:
                    continue
        return out

    no_conf = _extract_confidences(no_events)
    h_conf = _extract_confidences(h_events)
    no_margin = _extract_margins(no_events)
    h_margin = _extract_margins(h_events)

    out_image = Path(args.output_dir) / "helix_vs_nohelix.png"
    plot_results(no_series, h_series, no_counts, h_counts, no_conf, h_conf, no_margin, h_margin, out_image)


if __name__ == "__main__":
    main()
