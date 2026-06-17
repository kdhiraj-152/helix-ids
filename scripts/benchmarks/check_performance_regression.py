#!/usr/bin/env python3
"""CI performance regression gate for HELIX-IDS benchmarks.

Compares the just-generated benchmark output against the stored reference
baseline and fails if any CI gate threshold is breached.

Usage
-----
    # Normal CI check (compare current run vs blessed reference):
    python check_performance_regression.py

    # First-time setup (bless current output as the reference):
    python check_performance_regression.py --bless

    # Custom paths:
    python check_performance_regression.py \\
        --baseline benchmarks/baseline.reference.json \\
        --current benchmarks/baseline.json

Exit codes
----------
0 — all gates pass (or --bless succeeded)
1 — at least one gate fails, or reference/current file missing
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"

DEFAULT_BASELINE = BENCHMARKS_DIR / "baseline.reference.json"
DEFAULT_CURRENT = BENCHMARKS_DIR / "baseline.json"

# Gates: which stages x which metric x regression threshold
# direction="up"   → higher value is a regression (latency)
# direction="down" → lower value is a regression (throughput)
GATES: list[dict] = [
    {"stage": "training_step", "metric": "mean", "regression_pct": 5.0,
     "label": "Training step"},
    {"stage": "inference", "metric": "mean", "regression_pct": 5.0,
     "label": "Inference"},
    {"stage": "checkpoint_save", "metric": "throughput_mbps", "regression_pct": 10.0,
     "label": "Checkpoint save throughput",
     "direction": "down"},
    {"stage": "checkpoint_load", "metric": "throughput_mbps", "regression_pct": 10.0,
     "label": "Checkpoint load throughput",
     "direction": "down"},
]


def load_json(path: Path) -> dict:
    """Read and parse a JSON file."""
    with open(path) as f:
        return json.load(f)


def check_gate(
    baseline: dict,
    current: dict,
    stage: str,
    metric: str,
    regression_pct: float,
    label: str,
    direction: str = "up",
) -> list[str]:
    """Return a list of failure messages (empty = pass)."""
    base_val = baseline.get(stage, {}).get(metric)
    cur_val = current.get(stage, {}).get(metric)

    if base_val is None:
        return [f"  {label}: baseline has no value for {stage}.{metric}"]
    if cur_val is None:
        return [f"  {label}: run has no value for {stage}.{metric}"]
    if base_val == 0:
        return [f"  {label}: baseline value is 0, cannot compute regression"]

    if direction == "up":
        # Higher means regression (latency)
        change_pct = (cur_val - base_val) / base_val * 100
        if change_pct > regression_pct:
            return [
                f"  FAIL: {label} degraded {change_pct:+.1f}% "
                f"(baseline={base_val:.4f}, current={cur_val:.4f}, "
                f"threshold={regression_pct}%)"
            ]
    else:
        # Lower means regression (throughput)
        change_pct = (base_val - cur_val) / base_val * 100
        if change_pct > regression_pct:
            return [
                f"  FAIL: {label} degraded {change_pct:+.1f}% "
                f"(baseline={base_val:.2f}, current={cur_val:.2f}, "
                f"threshold={regression_pct}%)"
            ]

    return []


def do_bless(current_path: Path, baseline_path: Path) -> None:
    """Copy current benchmark output to the reference baseline path.

    Called when ``--bless`` is passed.  Exits with code 0 on success,
    1 if the current output file does not exist.
    """
    if not current_path.exists():
        print(f"FAIL: No current benchmark output found at {current_path}")
        print("Run benchmark_pipeline.py first to produce benchmark data.")
        sys.exit(1)

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(current_path, baseline_path)
    print(f"✓ Blessed baseline written to {baseline_path}")
    print("  Commit this file to track it as the performance reference.")
    sys.exit(0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CI performance regression gate for HELIX-IDS benchmarks.",
    )
    parser.add_argument(
        "--baseline",
        default=DEFAULT_BASELINE,
        type=Path,
        help=(
            f"Path to the reference (blessed) baseline "
            f"(default: {DEFAULT_BASELINE})"
        ),
    )
    parser.add_argument(
        "--current",
        default=DEFAULT_CURRENT,
        type=Path,
        help=(
            f"Path to the just-run benchmark output "
            f"(default: {DEFAULT_CURRENT})"
        ),
    )
    parser.add_argument(
        "--bless",
        action="store_true",
        help=(
            "Copy current benchmark output to the reference path, "
            "blessing it as the new performance baseline"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the regression gate.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments (default: ``sys.argv[1:]``).  Provided
        for testability; production callers omit this.
    """
    args = parse_args(argv)

    baseline_path: Path = args.baseline
    current_path: Path = args.current

    # ── --bless mode: promote current output to reference ────────────────
    if args.bless:
        do_bless(current_path, baseline_path)
        return  # unreachable — do_bless always exits

    # ── Normal mode: compare ─────────────────────────────────────────────
    if not baseline_path.exists():
        print(f"FAIL: No reference baseline found at {baseline_path}")
        print("Run with --bless to create one from current benchmark output.")
        sys.exit(1)

    if not current_path.exists():
        print(f"FAIL: No current benchmark output found at {current_path}")
        print("Run benchmark_pipeline.py first.")
        sys.exit(1)

    baseline = load_json(baseline_path)
    current = load_json(current_path)

    failures: list[str] = []
    for gate in GATES:
        failures.extend(check_gate(baseline, current, **gate))

    if failures:
        print("PERFORMANCE REGRESSION FAILURES:")
        for f in failures:
            print(f)
        sys.exit(1)

    print("All performance gates PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
