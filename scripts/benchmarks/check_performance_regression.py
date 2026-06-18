#!/usr/bin/env python3
"""Performance regression checker — compares benchmark results against a baseline.

Usage:
    check_performance_regression.py --baseline <reference.json> --current <current.json>
    check_performance_regression.py --bless --baseline <reference.json> --current <current.json>

Gates are defined by a list of check configurations. Each gate specifies
a metric path, a value key, a regression threshold (as a fraction), and
a direction ('up' for latency metrics where higher is worse, 'down' for
throughput where lower is worse).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# ── Gate definitions ──────────────────────────────────────────────────────────

GATES: list[dict[str, Any]] = [
    {
        "name": "Training step",
        "key": "training_step",
        "value_key": "mean",
        "threshold": 0.05,  # 5%
        "direction": "up",
    },
    {
        "name": "Inference",
        "key": "inference",
        "value_key": "mean",
        "threshold": 0.05,
        "direction": "up",
    },
    {
        "name": "Checkpoint save",
        "key": "checkpoint_save",
        "value_key": "throughput_mbps",
        "threshold": 0.05,
        "direction": "down",
    },
    {
        "name": "Checkpoint load",
        "key": "checkpoint_load",
        "value_key": "throughput_mbps",
        "threshold": 0.05,
        "direction": "down",
    },
]


# ── Core logic ────────────────────────────────────────────────────────────────


def check_gate(
    baseline: dict[str, Any],
    current: dict[str, Any],
    *,
    name: str,
    key: str,
    value_key: str,
    threshold: float,
    direction: str,
) -> list[str]:
    """Compare *current* against *baseline* for a single gate.

    Returns a list of failure messages (empty when the gate passes).
    """
    failures: list[str] = []

    base_section = baseline.get(key)
    cur_section = current.get(key)

    if not base_section or value_key not in base_section:
        failures.append(f"FAIL [{name}]: no value in baseline for '{key}.{value_key}'")
        return failures

    if not cur_section or value_key not in cur_section:
        failures.append(f"FAIL [{name}]: no value in current for '{key}.{value_key}'")
        return failures

    base_val: float = base_section[value_key]
    cur_val: float = cur_section[value_key]

    if base_val == 0.0:
        failures.append(
            f"FAIL [{name}]: baseline value is 0, cannot compute regression percentage"
        )
        return failures

    ratio = cur_val / base_val

    if direction == "up":
        # Higher is worse (e.g. latency)
        if ratio > (1.0 + threshold):
            pct = (ratio - 1.0) * 100.0
            failures.append(
                f"FAIL [{name}]: {cur_val:.4f} is {pct:+.2f}% vs baseline {base_val:.4f} "
                f"(threshold +{threshold * 100:.0f}%)"
            )
    elif direction == "down":
        # Lower is worse (e.g. throughput)
        if ratio < (1.0 - threshold):
            pct = (1.0 - ratio) * 100.0
            failures.append(
                f"FAIL [{name}]: {cur_val:.4f} is {pct:.2f}% lower than baseline "
                f"{base_val:.4f} (threshold -{threshold * 100:.0f}%)"
            )

    return failures


def run_checks(
    baseline: dict[str, Any],
    current: dict[str, Any],
    gates: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Run all gate checks and return the combined list of failures."""
    if gates is None:
        gates = GATES

    all_failures: list[str] = []
    for gate in gates:
        all_failures.extend(check_gate(baseline, current, **gate))
    return all_failures


# ── CLI ────────────────────────────────────────────────────────────────────────


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return dict(json.load(f))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check CI benchmark regressions.")
    parser.add_argument(
        "--baseline",
        required=True,
        type=Path,
        help="Path to the baseline reference JSON file.",
    )
    parser.add_argument(
        "--current",
        required=True,
        type=Path,
        help="Path to the current benchmark JSON file.",
    )
    parser.add_argument(
        "--bless",
        action="store_true",
        help="Copy current output to the baseline path (bootstrap / update reference).",
    )
    args = parser.parse_args(argv)

    ref_path: Path = args.baseline
    cur_path: Path = args.current

    if args.bless:
        if not cur_path.exists():
            print(f"ERROR: --bless mode — current file not found: {cur_path}", file=sys.stderr)
            sys.exit(1)
        # Copy current to reference
        ref_path.write_text(cur_path.read_text())
        print(f"Blessed: copied {cur_path} -> {ref_path}")
        sys.exit(0)

    # Normal check mode
    if not ref_path.exists():
        print(f"ERROR: baseline file not found: {ref_path}", file=sys.stderr)
        sys.exit(1)

    if not cur_path.exists():
        print(f"ERROR: current file not found: {cur_path}", file=sys.stderr)
        sys.exit(1)

    baseline = _load_json(ref_path)
    current = _load_json(cur_path)

    failures = run_checks(baseline, current)
    for msg in failures:
        print(msg, file=sys.stderr)

    if failures:
        sys.exit(1)
    else:
        print("All gates passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
