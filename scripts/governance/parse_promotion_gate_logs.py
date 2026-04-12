#!/usr/bin/env python3
"""Strict CI-only parser for promotion gate JSON logs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REQUIRED_FIELDS = {
    "run_id",
    "stage",
    "gate",
    "status",
    "reason_code",
    "metric",
    "threshold",
    "dataset",
    "seed",
    "timestamp",
    "fingerprint",
}


def _reject(message: str) -> int:
    print(message)
    return 1


def _require_ci_context() -> bool:
    return (
        os.environ.get("CI", "").lower() == "true"
        or os.environ.get("GITHUB_ACTIONS", "").lower() == "true"
    )


def parse_events(events_path: Path) -> tuple[int, list[dict[str, object]]]:
    if not events_path.exists():
        return 1, []

    events: list[dict[str, object]] = []
    with events_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                print(f"E-PROMOTION-MALFORMED-JSON-INVALID:{line_no}")
                return 1, []

            if not isinstance(event, dict):
                print(f"E-PROMOTION-NONOBJECT-JSON-INVALID:{line_no}")
                return 1, []

            missing = REQUIRED_FIELDS - set(event.keys())
            if missing:
                print(f"E-PROMOTION-MISSING-FIELDS-INVALID:{line_no}:{','.join(sorted(missing))}")
                return 1, []

            status = str(event.get("status"))
            if status not in {"PASS", "FAIL", "INVALID"}:
                print(f"E-PROMOTION-BAD-STATUS-INVALID:{line_no}")
                return 1, []

            events.append(event)

    if not events:
        print("E-PROMOTION-EMPTY-EVENTS-INVALID")
        return 1, []

    return 0, events


def evaluate_promotion(events: list[dict[str, object]]) -> int:
    target_run = str(events[-1]["run_id"])
    run_events = [event for event in events if str(event.get("run_id")) == target_run]

    if any(str(event.get("status")) in {"FAIL", "INVALID"} for event in run_events):
        return _reject("E-PROMOTION-RUN-HAS-FAILURES-INVALID")

    promotion_contract = [
        event
        for event in run_events
        if str(event.get("stage")) == "prepromote"
        and str(event.get("gate")) == "promotion_contract"
        and str(event.get("status")) == "PASS"
    ]
    if not promotion_contract:
        return _reject("E-PROMOTION-CONTRACT-MISSING-INVALID")

    reproducibility = [
        event
        for event in run_events
        if str(event.get("stage")) == "prepromote"
        and str(event.get("gate")) == "reproducibility_check"
        and str(event.get("status")) == "PASS"
    ]
    if not reproducibility:
        return _reject("E-PROMOTION-REPRODUCIBILITY-MISSING-INVALID")

    print(
        json.dumps(
            {
                "run_id": target_run,
                "promotion_state": "PASS",
                "events_evaluated": len(run_events),
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse and enforce promotion gate events in CI")
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("results/gates/gate_events.jsonl"),
        help="Path to gate event JSONL file",
    )
    args = parser.parse_args()

    if not _require_ci_context():
        return _reject("E-PROMOTION-LOCAL-OVERRIDE-INVALID")

    code, events = parse_events(args.events)
    if code != 0:
        return code

    return evaluate_promotion(events)


if __name__ == "__main__":
    raise SystemExit(main())
