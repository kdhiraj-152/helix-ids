"""Tests for strict CI promotion gate JSON parser."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "parse_promotion_gate_logs.py"


def _run_parser(events_path: Path, *, ci: bool) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["CI"] = "true" if ci else "false"
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--events", str(events_path)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_parser_rejects_local_override(tmp_path: Path):
    events_path = tmp_path / "events.jsonl"
    events_path.write_text("", encoding="utf-8")

    result = _run_parser(events_path, ci=False)

    assert result.returncode == 1
    assert "E-PROMOTION-LOCAL-OVERRIDE-INVALID" in result.stdout


def test_parser_rejects_malformed_json(tmp_path: Path):
    events_path = tmp_path / "events.jsonl"
    events_path.write_text("{not-json}\n", encoding="utf-8")

    result = _run_parser(events_path, ci=True)

    assert result.returncode == 1
    assert "E-PROMOTION-MALFORMED-JSON-INVALID" in result.stdout


def test_parser_passes_valid_promotion_contract(tmp_path: Path):
    events_path = tmp_path / "events.jsonl"
    rows = [
        {
            "run_id": "run-1",
            "stage": "prepromote",
            "gate": "reproducibility_check",
            "status": "PASS",
            "reason_code": "OK",
            "metric": 0.005,
            "threshold": 0.01,
            "dataset": "dataset-a",
            "seed": 42,
            "timestamp": 1.0,
            "fingerprint": "fp-a",
        },
        {
            "run_id": "run-1",
            "stage": "prepromote",
            "gate": "promotion_contract",
            "status": "PASS",
            "reason_code": "OK",
            "metric": 1.0,
            "threshold": 1.0,
            "dataset": "dataset-a",
            "seed": 42,
            "timestamp": 2.0,
            "fingerprint": "fp-a",
        },
    ]
    events_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    result = _run_parser(events_path, ci=True)

    assert result.returncode == 0
    parsed = json.loads(result.stdout.strip())
    assert parsed["promotion_state"] == "PASS"
