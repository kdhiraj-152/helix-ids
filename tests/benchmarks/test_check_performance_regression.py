"""Tests for the CI benchmark regression gate (check_performance_regression.py).

P3 — Verifies that the regression gate correctly detects performance
regressions, passes when values are within threshold, handles edge cases,
and supports the --bless bootstrap mode.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.benchmarks.check_performance_regression as cpr

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_baseline(overrides: dict | None = None) -> dict:
    """Return a minimal valid baseline dict with default values."""
    data: dict = {
        "training_step": {"mean": 0.0030},
        "inference": {"mean": 0.0007},
        "checkpoint_save": {"throughput_mbps": 1200.0},
        "checkpoint_load": {"throughput_mbps": 900.0},
    }
    if overrides:
        # Replace top-level keys entirely (no recursive merge) so that
        # callers can pass ``{"training_step": {}}`` to clear a sub-dict.
        data.update(overrides)
    return data


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestCheckGate:
    """Pure-function tests for ``cpr.check_gate``."""

    def test_all_gates_pass_within_threshold(self):
        """No regression: current values are within threshold of baseline."""
        base = _make_baseline()
        cur = _make_baseline({"training_step": {"mean": 0.0031}})  # +3.3%, < 5%

        failures: list[str] = []
        for gate in cpr.GATES:
            failures.extend(cpr.check_gate(base, cur, **gate))
        assert failures == []

    def test_regression_detected_when_exceeding_threshold(self):
        """Latency degrades beyond threshold -> fails."""
        base = _make_baseline()
        cur = _make_baseline({"training_step": {"mean": 0.0045}})  # +50%, > 5%

        failures = []
        for gate in cpr.GATES:
            failures.extend(cpr.check_gate(base, cur, **gate))
        assert any("FAIL" in f and "Training step" in f for f in failures)

    def test_throughput_regression_detected(self):
        """Throughput drops below threshold -> fails (direction='down')."""
        base = _make_baseline()
        cur = _make_baseline({"checkpoint_save": {"throughput_mbps": 500.0}})

        failures = []
        for gate in cpr.GATES:
            failures.extend(cpr.check_gate(base, cur, **gate))
        assert any("FAIL" in f and "Checkpoint save" in f for f in failures)

    def test_throughput_improvement_passes(self):
        """Higher throughput is an improvement, not a regression."""
        base = _make_baseline()
        cur = _make_baseline({"checkpoint_save": {"throughput_mbps": 2400.0}})

        failures = []
        for gate in cpr.GATES:
            failures.extend(cpr.check_gate(base, cur, **gate))
        # No failures for checkpoint_save (checkpoint_load still passes too)
        checkpoint_fails = [
            f for f in failures if "Checkpoint save" in f
        ]
        assert checkpoint_fails == []

    def test_missing_baseline_value_returns_failure(self):
        """Missing key in baseline returns failure message, not crash."""
        base = _make_baseline({"training_step": {}})  # empty sub-dict
        cur = _make_baseline()

        failures = []
        for gate in cpr.GATES:
            failures.extend(cpr.check_gate(base, cur, **gate))
        assert any("no value" in f for f in failures)

    def test_missing_current_value_returns_failure(self):
        """Missing key in current returns failure message."""
        base = _make_baseline()
        cur = _make_baseline({"inference": {}})

        failures = []
        for gate in cpr.GATES:
            failures.extend(cpr.check_gate(base, cur, **gate))
        assert any("no value" in f for f in failures)

    def test_zero_baseline_value_returns_failure(self):
        """Baseline value of 0 cannot compute regression percentage."""
        base = _make_baseline({"inference": {"mean": 0.0}})
        cur = _make_baseline()

        failures = []
        for gate in cpr.GATES:
            failures.extend(cpr.check_gate(base, cur, **gate))
        assert any("baseline value is 0" in f for f in failures)

    def test_identical_values_pass(self):
        """Current identical to baseline -> passes."""
        base = _make_baseline()
        cur = _make_baseline()

        failures = []
        for gate in cpr.GATES:
            failures.extend(cpr.check_gate(base, cur, **gate))
        assert failures == []


class TestBlessMode:
    """Tests for the ``--bless`` bootstrap workflow."""

    def test_bless_creates_reference(self, tmp_path: Path):
        """--bless copies current output to the reference path."""
        cur = tmp_path / "current.json"
        ref = tmp_path / "baseline.reference.json"
        cur.write_text(json.dumps(_make_baseline()))

        with pytest.raises(SystemExit) as exc:
            cpr.main(["--bless", "--baseline", str(ref), "--current", str(cur)])
        assert exc.value.code == 0
        assert ref.exists()
        assert json.loads(ref.read_text()) == _make_baseline()

    def test_bless_exits_nonzero_without_current(self, tmp_path: Path):
        """--bless exits 1 when current output doesn't exist."""
        ref = tmp_path / "baseline.reference.json"
        cur = tmp_path / "nonexistent.json"

        with pytest.raises(SystemExit) as exc:
            cpr.main(["--bless", "--baseline", str(ref), "--current", str(cur)])
        assert exc.value.code == 1
        assert not ref.exists()  # nothing was written


class TestMainIntegration:
    """Integration tests for ``cpr.main()`` with real file I/O."""

    def test_missing_reference_file_exits_nonzero(self, tmp_path: Path):
        """Missing baseline.reference.json causes exit code 1."""
        ref = tmp_path / "nonexistent.reference.json"
        cur = tmp_path / "current.json"
        cur.write_text(json.dumps(_make_baseline()))

        with pytest.raises(SystemExit) as exc:
            cpr.main(["--baseline", str(ref), "--current", str(cur)])
        assert exc.value.code == 1

    def test_missing_current_file_exits_nonzero(self, tmp_path: Path):
        """Missing current benchmark output causes exit code 1."""
        ref = tmp_path / "baseline.reference.json"
        cur = tmp_path / "nonexistent.json"
        ref.write_text(json.dumps(_make_baseline()))

        with pytest.raises(SystemExit) as exc:
            cpr.main(["--baseline", str(ref), "--current", str(cur)])
        assert exc.value.code == 1

    def test_passes_with_identical_files(self, tmp_path: Path):
        """Identical reference and current -> exit 0."""
        ref = tmp_path / "baseline.reference.json"
        cur = tmp_path / "baseline.json"
        data = _make_baseline()
        ref.write_text(json.dumps(data))
        cur.write_text(json.dumps(data))

        with pytest.raises(SystemExit) as exc:
            cpr.main(["--baseline", str(ref), "--current", str(cur)])
        assert exc.value.code == 0

    def test_fails_on_regression(self, tmp_path: Path):
        """Current with degraded latency -> exit 1."""
        ref = tmp_path / "baseline.reference.json"
        cur = tmp_path / "baseline.json"
        ref.write_text(json.dumps(_make_baseline()))
        bad = _make_baseline({"training_step": {"mean": 0.010}})  # +233%
        cur.write_text(json.dumps(bad))

        with pytest.raises(SystemExit) as exc:
            cpr.main(["--baseline", str(ref), "--current", str(cur)])
        assert exc.value.code == 1
