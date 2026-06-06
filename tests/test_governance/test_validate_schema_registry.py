"""Tests for validate_schema_registry.py — Phase 4B objective 1.

Covers: valid chronology, invalid chronology, malformed version,
missing version field, retired-entry exception.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

# Ensure the module is importable with PYTHONPATH=src
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ci.validate_schema_registry import (
    _parse_version,
    _date_to_str,
    _VERSION_RE,
    validate_registry,
    _load_registry,
    _validate_entry,
)
from datetime import date


# ---------------------------------------------------------------------------
# Unit tests: _parse_version
# ---------------------------------------------------------------------------

class TestParseVersion:
    def test_valid_date(self):
        assert _parse_version("2026-06-02") == date(2026, 6, 2)

    def test_valid_edge_cases(self):
        assert _parse_version("2025-01-01") == date(2025, 1, 1)
        assert _parse_version("2026-12-31") == date(2026, 12, 31)

    def test_malformed_not_date(self):
        assert _parse_version("2026-13-01") is None   # month 13
        assert _parse_version("2026-06-32") is None   # day 32
        assert _parse_version("2026-6-2") is None     # short form
        assert _parse_version("06-02-2026") is None   # wrong order
        assert _parse_version("2026_06_02") is None   # wrong separator

    def test_malformed_not_string(self):
        assert _parse_version(None) is None
        assert _parse_version(42) is None
        assert _parse_version("") is None
        assert _parse_version("   ") is None

    def test_malformed_wrong_format(self):
        assert _parse_version("v1.0.0") is None
        assert _parse_version("20260602") is None
        assert _parse_version("June 2, 2026") is None
        assert _parse_version("2026/06/02") is None

    def test_version_re_matches_valid_format(self):
        valid = ["2026-06-02", "2025-01-01", "2099-12-31"]
        for v in valid:
            assert _VERSION_RE.match(v), f"{v} should match"

        invalid = ["v2026-06-02", "2026-6-2", "26-06-02", "2026-06-2", ""]
        for v in invalid:
            assert not _VERSION_RE.match(v), f"{v} should NOT match"


class TestDateToStr:
    def test_roundtrip(self):
        d = date(2026, 6, 2)
        assert _date_to_str(d) == "2026-06-02"
        assert _parse_version(_date_to_str(d)) == d


# ---------------------------------------------------------------------------
# Unit tests: _validate_entry
# ---------------------------------------------------------------------------

class TestValidateEntryMissingVersion:
    def test_missing_version(self):
        checks = _validate_entry({
            "schema_name": "test_schema",
            "status": "active",
            "owner": "team",
            "compatibility_window": "30d",
            "deprecation_policy": "announce-deprecate-then-retire",
            "approval_required": True,
            # current_version deliberately absent
        }, index=0)
        fails = [c for c in checks if c["status"] == "fail"]
        assert any("missing required fields" in c["message"] for c in fails)


class TestValidateEntryMalformedVersion:
    def test_invalid_calendar_date(self):
        checks = _validate_entry({
            "schema_name": "test_schema",
            "current_version": "2026-13-01",   # invalid month
            "status": "active",
            "owner": "team",
            "compatibility_window": "30d",
            "deprecation_policy": "announce-deprecate-then-retire",
            "approval_required": True,
        }, index=0)
        fails = [c for c in checks if c["status"] == "fail"]
        assert any("not a valid calendar date" in c["message"] for c in fails)

    def test_wrong_format(self):
        checks = _validate_entry({
            "schema_name": "test_schema",
            "current_version": "v1.0.0",
            "status": "active",
            "owner": "team",
            "compatibility_window": "30d",
            "deprecation_policy": "announce-deprecate-then-retire",
            "approval_required": True,
        }, index=0)
        fails = [c for c in checks if c["status"] == "fail"]
        assert any("YYYY-MM-DD" in c["message"] for c in fails)


class TestValidateEntryChronology:
    def test_current_earlier_than_previous(self):
        checks = _validate_entry({
            "schema_name": "test_schema",
            "current_version": "2026-01-01",
            "previous_version": "2026-06-02",
            "status": "active",
            "owner": "team",
            "compatibility_window": "30d",
            "deprecation_policy": "announce-deprecate-then-retire",
            "approval_required": True,
        }, index=0)
        fails = [c for c in checks if c["status"] == "fail"]
        assert any("earlier than previous_version" in c["message"] for c in fails)

    def test_current_equal_to_previous_is_valid(self):
        checks = _validate_entry({
            "schema_name": "test_schema",
            "current_version": "2026-06-02",
            "previous_version": "2026-06-02",
            "status": "active",
            "owner": "team",
            "compatibility_window": "30d",
            "deprecation_policy": "announce-deprecate-then-retire",
            "approval_required": True,
        }, index=0)
        fails = [c for c in checks if c["status"] == "fail" and "earlier than previous_version" in c["message"]]
        assert len(fails) == 0, "Equal versions should be valid"


# ---------------------------------------------------------------------------
# Integration tests: validate_registry against tmp files
# ---------------------------------------------------------------------------

def _make_registry_yaml(entries: list[dict]) -> str:
    """Build a YAML string for a registry with the given schema entries."""
    lines = ["schemas:"]
    for entry in entries:
        lines.append("  - schema_name: " + entry["schema_name"])
        if "current_version" in entry:
            lines.append("    current_version: '" + entry["current_version"] + "'")
        # If absent, omit the field so the validator tests the missing-field path.
        if "previous_version" in entry:
            lines.append("    previous_version: '" + entry["previous_version"] + "'")
        lines.append("    status: " + entry.get("status", "active"))
        lines.append("    owner: " + entry.get("owner", "team"))
        lines.append("    compatibility_window: " + entry.get("compatibility_window", "30d"))
        lines.append("    deprecation_policy: " + entry.get("deprecation_policy", "announce-deprecate-then-retire"))
        lines.append("    approval_required: " + str(entry.get("approval_required", True)).lower())
    return "\n".join(lines)


class TestValidateRegistryChronology:
    def test_valid_chronology_increasing(self, tmp_path: Path):
        registry = tmp_path / "registry.yaml"
        registry.write_text(_make_registry_yaml([
            {"schema_name": "schema_a", "current_version": "2026-01-01"},
            {"schema_name": "schema_b", "current_version": "2026-06-02"},
        ]), encoding="utf-8")
        passed, report = validate_registry(registry)
        fails = [c for c in report["checks"] if c["status"] == "fail" and "chronology violation" in c["message"]]
        assert passed, f"Expected pass, got failures: {fails}"
        assert len(fails) == 0

    def test_valid_chronology_equal(self, tmp_path: Path):
        registry = tmp_path / "registry.yaml"
        registry.write_text(_make_registry_yaml([
            {"schema_name": "schema_a", "current_version": "2026-06-02"},
            {"schema_name": "schema_b", "current_version": "2026-06-02"},
        ]), encoding="utf-8")
        passed, report = validate_registry(registry)
        fails = [c for c in report["checks"] if c["status"] == "fail" and "chronology violation" in c["message"]]
        assert passed, f"Expected pass, got failures: {fails}"

    def test_invalid_chronology_decreasing(self, tmp_path: Path):
        registry = tmp_path / "registry.yaml"
        registry.write_text(_make_registry_yaml([
            {"schema_name": "schema_b", "current_version": "2026-06-02"},
            {"schema_name": "schema_a", "current_version": "2026-01-01"},  # Out of order
        ]), encoding="utf-8")
        passed, report = validate_registry(registry)
        fails = [c for c in report["checks"] if c["status"] == "fail" and "chronology violation" in c["message"]]
        assert not passed, "Decreasing version order must fail"
        assert len(fails) == 1
        assert "schema_a" in fails[0]["message"]
        assert "schema_b" in fails[0]["message"]

    def test_retired_entry_does_not_constrain_chronology(self, tmp_path: Path):
        """A retired entry can appear before an earlier active entry (historical marker)."""
        registry = tmp_path / "registry.yaml"
        registry.write_text(_make_registry_yaml([
            {"schema_name": "retired_schema", "current_version": "2026-06-02", "status": "retired"},
            {"schema_name": "active_schema", "current_version": "2025-01-01", "status": "active"},
        ]), encoding="utf-8")
        passed, report = validate_registry(registry)
        fails = [c for c in report["checks"] if c["status"] == "fail" and "chronology violation" in c["message"]]
        assert passed, f"Retired entries should not constrain chronology: {fails}"
        assert len(fails) == 0

    def test_retired_entry_in_middle_violation(self, tmp_path: Path):
        """A retired entry followed by a newer active entry is fine; but an active entry
        after retired still must not go backwards compared to the most recent non-retired."""
        # This tests that retired entries DO update prev_date (so the next active
        # must be >= retired date, not >= last active date before retired).
        registry = tmp_path / "registry.yaml"
        registry.write_text(_make_registry_yaml([
            {"schema_name": "schema_a", "current_version": "2026-01-01", "status": "active"},
            {"schema_name": "retired_schema", "current_version": "2026-06-02", "status": "retired"},
            {"schema_name": "schema_b", "current_version": "2025-01-01", "status": "active"},
        ]), encoding="utf-8")
        passed, report = validate_registry(registry)
        fails = [c for c in report["checks"] if c["status"] == "fail" and "chronology violation" in c["message"]]
        # schema_b (2025-01-01) < retired_schema (2026-06-02) even though retired
        assert not passed, "Active entry after retired must still be >= retired date"


class TestValidateRegistryMalformedVersion:
    def test_malformed_version_fails(self, tmp_path: Path):
        registry = tmp_path / "registry.yaml"
        registry.write_text(_make_registry_yaml([
            {"schema_name": "bad_schema", "current_version": "v1.0.0"},
        ]), encoding="utf-8")
        passed, report = validate_registry(registry)
        assert not passed

    def test_missing_version_fails(self, tmp_path: Path):
        registry = tmp_path / "registry.yaml"
        registry.write_text(_make_registry_yaml([
            {"schema_name": "bad_schema"},  # no current_version
        ]), encoding="utf-8")
        passed, report = validate_registry(registry)
        assert not passed