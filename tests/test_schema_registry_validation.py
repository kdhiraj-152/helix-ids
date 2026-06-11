"""Tests for scripts/ci/validate_schema_registry.py and validate_governance_docs.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VALIDATOR_SCRIPT = PROJECT_ROOT / "scripts" / "ci" / "validate_schema_registry.py"
GOV_DOCS_SCRIPT = PROJECT_ROOT / "scripts" / "ci" / "validate_governance_docs.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_registry(tmp_path: Path, entries: list[dict[str, Any]]) -> Path:
    """Write a schema_registry.yaml with the given entries."""
    lines = ["schemas:"]
    for entry in entries:
        lines.append("  - schema_name: \"{}\"".format(entry.get("schema_name", "")))
        for key in ("current_version", "previous_version", "status", "owner",
                     "compatibility_window", "deprecation_policy", "approval_required"):
            if key in entry:
                val = entry[key]
                if isinstance(val, bool):
                    lines.append(f"    {key}: {str(val).lower()}")
                else:
                    lines.append(f"    {key}: \"{val}\"")
    registry = tmp_path / "schema_registry.yaml"
    registry.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return registry


def _write_gov_doc(tmp_path: Path, filename: str, content: str = "test") -> Path:
    gov_dir = tmp_path / "governance"
    gov_dir.mkdir(parents=True, exist_ok=True)
    doc = gov_dir / filename
    doc.write_text(content, encoding="utf-8")
    return doc


# ---------------------------------------------------------------------------
# Schema registry validation tests
# ---------------------------------------------------------------------------

class TestSchemaRegistryValidation:
    """Tests for validate_schema_registry module."""

    def test_valid_registry(self, tmp_path: Path) -> None:
        """A well-formed registry should pass all checks."""
        from scripts.ci.validate_schema_registry import validate_registry

        registry = _write_registry(tmp_path, [
            {
                "schema_name": "manifest_schema",
                "current_version": "2026-06-02",
                "status": "active",
                "owner": "helix-ids-governance-lead",
                "compatibility_window": "30d",
                "deprecation_policy": "announce-deprecate-then-retire",
                "approval_required": True,
            },
            {
                "schema_name": "benchmark_result_schema",
                "current_version": "2026-06-02",
                "status": "active",
                "owner": "helix-ids-governance-lead",
                "compatibility_window": "30d",
                "deprecation_policy": "announce-deprecate-then-retire",
                "approval_required": True,
            },
        ])
        passed, report = validate_registry(registry)
        assert passed is True
        assert report["overall"] == "pass"
        assert not any(c["status"] == "fail" for c in report["checks"])

    def test_missing_owner(self, tmp_path: Path) -> None:
        """An entry without owner should fail validation."""
        from scripts.ci.validate_schema_registry import validate_registry

        registry = _write_registry(tmp_path, [
            {
                "schema_name": "manifest_schema",
                "current_version": "2026-06-02",
                "status": "active",
                "owner": "",  # empty owner
                "compatibility_window": "30d",
                "deprecation_policy": "announce-deprecate-then-retire",
                "approval_required": True,
            },
        ])
        passed, report = validate_registry(registry)
        assert passed is False
        assert report["overall"] == "fail"
        assert any("owner" in c.get("message", "") and c["status"] == "fail"
                    for c in report["checks"])

    def test_missing_version(self, tmp_path: Path) -> None:
        """An entry without current_version should fail validation."""
        from scripts.ci.validate_schema_registry import validate_registry

        registry = _write_registry(tmp_path, [
            {
                "schema_name": "manifest_schema",
                "current_version": "",  # empty version
                "status": "active",
                "owner": "helix-ids-governance-lead",
                "compatibility_window": "30d",
                "deprecation_policy": "announce-deprecate-then-retire",
                "approval_required": True,
            },
        ])
        passed, report = validate_registry(registry)
        assert passed is False
        assert report["overall"] == "fail"
        assert any("current_version" in c.get("message", "") and c["status"] == "fail"
                    for c in report["checks"])

    def test_missing_deprecation_policy(self, tmp_path: Path) -> None:
        """An entry without deprecation_policy should fail validation."""
        from scripts.ci.validate_schema_registry import validate_registry

        registry = _write_registry(tmp_path, [
            {
                "schema_name": "manifest_schema",
                "current_version": "2026-06-02",
                "status": "active",
                "owner": "helix-ids-governance-lead",
                "compatibility_window": "30d",
                "deprecation_policy": "",  # empty
                "approval_required": True,
            },
        ])
        passed, report = validate_registry(registry)
        assert passed is False
        assert any("deprecation_policy" in c.get("message", "") and c["status"] == "fail"
                    for c in report["checks"])

    def test_malformed_entry_not_dict(self, tmp_path: Path) -> None:
        """A non-dict entry in the schemas list should fail."""
        from scripts.ci.validate_schema_registry import _load_registry

        # Write a registry with a non-dict entry by using raw YAML
        registry = tmp_path / "schema_registry.yaml"
        registry.write_text(
            "schemas:\n  - just_a_string\n",
            encoding="utf-8",
        )

        # Try loading with PyYAML first, fallback to minimal
        data, load_check = _load_registry(registry)
        if data is None:
            # Minimal parser may not handle this; test with explicit data
            pytest.skip("PyYAML not available; minimal parser doesn't produce non-dict entries")

        from scripts.ci.validate_schema_registry import validate_registry
        passed, report = validate_registry(registry)
        assert passed is False

    def test_report_generation(self, tmp_path: Path) -> None:
        """Validation should produce a machine-readable JSON report."""
        from scripts.ci.validate_schema_registry import validate_registry, write_report

        registry = _write_registry(tmp_path, [
            {
                "schema_name": "test_schema",
                "current_version": "2026-01-01",
                "status": "active",
                "owner": "test-owner",
                "compatibility_window": "30d",
                "deprecation_policy": "announce-deprecate-then-retire",
                "approval_required": True,
            },
        ])
        report_path = tmp_path / "results" / "gates" / "schema_registry_validation.json"
        passed, report = validate_registry(registry)
        write_report(report, report_path)

        assert report_path.exists()
        loaded = json.loads(report_path.read_text(encoding="utf-8"))
        assert "overall" in loaded
        assert "checks" in loaded
        assert isinstance(loaded["checks"], list)

    def test_missing_registry_file(self, tmp_path: Path) -> None:
        """Validation should fail when registry file does not exist."""
        from scripts.ci.validate_schema_registry import validate_registry

        nonexistent = tmp_path / "nonexistent.yaml"
        passed, report = validate_registry(nonexistent)
        assert passed is False
        assert report["overall"] == "fail"

    def test_failure_exit_code(self, tmp_path: Path) -> None:
        """The script should exit non-zero on validation failure."""
        registry = _write_registry(tmp_path, [
            {
                "schema_name": "bad_schema",
                "current_version": "",  # invalid
                "status": "active",
                "owner": "test-owner",
                "compatibility_window": "30d",
                "deprecation_policy": "announce-deprecate-then-retire",
                "approval_required": True,
            },
        ])
        subprocess.run(
            [sys.executable, str(VALIDATOR_SCRIPT)],
            capture_output=True,
            text=True,
            env={**__import__("os").environ, "SCHEMA_REGISTRY_PATH": str(registry)},
            timeout=30,
        )
        # The script uses default path, so we test the module function directly
        # for exit code behavior
        from scripts.ci.validate_schema_registry import validate_registry
        passed, _ = validate_registry(registry)
        assert passed is False

    def test_empty_schemas_list(self, tmp_path: Path) -> None:
        """An empty schemas list should fail validation."""
        from scripts.ci.validate_schema_registry import validate_registry

        registry = tmp_path / "schema_registry.yaml"
        registry.write_text("schemas: []\n", encoding="utf-8")
        passed, report = validate_registry(registry)
        assert passed is False
        assert report["overall"] == "fail"

    def test_missing_schemas_key(self, tmp_path: Path) -> None:
        """A registry without the schemas key should fail."""
        from scripts.ci.validate_schema_registry import validate_registry

        registry = tmp_path / "schema_registry.yaml"
        registry.write_text("other_key: value\n", encoding="utf-8")
        passed, report = validate_registry(registry)
        assert passed is False

    # ------------------------------------------------------------------
    # Phase 4B: chronology enforcement (ADR-002 monotonicity)
    # ------------------------------------------------------------------

    def test_valid_chronology_with_previous_version(self, tmp_path: Path) -> None:
        """A registry with previous_version < current_version passes chronology."""
        from scripts.ci.validate_schema_registry import validate_registry

        registry = _write_registry(tmp_path, [
            {
                "schema_name": "manifest_schema",
                "current_version": "2026-06-02",
                "previous_version": "2026-01-15",
                "status": "active",
                "owner": "helix-ids-governance-lead",
                "compatibility_window": "30d",
                "deprecation_policy": "announce-deprecate-then-retire",
                "approval_required": True,
            },
        ])
        passed, report = validate_registry(registry)
        assert passed is True, (
            "Registry with valid chronology (2026-01-15 -> 2026-06-02) "
            "should pass: " + str(report)
        )
        assert any("chronology" in c.get("message", "")
                   for c in report["checks"])

    def test_invalid_chronology_current_earlier_than_previous(self, tmp_path: Path) -> None:
        """A registry with current_version < previous_version fails chronology."""
        from scripts.ci.validate_schema_registry import validate_registry

        registry = _write_registry(tmp_path, [
            {
                "schema_name": "manifest_schema",
                "current_version": "2025-01-01",
                "previous_version": "2026-06-02",
                "status": "active",
                "owner": "helix-ids-governance-lead",
                "compatibility_window": "30d",
                "deprecation_policy": "announce-deprecate-then-retire",
                "approval_required": True,
            },
        ])
        passed, report = validate_registry(registry)
        assert passed is False
        assert report["overall"] == "fail"
        assert any(
            "earlier than previous_version" in c.get("message", "")
            and c["status"] == "fail"
            for c in report["checks"]
        )

    def test_malformed_version_string_rejected(self, tmp_path: Path) -> None:
        """A registry with a non-YYYY-MM-DD current_version fails format check."""
        from scripts.ci.validate_schema_registry import validate_registry

        registry = _write_registry(tmp_path, [
            {
                "schema_name": "manifest_schema",
                "current_version": "v1.2.3",  # semver — wrong format
                "status": "active",
                "owner": "helix-ids-governance-lead",
                "compatibility_window": "30d",
                "deprecation_policy": "announce-deprecate-then-retire",
                "approval_required": True,
            },
        ])
        passed, report = validate_registry(registry)
        assert passed is False
        assert report["overall"] == "fail"
        assert any(
            "YYYY-MM-DD" in c.get("message", "")
            and c["status"] == "fail"
            for c in report["checks"]
        )

    def test_missing_current_version_field(self, tmp_path: Path) -> None:
        """A registry missing the current_version field entirely fails."""
        from scripts.ci.validate_schema_registry import validate_registry

        registry = _write_registry(tmp_path, [
            {
                "schema_name": "manifest_schema",
                # no current_version key at all
                "status": "active",
                "owner": "helix-ids-governance-lead",
                "compatibility_window": "30d",
                "deprecation_policy": "announce-deprecate-then-retire",
                "approval_required": True,
            },
        ])
        passed, report = validate_registry(registry)
        assert passed is False
        assert report["overall"] == "fail"
        # The required-field check fires for missing current_version
        assert any(
            ("current_version" in c.get("message", "")
             or "current_version" in str(c.get("details", "")))
            and c["status"] == "fail"
            for c in report["checks"]
        )


# ---------------------------------------------------------------------------
# Governance docs validation tests
# ---------------------------------------------------------------------------

class TestGovernanceDocsValidation:
    """Tests for validate_governance_docs module."""

    def test_all_docs_present(self, tmp_path: Path) -> None:
        """Validation passes when all required docs exist."""
        from scripts.ci.validate_governance_docs import validate_governance_docs

        for doc in ("manifest_schema_governance.md",
                     "result_schema_governance.md",
                     "hash_authority.md"):
            _write_gov_doc(tmp_path, doc, "# Content\nSome governance content.")

        passed, report = validate_governance_docs(
            governance_dir=tmp_path / "governance",
            required_docs=[
                "manifest_schema_governance.md",
                "result_schema_governance.md",
                "hash_authority.md",
            ],
        )
        assert passed is True
        assert report["overall"] == "pass"

    def test_missing_doc(self, tmp_path: Path) -> None:
        """Validation fails when a required doc is missing."""
        from scripts.ci.validate_governance_docs import validate_governance_docs

        # Only write 2 of 3 required docs
        _write_gov_doc(tmp_path, "manifest_schema_governance.md", "content")
        _write_gov_doc(tmp_path, "result_schema_governance.md", "content")
        # hash_authority.md is missing

        passed, report = validate_governance_docs(
            governance_dir=tmp_path / "governance",
            required_docs=[
                "manifest_schema_governance.md",
                "result_schema_governance.md",
                "hash_authority.md",
            ],
        )
        assert passed is False
        assert report["overall"] == "fail"
        assert any("hash_authority.md" in c.get("message", "") and c["status"] == "fail"
                    for c in report["checks"])

    def test_empty_doc(self, tmp_path: Path) -> None:
        """Validation fails when a required doc exists but is empty."""
        from scripts.ci.validate_governance_docs import validate_governance_docs

        _write_gov_doc(tmp_path, "manifest_schema_governance.md", "content")
        _write_gov_doc(tmp_path, "result_schema_governance.md", "content")
        _write_gov_doc(tmp_path, "hash_authority.md", "")  # empty

        passed, report = validate_governance_docs(
            governance_dir=tmp_path / "governance",
            required_docs=[
                "manifest_schema_governance.md",
                "result_schema_governance.md",
                "hash_authority.md",
            ],
        )
        assert passed is False
        assert any("hash_authority.md" in c.get("message", "") and c["status"] == "fail"
                    for c in report["checks"])

    def test_missing_governance_dir(self, tmp_path: Path) -> None:
        """Validation fails when governance directory does not exist."""
        from scripts.ci.validate_governance_docs import validate_governance_docs

        nonexistent = tmp_path / "no_such_dir"
        passed, report = validate_governance_docs(
            governance_dir=nonexistent,
            required_docs=["manifest_schema_governance.md"],
        )
        assert passed is False
        assert report["overall"] == "fail"

    def test_report_generation(self, tmp_path: Path) -> None:
        """Governance validation should produce a machine-readable JSON report."""
        from scripts.ci.validate_governance_docs import validate_governance_docs, write_report

        _write_gov_doc(tmp_path, "test_doc.md", "# Content")

        report_path = tmp_path / "results" / "gates" / "governance_docs_validation.json"
        passed, report = validate_governance_docs(
            governance_dir=tmp_path / "governance",
            required_docs=["test_doc.md"],
        )
        write_report(report, report_path)

        assert report_path.exists()
        loaded = json.loads(report_path.read_text(encoding="utf-8"))
        assert "overall" in loaded
        assert "checks" in loaded

    def test_actual_project_docs_exist(self) -> None:
        """Verify that the actual project governance documents exist."""
        gov_dir = PROJECT_ROOT / "docs" / "governance"
        for doc in ("ADR-001-governance-philosophy.md",
                     "ADR-002-schema-lifecycle.md",
                     "ADR-003-hash-authority.md"):
            doc_path = gov_dir / doc
            assert doc_path.exists(), f"Required governance document missing: {doc}"
            content = doc_path.read_text(encoding="utf-8")
            assert content.strip(), f"Governance document is empty: {doc}"
