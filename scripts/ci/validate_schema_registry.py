#!/usr/bin/env python3
"""CI validator for the schema lifecycle registry.

Validates that ``schema_registry.yaml`` exists, is well-formed, and that every
entry satisfies governance requirements (owner, version, deprecation policy).
Emits a machine-readable JSON report to ``results/gates/schema_registry_validation.json``
and exits non-zero on any validation failure.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REGISTRY_PATH = PROJECT_ROOT / "schema_registry.yaml"
REPORT_PATH = PROJECT_ROOT / "results" / "gates" / "schema_registry_validation.json"

REQUIRED_ENTRY_FIELDS = {
    "schema_name",
    "current_version",
    "status",
    "owner",
    "compatibility_window",
    "deprecation_policy",
    "approval_required",
}

VALID_STATUSES = {"active", "deprecated", "retired"}

# Version format: YYYY-MM-DD date-stamp (per ADR-002 and manifest/result schema governance).
# Strict format check; semantic validation is delegated to ``date.fromisoformat``.
_VERSION_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _parse_version(version: Any) -> date | None:
    """Parse a YYYY-MM-DD version string into a ``date``.

    Returns ``None`` if the value is not a string in the required format
    (caller is expected to record a failure for malformed values).
    """
    if not isinstance(version, str) or not _VERSION_RE.match(version):
        return None
    try:
        return date.fromisoformat(version)
    except ValueError:
        return None


def _date_to_str(d: date) -> str:
    """Serialize a date back to YYYY-MM-DD string."""
    return d.isoformat()


def _fail(msg: str, details: Any = None) -> dict[str, Any]:
    return {"status": "fail", "message": msg, "details": details}


def _pass(msg: str, details: Any = None) -> dict[str, Any]:
    return {"status": "pass", "message": msg, "details": details}


def _load_registry(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Load the YAML registry. Returns (data, check_result)."""
    if not path.exists():
        return None, _fail("Schema registry file not found", str(path))

    try:
        import yaml  # noqa: F811 – deferred import
    except ImportError:
        # Minimal YAML parser fallback: only handles our flat structure
        return _load_registry_minimal(path)

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except Exception as exc:
        return None, _fail("Registry YAML parse error", str(exc))

    if not isinstance(data, dict):
        return None, _fail("Registry root is not a mapping", type(data).__name__)

    return data, _pass("Registry loaded")


def _load_registry_minimal(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Minimal YAML parser for the registry when PyYAML is unavailable."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return None, _fail("Registry read error", str(exc))

    entries: list[dict[str, str]] = []
    current: dict[str, str] = {}
    in_schemas_list = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.lstrip()

        # Skip comments and blanks
        if not stripped or stripped.startswith("#"):
            continue

        # Detect schemas: key
        if stripped.startswith("schemas:"):
            in_schemas_list = True
            continue

        if not in_schemas_list:
            continue

        # List item marker
        if stripped.startswith("- "):
            if current:
                entries.append(current)
            current = {}
            stripped = stripped[2:].strip()

        # Key: value
        if ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if value in ("true", "True"):
                value = True
            elif value in ("false", "False"):
                value = False
            current[key] = value
            continue

    if current:
        entries.append(current)

    data: dict[str, Any] = {"schemas": entries}
    return data, _pass("Registry loaded (minimal parser)")


def _validate_entry(entry: dict[str, Any], index: int) -> list[dict[str, Any]]:
    """Validate a single registry entry. Returns list of check dicts."""
    checks: list[dict[str, Any]] = []
    entry_name = entry.get("schema_name", f"<entry-{index}>")

    # Required fields
    missing = sorted(REQUIRED_ENTRY_FIELDS - set(entry.keys()))
    if missing:
        checks.append(_fail(
            f"Entry '{entry_name}' missing required fields",
            {"missing_fields": missing},
        ))
        return checks  # Cannot continue validating a malformed entry

    checks.append(_pass(f"Entry '{entry_name}' has all required fields"))

    # Validate version is a non-empty string in YYYY-MM-DD format
    version = entry.get("current_version")
    if not isinstance(version, str) or not version.strip():
        checks.append(_fail(
            f"Entry '{entry_name}' current_version is missing or not a string",
            {"current_version": repr(version)},
        ))
    elif not _VERSION_RE.match(version):
        checks.append(_fail(
            f"Entry '{entry_name}' current_version is not a YYYY-MM-DD date-stamp",
            {"current_version": repr(version), "expected_format": "YYYY-MM-DD"},
        ))
    else:
        parsed = _parse_version(version)
        if parsed is None:
            checks.append(_fail(
                f"Entry '{entry_name}' current_version is not a valid calendar date",
                {"current_version": repr(version), "expected_format": "YYYY-MM-DD"},
            ))
        else:
            checks.append(_pass(f"Entry '{entry_name}' current_version valid", version))

    # Validate previous_version format (if present) and chronology
    previous_version = entry.get("previous_version")
    parsed_current = _parse_version(version) if isinstance(version, str) and _VERSION_RE.match(version or "") else None
    if previous_version is not None:
        if not isinstance(previous_version, str) or not previous_version.strip():
            checks.append(_fail(
                f"Entry '{entry_name}' previous_version is present but not a non-empty string",
                {"previous_version": repr(previous_version)},
            ))
        elif not _VERSION_RE.match(previous_version):
            checks.append(_fail(
                f"Entry '{entry_name}' previous_version is not a YYYY-MM-DD date-stamp",
                {"previous_version": repr(previous_version), "expected_format": "YYYY-MM-DD"},
            ))
        else:
            parsed_previous = _parse_version(previous_version)
            if parsed_previous is None:
                checks.append(_fail(
                    f"Entry '{entry_name}' previous_version is not a valid calendar date",
                    {"previous_version": repr(previous_version), "expected_format": "YYYY-MM-DD"},
                ))
            elif parsed_current is not None and parsed_current < parsed_previous:
                checks.append(_fail(
                    f"Entry '{entry_name}' current_version is earlier than previous_version",
                    {
                        "schema_name": entry_name,
                        "current_version": version,
                        "previous_version": previous_version,
                    },
                ))
            else:
                checks.append(_pass(
                    f"Entry '{entry_name}' version chronology valid (previous={previous_version} <= current={version})",
                ))

    # Validate owner is a non-empty string
    owner = entry.get("owner")
    if not isinstance(owner, str) or not owner.strip():
        checks.append(_fail(
            f"Entry '{entry_name}' owner is missing or not a string",
            {"owner": repr(owner)},
        ))
    else:
        checks.append(_pass(f"Entry '{entry_name}' owner valid", owner))

    # Validate deprecation_policy is a non-empty string
    dep_policy = entry.get("deprecation_policy")
    if not isinstance(dep_policy, str) or not dep_policy.strip():
        checks.append(_fail(
            f"Entry '{entry_name}' deprecation_policy is missing or not a string",
            {"deprecation_policy": repr(dep_policy)},
        ))
    else:
        checks.append(_pass(f"Entry '{entry_name}' deprecation_policy valid", dep_policy))

    # Validate status
    status = entry.get("status")
    if not isinstance(status, str) or status not in VALID_STATUSES:
        checks.append(_fail(
            f"Entry '{entry_name}' status is invalid",
            {"status": repr(status), "valid_statuses": sorted(VALID_STATUSES)},
        ))
    else:
        checks.append(_pass(f"Entry '{entry_name}' status valid", status))

    # Validate approval_required is boolean
    approval = entry.get("approval_required")
    if not isinstance(approval, bool):
        checks.append(_fail(
            f"Entry '{entry_name}' approval_required is not boolean",
            {"approval_required": repr(approval)},
        ))
    else:
        checks.append(_pass(f"Entry '{entry_name}' approval_required valid", approval))

    # Validate compatibility_window is a non-empty string
    cw = entry.get("compatibility_window")
    if not isinstance(cw, str) or not cw.strip():
        checks.append(_fail(
            f"Entry '{entry_name}' compatibility_window is missing or not a string",
            {"compatibility_window": repr(cw)},
        ))
    else:
        checks.append(_pass(f"Entry '{entry_name}' compatibility_window valid", cw))

    return checks


def validate_registry(registry_path: Path = REGISTRY_PATH) -> tuple[bool, dict[str, Any]]:
    """Run full validation. Returns (passed, report_dict)."""
    report: dict[str, Any] = {"registry_path": str(registry_path), "checks": []}
    any_failed = False

    # 1. Load
    data, load_check = _load_registry(registry_path)
    report["checks"].append(load_check)
    if data is None:
        any_failed = True
        report["overall"] = "fail"
        return not any_failed, report

    # 2. Top-level 'schemas' key must exist
    schemas = data.get("schemas")
    if schemas is None:
        report["checks"].append(_fail("Registry missing top-level 'schemas' key"))
        any_failed = True
        report["overall"] = "fail"
        return not any_failed, report

    if not isinstance(schemas, list):
        report["checks"].append(_fail("Registry 'schemas' is not a list", type(schemas).__name__))
        any_failed = True
        report["overall"] = "fail"
        return not any_failed, report

    if len(schemas) == 0:
        report["checks"].append(_fail("Registry 'schemas' list is empty"))
        any_failed = True
        report["overall"] = "fail"
        return not any_failed, report

    report["checks"].append(_pass(f"Registry has {len(schemas)} schema entries"))

    # 3. Validate each entry
    for idx, entry in enumerate(schemas):
        if not isinstance(entry, dict):
            report["checks"].append(_fail(f"Schema entry {idx} is not a mapping", type(entry).__name__))
            any_failed = True
            continue
        entry_checks = _validate_entry(entry, idx)
        report["checks"].extend(entry_checks)
        if any(c.get("status") == "fail" for c in entry_checks):
            any_failed = True

    # 4. Entry-list chronology: versions must appear in monotonically non-decreasing order.
    #    Entries without a valid parseable current_version are skipped for chronology.
    #    Entries with status=retired are historical markers; they do NOT constrain the
    #    active-entry chronology. Two anchors are maintained:
    #      - prev_date:       latest date overall (including retired); used for error reporting
    #      - prev_active_date: latest date among non-retired entries; used for comparison
    _chronology_failures: list[dict[str, Any]] = []
    prev_active_date: date | None = None
    prev_active_name: str = ""
    for entry in schemas:
        if not isinstance(entry, dict):
            continue
        name = entry.get("schema_name", "<unknown>")
        version_str = entry.get("current_version", "")
        status = entry.get("status", "")
        parsed = _parse_version(version_str) if isinstance(version_str, str) else None
        if parsed is None:
            continue  # Malformed version already caught per-entry; skip chronology
        if status == "retired":
            continue
        # Active entry: compare only against prev_active_date (last non-retired date)
        if prev_active_date is not None and parsed < prev_active_date:
            _chronology_failures.append(_fail(
                f"Entry-list chronology violation: '{name}' version {version_str} "
                f"is earlier than '{prev_active_name}' version {_date_to_str(prev_active_date)}",
                {
                    "offending_entry": name,
                    "offending_version": version_str,
                    "previous_entry": prev_active_name,
                    "previous_version": _date_to_str(prev_active_date),
                },
            ))
        else:
            prev_active_date = parsed
            prev_active_name = name
    report["checks"].extend(_chronology_failures)
    if _chronology_failures:
        any_failed = True

    report["overall"] = "pass" if not any_failed else "fail"
    return not any_failed, report


def write_report(report: dict[str, Any], report_path: Path = REPORT_PATH) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main() -> None:
    passed, report = validate_registry()
    write_report(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        print("\nSchema registry validation FAILED", file=sys.stderr)
        sys.exit(1)
    print("\nSchema registry validation PASSED")


if __name__ == "__main__":
    main()
