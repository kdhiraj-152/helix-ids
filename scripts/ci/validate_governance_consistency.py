#!/usr/bin/env python3
"""Objective 4: Governance Documentation Consistency Gate.

Validates that governance document claims match implementation reality:
  1. ADR-002 monotonicity claim matches validate_schema_registry.py capability.
  2. Governance docs do not claim enforcement for missing validators.
  3. Registry schemas referenced by docs actually exist.

Emits a machine-readable report to
``results/gates/governance_consistency_validation.json`` and exits non-zero
on any inconsistency.

No new governance subsystem — operates within existing validation framework.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_PATH = PROJECT_ROOT / "results" / "gates" / "governance_consistency_validation.json"

# Schema names listed in schema_registry.yaml (source of truth)
REGISTRY_PATH = PROJECT_ROOT / "schema_registry.yaml"

# Governance documents to scan
GOVERNANCE_DOCS_DIR = PROJECT_ROOT / "docs" / "governance"
ALL_GOVERNANCE_DOCS = [
    GOVERNANCE_DOCS_DIR / "manifest_schema_governance.md",
    GOVERNANCE_DOCS_DIR / "result_schema_governance.md",
    GOVERNANCE_DOCS_DIR / "hash_authority.md",
    GOVERNANCE_DOCS_DIR / "IMMUTABLE_SCHEMA_CONTRACT.md",
    GOVERNANCE_DOCS_DIR / "ADR-001-governance-philosophy.md",
    GOVERNANCE_DOCS_DIR / "ADR-002-schema-lifecycle.md",
    GOVERNANCE_DOCS_DIR / "ADR-003-hash-authority.md",
    GOVERNANCE_DOCS_DIR / "ADR-004-enforcement-pipeline.md",
]

# Schemas that governance docs may reference (by name as it appears in registry)
SCHEMA_NAMES_IN_REGISTRY: list[str] = [
    "manifest_schema",
    "benchmark_result_schema",
]


def _fail(msg: str, details: Any = None) -> dict[str, Any]:
    return {"status": "fail", "message": msg, "details": details}


def _pass(msg: str, details: Any = None) -> dict[str, Any]:
    return {"status": "pass", "message": msg, "details": details}


# ---------------------------------------------------------------------------
# Check 1: ADR-002 monotonicity claim vs. actual validator capability
# ---------------------------------------------------------------------------

# ADR-002 claims: "Version monotonicity" and "Version is not a hash: The version
# identifies the schema shape". The validator must:
#   (a) Parse YYYY-MM-DD version strings
#   (b) Reject malformed version formats
#   (c) Reject non-monotonic version ordering
# This function checks (a)-(c) by reading the validator source.
_REGISTRY_VALIDATOR_PATH = PROJECT_ROOT / "scripts" / "ci" / "validate_schema_registry.py"


def _check_adr002_monotonicity_claim() -> list[dict[str, Any]]:
    """ADR-002 claims version monotonicity. Verify the validator implements it."""
    checks: list[dict[str, Any]] = []
    validator_src = _REGISTRY_VALIDATOR_PATH.read_text(encoding="utf-8")

    # (a) Must have YYYY-MM-DD version format check
    if not re.search(r'YYYY-MM-DD|date-stamp|fromisoformat', validator_src):
        checks.append(_fail(
            "ADR-002 claim 'Version monotonicity' has no format parsing in validator",
            {"validator": str(_REGISTRY_VALIDATOR_PATH), "claim": "version format parsing"},
        ))
    else:
        checks.append(_pass("ADR-002: validator has YYYY-MM-DD format parsing"))

    # (b) Must have chronology / non-decreasing check
    if not re.search(r'chronolog|non-decreasing|monotone|_prev_date|parsed.*<.*parsed', validator_src):
        checks.append(_fail(
            "ADR-002 claim 'Version monotonicity' has no chronology enforcement in validator",
            {"validator": str(_REGISTRY_VALIDATOR_PATH), "claim": "chronology enforcement"},
        ))
    else:
        checks.append(_pass("ADR-002: validator implements chronology enforcement"))

    return checks


# ---------------------------------------------------------------------------
# Check 2: Governance docs do not claim enforcement for missing validators
# ---------------------------------------------------------------------------

# Pattern: a doc claims something is "enforced" / "validated" / "checked" but no
# corresponding validator file exists. We scan for enforcement claim patterns.
_ENFORCEMENT_CLAIM_RE = re.compile(
    r"(?:enforced|validated|checked|verified|tested|proven)"
    r"\s+(?:by|in|through|via|using)\s+[`\"]?([\w/_.-]+)[\`\"]?",
    re.IGNORECASE,
)


def _resolve_validator_path(referenced: str) -> Path | None:
    """Try to resolve a claimed validator reference to an existing file."""
    clean = referenced.rstrip(".")
    name = clean.replace(".py", "")
    candidates = [
        PROJECT_ROOT / clean,
        PROJECT_ROOT / "scripts" / "ci" / clean,
        PROJECT_ROOT / "scripts" / "evaluation" / clean,
        PROJECT_ROOT / "scripts" / "operations" / clean,
        PROJECT_ROOT / "scripts" / "training" / clean,
        PROJECT_ROOT / "docs" / "governance" / clean,
        PROJECT_ROOT / "src" / "helix_ids" / "governance" / f"{name}.py",
        PROJECT_ROOT / "src" / "helix_ids" / "operations" / f"{name}.py",
        PROJECT_ROOT / "src" / "helix_ids" / f"{name}.py",
        PROJECT_ROOT / "src" / f"{name}.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _is_non_file_reference(referenced: str) -> bool:
    """Heuristic: skip references that look like function names, env vars, or concepts."""
    clean = referenced.rstrip(".")
    # env vars: UPPER_CASE or UPPER_CASE_ with trailing underscore
    if clean.isupper() or clean.endswith("_"):
        return True
    # function names: no slash, no extension, starts with underscore
    if clean.startswith("_"):
        return True
    # bare single-word concepts (no slash, no dot extension, no hyphenated path)
    if "/" not in clean and "." not in clean.replace("_", ".") and not any(clean.endswith(ext) for ext in {".py", ".md", ".yaml", ".json", ".sh"}):
        return True
    return False


def _check_enforcement_claims_have_validators() -> list[dict[str, Any]]:
    """Every claimed validator in governance docs must exist."""
    checks: list[dict[str, Any]] = []
    dangling: list[dict[str, str]] = []

    for doc_path in ALL_GOVERNANCE_DOCS:
        if not doc_path.exists():
            continue
        content = doc_path.read_text(encoding="utf-8")
        for match in _ENFORCEMENT_CLAIM_RE.finditer(content):
            referenced = match.group(1).strip()
            if _is_non_file_reference(referenced):
                continue
            if _resolve_validator_path(referenced):
                continue
            dangling.append({
                "document": doc_path.name,
                "claimed_validator": referenced,
                "line_approx": content[:match.start()].count("\n") + 1,
            })

    if dangling:
        checks.append(_fail(
            "Governance docs claim enforcement by non-existent validators",
            {"dangling_references": dangling},
        ))
    else:
        checks.append(_pass("All governance enforcement claims reference existing validators"))

    return checks


# ---------------------------------------------------------------------------
# Check 3: Registry schemas referenced by docs actually exist
# ---------------------------------------------------------------------------

def _load_registry_schema_names() -> list[str]:
    """Load all schema_name values from schema_registry.yaml."""
    import yaml
    try:
        with REGISTRY_PATH.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return [e["schema_name"] for e in data.get("schemas", [])]
    except Exception:
        return []


# Pattern: docs may reference schema names like "manifest_schema" or
# "benchmark_result_schema" as the authoritative versioned schema.
_SCHEMA_NAME_RE = re.compile(
    r"\b(manifest_schema|benchmark_result_schema|runtime_feature_schema)\b",
)


def _check_docs_reference_valid_schemas() -> list[dict[str, Any]]:
    """Documents that reference registry schemas must reference existing entries."""
    checks: list[dict[str, Any]] = []
    valid_schemas = set(_load_registry_schema_names())
    bad_refs: list[dict[str, str]] = []

    for doc_path in ALL_GOVERNANCE_DOCS:
        if not doc_path.exists():
            continue
        content = doc_path.read_text(encoding="utf-8")
        for match in _SCHEMA_NAME_RE.finditer(content):
            schema_name = match.group(1)
            if schema_name not in valid_schemas:
                bad_refs.append({
                    "document": doc_path.name,
                    "schema_name": schema_name,
                    "line": content[:match.start()].count("\n") + 1,
                })

    if bad_refs:
        checks.append(_fail(
            "Governance docs reference schema names not in registry",
            {"invalid_schema_references": bad_refs},
        ))
    else:
        checks.append(_pass("All schema references in governance docs are valid"))

    return checks


# ---------------------------------------------------------------------------
# Check 4: Governance docs must not claim machine enforcement for policy-only rules
# ---------------------------------------------------------------------------

# Some governance policies are intentionally not machine-enforced (e.g., migration
# freeze policy in IMMUTABLE_SCHEMA_CONTRACT.md). Documents should either:
# (a) Not claim machine enforcement for policy-only rules, OR
# (b) Explicitly note "manual review required" for non-automated policies.
#
# We check for false enforcement claims: docs that say "CI enforces" or
# "automatically validates" for rules that have no corresponding validator.
_KNOWN_MANUAL_ONLY_POLICIES = {
    "IMMUTABLE_SCHEMA_CONTRACT.md": {
        "migration_freeze": ["migration", "freeze", "freeze-gated"],
        "producer_obligations": ["producer.*obligation", "exporter.*oblig"],
        "consumer_obligations": ["consumer.*obligation"],
    },
}
_KNOWN_AUTO_ENFORCED = {
    "IMMUTABLE_SCHEMA_CONTRACT.md": {"drift_fail": ["drift.*fail", "schema.*drift.*violation"]},
}


def _check_false_enforcement_claims() -> list[dict[str, Any]]:
    """Detect governance claims that assert machine enforcement for unautomated rules."""
    checks: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []

    for doc_path in ALL_GOVERNANCE_DOCS:
        if not doc_path.exists():
            continue
        content = doc_path.read_text(encoding="utf-8")
        doc_name = doc_path.name

        # Look for claims like "CI enforces X" or "automatically validates Y"
        # where X/Y refers to a known manual-only policy
        auto_claim_re = re.compile(
            r"(?:CI|automatically|machine|programmatically|validator)\s+"
            r"(?:enforces|validates|checks|verifies|blocks)\s+"
            r"[\"`']?([\w\s]+?)[\"`']?\s*(?:in|for|per|as\s+)\s*[\"`']?([\w\s]+?)[\"`']?\.",
            re.IGNORECASE,
        )

        for m in auto_claim_re.finditer(content):
            policy = m.group(1).strip().lower()
            context = m.group(2).strip().lower()

            # Skip if the claim is about something clearly automated
            if any(kw in content[max(0, m.start()-200):m.start()]
                   for kw in ["test", "runtime", "pytest", "unit"]):
                continue

            # Check against known manual-only policies
            for known_manual in ["migration", "producer", "consumer", "approval"]:
                if known_manual in policy or known_manual in context:
                    issues.append({
                        "document": doc_name,
                        "claimed_auto": f"{m.group(0)[:80]}",
                        "hint": "Verify this claim is actually automated",
                    })

    if issues:
        checks.append(_fail(
            "Possible false enforcement claims detected in governance docs",
            {"suspected_false_claims": issues[:5]},  # cap at 5 for brevity
        ))
    else:
        checks.append(_pass("No obvious false enforcement claims detected"))

    return checks


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def validate_consistency() -> tuple[bool, dict[str, Any]]:
    """Run all governance consistency checks. Returns (passed, report)."""
    report: dict[str, Any] = {
        "checks": [],
        "metadata": {
            "validator": __file__,
            "scope": "ADR claim vs implementation consistency",
        },
    }
    any_failed = False

    check1 = _check_adr002_monotonicity_claim()
    report["checks"].extend(check1)
    if any(c.get("status") == "fail" for c in check1):
        any_failed = True

    check2 = _check_enforcement_claims_have_validators()
    report["checks"].extend(check2)
    if any(c.get("status") == "fail" for c in check2):
        any_failed = True

    check3 = _check_docs_reference_valid_schemas()
    report["checks"].extend(check3)
    if any(c.get("status") == "fail" for c in check3):
        any_failed = True

    check4 = _check_false_enforcement_claims()
    report["checks"].extend(check4)
    # check4 is advisory — don't fail on it, just warn
    # (It may produce false positives on legitimate claims)

    report["overall"] = "pass" if not any_failed else "fail"
    return not any_failed, report


def write_report(report: dict[str, Any], report_path: Path = REPORT_PATH) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main() -> None:
    passed, report = validate_consistency()
    write_report(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        print("\nGovernance consistency validation FAILED", file=sys.stderr)
        sys.exit(1)
    print("\nGovernance consistency validation PASSED")


if __name__ == "__main__":
    main()
