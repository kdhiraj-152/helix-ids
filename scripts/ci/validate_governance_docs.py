#!/usr/bin/env python3
"""CI validator for governance documentation completeness.

Verifies that all required governance documents exist and are non-empty.
Emits a machine-readable JSON report to
``results/gates/governance_docs_validation.json`` and exits non-zero when
any required document is missing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GOVERNANCE_DIR = PROJECT_ROOT / "docs" / "governance"
REPORT_PATH = PROJECT_ROOT / "results" / "gates" / "governance_docs_validation.json"

REQUIRED_DOCS = [
    "manifest_schema_governance.md",
    "result_schema_governance.md",
    "hash_authority.md",
    "IMMUTABLE_SCHEMA_CONTRACT.md",
]


def _fail(msg: str, details: Any = None) -> dict[str, Any]:
    return {"status": "fail", "message": msg, "details": details}


def _pass(msg: str, details: Any = None) -> dict[str, Any]:
    return {"status": "pass", "message": msg, "details": details}


def validate_governance_docs(
    governance_dir: Path = GOVERNANCE_DIR,
    required_docs: list[str] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Run full governance documentation validation.

    Returns (passed, report_dict).
    """
    if required_docs is None:
        required_docs = list(REQUIRED_DOCS)

    report: dict[str, Any] = {
        "governance_dir": str(governance_dir),
        "required_docs": required_docs,
        "checks": [],
    }
    any_failed = False

    # 1. Governance directory must exist
    if not governance_dir.exists():
        report["checks"].append(_fail(
            "Governance directory does not exist",
            {"path": str(governance_dir)},
        ))
        any_failed = True
        report["overall"] = "fail"
        return not any_failed, report

    report["checks"].append(_pass("Governance directory exists", str(governance_dir)))

    # 2. Each required document must exist and be non-empty
    for doc_name in required_docs:
        doc_path = governance_dir / doc_name

        if not doc_path.exists():
            report["checks"].append(_fail(
                f"Required governance document missing: {doc_name}",
                {"path": str(doc_path)},
            ))
            any_failed = True
            continue

        try:
            content = doc_path.read_text(encoding="utf-8")
        except Exception as exc:
            report["checks"].append(_fail(
                f"Cannot read governance document: {doc_name}",
                {"path": str(doc_path), "error": str(exc)},
            ))
            any_failed = True
            continue

        if not content.strip():
            report["checks"].append(_fail(
                f"Governance document is empty: {doc_name}",
                {"path": str(doc_path)},
            ))
            any_failed = True
            continue

        report["checks"].append(_pass(
            f"Governance document present: {doc_name}",
            {"path": str(doc_path), "size_bytes": len(content.encode("utf-8"))},
        ))

    report["overall"] = "pass" if not any_failed else "fail"
    return not any_failed, report


def write_report(report: dict[str, Any], report_path: Path = REPORT_PATH) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main() -> None:
    passed, report = validate_governance_docs()
    write_report(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        print("\nGovernance documentation validation FAILED", file=sys.stderr)
        sys.exit(1)
    print("\nGovernance documentation validation PASSED")


if __name__ == "__main__":
    main()
