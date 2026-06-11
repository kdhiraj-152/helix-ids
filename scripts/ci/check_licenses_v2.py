#!/usr/bin/env python3
"""Expanded license compliance checker with machine-readable compliance report.

Produces:
    results/licenses/compliance-report.json — machine-readable compliance report
    stdout — human-readable summary with violations

Usage:
    python3 scripts/ci/check_licenses_v2.py results/licenses/licenses.csv

Exit code:
    0 — all licenses acceptable or known exceptions
    1 — one or more disallowed licenses found
"""

import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# --- Configuration ---

DISALLOWED_PATTERNS = [
    r"gnu\s+(a-)?gpl",
    r"gnu\s+lesser\s+general\s+public\s+license",
    r"lgpl",
    r"agpl",
    r"\bgpl[\s-]",
    r"\bgpl$",
    r"copyheart",
    r"european\s+union\s+public\s+license",
    r"eupl",
    r"common\s+public\s+license",
    r"cpl-1",
    r"reciprocal\s+public\s+license",
    r"rpl-1",
    r"affero",
]

ACCEPTABLE_LICENSES = [
    "mit", "mit license",
    "bsd", "bsd license", "bsd-2-clause", "bsd-3-clause",
    "apache", "apache-2.0", "apache software license", "apache license 2.0",
    "apache-2.0 and mit", "apache-2.0 or bsd-3-clause",
    "isc", "isc license (iscl)",
    "psf", "python software foundation license",
    "mpl-2.0", "mozilla public license 2.0 (mpl 2.0)", "mozilla public license 2.0",
    "cc0-1.0",
    "unicode-dfs-2016",
    "unlicense",
    "zlib",
    "unknown",  # accepted with review
    "public domain",
    "python",
    "pkg-config",  # build tool
    # Additional BSD variants
    "bsd-4-clause (bsd-4-clause)",
    "historical permission notice and disclaimer (bsd-4-clause)",
    # OpenSSL / MIT-style
    "openSSL",
    # CNRI
    "cnri open source license (cnri)",
]

# Known transitive dependencies with accepted GPL/LGPL exceptions
# These are documented in docs/compliance/LICENSE_POLICY.md §7
ACCEPTED_EXCEPTIONS = [
    {
        "name": "PyMuPDF",
        "version": "1.27.2.2",
        "license": "Dual Licensed - GNU AFFERO GPL 3.0 or Artifex Commercial License",
        "classified_as": "AGPL/GPL (dual)",
        "rationale": "Transitive dependency of pdf2docx (MIT). pdf2docx is not in requirements-lock.txt (manually installed for documentation generation). Not part of the production dependency tree.",
        "review_date": "2026-06-12",
        "action": "ACCEPTED — not in production dependency tree; can use Artifex commercial license option",
    },
    {
        "name": "chardet",
        "version": "5.2.0",
        "license": "GNU Lesser General Public License v2 or later (LGPLv2+)",
        "classified_as": "LGPL-2.0-or-later",
        "rationale": "Transitive dependency of cyclonedx-bom (SBOM generation) and pygount (code counting). Both are CI/dev tools only — never shipped in production image. LGPL v2+ is weak copyleft with linking exception.",
        "review_date": "2026-06-12",
        "action": "ACCEPTED — dev-only transitive dependency; LGPL linking exception applies",
    },
    {
        "name": "yattag",
        "version": "1.16.1",
        "license": "GNU Library or Lesser General Public License (LGPL)",
        "classified_as": "LGPL",
        "rationale": "Transitive dependency of cosmic_ray (mutation testing tool). Cosmic-Ray is a CI-only development tool — never shipped in production image. LGPL is weak copyleft with linking exception.",
        "review_date": "2026-06-12",
        "action": "ACCEPTED — dev-only transitive dependency; LGPL linking exception applies",
    },
]


def normalize(s: str) -> str:
    return s.strip().lower()


def is_acceptable(license_str: str) -> bool:
    n = normalize(license_str)
    if n in ACCEPTABLE_LICENSES:
        return True
    for acc in ACCEPTABLE_LICENSES:
        if acc in n:
            return True
    return False


def is_disallowed(license_str: str) -> bool:
    n = normalize(license_str)
    for pattern in DISALLOWED_PATTERNS:
        if re.search(pattern, n):
            return True
    return False


def classify_license(license_str: str) -> str:
    """Classify a license string into a category."""
    n = normalize(license_str)

    if re.search(r"mit", n):
        return "MIT"
    if re.search(r"bsd", n):
        return "BSD"
    if re.search(r"apache", n):
        return "Apache-2.0"
    if re.search(r"isc", n):
        return "ISC"
    if re.search(r"psf|python (software|license)", n):
        return "PSF"
    if re.search(r"mpl|mozilla", n):
        return "MPL-2.0"
    if re.search(r"cc0", n):
        return "CC0-1.0"
    if re.search(r"unicode|dfs", n):
        return "Unicode-DFS-2016"
    if re.search(r"unlicense|public.department|public.dom", n):
        return "Unlicense"
    if re.search(r"zlib", n):
        return "Zlib"
    if re.search(r"gpl", n):
        return "GPL"
    if re.search(r"lgpl", n):
        return "LGPL"
    if re.search(r"agpl|affero", n):
        return "AGPL"
    if re.search(r"eupl", n):
        return "EUPL"
    if re.search(r"cpl", n):
        return "CPL"
    if re.search(r"rpl", n):
        return "RPL"
    if n == "unknown":
        return "Unknown"
    return "Other"


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: check_licenses_v2.py <licenses.csv>")
        return 1

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"ERROR: License file not found: {csv_path}")
        return 1

    packages = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name", "?").strip()
            version = row.get("Version", "?").strip()
            license_str = row.get("License", "UNKNOWN").strip()
            packages.append({
                "name": name,
                "version": version,
                "license": license_str,
                "classified_as": classify_license(license_str),
                "is_disallowed": is_disallowed(license_str),
                "is_acceptable": is_acceptable(license_str),
            })

    # Build accepted exception names set
    accepted_names = {e["name"] for e in ACCEPTED_EXCEPTIONS}

    # Apply accepted exceptions
    for p in packages:
        if p["name"] in accepted_names:
            p["is_disallowed"] = False
            p["is_acceptable"] = True

    # Build compliance report
    report = {
        "$schema": "https://helix-ids.dev/schemas/license-compliance-report-v1.json",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "check_licenses_v2.py",
        "tool": "pip-licenses + custom checker",
        "source_file": str(csv_path),
        "summary": {
            "total_packages": len(packages),
            "acceptable": sum(1 for p in packages if p["is_acceptable"]),
            "disallowed": sum(1 for p in packages if p["is_disallowed"]),
            "unrecognized": sum(1 for p in packages if not p["is_acceptable"] and not p["is_disallowed"]),
        },
        "compliance_threshold": {
            "disallowed": 0,
            "unrecognized_warning": True,
        },
        "packages": packages,
        "accepted_exceptions": ACCEPTED_EXCEPTIONS,
        "policy_reference": "docs/compliance/LICENSE_POLICY.md",
    }

    # Write machine-readable report
    report_dir = csv_path.parent
    report_path = report_dir / "compliance-report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Compliance report written: {report_path}")

    # Human-readable output
    disallowed = [p for p in packages if p["is_disallowed"]]
    unrecognized = [p for p in packages if not p["is_acceptable"] and not p["is_disallowed"]]

    print(f"\n{'='*60}")
    print("LICENSE COMPLIANCE REPORT")
    print(f"{'='*60}")
    print(f"Total packages:  {report['summary']['total_packages']}")
    print(f"Acceptable:      {report['summary']['acceptable']}")
    print(f"Disallowed:      {report['summary']['disallowed']}")
    print(f"Unrecognized:    {report['summary']['unrecognized']}")
    print(f"{'='*60}\n")

    exit_code = 0

    if disallowed:
        print("::error title=Disallowed licenses::The following packages violate policy:\n")
        print(f"{'Package':<45} {'Version':<15} {'License':<25} {'Category'}")
        print("-" * 100)
        for p in disallowed:
            print(f"{p['name']:<45} {p['version']:<15} {p['license']:<25} {p['classified_as']}")
            print(f"::error ::{p['name']} {p['version']} — {p['license']} (DISALLOWED: {p['classified_as']})")
        print()
        exit_code = 1
    else:
        print("DISALLOWED LICENSES: 0 violations ✓")

    if unrecognized:
        print("\n::warning title=Unrecognized licenses::The following packages need manual review:\n")
        print(f"{'Package':<45} {'Version':<15} {'License':<25} {'Category'}")
        print("-" * 100)
        for p in unrecognized:
            print(f"{p['name']:<45} {p['version']:<15} {p['license']:<25} {p['classified_as']}")
            print(f"::warning ::{p['name']} {p['version']} — {p['license']} (unrecognized)")
        print()
    else:
        print("UNRECOGNIZED LICENSES: 0 ✓")

    # Print summary by license category
    print(f"\n{'License Distribution':-^50}")
    categories: dict[str, int] = {}
    for p in packages:
        cat = p["classified_as"]
        categories[cat] = categories.get(cat, 0) + 1
    for cat in sorted(categories, key=lambda c: categories[c], reverse=True):
        bar = "█" * max(1, categories[cat] // 2)
        print(f"  {cat:<18} {categories[cat]:>4} {bar}")

    if exit_code == 0:
        print(f"\n{'='*60}")
        print("LICENSE COMPLIANCE: PASSED ✓")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("LICENSE COMPLIANCE: FAILED — disallowed licenses found")
        print(f"{'='*60}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
