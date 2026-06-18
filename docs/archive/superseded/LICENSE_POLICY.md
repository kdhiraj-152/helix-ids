# License Compliance Policy

**Document:** LICENSE_POLICY.md
**Version:** 2.0
**Last Updated:** 2026-06-12
**Phase:** 10B — Artifact Provenance, Container Trust & Compliance Hardening

This document defines the license compliance policy for the HELIX-IDS project.
All direct and transitive dependencies MUST comply with this policy.

---

## 1. Allowlisted Licenses

The following licenses are explicitly permitted and may be used by any
dependency without further review:

| License | SPDX Identifier |
|---------|-----------------|
| MIT | MIT |
| BSD 2-Clause | BSD-2-Clause |
| BSD 3-Clause | BSD-3-Clause |
| Apache 2.0 | Apache-2.0 |
| ISC | ISC |
| Python Software Foundation License | PSF |
| Unicode License | Unicode-DFS-2016 |
| Mozilla Public License 2.0 | MPL-2.0 |
| CC0 1.0 Universal | CC0-1.0 |
| Unlicense | Unlicense |
| Zlib | Zlib |

## 2. Disallowed Licenses

The following licenses are **explicitly disallowed** in any direct or transitive
dependency. If a dependency is detected using one of these licenses, the CI
workflow MUST fail:

| License | SPDX Identifier | Reason |
|---------|-----------------|--------|
| GNU General Public License v2+ | GPL-2.0-only, GPL-2.0-or-later, GPL-3.0-only, GPL-3.0-or-later | Copyleft — may impose source distribution obligations |
| GNU Affero General Public License v3+ | AGPL-3.0-only, AGPL-3.0-or-later | Network copyleft — broader source distribution requirements |
| GNU Lesser General Public License v2.1+ | LGPL-2.1-only, LGPL-2.1-or-later, LGPL-3.0-only, LGPL-3.0-or-later | Weak copyleft — restricted linking |
| European Union Public License | EUPL-1.1, EUPL-1.2 | Copyleft by EU law |
| Common Public License | CPL-1.0 | Copyleft |
| Reciprocal Public License | RPL-1.5 | Strong copyleft |

## 3. Conditional (Review Required)

Licenses not listed in either allowlist or disallowed list require manual review
before they may be used. The review MUST be documented in this file with a
rationale and date.

## 4. CI Enforcement

License compliance is enforced by an automated CI step using `pip-licenses`:

```bash
# Generate machine-readable JSON for the inventory
pip-licenses --format=json --output-file=results/licenses/licenses.json

# Generate CSV for the compliance checker
pip-licenses --format=csv --output-file=results/licenses/licenses.csv

# Run the expanded compliance checker (Phase 10B)
python3 scripts/ci/check_licenses_v2.py results/licenses/licenses.csv
```

### Expanded Checker Features (Phase 10B)

The v2 checker (`check_licenses_v2.py`) provides:

1. **Machine-readable report**: `results/licenses/compliance-report.json`
   - Full classification per package (MIT, BSD, Apache, GPL, etc.)
   - Disallowed/unrecognized tracking
   - License distribution histogram
   - Schema-versioned report format

2. **Enhanced classification**:
   - Standard categories: MIT, BSD, Apache-2.0, ISC, PSF, MPL-2.0, CC0-1.0, GPL, LGPL, AGPL, etc.
   - Each package classified and flagged as acceptable/disallowed/unrecognized

### Workflow

1. On every CI run, `pip-licenses` scans all installed packages.
2. The custom checker script flags disallowed licenses.
3. The workflow fails if any disallowed license is detected.
4. License inventory and compliance report are uploaded as CI artifacts.

## 5. Review Process

- **New dependency PRs:** License is checked automatically by the CI gate.
- **Quarterly review:** Full license inventory audited against this policy.
- **Policy updates:** Requires maintainer approval and must be documented here.

## 6. Current Inventory

The current license inventory is generated each CI run and stored at:
- `results/licenses/licenses.json` (machine-readable)
- `results/licenses/licenses.csv` (human-readable)
- `results/licenses/compliance-report.json` (machine-readable report)

Refer to these files for the complete dependency-to-license mapping.

## 7. Accepted Exceptions

The following transitive dependencies carry licenses that require documented
acceptance. These are reviewed quarterly.

*No exceptions currently recorded.*

## 8. Remediation Plan for Copyleft Findings

If a GPL/AGPL/LGPL dependency is found:

1. **Immediate**: The CI gate blocks the workflow. The package name, version, and license are logged.
2. **Analysis**: Determine if the dependency is direct or transitive, and whether it can be replaced.
3. **Remediation options** (in priority order):
   a. Replace with a permissively-licensed alternative
   b. Pin to a specific version that uses a different license
   c. Exclude the transitive dependency via `pip install --no-deps` for the offending package
   d. Document an exception with maintainer approval if replacement is impossible
4. **Escalation**: If remediation is not possible within 30 days, escalate to project maintainers.
