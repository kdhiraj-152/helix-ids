# Reproducible Build Guide

**Document:** REPRODUCIBLE_BUILD_GUIDE.md
**Version:** 1.0
**Last Updated:** 2026-06-10
**Phase:** 8B — Reproducible Builds, SBOM & Release Integrity

This document describes how to perform a clean, reproducible build of the
HELIX-IDS project and verify its integrity.

---

## 1. Prerequisites

| Requirement | Minimum Version | Notes |
|-------------|-----------------|-------|
| Python | 3.11 | Must match CI runner (ubuntu-24.04) |
| pip | 24.0+ | Bundled with Python 3.11 |
| Git | 2.40+ | For commit verification |
| Operating System | Linux (amd64) or macOS | Windows not officially supported |

---

## 2. Clean Rebuild

### 2.1 Clone the Repository

```bash
git clone <repository-url> helix-ids
cd helix-ids
git checkout <tag-or-commit>
```

### 2.2 Verify the Lockfile

Before proceeding, confirm the lockfile is synchronized with the source
requirements:

```bash
# Install pip-tools
pip install piptools

# Recompile from source and compare
pip-compile --quiet --output-file=/tmp/lockfile-check.txt requirements.in
diff -q requirements-lock.txt /tmp/lockfile-check.txt && echo "LOCKFILE: synchronized"
```

If the lockfile is stale (`diff` reports differences), regenerate it:

```bash
pip-compile --generate-hashes --output-file=requirements-lock.txt requirements.in
```

### 2.3 Create a Virtual Environment

```bash
# Create a fresh venv
python3 -m venv .venv
source .venv/bin/activate

# Verify clean state
pip list --format=columns
```

### 2.4 Install Dependencies from Lockfile

```bash
# Install exactly the pinned versions with hash verification
pip install -r requirements-lock.txt
```

This is the **critical reproducibility step**: the lockfile pins exact versions
with SHA-256 hashes for every transitive dependency. Installing from the
lockfile guarantees the same dependency tree on every machine.

### 2.5 Build and Test

```bash
# Set project root for imports
export PYTHONPATH=src

# Run all tests with coverage gate
python -m pytest -q --cov-fail-under=55
```

Expected output:
- **677+ passed**
- **0 failed**
- **Coverage >= 55%**

---

## 3. Verification Steps

After a clean rebuild, verify all quality and security gates pass.

### 3.1 Ruff (Linter)

```bash
ruff check src scripts tests
```

Expected: **0 violations**

### 3.2 Mypy (Type Checker)

```bash
mypy src
```

Expected: **0 errors**

### 3.3 Bandit (Security Linter)

```bash
pip install bandit
bandit -r src --severity-level high
```

Expected: **0 HIGH-severity findings**

### 3.4 pip-audit (Dependency Vulnerability Scan)

```bash
pip install pip-audit
pip-audit --strict --desc on
```

Expected: **0 unknown vulnerabilities**. Only documented accepted
vulnerabilities (see `docs/security/SECURITY_POSTURE.md`) may appear.

### 3.5 Pytest (Full Test Suite)

```bash
python -m pytest -q
```

Expected: **677+ passed, 0 failed**

### 3.6 SBOM Generation

```bash
pip install cyclonedx-bom
cyclonedx-py requirements \
  requirements-lock.txt \
  --output-reproducible \
  --output-file sbom.json \
  --output-format JSON
```

Expected: SBOM generated successfully. Validate with:

```bash
cyclonedx-py requirements \
  requirements-lock.txt \
  --output-reproducible \
  --output-file /dev/null \
  --output-format JSON \
  --validate
```

---

## 4. Artifact Verification

### 4.1 Checksum Verification

Verify that key artifacts match their expected checksums:

```bash
sha256sum -c checksums.sha256
```

The `checksums.sha256` file is generated during the CI build and contains
SHA-256 hashes for:

- `requirements-lock.txt`
- `sbom.json`
- `coverage.xml`

When verifying locally, obtain the authoritative `checksums.sha256` from the
most recent CI build artifact (named `sbom`).

### 4.2 SBOM Verification

The SBOM can be validated programmatically:

```bash
# Structural validation is performed by cyclonedx-py --validate
# For deeper verification, use a CycloneDX validator:
pip install cyclonedx-bom
cyclonedx-py requirements \
  requirements-lock.txt \
  --output-reproducible \
  --output-file /dev/null \
  --validate
```

The SBOM (`sbom.json`) contains:
- Every direct and transitive dependency with exact version
- SHA-256 hashes for each package distribution
- PURL identifiers for each component
- Dependency graph relationships

---

## 5. CI Artifacts

The CI pipeline generates and retains the following reproducibility artifacts:

| Artifact | Job | Retention | Description |
|----------|-----|-----------|-------------|
| `sbom.json` | `sbom_generation` | 30 days | CycloneDX SBOM (JSON) |
| `sbom.sha256` | `sbom_generation` | 30 days | SBOM checksum |
| `checksums.sha256` | `sbom_generation` | 30 days | Aggregate checksums |
| `pip-audit-report.txt` | `dependency_vulnerability_scan` | 90 days | Vulnerability scan |
| `dependency-freshness.md` | `dependency_freshness` | 90 days | Outdated package report |
| `checks-junit-report.xml` | `checks` | 90 days | Test results XML |
| `release-integrity-report` | `release-integrity` | 90 days | Release verification report |

---

## 6. Release Verification Workflow

For tagged releases (`v*`), a dedicated **release-integrity** workflow runs
automatically:

```yaml
.on:
  push:
    tags:
      - "v*"
```

This workflow verifies:
1. Lockfile is synchronized with `requirements.in`
2. SBOM is generated and valid
3. Coverage threshold (>=55%) is met
4. Ruff passes (0 violations)
5. Mypy passes (0 errors)
6. pip-audit passes (0 unknown vulnerabilities)
7. Artifact checksums are generated

The workflow **does not** publish to PyPI, create GitHub Releases, or deploy
any code. It is a pure verification gate.

---

## 7. Deterministic Environment

All CI workflows now pin the runner image:

```yaml
runs-on: ubuntu-24.04
```

This prevents silent runner migrations that could introduce subtle
reproducibility differences. Ubuntu 24.04 provides:

- Python 3.11 by default
- glibc 2.39
- OpenSSL 3.0.x
- Consistent system package versions

---

## 8. Scorecard

| Metric | Current (Phase 8B) | Target |
|--------|-------------------|--------|
| Build Reproducibility | 70+/100 | >=70/100 |
| Supply-Chain Security | 75+/100 | >=75/100 |
| CI Maturity | 98+/100 | >=98/100 |
| Lockfile Coverage | 100% | 100% |
| SBOM Generation | Active | Active |
| Runner Determinism | ubuntu-24.04 across all workflows | Fixed runner image |
| Artifact Checksums | Generated per-run | Available |
| Release Verification | Dedicated workflow | Active |

---

## 9. Troubleshooting

### 9.1 Lockfile Mismatch

```
ERROR: Lockfile is stale.
```

The lockfile is out of sync with `requirements.in`. Regenerate:

```bash
pip-compile --generate-hashes --output-file=requirements-lock.txt requirements.in
```

### 9.2 SBOM Generation Fails

Ensure `cyclonedx-bom` is installed and the lockfile is valid:

```bash
pip install cyclonedx-bom
cyclonedx-py requirements requirements-lock.txt --output-file /dev/null
```

### 9.3 Test Failures in Clean Build

Verify the Python version matches:

```bash
python3 --version  # Should show 3.11.x
```

Ensure the virtual environment is activated and dependencies are installed:

```bash
source .venv/bin/activate
pip install -r requirements-lock.txt
export PYTHONPATH=src
```

---

*This document is maintained as part of the Phase 8B reproducibility
deliverable. Update on any change to the build process or verification
procedures.*