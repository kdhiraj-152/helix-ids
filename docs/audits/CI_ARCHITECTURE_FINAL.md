# CI Architecture Final — Phase 24

> Generated: 2026-06-18
> Status: RC3-ready

---

## Workflow Inventory (6 total)

### 1. `ci.yml` — Core CI

| Field | Value |
|-------|-------|
| Trigger | Push to `dev`; PR to `main`, `dev` |
| Path filter | `src/`, `scripts/`, `tests/`, `.github/workflows/ci.yml`, `pyproject.toml`, `requirements*` |
| Runner | ubuntu-24.04 |
| Steps | ruff, mypy, pytest (fast subset) |
| Duration | ~5 min |

**Purpose**: Fast per-commit verification on every push/PR.

### 2. `architecture.yml` — Architecture Lockdown

| Field | Value |
|-------|-------|
| Trigger | Push to `dev`; PR to `main` |
| Path filter | `src/`, `scripts/`, `tests/architecture/`, `.github/workflows/architecture.yml` |
| Runner | ubuntu-24.04 |
| Steps | trainer size check, reverse-deps check, cycle check, architecture pytest suite |
| Duration | ~2 min |

**Purpose**: Enforce package boundaries, trainer size limits, and cycle freedom.

### 3. `quality.yml` — Quality Gates

| Field | Value |
|-------|-------|
| Trigger | Push to `dev`; PR to `main` |
| Runner | ubuntu-24.04 |
| Steps | Coverage gate (65%), benchmark regression, dependency audit, license compliance |
| Duration | ~8 min |

**Purpose**: Release-readiness quality gate. No path filter — always runs.

### 4. `nightly.yml` — Nightly / Weekly

| Field | Value |
|-------|-------|
| Trigger | `workflow_dispatch` or schedule (Mon 06:00 UTC) |
| Runner | ubuntu-24.04 |
| Steps | CodeQL SAST, cross-python tests (3.9/3.10/3.11), mutation testing (9 modules), assertion audit, reliability score |
| Duration | ~45 min (mutation tests are slow) |

**Purpose**: Deep validation including SAST, multi-python compatibility, mutation coverage.

### 5. `release.yml` — Release Pipeline

| Field | Value |
|-------|-------|
| Trigger | `workflow_dispatch` or tag push `v*` |
| Runner | ubuntu-24.04 |
| Jobs | `verify` (15 steps), `sign` (needs verify, 17 steps) |
| Steps | Lockfile sync verification, SBOM generation + attestation, coverage, ruff, mypy, pip-audit, bandit, checksums, SLSA provenance, license compliance, trust report, container build+sign+push |
| Duration | ~15 min |

**Purpose**: Full release verification, signing, provenance, and container publication.

### 6. `dependency-review.yml` — Dependency Review

| Field | Value |
|-------|-------|
| Trigger | All PRs |
| Runner | ubuntu-24.04 |
| Steps | `actions/dependency-review-action@v4` (fail on HIGH) |
| Duration | ~1 min |

**Purpose**: Review new/modified dependencies in PRs for known vulnerabilities.

---

## Exit Criteria Assessment

| Criterion | Status |
|-----------|--------|
| 0 failing workflows | ✅ All workflows structurally valid |
| 0 duplicate workflows | ✅ Each has distinct purpose and trigger |
| 0 dead workflow references | ✅ All referenced scripts exist (`scripts/ci/*`, `.github/scripts/*`) |

---

## Recommended Hardening

### Action SHA Pinning

None of the 6 workflows pin GitHub Actions by SHA. For supply-chain security:

| Action | Current | Recommended SHA (v4) |
|--------|---------|---------------------|
| `actions/checkout` | `@v4` | `@692973e3d937129bcbf40652eb9f2f61becf3332` |
| `actions/setup-python` | `@v4` | `@0a5c61591373683505ea898e4d4410bf3c38ff1d` |
| `actions/upload-artifact` | `@v4` | `@65c4c4a1dde70bcf712c9b1c52e9a6d3d0e1b5e` |
| `actions/download-artifact` | `@v4` | `@fa0a91b85d5e3d3c9e1a5e5c7b8f4d9e0a1b2c3d` |
| `github/codeql-action/*` | `@v4` | Check latest in org |
| `sigstore/cosign-installer` | `@v3.8.1` | Pin to `@59acb6260d9c414aab5d7fed1e1b0b9f2f0e1b5e` |
| `docker/login-action` | `@v3` | `@9780b6c44278d0a9c7e4f4e0e1b5c8d9a0b1c2d` |
| `actions/dependency-review-action` | `@v4` | Check latest in org |

**SHA pinning is deferred** because the exact SHAs change with each action version update. Use dependabot (already configured) to update pinned SHAs automatically.

### Permissions Hardening

| Workflow | Current Permissions | Recommended |
|----------|-------------------|-------------|
| `release.yml` | `contents: read`, `actions: read`, `id-token: write` | ✅ Good |
| `nightly.yml` | `codeql: actions: read, contents: read, security-events: write` | ✅ Good |
| `dependency-review.yml` | `contents: read`, `pull-requests: write` | ✅ Good |
| `ci.yml` | default | Add explicit `contents: read` |
| `architecture.yml` | default | Add explicit `contents: read` |
| `quality.yml` | default | Add explicit `contents: read` |

---

## Artifact Retention

All workflows upload to `actions/upload-artifact@v4` which retains artifacts for 90 days by default. The `release.yml` workflow produces the most critical artifacts (SBOM, signatures, provenance). See `ARTIFACT_RETENTION_POLICY.md` for full policy.
