# Release Process

> Last updated: 2026-06-18

## Release Pipeline Overview

The release pipeline produces signed, attested artifacts with full provenance chain.

1. Code merged to `main` via squash-merge PR from `dev`
2. Release workflow triggered by tag push (`v*.*.*`)
3. CI validates: lint, types, tests, coverage, architecture, security
4. SBOM generated via CycloneDX (reproducible JSON)
5. SBOM attested via in-toto v1 statement
6. SLSA provenance generated (`generate_slsa_provenance.py`)
7. Docker image built with digest-pinned base
8. Release artifact uploaded with provenance verification

## CI Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | push/PR to main, dev | Lint, types, tests, coverage |
| `quality.yml` | push/PR to main | Ruff, mypy, bandit, pip-audit |
| `architecture.yml` | push/PR to main | Architecture boundary tests |
| `dependency-review.yml` | PR to main | Dependency vulnerability check |
| `release.yml` | tag v*.*.* | Build, sign, attest, upload |
| `nightly.yml` | Schedule (daily) | Fuzz, chaos, performance regression |

## Versioning

- **Format**: `v<major>.<minor>.<patch>` (semantic)
- **Git tags**: Pushed to trigger release workflow
- **SBOM version**: Aligned with git tag

## Release Integrity

| Check | Method |
|-------|--------|
| Lockfile sync | `uv pip compile` comparison |
| SBOM generation | `cyclonedx-py` reproducible JSON |
| SBOM attestation | in-toto v1 statement |
| SLSA provenance | Generate + verify scripts |
| Container base | Digest-pinned (`sha256:...`) |
| License compliance | CI gate via `check_licenses.py` |

## Branch Governance

| Branch | Protection | Merge Path |
|--------|-----------|------------|
| `main` | Full protection: PR required, status checks, 1 approval, squash-merge, no force-push | dev → main via PR |
| `dev` | Light protection: block deletion, allow force-push | Feature branches → dev |

### Main Protection Details
- `required_status_checks`: strict=true, contexts=[CI, Quality, Architecture]
- `required_pull_request_reviews`: 1 approval, dismiss_stale_reviews=true
- `allow_force_pushes`: false
- `allow_deletions`: false
- `enforce_admins`: false (admin bypass)
- `squash_merge`: true; `merge_commit` + `rebase`: false

## Promotion Gate

Deployable checkpoints require:
1. Multi-seed consensus (3 seeds minimum)
2. Variance threshold validation
3. Provenance chain verification
4. Staging gate check pass (override_rate <= 0.02, degraded_state == 0)
5. Traffic expansion guard validation

**Single-seed runs** produce research-only artifacts — not eligible for deployment.
