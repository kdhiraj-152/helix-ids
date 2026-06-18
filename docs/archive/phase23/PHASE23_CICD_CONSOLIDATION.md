# Phase 23 — CI/CD Consolidation

Date: 2026-06-18
Status: Certified ✓

## Summary

Reduced from 13 workflow files to 6 well-separated pipelines.
Removed all dead/duplicate configurations.
Lockfile regenerated with fresh package versions + correct CUDA hashes.

## Final Workflow Layout

```
.github/workflows/
├── ci.yml                 # Fast path (<6 min)
├── quality.yml            # Medium path (~10–15 min)
├── architecture.yml       # Governance layer (~3 min)
├── dependency-review.yml  # Pre-CI gate (~1 min)
├── release.yml            # Release pipeline (verify → sign → publish)
└── nightly.yml            # Weekly long-running
```

## Workflow Details

### ci.yml — Fast Path (<6 min)

| Step | Tool | Expected |
|------|------|----------|
| 1 | ruff check | 0 violations |
| 2 | mypy (src) | 0 errors |
| 3 | pytest (fast subset) | All passing |

**Triggers:** push → dev, pull_request → main, dev
**Scope excludes:** architecture, training, operations, test_data

### quality.yml — Medium Path (~10–15 min)

| Step | Tool | Expected |
|------|------|----------|
| 1 | Coverage gate | ≥65% |
| 2 | Benchmark regression | No regression |
| 3 | Dependency audit | No reverse deps, no self-imports |
| 4 | License compliance | Policy-compliant |

**Triggers:** push → dev, pull_request → main

### architecture.yml — Governance Layer (~3 min)

| Step | Tool | Expected |
|------|------|----------|
| 1 | Trainer size limit | ≤2000 LOC |
| 2 | Reverse deps check | 0 src→scripts imports |
| 3 | Cycle detection | 0 cycles |
| 4 | Architecture pytest | All passing |

**Triggers:** push → dev, pull_request → main
**Path filters:** src/**, scripts/**, tests/architecture/**

### dependency-review.yml — Pre-CI Gate (~1 min)

| Check | What it catches |
|-------|-----------------|
| Vulnerable packages | Known CVEs |
| License changes | SPDX policy violations |
| Dependency explosion | Unbounded transitive deps |

**Trigger:** pull_request (any branch)

### release.yml — Release Pipeline

| Phase | Steps |
|-------|-------|
| verify | Lockfile sync, SBOM, coverage, ruff, mypy, pip-audit, bandit, checksums, SLSA, license, trust report |
| sign | SBOM, license inventory, SLSA, signing (Cosign keyless), container build/push/sign |

**Trigger:** tags v*, workflow_dispatch

### nightly.yml — Weekly Long-Running

| Job | Scope |
|-----|-------|
| codeql | Python SAST (security-and-quality queries) |
| cross_python | pytest + coverage on 3.9, 3.10, 3.11 |
| mutation | Cosmic-ray on 9 modules |

**Trigger:** schedule (Mon 06:00 UTC), workflow_dispatch

## Consolidation Map

| Removed Workflow | Absorbed Into | Rationale |
|------------------|----------------|-----------|
| codeql.yml | nightly.yml | Now weekly-only |
| performance-regression.yml | quality.yml | Part of quality gate |
| release-integrity.yml | release.yml | Merged into single pipeline |
| runtime-monitoring-hardening.yml | — | Superseded by quality + architecture |
| sign-release.yml | release.yml | Merged into single pipeline |
| test-reliability.yml | nightly.yml | Now weekly-only |

## Key Fixes Applied

| Issue | Fix | Commit |
|-------|-----|--------|
| Lockfile had stale package versions | Regenerated with `uv pip compile --refresh --python-platform=linux` | 6 |
| Release pipeline used `pip-compile` (platform-dependent) | Changed to `uv pip compile --python-platform=linux` | 6 |
| Lockfile check failed on header comment diff | Made diff header-agnostic (`sed '1,5d'`) | 6 |

## Trigger Summary

| Workflow | push | pull_request | schedule | workflow_dispatch | tag v* |
|----------|------|--------------|----------|-------------------|--------|
| ci | dev | main, dev | — | — | — |
| quality | dev | main | — | — | — |
| architecture | dev | main | — | — | — |
| dependency-review | — | all | — | — | — |
| release | — | — | — | ✓ | ✓ |
| nightly | — | — | Mon 06:00 | ✓ | — |

## Verification

- [x] All 6 commits cleanly applied (main: 328a83e → 64365c9)
- [x] 6 obsolete workflows removed
- [x] Zero dead workflow configurations
- [x] All branch triggers validated
- [x] Path filters prevent unnecessary runs
- [x] Lockfile synchronized (uv check passes)
- [x] All YAML files lint-valid
