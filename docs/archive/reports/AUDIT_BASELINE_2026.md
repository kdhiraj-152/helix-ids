# Audit Baseline — 2026

> Last updated: 2026-06-18  
> Consolidated findings from the Phase 23 audit cycle and earlier audits.

## Repository Statistics

| Metric | Value |
|--------|-------|
| Git-tracked files | ~241 |
| Python source files | ~158 tracked |
| Documentation files | ~47 `.md` files tracked (pre-consolidation) |
| Total test functions | ~2,500+ collected |
| CI workflow files | 6 active |

## Quality Gates

| Gate | Status | Threshold |
|------|--------|-----------|
| Coverage | ≥70% | ≥65% |
| Mutation score | 100% killed (15 modules) | ≥90% |
| Ruff lint | Clean | 0 violations |
| Mypy | Pass | 0 errors |
| Bandit | Pass | Clean |

## Test Health

| Metric | Value |
|--------|-------|
| Total tests | ~2,500+ |
| CI passing rate | ≥95% |
| Architecture tests | 6 files, 34 tests |
| Mutation coverage | 8,479 mutants, 100% killed |

## Reproducibility

| Guarantee | Status |
|-----------|--------|
| pip-compile lockfile | `requirements-lock.txt` (hashed, pinned) |
| SLSA provenance | Active in release workflow |
| SBOM generation | CycloneDX JSON |
| Docker build | Digest-pinned base |
| Deterministic training | Verified via cosmic-ray determinism config |

## Architecture Freeze

| Metric | Value | Status |
|--------|-------|--------|
| Dependency graph nodes | 256 | FROZEN |
| Dependency graph edges | 590 | FROZEN |
| Package-level cycles | 0 | PASS |
| Reverse deps (src → scripts) | 0 | PASS |
| Forbidden imports (src. prefix) | 0 | PASS |
| HelixFullTrainer LOC | ≤2,000 | PASS |
| HelixFullTrainer methods | ≤100 | PASS |

## Deployment Certification

| Check | Status |
|-------|--------|
| Lockfile dependency fix | ✅ |
| Workflow consolidation (13→6) | ✅ |
| Branch protection (main) | ✅ |
| Branch protection (dev) | ✅ |
| Stale branch cleanup | ✅ |
| SBOM generation | ✅ |
| SBOM attestation | ✅ |
| SLSA provenance | ✅ |

## Technical Debt Register (Key Items)

| ID | Issue | Severity | Phase |
|----|-------|----------|-------|
| TDR-001 | ENGINEERED_FEATURE_NAMES duplication | MEDIUM | Resolved |
| TDR-002 | Partial delegation anti-pattern (17 wrappers) | HIGH | Phase 21 |
| TDR-003 | Loss logic inline (9 functions) | MEDIUM | Phase 22 |
| TDR-004 | Config version mismatch (41 vs 17) | MEDIUM | Phase 22 |
| TDR-005 | 11 unused config fields | LOW | Phase 22A |
| TDR-006 | schema_registry.yaml orphaned | MEDIUM | Phase 22C |
| TDR-007 | Pre-commit hooks missing | LOW | Phase 23 |

## Security Posture

| Control | Status |
|---------|--------|
| Ruff lint | Active (blocking CI) |
| Mypy types | Active (blocking CI) |
| Coverage ≥65% | Active (blocking CI) |
| pip-audit | Active (strict mode) |
| Dependabot | Active (weekly) |
| Dependency review | Active (PR gate) |
| CodeQL | Active (PR + scheduled) |
| Bandit | Active (blocking CI) |
| SLSA provenance | Active (release) |
| SBOM generation | Active (release) |

## Known Gaps

- **Coverage**: Pre-existing gap from modified source files (metrics.py, loss.py, provenance.py)
- **Testing**: No performance regression tests (Phase 21 finding — resolved in C1)
- **Config**: `helix_config.yaml` uses `input_features: 41` vs canonical 17
- **Cycles**: Zero cross-boundary cycles; single internal cycle (architecture-frozen)
