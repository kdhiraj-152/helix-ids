# Phase 23 — Documentation Rationalization

> Generated: 2026-06-18

---

## Rationalization Actions Completed

### 1. Archive Created for Historical Phase Docs

Created `docs/archive/` with per-phase subdirectories:

| Archive | Files Moved | Original Location |
|---------|-------------|-------------------|
| `docs/archive/phase4/` | `PHASE_4A_GOVERNANCE_COVERAGE_AUDIT.md`, `PHASE_4B_ASSUMPTION_ELIMINATION.md` | `docs/governance/` |
| `docs/archive/phase11a/` | `PHASE_11A_CLEANUP_REPORT.md` | `docs/development/` |
| `docs/archive/phase13/` | `PHASE13B_AUDIT.md` | `docs/architecture/` |
| `docs/archive/phase19/` | `PHASE19_ARCHITECTURE_FREEZE.md` | `docs/architecture/` |
| `docs/archive/phase22/` | `PHASE22_RELIABILITY_PLAN.md` | `docs/architecture/` |
| `docs/archive/phase23/` | `PHASE23_CICD_CONSOLIDATION.md` | `docs/operations/` |
| `docs/archive/superseded/` | `DEAD_CODE_AUDIT.md`, `DEPENDENCY_LOCKDOWN.md`, `PERFORMANCE_BASELINE.md`, `REPRODUCIBILITY_GAP.md` (×2) | Various |

### 2. `docs/README.md` Rewritten

- Replaced stale layout (referenced non-existent `docs/archives/`, `docs/fig_revamp/`, `HASH_AUTHORITY.md` → `hash_authority.md`, phantom `CHECKPOINT_AUDIT.md`, etc.)
- Now matches actual directory tree
- Archive section points to real archive directories

---

## Active vs Archived Documentation

### Active Documentation (Stays in Place)

| Directory | Purpose | Count |
|-----------|---------|-------|
| `docs/architecture/` | System architecture, schemas, design docs | 18 docs |
| `docs/compliance/` | License policy, supply chain | 2 docs |
| `docs/development/` | Project status | 1 doc |
| `docs/governance/` | ADRs, schema governance, hash authority | 8 docs |
| `docs/manuscript/` | Paper drafts | 2 docs |
| `docs/operations/` | Runbooks, branch governance, certification | 5 docs |
| `docs/releases/` | RC readiness and certification | 3 docs |
| `docs/reports/` | Benchmark protocols, dataset reports | 4 docs |
| `docs/reproducibility/` | Build guides, container reproduction | 3 docs |
| `docs/security/` | Security posture, review | 2 docs |
| `docs/audits/` | Phase 23 audit deliverables | 9 docs |
| **Total active** | | **~57 docs** |

### Archived Documentation

| Archive | Count | Traceability |
|---------|-------|-------------|
| `docs/archive/phase4/` | 2 | Phase 4 governance audits (historical) |
| `docs/archive/phase11a/` | 1 | Phase 11A cleanup report |
| `docs/archive/phase13/` | 1 | Phase 13B architecture audit |
| `docs/archive/phase19/` | 1 | Phase 19 architecture freeze |
| `docs/archive/phase22/` | 1 | Phase 22 reliability plan |
| `docs/archive/phase23/` | 1 | Phase 23 CI/CD consolidation |
| `docs/archive/superseded/` | 5 | Superseded docs (dead code, dependency, etc.) |
| **Total archived** | | **12 docs** |

---

## Remaining Issues

| Issue | Status | Action |
|-------|--------|--------|
| `docs/figures/fig*.png` (5.9 MB) committed | Open | Consider: (a) remove from git and regenerate, (b) LFS, (c) keep as-is |
| `docs/architecture/FINAL_METRICS.md` | Possibly outdated | Review if still accurate |
| `docs/architecture/RC3_READINESS_VERDICT.md` | Pre-Phase-23 | May need update after Phase 23 complete |
| `docs/releases/RC1_READINESS.md` | Superseded by RC2 docs | Could archive (post-review) |
| `docs/reproducibility/REPRODUCIBILITY.md` | Duplicate of REPRODUCIBLE_BUILD_GUIDE.md? | Review for consolidation |
