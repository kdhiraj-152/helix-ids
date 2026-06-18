# Phase 23 — Delete Candidates

> This document lists files that are safe to delete per the Dead File Audit.
> No deletions have been executed. This is a recommendation only.

---

## Immediate Delete Candidates (SAFE_DELETE)

These files have no runtime or documentation dependencies:

```bash
# One-time phase helpers
rm scripts/training/_rewrite_main.py
rm scripts/smoke_save_checkpoint.py

# Superseded docs (no traceability value)
rm docs/architecture/DEAD_CODE_AUDIT.md
rm docs/architecture/DEPENDENCY_LOCKDOWN.md
rm docs/architecture/PERFORMANCE_BASELINE.md
rm docs/reproducibility/REPRODUCIBILITY_GAP.md
```

## Archive Candidates (move to docs/archive/)

These historical phase docs should be moved rather than deleted:

```bash
mkdir -p docs/archive/phase13
mkdir -p docs/archive/phase19
mkdir -p docs/archive/phase22
mkdir -p docs/archive/phase11a
mkdir -p docs/archive/phase4
mkdir -p docs/archive/phase23

mv docs/architecture/PHASE13B_AUDIT.md docs/archive/phase13/
mv docs/architecture/PHASE19_ARCHITECTURE_FREEZE.md docs/archive/phase19/
mv docs/architecture/PHASE22_RELIABILITY_PLAN.md docs/archive/phase22/
mv docs/development/PHASE_11A_CLEANUP_REPORT.md docs/archive/phase11a/
mv docs/governance/PHASE_4A_GOVERNANCE_COVERAGE_AUDIT.md docs/archive/phase4/
mv docs/governance/PHASE_4B_ASSUMPTION_ELIMINATION.md docs/archive/phase4/
mv docs/operations/PHASE23_CICD_CONSOLIDATION.md docs/archive/phase23/
```

## Rename Candidates (covered in NAMING_STANDARDIZATION.md)

These files are not dead but have non-standard names:

| Current Name | Issue |
|--------------|-------|
| `scripts/training/train_multidataset_v2_fixed.py` | `v2_fixed` suffix — historical artifact |
| `scripts/evaluation/benchmark_e2e_v2_fixed.py` | `v2_fixed` suffix |
| `scripts/evaluation/holdout_evaluation_v2.py` | `_v2` suffix |
| `scripts/training/adversarial_training_v2.py` | `_v2` suffix |
| `scripts/training/train_unsw_only_cleaned.py` | `_cleaned` suffix |
| `results/v2_fixed/` directory | Historical naming |

## Execution Order

1. Run `git rm` for SAFE_DELETE files
2. Run `git mv` for ARCHIVE files
3. Execute renames (if approved) per NAMING_STANDARDIZATION.md
4. Run full test suite to confirm no breakage
5. Commit in order: deletes → archives → renames → post-cleanup
