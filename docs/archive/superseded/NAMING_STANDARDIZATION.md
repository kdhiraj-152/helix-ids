# Phase 23 — Naming Standardization Audit

> Generated: 2026-06-18
> Targets: `scripts/`, `tests/`, `docs/`, `config/`, `benchmarks/`, `results/`

---

## Naming Conventions (Repository Standard)

| Category | Convention | Example |
|----------|-----------|---------|
| Python files | `snake_case.py` | `feature_harmonization.py` |
| Test files | `test_<snake_case>.py` | `test_feature_harmonization.py` |
| Config files | `kebab-case.yaml` or `snake_case.yaml` | `helix_config.yaml` |
| Workflow files | `kebab-case.yml` | `ci.yml`, `dependency-review.yml` |
| Doc files | `UPPER_SNAKE_CASE.md` or `kebab-case.md` | `ARCHITECTURE.md`, `ADR-001-governance-philosophy.md` |
| Directories | `snake_case/` | `feature_harmonization/` |
| Benchmark/results | `snake_case` | `baseline.reference.json` |

---

## Violations Detected

### 1. Scripts — Version Suffixes (`_v2`, `_v2_fixed`, `_cleaned`)

These suffixes indicate historical iteration artifacts. They should be removed to reflect that these are now the primary (or only) versions.

| Current Name | Issue | Risk |
|-------------|-------|------|
| `scripts/training/train_multidataset_v2_fixed.py` | `v2_fixed` | Imported by `tests/test_data_loading.py`; referenced in 4 experiment configs as `entrypoint`? Actually configs use `benchmark_e2e_v2_fixed` |
| `scripts/evaluation/benchmark_e2e_v2_fixed.py` | `v2_fixed` | Entrypoint in all 4 experiment configs |
| `scripts/evaluation/holdout_evaluation_v2.py` | `_v2` | Referenced in reproducibility docs |
| `scripts/training/adversarial_training_v2.py` | `_v2` | Referenced in `src/helix_ids/cli.py` |
| `scripts/training/train_unsw_only_cleaned.py` | `_cleaned` | Referenced in training methodology doc |
| `scripts/training/train_unified_rebalanced.py` | OK | Clean name |

**Recommendation:** Rename these to drop the suffix (e.g. `train_multidataset.py`, `benchmark_e2e.py`, `holdout_evaluation.py`, `adversarial_training.py`, `train_unsw_only.py`). Update all importers and config references atomically.

### 2. Results Directory — `v2_fixed/`

| Path | Issue |
|------|-------|
| `results/v2_fixed/training_v2.log` | Historical naming — should be `results/benchmarks/training.log` or similar |

**Recommendation:** Archive `training_v2.log` and remove `results/v2_fixed/`.

### 3. Doc File Naming — Inconsistent Styles

| Current Name | Issue | Better Name |
|-------------|-------|------------|
| `docs/architecture/PHASE13B_AUDIT.md` | Phase prefix + all-caps | → archive: `docs/archive/phase13/ARCHITECTURE_AUDIT.md` |
| `docs/architecture/PHASE19_ARCHITECTURE_FREEZE.md` | Phase prefix | → archive |
| `docs/architecture/PHASE22_RELIABILITY_PLAN.md` | Phase prefix | → archive |
| `docs/development/PHASE_11A_CLEANUP_REPORT.md` | Phase prefix | → archive |
| `docs/governance/PHASE_4A_GOVERNANCE_COVERAGE_AUDIT.md` | Phase prefix | → archive |
| `docs/governance/PHASE_4B_ASSUMPTION_ELIMINATION.md` | Phase prefix | → archive |
| `docs/operations/PHASE23_CICD_CONSOLIDATION.md` | Phase prefix | → archive |
| `docs/operations/BRANCH_GOVERNANCE_APPLIED.md` | OK but could be clearer | `BRANCH_GOVERNANCE_ENFORCEMENT.md`? |
| `docs/operations/BRANCH_GOVERNANCE_FINAL.md` | `_FINAL` suffix | Drop `_FINAL` → `BRANCH_GOVERNANCE.md` |
| `docs/operations/RELEASE_PIPELINE_CERTIFICATION.md` | OK | — |
| `docs/operations/OPERATIONS_CERTIFICATION.md` | OK | — |
| `docs/governance/REPRODUCIBILITY_GAP.md` | Duplicate of `docs/reproducibility/REPRODUCIBILITY_GAP.md`? | Check if different |

**Recommendation:** Consolidate `_FINAL` → plain; archive phase-prefixed docs; eliminate duplicates.

### 4. Doc Duplicate Names

| Duplicate Pairs | Issue |
|-----------------|-------|
| `docs/governance/REPRODUCIBILITY_GAP.md` vs `docs/reproducibility/REPRODUCIBILITY_GAP.md` | Two files with same name in different dirs — check if different content |
| `docs/governance/hash_authority.md` vs `docs/governance/HASH_AUTHORITY.md` | Both exist! Different casing |
| `docs/reproducibility/REPRODUCIBILITY.md` vs `docs/architecture/REPRODUCIBILITY_AUDIT.md` | Related but different scope |

**Check needed:** The duplicate `hash_authority.md` (lowercase) vs `hash_authority.md` (referenced in docs/README.md with uppercase). Let me verify.

### 5. Test File Naming — Flat Root

50+ test files at `tests/` root follow `test_*.py` convention correctly, but the density indicates they should be organized:

| Pattern | Count | Example |
|---------|-------|---------|
| `tests/test_*.py` (flat) | 50+ | `test_feature_harmonization.py` |
| `tests/test_data/test_*.py` | 5 | `test_unified_loader.py` |
| `tests/test_governance/test_*.py` | 10 | `test_ast_validator.py` |
| `tests/test_training/test_*.py` | 9 | `test_execution_batch_processor.py` |
| `tests/training/test_*.py` | 8 | `test_checkpoint_recovery.py` |

**Observation:** The naming convention itself is correct (all `test_*.py`). The restructuring priority is purely about directory organization (covered in REPOSITORY_STRUCTURE.md).

### 6. Config Naming

All config files follow good conventions (`kebab-case.yaml` and `snake_case.yaml`). No issues.

### 7. Workflow Naming

All workflow YAML files use `kebab-case.yml`. No issues.

### 8. Benchmark Naming

| File | Convention | Status |
|------|-----------|--------|
| `benchmarks/baseline.json` | snake_case | OK |
| `benchmarks/baseline.reference.json` | dotted snake_case | OK |
| `benchmarks/load_test_results.json` | snake_case | OK |

No issues.

---

## Summary of Required Actions

| Priority | File/Path | Action | Impact |
|----------|-----------|--------|--------|
| HIGH | `scripts/evaluation/benchmark_e2e_v2_fixed.py` | Rename → `benchmark_e2e.py` | Update 4 config files + 1 test + CLI |
| HIGH | `scripts/evaluation/holdout_evaluation_v2.py` | Rename → `holdout_evaluation.py` | Update 2 docs |
| HIGH | `scripts/training/adversarial_training_v2.py` | Rename → `adversarial_training.py` | Update `cli.py` |
| MEDIUM | `scripts/training/train_multidataset_v2_fixed.py` | Rename → `train_multidataset.py` | Update 1 test import |
| MEDIUM | `scripts/training/train_unsw_only_cleaned.py` | Rename → `train_unsw_only.py` | Update 1 doc |
| LOW | `results/v2_fixed/` | Archive + remove |
| LOW | `docs/governance/hash_authority.md` | Deduplicate with `HASH_AUTHORITY.md` |
| LOW | `docs/governance/REPRODUCIBILITY_GAP.md` | Resolve duplicate with `docs/reproducibility/REPRODUCIBILITY_GAP.md` |

**Note:** Renames should be done as `git mv` with atomic import reference updates, not as copy-delete.
