# Repository Audit Remediation Plan

> **Goal:** Fix all structural, naming, and stale-reference issues identified in the July 2 comprehensive audit, fully completing the rp-2 → helix-ids transition and hardening the repository for production use.

**Audit source:** Full folder audit performed 2026-07-02 (see session transcript).

**Architecture:** Six sequential phases, each with bite-sized tasks. Every task has a verification step. RED items first, then YELLOW, then GREEN.

**Tech Stack:** Python 3.11, bash, git, macOS

**Total estimated tasks:** ~25 across 6 phases

---

## Phase 0: Complete rp-2 → helix-ids Rename

**Objective:** Eliminate every stale reference to "RP-2" in source code and active operational files. Archived research docs (under docs/archive/) retain historical RP-2 references as-is — they are historical records, not active code.

---

### Task 0.1: Update cleanup scripts hardcoded paths

**Files:**
- Modify: `cleanup/generate_inventory.py:10` — `REPO = "/Users/kdhiraj/Downloads/RP-2"`
- Modify: `cleanup/generate_audit.py:9` — `REPO = "/Users/kdhiraj/Downloads/RP-2"`
- Modify: `cleanup/final_classify.py:7` — `REPO = "/Users/kdhiraj/Downloads/RP-2"`
- Modify: `cleanup/cleanup_execution_report.md:4` — references `/Users/kdhiraj/Downloads/RP-2`

**Step 1:** For each `.py` file, patch `REPO = "/Users/kdhiraj/Downloads/RP-2"` → `REPO = "/Users/kdhiraj/Downloads/helix-ids"`

**Step 2:** For the `.md` file, patch the RP-2 path reference.

**Step 3:** Verify with grep that no RP-2 absolute paths remain in cleanup/.

Run: `grep -rn "RP-2" cleanup/` → Expected: 0 matches

**Step 4:** Commit.

```bash
git add cleanup/generate_inventory.py cleanup/generate_audit.py \
       cleanup/final_classify.py cleanup/cleanup_execution_report.md
git commit -m "fix: update stale RP-2 absolute paths in cleanup scripts"
```

---

### Task 0.2: Update analysis scripts hardcoded paths

**Files (11 files):**
- `scripts/analysis/phase43a_fingerprint_attribution.py:14`
- `scripts/analysis/phase43b_fingerprint_ablation.py:20`
- `scripts/analysis/phase43c_coral_domain_alignment.py:16`
- `scripts/analysis/phase43d_semantic_preservation.py:19`
- `scripts/analysis/phase43e_cross_dataset_transfer.py:19`
- `scripts/analysis/phase43g_causal_attribution.py:24`
- `scripts/analysis/phase43h_failure_mode_taxonomy.py:16`
- `scripts/analysis/phase44c_diagnostics.py:12`
- `scripts/analysis/phase44c_transfer_baseline.py:9`
- `scripts/analysis/phase44c_transfer_coral.py:9`
- `scripts/analysis/phase45_causal_discovery.py:11`
- `scripts/analysis/architecture_audit_part_a.py:15`
- `scripts/analysis/phase52_runner.sh:8`
- `scripts/data/phase44c_ingestion.py:11`
- `scripts/data/link_raw_datasets_for_testing.sh:5`

Each has a line like `cd /Users/kdhiraj/Downloads/RP-2` or `Path("/Users/kdhiraj/Downloads/RP-2")` or `RAW_SOURCE_ROOT="${RAW_SOURCE_ROOT:-/Users/kdhiraj/Datasets/RP-2-raw}"`

**Step 1:** Patch all `"RP-2"` path strings → `"helix-ids"` and `"RP-2-raw"` → `"helix-ids-raw"` (the actual target directory name).

**Step 2:** Verify.

Run: `grep -rn "/RP-2" scripts/` → Expected: 0 matches

**Step 3:** Commit.

```bash
git add scripts/analysis/phase43*.py scripts/analysis/phase44*.py \
       scripts/analysis/phase45*.py scripts/analysis/architecture_audit_part_a.py \
       scripts/analysis/phase52_runner.sh scripts/data/phase44c_ingestion.py \
       scripts/data/link_raw_datasets_for_testing.sh
git commit -m "fix: update stale RP-2 absolute paths in analysis scripts"
```

---

### Task 0.3: Update data symlinks target name

**Files:**
- `data/cicids2018/raw` → symlink to `/Users/kdhiraj/Datasets/RP-2-raw/cicids2018/raw`
- `data/unsw_nb15/raw` → symlink to `/Users/kdhiraj/Datasets/RP-2-raw/unsw_nb15/raw`
- `data/nsl_kdd/raw` → symlink to `/Users/kdhiraj/Datasets/RP-2-raw/nsl_kdd/raw`

**Step 1:** Check if `/Users/kdhiraj/Datasets/helix-ids-raw` exists or if we should just leave the symlinks as-is (they resolve correctly since `RP-2-raw` exists).

**Important:** The target directory `/Users/kdhiraj/Datasets/RP-2-raw` still exists and the symlinks work. The RP-2-raw naming is a datasets storage concern, not a repo concern. If the user wants to rename that too, they must rename the directory first.

**Step 2:** If directory is renamed, remove each symlink and recreate with new target:

```bash
cd data
rm cicids2018/raw unsw_nb15/raw nsl_kdd/raw
ln -s /Users/kdhiraj/Datasets/helix-ids-raw/cicids2018/raw cicids2018/raw
ln -s /Users/kdhiraj/Datasets/helix-ids-raw/unsw_nb15/raw unsw_nb15/raw
ln -s /Users/kdhiraj/Datasets/helix-ids-raw/nsl_kdd/raw nsl_kdd/raw
```

**Step 3:** Also update the script reference in `scripts/data/link_raw_datasets_for_testing.sh:5` to match.

---

### Task 0.4: Update results phase reports with stale checkpoint paths

**Files:**
- `results/phase62/phase62_report.md:5` — `Checkpoint: /Users/kdhiraj/Downloads/RP-2/models/...`
- `results/phase63/phase63_report.md:5` — same pattern

**Step 1:** Patch `/Users/kdhiraj/Downloads/RP-2` → `/Users/kdhiraj/Downloads/helix-ids` in both files.

**Step 2:** Verify.

Run: `grep -rn "RP-2" results/` → Expected: 0 matches

**Step 3:** Commit.

```bash
git add results/phase62/phase62_report.md results/phase63/phase63_report.md
git commit -m "fix: update stale RP-2 checkpoint paths in results reports"
```

---

### Task 0.5: Final RP-2 sweep

**Step 1:** Run comprehensive grep to verify zero remaining RP-2 references in active code.

Run: `grep -rn "RP-2" --include="*.py" --include="*.sh" --include="*.toml" --include="*.yaml" --include="*.yml" . 2>/dev/null | grep -v ".git/" | grep -v ".venv311/" | grep -v "docs/archive/" | grep -v "__pycache__"`

Expected: 0 matches. (References in `docs/archive/` are historical records and intentionally preserved.)

**Step 2:** Report any remaining hits.

---

## Phase 1: Structural Cleanup (RED items)

---

### Task 1.1: Consolidate archive/ → docs/archive/

**Current state:**
- `archive/phase24a/` — 6 files (subset of old source code)
- `docs/archive/phase24a/` — already has docs for phase24a

The `archive/` at root appears to be leftover from a flattening operation. It contains old code snapshots that don't belong at root level.

**Decision:**
- Move `archive/` content to `docs/archive/phase24a/code_snapshot/` so it becomes part of the phase documentation.
- Remove the now-empty `archive/` top-level dir.

**Step 1:** Move code files into docs/archive/phase24a/.

```bash
mkdir -p docs/archive/phase24a/code_snapshot
cp -r archive/phase24a/* docs/archive/phase24a/code_snapshot/
git add docs/archive/phase24a/code_snapshot/
git rm -r archive/
git commit -m "refactor: consolidate archive/ into docs/archive/phase24a/code_snapshot/"
```

**Step 2:** Remove empty dirs left behind.

```bash
rmdir archive/ 2>/dev/null || true
```

---

### Task 1.2: Move cleanup/ to docs/archive/cleanup/

**Current state:** `cleanup/` at root has 14 files — operational plans and scripts from past cleanup phases. These are historical artifacts, not active code.

**Step 1:** Move the entire directory.

```bash
mkdir -p docs/archive/cleanup
mv cleanup/* docs/archive/cleanup/
git mv cleanup docs/archive/cleanup
# Or: git rm --cached cleanup/* and git add docs/archive/cleanup/*
```

Actually, simpler:

```bash
mkdir -p docs/archive/cleanup
git mv cleanup/* docs/archive/cleanup/
git rm -r cleanup 2>/dev/null; git add docs/archive/cleanup/
git commit -m "refactor: move cleanup/ to docs/archive/ (historical artifact)"
```

**Wait — problem:** `cleanup/` may have files tracked in git. Check first.

Run: `git ls-files cleanup/` to check if any cleanup files are tracked. If yes, use `git mv`. If no (they're untracked), just `mv` and add.

---

### Task 1.3: Ensure coverage.xml is properly gitignored

**Current state:** `coverage.xml` (409KB) is listed in `.gitignore` but was previously committed. Verify current status.

**Step 1:** Check if git still tracks coverage.xml.

Run: `git ls-files coverage.xml`

If tracked: `git rm --cached coverage.xml` and commit.

**Step 2:** Verify .gitignore pattern works.

Run: `git check-ignore coverage.xml` → Expected: `coverage.xml`

---

### Task 1.4: Remove duplicate coral_loss.py

**Files:**
- `src/helix_ids/training/coral_loss.py`
- `src/helix_ids/models/adaptation/coral_loss.py`

**Step 1:** Check if they're identical.

Run: `diff src/helix_ids/training/coral_loss.py src/helix_ids/models/adaptation/coral_loss.py`

**If identical** — remove the training/ copy and update any imports.
**If different** — investigate purpose, document the difference.

**Step 2:** Check imports across codebase to determine which one is actually used.

Run: `grep -rn "from.*coral_loss\|import.*coral_loss" --include="*.py" src/ scripts/ tests/`

**Step 3:** If safe to remove:

```bash
git rm src/helix_ids/training/coral_loss.py
git commit -m "refactor: remove duplicate coral_loss.py (identical copy in models/adaptation/)"
```

---

### Task 1.5: Clean empty directories

**34 empty dirs** identified:
- `results/gates/`, `results/manifests/`, `results/metrics/`
- `results/phase{48,49,50,51,52,53,54,58,59,60,61,62,63,64,65}/*/` (tables, latents, matrices, etc.)
- `archive/phase24a/src/helix_ids/{adaptation,data}/`

**Step 1:** Check if any are tracked by git.

Run: `for d in results/gates results/manifests results/metrics; do echo "$d: $(git ls-files $d | wc -l) tracked"; done`

**Step 2:** Remove empty dirs that aren't tracked.

```bash
find results/ -type d -empty -delete 2>/dev/null
find archive/ -type d -empty -delete 2>/dev/null
```

**Step 3:** For empty dirs that ARE tracked, remove them from git.

```bash
git ls-files results/gates results/manifests results/metrics | xargs git rm --cached 2>/dev/null
# Then git commit
```

---

## Phase 2: Test Organization (YELLOW items)

---

### Task 2.1: Consolidate tests/training/ and tests/test_training/

**Current state:**
- `tests/training/` — 18 files (newer extraction tests)
- `tests/test_training/` — 2 files (older tests)

**Step 1:** Check if the 2 files in `tests/test_training/` are covered by the 18 files in `tests/training/`.

Run: `diff <(ls tests/test_training/test_*.py) <(ls tests/training/test_*.py)`

**Step 2:** If they're distinct tests, move them into `tests/training/`.

```bash
git mv tests/test_training/test_extracted_data_components.py tests/training/
git mv tests/test_training/test_extracted_evaluation.py tests/training/
```

**Step 3:** Remove empty `tests/test_training/`.

```bash
git rm tests/test_training/__init__.py 2>/dev/null; rmdir tests/test_training/
```

**Step 4:** Fix any imports in the moved files that reference relative test paths.

**Step 5:** Commit.

```bash
git commit -m "refactor: consolidate tests/training/ and tests/test_training/"
```

---

### Task 2.2: Migrate root-level tests into subdirectories

**Current state:** 50 test files at `tests/` root. These should be organized by domain.

**Proposed mapping (group by src module):**

| Root test file | Target subdir | Rationale |
|---|---|---|
| `test_adversarial_robustness.py` | `test_operations/` | Tests inference robustness |
| `test_callbacks.py` | `test_models/` | Model callbacks |
| `test_classifier.py` | `test_models/` | Classifier model test |
| `test_coral_loss.py` | `test_models/` | Model adaptation loss |
| `test_combined_da.py` | `test_models/` | Domain adaptation |
| `test_label_aware_da.py` | `test_models/` | Domain adaptation |
| `test_mmd_loss.py` | `test_models/` | Model adaptation loss |
| `test_transfer_learning_da_schedule.py` | `test_models/` | Transfer learning |
| `test_loss_unit.py` | `test_models/` | Loss function tests |
| `test_model_inference.py` | `test_models/` | Model inference |
| `test_feature_harmonization.py` | `test_data/` | Feature pipeline |
| `test_feature_engineering.py` | `test_data/` | Feature pipeline |
| `test_preprocessing.py` | `test_data/` | Data preprocessing |
| `test_data_loading.py` | `test_data/` | Data loading |
| `test_data_integrity_guards.py` | `test_data/` | Data integrity |
| `test_dataset_corruption.py` | `test_data/` | Dataset corruption |
| `test_schema_contract.py` | `test_data/` | Schema contract |
| `test_regression_taxonomy_contracts.py` | `test_data/` | Taxonomy contracts |
| `test_regression_threat_weights.py` | `test_data/` | Threat weights |
| `test_export.py` | `test_operations/` | Export pipeline |
| `test_export_contract.py` | `test_operations/` | Export contracts |
| `test_export_quantization_deployment.py` | `test_operations/` | Deploy export |
| `test_export_unit.py` | `test_operations/` | Export unit tests |
| `test_provenance.py` | `test_governance/` | Provenance |
| `test_benchmark_formalization.py` | `tests/benchmarks/` (new) | Benchmark tests |
| `test_benchmark_output_validator.py` | `tests/benchmarks/` (new) | Benchmark validation |
| `test_check_performance_regression.py` | `tests/benchmarks/` (new) | Regression checks |
| `test_runtime_invariants.py` | `test_operations/` | Runtime |
| `test_runtime_monitoring_hardening.py` | `test_operations/` | Monitoring |
| `test_per_class_metrics.py` | `test_utils/` | Metrics |
| `test_fn_tracker.py` | `test_utils/` | FN tracker |
| `test_helix_ids_unit.py` | `test_models/` | Unit tests |
| `test_fault_injection.py` | `test_operations/` | Fault injection |
| `test_fuzz.py` | `test_operations/` | Fuzzing |
| `test_memory_leak_detection.py` | `test_operations/` | Memory |
| `test_checkpoint_chaos.py` | `test_operations/` | Checkpoint |
| `test_checkpoint_contracts.py` | `test_operations/` | Checkpoint |
| `test_validation_artifacts.py` | `test_governance/` | Validation |
| `test_validation_calibrator.py` | `test_governance/` | Calibration |
| `test_validation_evaluator.py` | `test_governance/` | Evaluation |
| `test_schema_registry_validation.py` | `test_governance/` | Schema registry |
| `test_lifecycle_verifier.py` | `test_governance/` | Lifecycle |
| `test_critical_pipeline_invariants.py` | `test_governance/` | Pipeline |
| `test_entropy_deployment_validation.py` | `test_operations/` | Deployment |
| `test_training_pipeline_v2.py` | `tests/training/` | Training pipeline |
| `test_training_direct_adaptation_eval.py` | `tests/training/` | Training eval |
| `test_prepare_canonical_artifacts.py` | `tests/training/` | Training artifacts |
| `test_e2e_smoke.py` | `tests/training/` or new `tests/e2e/` | Smoke test |
| `test_property_based.py` | `tests/` root (keep) | Property-based = cross-cutting |
| `conftest.py` | `tests/` root (keep) | Fixtures |

**Step 1:** Create new subdirs as needed and `git mv` each group, updating relative imports.

Run: `mkdir -p tests/benchmarks tests/e2e`

**Step 2:** For each moved file, update any `from .` or `from tests.` relative imports.

**Step 3:** Run full test suite to verify nothing broke.

Run: `PYTHONPATH=src pytest -q --tb=short 2>&1 | tail -20`

**Step 4:** Commit after each group move, or batch the whole thing.

```bash
git commit -m "refactor: organize root-level tests into domain subdirectories"
```

---

## Phase 3: Naming Conventions (GREEN items)

---

### Task 3.1: Rename trust-report directory

**Current:** `results/trust-report/`
**Target:** `results/trust_report/`

**Step 1:** Check if any file references the old path.

Run: `grep -rn "trust-report" --include="*.py" --include="*.md" --include="*.json" .`

**Step 2:** Rename.

```bash
git mv results/trust-report results/trust_report
```

**Step 3:** Update any references found in step 1.

**Step 4:** Commit.

```bash
git commit -m "style: rename trust-report -> trust_report (kebab to snake)"
```

---

### Task 3.2: Rename _monitor directory in soak artifacts

**Current:** `artifacts/soak/_monitor/`
This uses an underscore prefix which is fine — it signals "private/internal" in Python convention. However, it's not a Python package. Still, it's consistent with the codebase convention. **Recommend keeping as-is** — the underscore prefix is conventional for internal/status dirs.

**Verdict:** Skip. Low value, no consistency gain.

---

## Phase 4: CI Gate Compliance

---

### Task 4.1: Assess train_helix_ids_full.py LOC breach

**Current:** 4,605 LOC (CI gate: ≤2,000 LOC — defined in `.github/scripts/trainer_size_check.py`)

**Step 1:** Read the existing extraction subdirs to understand what's already been extracted.

Existing extraction targets:
- `scripts/training/core/` — 4 files (recovery, facade, factory, state)
- `scripts/training/data/` — 3 files (builder, samplers, validators)
- `scripts/training/diagnostics/` — 3 files
- `scripts/training/evaluation/` — 2 files (evaluator, orchestrator)
- `scripts/training/execution/` — 4 files
- `scripts/training/governance/` — 4 files
- `scripts/training/losses/` — 4 files
- `scripts/training/orchestration/` — 3 files
- `scripts/training/representation/` — 2 files
- `scripts/training/scheduler/` — 5 files
- `scripts/training/validation/` — 3 files

**This is NOT a Phase 4 task** — extraction is an ongoing effort beyond the scope of this cleanup plan. Document the status and move on.

**Verdict:** Deferred. Extraction effort requires its own dedicated plan.

---

## Phase 5: Documentation & Verification

---

### Task 5.1: Update AGENTS.md file counts

**Step 1:** Re-count files after all moves.

Run:
```bash
echo "Python files: $(find . -not -path './.git/*' -not -path './.venv311/*' -name '*.py' -type f | wc -l)"
echo "Markdown files: $(find . -not -path './.git/*' -not -path './.venv311/*' -name '*.md' -type f | wc -l)"
echo "Core package (src/helix_ids/): $(find src/helix_ids -name '*.py' -type f | wc -l)"
echo "Scripts: $(find scripts -name '*.py' -type f | wc -l)"
echo "Tests: $(find tests -name '*.py' -type f | wc -l)"
```

**Step 2:** Update the file count section in AGENTS.md (section 13) to reflect new totals.

---

### Task 5.2: Final smoke test

**Step 1:** Run lint and test suite.

```bash
ruff check src scripts tests
PYTHONPATH=src pytest -q --ignore=tests/architecture --ignore=tests/training 2>&1 | tail -10
```

**Step 2:** Verify no regressions introduced by the moves.

---

## Execution Order Summary

| Phase | Priority | Tasks | Est. Time |
|-------|----------|-------|-----------|
| 0 | 🔴 Critical | 5 tasks — complete rp-2 rename | 15 min |
| 1 | 🔴 High | 5 tasks — structural cleanup | 20 min |
| 2 | 🟡 Medium | 2 tasks — test organization | 30 min |
| 3 | 🟢 Low | 1 task — naming fix | 5 min |
| 4 | 🟡 Medium | Assessment only — LOC breach | 5 min |
| 5 | 🟢 Low | 2 tasks — docs + verification | 10 min |
| **Total** | | **~16 tasks** | **~85 min** |

---

## Key Decisions Documented

1. **Historical docs preserve RP-2** — `docs/archive/` research docs are historical records. Updating them would destroy provenance. Only active code paths are fixed.
2. **`_monitor` naming kept** — underscore prefix is valid Python convention for internal status directories.
3. **`train_helix_ids_full.py` extraction deferred** — requires a dedicated extraction plan, outside cleanup scope.
4. **`data/` symlinks left if target dir unchanged** — the dataset storage path is an external concern. Only update if `/Users/kdhiraj/Datasets/RP-2-raw` is renamed.
5. **50 root tests migration is optional** — high-value but disruptive. Consider deferring if the test suite is in active development.
