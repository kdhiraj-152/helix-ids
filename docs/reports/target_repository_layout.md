## Proposed Target Layout (finalized)

Below is the recommended, production-ready repository layout and a concise set of naming and organizational conventions — opinionated, minimal, and maintainable.

Root (high level)
- README.md                — project README, quickstart, contact
- LICENSE
- pyproject.toml           — build / packaging / tooling
- .github/                 — CI workflows and automation
- .gitignore               — explicit ignore patterns

Top-level directories (purpose-driven)

- `config/`                — runtime and deployment configs (small, tracked)

- `src/helix_ids/`         — Python package; all source code lives here.
   - `helix_ids/`           — package module structure
      - `__init__.py`
      - `cli.py`
      - `config/`
      - `data/`              — processing code, feature harmonizers
      - `governance/`        — validators, lifecycle, fingerprinting
      - `metrics/`
      - `models/`            — model definitions and small deploy configs
      - `operations/`
      - `pipeline/`
      - `utils/`

- `scripts/`               — runnable entry scripts grouped by domain
   - `ci/`                  — CI helpers and validators (authoritative)
   - `data/`                — dataset download and processing entrypoints
   - `evaluation/`          — evaluation harnesses (authoritative files only)
   - `governance/`          — governance helpers and log parsers
   - `operations/`          — serving, staging, monitoring helpers
   - `training/`            — training entrypoints and experiment runners

- `docs/`                  — canonical, curated documentation
   - `governance/`          — ADRs, schema contracts, governance policies
   - `audits/`              — research/audit outputs (small curated set)
   - `archives/`            — historical phase reports (read-only)
   - `fig/`                 — canonical figures (keep single variant)

- `tests/`                 — pytest test suite (all tests tracked)

- `artifacts/`, `checkpoints/`, `results/`, `logs/` — gitignored runtime/generated outputs

Naming and conventions (brief)

- Script names: use explicit verbs and underscores, e.g. `train_*.py`, `serve_*.py`, `export_*.py`.
- One authoritative entrypoint per responsibility — no root-level shims.
- Generated artifacts must not live in `docs/` or `src/`.
- `src/` is the canonical Python package root; CI and dev tooling must run with `PYTHONPATH=src`.
- Tests are fully tracked; CI runs a minimal gateset but maintainers run full suite locally.

Immediate next-step recommendations (ordered)

1. Fix `.gitignore` to avoid accidentally excluding `src/` subdirectories (replace broad `models/` line with explicit patterns: `models/*.pt`, `models/*.pth`, `checkpoints/`). (P0)
2. Add the 9 untracked test files to git and run `pytest -q`; fix import breakages iteratively. (P0)
3. Ensure CI workflows reference authoritative `scripts/` paths and remove deprecated shims. (P1)
4. Move all phase4/phase5 historical reports into `docs/archives/phase5/` (archived). (P1)
5. Move audit/research docs into `docs/audits/` and keep `docs/` concise. (P1)
6. Remove remaining generated root-level diagnostics (done in this run). (P1)
7. Optionally run a migration script that applies file moves and updates imports in a single atomic commit; review before merging. (P2)

Actions already performed in this run

- Archived Phase 5 reports to `docs/archives/phase5/`.
- Deleted generated root reports: `coverage.xml`, `repo_tree.txt`, `clean_tree.txt`, `dead_code.txt`, `duplicate_hashes.txt`, `tracked_files.txt`, `size_audit.txt`, `git_status_audit.txt`, `finding-structured.json`, `governance_test_analysis.md`.
- Removed duplicate shim scripts: `scripts/parse_promotion_gate_logs.py`, `scripts/train_multidataset_v2_fixed.py`.

Next choices for me to take (pick one)

- (A) Apply `.gitignore` fix + `git add` the 9 missing test files, run `pytest`, and iteratively fix import errors. I will create a branch and open a draft PR-ready patch.
- (B) Scaffold a migration script that will perform file moves and update imports; I will produce a preview patch showing all changes before applying.
- (C) Stop and wait for your approval/edits to the proposed layout.

Which option do you want me to do now? (A/B/C)

│   ├── test_data/
│   ├── test_governance/
│   ├── test_models/
│   ├── test_operations/
│   ├── test_utils/
│   └── test_*.py                ← 9 test files UNTRACKED, need git add
└── (root-level clutter)
    ├── _check_validators.py     ← DELETE
    ├── artifacts/               ← gitignore
    ├── checkpoints/             ← gitignore
    ├── clean_tree.txt           ← DELETE (generated)
    ├── coverage.xml             ← DELETE (generated)
    ├── dead_code.txt            ← DELETE (generated)
    ├── duplicate_hashes.txt      ← DELETE (generated)
    ├── finding-structured.json  ← DELETE (git-tracked, generated)
    ├── git_status_audit.txt     ← DELETE (generated)
    ├── governance_test_analysis.md  ← DELETE (generated)
    ├── jscpd-report/            ← gitignore
    ├── models/                  ← gitignore (top-level ML model storage)
    ├── phase5R_*.md             ← ARCHIVE
    ├── phase5_*.md              ← ARCHIVE
    ├── repo_hygiene_bundle.txt  ← DELETE (generated)
    ├── repo_tree.txt            ← DELETE (generated)
    ├── results/                 ← gitignore
    ├── schema_registry.yaml     ← DELETE (generated)
    ├── session_logs/            ← gitignore
    ├── size_audit.txt           ← DELETE (generated)
    ├── tmp_artifact/            ← DELETE (git-tracked, temp)
    └── tracked_files.txt        ← DELETE (generated)
```

---

## Problems Identified

### P0: Structural

1. **`models/` gitignore bug** — `src/helix_ids/models/` Python files are
   excluded from git (35 source files untracked).

2. **9 test files not in git** — `tests/test_benchmark_formalization.py`,
   `test_benchmark_output_validator.py`, `test_governance/test_ast_validator.py`,
   `test_governance/test_enforcement_completeness.py`, `test_governance/test_legacy_policy.py`,
   `test_governance/test_nested_schema_validation.py`, `test_governance/test_validate_schema_registry.py`,
   `test_lifecycle_verifier.py`, `test_schema_registry_validation.py`.

3. **Root-level clutter** — 15+ generated/investigative files at root.

### P1: Organization

4. **Duplicate shim scripts** — `scripts/parse_promotion_gate_logs.py` (317 B)
   and `scripts/train_multidataset_v2_fixed.py` (618 B) are shims whose canonical
   location is `scripts/governance/parse_promotion_gate_logs.py` and
   `scripts/training/train_multidataset_v2_fixed.py` respectively.

5. **docs/results/** artifact leakage — `staging_validation.json` and
   `staging_baseline_checkpoint.pt` (6 MB) in docs/results/ are generated
   artifacts that should be in `checkpoints/` (gitignored).

6. **docs/fig/ and docs/fig_revamp/** — duplicate figure sets (12 files,
   potentially 15 MB). Keep one.

### P2: Cleanup

7. **Phase 4/5 archive candidates** — 4 governance docs + 6 phase5 reports
   should move to archive.

8. **`coverage.xml`** at root (490 KB) should be deleted.

9. **docs/.DS_Store** files (2× 10 KB) should be deleted.

---

## Proposed Target Layout

```
/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── pyrightconfig.json
├── requirements.txt
│
├── .gitignore                   ← Fixed: models/ → models/*.pt, models/*.pth, checkpoints/
│
├── config/                      ← KEEP as-is
│
├── data/                        ← Restructure: keep splits/processed, gitignore raw
│   ├── splits/
│   │   ├── nsl_kdd_train.csv
│   │   ├── nsl_kdd_test.csv
│   │   └── unsw_nb15_*.csv
│   └── processed/               ← .gitignore large processed files
│       ├── multi_dataset_v1/
│       └── multi_dataset_v1_nsl_repaired/
│   (raw/ should be gitignored — too large)
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── CHECKPOINT_AUDIT.md
│   ├── OPERATIONS_DEPLOYMENT_RUNBOOK.md
│   ├── SCHEMA_CONTRACT.md
│   ├── HELIX_FORENSIC_CANONICALIZATION_AUDIT.md
│   │
│   ├── governance/
│   │   ├── ADR-001-governance-philosophy.md
│   │   ├── ADR-002-schema-lifecycle.md
│   │   ├── ADR-003-hash-authority.md
│   │   ├── ADR-004-enforcement-pipeline.md
│   │   ├── hash_authority.md
│   │   ├── manifest_schema_governance.md
│   │   ├── result_schema_governance.md
│   │   └── IMMUTABLE_SCHEMA_CONTRACT.md  ← REVIEW: merge or delete
│   │
│   ├── audits/                  ← NEW: research audits
│   │   ├── EXPORT_CONTRACT_REPORT.md
│   │   ├── CHECKPOINT_AUDIT.md
│   │   └── HELIX_FORENSIC_*.md
│   │
│   ├── manuscript/
│   │   ├── HELIX_submission_ready.md
│   │   ├── HELIX_ieee_variant.md
│   │   ├── IEEE_variant_revamp_plan.md
│   │   └── conference-template-letter.docx
│   │
│   ├── fig/                     ← KEEP one variant (fig/ is larger, keep that)
│   │   └── (keep only the cleaner figure set)
│   │
│   └── archives/                ← NEW: historical phase docs
│       ├── phase4/
│       │   ├── phase4a_governance_coverage_audit.md
│       │   ├── phase4b_assumption_elimination.md
│       │   └── reproducibility_gap_analysis.md
│       └── phase5/
│           ├── phase5_governance_coverage_report.md
│           ├── phase5_drift_report.md
│           ├── phase5_risk_register.md
│           ├── phase5_closure_statement.md
│           ├── phase5_assumption_report.md
│           └── phase5R_governance_remediation_report.md
│
├── scripts/
│   ├── README.md
│   ├── ci/
│   │   ├── validate_benchmark_outputs.py
│   │   ├── validate_governance_consistency.py
│   │   ├── validate_governance_docs.py
│   │   ├── validate_schema_registry.py
│   │   └── verify_contract_sidecars.py
│   ├── data/
│   │   ├── download_datasets.py
│   │   ├── link_raw_datasets_for_testing.sh
│   │   ├── process_cicids.py
│   │   ├── process_nsl_kdd.py
│   │   ├── process_unsw_nb15.py
│   │   └── unlink_raw_datasets_for_deployment.sh
│   ├── deployment/
│   │   └── deploy.py
│   ├── evaluation/
│   │   ├── benchmark_e2e_v2_fixed.py  ← AUTHORITATIVE
│   │   ├── benchmarks.py               ← DUPLICATE → DELETE
│   │   ├── benchmark_result.schema.json
│   │   ├── holdout_evaluation_v2.py
│   │   └── test_phase3_smoke.py
│   ├── governance/                   ← canonical location for governance scripts
│   │   ├── parse_promotion_gate_logs.py
│   │   └── (scripts/parse_promotion_gate_logs.py → DELETE shim)
│   ├── operations/
│   │   ├── export_inference_bundle.py
│   │   ├── freeze_baseline.py
│   │   ├── serve_rest.py
│   │   ├── staging_gate_check.py
│   │   ├── stress_validate_baseline.py
│   │   ├── traffic_expansion_guard.py
│   │   └── visualize_helix_demo.py
│   └── training/
│       ├── train_helix_ids_full.py    ← AUTHORITATIVE
│       ├── train_multidataset_v2_fixed.py  ← AUTHORITATIVE
│       ├── train_unified_rebalanced.py
│       ├── train_unsw_only_cleaned.py
│       ├── adversarial_training_v2.py
│       ├── prepare_canonical_artifacts.py
│       └── train_edge_models.py
│       (scripts/train_multidataset_v2_fixed.py → DELETE shim)
│
├── src/helix_ids/               ← After .gitignore fix: all tracked
│   ├── __init__.py
│   ├── adaptation/
│   ├── cli.py
│   ├── config/
│   ├── contracts/
│   ├── data/
│   ├── governance/
│   ├── metrics/
│   ├── models/                  ← Will be tracked after .gitignore fix
│   ├── operations/
│   ├── pipeline/
│   └── utils/
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── fixtures/
│   ├── test_data/
│   ├── test_governance/
│   ├── test_models/
│   ├── test_operations/
│   ├── test_utils/
│   └── test_*.py               ← All 56 tests should be tracked
│
├── checkpoints/                ← gitignored, for .pt artifacts
├── results/                    ← gitignored, for evaluation outputs
├── artifacts/                 ← gitignored, for generated artifacts
└── logs/                       ← gitignored, for runtime logs
```

---

## Summary of Structural Changes

| Change | Description | Priority |
|---|---|---|
| Fix `.gitignore` line 35 | Change `models/` to `models/*.pt`, `models/*.pth`, `checkpoints/` | P0 |
| Git add 9 test files | Add untracked tests to version control | P0 |
| Delete root clutter | Remove 15+ generated/investigative files at root | P0 |
| Move shim scripts | `scripts/parse_promotion_gate_logs.py` shim → governance/; `scripts/train_multidataset_v2_fixed.py` shim → training/ | P1 |
| Create `docs/archives/` | Archive phase4/phase5 reports | P1 |
| Create `docs/audits/` | Move research audit docs to dedicated folder | P1 |
| Delete `docs/results/` | Move or delete staging artifacts in docs/results/ | P1 |
| Dedupe figures | Keep only `docs/fig/`, delete `docs/fig_revamp/` | P2 |
| Review `IMMUTABLE_SCHEMA_CONTRACT.md` | Compare with SCHEMA_CONTRACT.md; merge or delete | P2 |
| Delete `coverage.xml` | Generated coverage report | P1 |