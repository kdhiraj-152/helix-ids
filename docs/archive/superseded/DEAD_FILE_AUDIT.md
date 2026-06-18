# Phase 23 — Dead File Audit

> Generated: 2026-06-18
> Repository: HELIX-IDS (RP-2)

---

## Methodology

Scanned every file in the repository (excluding `.git/`, `.venv*/`, cache dirs).
Each candidate was cross-referenced against:

- Runtime imports (code paths)
- Doc references (documentation mentions)
- Config references (YAML/TOML entrypoints)
- Test imports
- CLI entrypoints (`src/helix_ids/cli.py`)
- Dependency graph (`docs/architecture/dependency_graph.json`)
- Git history / usage patterns

---

## Classification Categories

| Label | Meaning |
|-------|---------|
| **SAFE_DELETE** | Dead code, no runtime or documentation dependency. Can be deleted immediately. |
| **ARCHIVE** | Historical or phase-specific artifact that should be moved to `docs/archive/` rather than deleted. Preserves traceability. |
| **KEEP** | Still active, referenced, or serving a documented purpose. May need renaming or relocation (covered in other Phase 23 audits). |

---

## 1. Scripts

### SAFE_DELETE

| File | Reason |
|------|--------|
| `scripts/training/_rewrite_main.py` | One-time Phase 13A-4 extraction helper. Contains hardcoded absolute path (`/Users/kdhiraj/Downloads/RP-2/`). Not importable, not referenced by any workflow/config/doc. |
| `scripts/smoke_save_checkpoint.py` | One-off smoke test helper for checkpoint callback. Not referenced by any test, config, doc, or CLI entrypoint. Writes to `artifacts/test_smoke/`. |

### KEEP

| File | Justification | Notes |
|------|---------------|-------|
| `scripts/training/train_multidataset_v2_fixed.py` | Imported by `tests/test_data_loading.py` (`SafeDataLoader`). 1125 lines. | Naming contains `v2_fixed` → covered in naming audit. |
| `scripts/training/train_unified_rebalanced.py` | Imported by `tests/test_training_direct_adaptation_eval.py`. 378 lines. | — |
| `scripts/training/train_unsw_only_cleaned.py` | Referenced in `docs/architecture/TRAINING_METHODOLOGY.md`. 277 lines. | Naming contains `_cleaned` → covered in naming audit. |
| `scripts/training/adversarial_training_v2.py` | Referenced by `src/helix_ids/cli.py` via `scripts.adversarial_training_v2`. 261 lines. | Naming contains `_v2` → covered in naming audit. |
| `scripts/evaluation/benchmark_e2e_v2_fixed.py` | Entrypoint in all 4 experiment configs (`smoke.yaml`, `drift_robustness.yaml`, `edge_latency.yaml`, `governance_ablation.yaml`). 249 lines. | Renaming candidate — see naming audit. |
| `scripts/evaluation/holdout_evaluation_v2.py` | Referenced in `docs/reproducibility/REPRODUCIBILITY.md` and `docs/architecture/EXPERIMENTAL_SETUP.md`. 319 lines. | Renaming candidate — see naming audit. |

---

## 2. Documentation

### ARCHIVE (move to `docs/archive/`)

These are phase-specific reports that document completed work. They are valuable for traceability but do not describe the current system state.

| File | Phase | Size |
|------|-------|------|
| `docs/architecture/PHASE13B_AUDIT.md` | Phase 13B — Architecture Audit | 374 lines |
| `docs/architecture/PHASE19_ARCHITECTURE_FREEZE.md` | Phase 19 — Architecture Freeze | 301 lines |
| `docs/architecture/PHASE22_RELIABILITY_PLAN.md` | Phase 22 — Reliability Plan | 133 lines |
| `docs/development/PHASE_11A_CLEANUP_REPORT.md` | Phase 11A — Cleanup Report | 554 lines |
| `docs/governance/PHASE_4A_GOVERNANCE_COVERAGE_AUDIT.md` | Phase 4A — Governance Coverage | — |
| `docs/governance/PHASE_4B_ASSUMPTION_ELIMINATION.md` | Phase 4B — Assumption Elimination | — |
| `docs/operations/PHASE23_CICD_CONSOLIDATION.md` | Phase 23 — CI/CD Consolidation | 127 lines |

### SAFE_DELETE (no traceability value — superseded)

| File | Reason |
|------|--------|
| `docs/architecture/DEAD_CODE_AUDIT.md` | Dead code audit from earlier phase. Current state has moved on; this audit supersedes it. |
| `docs/architecture/DEPENDENCY_LOCKDOWN.md` | Dependency lockdown from earlier phase. Current dependency state is covered by lockfiles and `requirements.in`. |
| `docs/architecture/PERFORMANCE_BASELINE.md` | Performance baseline from earlier phase. Superseded by `benchmarks/baseline.reference.json` and `benchmarks/README.md`. |
| `docs/reproducibility/REPRODUCIBILITY_GAP.md` | Reproducibility gap analysis from earlier phase. Superseded by `REPRODUCIBILITY_AUDIT.md` and `CONTAINER_REPRODUCIBILITY.md`. |

### KEEP (active, referenced, or still needed)

All other docs under `docs/architecture/`, `docs/compliance/`, `docs/governance/`, `docs/operations/`, `docs/releases/`, `docs/reports/`, `docs/reproducibility/`, `docs/security/`, and `docs/manuscript/` — except those listed above.

---

## 3. Config Files

### KEEP (all still active)

| File | Use |
|------|-----|
| `config/experiments/drift_robustness.yaml` | Referenced by benchmark entrypoints |
| `config/experiments/edge_latency.yaml` | Referenced by benchmark entrypoints |
| `config/experiments/governance_ablation.yaml` | Referenced by benchmark entrypoints |
| `config/experiments/smoke.yaml` | Referenced by benchmark entrypoints + reproducibility docs |
| All `config/mutation/*.toml` | Active cosmic-ray mutation testing configs |
| `config/attack_params.yaml` | Active adversarial config |
| `config/helix_config.yaml` | Core runtime config |
| `config/platform_configs.yaml` | Platform deployment configs |
| `config/schema_registry.yaml` | Schema governance |
| `config/training.yaml` | Training config |

---

## 4. JSON / Data Files

### SAFE_DELETE (generated/dead artifacts)

| File | Reason |
|------|--------|
| `.vscode/settings.json` | Empty file (`{}`). No editor configuration applied. |
| `artifacts/soak/smoke_test_20260618_032231/snapshot_2026-06-17T21-52-31.864494+00-00.json` | One-time soak test snapshot. Already gitignored (`artifacts/` in `.gitignore`). Not tracked. |
| `artifacts/operations/live_events.jsonl` | Runtime live event log. Already gitignored. Not tracked. |
| `session_logs/session_772781e6-9328-43f5-999d-d8512944e3fd_20260618_152149.json` | Agent session log. Already gitignored (`session_logs/` in `.gitignore`). |

### KEEP

All JSON/data files in `data/`, `benchmarks/`, `docs/architecture/`, `results/`, `models/`.

---

## 5. Benchmark / Results Artifacts

### KEEP (but note status)

| File | Status | Notes |
|------|--------|-------|
| `benchmarks/baseline.json` | Output artifact (gitignored) | Current run output. |
| `benchmarks/baseline.reference.json` | Tracked reference | Blessed baseline. Committed. |
| `benchmarks/load_test_results.json` | Output artifact (gitignored) | Load test output. |
| `results/v2_fixed/training_v2.log` | Historical benchmark log | From `benchmark_e2e_v2_fixed.py` run. Should be archived or deleted. |

---

## 6. Lockfiles

### KEEP

| File | Lines | Purpose |
|------|-------|---------|
| `requirements-lock.txt` | 753 | Core runtime dependencies (generated from `requirements.in`) |
| `requirements-dev-lock.txt` | 1,018 | Dev dependencies (subset of all) |
| `requirements-all-lock.txt` | 3,152 | Full dependency tree (all extras) |
| `requirements.in` | 11 | Source dependency manifest |

**Note:** `requirements-dev-lock.txt` and `requirements-all-lock.txt` overlap. Consider whether the dev lockfile is redundant (see Dependency Audit).

---

## Summary

| Category | SAFE_DELETE | ARCHIVE | KEEP |
|----------|-------------|---------|------|
| Scripts | 2 | 0 | 6+ |
| Documentation | 4 | 7 | 40+ |
| Config | 0 | 0 | 15+ |
| JSON/Data | 4 | 0 | 20+ |
| Benchmark/Results | 0 | 0 | 5 |
| Lockfiles | 0 | 0 | 4 |
| **Total** | **10** | **7** | **~90** |
