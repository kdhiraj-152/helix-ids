# Phase 23 — Repository Structure Audit

> Generated: 2026-06-18
> Targets: `src/`, `tests/`, `scripts/`, `docs/`, `artifacts/`, `configs/`, `.github/`

---

## 1. Top-Level Layout

```
/
├── src/                    # Core package (helix_ids)
├── scripts/                # Training, evaluation, operations, data, CI scripts
├── tests/                  # All test suites
├── docs/                   # Documentation
├── config/                 # Configuration (YAML, TOML)
├── .github/                # CI/CD workflows + scripts
├── benchmarks/             # Benchmark baselines + README
├── data/                   # Dataset storage + processed artifacts
├── models/                 # Trained model checkpoints + deployment artifacts
├── results/                # Generated results (provenance, SBOM, trust reports)
├── artifacts/              # Runtime operation artifacts (gitignored)
├── session_logs/           # Agent session logs (gitignored)
├── .vscode/                # Editor settings (empty)
├── .hypothesis/            # Hypothesis test cache (gitignored)
├── .mypy_cache/            # Mypy cache (gitignored)
├── .pytest_cache/          # Pytest cache (gitignored)
├── .ruff_cache/            # Ruff cache (gitignored)
├── .venv311/               # Virtual environment (gitignored)

Root-level files:
├── pyproject.toml
├── README.md
├── AGENTS.md
├── Dockerfile
├── requirements.in
├── requirements-lock.txt
├── requirements-dev-lock.txt
├── requirements-all-lock.txt
├── pyrightconfig.json
├── .gitignore
├── .dockerignore
├── .coverage               # (gitignored)
├── coverage.xml            # (gitignored)
```

**Observations:**
- Layout is clean and well-organized overall.
- Cache directories and virtual environments are properly gitignored.
- No temp/scratch files at root level.

---

## 2. `src/helix_ids/` — Package Structure

```
src/helix_ids/
├── __init__.py
├── cli.py
├── adaptation/             # Domain adaptation (feature_harmonization, online_finetune)
├── config/                 # Environment + platform loading
├── contracts/              # Schema contracts, taxonomies, constants
├── data/                   # Data loading, preprocessing, augmentations
├── governance/             # AST validation, determinism, fingerprinting, provenance
├── metrics/                # Adversarial tests, per-class metrics, FN tracker
├── models/                 # Core models + sub-packages
│   ├── adaptation/         # DA losses (CORAL, MMD, DANN, etc.)
│   ├── attention.py
│   ├── classifier.py
│   ├── core.py
│   ├── full.py
│   ├── helix_ids.py
│   ├── helix_ids_full.py
│   └── loss.py
├── operations/             # Inference, baseline freeze, monitoring
│   ├── logging/            # Structured logging
│   ├── recovery/           # Restart manager
│   └── safety/             # Circuit breaker
└── utils/                  # Callbacks, export, metrics, entropy diagnostics
```

**Status: HEALTHY** — Clean package boundaries, well-organized sub-packages.

**Issues identified:**
- `src/helix_ids/data/feature_harmonization.py` and `src/helix_ids/adaptation/feature_harmonization.py` — Two files with the same name in different subpackages. Different concerns (data-level harmonization vs model-level adaptation), but naming collision is confusing.
- `_unused/` directory exists in gitignore pattern but directory doesn't exist on disk.

---

## 3. `tests/` — Test Structure

```
tests/
├── __init__.py
├── conftest.py
├── architecture/           # Architecture freeze, dependency checks, trainer boundary
├── config/                 # Environment loader
├── fixtures/               # Test data fixtures (CICIDS snapshot)
├── test_data/              # Data pipeline tests (diagnostic contracts, loaders, harmonization)
├── test_governance/        # Governance tests (AST validator, enforcement, promotion, schema)
├── test_models/            # Model tests (helix_full, helix_ids, loss)
├── test_operations/        # Operations tests (baseline, deployment, inference, monitoring, etc.)
├── test_training/          # Extracted component tests (data, diagnostics, losses, orchestration, etc.)
├── test_utils/             # Metrics tests
├── training/               # Checkpoint recovery, evaluation orchestrator, execution tests
└── (root-level test files) # 50+ test files at root
```

**Observation — NOTABLE ISSUE:**

There are **50+ test files at the root of `tests/`** rather than sorted into subdirectories. This indicates incomplete migration from a flat structure. While `test_*.py` is valid anywhere inside `tests/`, the density of root-level test files weakens the organizational signal.

| Root-level files (50+) | Suggested home |
|------------------------|----------------|
| `test_benchmark_formalization.py` | `tests/test_operations/`? |
| `test_checkpoint_chaos.py` | `tests/training/` |
| `test_checkpoint_contracts.py` | `tests/training/` |
| `test_data_loading.py` | `tests/test_data/` |
| `test_data_integrity_guards.py` | `tests/test_data/` |
| `test_dataset_corruption.py` | `tests/test_data/` |
| `test_feature_engineering.py` | `tests/test_data/` |
| `test_feature_harmonization.py` | `tests/test_data/` |
| `test_fuzz.py` | `tests/operations/` |
| `test_memory_leak_detection.py` | `tests/operations/` |
| `test_fault_injection.py` | `tests/operations/` |
| `test_property_based.py` | `tests/operations/` |
| `test_runtime_invariants.py` | `tests/operations/` |
| `test_runtime_monitoring_hardening.py` | `tests/operations/` |
| `test_schema_contract.py` | `tests/test_governance/` |
| `test_schema_registry_validation.py` | `tests/test_governance/` |
| ... and 30+ more | |

---

## 4. `scripts/` — Script Layout

```
scripts/
├── README.md
├── benchmarks/             # Benchmark pipeline, load test, soak tests, regression check
├── ci/                     # CI helper scripts (license checks, provenance, SLSA, validation)
├── data/                   # Dataset download, processing, linking
├── deployment/             # Deployment script
├── evaluation/             # Benchmark entrypoints, evaluation, smoke tests
├── governance/             # Governance logs parsing
├── operations/             # Rest serving, staging gates, baseline, traffic expansion
├── training/               # Main training scripts + extracted subpackages
│   ├── core/               # Trainer facade, factory, state, recovery
│   ├── data/               # Dataset builder, samplers, validators
│   ├── diagnostics/        # Cluster analyzer, geometry, rep diagnostics
│   ├── evaluation/         # Evaluation orchestrator, evaluator
│   ├── execution/          # Batch processor, epoch runner, training orchestrator
│   ├── governance/         # AB testing, orchestration, promotion, reporting
│   ├── losses/             # Contrastive, energy, regularization losses
│   ├── orchestration/      # Config parsing, governance pipeline, run orchestrator
│   ├── representation/     # Centroid manager, representation coordinator
│   ├── scheduler/          # Early stopping, freeze manager, LR scheduler, phase manager
│   └── validation/         # Artifacts, calibrator, evaluator, validation orchestrator
├── smoke_save_checkpoint.py  # ← DEAD (see Dead File Audit)
└── _rewrite_main.py          # ← DEAD (see Dead File Audit)
```

**Status: HEALTHY** — Well-organized. The `training/` subpackage has proper extraction into subdomains.

**Issues:**
- `scripts/training/_rewrite_main.py` — One-time helper with hardcoded absolute path.
- `scripts/smoke_save_checkpoint.py` — Orphan at `scripts/` root level, not referenced.

---

## 5. `docs/` — Documentation Structure

```
docs/
├── README.md               # Stale — references non-existent paths
├── architecture/           # 25+ docs — some active, some phase-historical
├── compliance/             # License policy, supply chain
├── development/            # Phase 11A cleanup report, project status
├── figures/                # 6 PNG figures (committed)
├── governance/             # ADRs, hash authority, schema governance, phase audits
├── manuscript/             # Paper drafts (TeX + Markdown)
├── operations/             # Runbooks, branch governance, phase docs
├── releases/               # RC1 readiness, RC2 certification
├── reports/                # Benchmark protocol, dataset report, mutation scorecard
├── reproducibility/        # Container reproducibility, data pipeline, build guide
└── security/               # Security posture, review
```

**Issues:**
- `docs/README.md` references paths that don't exist (`docs/archives/`, `docs/fig_revamp/`, `docs/operations/CHECKPOINT_AUDIT.md`, multiple `docs/development/` docs that moved, `docs/reports/` docs that don't exist).
- Phase-specific docs mixed with active architecture docs (see Dead File Audit).
- `docs/figures/fig*.png` — 5.9 MB of committed figures. Should these be in git or artifact storage?

---

## 6. `config/` — Configuration

```
config/
├── attack_params.yaml
├── helix_config.yaml
├── platform_configs.yaml
├── schema_registry.yaml
├── training.yaml
├── experiments/            # 4 experiment configs (smoke, drift, edge, governance)
└── mutation/               # 14 cosmic-ray TOML configs
```

**Status: CLEAN** — Well-organized. No stale configs identified.

---

## 7. `.github/` — CI/CD

```
.github/
├── dependabot.yml
├── scripts/                # 3 CI helper scripts (cycle check, reverse deps, trainer size)
└── workflows/              # 6 workflows (ci, quality, architecture, dependency-review, release, nightly)
```

**Status: CLEAN** — Already consolidated from 13 to 6 workflows in prior phase.

---

## 8. Structural Issues Summary

| # | Issue | Severity | Recommendation |
|---|-------|----------|----------------|
| S1 | `tests/` has 50+ flat test files at root | Medium | Migrate into subdirectories (test_data, test_governance, test_operations, training) |
| S2 | `docs/README.md` references non-existent paths | Medium | Update to match actual layout |
| S3 | `src/helix_ids/data/feature_harmonization.py` vs `src/helix_ids/adaptation/feature_harmonization.py` | Low | Same filename in different subpackages — confusing but different concerns |
| S4 | `docs/figures/fig*.png` (5.9 MB) committed to git | Low | Consider LFS or generating on demand |
| S5 | `results/v2_fixed/` directory — historical naming | Low | Should be renamed or archived |
