# Architecture Findings

## Dependency Graph Summary

| Metric | Value |
|--------|-------|
| Total files parsed | 290 |
| Total LOC | 103,656 |
| Total functions | 4,747 |
| Total classes | 690 |
| Graph nodes (files with cross-project imports) | 80 |
| Graph edges (cross-project imports) | 171 |
| Circular import cycles | 3 (all benign __init__ ↔ submodule in orchestration/) |

## Fan-In Leaders (Most Imported)

| Rank | File | Fan-In | Risk |
|------|------|--------|------|
| 1 | `scripts/training/losses/__init__.py` | 9 | Ripple effect from loss changes |
| 2 | `scripts/training/core/trainer_state.py` | 7 | Training state changes affect 7 consumers |
| 3 | `scripts/training/scheduler/__init__.py` | 7 | Scheduling changes propagate broadly |
| 4 | `scripts/training/evaluation/__init__.py` | 6 | Evaluation contract changes |
| 5 | `scripts/training/validation/__init__.py` | 6 | Validation contract changes |

## Fan-Out Leaders (Most Coupled)

| Rank | File | Fan-Out | Risk |
|------|------|---------|------|
| 1 | `scripts/training/train_helix_ids_full.py` | 13 | Tightly coupled to 13 modules — extraction candidate |
| 2 | `scripts/training/core/trainer_facade.py` | 9 | Orchestrator pattern, expected high fan-out |
| 3 | `scripts/training/core/trainer_state.py` | 7 | Central state object |
| 4 | `scripts/training/scheduler/__init__.py` | 5 | Scheduler re-exports |
| 5 | Various | 4 | |

## God Files (>1000 LOC)

| File | LOC | Functions | Classes | Risk |
|------|-----|-----------|---------|------|
| `scripts/training/train_helix_ids_full.py` | 4,606 | 73 | 3 | HIGH — SRP violation, hard to test |
| `scripts/training/core/trainer_facade.py` | 2,818 | 32 | 1 | HIGH — monolithic trainer |
| `src/helix_ids/governance/contracts/learnability_contract.py` | 2,093 | 47 | 1 | MEDIUM — governance, expected size |
| `src/helix_ids/data/multi_dataset_loader.py` | 1,577 | 51 | 1 | HIGH — data loading monolith |
| `src/helix_ids/data/feature_engineering.py` | 1,422 | 36 | 1 | MEDIUM — centralized feature logic |
| `src/helix_ids/models/adaptation/transfer_learning.py` | 1,270 | 44 | 1 | MEDIUM — adaptation logic |
| `src/helix_ids/operations/inference_runtime.py` | 1,100 | 30 | 2 | MEDIUM |
| `scripts/training/orchestration/run_orchestrator.py` | 1,057 | 24 | 2 | MEDIUM |
| `src/helix_ids/utils/export.py` | 1,056 | 28 | 1 | MEDIUM |
| `scripts/training/train_unified_rebalanced.py` | 1,039 | 21 | 1 | MEDIUM — training entrypoint |
| `scripts/training/train_multidataset.py` | 1,015 | 17 | 1 | MEDIUM |
| `scripts/evaluation/benchmark_e2e.py` | 1,010 | 23 | 1 | LOW — test infrastructure |
| 5 additional files | 1000-1049 | | | LOW |

## God Classes (>500 LOC)

| Class | File | LOC | Risk |
|-------|------|------|------|
| `HelixFullTrainer` | `trainer_facade.py` | 1,929 | HIGH — 32 methods, 109 method calls |
| `MultiDatasetLoader` | `multi_dataset_loader.py` | 1,443 | HIGH — SRP violation |
| `MultiDatasetPretrainer` | `transfer_learning.py` | 1,049 | MEDIUM |
| 9 additional classes | various | 500-900 | MEDIUM |

## God Functions (>100 LOC)

**58 functions > 100 LOC** found. Top offenders:

| Function | File | LOC | Risk |
|----------|------|------|------|
| `run_orchestration` | `run_orchestrator.py` | 900 | HIGH — orchestrator complexity |
| `HelixFullTrainer.__init__` | `trainer_facade.py` | 392 | HIGH — constructor doing too much |
| `main` (train_multidataset) | `train_multidataset.py` | 368 | HIGH — monolith entrypoint |
| 55 more | various | 100-350 | MEDIUM |

## Layer Violations

| Finding | Severity | Location | Evidence |
|---------|----------|----------|----------|
| Script imports from src/ (expected) | INFO | All scripts | Scripts are entrypoints |
| governance imports models | MEDIUM | `governance/orchestrator.py` | Imports model classes directly |
| data imports governance | MEDIUM | `data/fingerprinting.py` | Data layer depends on governance |
| operations imports from scripts/training | MEDIUM | `operations/inference_runtime.py` | Runtime depends on training code |

## Hidden Coupling

`ENGINEERED_FEATURE_NAMES` is imported from `scripts/training/` into `src/helix_ids/`:
- `scripts/training/data/feature_engineering.py` defines `ENGINEERED_FEATURE_NAMES`
- `src/helix_ids/contracts/schema_contract.py` references it (via import from scripts)
- **Violation**: Core package depends on training scripts — should be in `src/helix_ids/contracts/`

## Import Hygiene

- **0 unused imports** detected by ruff (F401) ✅
- 2 `E741` ambiguous-variable-name (single-letter vars)
- 2 `I001` unsorted-imports

## Circular Import Analysis

3 cycles detected, all within `scripts/training/orchestration/`:
- `__init__.py` ↔ `config_parser.py`
- `__init__.py` ↔ `governance_pipeline.py`
- `__init__.py` ↔ `run_orchestrator.py`

**Assessment**: Benign. These are Python package re-export patterns where `__init__.py` imports submodules, and submodules reference other names from `__init__.py`. No actual runtime circular dependency exists because Python processes the module cache before the cycle completes.

## Architecture Scorecard

| Gate | Status | Evidence |
|------|--------|----------|
| Acyclic import graph | ✅ PASS | 3 benign cycles only |
| Module boundaries clear | ✅ PASS | src/helix_ids organized into 9 subpackages |
| No layer violations | ⚠️ PARTIAL | 3 cross-layer coupling issues |
| Files < 1000 LOC | ❌ FAIL | 17 files exceed 1000 LOC |
| Classes < 500 LOC | ❌ FAIL | 12 classes exceed 500 LOC |
| Functions < 100 LOC | ❌ FAIL | 58 functions exceed 100 LOC |
| Dead code minimal | ✅ PASS | Phase 23 removed dead code |
| Consistent naming | ✅ PASS | snake_case throughout |
| Package exports explicit | ✅ PASS | __init__.py re-exports |
