# Phase 24A — Legacy Model & Dead-Code Elimination Audit

**Date:** 2026-06-20  
**Scope:** Entire HelixIDS codebase under `src/helix_ids/`, `scripts/`, `tests/`  
**Production Baseline:** `HelixIDS-Full` (via `src/helix_ids/models/helix_ids_full.py` / `full.py`)  
**Status:** Audit Only — NO deletions executed

---

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| Total Python files audited (src/) | 43 |
| Total Python files audited (scripts/) | 14 |
| Total Python files audited (tests/) | 56 |
| Candidates recommended for DELETE | 1 file (277 lines) |
| Candidates recommended for ARCHIVE | 4 files + 1 sub-package (10 files, ~3,531 lines) |
| Candidates recommended for KEEP | ~43 files |
| Estimated deletion savings | 277 lines / 11 KB |
| Estimated archive savings | 3,531 lines / 141 KB |
| Overall risk | LOW (see risk assessment §7) |

---

## 2. Model Audit: `src/helix_ids/models/`

### 2.1 Model-by-Model Analysis

| File | Lines | Referenced? | Imported By? | Tested? | Used In Training? | Used In Inference? | Recommendation |
|------|-------|-------------|--------------|---------|-------------------|--------------------|---------------|
| `helix_ids_full.py` | 556 | YES (heavy) | `full.py`, tests, training scripts | YES | YES (production) | YES | **KEEP** |
| `full.py` | 20 | YES | `train_helix_ids_full.py`, test_models/ | YES | YES | YES | **KEEP** |
| `core.py` | 8 | YES | `src/helix_ids/__init__.py` | NO | No (via __init__) | No | **KEEP** (public API alias) |
| `__init__.py` | 31 | YES | Package import | NO | YES | YES | **KEEP** |
| `helix_ids.py` | 551 | LIMITED | `core.py`, `test_helix_ids_unit.py`, `test_models/test_helix_ids.py` | YES | NO | NO | **ARCHIVE** |
| `classifier.py` | 632 | LIMITED | `helix_ids.py`, test_classifier.py, train_multidataset.py | YES | NO (legacy) | NO | **ARCHIVE** |
| `attention.py` | 481 | LIMITED | `helix_ids.py`, test_classifier.py | YES | NO (legacy) | NO | **ARCHIVE** |
| `loss.py` | 635 | LIMITED | `helix_ids.py`, `models/__init__.py`, test_loss_unit.py, test_models/test_loss.py | YES | NO (legacy loss) | NO | **ARCHIVE** |

### 2.2 Evidence: `helix_ids.py` (Legacy Model)

**Inbound references (src/):**
- `src/helix_ids/models/core.py` — re-exports via `from .helix_ids import HELIXIDS`
- `src/helix_ids/__init__.py` — exports `HELIXIDS` through `core.py` chain

**Inbound references (scripts/):**
- NONE. All production scripts use `full.py` → `helix_ids_full.py`

**Inbound references (tests/):**
- `tests/test_helix_ids_unit.py` — imports `HELIXIDS`, `HELIXConfig`, `HELIXEnsemble`, `HELIXNano`, `HELIXLite`, `HELIXFull`, `create_helix_model`
- `tests/test_models/test_helix_ids.py` — imports `HELIX_VARIANTS`, `HELIXIDS`, `FeatureBackbone`, `HELIXConfig`
- `tests/test_model_inference.py` — imports `create_helix_model` (only to construct model for edge inference tests)

**Production runtime usage:** ZERO. The training pipeline (`train_helix_ids_full.py`) and inference runtime (`inference_runtime.py`) both import from `full.py` → `helix_ids_full.py`.

**Verdict:** Legacy model superseded by `HelixIDSFull`. Variants (`HELIXNano`, `HELIXLite`, `HELIXFull`) were designed for edge deployment but are not used in production. The edge deployment scripts (`train_edge_models.py`) have their own model architecture definitions. However, archival preserves test reproducibility.

### 2.3 Evidence: `loss.py` (MultiTaskLoss — disconnected from production)

**Critical finding:** There are TWO independent `MultiTaskLoss` implementations:

| Implementation | Location | Lines | Used By |
|---|---|---|---|
| **Legacy MultiTaskLoss** | `loss.py:324-635` | ~311 | `helix_ids.py` only |
| **Production MultiTaskLoss** | `helix_ids_full.py:187-543` | ~356 | `HelixIDSFull` via `full.py` |

- `helix_ids_full.py` does NOT import from `loss.py` — it has its own self-contained `MultiTaskLoss`
- `full.py` re-exports the `helix_ids_full.py` version, NOT the `loss.py` version
- `loss.py` also exports `ThreatAwareFocalLoss`, `CalibrationLoss`, `FocalLoss`, `create_loss_function` — all used exclusively by `helix_ids.py`

### 2.4 Evidence: `models/adaptation/` Sub-package

| File | Lines | Inbound src/ refs | Inbound test refs | Inbound script refs |
|------|-------|-------------------|-------------------|---------------------|
| `__init__.py` | 53 | Via transfer_learning.py | Via individual test imports | Via train_unified_rebalanced.py |
| `transfer_learning.py` | 1,280 | Via `train_unified_rebalanced.py` | test_transfer_learning_da_schedule.py, test_deployment_manifest_injection.py | train_unified_rebalanced.py (legacy) |
| `dann.py` | 430 | Via transfer_learning.py | test_label_aware_da.py | — |
| `mmd_loss.py` | 294 | Via transfer_learning.py | test_mmd_loss.py | — |
| `coral_loss.py` | 212 | Via transfer_learning.py | test_coral_loss.py | — |
| `combined_da.py` | 335 | Via transfer_learning.py | test_combined_da.py | — |
| `label_aware_da.py` | 789 | Via transfer_learning.py | test_label_aware_da.py | — |

**Total: 3,393 lines** in a cohesive domain-adaptation framework. Only ONE legacy training script (`train_unified_rebalanced.py`) uses it. The 7 test files provide coverage.

**Recommendation: ARCHIVE entire sub-package.** All dependents are legacy (see training audit below).

---

## 3. Training Pipeline Audit: `scripts/training/`

| File | Lines | Production Usage? | CLI Binding? | Test Dependency? | Recommendation |
|------|-------|-------------------|-------------|-----------------|---------------|
| `train_helix_ids_full.py` | 4,605 | YES — active training pipeline | No | YES | **KEEP** |
| `prepare_canonical_artifacts.py` | 226 | YES — artifact prep | No | test_prepare_canonical_artifacts.py | **KEEP** |
| `train_edge_models.py` | 340 | YES — edge model training | cli.py: `train_edge` | NO | **KEEP** |
| `adversarial_training.py` | 261 | YES — bound via CLI | cli.py: `adversarial` | NO | **KEEP** |
| `train_multidataset.py` | 1,125 | NO (superseded by train_helix_ids_full) | cli.py: `train` | test_data_loading.py, adversarial_training.py, holdout_eval, benchmark | **ARCHIVE** (multiple dependents) |
| `train_unified_rebalanced.py` | 378 | NO | NO | test_training_direct_adaptation_eval.py | **ARCHIVE** |
| `train_unsw_only.py` | 277 | NO | NO | NO | **DELETE** |
| `_constants.py` | 21 | YES — ENGINEERED_FEATURE_NAMES | No | architecture tests | **KEEP** |

### 3.1 Legacy Training Paths Evidence

**`train_multidataset.py`** (1,125 lines, "v2 Fixed"):  
- Imports: Re-used by `adversarial_training.py` (`HELIXMLP5Class`, `SafeDataLoader`)  
- Imports: Re-used by `holdout_evaluation.py` (`ImprovedTrainer`, `SafeDataLoader`, `HELIXMLP5Class`)  
- Imports: Re-used by `benchmark_e2e.py` (`HELIXMLP5Class`, `SafeDataLoader`)  
- CLI binding: `cli.py:main():train` → `scripts.train_multidataset`  
- **This is an architectural dead end** — the production `HelixIDSFull` training is in `train_helix_ids_full.py`. The `SafeDataLoader` and `HELIXMLP5Class` inside this file are completely separate from the production data pipeline.

**`train_unified_rebalanced.py`** (378 lines):
- Direct domain-adaptation training via transfer_learning modules
- Bypasses the harmonization/probe gate stack (as noted in its own docstring)
- Only 1 test reference: `test_training_direct_adaptation_eval.py`

**`train_unsw_only.py`** (277 lines):
- Standalone UNSW-only training with anomaly filtering
- **Zero inbound references from ANY production code, test, CLI, config, or doc**
- Only self-referencing line: `"origin": "train_unsw_only"` in its own checkpoint export

---

## 4. Data Module Audit: `src/helix_ids/data/`

| File | Lines | Production Use | Status | Recommendation |
|------|-------|---------------|--------|---------------|
| `__init__.py` | 28 | Package exports | Active | **KEEP** |
| `learnability_contract.py` | 2,092 | Used by train_helix_ids_full.py | Active | **KEEP** |
| `feature_harmonization.py` | 1,158 | Used by training scripts | Active | **KEEP** |
| `multi_dataset_loader.py` | 1,576 | Used by training scripts | Active | **KEEP** |
| `feature_engineering.py` | 1,421 | Exported from __init__.py | Active | **KEEP** |
| `loader_core.py` | 531 | Used by legacy scripts | Active | **KEEP** |
| `preprocessing.py` | 396 | Exported from __init__.py | Active | **KEEP** |
| `geometric_representation_fixes.py` | 563 | Imported by loader_core.py | Active | **KEEP** |
| `dataset_config.py` | 264 | Imported by label_mapping.py | Active | **KEEP** |
| `label_mapping.py` | 263 | Imported by loader_core.py | Active | **KEEP** |
| `feature_io.py` | 246 | Imported by loader_core.py | Active | **KEEP** |
| `unified_loader.py` | 36 | Exported from __init__.py | Active | **KEEP** |
| `augmentation.py` | 885 | Public API export | Active (only test usage) | **KEEP** (public API) |
| `data_audit.py` | 590 | **ZERO production usage** | Dead | **ARCHIVE** |

### 4.1 Evidence: `data_audit.py`

- **Zero inbound src/ references**: No `src/helix_ids/` file imports it
- **Zero production script references**: No training or evaluation script imports it
- **Test-only usage**:
  - `tests/test_dataset_corruption.py` (15 reference blocks)
  - `tests/test_fuzz.py` (4 reference blocks)
  - `tests/test_property_based.py` (1 reference block)

**Verdict:** No production dependency. The `DataAudit` class performs data quality validation that was never integrated into the production data pipeline (which uses `learnability_contract.py` for contract enforcement instead).

---

## 5. Dual Adaptation Audit

The codebase has **TWO separate adaptation packages** — this is a structural anomaly.

| Package | Lines | Content | Used By |
|---------|-------|---------|---------|
| `src/helix_ids/adaptation/` | 911 | FeatureHarmonizer, OnlineFineTuner | `tests/test_feature_harmonization.py` ONLY |
| `src/helix_ids/models/adaptation/` | 3,393 | DANN, MMD, CORAL, TransferLearning | Legacy training + tests |

### 5.1 Evidence: `src/helix_ids/adaptation/` (Top-Level)

**Zero production references:**
- No script in `scripts/training/` imports from `helix_ids.adaptation`
- No module in `src/helix_ids/` imports it
- Only test file `tests/test_feature_harmonization.py` imports `FeatureHarmonizer` and `create_cross_dataset_pipeline`

**Duplicate functionality with `data/feature_harmonization.py`:**
- `adaptation/feature_harmonization.py` is 704 lines defining `FeatureHarmonizer`, `create_cross_dataset_pipeline`, `harmonize_dataset_pair`
- `data/feature_harmonization.py` is 1,158 lines defining `compute_schema_hash`, `validate_mapping`, feature order constants
- These are completely separate modules with NO shared code and different purposes

**Recommendation:** ARCHIVE `src/helix_ids/adaptation/` entirely. If cross-dataset harmonization is needed later, it should be built on top of the `data/feature_harmonization.py` pipeline, not this dead module.

---

## 6. Checkpoint Compatibility Audit

### 6.1 Production Checkpoints

Only 4 `.pt` files exist, all HelixIDS-Full variants:
- `models/helix_full/helix_full_nsl_kdd_best.pt`
- `models/helix_full/helix_full_nsl_kdd_final.pt`
- `models/helix_full/helix_full_unsw_nb15_best.pt`
- `models/helix_full/helix_full_unsw_nb15_final.pt`

### 6.2 Legacy Checkpoint References

No legacy checkpoints exist on disk that reference `helix_ids.py` model architecture or legacy loss functions. The checkpoint paths referenced in legacy scripts:
- `helix_transfer_direct_adaptation.pt` (train_unified_rebalanced.py) — NOT on disk
- `model_adversarial.pt` (adversarial_training.py) — NOT on disk
- `model_v2.pt` (train_multidataset.py) — NOT on disk

### 6.3 Migration Code

No migration scripts reference the legacy model architectures. The `prepare_canonical_artifacts.py` script is the only migration-related tool and it operates on the HelixIDS-Full pipeline exclusively.

### 6.4 Verdict

**Safe to delete/archive all legacy candidates without checkpoint compatibility risk.** No serialized artifacts depend on legacy model classes.

---

## 7. Risk Assessment

| Candidate | Risk Level | Rationale |
|-----------|-----------|-----------|
| `train_unsw_only.py` (DELETE) | **LOW** | Zero references, zero checkpoint dependency, zero test dependency |
| `train_unified_rebalanced.py` (ARCHIVE) | **LOW** | 1 test dependency — archive the test too, or convert to integration test |
| `src/helix_ids/adaptation/` (ARCHIVE) | **LOW** | 1 test dependency — archive or update test |
| `models/adaptation/` (ARCHIVE) | **MEDIUM** | 7 test files + 1 legacy script depend on it. Archiving requires coordinating with those |
| `loss.py` (ARCHIVE) | **MEDIUM** | Legacy losses used by `helix_ids.py` + 2 test files. Production `MultiTaskLoss` is independent |
| `helix_ids.py` + `classifier.py` + `attention.py` (ARCHIVE) | **MEDIUM** | Legacy model triad. 4 test files depend on it. Core `__init__.py` exports it |
| `data_audit.py` (ARCHIVE) | **LOW** | 3 test files depend on it. No production code does |

---

## 8. Detailed Recommendation Matrix

### DELETE (no process change, no test breakage)
```
scripts/training/train_unsw_only.py          (277 lines, 0 references)
```

### ARCHIVE (move to docs/archive/ — preserve for history, remove from active tree)
```
src/helix_ids/adaptation/__init__.py         (16 lines)
src/helix_ids/adaptation/feature_harmonization.py  (704 lines)
src/helix_ids/adaptation/online_finetune.py  (191 lines)
src/helix_ids/data/data_audit.py             (590 lines)
scripts/training/train_multidataset.py       (1,125 lines)
scripts/training/train_unified_rebalanced.py (378 lines)
```

### ARCHIVE (conditional — requires test coordination)
```
src/helix_ids/models/helix_ids.py            (551 lines) — 4 test files depend on it
src/helix_ids/models/classifier.py           (632 lines) — depends on helix_ids.py usage
src/helix_ids/models/attention.py            (481 lines) — depends on helix_ids.py usage
src/helix_ids/models/loss.py                 (635 lines) — 2 test files
src/helix_ids/models/adaptation/             (3,393 lines, 7 files) — 7 test files
```

### KEEP (production-critical or actively used)
```
src/helix_ids/models/helix_ids_full.py       (556 lines)
src/helix_ids/models/full.py                 (20 lines)
src/helix_ids/models/__init__.py             (31 lines)
src/helix_ids/models/core.py                 (8 lines)
scripts/training/train_helix_ids_full.py     (4,605 lines)
scripts/training/train_edge_models.py        (340 lines)
scripts/training/adversarial_training.py     (261 lines)
scripts/training/_constants.py               (21 lines)
scripts/training/prepare_canonical_artifacts.py (226 lines)
src/helix_ids/data/ (13 files) — all active
```

---

## 9. Key Duplication Findings

1. **TWO MultiTaskLoss implementations**: `loss.py:324` (legacy, 311 lines) and `helix_ids_full.py:187` (production, 356 lines). They share no code and have different interfaces.

2. **TWO adaptation packages**: `src/helix_ids/adaptation/` (911 lines, dead) and `src/helix_ids/models/adaptation/` (3,393 lines, legacy). Both are unused in production.

3. **TWO feature_harmonization modules**: `adaptation/feature_harmonization.py` (704 lines, dead) has different API from `data/feature_harmonization.py` (1,158 lines, active).

4. **TWO training pipelines**: `train_multidataset.py` (legacy, 1,125 lines) and `train_helix_ids_full.py` (production, 4,605 lines). The legacy pipeline has remaining CLI bindings and downstream import dependencies.

---

## 10. Maintenance Impact

| Action | Lines Removed | Estimated File Count | KB Saved | Maintenance Reduction |
|--------|--------------|---------------------|----------|----------------------|
| DELETE train_unsw_only.py | 277 | 1 | 11 KB | Negligible |
| ARCHIVE entire legacy model triad | 1,664 | 3 | 60 KB | Moderate (no more false-positive coverage gaps) |
| ARCHIVE adaptation sub-packages | 4,304 | 10 | 152 KB | Significant (dual adaptation confusion eliminated) |
| ARCHIVE data_audit.py | 590 | 1 | 20 KB | Minor |
| **TOTAL** | **~6,835** | **~15** | **~243 KB** | **Significant** |

Production HelixIDS-Full source is ~15,000 lines. Legacy candidate removal would reduce the tree by ~31% (source lines), but the real savings is in reduced cognitive overhead from dual/dead code paths, not storage.
