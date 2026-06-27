# Phase 24A — Archive List

Files worth preserving historically but not needed operationally.

---

## Summary

| # | Path | Lines | Category | Tests Affected | Priority |
|---|------|-------|----------|---------------|----------|
| 1 | `src/helix_ids/models/helix_ids.py` | 551 | Legacy Model | 4 test files | MEDIUM |
| 2 | `src/helix_ids/models/classifier.py` | 632 | Legacy Component | 1 test file | MEDIUM |
| 3 | `src/helix_ids/models/attention.py` | 481 | Legacy Component | 1 test file | MEDIUM |
| 4 | `src/helix_ids/models/loss.py` | 635 | Legacy Loss | 2 test files | MEDIUM |
| 5 | `src/helix_ids/adaptation/__init__.py` | 16 | Dead Adaptation | 1 test file | LOW |
| 6 | `src/helix_ids/adaptation/feature_harmonization.py` | 704 | Dead Adaptation | 1 test file | LOW |
| 7 | `src/helix_ids/adaptation/online_finetune.py` | 191 | Dead Adaptation | 1 test file | LOW |
| 8 | `src/helix_ids/models/adaptation/` (7 files) | 3,393 | Legacy Domain Adaptation | 7 test files | HIGH |
| 9 | `src/helix_ids/data/data_audit.py` | 590 | Dead Data Audit | 3 test files | LOW |
| 10 | `scripts/training/train_multidataset.py` | 1,125 | Superseded Pipeline | 4 scripts + tests | HIGH |
| 11 | `scripts/training/train_unified_rebalanced.py` | 378 | Superseded Pipeline | 1 test file | LOW |

---

## Detailed Entries

### 1. `src/helix_ids/models/helix_ids.py` (551 lines)

**Category:** Legacy Model  
**Variant:** Original `HELIXIDS` with `HELIXNano`, `HELIXLite`, `HELIXFull` subclasses  
**Superseded by:** `helix_ids_full.py` → `HelixIDSFull`  
**Dependents:**
- `src/helix_ids/models/core.py` (re-exports)
- `src/helix_ids/__init__.py` (public API exports through core chain)
- `tests/test_helix_ids_unit.py` (imports model class + variants)
- `tests/test_models/test_helix_ids.py` (imports model class)
- `tests/test_model_inference.py` (imports `create_helix_model` for edge tests)
- `tests/test_classifier.py` (imports classifier components used by helix_ids.py)

**Checkpoint dependency:** None — no checkpoints on disk use `helix_ids.HELIXIDS`  
**Archive impact:** Breaks 4 test files + public `__init__.py` exports  
**Recommended action before archiving:** Migrate tests to `HelixIDSFull`, remove `core.py` re-exports and `__init__.py` exports of legacy variants

### 2. `src/helix_ids/models/classifier.py` (632 lines)

**Category:** Legacy Component  
**Contains:** `HierarchicalClassifier`, `ClassifierConfig`, factory functions  
**Superseded by:** `HelixIDSFull` has its own classifier head in `helix_ids_full.py`  
**Dependents:**
- `src/helix_ids/models/helix_ids.py` (uses `HierarchicalClassifier`)
- `tests/test_classifier.py` (full test coverage)
- `scripts/training/train_multidataset.py` (uses `HELIXMLP5Class` — a separate local class, not from classifier.py)

**Note:** The `HELIXMLP5Class` in `train_multidataset.py` is a local class defined IN that file (line 451), not imported from `classifier.py`. So `classifier.py` is only used by `helix_ids.py` and tests.

### 3. `src/helix_ids/models/attention.py` (481 lines)

**Category:** Legacy Component  
**Contains:** `TemporalAttentionModule`, `create_tam`  
**Superseded by:** HelixIDS-Full uses feed-forward architecture with no temporal attention  
**Dependents:**
- `src/helix_ids/models/helix_ids.py` (imports temporal attention)
- `tests/test_classifier.py` (references in test hierarchy)

### 4. `src/helix_ids/models/loss.py` (635 lines)

**Category:** Legacy Loss Functions  
**Contains:** `MultiTaskLoss` (legacy), `ThreatAwareFocalLoss`, `CalibrationLoss`, `FocalLoss`, `create_loss_function`  
**Superseded by:** `helix_ids_full.py:MultiTaskLoss` (self-contained, independent implementation)  
**Dependents:**
- `src/helix_ids/models/helix_ids.py` (imports `MultiTaskLoss`, `create_loss_function`)
- `src/helix_ids/models/__init__.py` (exports `MultiTaskLoss`, `ThreatAwareFocalLoss`)
- `tests/test_models/test_loss.py` (imports from `loss.py`)
- `tests/test_loss_unit.py` (imports from `loss.py`)

**Key finding:** The PRODUCTION `MultiTaskLoss` in `helix_ids_full.py:187` is a SEPARATE IMPLEMENTATION from the legacy one in `loss.py:324`. The production one has 30 constructor parameters, family margin loss, class-4 penalties, feature separation, entropy regularization. The legacy one has curriculum learning schedule, 4-tasks (binary/family/fine/calibration), 4-phases.

### 5-7. `src/helix_ids/adaptation/*.py` (3 files, 911 lines total)

**Category:** Dead Adaptation Package  
**Contains:** `FeatureHarmonizer`, `create_cross_dataset_pipeline`, `harmonize_dataset_pair`, `OnlineFineTuner`, `quick_calibrate`  
**Dependents:**
- `tests/test_feature_harmonization.py` only

**Note:** This is a SEPARATE package from `models/adaptation/`. The `data/feature_harmonization.py` (1,158 lines) is the active feature harmonization system used by training scripts. This top-level `adaptation/` package is completely disconnected.

### 8. `src/helix_ids/models/adaptation/` (7 files, 3,393 lines)

**Category:** Legacy Domain Adaptation Framework  
**Contains:** `DANN`, `MMD`, `CORAL`, `LabelAwareDA`, `TransferLearning`, `CombinedDA`  
**Dependents (tests):**
- `tests/test_mmd_loss.py`
- `tests/test_coral_loss.py`
- `tests/test_combined_da.py`
- `tests/test_label_aware_da.py`
- `tests/test_transfer_learning_da_schedule.py`
- `tests/test_operations/test_deployment_manifest_injection.py` (imports from `transfer_learning`)
- `tests/test_training_direct_adaptation_eval.py` (indirect via `train_unified_rebalanced.py`)

**Dependents (scripts):**
- `scripts/training/train_unified_rebalanced.py` (imports `MultiDatasetPretrainer`, `TransferLearningConfig`)

**Dependents (src/):**
- Internal — `transfer_learning.py` imports from sibling modules in this package

**Note:** This is a large, cohesive framework. Archiving requires simultaneous archiving of all 7 files plus the dependent test files (or converting them to legacy-reference tests).

### 9. `src/helix_ids/data/data_audit.py` (590 lines)

**Category:** Dead Data Audit  
**Contains:** `DataAudit`, `DataAuditConfig` — data quality validation  
**Dependents:**
- `tests/test_dataset_corruption.py` (15 reference blocks)
- `tests/test_fuzz.py` (4 reference blocks)
- `tests/test_property_based.py` (1 reference block)

**Note:** The production data pipeline enforces quality via `learnability_contract.py` (schema hashes, contract checks, preprocessing thresholds). The `DataAudit` class was an earlier attempt at quality validation that was never integrated into production.

### 10. `scripts/training/train_multidataset.py` (1,125 lines)

**Category:** Superseded Training Pipeline  
**Variant:** "v2 Fixed"  
**Contains:** `SafeDataLoader`, `HELIXMLP5Class`, `ImprovedTrainer` — a completely separate training pipeline from production  
**Superseded by:** `train_helix_ids_full.py` (multi-dataset, contract-governed training)  
**Dependents:**
- `src/helix_ids/cli.py` — bound to `cli train` command
- `scripts/training/adversarial_training.py` — imports `HELIXMLP5Class`, `SafeDataLoader`
- `scripts/evaluation/holdout_evaluation.py` — imports `ImprovedTrainer`, `SafeDataLoader`, `HELIXMLP5Class`
- `scripts/evaluation/benchmark_e2e.py` — imports `HELIXMLP5Class`, `SafeDataLoader`
- `tests/test_data_loading.py` — imports `SafeDataLoader`

**Archive complexity:** HIGH — 4 downstream dependents including CLI entry points. Requires migration of CLI bindings to production pipeline.

### 11. `scripts/training/train_unified_rebalanced.py` (378 lines)

**Category:** Superseded Training Pipeline  
**Approach:** Direct domain-adaptation training bypassing harmonization stack  
**Dependents:**
- `tests/test_training_direct_adaptation_eval.py` (imports as `runner`)

**Note:** This script's own docstring says it "intentionally bypasses the harmonization/probe gate stack" — it was an experimental training path that is now a maintenance liability.

---

## Archive Procedure Recommendation

### Phase 1 (LOW risk, no test breakage)
1. `src/helix_ids/adaptation/` — move to `docs/archive/phase24a/src/helix_ids/adaptation/`
2. `src/helix_ids/data/data_audit.py` — move to `docs/archive/phase24a/src/helix_ids/data/`
3. `scripts/training/train_unified_rebalanced.py` — move to `docs/archive/phase24a/scripts/training/`

### Phase 2 (MEDIUM risk — requires test coordination)
4. `src/helix_ids/models/adaptation/` — archive all 7 files (affects 7 test files)
5. `src/helix_ids/models/helix_ids.py` + `classifier.py` + `attention.py` + `loss.py` — archive legacy model triad + losses (affects 4 test files)
6. `scripts/training/train_multidataset.py` — requires migrating 4 dependents (affects `cli.py`, eval scripts, tests)

### Test Migration Strategy
For archived module tests, options in order of preference:
1. **Convert to legacy-reference integration tests** — update imports to point to archive location; run once for coverage, not for regression
2. **Remove redundant tests** — if coverage is already duplicated in `test_models/test_helix_full.py` or production tests
3. **Keep as-is with ARCHIVE note** — acceptable for test-only dependencies if maintenance cost is low
