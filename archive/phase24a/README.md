# Phase 24A Archive

**Date archived:** 2026-06-20  
**Reason:** Legacy model cleanup — HelixIDS-Full is the sole production-certified model.

This directory contains implementation files and scripts that were part of the active codebase during earlier phases but have been superseded by the current production architecture.

## Contents

### `code/helix_ids/adaptation/` (3 files, 911 lines)
Top-level adaptation package containing:
- `feature_harmonization.py` — Early cross-dataset feature harmonization (superseded by `src/helix_ids/data/feature_harmonization.py`)
- `online_finetune.py` — Online fine-tuning for domain shift (superseded by production inference pipeline)
- Both were only referenced by `tests/test_feature_harmonization.py`

### `code/helix_ids/data/data_audit.py` (590 lines)
- Dataset quality auditing module
- Only referenced by `tests/test_dataset_corruption.py`
- Not used in any production training, inference, or evaluation pipeline

### `scripts/training/train_unified_rebalanced.py` (378 lines)
- Direct domain adaptation training entrypoint
- Only referenced by `tests/test_training_direct_adaptation_eval.py`
- Production equivalent: `scripts/training/train_helix_ids_full.py`

### `scripts/training/train_unsw_only.py` (277 lines) — DELETED
- Standalone UNSW-NB15 training script
- Zero inbound references from any code, test, config, or doc
- Safely removed per Phase 24A audit recommendation

## Status

| Item | Status | Lines |
|------|--------|-------|
| `adaptation/` | Archived | 911 |
| `data_audit.py` | Archived | 590 |
| `train_unified_rebalanced.py` | Archived | 378 |
| `train_unsw_only.py` | Deleted | 277 |

**Total moved to archive:** 1,879 lines across 5 files  
**Total deleted:** 277 lines across 1 file

These components are preserved for historical reference and backward compatibility of test references. They are **not** part of the active production system.
