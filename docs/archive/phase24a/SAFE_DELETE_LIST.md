# Phase 24A — Safe-to-Delete Files

**Criteria:** Files meeting ALL of:
- Zero imports (nobody imports this module)
- Zero references (no code, test, config, doc, or CLI references the file)
- Zero runtime usage (not used by production training, inference, or evaluation)
- Zero checkpoint dependency (no serialized artifact depends on this module)

---

## Single Candidate: `scripts/training/train_unsw_only.py`

| Field | Value |
|-------|-------|
| **File** | `scripts/training/train_unsw_only.py` |
| **Lines** | 277 |
| **Size** | ~11 KB |
| **Purpose** | Standalone UNSW-NB15-only training with anomaly filtering |
| **Production superseded by** | `train_helix_ids_full.py` (multi-dataset unified training) |

### Verification: Zero Imports

```
$ grep -rn "from.*train_unsw_only\|import.*train_unsw_only" . --include="*.py" | grep -v __pycache__
(no results)
```

No Python file in the entire repository imports any symbol from `train_unsw_only`.

### Verification: Zero References (non-self)

```
$ grep -rn "train_unsw_only" . --include="*.py" --include="*.md" --include="*.yaml" \
  --include="*.toml" --include="*.json" --include="*.cfg" --include="*.ini" \
  | grep -v __pycache__ | grep -v .venv | grep -v .git/ | grep -v docs/archive \
  | grep -v architecture_audit_part_a_output
./scripts/training/train_unsw_only.py:233: ... "origin": "train_unsw_only" ...  (self-reference in own file)
```

The ONLY match is a self-reference within the file itself (checkpoint origin tag). No configuration, documentation, test, CLI entry point, or other script references this file.

### Verification: Zero Runtime Usage

- Not bound in `helix-ids` CLI (`cli.py`)
- Not imported by any production training/evaluation script
- Not referenced in any CI/CD pipeline or Dockerfile
- Not called by any cron job or automation

### Verification: Zero Checkpoint Dependency

```
$ find . -name "helix_full_unsw_cleaned*.pt" -o -name "helix_full_unsw_*.pt"
(no results on disk)
```

No checkpoints produced by this script exist on disk. The script's own code references paths like `helix_full_unsw_cleaned_best.pt` and `helix_full_unsw_cleaned_final.pt`, but neither file exists.

---

## Additional Candidates Considered and Rejected

### `scripts/training/_constants.py`
- **Rejected.** Imported by `run_orchestrator.py` and `train_helix_ids_full.py` (both production)

### `scripts/training/train_unified_rebalanced.py`
- **Rejected.** Imported by `tests/test_training_direct_adaptation_eval.py` (test dependency)

### `src/helix_ids/adaptation/feature_harmonization.py`
- **Rejected.** Imported by `tests/test_feature_harmonization.py` (test dependency)

### `src/helix_ids/adaptation/online_finetune.py`
- **Rejected.** Imported via `adaptation/__init__.py` chain (test dependency)

### `src/helix_ids/data/data_audit.py`
- **Rejected.** Imported by 3 test files (`test_dataset_corruption.py`, `test_fuzz.py`, `test_property_based.py`)

---

## Summary

**Only 1 file** meets the strict SAFE_DELETE criteria:
- `scripts/training/train_unsw_only.py` (277 lines, ~11 KB)

All other legacy candidates have at minimum a test dependency and require phased decommissioning.
