#!/usr/bin/env bash
# =============================================================================
# Phase 24A — DELETE SCRIPT
# Generated: 2026-06-20
# Status: AUDIT ONLY — DO NOT EXECUTE
# This script lists all commands that WOULD be executed after SAFE_DELETE_LIST
# and ARCHIVE_LIST are approved.
# =============================================================================
# WARNING: This file is GENERATED for review. It MUST be reviewed and approved
# before any deletion occurs. Verification gates must pass first.
# =============================================================================

set -euo pipefail

echo "=== Phase 24A — DELETE & ARCHIVE COMMANDS ==="
echo ""

# =============================================================================
# SECTION 1: SAFE DELETE (zero dependencies — no coordination needed)
# =============================================================================
echo "--- SECTION 1: SAFE DELETE ---"
echo ""

# train_unsw_only.py — 277 lines, 0 imports, 0 references, 0 checkpoint dependency
echo "# scripts/training/train_unsw_only.py — 277 lines (SAFE DELETE)"
echo "rm scripts/training/train_unsw_only.py"
echo ""

# =============================================================================
# SECTION 2: ARCHIVE — LOW RISK (test-only dependencies, no production impact)
# =============================================================================
echo "--- SECTION 2: ARCHIVE (LOW RISK) ---"
echo ""

echo "# src/helix_ids/adaptation/ — 3 files, 911 lines (ARCHIVE: dead adaptation pkg)"
echo "mkdir -p docs/archive/phase24a/src/helix_ids/adaptation/"
echo "mv src/helix_ids/adaptation/__init__.py docs/archive/phase24a/src/helix_ids/adaptation/"
echo "mv src/helix_ids/adaptation/feature_harmonization.py docs/archive/phase24a/src/helix_ids/adaptation/"
echo "mv src/helix_ids/adaptation/online_finetune.py docs/archive/phase24a/src/helix_ids/adaptation/"
echo "rmdir src/helix_ids/adaptation/"
echo ""

echo "# src/helix_ids/data/data_audit.py — 590 lines (ARCHIVE: test-only)"
echo "mkdir -p docs/archive/phase24a/src/helix_ids/data/"
echo "mv src/helix_ids/data/data_audit.py docs/archive/phase24a/src/helix_ids/data/"
echo ""

echo "# scripts/training/train_unified_rebalanced.py — 378 lines (ARCHIVE: legacy path)"
echo "mkdir -p docs/archive/phase24a/scripts/training/"
echo "mv scripts/training/train_unified_rebalanced.py docs/archive/phase24a/scripts/training/"
echo ""

# =============================================================================
# SECTION 3: ARCHIVE — MEDIUM RISK (requires test coordination)
# =============================================================================
echo "--- SECTION 3: ARCHIVE (MEDIUM RISK — requires test coordination) ---"
echo ""

echo "# src/helix_ids/models/adaptation/ — 7 files, 3,393 lines (ARCHIVE: legacy DA framework)"
echo "# AFFECTS: test_mmd_loss.py, test_coral_loss.py, test_combined_da.py,"
echo "#          test_label_aware_da.py, test_transfer_learning_da_schedule.py,"
echo "#          test_deployment_manifest_injection.py, test_training_direct_adaptation_eval.py"
echo "mkdir -p docs/archive/phase24a/src/helix_ids/models/"
echo "mv src/helix_ids/models/adaptation/ docs/archive/phase24a/src/helix_ids/models/"
echo ""

echo "# src/helix_ids/models/helix_ids.py — 551 lines (ARCHIVE: legacy model)"
echo "# AFFECTS: test_helix_ids_unit.py, test_models/test_helix_ids.py,"
echo "#          test_model_inference.py, test_classifier.py"
echo "# ALSO: must update core.py, models/__init__.py, src/__init__.py"
echo "mkdir -p docs/archive/phase24a/src/helix_ids/models/"
echo "# Step 1: remove legacy exports from core.py and __init__.py"
echo "# Step 2: remove helix_ids.py from models/__init__.py export chain"
echo "# Step 3: mv src/helix_ids/models/helix_ids.py docs/archive/phase24a/src/helix_ids/models/"
echo ""

echo "# src/helix_ids/models/classifier.py — 632 lines (ARCHIVE: component)"
echo "mv src/helix_ids/models/classifier.py docs/archive/phase24a/src/helix_ids/models/"
echo ""

echo "# src/helix_ids/models/attention.py — 481 lines (ARCHIVE: component)"
echo "mv src/helix_ids/models/attention.py docs/archive/phase24a/src/helix_ids/models/"
echo ""

echo "# src/helix_ids/models/loss.py — 635 lines (ARCHIVE: legacy losses)"
echo "# AFFECTS: helix_ids.py (already archived), test_models/test_loss.py, test_loss_unit.py"
echo "# ALSO: removes MultiTaskLoss export from models/__init__.py"
echo "mv src/helix_ids/models/loss.py docs/archive/phase24a/src/helix_ids/models/"
echo ""

echo "# scripts/training/train_multidataset.py — 1,125 lines (ARCHIVE: superseded pipeline)"
echo "# AFFECTS: cli.py train command, adversarial_training.py,"
echo "#          holdout_evaluation.py, benchmark_e2e.py, test_data_loading.py"
echo "# REQUIRES: CLI rewire + dependent migration"
echo "mkdir -p docs/archive/phase24a/scripts/training/"
echo "mv scripts/training/train_multidataset.py docs/archive/phase24a/scripts/training/"
echo ""

# =============================================================================
# SUMMARY
# =============================================================================
echo "============================================"
echo "SUMMARY OF ALL PROPOSED OPERATIONS"
echo "============================================"
echo ""
echo "DELETE (safe, no tests affected):"
echo "  1 file — scripts/training/train_unsw_only.py (277 lines)"
echo ""
echo "ARCHIVE LOW (test-only dep, no prod effect):"
echo "  5 files — adaptation/ (3) + data_audit.py + train_unified_rebalanced.py (1,879 lines)"
echo ""
echo "ARCHIVE MEDIUM (test coordination required):"
echo "  10 files — models/adaptation/ (7) + legacy model triad + loss.py + train_multidataset.py"
echo "             (6,817 lines)"
echo ""
echo "TOTAL LINES AFFECTED: 8,973"
echo "TOTAL FILES AFFECTED: 16 (1 delete + 15 archive)"
echo ""
echo "============================================"
echo "END — DO NOT EXECUTE WITHOUT APPROVAL"
echo "============================================"
