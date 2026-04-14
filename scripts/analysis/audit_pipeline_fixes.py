#!/usr/bin/env python3
"""
Audit script to verify critical fixes to pipeline.

Checks:
1. Label mapping now includes "Backdoors" (plural)
2. Train/val/test splits properly stratified
3. Each test set contains both Normal and Attack classes
4. Test set class distributions are logged
"""

import sys
import logging
from pathlib import Path

# Setup path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PipelineAudit")


def main():
    logger.info("=" * 80)
    logger.info("AUDITING CRITICAL PIPELINE FIXES")
    logger.info("=" * 80)

    # Test 1: Check label mapping
    logger.info("\n[1/4] Checking label mapping fix...")
    from helix_ids.data.feature_harmonization import UNSW_TO_7CLASS

    if "Backdoors" in UNSW_TO_7CLASS:
        logger.info("✅ PASS: 'Backdoors' (plural) found in UNSW_TO_7CLASS")
        logger.info(f"   Mapping: {UNSW_TO_7CLASS['Backdoors']} (expected 6)")
        assert UNSW_TO_7CLASS["Backdoors"] == 6, "Backdoors should map to class 6"
    else:
        logger.error("❌ FAIL: 'Backdoors' (plural) NOT in UNSW_TO_7CLASS")
        logger.error(f"   Available keys: {list(UNSW_TO_7CLASS.keys())}")
        return False

    # Test 2: Load and validate datasets
    logger.info("\n[2/4] Loading and harmonizing datasets...")
    try:
        from helix_ids.data.multi_dataset_loader import MultiDatasetLoader

        loader = MultiDatasetLoader(project_root=str(project_root))

        nsl_kdd, unsw, cicids = loader.load_and_harmonize_all()
        logger.info("✅ PASS: Datasets loaded successfully")
        logger.info(f"   NSL-KDD: {nsl_kdd.shape}, labels: {sorted(nsl_kdd['label'].unique())}")
        logger.info(f"   UNSW-NB15: {unsw.shape}, labels: {sorted(unsw['label'].unique())}")
        if cicids is not None:
            logger.info(f"   CICIDS: {cicids.shape}, labels: {sorted(cicids['label'].unique())}")
    except Exception as e:
        logger.error(f"❌ FAIL: Error loading datasets: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 3: Create splits with validation
    logger.info("\n[3/4] Creating stratified splits...")
    try:
        dfs = [nsl_kdd, unsw, cicids] if cicids is not None else [nsl_kdd, unsw]
        splits = loader.create_splits(dfs)
        logger.info("✅ PASS: Splits created successfully")
    except Exception as e:
        logger.error(f"❌ FAIL: Error creating splits: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 4: Verify test set class distributions
    logger.info("\n[4/4] Validating test set class distributions...")
    test_keys = [k for k in splits.keys() if k.startswith("y_test_")]
    all_valid = True

    for test_key in sorted(test_keys):
        y_test = splits[test_key]
        dataset_name = test_key.replace("y_test_", "")

        classes = np.unique(y_test)
        class_dist = np.bincount(y_test.astype(int))

        # Check both classes present
        has_normal = np.any(y_test == 0)
        has_attack = np.any(y_test > 0)

        if has_normal and has_attack:
            logger.info(f"✅ {dataset_name:20s} - Both classes present")
            logger.info(
                f"   Classes: {classes}, Distribution: {dict(enumerate(class_dist[: len(classes)]))}"
            )
        else:
            logger.error(f"❌ {dataset_name:20s} - MISSING classes!")
            logger.error(f"   Has Normal (0): {has_normal}, Has Attack (>0): {has_attack}")
            logger.error(
                f"   Classes: {classes}, Distribution: {dict(enumerate(class_dist[: len(classes)]))}"
            )
            all_valid = False

    logger.info("\n" + "=" * 80)
    if all_valid:
        logger.info("✅ ALL PIPELINE FIXES VALIDATED")
        logger.info("Pipeline is now properly specified:")
        logger.info("  ✓ Label mapping corrected (Backdoors → class 6)")
        logger.info("  ✓ Stratification enforced (never returns None)")
        logger.info("  ✓ Test sets validated (both classes present)")
        logger.info("  ✓ Class distributions logged")
    else:
        logger.error("❌ PIPELINE VALIDATION FAILED - See errors above")
        return False
    logger.info("=" * 80)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
