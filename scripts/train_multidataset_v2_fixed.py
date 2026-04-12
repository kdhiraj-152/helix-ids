#!/usr/bin/env python3
"""Backward-compatible wrapper. Canonical script moved to scripts/training/train_multidataset_v2_fixed.py."""

import os
import sys

SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

TARGET = os.path.join(SCRIPT_DIR, "training", "train_multidataset_v2_fixed.py")

if __name__ == "__main__":
    os.execv(sys.executable, [sys.executable, TARGET, *sys.argv[1:]])
else:
    from training.train_multidataset_v2_fixed import HELIXMLP5Class, ImprovedTrainer, SafeDataLoader

    __all__ = ["HELIXMLP5Class", "ImprovedTrainer", "SafeDataLoader"]
