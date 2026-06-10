#!/usr/bin/env python3
"""Compatibility shim for legacy import path.

This file delegates to the canonical implementation at
`scripts/training/train_multidataset_v2_fixed.py` to preserve
backwards-compatible imports used by tests and CI.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    _mod = importlib.import_module("scripts.training.train_multidataset_v2_fixed")
except Exception:  # pragma: no cover - best-effort shim
    # Fall back to direct module name if available
    _mod = importlib.import_module("train_multidataset_v2_fixed")

# Re-export common symbols expected by callers/tests
SafeDataLoader = getattr(_mod, "SafeDataLoader", None)
HELIXMLP5Class = getattr(_mod, "HELIXMLP5Class", None)
ImprovedTrainer = getattr(_mod, "ImprovedTrainer", None)
compute_class_weights = getattr(_mod, "compute_class_weights", None)

__all__ = [name for name in ("SafeDataLoader", "HELIXMLP5Class", "ImprovedTrainer", "compute_class_weights") if globals().get(name) is not None]
