#!/usr/bin/env python3
"""Backward-compatible wrapper. Canonical script moved to scripts/training/train_edge_models.py."""

import os
import sys

TARGET = os.path.join(os.path.dirname(__file__), "training", "train_edge_models.py")
os.execv(sys.executable, [sys.executable, TARGET, *sys.argv[1:]])
