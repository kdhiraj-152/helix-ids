#!/usr/bin/env python3
"""Backward-compatible wrapper. Canonical script moved to scripts/evaluation/holdout_evaluation_v2.py."""

import os
import sys

TARGET = os.path.join(os.path.dirname(__file__), "evaluation", "holdout_evaluation_v2.py")
os.execv(sys.executable, [sys.executable, TARGET, *sys.argv[1:]])
