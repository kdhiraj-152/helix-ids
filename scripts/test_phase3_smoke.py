#!/usr/bin/env python3
"""Backward-compatible wrapper. Canonical script moved to scripts/evaluation/test_phase3_smoke.py."""

import os
import sys

TARGET = os.path.join(os.path.dirname(__file__), "evaluation", "test_phase3_smoke.py")
os.execv(sys.executable, [sys.executable, TARGET, *sys.argv[1:]])
