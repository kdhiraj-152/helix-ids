#!/usr/bin/env python3
"""Backward-compatible wrapper. Canonical script moved to scripts/analysis/quick_verify_fixes.py."""

import os
import sys

TARGET = os.path.join(os.path.dirname(__file__), "analysis", "quick_verify_fixes.py")
os.execv(sys.executable, [sys.executable, TARGET, *sys.argv[1:]])
