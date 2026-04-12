#!/usr/bin/env python3
"""Backward-compatible wrapper. Canonical script moved to scripts/evaluation/benchmark_e2e_v2_fixed.py."""

import os
import sys

TARGET = os.path.join(os.path.dirname(__file__), "evaluation", "benchmark_e2e_v2_fixed.py")
os.execv(sys.executable, [sys.executable, TARGET, *sys.argv[1:]])
