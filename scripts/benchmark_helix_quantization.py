#!/usr/bin/env python3
"""Backward-compatible wrapper. Canonical script moved to scripts/quantization/benchmark_helix_quantization.py."""

import os
import sys

TARGET = os.path.join(os.path.dirname(__file__), "quantization", "benchmark_helix_quantization.py")
os.execv(sys.executable, [sys.executable, TARGET, *sys.argv[1:]])
