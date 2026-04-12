#!/usr/bin/env python3
"""Backward-compatible wrapper. Canonical script moved to scripts/quantization/quantize_helix_lite.py."""

import os
import sys

TARGET = os.path.join(os.path.dirname(__file__), "quantization", "quantize_helix_lite.py")
os.execv(sys.executable, [sys.executable, TARGET, *sys.argv[1:]])
