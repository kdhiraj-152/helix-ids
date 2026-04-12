#!/usr/bin/env python3
"""Backward-compatible wrapper. Canonical script moved to scripts/data/process_cicids.py."""

import os
import sys

TARGET = os.path.join(os.path.dirname(__file__), "data", "process_cicids.py")
os.execv(sys.executable, [sys.executable, TARGET, *sys.argv[1:]])
