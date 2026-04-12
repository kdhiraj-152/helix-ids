#!/usr/bin/env python3
"""Backward-compatible wrapper. Canonical script moved to scripts/maintenance/create_test_checkpoint.py."""

import os
import sys

TARGET = os.path.join(os.path.dirname(__file__), "maintenance", "create_test_checkpoint.py")
os.execv(sys.executable, [sys.executable, TARGET, *sys.argv[1:]])
