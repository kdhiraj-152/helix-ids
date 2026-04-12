#!/usr/bin/env python3
"""Backward-compatible wrapper. Canonical script moved to scripts/governance/parse_promotion_gate_logs.py."""

import os
import sys

TARGET = os.path.join(os.path.dirname(__file__), "governance", "parse_promotion_gate_logs.py")
os.execv(sys.executable, [sys.executable, TARGET, *sys.argv[1:]])
