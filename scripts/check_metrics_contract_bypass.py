#!/usr/bin/env python3
"""Backward-compatible wrapper. Canonical script moved to scripts/governance/check_metrics_contract_bypass.py."""

import os
import sys

TARGET = os.path.join(os.path.dirname(__file__), "governance", "check_metrics_contract_bypass.py")
os.execv(sys.executable, [sys.executable, TARGET, *sys.argv[1:]])
