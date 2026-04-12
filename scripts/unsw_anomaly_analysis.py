#!/usr/bin/env python3
"""Backward-compatible wrapper. Canonical script moved to scripts/analysis/unsw_anomaly_analysis.py."""

import os
import sys

TARGET = os.path.join(os.path.dirname(__file__), "analysis", "unsw_anomaly_analysis.py")
os.execv(sys.executable, [sys.executable, TARGET, *sys.argv[1:]])
