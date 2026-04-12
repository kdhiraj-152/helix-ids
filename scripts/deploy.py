#!/usr/bin/env python3
"""Backward-compatible wrapper. Canonical script moved to scripts/deployment/deploy.py."""

import os
import sys

TARGET = os.path.join(os.path.dirname(__file__), "deployment", "deploy.py")
os.execv(sys.executable, [sys.executable, TARGET, *sys.argv[1:]])
