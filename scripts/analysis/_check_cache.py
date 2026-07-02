#!/usr/bin/env python3
"""Check phase52 cache and existing results."""
import numpy as np
from pathlib import Path

cache = Path("data/processed/phase52_cache")
print("=== Phase52 cache contents ===")
for f in sorted(cache.glob("*.npy")):
    arr = np.load(f, mmap_mode="r")
    print(f"  {f.name}: {arr.shape} {arr.dtype}")

print("\n=== Existing phase50 models ===")
phase50 = Path("results/phase50")
for f in sorted(phase50.glob("*.pt")):
    print(f"  {f.name}: {f.stat().st_size / 1024:.0f} KB")

print("\n=== Phase50 results ===")
for f in sorted(phase50.glob("*.csv")):
    print(f"  {f.name}: {f.stat().st_size / 1024:.0f} KB")
for f in sorted(phase50.glob("*.json")):
    print(f"  {f.name}: {f.stat().st_size / 1024:.0f} KB")
