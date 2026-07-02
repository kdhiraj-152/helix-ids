#!/usr/bin/env python3
"""Run experiments F and G only."""
import sys, os, time, gc, torch, traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import after path setup
from phase56_main import experiment_f, experiment_g

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'phase56')
os.makedirs(RESULTS_DIR, exist_ok=True)

t0 = time.time()
print(">>> Experiment F")
try:
    result_f = experiment_f()
    print(f"  OK ({time.time()-t0:.1f}s)")
except Exception as e:
    print(f"  FAILED ({time.time()-t0:.1f}s): {e}")
    traceback.print_exc()

print()

t1 = time.time()
print(">>> Experiment G")
if torch.mps.is_available():
    torch.mps.empty_cache()
gc.collect()
try:
    result_g = experiment_g()
    print(f"  OK ({time.time()-t1:.1f}s)")
except Exception as e:
    print(f"  FAILED ({time.time()-t1:.1f}s): {e}")
    traceback.print_exc()

print(f"\nDone. Total: {time.time()-t0:.1f}s")
