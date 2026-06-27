#!/usr/bin/env python3
"""Single-process Phase 52 runner.
Avoids MPS driver memory fragmentation by running all 6 experiments
in ONE Python process with internal cleanup between configs.
"""
import subprocess, sys
from pathlib import Path

BASE_CMD = [
    sys.executable, "-u", "scripts/analysis/phase52_main.py",
    "--skip-stats",
]

RESULTS_DIR = Path("results/phase52")
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

def exp_done(experiment_id, expected_count):
    """Check if experiment has all expected results files."""
    prefix = f"exp{experiment_id}"
    files = list(TABLES_DIR.glob(f"{prefix}_*_metrics.json"))
    return len(files) >= expected_count

# Expected counts per experiment
EXPECTED = {"A": 6, "B": 5, "C": 6, "D": 5, "E": 5, "F": 5}

# Run order: C first (most MPS-sensitive), then D, E, F, A, B
ORDER = ["C", "D", "E", "F", "A", "B"]

for exp_id in ORDER:
    expected = EXPECTED[exp_id]
    if exp_done(exp_id, expected):
        print(f"[{exp_id}] Skipping ({expected} files found)")
        continue
    
    print(f"[{exp_id}] Running {expected} configs...")
    cmd = BASE_CMD + ["--experiments", exp_id]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        actual = len(list(TABLES_DIR.glob(f"exp{exp_id}_*_metrics.json")))
        if actual >= expected:
            print(f"[{exp_id}] OK ({actual}/{expected} files)")
        else:
            print(f"[{exp_id}] PARTIAL ({actual}/{expected} files): {result.returncode}")
            print(result.stdout[-500:])
            print(result.stderr[-500:])
    else:
        print(f"[{exp_id}] FAILED (exit {result.returncode})")
        # Print last lines of output
        out = (result.stdout or "") + (result.stderr or "")
        print(out[-1000:] if out else "(no output)")
        # Try running directly to preserve MPS freshness
        print(f"[{exp_id}] Will retry with fresh MPS after others...")
        # Re-queue at end
        ORDER.append(exp_id)

# Check final state
print("\n=== Final Results ===")
for exp_id in sorted(EXPECTED.keys()):
    files = list(TABLES_DIR.glob(f"exp{exp_id}_*_metrics.json"))
    print(f"  Exp {exp_id}: {len(files)}/{EXPECTED[exp_id]} files {'OK' if len(files) >= EXPECTED[exp_id] else 'MISSING'}")

if all(len(list(TABLES_DIR.glob(f"exp{k}_*_metrics.json"))) >= v for k, v in EXPECTED.items()):
    print("\nAll experiments complete. Generating report...")
    result = subprocess.run(
        [sys.executable, "-u", "scripts/analysis/phase52_main.py",
         "--experiments", "A,B,C,D,E,F"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("Report generated successfully")
        print(result.stdout[-2000:])
    else:
        print(f"Report generation failed: {result.returncode}")
        print(result.stderr[-500:])
else:
    print(f"\nNot all experiments completed. Cannot generate report.")
