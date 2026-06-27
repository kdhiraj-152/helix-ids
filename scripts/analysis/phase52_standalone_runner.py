#!/usr/bin/env python3
"""
Phase 52 Standalone Runner — bypasses argparse/main() SIGKILL bug.

main() crashes with SIGKILL (-9) when calling run_experiment_c, but calling
run_experiment_c directly from an import works. This runner loads data once,
calls each experiment function by direct import (no argparse, no ifmain),
and generates the final report.

Usage:
  source .venv311/bin/activate
  PYTHONPATH=src python3 scripts/analysis/phase52_standalone_runner.py
"""
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import numpy as np

# ── Setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "phase52"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase52_runner")
fh = logging.FileHandler(RESULTS_DIR / "phase52_runner.log")
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(fh)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Standalone runner starting — device={DEVICE}")

# ── Safe memory cleanup ──────────────────────────────────────────────────
def cleanup():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

# ── Import module (bypass main() ifname guard) ──────────────────────────
import importlib
pm = importlib.import_module("phase52_main")

# ── Load data once ───────────────────────────────────────────────────────
logger.info("Loading datasets...")
data_dict = pm.load_all_datasets()
logger.info(f"Loaded {len(data_dict)} datasets:")
for name, d in sorted(data_dict.items()):
    logger.info(f"  {name}: {d['X'].shape}")

experiment_results = {}

# ── Exp C — Temperature Sweep ────────────────────────────────────────────
if True:
    logger.info("─── Experiment C — Temperature Sweep ───")
    logger.info(f"Temperatures: {pm.TEMPERATURES}")
    exp_c_results = pm.run_experiment_c(data_dict)
    experiment_results["exp_c"] = exp_c_results
    cleanup()
    logger.info(f"  Exp C done: {len(exp_c_results)} configs")

# ── Exp D — Loss Weight Ablation ─────────────────────────────────────────
if True:
    logger.info("─── Experiment D — Loss Weight Ablation ───")
    logger.info(f"Weights: {pm.LOSS_WEIGHTS}")
    exp_d_results = pm.run_experiment_d(data_dict)
    experiment_results["exp_d"] = exp_d_results
    cleanup()
    logger.info(f"  Exp D done: {len(exp_d_results)} configs")

# ── Exp E — Label Noise Robustness ───────────────────────────────────────
if True:
    logger.info("─── Experiment E — Label Noise Robustness ───")
    logger.info(f"Noise rates: {pm.NOISE_RATES}")
    exp_e_results = pm.run_experiment_e(data_dict)
    experiment_results["exp_e"] = exp_e_results
    cleanup()
    logger.info(f"  Exp E done: {len(exp_e_results)} configs")

# ── Exp F — Sample Efficiency ────────────────────────────────────────────
if True:
    logger.info("─── Experiment F — Sample Efficiency ───")
    logger.info(f"Fractions: {pm.SAMPLE_FRACTIONS}")
    exp_f_results = pm.run_experiment_f(data_dict)
    experiment_results["exp_f"] = exp_f_results
    cleanup()
    logger.info(f"  Exp F done: {len(exp_f_results)} configs")

# ── Also load existing Exp A & B results ─────────────────────────────────
for exp_key, exp_func, exp_name, param_key, param_vals in [
    ("exp_a", "run_experiment_a", "Exp A", "latent_dim", None),
    ("exp_b", "run_experiment_b", "Exp B", "architecture", None),
]:
    # Check if we already have metrics files
    if exp_key == "exp_a":
        existing = sorted(RESULTS_DIR.glob("tables/expA_*_metrics.json"))
    else:
        existing = sorted(RESULTS_DIR.glob("tables/expB_*_metrics.json"))

    if existing:
        logger.info(f"Loading existing {exp_name} results from {len(existing)} files")
        results = []
        for fp in existing:
            with open(fp) as f:
                results.append(json.load(f))
        experiment_results[exp_key] = results
    else:
        logger.info(f"Running {exp_name}...")
        func = getattr(pm, exp_func)
        r = func(data_dict)
        experiment_results[exp_key] = r
        cleanup()

# ── Statistical Analysis ─────────────────────────────────────────────────
logger.info("─── Statistical Analysis ───")
stats = pm.run_statistical_analysis(experiment_results)
logger.info("  Statistical analysis done")

# ── Generate Deliverables ────────────────────────────────────────────────
logger.info("─── Generating deliverables ───")
pm.generate_deliverables(experiment_results, stats)
logger.info("  Deliverables done")

# ── Print summary ────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  PHASE 52 — DELIVERABLES GENERATED")
print("=" * 72)
for exp_key, exp_name in [
    ("exp_a", "A: Latent Dimension"),
    ("exp_b", "B: Encoder Depth"),
    ("exp_c", "C: Temperature Sweep"),
    ("exp_d", "D: Loss Weight"),
    ("exp_e", "E: Label Noise"),
    ("exp_f", "F: Sample Efficiency"),
]:
    exp_data = experiment_results.get(exp_key, [])
    n = len(exp_data)
    if n > 0:
        best = max(exp_data, key=lambda x: x.get("mean_off_diag_mf1", 0))
        bc = best.get("run_name", "?")
        bm = best.get("mean_off_diag_mf1", 0)
        print(f"  {exp_name}: {n} configs, best={bm:.4f} ({bc})")
    else:
        print(f"  {exp_name}: NO DATA")
print(f"\n  Report: {RESULTS_DIR / 'FINAL_REPORT.md'}")
print(f"  Tables: {RESULTS_DIR / 'tables/'}")
print("=" * 72)

logger.info("Phase 52 standalone runner complete")
