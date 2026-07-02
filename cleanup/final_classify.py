#!/usr/bin/env python3
"""Final audit report generator — produces accurate KEEP/ARCHIVE/DELETE lists."""
import csv
import os
from pathlib import Path

REPO = "/Users/kdhiraj/Downloads/RP-2"
CACHE_SKIP = {".git", ".venv311", ".venv", "node_modules"}

def get_classification(rel, fp):
    """Classify a file into KEEP/ARCHIVE/DELETE with reason."""
    size = fp.stat().st_size
    
    # === Cache files (SAFE to delete) ===
    if fp.suffix in (".pyc", ".pyo"):
        return "DELETE", "Python bytecode cache — auto-generated"
    if ".mypy_cache" in rel:
        return "DELETE", "Mypy type checking cache — auto-generated"
    if ".hypothesis" in rel:
        return "DELETE", "Hypothesis test examples — auto-generated"
    if ".pytest_cache" in rel:
        return "DELETE", "Pytest cache — auto-generated"
    if ".ruff_cache" in rel:
        return "DELETE", "Ruff lint cache — auto-generated"
    if ".understand-anything/.trash" in rel:
        return "DELETE", "Understand-Anything trash — not needed"
    
    # === Duplicate checkpoints ===
    if rel == "results/phase63/checkpoint_immutable.pt":
        return "DELETE", "Identical to models/helix_full/helix_full_nsl_kdd_best.pt"
    
    # === Duplicate scaler.pkl ===
    if rel in ("models/esp32/scaler.pkl", "models/rpi_4/scaler.pkl", "models/rpi_zero/scaler.pkl"):
        return "DELETE", "Duplicate of models/production/scaler.pkl"
    
    # === Duplicate feature_names.json ===
    if rel in ("models/esp32/feature_names.json", "models/rpi_4/feature_names.json", "models/rpi_zero/feature_names.json"):
        return "DELETE", "Duplicate of models/production/feature_names.json"
    
    # === Duplicate labels (phase52_cache vs multi_dataset_v1) ===
    if rel == "data/processed/phase52_cache/nsl_kdd_y_train.npy":
        return "DELETE", "Identical to multi_dataset_v1/y_train_nsl_kdd.npy"
    if rel == "data/processed/phase52_cache/nsl_kdd_y_test.npy":
        return "DELETE", "Identical to multi_dataset_v1/y_test_nsl_kdd.npy"
    if rel == "data/processed/phase52_cache/unsw_nb15_y_train.npy":
        return "DELETE", "Identical to multi_dataset_v1/y_train_unsw_nb15.npy"
    if rel == "data/processed/phase52_cache/unsw_nb15_y_test.npy":
        return "DELETE", "Identical to multi_dataset_v1/y_test_unsw_nb15.npy"
    
    # === Duplicate CSVs in phase51 ===
    if "results/phase51/tables/failure_attribution_matrix_detail.csv" in rel:
        return "DELETE", "Duplicate of failure_attribution.csv"
    if "results/phase51/tables/pairwise_transferability_atlas.csv" in rel:
        return "DELETE", "Duplicate of all_transfer_results.csv"
    if "results/phase51/tables/predictor_ranking.csv" in rel:
        return "DELETE", "Duplicate of similarity_correlations.csv"
    if "results/phase51/tables/class_transfer_atlas.csv" in rel:
        return "DELETE", "Duplicate of class_transfer_matrix.csv"
    
    # === Duplicate labels ===
    if "results/phase55/latents/expF_dim1_labels.npz" in rel:
        return "DELETE", "Identical to expF_dim32_labels.npz"
    
    # === Duplicate pwcca ===
    if rel == "results/phase47/pwcca_matrix.csv":
        return "DELETE", "Identical to svcca_matrix.csv"
    
    # === Duplicate log ===
    if "phase59_console.log" in rel:
        return "DELETE", "Duplicate of phase59_run.log"
    
    # === Empty placeholder files ===
    if rel in ("data/processed/multi_dataset_v1/X_train_cicids.npy",
               "data/processed/multi_dataset_v1/y_train_cicids.npy",
               "data/processed/multi_dataset_v1/X_val_cicids.npy",
               "data/processed/multi_dataset_v1/y_val_cicids.npy"):
        return "DELETE", "Empty placeholder array (shape (0,) or (0,17))"
    
    # === Archive intermediate checkpoints ===
    # Phase 48 models
    if "results/phase48/models/" in rel and fp.suffix in (".pt", ".pkl"):
        return "ARCHIVE", "Intermediate model — report captures all results"
    # Phase 50 models
    if "results/phase50/models/" in rel and fp.suffix == ".pt":
        return "ARCHIVE", "Intermediate model — report captures all results"
    # Phase 52 encoders (except canonical dim32)
    if "results/phase52/latents/encoder_supcon_dim32.pt" in rel:
        return "KEEP", "Canonical optimal encoder (dim=32)"
    if "results/phase52/latents/encoder_" in rel and fp.suffix == ".pt":
        return "ARCHIVE", "Ablation encoder — optimal configuration known"
    # Phase 55 models (except best)
    if "results/phase55/models/expA_supcon.pt" in rel:
        return "KEEP", "Best SupCon encoder (Exp A)"
    if "results/phase55/models/" in rel and fp.suffix == ".pt":
        return "ARCHIVE", "Ablation checkpoint — report captures results"
    # Phase 58 models
    if "results/phase58/models/" in rel and fp.suffix == ".pt":
        return "ARCHIVE", "Intermediate model — can be retrained"
    
    # === Default: KEEP ===
    return "KEEP", "Critical or not flagged for removal"

# Walk and classify
keeps, archives, deletes = [], [], []

for root, dirs, files in os.walk(REPO):
    d = Path(root)
    skip = False
    for skip_dir in CACHE_SKIP:
        if skip_dir in d.parts:
            skip = True
            break
    if skip:
        continue
    
    for f in files:
        fp = d / f
        try:
            rel = str(fp.relative_to(REPO))
            size = fp.stat().st_size
        except (OSError, ValueError):
            continue
        
        action, reason = get_classification(rel, fp)
        
        if action == "KEEP":
            keeps.append((rel, size, reason))
        elif action == "ARCHIVE":
            archives.append((rel, size, reason))
        elif action == "DELETE":
            deletes.append((rel, size, reason))

# Write lists
for name, data in [("KEEP_LIST", keeps), ("ARCHIVE_LIST", archives), ("DELETE_LIST", deletes)]:
    out = os.path.join(REPO, f"cleanup/{name}/inventory.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "size_bytes", "reason"])
        for item in sorted(data, key=lambda x: x[0]):
            w.writerow(item)

print(f"KEEP: {len(keeps)} files ({sum(k[1] for k in keeps)/1024/1024:.0f} MB)")
print(f"ARCHIVE: {len(archives)} files ({sum(a[1] for a in archives)/1024/1024:.0f} MB)")
print(f"DELETE: {len(deletes)} files ({sum(d[1] for d in deletes)/1024/1024:.0f} MB)")
print(f"\nTotal files classified: {len(keeps) + len(archives) + len(deletes)}")
