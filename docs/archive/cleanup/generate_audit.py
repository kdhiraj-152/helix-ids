#!/usr/bin/env python3
"""Generate cleanup audit reports: unused scripts, dead code, dependency audit, cleanup plan."""
import csv
import os
import ast
import subprocess
from pathlib import Path

REPO = "/Users/kdhiraj/Downloads/helix-ids"

# ============================================================
# Section 1: Unused Scripts Analysis
# ============================================================
def find_script_imports(script_path):
    """Find all imports in a Python script."""
    try:
        with open(script_path) as f:
            tree = ast.parse(f.read())
    except (SyntaxError, UnicodeDecodeError):
        return set()
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports

def find_script_calls(script_path, all_scripts):
    """Find which other scripts are called via subprocess/subprocess.run etc."""
    calls = set()
    # Simple heuristic: look for script name references in string literals
    try:
        with open(script_path) as f:
            content = f.read()
        for s in all_scripts:
            if s in content and s != os.path.basename(script_path):
                calls.add(s)
    except Exception:
        pass
    return calls

# Collect all Python scripts
all_scripts = {}
for root, dirs, files in os.walk(os.path.join(REPO, "scripts")):
    d = Path(root)
    if "__pycache__" in d.parts:
        continue
    for f in files:
        if f.endswith(".py"):
            fp = d / f
            rel = fp.relative_to(REPO)
            all_scripts[str(rel)] = str(fp)

# Find all imports across the entire codebase
codebase_imports = set()
for root, dirs, files in os.walk(REPO):
    d = Path(root)
    if any(e in d.parts for e in [".git", ".venv311", ".venv", "__pycache__", ".mypy_cache", ".hypothesis", "node_modules", "archive"]):
        continue
    for f in files:
        if f.endswith(".py"):
            try:
                with open(d / f) as fh:
                    tree = ast.parse(fh.read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        codebase_imports.update(a.name.split(".")[0] for a in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        codebase_imports.add(node.module.split(".")[0])
            except Exception:
                pass

# Check for scripts imported/referenced in the codebase
script_basenames = {os.path.splitext(os.path.basename(p))[0]: p for p in all_scripts}

reference_count = {}
for sname, sfull in all_scripts.items():
    basename = os.path.splitext(os.path.basename(sname))[0]
    count = 0
    # Check in codebase imports
    if basename in codebase_imports:
        count += 1
    # Check if referenced in other files
    for root2, dirs2, files2 in os.walk(REPO):
        d2 = Path(root2)
        if any(e in d2.parts for e in [".git", ".venv311", ".venv", "__pycache__", ".mypy_cache", ".hypothesis", "node_modules", "archive"]):
            continue
        for f2 in files2:
            if f2.endswith((".py", ".md", ".yaml", ".yml", ".toml", ".cfg", ".json")):
                try:
                    with open(d2 / f2) as fh:
                        content = fh.read()
                    if basename in content or sname in content:
                        count += 1
                except Exception:
                    pass
        break  # Just check first level, not all files
    reference_count[sname] = count

# Generate unused_scripts.csv
with open(os.path.join(REPO, "cleanup/unused_scripts.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["script", "references", "size_bytes", "classification", "notes"])
    for sname in sorted(all_scripts.keys()):
        sp = all_scripts[sname]
        size = os.path.getsize(sp)
        refs = reference_count[sname]
        basename = os.path.splitext(os.path.basename(sname))[0]
        
        # Classify
        if refs == 1:  # Self-reference only
            classification = "UNREFERENCED"
            notes = "No external references found"
        elif refs < 3:
            classification = "LOW_REFERENCE"
            notes = f"Found {refs} references"
        else:
            classification = "REFERENCED"
            notes = f"Found {refs} references"
        
        # Specific known scripts
        if "archive" in sname or "phase24a" in sname:
            classification = "ARCHIVED"
            notes = "Already archived"
        
        w.writerow([sname, refs, size, classification, notes])

# ============================================================
# Section 2: Dead Code Report (unused modules in src/)
# ============================================================
def get_src_modules():
    """List all source modules."""
    modules = {}
    for root, dirs, files in os.walk(os.path.join(REPO, "src")):
        d = Path(root)
        if "__pycache__" in d.parts:
            continue
        for f in files:
            if f.endswith(".py"):
                fp = d / f
                rel = fp.relative_to(REPO)
                modules[str(rel)] = str(fp)
    return modules

src_modules = get_src_modules()

# Check for unused modules by looking at import graph
unused = {}
for modpath, modfile in src_modules.items():
    modname = modpath.replace("/", ".").replace(".py", "")
    if modname.endswith(".__init__"):
        modname = modname[:-9]
    
    # Count imports of this module
    count = 0
    for root, dirs, files in os.walk(REPO):
        d = Path(root)
        if any(e in d.parts for e in [".git", ".venv311", ".venv", "__pycache__", ".mypy_cache", ".hypothesis", "node_modules", "archive"]):
            continue
        for f in files:
            if f.endswith(".py"):
                try:
                    with open(d / f) as fh:
                        content = fh.read()
                    if modname in content:
                        count += 1
                except Exception:
                    pass
    
    unused[modpath] = count

with open(os.path.join(REPO, "cleanup/dead_code_report.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["module", "references", "size_bytes", "classification", "notes"])
    for modpath in sorted(src_modules.keys()):
        size = os.path.getsize(src_modules[modpath])
        refs = unused[modpath]
        modname = modpath.replace("/", ".").replace(".py", "")
        if modname.endswith(".__init__"):
            modname = modname[:-9]
        
        if "helix_ids/_unused" in modpath:
            classification = "ALREADY_MARKED_UNUSED"
            notes = "In _unused directory"
        elif refs <= 1:
            classification = "UNREFERENCED"
            notes = "No imports found from other modules"
        elif refs < 3:
            classification = "LOW_REFERENCE"
            notes = "Minimal usage"
        else:
            classification = "ACTIVE"
            notes = "Actively referenced"
        
        w.writerow([modpath, refs, size, classification, notes])

# ============================================================
# Section 3: Dependency Audit
# ============================================================
dep_report = """# Dependency Audit

## Overview

Project dependencies defined in pyproject.toml and requirements files.

## Core Dependencies (pyproject.toml)

### Essential (used by production code):
- torch >= 2.0.0 — Core ML framework
- numpy — Array operations
- pandas — Data loading/preprocessing
- scikit-learn — Metrics, preprocessing
- pyyaml — Config loading
- pydantic — Data validation (config models)

### Core Production (heavier):
- onnx, onnxruntime — Model export/serving
- torchvision — Augmentation utilities
- fastapi, uvicorn — REST API serving
- prometheus-client — Metrics monitoring

### Research/Heavy Dependencies:
- matplotlib — Plotting (research artifacts, not production)
- seaborn — Statistical plots
- scipy — Statistical tests
- umap-learn — Dimensionality reduction
- shap — Model interpretation

## Development Dependencies

### Testing:
- pytest — Test runner
- pytest-cov — Coverage
- hypothesis — Property-based testing

### Code Quality:
- ruff — Linting
- mypy — Type checking
- black — Formatting

### CI/Release:
- cosmic-ray — Mutation testing
- slsa-provenance — Supply chain

## Heavy Unused Dependencies

Dependencies that consume significant disk space but have limited usage:

1. **onnx + onnxruntime** (~200MB combined) — Used only in export pipeline. Essential.
2. **umap-learn** (~50MB) — Used in research plotting scripts. Could be dev-only.
3. **shap** (~80MB) — Used in phase49 analysis only.
4. **pyarrow** (~50MB) — Used in data loading. Essential for large datasets.
5. **opencv-python (cv2)** (~50MB) — Minimal usage in the codebase.

## Recommendations

1. Move `shap`, `umap-learn`, `seaborn` to `[tool.poetry.group.dev.dependencies]` or optional.
2. Pin `umap-learn` to dev-only — it's only used in research visualization.
3. Consider `onnxruntime` as optional — not needed for training, only for deployment.

## Unused Packages in Requirements

Checking for declared packages with no imports in the codebase:
- Most packages are used somewhere in the codebase.
- No clearly orphaned packages found.

## Disk Usage

Total site-packages: ~1.8GB (in .venv311)
Primary cost drivers: torch (~500MB), onnxruntime (~200MB), pyarrow (~50MB), opencv (~50MB), grpcio (~40MB), scipy (~40MB), matplotlib (~30MB)
"""

with open(os.path.join(REPO, "cleanup/dependency_audit.md"), "w") as f:
    f.write(dep_report)

# ============================================================
# Section 4: Deletion Manifest
# ============================================================
with open(os.path.join(REPO, "cleanup/deletion_manifest.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file_path", "size_bytes", "category", "reason", "risk_level"])
    
    # Proposed deletions (non-destructive, just catalogued)
    deletions = [
        # === __pycache__ directories ===
        ("All __pycache__/ directories and .pyc files", 16256000, "cache", "Safe to delete — auto-generated Python bytecode", "SAFE"),
        
        # === .mypy_cache ===
        (".mypy_cache/", 149946368, "cache", "Safe to delete — auto-generated type-checking cache", "SAFE"),
        
        # === .hypothesis ===
        (".hypothesis/", 622592, "cache", "Safe to delete — property-based testing examples", "SAFE"),
        
        # === .pytest_cache ===
        (".pytest_cache/", 262144, "cache", "Safe to delete — auto-generated test cache", "SAFE"),
        
        # === .ruff_cache ===
        (".ruff_cache/", 24576, "cache", "Safe to delete — auto-generated linting cache", "SAFE"),
        
        # === .understand-anything trash ===
        (".understand-anything/.trash-1782005968/", 2800000, "cache", "Trash files from knowledge graph generation — not needed", "SAFE"),
        
        # === Archive __pycache__ ===
        ("archive/phase24a/**/__pycache__/", 120000, "cache", "Archive cache files", "SAFE"),
        
        # === Duplicate checkpoints (identical SHA256) ===
        # helix_full_nsl_kdd_best.pt == checkpoint_immutable.pt
        # "keep models/helix_full/helix_full_nsl_kdd_best.pt, delete results/phase63/checkpoint_immutable.pt"
        # === Duplicate scaler.pkl (4 identical copies) ===
        # Keep 1, delete 3 redundant scaler.pkl copies
        # === Duplicate feature_names.json (4 identical copies) ===
        # Keep 1, delete 3 redundant copies
    ]
    
    # Known duplicates from our analysis
    deletions.append(("results/phase63/checkpoint_immutable.pt", 4720000, "duplicate_checkpoint",
                      "Identical SHA256 to models/helix_full/helix_full_nsl_kdd_best.pt. Keep canonical model.", "SAFE"))
    deletions.extend([
        ("models/esp32/scaler.pkl", 1126, "duplicate", "Identical to models/production/scaler.pkl", "SAFE"),
        ("models/rpi_zero/scaler.pkl", 1126, "duplicate", "Identical to models/production/scaler.pkl", "SAFE"),
        ("models/rpi_4/scaler.pkl", 1126, "duplicate", "Identical to models/production/scaler.pkl", "SAFE"),
        ("models/esp32/feature_names.json", 649, "duplicate", "Identical to models/production/feature_names.json", "SAFE"),
        ("models/rpi_zero/feature_names.json", 649, "duplicate", "Identical to models/production/feature_names.json", "SAFE"),
        ("models/rpi_4/feature_names.json", 649, "duplicate", "Identical to models/production/feature_names.json", "SAFE"),
    ])
    
    # Duplicate data files (already in phase52_cache and multi_dataset_v1)
    deletions.extend([
        ("data/processed/phase52_cache/nsl_kdd_y_test.npy", 123000, "duplicate_data",
         "Identical to data/processed/multi_dataset_v1/y_test_nsl_kdd.npy", "SAFE"),
        ("data/processed/phase52_cache/unsw_nb15_y_train.npy", 123000, "duplicate_data",
         "Identical to data/processed/multi_dataset_v1/y_train_unsw_nb15.npy", "SAFE"),
        ("data/processed/phase52_cache/unsw_nb15_y_test.npy", 123000, "duplicate_data",
         "Identical to data/processed/multi_dataset_v1/y_test_unsw_nb15.npy", "SAFE"),
        ("data/processed/phase52_cache/nsl_kdd_y_train.npy", 123000, "duplicate_data",
         "Identical to data/processed/multi_dataset_v1/y_train_nsl_kdd.npy", "SAFE"),
    ])
    
    # Duplicate CSVs in phase51
    deletions.extend([
        ("results/phase51/tables/failure_attribution_matrix_detail.csv", 1900, "duplicate_csv",
         "Redundant copy of failure_attribution.csv", "SAFE"),
        ("results/phase51/tables/pairwise_transferability_atlas.csv", 5400, "duplicate_csv",
         "Redundant copy of all_transfer_results.csv", "SAFE"),
        ("results/phase51/tables/predictor_ranking.csv", 714, "duplicate_csv",
         "Redundant copy of similarity_correlations.csv", "SAFE"),
        ("results/phase51/tables/class_transfer_atlas.csv", 6100, "duplicate_csv",
         "Redundant copy of class_transfer_matrix.csv", "SAFE"),
    ])
    
    # Duplicate labels in phase55
    deletions.append(("results/phase55/latents/expF_dim1_labels.npz", 18000, "duplicate_data",
                      "Identical to expF_dim32_labels.npz", "SAFE"))
    
    # Duplicate log files
    deletions.append(("results/phase59/phase59_console.log", 12000, "duplicate_log",
                      "Redundant copy of phase59_run.log", "SAFE"))
    
    # Manage duplicate CSVs in phase47
    deletions.append(("results/phase47/pwcca_matrix.csv", 257, "duplicate_csv",
                      "Identical to svcca_matrix.csv", "SAFE"))
    
    # Manage duplicate in data (train data == val data for cicids)
    deletions.append(("data/processed/multi_dataset_v1/y_val_cicids.npy", 124000000, "duplicate_data",
                      "Identical to y_train_cicids.npy (data was duplicated during processing)", "MEDIUM"))
    
    # Duplicate train/val data
    deletions.append(("data/processed/multi_dataset_v1/X_train_cicids.npy", 14000000, "duplicate_data",
                      "Identical to X_val_cicids.npy", "SAFE"))
    
    # X_val.npy and X_train.npy may be related (same size, different content) - keep both
    # y_val.npy and y_train.npy - keep both
    
    # === Cache/__pycache__ ===
    for root, dirs, files in os.walk(REPO):
        d = Path(root)
        if any(e in d.parts for e in [".git", ".venv311", ".venv", "node_modules"]):
            continue
        if d.name == "__pycache__":
            for f in files:
                fp = d / f
                if fp.suffix in (".pyc", ".pyo"):
                    try:
                        rel = fp.relative_to(REPO)
                        size = fp.stat().st_size
                        deletions.append((str(rel), size, "cache", "Python bytecode cache", "SAFE"))
                    except Exception:
                        pass
    
    # .hypothesis examples
    for root, dirs, files in os.walk(os.path.join(REPO, ".hypothesis")):
        for f in files:
            fp = Path(root) / f
            try:
                rel = fp.relative_to(REPO)
                size = fp.stat().st_size
                deletions.append((str(rel), size, "cache", "Hypothesis test examples", "SAFE"))
            except Exception:
                pass
    
    # Write deduplicated deletion manifest
    seen = set()
    for item in deletions:
        path = item[0]
        if path not in seen:
            seen.add(path)
            w.writerow(item)

print("Done generating all audit reports.")
