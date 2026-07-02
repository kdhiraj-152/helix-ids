#!/usr/bin/env python3
"""Generate repository inventory CSVs."""
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

REPO = "/Users/kdhiraj/Downloads/helix-ids"

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

EXCLUDE_DIRS = {".git", ".venv311", ".venv", "node_modules", "__pycache__", ".mypy_cache", ".hypothesis"}

def should_exclude(p):
    parts = p.resolve().parts
    for e in EXCLUDE_DIRS:
        if e in parts:
            return True
    # Exclude .pyc files
    if p.suffix == ".pyc":
        return True
    return False

# Collect all files
all_files = []
for root, dirs, files in os.walk(REPO):
    root_p = Path(root)
    # Skip excluded dirs
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".venv")]
    if any(e in root_p.parts for e in EXCLUDE_DIRS):
        continue
    for f in files:
        fp = root_p / f
        if fp.suffix == ".pyc":
            continue
        rel = fp.relative_to(REPO)
        try:
            size = fp.stat().st_size
        except OSError:
            continue
        ext = fp.suffix.lower()
        all_files.append({
            "path": str(rel),
            "size": size,
            "size_kb": round(size / 1024, 1),
            "ext": ext,
            "abs": str(fp),
        })

# --- repository_inventory.csv ---
with open(os.path.join(REPO, "cleanup/repository_inventory.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["path", "size_bytes", "size_kb", "extension", "category"])
    cats = {
        "py": "source",
        "md": "documentation",
        "csv": "data",
        "json": "data",
        "yaml": "config",
        "yml": "config",
        "toml": "config",
        "cfg": "config",
        "ini": "config",
        "npy": "data",
        "npz": "data",
        "pt": "model",
        "pth": "model",
        "pkl": "model",
        "png": "figure",
        "jpg": "figure",
        "jpeg": "figure",
        "pdf": "document",
        "html": "document",
        "tex": "document",
        "bib": "document",
        "txt": "text",
        "log": "log",
        "sh": "script",
        "pyx": "source",
        "pxd": "source",
        "zip": "archive",
        "gz": "archive",
        "tar": "archive",
        "": "other",
    }
    for af in sorted(all_files, key=lambda x: x["path"]):
        cat = cats.get(af["ext"].lstrip("."), "other")
        w.writerow([af["path"], af["size"], af["size_kb"], af["ext"], cat])

print(f"Total files in inventory: {len(all_files)}")

# --- folder_size_report.csv ---
from collections import defaultdict
folder_sizes = defaultdict(lambda: {"files": 0, "bytes": 0})
for af in all_files:
    p = Path(af["path"])
    folder = str(p.parent) if str(p.parent) != "." else "(root)"
    folder_sizes[folder]["files"] += 1
    folder_sizes[folder]["bytes"] += af["size"]

with open(os.path.join(REPO, "cleanup/folder_size_report.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["folder", "file_count", "size_bytes", "size_kb", "size_mb"])
    for folder in sorted(folder_sizes.keys()):
        info = folder_sizes[folder]
        w.writerow([folder, info["files"], info["bytes"],
                     round(info["bytes"]/1024, 1),
                     round(info["bytes"]/1024/1024, 2)])

# --- largest_files.csv ---
sorted_by_size = sorted(all_files, key=lambda x: x["size"], reverse=True)
with open(os.path.join(REPO, "cleanup/largest_files.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["rank", "path", "size_bytes", "size_kb", "size_mb", "extension"])
    for i, af in enumerate(sorted_by_size[:200], 1):
        w.writerow([i, af["path"], af["size"], af["size_kb"],
                     round(af["size"]/1024/1024, 2), af["ext"]])

# --- duplicate_files.csv ---
# SHA256 all non-cache files
print("Computing SHA256 for duplicate detection...")
sha_map = defaultdict(list)
for af in all_files:
    try:
        h = sha256_file(af["abs"])
        sha_map[h].append(af["path"])
    except Exception:
        pass

with open(os.path.join(REPO, "cleanup/duplicate_files.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sha256", "file_count", "total_size_bytes", "paths"])
    for h, paths in sorted(sha_map.items(), key=lambda x: -len(x[1])):
        if len(paths) > 1:
            total = sum(next(af["size"] for af in all_files if af["path"] == p) for p in paths)
            w.writerow([h, len(paths), total, "; ".join(paths)])

# --- duplicate_checkpoints.csv ---
checkpoint_exts = {".pt", ".pth", ".pkl"}
checkpoints = [af for af in all_files if af["ext"] in checkpoint_exts]
cp_sha = defaultdict(list)
for af in checkpoints:
    try:
        h = sha256_file(af["abs"])
        cp_sha[h].append(af["path"])
    except Exception:
        pass

with open(os.path.join(REPO, "cleanup/duplicate_checkpoints.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sha256", "file_count", "total_size_bytes", "paths"])
    for h, paths in sorted(cp_sha.items(), key=lambda x: -len(x[1])):
        if len(paths) > 1:
            total = sum(next(af["size"] for af in all_files if af["path"] == p) for p in paths)
            w.writerow([h, len(paths), total, "; ".join(paths)])

print("Done generating inventory CSVs.")

# Print summary
total_size = sum(af["size"] for af in all_files)
print(f"\nTotal tracked files (excluding .git, .venv, caches): {len(all_files)}")
print(f"Total size: {total_size:,} bytes ({total_size/1024/1024/1024:.2f} GB)")
