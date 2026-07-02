#!/usr/bin/env bash
# release_package.sh — Create a publication-ready snapshot of HELIX-IDS
# Usage: bash cleanup/release_package.sh [--dest /path/to/output]
#
# Produces a clean directory tree suitable for release/archival:
#   HELIX-IDS-release/
#   ├── src/              (production code)
#   ├── scripts/          (operational scripts)
#   ├── docs/             (documentation + paper)
#   ├── models/           (production checkpoints only)
#   ├── config/           (configuration)
#   ├── tests/            (test suite)
#   ├── datasets/         (canonical data — raw, not processed cache)
#   ├── artifacts/        (compressed research archives)
#   ├── paper/            (manuscript)
#   ├── reproducibility/  (reproducibility guide + manifests)
#   └── cleanup/          (cleanup logs + reports)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# Parse --dest flag
DEST="$REPO_DIR/releases/helix-ids-publication"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest) DEST="$2"; shift 2 ;;
        *)      echo "Unknown: $1"; exit 1 ;;
    esac
done
mkdir -p "$DEST"

echo "Building publication snapshot at: $DEST"

# --- 1. Production code ---
echo "  Copying src/ ..."
cp -r "$REPO_DIR/src" "$DEST/src"

# --- 2. Scripts ---
echo "  Copying scripts/ ..."
cp -r "$REPO_DIR/scripts" "$DEST/scripts"

# --- 3. Documentation ---
echo "  Copying docs/ ..."
cp -r "$REPO_DIR/docs" "$DEST/docs"

# --- 4. Models (production checkpoints only) ---
echo "  Copying models/ ..."
mkdir -p "$DEST/models"
cp -r "$REPO_DIR/models/helix_full" "$DEST/models/helix_full" 2>/dev/null || true
cp -r "$REPO_DIR/models/dann_production" "$DEST/models/dann_production" 2>/dev/null || true
cp -r "$REPO_DIR/models/production" "$DEST/models/production" 2>/dev/null || true

# --- 5. Config ---
echo "  Copying config/ ..."
cp -r "$REPO_DIR/config" "$DEST/config"

# --- 6. Tests ---
echo "  Copying tests/ ..."
cp -r "$REPO_DIR/tests" "$DEST/tests"

# --- 7. Datasets (raw canonical only) ---
echo "  Copying raw datasets ..."
if [ -d "$REPO_DIR/data/raw" ]; then
    mkdir -p "$DEST/datasets"
    cp -r "$REPO_DIR/data/raw" "$DEST/datasets/raw"
fi

# --- 8. Artifacts (compressed research archives) ---
echo "  Copying compressed archives ..."
if [ -d "$REPO_DIR/cleanup/archives" ]; then
    mkdir -p "$DEST/artifacts"
    cp -r "$REPO_DIR/cleanup/archives" "$DEST/artifacts/compressed"
fi

# --- 9. Paper ---
echo "  Copying manuscript ..."
mkdir -p "$DEST/paper"
cp "$REPO_DIR/docs/manuscript/HELIX_submission_ready.md" "$DEST/paper/" 2>/dev/null || true
# Copy any PDF if present
for f in "$REPO_DIR/docs/manuscript"/*.pdf; do
    [ -f "$f" ] && cp "$f" "$DEST/paper/"
done

# --- 10. Reproducibility ---
echo "  Creating reproducibility/ ..."
mkdir -p "$DEST/reproducibility"
cat > "$DEST/reproducibility/README.md" << 'REPRO'
# Reproducibility Guide — HELIX-IDS

## Quick Start
```bash
python3 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements-lock.txt
PYTHONPATH=src python scripts/training/train_helix_ids_full.py \
  --config config/helix_config.yaml \
  --output models/helix_full
```

## Verification
```bash
PYTHONPATH=src pytest -q
```

## Artifact Restoration
Compressed research archives are in `../artifacts/compressed/`.
Extract with: `tar -xJf <archive>.tar.xz -C <target>`
REPRO

cp "$REPO_DIR/requirements-lock.txt" "$DEST/reproducibility/" 2>/dev/null || true
cp "$REPO_DIR/requirements.in" "$DEST/reproducibility/" 2>/dev/null || true
cp "$REPO_DIR/pyproject.toml" "$DEST/reproducibility/" 2>/dev/null || true

# Copy SHA256 manifests and reports
if [ -d "$REPO_DIR/cleanup" ]; then
    mkdir -p "$DEST/reproducibility/cleanup-reports"
    cp "$REPO_DIR/cleanup"/*.md "$DEST/reproducibility/cleanup-reports/" 2>/dev/null || true
    cp "$REPO_DIR/cleanup"/*.csv "$DEST/reproducibility/cleanup-reports/" 2>/dev/null || true
fi

# --- 11. README ---
echo "  Creating top-level README ..."
cat > "$DEST/README.md" << 'README'
# HELIX-IDS — Hierarchical Edge-optimized Lightweight Intrusion eXpert

Network intrusion detection system for resource-constrained edge devices.

## Contents
- `src/` — Core package
- `scripts/` — Operational scripts (training, serving, evaluation)
- `docs/` — Full documentation and manuscript
- `models/` — Production checkpoints
- `tests/` — Test suite (76+ tests)
- `datasets/` — Raw canonical datasets
- `artifacts/` — Compressed research archives
- `paper/` — IEEE-format manuscript
- `reproducibility/` — Reproducibility guide

## Citation
See `paper/HELIX_submission_ready.md`
README

echo "  Removing non-essential files from snapshot..."
find "$DEST" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$DEST" -name ".DS_Store" -delete 2>/dev/null || true
find "$DEST" -name ".mypy_cache" -type d -exec rm -rf {} + 2>/dev/null || true
find "$DEST" -name ".ruff_cache" -type d -exec rm -rf {} + 2>/dev/null || true

echo ""
echo "Publication snapshot complete: $(du -sh "$DEST" | cut -f1)"
echo "  Location: $DEST"
