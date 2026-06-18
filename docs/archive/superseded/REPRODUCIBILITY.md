# Reproducibility Guide

> How a third party can reproduce all HELIX-IDS results without consulting source code.

Last updated: 2026-06-09

## Prerequisites

### Hardware
- **Minimum**: 4 CPU cores, 8 GB RAM, 10 GB free disk
- **Recommended**: NVIDIA GPU (8 GB+ VRAM), 16 GB RAM, 50 GB SSD
- **Edge targets** (for deployment): RPi 4 (4 GB), RPi Zero (512 MB), ESP32

### Software
- Ubuntu 22.04 / macOS 14+ (Windows untested)
- Python 3.11
- Git
- 10 GB free disk (for datasets + checkpoints)

## Step 1: Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd <repository-directory>

# Create virtual environment
python3.11 -m venv .venv311
source .venv311/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "import torch; import sklearn; import numpy; print('OK')"
```

## Step 2: Acquire Datasets

```bash
# Option A: Automated download
PYTHONPATH=src python scripts/data/download_datasets.py

# Option B: Manual (if download fails)
# NSL-KDD: Download from https://www.unb.ca/cic/datasets/nsl.html
# UNSW-NB15: Download from https://research.unsw.edu.au/projects/unsw-nb15-dataset
# CICIDS-2018: Download from https://www.unb.ca/cic/datasets/ids-2018.html
# Place files in data/nsl_kdd/, data/unsw_nb15/, data/cicids2018/raw/
```

**Note:** CICIDS-2018 download requires ~50 GB of raw data. The processing script (`process_cicids.py`) generates a 704 MB cleaned CSV.

## Step 3: Process Datasets

```bash
PYTHONPATH=src python scripts/data/process_nsl_kdd.py
PYTHONPATH=src python scripts/data/process_unsw_nb15.py
PYTHONPATH=src python scripts/data/process_cicids.py
PYTHONPATH=src python scripts/training/prepare_canonical_artifacts.py
```

Expected output: `data/processed/multi_dataset_v1/` with X_train/y_train/etc. `.npy` files.

## Step 4: Verify Data Integrity

```bash
# Run data pipeline tests
pytest tests/test_data/ -v

# Run schema contract tests
pytest tests/test_schema_contract.py -v
```

## Step 5: Train Models

### Single-dataset training (NSL-KDD):
```bash
PYTHONPATH=src python scripts/training/train_helix_ids_full.py     --config config/experiments/smoke.yaml     --output /tmp/helix_output     --device cpu     --epochs 50     --seed 42     --batch-size 256
```

### With GPU:
```bash
PYTHONPATH=src python scripts/training/train_helix_ids_full.py     --config config/experiments/smoke.yaml     --output /tmp/helix_output     --device cuda:0     --epochs 100     --seed 42
```

Expected training time:
- CPU: ~4 hours for 50 epochs
- GPU (A100): ~15 minutes for 50 epochs

## Step 6: Evaluate

```bash
# Holdout evaluation
PYTHONPATH=src python scripts/evaluation/holdout_evaluation_v2.py     --checkpoint /tmp/helix_output/helix_full_nsl_kdd_best.pt

# Full benchmark suite
PYTHONPATH=src python scripts/evaluation/benchmarks.py     --manifest config/experiments/smoke.yaml
```

## Step 7: Verify Artifacts

```bash
# Verify checkpoint manifest
PYTHONPATH=src python scripts/ci/verify_contract_sidecars.py     --checkpoint /tmp/helix_output/helix_full_nsl_kdd_best.pt

# Verify schema registry
PYTHONPATH=src python scripts/ci/validate_schema_registry.py

# Verify governance consistency
PYTHONPATH=src python scripts/ci/validate_governance_consistency.py
```

## Step 8: Run Tests

```bash
# Full test suite
pytest -q

# Specific test suites
pytest tests/test_models/ -v
pytest tests/test_operations/ -v
pytest tests/test_governance/ -v
```

## Step 9: Deploy

```bash
# Freeze baseline
PYTHONPATH=src python scripts/operations/freeze_baseline.py     --checkpoint /tmp/helix_output/helix_full_nsl_kdd_best.pt

# Start REST server
PYTHONPATH=src python scripts/operations/serve_rest.py     --checkpoint /tmp/helix_output/helix_full_nsl_kdd_best.pt     --port 8080
```

## Reproducibility Scorecard

| Requirement | Status | Notes |
|------------|--------|-------|
| Environment pinned | ✓ PARTIAL | requirements.txt has loose version pins |
| Deterministic training | ✓ DEPLOYED | determinism.py seeds all RNGs |
| Dataset fingerprint | ✓ DEPLOYED | SHA-256 in learnability contract |
| Data splits recorded | ✓ DEPLOYED | JSON files in data/splits/ |
| Config captured | ✓ DEPLOYED | Config hash in manifest |
| Git commit tracked | ✓ DEPLOYED | git commit in manifest |
| CICIDS acquisition automated | ✗ PARTIAL | Symlink to external storage required |
| Training environment containerized | ✗ MISSING | No Dockerfile |
| Result verification script | ✗ MISSING | No automated reproduction validation |
| Hardware configuration recorded | ✗ MISSING | Not captured automatically |
| Seed strategy documented | ✗ MISSING | Single seed default; multi-seed documented but no results |
| Cross-platform test | ✗ MISSING | Only tested on macOS/Ubuntu |
