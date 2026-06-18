# Experimental Setup

Last updated: 2026-06-09

## Hardware Configuration

### Training Server

| Component | Specification |
|-----------|---------------|
| CPU | (Not documented — varies by experiment) |
| GPU | (Not documented — varies by experiment) |
| RAM | (Not documented — varies by experiment) |
| Storage | (Not documented — varies by experiment) |

### Deployment Targets

| Target | CPU | RAM | Storage | Network |
|--------|-----|-----|---------|---------|
| RPi 4 | Cortex-A72, 4 cores @ 1.5GHz | 4 GB | 32 GB SD | Gigabit Ethernet |
| RPi Zero | ARM11, 1 core @ 1GHz | 512 MB | 8 GB SD | 802.11n WiFi |
| ESP32 | Xtensa LX6, 2 cores @ 240MHz | 520 KB SRAM | 4 MB Flash | WiFi/BT |

### Server Benchmarks (Reference)

(NOT YET DOCUMENTED — see docs/BENCHMARK_PROTOCOL.md for procedure)

## Software Environment

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.11 | Virtual environment at `.venv311/` |
| PyTorch | ≥2.0.0 | CPU/CUDA deployment |
| scikit-learn | ≥1.3.0 | Metrics and preprocessing |
| NumPy | ≥1.24.0 | Array operations |
| CUDA | (Optional) | For GPU training |

Full environment: `requirements.txt` (16 packages)

## Seeds

| Seed | Purpose |
|------|---------|
| 42 | Default training seed |
| (Varies) | Multi-seed consensus runs (3 seeds required) |

Seeds control: Python RNG, NumPy RNG, PyTorch RNG, CUDA RNG, DataLoader worker RNG.

Implementation: `governance/determinism.py` → `set_global_determinism(seed)`

## Dataset Versions

| Dataset | Version Identifier | Location | Provenance |
|---------|-------------------|----------|------------|
| NSL-KDD | v1 (processed 2024) | `data/nsl_kdd/train.csv`, `data/nsl_kdd/test.csv` | SHA-256 via learnability contract |
| UNSW-NB15 | v1 (processed 2024) | `data/unsw_nb15/train.csv`, `data/unsw_nb15/test.csv` | SHA-256 via learnability contract |
| CICIDS-2018 | v1 (processed 2024) | `data/processed/cicids2018_cleaned.csv` | SHA-256 via learnability contract |

All three datasets are harmonized into a single canonical version at `data/processed/multi_dataset_v1/`.

## Training Settings

### Standard Configuration

```yaml
# From config/experiments/smoke.yaml
model:
  input_dim: 41
  hidden_dim: 256
  embedding_dim: 64
  num_family_classes: 7
  num_binary_classes: 2
  
training:
  batch_size: 256
  epochs: 100
  learning_rate: 0.001
  weight_decay: 1e-5
  grad_clip: 1.0
  label_smoothing: 0.1
  warmup_epochs: 10
  focal_gamma: 1.5
  patience: 15
  seed: 42
  
class_balance:
  strategy: weighted_ce    # Options: none, weighted_ce, sqrt_weighted_ce, focal
  sampler: interleaved_rr  # Options: interleaved_rr, weighted_random
  family_margin_loss_weight: 0.5
  family_margin_tau: 0.1
  class4_logit_penalty_weight: 0.3
```

## Evaluation Settings

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Macro F1 | >0.85 | Primary metric for promotion |
| Binary F1 | >0.95 | Secondary metric |
| Threat-weighted F1 | >0.80 | Rare-class sensitive |
| Per-class F1 (class 4 / U2R) | >0.70 | Minimum for rare class |

## Benchmark Settings

(NOT YET DOCUMENTED — procedure defined in docs/BENCHMARK_PROTOCOL.md)

## Reproduction Steps

### End-to-end reproduction:

```bash
# Step 1: Environment
git clone <repo>
cd <repo>
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt

# Step 2: Data
PYTHONPATH=src python scripts/data/download_datasets.py
PYTHONPATH=src python scripts/data/process_nsl_kdd.py
PYTHONPATH=src python scripts/data/process_unsw_nb15.py
PYTHONPATH=src python scripts/data/process_cicids.py
PYTHONPATH=src python scripts/training/prepare_canonical_artifacts.py

# Step 3: Train
PYTHONPATH=src python scripts/training/train_helix_ids_full.py \
    --config config/experiments/smoke.yaml \
    --output /tmp/helix_output \
    --epochs 50 \
    --seed 42

# Step 4: Evaluate
PYTHONPATH=src python scripts/evaluation/holdout_evaluation_v2.py \
    --checkpoint /tmp/helix_output/helix_full_nsl_kdd_best.pt

# Step 5: Benchmark
PYTHONPATH=src python scripts/evaluation/benchmarks.py \
    --manifest config/experiments/smoke.yaml

# Step 6: Run tests
pytest -q
```

### Known reproducibility issues:
1. **CICIDS-2018 acquisition not fully automated**: The raw dataset lives at an external path (`data/cicids2018/raw/` is a symlink). Downloading requires manual interaction.
2. **GPU determinism**: CUDA operations may produce slightly different results across GPU architectures even with fixed seed.
3. **PyTorch version sensitivity**: Minor PyTorch versions may produce different results.
4. **Randomness in augmentation**: Data augmentation (if enabled) adds non-deterministic noise.
5. **Multi-seed consensus required**: Single-seed results may not generalize. Three seeds recommended for confidence.