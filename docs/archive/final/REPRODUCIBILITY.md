# Reproducibility Package

**Project:** Helix IDS — Cross-Dataset Transfer Learning for Network Intrusion Detection
**Date:** 2026-06-24

---

## 1. Dataset Versions

| Dataset | Source | Version | Date | Samples |
|---------|--------|---------|:----:|-------:|
| NSL-KDD | https://www.unb.ca/cic/datasets/nsl.html | KDDTrain+.txt / KDDTest+.txt | 2009 | 148,517 |
| UNSW-NB15 | https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/ | UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv | 2015 | 175,341 |
| CICIDS2018 | https://www.unb.ca/cic/datasets/ids-2018.html | AWS traffic captures (CSV format) | 2018 | 16,232,943 |
| TON-IoT | https://research.unsw.edu.au/projects/toniot-dataset | Train_Test_Network.csv + IoT sensor telemetry | 2021 | 461,043 |

### Harmonization

All datasets are harmonized to the **17-canonical-feature** schema defined by:

- **Schema version:** `SCHEMA_VERSION="2026-05-25"`
- **Schema contract file:** `src/helix_ids/contracts/canonical_features.py`
- **Harmonization pipeline:** `src/helix_ids/data/feature_harmonization.py`
- **Multi-dataset loader:** `src/helix_ids/data/multi_dataset_loader.py`

### Dataset Contracts

Each dataset has a formal schema contract verified during loading:

| Dataset | Contract | Verification |
|---------|----------|:------------:|
| NSL-KDD | `contracts/nsl_kdd_contract.py` | Phase 25A certification |
| UNSW-NB15 | `contracts/unsw_nb15_contract.py` | Phase 25A certification |
| CICIDS2018 | `contracts/cicids2018_contract.py` | Phase 25A certification |
| TON-IoT | `contracts/ton_iot_contract.py` | Phase 25C certification |

### Label Taxonomy

7-class unified ontology (mapping in `data/label_mapping.py`):

| Index | Family |
|:-----:|--------|
| 0 | Normal |
| 1 | DoS |
| 2 | Probe |
| 3 | R2L |
| 4 | U2R |
| 5 | Generic |
| 6 | Backdoor |

---

## 2. Hyperparameters

### DANNHelixModel Architecture

| Parameter | Value |
|-----------|-------|
| Shared layers | [128, 64, 32] |
| Activation | ReLU |
| Batch normalization | Yes (after each shared layer) |
| Dropout | 0.3 |
| Family classifier | 32 → 7 (softmax) |
| Binary head | 32 → 2 (softmax) |
| Domain classifier | 32 → 16 → 1 (sigmoid) |
| Gradient reversal λ | 0.5 (mode best from Phase 28A sweep) |

### Training

| Parameter | Phase 26A | Phase 26B+ |
|-----------|:---------:|:----------:|
| Optimizer | Adam | Adam |
| Learning rate | 0.001 | 0.001 |
| Batch size | 128 | 128 |
| Max epochs | 100 | 100 |
| Patience | 20 | 20 |
| Weight decay | 0 | 0 |
| Loss weighting | Weighted cross-entropy | Weighted cross-entropy |
| Gradient clipping | None | None |

### CORAL (Phase 27)

| Parameter | Values |
|-----------|--------|
| λ_coral sweep | {0.01, 0.05, 0.1, 0.5, 1.0} |
| Best λ | 0.5 |

### DANN (Phase 28)

| Parameter | Values |
|-----------|--------|
| λ_dann sweep | {0.01, 0.05, 0.1, 0.25, 0.5} |
| Best λ (mode) | 0.5 |
| Domain classifier | 32 → 16 → 1 MLP |

### Data Sampling

| Parameter | Phase 26A | Phase 26B+ |
|-----------|:---------:|:----------:|
| Train cap per source | 50,000 | 200,000 |
| Test cap per target | 10,000 | 50,000 |
| Stratification | By attack family | By attack family |

---

## 3. Seeds

| Experiment | Seeds | Purpose |
|-----------|:-----:|---------|
| Phase 26A | 42 | Baseline |
| Phase 26B | 42 | Production baseline |
| Phase 27A | 42 | CORAL pilot |
| Phase 27B | 42 | CORAL multi-dataset |
| Phase 28A | 42 (×5 λ) | DANN development |
| Phase 28C | {42, 1337, 2026, 141, 256} (×8 exp) | DANN stability |
| Phase 29 | {42, 1337, 2026} | Production deployment |
| Phase 30 | {42, 1337, 2026} | Forensic audit |
| Phase 31 | 42 | Fingerprint analysis |
| Phase 32 | 42 | Schema redesign |
| Phase 33 | 42 (+ bootstrap) | Incompatibility proof |
| Phase 34 | 42 | Ceiling validation |

Default seed: **42** (used for all single-seed experiments).

---

## 4. Hardware

| Component | Specification |
|-----------|---------------|
| **Device** | Apple Mac (Apple Silicon) |
| **SoC** | M-series (M1 Pro / M2 Max / M3 Max) |
| **Compute backend** | MPS (Metal Performance Shaders) |
| **Precision** | FP32 |
| **RAM** | 32–64 GB unified memory |
| **Storage** | NVMe SSD |
| **Training time** | 20–125 seconds per experiment (Phase 28A) |
| **Inference latency** | 0.39 ms/sample |
| **Peak throughput** | 639,964 samples/s |

---

## 5. Software Versions

| Component | Version |
|-----------|---------|
| Python | 3.11 |
| PyTorch | 2.x (MPS-compatible) |
| NumPy | 1.24+ |
| pandas | 2.0+ |
| scikit-learn | 1.3+ |
| matplotlib | 3.7+ |
| seaborn | 0.12+ |
| umap-learn | 0.5+ |
| scipy | 1.11+ |
| macOS | 14.x (Sonoma) or later |

---

## 6. Command Lines

### Phase 26A — Cross-Dataset Baseline

```bash
cd /path/to/RP-2
source .venv311/bin/activate
PYTHONPATH=src python scripts/benchmarks/phase28a.py \
    --max-samples 50000 \
    --epochs 100 \
    --patience 20 \
    --seed 42 \
    --device mps
```

### Phase 26B — Production-Scale Baseline

```bash
PYTHONPATH=src python scripts/benchmarks/phase27b.py \
    --max-samples 200000 \
    --epochs 100 \
    --patience 20 \
    --seed 42 \
    --device mps
```

### Phase 27A — CORAL Pilot

```bash
PYTHONPATH=src python scripts/training/coral.py \
    --source nsl_kdd \
    --target unsw_nb15 \
    --lambda-coral 0.5 \
    --epochs 100 \
    --seed 42 \
    --device mps
```

### Phase 27B — CORAL Multi-Dataset

```bash
PYTHONPATH=src python scripts/benchmarks/phase27b.py \
    --max-samples 200000 \
    --lambda-coral 0.5 \
    --epochs 100 \
    --patience 20 \
    --seed 42 \
    --device mps
```

### Phase 28A — DANN Development

```bash
PYTHONPATH=src python scripts/benchmarks/phase28a.py \
    --max-samples 200000 \
    --epochs 100 \
    --patience 20 \
    --seed 42 \
    --device mps
```

### Phase 28C — DANN Production

```bash
PYTHONPATH=src python scripts/benchmarks/phase28c.py \
    --max-samples 200000 \
    --epochs 100 \
    --patience 20 \
    --seeds 42 1337 2026 141 256 \
    --device mps
```

### Phase 29 — Production Deployment

```bash
PYTHONPATH=src python scripts/training/train_helix_ids_full.py \
    --config configs/production_dann.yaml \
    --seed 42 \
    --device mps
```

### Phase 30 — Forensic Audit

```bash
# Domain generalization
PYTHONPATH=src python scripts/benchmarks/phase30_domain_gen.py \
    --seeds 42 1337 2026 \
    --epochs 100 \
    --patience 20 \
    --device mps

# Random label test
PYTHONPATH=src python scripts/benchmarks/phase30_random_label.py \
    --seeds 42 1337 2026 \
    --device mps

# Dataset-ID prediction
PYTHONPATH=src python scripts/analysis/phase30_dataset_id.py \
    --seed 42
```

### Phase 31 — Fingerprint Analysis

```bash
PYTHONPATH=src python scripts/analysis/phase31_fingerprint_elimination.py \
    --seed 42
```

### Phase 32 — Schema Redesign

```bash
PYTHONPATH=src python scripts/analysis/phase32_harmonization_redesign.py \
    --seed 42
```

### Phase 33 — Incompatibility Proof

```bash
# Covariate shift analysis
PYTHONPATH=src python scripts/analysis/phase33_covariate_shift.py \
    --seed 42

# Label shift analysis
PYTHONPATH=src python scripts/analysis/phase33_label_shift.py \
    --seed 42

# Semantic overlap analysis
PYTHONPATH=src python scripts/analysis/phase33_semantic_overlap.py

# Domain divergence (Ben-David bound)
PYTHONPATH=src python scripts/analysis/phase33_domain_divergence.py \
    --seed 42
```

### Phase 34 — Ceiling Validation

```bash
# Oracle (within-dataset) performance
PYTHONPATH=src python scripts/benchmarks/phase34_oracle.py \
    --seed 42 \
    --epochs 100 \
    --device mps

# Shared-class transfer
PYTHONPATH=src python scripts/benchmarks/phase34_shared_class.py \
    --seed 42 \
    --epochs 100 \
    --device mps

# Transfer ratio analysis
PYTHONPATH=src python scripts/benchmarks/phase34_transfer_ratio.py \
    --seed 42

# Subspace analysis
PYTHONPATH=src python scripts/analysis/phase34_subspace_analysis.py \
    --seed 42
```

---

## 7. Configuration Files

- **Training config:** `configs/production_dann.yaml` (Phase 29 deployment)
- **Model definition:** `src/helix_ids/models/helix_ids_full.py` (DANNHelixModel)
- **Data pipeline:** `src/helix_ids/data/unified_loader.py`
- **Feature harmonization:** `src/helix_ids/data/feature_harmonization.py`
- **Schema contracts:** `src/helix_ids/contracts/`
- **Benchmark scripts:** `scripts/benchmarks/phase28a.py`, `phase28c.py`, `phase27b.py`

---

## 8. Phase Manager

For running experiments in sequence with automation:

```bash
PYTHONPATH=src python scripts/training/scheduler/phase_orchestrator.py \
    --phases 26A 26B 27A 27B 28A 28C 29 30 31 32 33 34 \
    --seed 42 \
    --device mps
```

---

## 9. Expected Output Locations

| Artifact | Location |
|----------|----------|
| Benchmark results | `benchmarks/phase*_results.json` |
| Certification documents | `docs/releases/PHASE*_CERTIFICATION.md` |
| Analysis results | `docs/phase*/` |
| Plots | `docs/phase*/plots/` |
| Analysis scripts | `scripts/analysis/phase*_*.py` |
| Harmonized cache | `benchmarks/phase*_harmonized_cache_*.npz` |

---

*Generated: 2026-06-24*
