# HELIX-IDS Feature Harmonization

## Purpose

This document defines the active feature contract used for multi-dataset training.

The repository currently trains on a shared 31-feature representation:

- 28 harmonized common features
- 3 dataset-origin one-hot indicators

Canonical implementation: `src/helix_ids/data/feature_harmonization.py`

## Input Contract

Model input dimension is fixed at 31:

- Common features (28): network, count/rate, packet-length, timing, and login indicators
- Dataset-origin flags (3):
  - `is_nsl_kdd`
  - `is_unsw`
  - `is_cicids`

This contract is consumed by `src/helix_ids/models/helix_ids_full.py`.

## Dataset Mapping

Three dataset-specific mappers normalize field naming and semantics into the common space:

- NSL-KDD mapper
- UNSW-NB15 mapper
- CICIDS mapper

The mapping layer handles inconsistent source column names (case/spacing variants) via normalized matching.

## Label Taxonomy

The training target uses a 7-class family taxonomy plus binary head:

- Binary: Normal vs Attack
- Family classes:
  - Normal
  - DoS
  - Probe
  - R2L
  - U2R
  - Generic
  - Backdoor

Taxonomy and dataset-specific mappings are maintained in `src/helix_ids/data/feature_harmonization.py`.

## Normalization and Leakage Prevention

Normalization is applied per dataset to prevent identity leakage across domains.

Canonical loader and split behavior: `src/helix_ids/data/multi_dataset_loader.py`

- Train/val/test splitting is stratified when feasible
- Scalers are fit per dataset partitioning policy
- Harmonized outputs feed script-level training entrypoints

## Validation Expectations

Any change to harmonization should be validated with:

- Unit tests under `tests/`
- Loader/harmonization tests under data-related test modules
- Quick governance checks when training scripts are modified

Recommended project checks:

```bash
pytest tests/ -v --cov=src/helix_ids --cov-report=xml --cov-report=term-missing
ruff check src/ scripts/
ruff format --check src/ scripts/
```
