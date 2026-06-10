# Dataset Report

> Statistical characterization of all three benchmark datasets used for HELIX-IDS evaluation.

Last updated: 2026-06-09

## Overview

HELIX-IDS uses three standard network intrusion detection datasets. All three have been harmonized to a 41-feature, 7-class schema.

## Dataset Comparison

| Property | NSL-KDD | UNSW-NB15 | CICIDS-2018 |
|----------|---------|-----------|-------------|
| Year | 2009 | 2015 | 2018 |
| Sample type | Synthetic | Real + synthetic | Real |
| Network type | Simulated | Modern | Modern enterprise |
| Traffic protocols | TCP, UDP, ICMP | TCP, UDP, ICMP, HTTP | Full (HTTP, HTTPS, FTP, SSH, SMTP, DNS) |
| Raw features | 41 | 49 | ~80 |
| Harmonized features | 41 | 41 | 41 |
| Attack classes | 22 | 9 | 7 |
| Label type | Per-attack | Per-attack | Per-attack |
| Label mapping | 5-7 class | 5-7 class | 7 class |

## Class Distributions

### NSL-KDD

| Family Class | Train Count | Train % | Test Count | Test % |
|-------------|------------|---------|------------|--------|
| Normal (0) | 67,343 | 53% | 9,711 | 43% |
| DoS (1) | 45,927 | 36% | 7,458 | 33% |
| Probe (2) | 11,656 | 9% | 2,421 | 11% |
| R2L (3) | 995 | <1% | 2,754 | 12% |
| U2R (4) | 52 | <0.1% | 200 | <1% |
| **Total** | **125,973** | **100%** | **22,544** | **100%** |

### UNSW-NB15

(NOT YET DOCUMENTED — class distribution statistics not extracted)

### CICIDS-2018

(NOT YET DOCUMENTED — class distribution statistics not extracted; dataset is 704 MB)

## Feature Distributions

(NOT YET DOCUMENTED — feature-wise statistics not extracted)

For each of the 41 canonical features, the following should be reported:
- Mean, median, std, min, max
- Missing value count
- Feature type (continuous, discrete, categorical)
- Per-dataset value ranges

## Data Quality Issues

### NSL-KDD
1. **Synthetic data**: Does not reflect real network traffic characteristics
2. **Class imbalance**: U2R class has only 52 samples (0.04%) — insufficient for reliable learning
3. **Outdated attack patterns**: No reflection of modern attacks (2017+)
4. **Label ambiguity**: Some attack types straddle family boundaries

### UNSW-NB15
1. **Mixed provenance**: Contains both real and synthetic traffic
2. **Feature heterogeneity**: Mixing flow-level and packet-level features
3. **Label imbalance**: Some attack families underrepresented

### CICIDS-2018
1. **Large size**: 704 MB processed CSV — requires significant storage
2. **External storage dependency**: Symlink to non-repository path
3. **Feature naming inconsistency**: Column names changed between CICIDS versions
4. **Memory constraints**: Cannot load entire dataset on edge devices
5. **Timestamp artifacts**: Features may contain temporal leakage if not properly split

## Domain Shifts

Between datasets:

| Shift Type | NSL-KDD → UNSW-NB15 | NSL-KDD → CICIDS-2018 |
|------------|--------------------|------------------------|
| Protocol distribution | Major shift | Major shift |
| Feature definitions | 30/41 direct match | 25/41 direct match |
| Class definitions | 5 families → 7 | 5 families → 7 |
| Traffic patterns | Simulated | Real |
| Background noise | None | Significant |

The multi-dataset pretrainer (`transfer_learning.py`) and domain adaptation modules (`dann.py`, `mmd_loss.py`, etc.) are designed to mitigate these shifts, but their effectiveness varies by dataset pair.

## Dataset Limitations

1. **Single benchmark**: Three datasets, all from academic benchmarks — no real production data
2. **Feature engineering dependence**: Results are sensitive to the 41-feature harmonization
3. **Temporal validity**: NSL-KDD (2009) and UNSW-NB15 (2015) may not reflect current attack patterns
4. **Label quality**: Attack labels are dataset-defined, not ground-truth verified
5. **Missing data**: CICIDS-2018 has known missing values in some feature columns
6. **Split comparability**: Different datasets use different train/test splits, complicating cross-dataset evaluation
