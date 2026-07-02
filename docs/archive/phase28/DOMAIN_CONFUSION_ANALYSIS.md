# Domain Confusion Analysis

## Overview

Domain confusion measures how well the adversarial training has suppressed dataset-specific features. A well-confused model should have target domain classifier accuracy near random chance (50% for binary source/target distinction).

## Per-Experiment Domain Accuracy

We approximate domain confusion by the inverse of silhouette_dataset: lower silhouette_dataset = stronger domain confusion.

| Experiment | λ | Sil-Dataset | Domain Alignment |
|-----------|--:|-----------:|-----------------:|
| exp01_pairwise_nsl_to_unsw | 0.500 | 0.4444 | Weak |
| exp02_pairwise_unsw_to_cicids | 0.010 | 0.1786 | Weak |
| exp03_pairwise_cicids_to_ton | 0.250 | 0.1250 | Moderate |
| exp04_pairwise_ton_to_nsl | 0.500 | 0.1215 | Moderate |
| exp05_holdout_3src_to_ton | 0.250 | 0.0843 | Moderate |
| exp06_holdout_3src_to_cicids | 0.500 | 0.1538 | Weak |
| exp07_holdout_3src_to_nsl | 0.500 | 0.2474 | Weak |
| exp08_holdout_3src_to_unsw | 0.010 | 0.4304 | Weak |


## Interpretation

- **Strong alignment** (silhouette <= 0.05): Features are nearly
  indistinguishable between source and target domains.
- **Moderate alignment** (0.05 < silhouette <= 0.15): Partial
  domain overlap; some dataset-specific features remain.
- **Weak alignment** (silhouette > 0.15): Source and target
  remain clearly separable; DANN has not eliminated domain gap.
