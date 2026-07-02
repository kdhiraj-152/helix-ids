# Embedding Audit

## Methodology

For each experiment, backbone embeddings (64-dim) are extracted from both baseline and CORAL models for source and target test data. We compute:

- **silhouette_dataset**: Silhouette score of source vs target embeddings.
  Lower = better domain-invariant alignment.
- **silhouette_family**: Silhouette score of attack-family clusters within target.
  Higher = better class separability.

| Experiment | Model | Sil-Dataset (↓better) | Sil-Family (↑better) |
|-----------|------:|---------------------:|--------------------:|
| exp01_pairwise_nsl_to_unsw | baseline | 0.3282 | -0.0373 |
| exp01_pairwise_nsl_to_unsw | coral | 0.3898 | -0.0429 |
| exp02_pairwise_unsw_to_cicids | baseline | 0.1043 | -0.7365 |
| exp02_pairwise_unsw_to_cicids | coral | 0.1823 | -0.5578 |
| exp03_pairwise_cicids_to_ton | baseline | 0.0468 | -0.6179 |
| exp03_pairwise_cicids_to_ton | coral | 0.2513 | -0.0880 |
| exp04_pairwise_ton_to_nsl | baseline | 0.1312 | 0.1374 |
| exp04_pairwise_ton_to_nsl | coral | 0.1714 | 0.1344 |
| exp05_holdout_3src_to_ton | baseline | 0.1549 | -0.0405 |
| exp05_holdout_3src_to_ton | coral | 0.1624 | -0.0325 |
| exp06_holdout_3src_to_cicids | baseline | 0.1037 | -0.7357 |
| exp06_holdout_3src_to_cicids | coral | 0.1551 | -0.6026 |
| exp07_holdout_3src_to_nsl | baseline | 0.0270 | -0.5358 |
| exp07_holdout_3src_to_nsl | coral | 0.0261 | -0.7779 |
| exp08_holdout_3src_to_unsw | baseline | 0.2975 | -0.0680 |
| exp08_holdout_3src_to_unsw | coral | 0.3498 | -0.0091 |


## Summary

- **Average silhouette_dataset delta**: 0.0618
- **CORAL increases domain separation** — feature alignment degraded.
