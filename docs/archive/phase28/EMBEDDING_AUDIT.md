# Embedding Audit (DANN)

## Methodology

For each experiment, backbone embeddings (64-dim) are extracted from DANN models for source and target test data. We compute:

- **silhouette_dataset**: Silhouette score of source vs target embeddings.
  Lower = better domain-invariant alignment.
- **silhouette_family**: Silhouette score of attack-family clusters within target.
  Higher = better class separability.

| Experiment | λ | Model | Sil-Dataset (↓better) | Sil-Family (↑better) |
|-----------|--|------:|---------------------:|--------------------:|
| exp01_pairwise_nsl_to_unsw | 0.500 | DANN | 0.4444 (+0.1162 vs baseline) | -0.0789 |
| exp02_pairwise_unsw_to_cicids | 0.010 | DANN | 0.1786 (+0.0743 vs baseline) | -0.5683 |
| exp03_pairwise_cicids_to_ton | 0.250 | DANN | 0.1250 (+0.0782 vs baseline) | -0.2345 |
| exp04_pairwise_ton_to_nsl | 0.500 | DANN | 0.1215 (-0.0097 vs baseline) | 0.1607 |
| exp05_holdout_3src_to_ton | 0.250 | DANN | 0.0843 (-0.0706 vs baseline) | -0.0236 |
| exp06_holdout_3src_to_cicids | 0.500 | DANN | 0.1538 (+0.0501 vs baseline) | -0.6220 |
| exp07_holdout_3src_to_nsl | 0.500 | DANN | 0.2474 (+0.2204 vs baseline) | 0.1139 |
| exp08_holdout_3src_to_unsw | 0.010 | DANN | 0.4304 (+0.1329 vs baseline) | 0.0062 |


## Summary

- **Average silhouette_dataset delta vs Phase 26B**: 0.0740
- **DANN increases domain separation** — feature alignment degraded.
