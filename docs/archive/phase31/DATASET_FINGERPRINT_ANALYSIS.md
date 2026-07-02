# Dataset Fingerprint Analysis — Phase 31

**Date:** 2026-06-24
**Experiment:** Phase 31, RP-2 HELIX-IDS
**Status:** Complete — Negative Result

## Summary

Dataset-ID classification achieves **100% accuracy** on the 17 canonical harmonized features across all three datasets (NSL-KDD, UNSW-NB15, CICIDS-2018). Dataset fingerprinting is **not driven by individual features** but by the joint distribution across all features. Ablating the top 15 of 17 features still yields **57.6% accuracy** (well above chance 33.3%).

## Experimental Setup

- **Classifier:** Random Forest (200 trees, 3-fold stratified CV)
- **Features:** 17 canonical harmonized features (`SCHEMA_VERSION="2026-05-25"`)
- **Training samples:** 356,117 (107,077 NSL-KDD + 149,040 UNSW-NB15 + 100,000 CICIDS-2018 subsampled)
- **Target:** 3-class dataset origin classification

## Results

| Metric | Value |
|--------|-------|
| Dataset-ID Accuracy (all 17 features) | **100.0%** |
| Dataset-ID Accuracy (remove top-15 features, keep 2) | **57.6%** |
| Dataset-ID Accuracy (remove top-10 features, keep 7) | **99.99%** |
| Chance level | **33.3%** |

## Feature Importance Ranking (Gini)

| Rank | Feature | Importance | Cumulative |
|------|---------|-----------|------------|
| 1 | flag | 0.2398 | 24.0% |
| 2 | src_dst_bytes_ratio | 0.1938 | 43.4% |
| 3 | dst_src_bytes_ratio | 0.1424 | 57.6% |
| 4 | connection_state | 0.1291 | 70.5% |
| 5 | protocol_service_flag | 0.0756 | 78.1% |
| 6 | count_x_srv_count | 0.0703 | 85.1% |
| 7 | duration | 0.0687 | 91.8% |
| 8 | diff_srv_rate_x_flag | 0.0220 | 93.8% |
| 9 | src_bytes | 0.0184 | 95.7% |
| 10 | dst_bytes | 0.0108 | 96.7% |
| 11 | log_dst_bytes | 0.0089 | 97.6% |
| 12 | log_src_bytes | 0.0087 | 98.5% |
| 13 | same_host_rate_x_service | 0.0072 | 99.2% |
| 14 | service_tier | 0.0037 | 99.6% |
| 15 | protocol_type | 0.0003 | 99.9% |
| 16 | has_rst | 0.0002 | 99.9% |
| 17 | traffic_direction | 0.0001 | 100.0% |

**Top-5 features account for 78.1% of total importance.**
**Top-10 features account for 97.1%.**

## Permutation Importance

Only `flag` shows significant permutation importance (0.259 ± 0.001). All other features show zero permutation importance, indicating that:
1. `flag` is the single most informative feature for dataset discrimination
2. Other features contribute to dataset separability only through their joint multi-dimensional distribution, not individually
3. The 100% accuracy is achieved through the ensemble combination of all features

## Embedding Silhouette Scores

| Method | Dataset Silhouette | Attack Silhouette |
|--------|-------------------|-------------------|
| t-SNE | 0.2252 | -0.0973 |
| UMAP | 0.0408 | -0.1964 |

- Dataset silhouette is **moderately positive** (0.2252 t-SNE), confirming dataset-level clustering
- Attack-family silhouette is **negative** in both embeddings, indicating attack classes are not separable in the harmonized feature space

## Plots

- `plots/feature_importance.png` — Gini importance with permutation importance overlay
- `plots/dataset_id_ablation.png` — Accuracy vs. features removed
- `plots/tsne_phase31.png` — t-SNE colored by dataset (left) and attack family (right)
- `plots/umap_phase31.png` — UMAP colored by dataset (left) and attack family (right)

## Conclusion

**Dataset fingerprint is encoded in the joint distribution of all 17 features and cannot be eliminated by feature-level interventions.** This is a fundamental property of the harmonized feature space, not a model capacity issue.
