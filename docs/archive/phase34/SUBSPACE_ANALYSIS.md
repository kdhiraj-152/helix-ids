# Domain-Invariant Subspace Analysis

## Intrinsic Dimensionality

Number of PCA components needed to explain 95% and 99% of variance.

| Dataset | Features | N Samples | Intrinsic Dim (95%) | Intrinsic Dim (99%) |
|---------|:--------:|:---------:|:-------------------:|:-------------------:|
| NSL-KDD | 17 | 49,998 | 9 | 11 |
| UNSW-NB15 | 17 | 49,997 | 7 | 8 |
| CICIDS2018 | 17 | 49,999 | 5 | 7 |

## Pairwise Manifold Overlap

Reconstruction error when projecting source data into target's PCA space.
Lower reconstruction error = more similar manifold structure.
Subspace alignment: mean cosine similarity between top-5 principal components.

| Source → Target | Recon Error | Subspace Alignment | PCA Dims |
|----------------|:----------:|:------------------:|:--------:|
| CICIDS2018 → NSL-KDD | 0.0 | 0.2168 | 17 |
| CICIDS2018 → UNSW-NB15 | 0.0 | 0.2828 | 17 |
| NSL-KDD → CICIDS2018 | 0.0 | 0.2168 | 17 |
| NSL-KDD → UNSW-NB15 | 0.0 | 0.183 | 17 |
| UNSW-NB15 → CICIDS2018 | 0.0 | 0.2828 | 17 |
| UNSW-NB15 → NSL-KDD | 0.0 | 0.183 | 17 |

## Common Subspace Size

The intersection of meaningful signal subspaces across datasets.

- Minimum intrinsic dimension: 5 (dataset with lowest complexity)
- Maximum intrinsic dimension: 9 (dataset with highest complexity)
- Gap: 4 dimensions

The common subspace is constrained by the dataset with the SIMPLEST structure (5 dimensions). Any model trained on 9-dimensional data will learn features specific to that dataset's excess complexity that do not transfer to the simpler domain — and vice versa.
