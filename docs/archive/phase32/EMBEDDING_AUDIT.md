# Embedding Audit — Phase 32

**Date:** 2026-06-24
**Experiment:** Phase 32, RP-2 HELIX-IDS
**Status:** Complete

## Summary

Silhouette scores for dataset identity and attack-family separability in each alternative schema representation. Computed over a balanced 30,000-sample subset (10,000 per dataset).

Embeddings: Raw feature space after StandardScaler normalization (no dimensionality reduction).

## Dataset Silhouette

Positive scores indicate dataset-level clustering in the feature space. **Higher is worse** for cross-dataset transfer.

| Schema | Dataset Silhouette | Interpretation |
|--------|-------------------|----------------|
| Phase31 Baseline (17-feat) | **0.2621** | Moderate dataset clustering |
| PCA-8 | **0.3020** | PCA condenses dataset-specific variance |
| RP-8 | **0.3317** | Random projection amplifies differences |
| Schema-C (Network-behavior) | **0.3329** | Connection features cluster by dataset |
| Schema-A (Conservative) | **0.2720** | Conservative selection still clusters |
| Schema-D (Minimal transfer) | **0.1502** | Lowest dataset clustering — still positive |
| Schema-B (Statistical) | **0.1371** | Also low — ratios/interactions compress inter-dataset differences |

**Key finding:** All scores are positive (dataset clustering exists in every representation). Schema-D and Schema-B have the lowest dataset silhouette, but even at 0.137–0.150, the clustering is sufficient for 100% dataset-ID classification.

## Attack Silhouette

Negative scores indicate attack classes overlap (expected: attacks should not form clean clusters in the feature space).

| Schema | Attack Silhouette | Interpretation |
|--------|------------------|----------------|
| Phase31 Baseline (17-feat) | **−0.0090** | Near-random — attack families overlap heavily |
| Schema-C (Network-behavior) | **−0.0173** | Slightly worse than baseline |
| PCA-8 | **−0.0287** | Worse |
| Schema-A (Conservative) | **−0.0254** | Worse |
| RP-8 | **−0.0375** | Worse |
| Schema-D (Minimal transfer) | **−0.0537** | Markedly worse |
| Schema-B (Statistical) | **−0.1117** | Worst — stats features alone cannot separate attacks |

**Key finding:** No schema improves attack-family silhouette relative to baseline. Schema-D and Schema-B are significantly worse, confirming that removing dataset-identifying features also removes attack-discriminating signal.

## Silhouette Comparison

```
Schema                                            Dataset Sil ████  Attack Sil ████
Phase31-Baseline (17-feat)  ██████████████████████████████  0.262 ████████████  -0.009
Schema-A (Conservative)     ██████████████████████████████  0.272 ████████████  -0.025
Schema-B (Statistical)      ██████████████████████████    0.137 ██████████████  -0.112
Schema-C (Network-behavior) ████████████████████████████████0.333 ████████████  -0.017
Schema-D (Minimal transfer) ████████████████████████      0.150 ██████████████  -0.054
PCA-8                       ████████████████████████████████0.302 ████████████  -0.029
RP-8                        █████████████████████████████████ 0.332 ████████████  -0.037
```

## Phase 31 Comparison

| Method | Phase 31 Dataset Sil | Phase 32 Dataset Sil (Baseline) | Change |
|--------|---------------------|--------------------------------|--------|
| Raw feature space | — | 0.2621 | — |
| t-SNE | 0.2252 | — | — |
| UMAP | 0.0408 | — | — |

**Notes:** Phase 31 reported silhouette on t-SNE/UMAP embeddings (non-linear DR). Phase 32 reports silhouette in the original feature space. The two are not directly comparable but both confirm positive dataset clustering.

## Conclusion

**No schema reduces dataset silhouette to zero or negative.** Every representation preserves some dataset-level structure in the feature space. Even Schema-D (minimal transfer), which deliberately removes the top-5 dataset-ID features, retains positive dataset silhouette (0.150) — sufficient for the RF classifier to achieve 99.97% accuracy.

The tertiary criterion (attack-family silhouette improvement) is **not met**: all schemas have attack-family silhouette equal to or worse than the baseline.
