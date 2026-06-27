# Embedding Audit — Phase 31

**Date:** 2026-06-24
**Experiment:** Visual and quantitative analysis of dataset vs. attack family separation in latent space

## Methodology

- **Sample:** 10,000 per dataset (30,000 total), standardized (z-score)
- **t-SNE:** perplexity=30, max_iter=1000
- **UMAP:** n_neighbors=30, min_dist=0.1
- **Silhouette score:** computed on 2D embeddings

## Silhouette Scores

| Embedding | Dataset Silhouette | Attack Silhouette |
|-----------|-------------------|-------------------|
| t-SNE | 0.2252 | −0.0973 |
| UMAP | 0.0408 | −0.1964 |

### Interpretation

- **Dataset silhouette positive (0.225 t-SNE):** Moderate dataset-level clustering. Points from the same dataset are closer to each other than to points from other datasets. This is consistent with the dataset-ID classifier's 100% accuracy.
- **Attack silhouette negative (both embeddings):** Attack family clusters are not compact. Negative silhouette means within-attack-family distances exceed between-family distances. The harmonized feature space does not organize attacks by family.

### Comparison to Phase 30 Baseline

*(Phase 30 did not compute explicit silhouette scores. Baseline for comparison is the t-SNE/UMAP visualization in Phase 30 certification, which showed dataset clusters but no attack family separation — consistent with these quantitative results.)*

## Visual Observations

### t-SNE (plots/tsne_phase31.png)

**Left panel — colored by dataset:**
- NSL-KDD and UNSW-NB15 form distinct, separable clusters
- CICIDS samples are more dispersed but form a recognizable third cluster
- Clear 3-way dataset separation visible to the naked eye

**Right panel — colored by attack family:**
- No clean attack-family clusters visible
- Normal samples intermix with attack classes
- All 7 classes overlap heavily

### UMAP (plots/umap_phase31.png)

**Left panel — colored by dataset:**
- Dataset separation is less pronounced than t-SNE
- Some overlap between datasets visible
- Consistent with lower dataset silhouette (0.041 vs 0.225)

**Right panel — colored by attack family:**
- Complete intermixing of all 7 classes
- No structure at attack-family level

## Key Insight

The embedding results directly confirm the Phase 30-31 narrative:

1. **What the features encode:** Dataset identity (moderate separation)
2. **What the features do NOT encode:** Attack family distinctions (near-random separation)
3. **Why feature-level interventions fail:** The dataset signature is a property of the joint 17-dimensional distribution, not a separable signal that can be removed without losing attack information

Dataset and attack information appear to be **entangled** in the same feature dimensions. Removing dataset-specific signal necessarily removes attack-generalization signal.

## Plots

- `plots/tsne_phase31.png`
  - Left: colored by dataset origin
  - Right: colored by attack family
- `plots/umap_phase31.png`
  - Left: colored by dataset origin
  - Right: colored by attack family

## Conclusion

The embedding audit confirms that the 17-feature harmonized space exhibits moderate dataset clustering and negligible attack-family organization. This structural property prevents effective cross-dataset generalization at the feature level.
