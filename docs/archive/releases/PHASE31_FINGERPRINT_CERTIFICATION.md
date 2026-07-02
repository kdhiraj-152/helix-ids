# Phase 31 — Fingerprint Elimination Certification

**Date:** 2026-06-24
**Certification Authority:** Phase 31 Analysis Pipeline
**Status:** ❌ **NOT APPROVED** — Cannot proceed to Phase 32 as designed

---

## Executive Summary

Phase 31 conducted a comprehensive investigation of dataset fingerprinting in the 17-canonical-feature harmonized space. The core finding is that **dataset identity is redundantly encoded in the joint distribution of all features** and cannot be eliminated through input-level interventions (feature ablation, normalization, or scaling).

---

## Success Criteria Assessment

### Primary: Dataset-ID Accuracy Reduction ≥ 30 Percentage Points

| Metric | Baseline | Best Achieved | Target | Result |
|--------|----------|---------------|--------|--------|
| Dataset-ID Accuracy | 100.0% | 57.6% (remove 15/17 features) | ≤70.0% | ❌ |
| Feature-Preserving Reduction | — | 0.01pp (remove 10 features) | 30pp | ❌ |
| Normalization Reduction | — | 0.0pp (all methods) | 30pp | ❌ |

**Verdict: FAIL.** Removing 15 of 17 features is functionally identical to abandoning the feature set. With 10 features removed (the maximum practical reduction), accuracy remains at 99.99%. No normalization method reduces dataset separability at all.

### Secondary: Average Leave-One-Dataset-Out Macro F1 Improvement ≥ 25%

| Intervention | Avg. Macro F1 | vs Baseline | Result |
|-------------|:------------:|:-----------:|:------:|
| Baseline | **0.099** | — | — |
| Ablate Top-5 | 0.077 | **−22%** | ❌ |
| Quantile Norm | 0.089 | **−10%** | ❌ |

**Verdict: FAIL.** No intervention improves average cross-dataset Macro F1. The best result is the baseline itself. Feature removal creates a zero-sum tradeoff (improving UNSW at the expense of CICIDS).

### Tertiary: Attack-Family Silhouette Maintained or Improved

| Method | Attack Silhouette (t-SNE) | vs Phase 30 | Result |
|--------|:------------------------:|:-----------:|:------:|
| Baseline Phase 31 | −0.097 | N/A (no Phase 30 baseline) | ⚪ |

**Verdict: INCONCLUSIVE** (no Phase 30 quantitative baseline). The negative attack silhouette (−0.097) indicates poor attack-family separation in the harmonized space, consistent with Phase 30's qualitative observations.

---

## Scientific Findings

### Finding 1: Dataset Fingerprint is Redundant and Pervasive

The dataset-ID classifier achieves 100% accuracy using the 17 canonical features, and this accuracy remains invariant to:
- Removal of the top-1, top-3, top-5, and top-10 features (still ≥99.99%)
- All tested normalization methods (z-score, robust, quantile, rank, per-dataset)
- Permutation of all but the most important feature

Only by removing 15 of 17 features does accuracy drop to 57.6% — still well above chance (33.3%).

### Finding 2: Feature Ablation Destroys Cross-Dataset Transfer

Removing dataset-identifying features systematically degrades cross-dataset Macro F1:
- Top-1 removal: −96.6%
- Top-3 removal: −96.3%
- Top-5 removal: −84.4%
- Top-10 removal: −46.0%

The features that encode dataset identity also carry the attack signal necessary for cross-dataset generalization. These are not separable signals.

### Finding 3: Normalization Cannot Erase the Fingerprint

All six normalization methods preserve 100% dataset-ID accuracy. Quantile and rank normalization — which break the original feature geometry — also destroy cross-dataset performance (−78% and −55% respectively), confirming that the fingerprint is not a marginal-distribution artifact.

### Finding 4: The Limitation is at the Feature Level, Not the Representation Level

The dataset-ID classifier achieves perfect accuracy using raw features (17-dim LR achieves 97%+, RF achieves 100%). This means the fingerprint is present in the input space and cannot be addressed by training a better feature extractor. Domain-adversarial training (as tested in Phase 30 with DANN) partially mitigates this at the representation level but cannot eliminate the information present in the input.

---

## Root Cause

The 17-feature harmonization scheme (`SCHEMA_VERSION="2026-05-25"`) maps heterogeneous dataset-specific features to a shared schema. This schema necessarily encodes dataset-specific distributional signatures because:

1. **Different raw features map to the same canonical feature** with different value distributions per dataset (e.g., `flag` values 0-10+ in NSL-KDD represent different semantics than mapped values in CICIDS)
2. **Missing and synthetic features** (e.g., `has_rst`, `traffic_direction`, `protocol_service_flag` in datasets that lack these raw attributes) are assigned dataset-specific default values
3. **Per-dataset log1p clipping** (applied to `duration`, `src_bytes`, `dst_bytes`) preserves dataset-specific scale characteristics even after standardization
4. **The joint distribution** of 17 features is sufficiently high-dimensional that any dataset's samples occupy a unique manifold

---

## Decision

**→ Conclude that the 17-feature harmonization itself imposes an upper bound on cross-dataset transfer and document the limitation.**

**Rationale:**
- Phase 31 demonstrates that feature-level interventions cannot simultaneously eliminate the fingerprint and preserve attack signal
- The Phase 30 DANN approach already represents the state of the art for this feature space
- Further work should focus on representation-level or model-level fingerprint mitigation (Phase 32 candidate: Fingerprint-Aware Domain Adaptation)

**Recommendation for documentation:**
- The 17-feature harmonized space achieves excellent in-distribution performance (Phase 24-30)
- Cross-dataset generalization is bounded by residual dataset fingerprints in the shared feature schema
- Future work on this axis should operate at the representation (embedding) level, not the input-feature level

---

## Artifacts

| File | Description |
|------|-------------|
| `docs/phase31/DATASET_FINGERPRINT_ANALYSIS.md` | Feature importance, permutation importance, ranking |
| `docs/phase31/FEATURE_ABLATION_RESULTS.md` | Progressive ablation experiment |
| `docs/phase31/NORMALIZATION_STUDY.md` | Normalization method comparison |
| `docs/phase31/CROSS_DATASET_RESULTS.md` | Leave-one-dataset-out validation |
| `docs/phase31/EMBEDDING_AUDIT.md` | t-SNE, UMAP, silhouette scores |
| `docs/phase31/plots/feature_importance.png` | Gini importance bar chart |
| `docs/phase31/plots/dataset_id_ablation.png` | Ablation curve |
| `docs/phase31/plots/cross_dataset_f1.png` | Cross-dataset grouped bar chart |
| `docs/phase31/plots/tsne_phase31.png` | t-SNE embedding |
| `docs/phase31/plots/umap_phase31.png` | UMAP embedding |
| `docs/phase31/results.json` | Full numerical results |
| `scripts/analysis/phase31_fingerprint_elimination.py` | Analysis pipeline |
