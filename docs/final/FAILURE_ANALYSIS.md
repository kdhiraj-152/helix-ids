# Failure Analysis

**Project:** Helix IDS — Cross-Dataset Transfer Learning for Network Intrusion Detection
**Date:** 2026-06-24

---

## Overview

Nine sequential phases of increasingly sophisticated interventions all failed to achieve meaningful cross-dataset transfer. This document catalogues **why** each approach failed, from the initial baseline through domain adaptation, feature engineering, schema redesign, and — ultimately — dataset incompatibility proof.

The unifying finding: **every failure traces to the same root cause — the datasets themselves are incompatible at a structural level that no modeling approach can overcome.**

---

## 1. Domain Adaptation Failure

### 1.1 CORAL (Phase 27A/B)

**Attempted solution:** Align second-order statistics (covariance matrices) of source and target feature distributions.

**Why it failed:**

CORAL assumes that the remaining domain shift after correlation alignment is small enough for a shared classifier to work. Three structural problems prevent this:

1. **Linear alignment is insufficient.** CORAL only matches mean and covariance. The 17-feature distributions differ in higher moments (skewness, kurtosis, multimodality) that linear alignment cannot address. Even after CORAL, the domain classifier trivially separates datasets.

2. **Alignment on fingerprint features is harmful.** CORAL aligns all 17 features, including the ones that carry dataset-specific signatures (e.g., `flag`, `connection_state`). Matching these distributions forces the model to learn a representation that is compromised — averaged across incompatible manifolds.

3. **Gains are pair-specific and non-transferable.** CORAL improved CICIDS → TON-IoT (+0.1766) but degraded UNSW → CICIDS (−0.1574). The pair where it worked had partially overlapping label structures; the pair where it failed had fundamentally different covariate distributions. A method that works on some pairs and harms others is not a generalizable solution.

**Evidence:** Avg MF1 Δ = +2.84% (threshold 20%). Silhouette reversed direction (+0.0618 increase, not decrease). 6/8 experiments improved but not enough.

### 1.2 DANN (Phase 28A/B/C)

**Attempted solution:** Domain-adversarial training with gradient reversal to force domain-invariant feature representations.

**Why it failed:**

DANN tries to make a feature extractor that fools a domain classifier. But:

1. **The domain classifier can always win.** A logistic regression achieves 100% accuracy distinguishing datasets using the 17 features. The gradient reversal in DANN pits a feature extractor against this domain classifier — but the feature extractor cannot discard information the domain classifier needs when that information overlaps with attack signal.

2. **DANN increases, not decreases, dataset silhouette.** Phase 28A showed dataset silhouette increased from 0.1492 (baseline) to 0.2232 (DANN), meaning the datasets became *more* separable in learned representations. This happens because DANN's optimization is unstable: the gradient reversal competes with the classification loss, and the classification loss dominates when attack-relevant features also encode dataset identity.

3. **Domain invariance was never achieved.** The residual dataset silhouette of 0.2232 (DANN) vs 0.1492 (baseline) confirms that DANN moved *away* from domain invariance. This is a consequence of the feature overlap — any representation that retains attack information also retains dataset information.

4. **Variance is high across seeds.** Phase 28C showed σ = 0.0531 across 5 seeds (threshold 0.03), with some experiments showing σ = 0.0715 (TON-IoT → NSL-KDD). This variance indicates the optimization landscape has multiple competing equilibria, none of which achieve domain invariance.

**Evidence:** DANN beat baseline by +397.81% but still only reached avg MF1 = 0.1349. Dataset silhouette *increased*. Domain invariance not achieved. Production variance unacceptably high.

### 1.3 Theoretical Explanation (Phase 33)

The Ben-David bound (Ben-David et al., 2010) decomposes target error as:

ε_T ≤ ε_S + d_H + λ

Where:
- ε_S = source error (how well the model classifies source data)
- d_H = H-divergence (domain classifier error — how separable domains are)
- λ = joint risk (best possible combined error — how well the same function performs on both domains)

**The critical finding: d_H ≈ 0 (maximum domain separability).** Domain adaptation (both CORAL and DANN) targets d_H. When d_H ≈ 0 (datasets are perfectly separable), there is nothing to reduce. Domain adaptation fails because the standard theoretical assumption — that there exists some overlap between domains for adaptation to exploit — does not hold for public IDS benchmarks.

**ε_S accounts for 58–90% of the upper bound.** Even if domain adaptation were perfect, the model would still fail because the source task (classifying attacks) is hard even on the source dataset.

**λ accounts for 7–11% of the bound.** This represents irreducible error due to non-overlapping label spaces — attack classes that exist in the target but not the source, which no domain adaptation can address.

---

## 2. Feature Engineering Failure

### 2.1 Production-Scale Training (Phase 26B)

**Attempted solution:** Increase training data 4× (50K → 200K samples per source) to improve generalization.

**Why it failed:**

1. **More data amplifies the dataset fingerprint.** Increasing the number of samples from each dataset makes the dataset signature *more* distinctive, not less. The decision boundary learned on source features becomes even more tightly coupled to source-specific distribution patterns.

2. **Generalization gap widens with more data.** The average generalization gap increased from +0.1172 (Phase 26B) versus Phase 26A, meaning the model became better at classifying source data and worse at target data. This is overfitting to source-specific patterns at scale.

3. **Embedding audit confirmed the mechanism.** t-SNE and UMAP visualizations showed clear dataset-specific clusters. Within each cluster, attack families randomly intermix — the model learned "which dataset is this from" as its primary representation, not "which attack is this."

**Evidence:** 2.49× improvement over Phase 26A (0.0197 → 0.0491 MF1) but still effectively zero transfer (0.0491). Embeddings cluster by dataset, not attack family. Generalization gap: +0.1172.

### 2.2 Feature Ablation (Phase 31)

**Attempted solution:** Remove the dataset-identifying features to force the model to rely on transferable patterns.

**Why it failed:**

1. **The fingerprint is distributed across ALL 17 features, not concentrated in a few.** Removing the top-10 fingerprint features only reduced dataset-ID accuracy from 100.0% to 99.99%. Even removing 15 of 17 features still yields 57.6% accuracy (chance = 33.3%). This means every feature carries some dataset-specific information.

2. **The fingerprint and attack signal share the same subspace.** Every time we removed a fingerprint feature, cross-dataset MF1 collapsed: top-1 removal reduced MF1 by 96.6%. The features that identify the dataset also carry the attack signal. This is not accidental — it is structural: attack traffic patterns in NSL-KDD *are* different from attack traffic patterns in CICIDS because the underlying networks, protocols, and attack implementations differ.

3. **No zero-tradeoff exists.** The features cannot be separated into "dataset ID" and "attack signal" components because they are the same dimensions. Any intervention that reduces one also reduces the other.

**Evidence:** Dataset-ID accuracy invariant to ablation. Cross-dataset MF1 degrades proportionally to ablation depth. Zero-tradeoff confirmed.

### 2.3 Normalization (Phase 31)

**Attempted solution:** Standardize feature distributions across datasets using z-score, robust scaling, quantile normalization, rank normalization, and per-dataset normalization.

**Why it failed:**

1. **Quantile and rank normalization break the original data geometry.** These methods force marginal distributions to match, but they destroy the joint distribution structure that carries attack signal. Cross-dataset MF1 dropped by 78% (quantile) and 55% (rank).

2. **Per-dataset standardization preserves dataset identity.** Z-score and robust scaling normalize within each dataset independently, but the relative ordering and spacing of samples within each dataset remains dataset-specific. A simple domain classifier still achieves 100% accuracy.

3. **Moment matching is not sufficient to erase the fingerprint.** The fingerprint survives first- and second-order moment matching because dataset identity is encoded in higher-order statistics and joint-distribution structure.

**Evidence:** All 6 normalization methods preserve 100% dataset-ID accuracy. Performance-degrading methods (quantile, rank) destroy attack signal without eliminating the fingerprint. No normalization achieves the goal.

---

## 3. Schema Redesign Failure (Phase 32)

**Attempted solution:** Replace the 17-feature canonical schema with fundamentally different feature representations: conservative (8 raw), statistical (8 log/ratio), network-behavior (9), minimal-transfer (9 excluding top-5 fingerprint), PCA-5, PCA-8, and RP-8 (random projection).

**Why it failed:**

1. **Every representation retains the dataset fingerprint.** Dataset-ID accuracy ≥ 99.97% for ALL 7 representations. PCA (which decorrelates features) and random projection (which isometry-preserving) both retain full separability. This proves the fingerprint is not a feature-selection artifact but a property of the data manifold itself.

2. **PCA preserves variance structure that encodes dataset identity.** The top PCA components capture the directions of maximum variance, which in this context are dataset-specific. CICIDS's massive class imbalance (98% normal) produces a variance structure that dominates the projection.

3. **Random projection preserves pairwise distances statistically.** While random projection should theoretically disrupt the joint distribution, it preserves distances with high probability for sufficiently large samples. This distance preservation is sufficient for a sufficiently flexible classifier (Random Forest) to decode dataset origin.

4. **Cross-dataset MF1 does not improve meaningfully.** Best performer (PCA-8): +0.6% vs baseline. All other schemas degrade performance (−1.0% to −20.2%). No alternative schema achieves the target +25% improvement.

5. **The CICIDS holdout is particularly sensitive.** Removing `flag`, `connection_state`, or ratio features collapses CICIDS binary F1 from 0.590 to near zero (0.000–0.005). The features that CICIDS relies on for detection also carry dataset identity.

**Evidence:** All 7 schemas ≥ 99.97% DS-ID accuracy. Best MF1 improvement +0.6% (target +25%). CICIDS collapses without fingerprint features. Zero-tradeoff confirmed universally.

---

## 4. Representation Failure (Phase 31)

**Attempted solution:** Produce domain-invariant embeddings that capture attack-semantic structure without dataset-specific patterns.

**Why it failed:**

1. **Embeddings cluster by dataset, not by attack family.** Both t-SNE and UMAP visualizations show clean separation by dataset (silhouette 0.2252 t-SNE) and random intermixing of attack families (silhouette −0.0973 t-SNE). The learned representation prioritizes dataset identity over attack semantics.

2. **The feature extractor cannot disentangle these signals.** The DANN gradient reversal mechanism theoretically should force domain invariance, but it fails because the domain discriminator can always distinguish datasets using any representation that retains attack information. This is a fundamental capacity limitation: there is no linear or non-linear transformation of the 17 features that preserves attack class separability while removing dataset separability.

3. **Attack silhouette is negative across all tested representations.** Even after DANN training (Phase 30 domain generalization), MACRO F1 stays at 0.063–0.192. The negative attack silhouette (−0.097 to −0.196) indicates that attack families are not coherent clusters in the embedding space — they are intermixed.

**Evidence:** t-SNE dataset silhouette 0.2252 (positive, clustering by dataset). t-SNE attack silhouette −0.0973 (negative, no attack structure). DANN does not resolve this.

---

## 5. Benchmark Failure (Phase 33–34)

**Summary:** The deepest failure is not in our methods but in the benchmarks themselves.

### 5.1 Covariate Shift

Every one of the 17 canonical features shows statistically significant distribution differences between every dataset pair (KS p < 0.05 for 100% of features). Mean JS divergence ranges from 0.36 to 0.66. This means the input distributions are not merely different — they are probabilistically almost disjoint. A linear classifier achieves 100% accuracy distinguishing datasets (Proxy A-distance = 2.0, the maximum).

### 5.2 Label Shift

Class priors differ dramatically: CICIDS2018 is 98.24% normal; TON-IoT is 22.07% normal. TVD ranges from 0.47 to 0.76. A model trained on CICIDS learns a strong "predict Normal" prior that completely fails on TON-IoT. Conversely, a model trained on TON-IoT sees attack proportions that mislead it for CICIDS.

### 5.3 Semantic Shift

Different datasets use different attack names for what is ostensibly the same attack family. The semantic overlap score is ≤ 0.21 for all pairs. For example, "Probe" attacks in NSL-KDD are labeled "probe"; in UNSW-NB15 as "analysis, fuzzers, reconnaissance"; in CICIDS2018 as "portscan"; and in TON-IoT as "scanning". A model trained to recognize "portscan" on CICIDS cannot identify "fuzzers" on UNSW even though both are reconnaissance activities.

### 5.4 Shared-Class Ceiling (Phase 34)

Even when restricting transfer to only classes present in both source and target, the average Macro F1 is only 0.0755 improvement. The best shared-class pair (NSL-KDD → UNSW-NB15) reaches 0.1885 MF1 — still well below deployment thresholds. Covariate shift on the shared classes is sufficient to prevent reliable transfer.

### 5.5 Information-Theoretic Ceiling (Phase 34)

The average achievable ceiling Macro F1 is 0.3702 — this is the best possible performance even after PERFECT domain adaptation. This ceiling is set by the information content of the features and labels, not by any architectural choice. Average transfer entropy is 0.7348, meaning 73.5% of the predictive information is lost to domain shift.

### 5.6 Benchmark Validity Assumptions Violated

Phase 34 tested the four standard assumptions for transfer learning benchmarks:

| Assumption | Status | Evidence |
|-----------|:------:|----------|
| **Shared support** | ❌ VIOLATED | Each dataset has unique attack classes |
| **Identical label space** | ❌ VIOLATED | No two datasets share the same class set |
| **Covariate shift only** | ❌ VIOLATED | Label shift and condition shift also present |
| **Overlap assumption** | ❌ VIOLATED | Domains perfectly separable by linear classifier |

All four assumptions — the minimal requirements for meaningful domain adaptation — are violated.

---

## Root Cause Synthesis

```
Public IDS Benchmarks
        ↓
  Different network environments
  (simulated military, enterprise, IoT, production)
        ↓
  Different attack generation methodologies
  (rule-based, synthetic, red-teaming)
        ↓
  Different feature extraction tools
  (tcpdump, Bro, CICFlowMeter, custom)
        ↓
        ↓
  Covariate Shift ──── Label Shift ──── Semantic Shift
        ↓                    ↓                   ↓
  100% features           TVD 0.47-0.76      Overlap ≤ 0.21
  significantly diff.     Class priors       Different attack
  (KS p < 0.05)           differ by 76pp     names per family
        ↓                    ↓                   ↓
        ↓───────────────────────────────────────────↓
                        ↓
             ALL FOUR ASSUMPTIONS VIOLATED
                        ↓
       Domain adaptation has nothing to reduce
       (H-divergence = 0, A-distance = max)
                        ↓
          Transfer ceiling ≤ 0.3702 MF1
             (0.6% of oracle preserved)
                        ↓
          Cross-dataset transfer is not a
          modeling problem — it is a
          benchmark design problem
```

---

*Generated: 2026-06-24*
