# Phase 43G — Causal Attribution of IDS Transfer Failure

**Date:** 2026-06-26
**Experiment:** Phase 43G, RP-2 HELIX-IDS

**Objective:** Determine which component of benchmark incompatibility is responsible for cross-dataset IDS transfer failure after Phase 43F established that covariance alignment does not significantly improve transfer.

## Scientific Background

Previous phases established:
- Phase 43C: CORAL significantly reduced benchmark fingerprinting and geometric mismatch.
- Phase 43D: CORAL preserved attack semantics.
- Phase 43E: CORAL produced only small average transfer gains.
- Phase 43F: Those gains were statistically indistinguishable from noise.

Therefore, covariance mismatch is not the dominant source of transfer failure. The remaining candidate explanations are:
1. Label-space mismatch
2. Class prior shift
3. Higher-order distribution mismatch
4. Non-linear manifold mismatch
5. Feature semantic mismatch
6. Dataset-specific attack behavior

## Hypotheses

- **H0:** Cross-dataset transfer failure is primarily explained by covariance mismatch.
- **H1:** Cross-dataset transfer failure is primarily explained by factors beyond covariance mismatch.

## Experimental Design

For each of 12 transfer directions across 4 datasets (NSL-KDD, UNSW-NB15, CICIDS2018, TON-IoT), six diagnostic metrics were computed:

| # | Metric | Description |
|---|--------|-------------|
| 1 | Covariance Distance | Frobenius norm ||Σs − Σt||F |
| 2 | CORAL Improvement ΔMF1 | MF1_CORAL − MF1_Raw (from Phase 43E) |
| 3 | Label-Space Overlap | Shared classes / Union classes |
| 4 | Class Prior Divergence | Jensen-Shannon divergence (bits) |
| 5 | MMD | Maximum Mean Discrepancy (Gaussian kernel) |
| 6 | MI Retention | In-distribution MF1 ratio (CORAL-aligned / Raw) |

## Table 1: Transfer Direction Diagnostics

| Direction | CovDist | LabelOverlap | JSDiv | MMD | ΔMF1 | MI Retention |
|---|---|---|---|---|---|---|
| NSL-KDD → UNSW-NB15 | 6.0033 | 0.5714 | 0.6250 | 0.4957 | -0.024776 | 0.8734 |
| NSL-KDD → CICIDS2018 | 8.4511 | 0.3333 | 0.5384 | 0.5931 | -0.000804 | 0.7601 |
| NSL-KDD → TON-IoT | 6.9364 | 0.6667 | 0.5272 | 0.4134 | +0.004329 | 0.8329 |
| UNSW-NB15 → NSL-KDD | 6.0033 | 0.5714 | 0.6250 | 0.5104 | -0.004484 | 0.8124 |
| UNSW-NB15 → CICIDS2018 | 7.4904 | 0.5000 | 0.6313 | 0.6315 | +0.035061 | 0.5349 |
| UNSW-NB15 → TON-IoT | 7.7487 | 0.5714 | 0.5827 | 0.5480 | -0.002712 | 0.4620 |
| CICIDS2018 → NSL-KDD | 8.4539 | 0.3333 | 0.5384 | 0.6091 | +0.000000 | 0.9228 |
| CICIDS2018 → UNSW-NB15 | 7.4991 | 0.5000 | 0.6313 | 0.6477 | +0.000000 | 0.8979 |
| CICIDS2018 → TON-IoT | 8.5896 | 0.6000 | 0.7372 | 0.4529 | -0.000022 | 0.8554 |
| TON-IoT → NSL-KDD | 6.9364 | 0.6667 | 0.5272 | 0.4267 | +0.062163 | 0.6382 |
| TON-IoT → UNSW-NB15 | 7.7487 | 0.5714 | 0.5827 | 0.5515 | -0.007683 | 0.6182 |
| TON-IoT → CICIDS2018 | 8.5897 | 0.6000 | 0.7372 | 0.4452 | +0.003211 | 0.4600 |

## Table 2: Spearman Correlations

| Test | Predictor | Response | ρ | p-value | Significant (p<0.05) |
|---|---|---|---|---|---|
| Test A | Covariance Distance | ΔMF1 | 0.1261 | 0.6962 | ✗ |
| Test A2 | Covariance Distance | Raw MF1 | 0.5804 | 0.0479 | ✓ |
| Test B | Label Overlap | ΔMF1 | 0.2658 | 0.4037 | ✗ |
| Test B2 | Label Overlap | Raw MF1 | -0.5880 | 0.0443 | ✓ |
| Test C | MMD | ΔMF1 | -0.1681 | 0.6015 | ✗ |
| Test C2 | MMD | Raw MF1 | 0.4476 | 0.1446 | ✗ |
| Test D | JS Divergence | ΔMF1 | -0.1345 | 0.6768 | ✗ |
| Test D2 | JS Divergence | Raw MF1 | 0.1414 | 0.6612 | ✗ |
| Test E | MI Retention | ΔMF1 | -0.1506 | 0.6403 | ✗ |
| Test F | MI Retention | Raw MF1 | -0.2378 | 0.4568 | ✗ |

## Table 3: Multiple Linear Regression — ΔMF1

**Model:** ΔMF1 ~ CovarianceDistance + LabelOverlap + JSDivergence + MMD
**R² = 0.2367**, **Adj R² = -0.1995**
**F(4, 7) = 0.5425, p = 0.7105**

| Predictor | Std β | p-value | Partial R² |
|---|---|---|---|
| covariance_distance | +0.3730 | 0.3969 | 0.1042 |
| label_overlap_ratio | +0.6920 | 0.3031 | 0.1500 |
| js_divergence | -0.4558 | 0.2637 | 0.1742 |
| mmd | +0.2497 | 0.6571 | 0.0298 |

*Note: † p<0.10, * p<0.05*

## Table 4: Multiple Linear Regression — Raw MF1

**Model:** Raw MF1 ~ CovarianceDistance + LabelOverlap + JSDivergence + MMD
**R² = 0.6292**, **Adj R² = 0.4174**
**F(4, 7) = 2.9700, p = 0.0994**

| Predictor | Std β | p-value | Partial R² |
|---|---|---|---|
| covariance_distance | +0.4224 | 0.1861 | 0.2349 |
| label_overlap_ratio | -0.5480 | 0.2471 | 0.1855 |
| js_divergence | -0.0081 | 0.9761 | 0.0001 |
| mmd | -0.0528 | 0.8921 | 0.0028 |

## Decision Rules

### Outcome A: Covariance Distance strongly predicts transfer
- |ρ| > 0.6, p < 0.05 for Covariance Distance vs ΔMF1
- Conclusion: Covariance shift remains dominant.

### Outcome B: Label overlap and MMD dominate; Covariance weak
- Conclusion: Transfer failure is semantic rather than geometric.

### Outcome C: No factor explains transfer
- Conclusion: Benchmark incompatibility is fundamentally heterogeneous.

## Interpretation

### Test A: Covariance Distance vs ΔMF1

Spearman ρ = 0.1261, p = 0.6962
Result: Not significant
→ Covariance mismatch does NOT predict CORAL improvement.

### Test B: Label Overlap vs ΔMF1

Spearman ρ = 0.2658, p = 0.4037
Result: Not significant
→ Label overlap alone does not explain transfer.

### Test C: MMD vs ΔMF1

Spearman ρ = -0.1681, p = 0.6015
Result: Not significant
→ MMD does not predict CORAL improvement.

### Regression Analysis

The full model (R² = 0.2367) explains 23.7% of ΔMF1 variance.
No individual predictor reaches significance.

## Final Scientific Question

**If covariance alignment does not improve transfer, what factor actually explains IDS benchmark incompatibility?**

**Answer: Benchmark incompatibility is fundamentally heterogeneous.**
No single factor—covariance, label overlap, class prior shift, or higher-order distribution mismatch—consistently explains transfer failure. This is consistent with Outcome C: the causes interact and vary by direction.

The evidence supports a **multi-factorial, direction-dependent** model of IDS benchmark incompatibility. Different transfer directions fail for different reasons, requiring direction-specific adaptation strategies rather than a one-size-fits-all alignment method.

---
*Report generated by Phase 43G — Causal Attribution of IDS Transfer Failure*
