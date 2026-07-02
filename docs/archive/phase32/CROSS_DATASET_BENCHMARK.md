# Cross-Dataset Benchmark — Phase 32

**Date:** 2026-06-24
**Experiment:** Phase 32, RP-2 HELIX-IDS
**Status:** Complete — Negative Result

## Summary

Leave-one-dataset-out evaluation for all 7 alternative representations compared against the Phase 31 17-feature baseline. **No representation achieves the ≥25% Macro F1 improvement target.** The best is PCA-8 at +0.6%.

## Protocol

- **Classifier:** LogisticRegression (max_iter=1000, L2 regularization)
- **Training:** Two datasets combined; evaluated on held-out third
- **Metrics:** Macro F1 (7-class), Binary F1, Precision, Recall, ROC-AUC
- **CICIDS subsample:** 100,000 rows (from 16.2M total) for computational feasibility

## Phase 31 Baseline (17 features)

| Holdout | Macro F1 | Binary F1 | Precision | Recall | ROC-AUC |
|---------|---------|----------|-----------|--------|---------|
| NSL-KDD | 0.3444 | 0.8936 | 0.9461 | 0.8466 | 0.8972 |
| UNSW-NB15 | 0.3630 | 0.7360 | 0.8116 | 0.6733 | 0.2402 |
| CICIDS-2018 | 0.4079 | 0.5903 | 0.7911 | 0.4708 | 0.7436 |
| **Average** | **0.3718** | **0.7400** | **0.8496** | **0.6636** | **0.6270** |

**Notes:** This baseline is higher than Phase 31's reported 0.099 because it includes CICIDS data in all training folds (as designed — Phase 31 used NSL+UNSW→CICIDS only; this uses all three leave-one-out permutations). The CICIDS holdout performance remains dominant. UNSW shows ROC-AUC below 0.5, indicating systematic prediction reversal.

## Schema-A: Conservative (8 features)

| Holdout | Macro F1 | Binary F1 | Precision | Recall | ROC-AUC |
|---------|---------|----------|-----------|--------|---------|
| NSL-KDD | 0.3531 | 0.8846 | 0.9445 | 0.8319 | 0.9295 |
| UNSW-NB15 | 0.4193 | 0.7391 | 0.8377 | 0.6613 | 0.5163 |
| CICIDS-2018 | 0.3319 | 0.0046 | 0.4444 | 0.0023 | 0.6431 |
| **Average** | **0.3681** | **0.5428** | **0.7422** | **0.4985** | **0.6963** |

**Key observation:** Conservative schema matches baseline on NSL-KDD and improves UNSW (+0.056) but collapses CICIDS binary F1 to near-zero (0.0046). The lack of derived features hurts CICIDS detection severely.

## Schema-B: Statistical (8 features)

| Holdout | Macro F1 | Binary F1 | Precision | Recall | ROC-AUC |
|---------|---------|----------|-----------|--------|---------|
| NSL-KDD | 0.3107 | 0.8977 | 0.9212 | 0.8755 | 0.8999 |
| UNSW-NB15 | 0.2866 | 0.6113 | 0.8742 | 0.4699 | 0.4893 |
| CICIDS-2018 | 0.3304 | 0.0000 | 0.0000 | 0.0000 | 0.5989 |
| **Average** | **0.3092** | **0.5030** | **0.5985** | **0.4485** | **0.6627** |

**Key observation:** Statistical features alone cannot detect CICIDS attacks at all (binary F1=0.0). The ratio and interaction features encode NSL/UNSW-specific patterns that don't transfer.

## Schema-C: Network-behavior (9 features)

| Holdout | Macro F1 | Binary F1 | Precision | Recall | ROC-AUC |
|---------|---------|----------|-----------|--------|---------|
| NSL-KDD | 0.3219 | 0.8889 | 0.9836 | 0.8108 | 0.9360 |
| UNSW-NB15 | 0.3623 | 0.7377 | 0.8483 | 0.6526 | 0.2203 |
| CICIDS-2018 | 0.4075 | 0.5480 | 0.6555 | 0.4708 | 0.7153 |
| **Average** | **0.3639** | **0.7248** | **0.8291** | **0.6447** | **0.6239** |

**Key observation:** Closest to baseline. Network-behavior features carry the most cross-dataset attack signal. But still fails CICIDS binary detection compared to baseline.

## Schema-D: Minimal Transfer (9 features)

| Holdout | Macro F1 | Binary F1 | Precision | Recall | ROC-AUC |
|---------|---------|----------|-----------|--------|---------|
| NSL-KDD | 0.3167 | 0.8566 | 0.8673 | 0.8463 | 0.9312 |
| UNSW-NB15 | 0.2824 | 0.6666 | 0.8907 | 0.5327 | 0.6618 |
| CICIDS-2018 | 0.3304 | 0.0000 | 0.0000 | 0.0000 | 0.6177 |
| **Average** | **0.3098** | **0.5078** | **0.5860** | **0.4597** | **0.7369** |

**Key observation:** Despite removing the top-5 fingerprint features (flag, both ratios, connection_state, protocol_service_flag), CICIDS binary detection collapses completely. Removing flag and connection_state destroys CICIDS attack signal.

## PCA-8 (8 principal components)

| Holdout | Macro F1 | Binary F1 | Precision | Recall | ROC-AUC |
|---------|---------|----------|-----------|--------|---------|
| NSL-KDD | 0.4730 | 0.8829 | 0.9299 | 0.8405 | 0.9234 |
| UNSW-NB15 | 0.3183 | 0.6670 | 0.8509 | 0.5485 | 0.3063 |
| CICIDS-2018 | 0.3304 | 0.0000 | 0.0000 | 0.0000 | 0.7058 |
| **Average** | **0.3739** | **0.5167** | **0.5936** | **0.4630** | **0.6452** |

**Key observation:** PCA-8 is the **best performer** — average MF1 is +0.6% above baseline. But NSL-KDD MF1 of 0.473 is an outlier: PCA captures dataset-specific structure that happens to help NSL-KDD generalization at the expense of CICIDS.

## RP-8 (8 random projections)

| Holdout | Macro F1 | Binary F1 | Precision | Recall | ROC-AUC |
|---------|---------|----------|-----------|--------|---------|
| NSL-KDD | 0.3454 | 0.8857 | 0.9653 | 0.8183 | 0.9528 |
| UNSW-NB15 | 0.2733 | 0.6622 | 0.7634 | 0.5847 | 0.2717 |
| CICIDS-2018 | 0.2717 | 0.4974 | 0.5273 | 0.4708 | 0.6663 |
| **Average** | **0.2968** | **0.6818** | **0.7520** | **0.6246** | **0.6303** |

**Key observation:** Random projection performs worst overall. Breaking the joint distribution via random projection destroys attack-signal structure without eliminating the dataset fingerprint (99.99% DS-ID accuracy).

## Aggregate Comparison

| Schema | Avg MF1 | Δ MF1 | Avg Bin F1 | Δ Bin F1 | Avg ROC-AUC |
|--------|---------|-------|-----------|----------|-------------|
| Phase31 Baseline | 0.3718 | — | 0.7400 | — | 0.6270 |
| PCA-8 | 0.3739 | **+0.6%** | 0.5167 | −30.2% | 0.6452 |
| Schema-A (Conservative) | 0.3681 | −1.0% | 0.5428 | −26.7% | 0.6963 |
| Schema-C (Network-behavior) | 0.3639 | −2.1% | 0.7248 | −2.0% | 0.6239 |
| Schema-D (Minimal transfer) | 0.3098 | −16.7% | 0.5078 | −31.4% | 0.7369 |
| Schema-B (Statistical) | 0.3092 | −16.8% | 0.5030 | −32.0% | 0.6627 |
| RP-8 | 0.2968 | −20.2% | 0.6818 | −7.9% | 0.6303 |

## Conclusion

- **No schema improves average cross-dataset MF1 by ≥25%** (best: PCA-8 at +0.6%)
- Every schema degrades binary F1 on at least one holdout
- The CICIDS holdout is the most sensitive: removing any feature set significantly hurts CICIDS attack detection
- PCA-8 marginally improves on the baseline (0.3739 vs 0.3718) but still has 100% dataset-ID accuracy
- The secondary criterion is **not met**

The tradeoff is structural: features that encode dataset identity also encode legitimate attack signal, and the two cannot be cleanly separated within the current harmonized space.
