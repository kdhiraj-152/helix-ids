# PHASE 28A — DANN Domain-Adversarial Certification Report

**Decision**: HOLD
**Recommended next phase**: Phase 28B — try alternative DANN variants or hyperparameters

## Executive Summary

Phase 28A validates whether domain-adversarial training (DANN) can eliminate dataset-specific representations and improve cross-dataset generalization where CORAL failed. We ran 8 experiments (4 pairwise + 4 holdout) across 5 lambda values each (40 total runs).

- **Average Macro F1**: 0.1311 (vs baseline 0.0263)
- **Average Macro F1 Δ vs Phase 26B**: +0.1048 (+397.81%)
- **Average Macro F1 Δ vs Phase 27B CORAL**: +0.0156
- **Average Silhouette Dataset**: 0.2232 (vs baseline 0.1492, -49.6% reduction)
- **Average Silhouette Family**: -0.1558 (vs baseline -0.3293)
- **Best Macro F1**: 0.2113
- **Worst Macro F1**: 0.0617
- **Wins/Losses vs Baseline**: 7/1
- **Domain collapse detected**: NO

## Success Criteria

### Primary: Average Macro F1 improvement >= 15% vs Phase 26B baseline

- **Improvement**: +397.81%
- **Threshold**: >= 15%
- **Result**: PASS ✅

### Secondary: Dataset silhouette reduction >= 30%

- **Silhouette reduction**: -49.6%
- **Threshold**: >= 30%
- **Result**: FAIL ❌

### Tertiary: Family silhouette increase >= 20%

- **Family silhouette (baseline)**: -0.3293
- **Family silhouette (DANN)**: -0.1558
- **Result**: EVALUATED (IMPROVED)

### Production Candidate: Average Macro F1 > 0.10

- **Average Macro F1**: 0.1311
- **Threshold**: > 0.10
- **Result**: PASS ✅

**Failure condition**: NOT TRIGGERED ✅

## Per-Experiment Macro F1

| Experiment | λ (best) | Phase 26B F1 | CORAL F1 | DANN F1 | Δ vs 26B | Δ vs CORAL |
|-----------|--------:|------------:|--------:|-------:|--------:|----------:|
| exp01_pairwise_nsl_to_unsw | 0.500 | 0.1068 | 0.0528 | 0.0645 | -0.0423 🔴 | +0.0117 🟢 |
| exp02_pairwise_unsw_to_cicids | 0.010 | 0.0196 | 0.0415 | 0.1299 | +0.1103 🟢 | +0.0884 🟢 |
| exp03_pairwise_cicids_to_ton | 0.250 | 0.0633 | 0.2531 | 0.1349 | +0.0716 🟢 | -0.1182 🔴 |
| exp04_pairwise_ton_to_nsl | 0.500 | 0.0067 | 0.1296 | 0.1498 | +0.1431 🟢 | +0.0202 🟢 |
| exp05_holdout_3src_to_ton | 0.250 | 0.0119 | 0.1537 | 0.2113 | +0.1994 🟢 | +0.0576 🟢 |
| exp06_holdout_3src_to_cicids | 0.500 | 0.0000 | 0.1684 | 0.1870 | +0.1870 🟢 | +0.0186 🟢 |
| exp07_holdout_3src_to_nsl | 0.500 | 0.0004 | 0.1083 | 0.1099 | +0.1095 🟢 | +0.0016 🟢 |
| exp08_holdout_3src_to_unsw | 0.010 | 0.0020 | 0.0167 | 0.0617 | +0.0597 🟢 | +0.0450 🟢 |

## Conclusions

### 1. DANN vs Baseline

DANN outperforms the Phase 26B baseline (Δ = +0.1048, +397.81%). The primary success criterion is MET.

### 2. DANN vs CORAL

DANN marginally beats CORAL (Δ = +0.0156). DANN wins 7/8 experiments over CORAL.

### 3. Dataset Silhouette Change

Dataset silhouette INCREASED from 0.1492 to 0.2232. DANN did not reduce dataset-specific clustering.

### 4. Family Silhouette Change

Family silhouette improved from -0.3293 to -0.1558, indicating better attack-family separation. However, values remain negative in most cases, suggesting cluster coherence is still poor.

### 5. Domain Invariance Assessment

Domain invariance was NOT achieved. Average silhouette_dataset = 0.2232 indicates source and target embeddings remain clearly separable.

### 6. GO / NO-GO Decision

**Decision**: HOLD

DANN shows partial improvement (+397.81%) but does not meet all success criteria. Consider Phase 28B with modified DANN architecture or alternative approaches.

## Lambda Sensitivity

| Experiment | Best λ | Best F1 |
|-----------|------:|-------:|
| exp01_pairwise_nsl_to_unsw | 0.500 | 0.0645 |
| exp02_pairwise_unsw_to_cicids | 0.010 | 0.1299 |
| exp03_pairwise_cicids_to_ton | 0.250 | 0.1349 |
| exp04_pairwise_ton_to_nsl | 0.500 | 0.1498 |
| exp05_holdout_3src_to_ton | 0.250 | 0.2113 |
| exp06_holdout_3src_to_cicids | 0.500 | 0.1870 |
| exp07_holdout_3src_to_nsl | 0.500 | 0.1099 |
| exp08_holdout_3src_to_unsw | 0.010 | 0.0617 |

Most frequent best λ: 0.5 (4/8 experiments)
