# Phase 43C — CORAL Domain Alignment Validation

**Date:** 2026-06-26
**Experiment:** Phase 43C, RP-2 HELIX-IDS
**Objective:** Determine whether covariance alignment (CORAL) can eliminate the remaining dataset fingerprint after Phase 43B feature ablation.

## Executive Summary

For each dataset pair, CORAL transformation was applied to align source covariance to target covariance. Dataset-ID classifiers (Logistic Regression, 5-fold CV) were trained on both raw and CORAL-aligned features to measure whether second-order statistics drive the remaining fingerprint.
Four conditions tested per pair: (1) Raw features, (2) CORAL alignment A→B, (3) CORAL alignment B→A, (4) Average of both CORAL directions. Each condition tested on full canonical features and on top-5 ablated features.

## Per-Pair Results

### CICIDS2018 vs TON-IoT

**Condition: Full canonical features**

- Raw DOS:           0.0001
- CORAL A→B DOS:     0.4878
- CORAL B→A DOS:     0.4879
- CORAL Avg DOS:     0.4878
- Δ DOS (Avg - Raw): +0.4878

**Condition: Top-5 ablated features**

- Raw DOS:           0.0166
- CORAL A→B DOS:     0.4878
- CORAL B→A DOS:     0.4878
- CORAL Avg DOS:     0.4878
- Δ DOS (Avg - Raw): +0.4712

---

### NSL-KDD vs CICIDS2018

**Condition: Full canonical features**

- Raw DOS:           0.0000
- CORAL A→B DOS:     0.3487
- CORAL B→A DOS:     0.3487
- CORAL Avg DOS:     0.3487
- Δ DOS (Avg - Raw): +0.3487

**Condition: Top-5 ablated features**

- Raw DOS:           0.2787
- CORAL A→B DOS:     0.3487
- CORAL B→A DOS:     0.3487
- CORAL Avg DOS:     0.3487
- Δ DOS (Avg - Raw): +0.0700

---

### NSL-KDD vs TON-IoT

**Condition: Full canonical features**

- Raw DOS:           0.0000
- CORAL A→B DOS:     0.3599
- CORAL B→A DOS:     0.3599
- CORAL Avg DOS:     0.3599
- Δ DOS (Avg - Raw): +0.3599

**Condition: Top-5 ablated features**

- Raw DOS:           0.0075
- CORAL A→B DOS:     0.3599
- CORAL B→A DOS:     0.3599
- CORAL Avg DOS:     0.3599
- Δ DOS (Avg - Raw): +0.3524

---

### NSL-KDD vs UNSW-NB15

**Condition: Full canonical features**

- Raw DOS:           0.0000
- CORAL A→B DOS:     0.4181
- CORAL B→A DOS:     0.4181
- CORAL Avg DOS:     0.4181
- Δ DOS (Avg - Raw): +0.4180

**Condition: Top-5 ablated features**

- Raw DOS:           0.1269
- CORAL A→B DOS:     0.4181
- CORAL B→A DOS:     0.4181
- CORAL Avg DOS:     0.4181
- Δ DOS (Avg - Raw): +0.2912

---

### UNSW-NB15 vs CICIDS2018

**Condition: Full canonical features**

- Raw DOS:           0.0000
- CORAL A→B DOS:     0.4270
- CORAL B→A DOS:     0.4270
- CORAL Avg DOS:     0.4270
- Δ DOS (Avg - Raw): +0.4270

**Condition: Top-5 ablated features**

- Raw DOS:           0.1674
- CORAL A→B DOS:     0.4270
- CORAL B→A DOS:     0.4270
- CORAL Avg DOS:     0.4270
- Δ DOS (Avg - Raw): +0.2596

---

### UNSW-NB15 vs TON-IoT

**Condition: Full canonical features**

- Raw DOS:           0.0000
- CORAL A→B DOS:     0.4390
- CORAL B→A DOS:     0.4390
- CORAL Avg DOS:     0.4390
- Δ DOS (Avg - Raw): +0.4390

**Condition: Top-5 ablated features**

- Raw DOS:           0.0924
- CORAL A→B DOS:     0.4390
- CORAL B→A DOS:     0.4390
- CORAL Avg DOS:     0.4390
- Δ DOS (Avg - Raw): +0.3466

---

## Main Comparison: Raw vs CORAL

| Dataset Pair | DOS Raw | DOS CORAL Avg | Δ DOS | CORAL A→B | CORAL B→A |
|---|---|---|---|---|---|
| CICIDS2018 vs TON-IoT | 0.0001 | 0.4878 | +0.4878 | 0.4878 | 0.4879 |
| NSL-KDD vs CICIDS2018 | 0.0000 | 0.3487 | +0.3487 | 0.3487 | 0.3487 |
| NSL-KDD vs TON-IoT | 0.0000 | 0.3599 | +0.3599 | 0.3599 | 0.3599 |
| NSL-KDD vs UNSW-NB15 | 0.0000 | 0.4181 | +0.4180 | 0.4181 | 0.4181 |
| UNSW-NB15 vs CICIDS2018 | 0.0000 | 0.4270 | +0.4270 | 0.4270 | 0.4270 |
| UNSW-NB15 vs TON-IoT | 0.0000 | 0.4390 | +0.4390 | 0.4390 | 0.4390 |

## Top-5 Ablated: Raw vs CORAL

| Dataset Pair | DOS Ablated Raw | DOS Ablated + CORAL | Δ DOS |
|---|---|---|---|
| CICIDS2018 vs TON-IoT | 0.0166 | 0.4878 | +0.4712 |
| NSL-KDD vs CICIDS2018 | 0.2787 | 0.3487 | +0.0700 |
| NSL-KDD vs TON-IoT | 0.0075 | 0.3599 | +0.3524 |
| NSL-KDD vs UNSW-NB15 | 0.1269 | 0.4181 | +0.2912 |
| UNSW-NB15 vs CICIDS2018 | 0.1674 | 0.4270 | +0.2596 |
| UNSW-NB15 vs TON-IoT | 0.0924 | 0.4390 | +0.3466 |

## Aggregate Results

### Aggregate Table (averaged across all pairs)

| Condition | Mean DOS Raw | Mean DOS CORAL | Δ DOS |
| --- | --- | --- | --- |
| Full canonical features | 0.0000 | 0.4134 | +0.4134 |
| Top-5 ablated features | 0.1149 | 0.4134 | +0.2985 |

## Per-Pair DOS Changes

| Dataset Pair | Δ DOS (Full) | Δ DOS (Top-5 Ablated) | Full DOS Coral Avg | Ablated DOS Coral Avg |
|---|---|---|---|---|
| CICIDS2018 vs TON-IoT | +0.4878 | +0.4712 | 0.4878 | 0.4878 |
| NSL-KDD vs CICIDS2018 | +0.3487 | +0.0700 | 0.3487 | 0.3487 |
| NSL-KDD vs TON-IoT | +0.3599 | +0.3524 | 0.3599 | 0.3599 |
| NSL-KDD vs UNSW-NB15 | +0.4180 | +0.2912 | 0.4181 | 0.4181 |
| UNSW-NB15 vs CICIDS2018 | +0.4270 | +0.2596 | 0.4270 | 0.4270 |
| UNSW-NB15 vs TON-IoT | +0.4390 | +0.3466 | 0.4390 | 0.4390 |

## Interpretation

- **Full features:** Raw mean DOS = 0.0000, CORAL mean DOS = 0.4134, Δ = +0.4134
- **Top-5 ablated:** Raw mean DOS = 0.1149, CORAL mean DOS = 0.4134, Δ = +0.2985

### Decision Criteria

**CORAL produces substantial improvement (Δ DOS > 0.10) on full features.** This suggests covariance mismatch is a significant component of the dataset fingerprint for some pairs.
- 6 pairs with large improvement (Δ > 0.10)
- 0 pairs with moderate improvement (0.05 < Δ ≤ 0.10)
- 0 pairs with minimal improvement (Δ ≤ 0.05)

### Phase 36 Threshold Analysis

- **Full features + CORAL:** 6/6 pairs reach DOS ≥ 0.3. Mean DOS = 0.4134
- **Top-5 ablated + CORAL:** 6/6 pairs reach DOS ≥ 0.3. Mean DOS = 0.4134

### Verdict

**CORAL SUCCESSFULLY removes dataset fingerprint** for some conditions (best mean DOS = 0.4134 ≥ 0.30). Covariance alignment, particularly when combined with feature ablation, achieves the Phase 36 benchmark-repair threshold.

---

*Report generated by Phase 43C — CORAL Domain Alignment Validation*
