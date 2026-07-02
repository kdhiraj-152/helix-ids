# PHASE 28C — Production Certification Report

**Decision**: HOLD ⚠️

## Executive Summary

Phase 28C validates whether Phase 28A DANN gains persist at full production scale across 5 random seeds and 8 experiments (40 total runs).

- **Global Mean Macro F1**: 0.1349
- **Global Std Macro F1**: 0.0531
- **95% Confidence Interval**: [0.1185, 0.1514]
- **Win Rate vs CORAL**: 65.0% (26/40)
- **Win Rate vs Baseline (26B)**: 90.0% (36/40)

## Success Criteria

### C1: Mean Macro F1 >= 0.12

- **Value**: 0.1349
- **Result**: ✅

### C2: Std Deviation <= 0.03

- **Value**: 0.0531
- **Result**: ❌

### C3: DANN beats CORAL >= 75%

- **Value**: 65.0%
- **Result**: ❌

### C4: DANN beats Baseline >= 75%

- **Value**: 90.0%
- **Result**: ✅

### Overall

**Criteria NOT met**: Std Deviation <= 0.03, DANN beats CORAL >= 75%

## DANN vs CORAL — Per-Experiment

| Experiment | CORAL | DANN μ | DANN σ | Δ | Wins/5 |
|-----------|-----:|------:|------:|--:|------:|
| exp01_pairwise_nsl_to_unsw | 0.0528 | 0.0943 | 0.0121 | +0.0415 🟢 | 5/5 |
| exp02_pairwise_unsw_to_cicids | 0.0415 | 0.1395 | 0.0026 | +0.0980 🟢 | 5/5 |
| exp03_pairwise_cicids_to_ton | 0.2531 | 0.1465 | 0.0587 | -0.1066 🔴 | 0/5 |
| exp04_pairwise_ton_to_nsl | 0.1296 | 0.1425 | 0.0715 | +0.0129 🟢 | 2/5 |
| exp05_holdout_3src_to_ton | 0.1537 | 0.1392 | 0.0445 | -0.0145 🔴 | 1/5 |
| exp06_holdout_3src_to_cicids | 0.1684 | 0.1902 | 0.0680 | +0.0218 🟢 | 3/5 |
| exp07_holdout_3src_to_nsl | 0.1083 | 0.1549 | 0.0226 | +0.0466 🟢 | 5/5 |
| exp08_holdout_3src_to_unsw | 0.0167 | 0.0721 | 0.0069 | +0.0554 🟢 | 5/5 |

## Expected Production Macro F1 Range

- **Expected range**: 0.1185 – 0.1514 (95% CI)
- **Worst-case (min across seeds/experiments)**: 0.0669
- **Best-case (max across seeds/experiments)**: 0.2534
- **σ estimate for macro F1 across experiments**: 0.0531

✅ **Expected Macro F1** is above the 0.12 production threshold.

## Final Verdict

### 1. Is DANN stable?
**NO ❌** — σ = 0.0531 (threshold 0.03)

### 2. Is DANN production-ready?
**YES ✅** — μ = 0.1349 (threshold 0.12)

### 3. Expected production Macro F1 range
**95% CI**: [0.1185, 0.1514]

### 4. GO / NO-GO for deployment training
**HOLD ⚠️** — Marginal pass on gains but needs review.

---
*Generated on 2026-06-23 01:06:41 IST*
