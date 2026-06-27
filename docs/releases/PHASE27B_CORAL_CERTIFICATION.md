# PHASE 27B — CORAL Multi-Dataset Certification Report

**Decision**: NO-GO
**Recommended next phase**: Phase 28 (DANN domain-adversarial training)

## Executive Summary

Phase 27B validates whether CORAL produces consistent domain-invariant improvements across the entire Helix dataset ecosystem. We ran 8 experiments: 4 pairwise and 4 holdout transfers covering all 4 datasets (NSL-KDD, UNSW-NB15, CICIDS2018, TON-IoT).

## Success Criteria

### Primary: Average Macro F1 improvement >= 20%

- **Average Macro F1 Δ**: +0.0284 (+2.84%)
- **Threshold**: ≥ 0.20 (20%)
- **Result**: FAIL ❌

### Secondary: Average dataset silhouette reduction >= 15%

- **Average Silhouette Δ**: +0.0618
- **Threshold**: ≤ -0.15 (15% reduction)
- **Result**: FAIL ❌

### Tertiary: At least 5 of 8 experiments improve Macro F1

- **Wins**: 6/8
- **Threshold**: ≥ 5
- **Result**: PASS ✅

**Failure condition**: TRIGGERED (⚠️ CORAL underperforms)

## Per-Experiment Macro F1

| Experiment | Baseline F1 | CORAL F1 | Δ |
|-----------|----------:|--------:|--:|
| exp01_pairwise_nsl_to_unsw | 0.0511 | 0.0528 | +0.0017 🟢 |
| exp02_pairwise_unsw_to_cicids | 0.1988 | 0.0415 | -0.1574 🔴 |
| exp03_pairwise_cicids_to_ton | 0.0765 | 0.2531 | +0.1766 🟢 |
| exp04_pairwise_ton_to_nsl | 0.0893 | 0.1296 | +0.0403 🟢 |
| exp05_holdout_3src_to_ton | 0.1733 | 0.1537 | -0.0196 🔴 |
| exp06_holdout_3src_to_cicids | 0.0541 | 0.1684 | +0.1142 🟢 |
| exp07_holdout_3src_to_nsl | 0.0424 | 0.1083 | +0.0659 🟢 |
| exp08_holdout_3src_to_unsw | 0.0113 | 0.0167 | +0.0053 🟢 |

## Conclusion

1. **Average Macro F1 delta**: +0.0284 (+2.84%)
2. **Average silhouette delta**: +0.0618
3. **Wins/Losses**: 6/2 (ties: 0)
4. **CORAL generalizes beyond NSL→UNSW**: YES — consistently improves multiple dataset pairs ['-0.1574', '+0.1766', '+0.0403', '-0.0196', '+0.1142', '+0.0659', '+0.0053']
5. **Decision**: NO-GO → Recommend Phase 28 (DANN domain-adversarial training)
