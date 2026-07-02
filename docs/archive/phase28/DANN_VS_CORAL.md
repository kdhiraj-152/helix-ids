# DANN vs CORAL — Multi-Seed Comparison

## Per-Experiment DANN vs CORAL Best F1

| Experiment | CORAL Best F1 | DANN μ F1 | DANN σ | Δ | DANN Wins/5 |
|-----------|-------------:|---------:|------:|--:|----------:|
| exp01_pairwise_nsl_to_unsw | 0.0528 | 0.0943 | 0.0121 | +0.0415 🟢 | 5/5 |
| exp02_pairwise_unsw_to_cicids | 0.0415 | 0.1395 | 0.0026 | +0.0980 🟢 | 5/5 |
| exp03_pairwise_cicids_to_ton | 0.2531 | 0.1465 | 0.0587 | -0.1066 🔴 | 0/5 |
| exp04_pairwise_ton_to_nsl | 0.1296 | 0.1425 | 0.0715 | +0.0129 🟢 | 2/5 |
| exp05_holdout_3src_to_ton | 0.1537 | 0.1392 | 0.0445 | -0.0145 🔴 | 1/5 |
| exp06_holdout_3src_to_cicids | 0.1684 | 0.1902 | 0.0680 | +0.0218 🟢 | 3/5 |
| exp07_holdout_3src_to_nsl | 0.1083 | 0.1549 | 0.0226 | +0.0466 🟢 | 5/5 |
| exp08_holdout_3src_to_unsw | 0.0167 | 0.0721 | 0.0069 | +0.0554 🟢 | 5/5 |

**Total CORAL wins**: 26/40 (65.0%)

⚠️ **DANN edges CORAL** but not decisively.
