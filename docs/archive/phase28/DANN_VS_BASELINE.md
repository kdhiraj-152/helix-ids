# DANN vs Baseline (Phase 26B) — Multi-Seed Comparison

## Per-Experiment DANN vs Baseline

| Experiment | 26B Baseline F1 | DANN μ F1 | Δ | DANN Wins/5 |
|-----------|---------------:|---------:|--:|----------:|
| exp01_pairwise_nsl_to_unsw | 0.1068 | 0.0943 | -0.0125 🔴 | 1/5 |
| exp02_pairwise_unsw_to_cicids | 0.0196 | 0.1395 | +0.1199 🟢 | 5/5 |
| exp03_pairwise_cicids_to_ton | 0.0633 | 0.1465 | +0.0832 🟢 | 5/5 |
| exp04_pairwise_ton_to_nsl | 0.0067 | 0.1425 | +0.1358 🟢 | 5/5 |
| exp05_holdout_3src_to_ton | 0.0119 | 0.1392 | +0.1273 🟢 | 5/5 |
| exp06_holdout_3src_to_cicids | 0.0000 | 0.1902 | +0.1902 🟢 | 5/5 |
| exp07_holdout_3src_to_nsl | 0.0004 | 0.1549 | +0.1545 🟢 | 5/5 |
| exp08_holdout_3src_to_unsw | 0.0020 | 0.0721 | +0.0701 🟢 | 5/5 |

**Total Baseline wins**: 36/40 (90.0%)

✅ **DANN dominates the Phase 26B baseline** at production scale.
