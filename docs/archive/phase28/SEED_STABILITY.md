# Seed Stability Analysis

## Per-Seed Macro F1 Across All Experiments

**Seeds tested**: [42, 1337, 2026, 7777, 9999]

| Experiment | Seed=42 | Seed=1337 | Seed=2026 | Seed=7777 | Seed=9999 | μ | σ |
|-----------|--------:|----------:|----------:|----------:|----------:|--:|--:|
| exp01_pairwise_nsl_to_unsw | 0.1014 | 0.0918 | 0.1099 | 0.0778 | 0.0907 | 0.0943 | 0.0121 |
| exp02_pairwise_unsw_to_cicids | 0.1399 | 0.1386 | 0.1400 | 0.1358 | 0.1430 | 0.1395 | 0.0026 |
| exp03_pairwise_cicids_to_ton | 0.1931 | 0.2204 | 0.1353 | 0.0876 | 0.0959 | 0.1465 | 0.0587 |
| exp04_pairwise_ton_to_nsl | 0.0819 | 0.0827 | 0.1266 | 0.2534 | 0.1680 | 0.1425 | 0.0715 |
| exp05_holdout_3src_to_ton | 0.1482 | 0.1208 | 0.0871 | 0.1319 | 0.2080 | 0.1392 | 0.0445 |
| exp06_holdout_3src_to_cicids | 0.2235 | 0.2486 | 0.1167 | 0.2460 | 0.1164 | 0.1902 | 0.0680 |
| exp07_holdout_3src_to_nsl | 0.1717 | 0.1475 | 0.1210 | 0.1560 | 0.1786 | 0.1549 | 0.0226 |
| exp08_holdout_3src_to_unsw | 0.0839 | 0.0711 | 0.0714 | 0.0669 | 0.0674 | 0.0721 | 0.0069 |

## Global Seed Variance

- **Global μ**: 0.1349
- **Global σ**: 0.0531
- **95% CI**: [0.1185, 0.1514]
- **❌ Seed stability FAIL**: σ = 0.0531 > 0.03
