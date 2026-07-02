# CORAL Effectiveness Analysis

## Summary Statistics

- **Total experiments**: 8
- **Wins (CORAL better)**: 6
- **Losses (Baseline better)**: 2
- **Ties**: 0
- **Average Macro F1 Δ**: +0.0284
- **Average Accuracy Δ**: +0.0691
- **Average Silhouette Dataset Δ**: +0.0618


## Per-Experiment Δ

| Experiment | Δ Macro F1 | Δ Accuracy | Δ Sil-DS | Win/Loss |
|-----------|----------:|----------:|--------:|----------:|
| exp01_pairwise_nsl_to_unsw | +0.0017 | +0.0015 | +0.0616 | WIN |
| exp02_pairwise_unsw_to_cicids | -0.1574 | -0.5985 | +0.0780 | LOSS |
| exp03_pairwise_cicids_to_ton | +0.1766 | +0.1721 | +0.2046 | WIN |
| exp04_pairwise_ton_to_nsl | +0.0403 | +0.1563 | +0.0403 | WIN |
| exp05_holdout_3src_to_ton | -0.0196 | +0.0096 | +0.0075 | LOSS |
| exp06_holdout_3src_to_cicids | +0.1142 | +0.5263 | +0.0514 | WIN |
| exp07_holdout_3src_to_nsl | +0.0659 | +0.2845 | -0.0008 | WIN |
| exp08_holdout_3src_to_unsw | +0.0053 | +0.0013 | +0.0523 | WIN |


## Decision

**GO for Phase 28 (DANN domain-adversarial training)**

- Average macro F1 improvement = 2.84% (< 20% threshold)
- Average improvement < 10% → failure condition triggered
