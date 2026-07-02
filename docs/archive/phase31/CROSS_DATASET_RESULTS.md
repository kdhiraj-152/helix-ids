# Cross-Dataset Generalization — Phase 31

**Date:** 2026-06-24
**Experiment:** Leave-one-dataset-out validation under three conditions

## Protocol

For each modification (Baseline, Ablate Top-5, Quantile Norm):
1. Train on two datasets
2. Evaluate on the held-out third dataset
3. Report Macro F1, Binary F1, Precision, Recall

**Classifier:** LogisticRegression (max_iter=1000, L2 regularization)

## Baseline — All 17 Features

| Holdout | Macro F1 | Binary F1 | Precision | Recall |
|---------|---------|----------|-----------|--------|
| NSL-KDD | 0.0998 | 0.0225 | 0.9524 | 0.0114 |
| UNSW-NB15 | 0.0025 | 0.7503 | 0.6006 | 0.9992 |
| CICIDS-2018 | **0.1935** | 0.0016 | 0.0011 | 0.0029 |
| **Average** | **0.0986** | 0.2581 | 0.5180 | 0.3378 |

### Analysis by Holdout

**NSL-KDD holdout** (trained on UNSW + CICIDS):
- Binary recall is very low (0.011): the model almost never predicts "attack" on NSL-KDD
- Binary precision is high (0.952): when it does predict attack, it's usually correct
- All 7-class Macro F1 is low (0.100): the model cannot distinguish attack subtypes

**UNSW-NB15 holdout** (trained on NSL + CICIDS):
- Macro F1 collapses to 0.0025: near-random classification across 7 classes
- Binary F1 is reasonable (0.750): the model can tell attack from normal on UNSW
- Recall is 0.999: it classifies nearly everything as attack (high true positive, low precision)

**CICIDS-2018 holdout** (trained on NSL + UNSW):
- Best Macro F1 at 0.194: CICIDS has the richest attack diversity
- Binary F1 is near zero (0.002): the model almost never predicts "attack" on CICIDS
- This is the most severe generalization failure — the model fails to trigger on CICIDS attacks

## Ablate Top-5 Features

Remove `{flag, src_dst_bytes_ratio, dst_src_bytes_ratio, connection_state, protocol_service_flag}`.

| Holdout | Macro F1 | Binary F1 | Change vs Baseline |
|---------|---------|----------|-------------------|
| NSL-KDD | 0.0907 | 0.1828 | −9.2% |
| UNSW-NB15 | 0.1110 | 0.7182 | +4338.8% |
| CICIDS-2018 | 0.0301 | 0.0374 | −84.4% |
| **Average** | **0.0773** | **0.3128** | **−21.6%** |

### Analysis

- UNSW-NB15 Macro F1 improves dramatically (0.003 → 0.111) — the ablated features were actively interfering with UNSW generalization
- CICIDS Macro F1 collapses 84% (0.194 → 0.030) — these features are critical for CICIDS detection
- **Zero-sum tradeoff:** removing dataset-ID features helps UNSW but hurts CICIDS

## Quantile Norm

| Holdout | Macro F1 | Binary F1 | Change vs Baseline |
|---------|---------|----------|-------------------|
| NSL-KDD | 0.0995 | 0.0102 | −0.3% |
| UNSW-NB15 | 0.0578 | 0.7513 | +2229.0% |
| CICIDS-2018 | 0.1094 | 0.0470 | −43.4% |
| **Average** | **0.0889** | **0.2695** | **−9.9%** |

### Analysis

- Quantile normalization slightly helps UNSW but hurts CICIDS
- The pattern mirrors feature ablation: suppressing dataset-specific distributional structure helps some datasets at the expense of others
- Average Macro F1 drops 9.9% — net negative

## Cross-Dataset Macro F1 Comparison

| Method | NSL-KDD | UNSW-NB15 | CICIDS | **Average** |
|--------|---------|-----------|--------|:-----------:|
| Baseline | 0.100 | 0.002 | 0.194 | **0.099** |
| Ablate Top-5 | 0.091 | **0.111** | 0.030 | **0.077** |
| Quantile Norm | 0.100 | 0.058 | 0.109 | **0.089** |

## Plot

See `plots/cross_dataset_f1.png`

## Conclusion

**No intervention improves average cross-dataset Macro F1.** The best result is the baseline (0.099). Both feature ablation and normalization produce a zero-sum tradeoff: improving generalization on UNSW at the expense of CICIDS, or vice versa. This confirms that cross-dataset generalization is fundamentally bounded by the harmonized 17-feature representation space.
