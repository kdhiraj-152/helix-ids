# Holdout Transfer Results

## Experiments Run: 4/4

| Experiment | Model | Accuracy | Macro F1 | Precision | Recall | Gen Gap | Epochs | Time (s) | Sil-DS | Sil-Fam |
|-----------|------:|--------:|---------:|----------:|------:|--------:|------:|--------:|-------:|--------:|
| 3-dataset→TON-IoT (baseline) | baseline | 0.2840 | 0.1733 | 0.1383 | 0.2840 | +0.5217 | 33 | 11.8 | 0.1549 | -0.0405 |
| 3-dataset→TON-IoT (coral) | coral | 0.2936 | 0.1537 | 0.2670 | 0.2936 | +0.4658 | 33 | 16.2 | 0.1624 | -0.0325 |
| 3-dataset→CICIDS2018 (baseline) | baseline | 0.0251 | 0.0541 | 0.0332 | 0.0251 | +0.7718 | 30 | 11.8 | 0.1037 | -0.7357 |
| 3-dataset→CICIDS2018 (coral) | coral | 0.5514 | 0.1684 | 0.6809 | 0.5514 | -0.3876 | 22 | 12.4 | 0.1551 | -0.6026 |
| 3-dataset→NSL-KDD (baseline) | baseline | 0.0749 | 0.0424 | 0.0782 | 0.0749 | +0.6699 | 26 | 10.4 | 0.0270 | -0.5358 |
| 3-dataset→NSL-KDD (coral) | coral | 0.3594 | 0.1083 | 0.6593 | 0.3594 | -0.1951 | 65 | 36.6 | 0.0261 | -0.7779 |
| 3-dataset→UNSW-NB15 (baseline) | baseline | 0.0197 | 0.0113 | 0.0488 | 0.0197 | +0.7827 | 29 | 10.7 | 0.2975 | -0.0680 |
| 3-dataset→UNSW-NB15 (coral) | coral | 0.0210 | 0.0167 | 0.1218 | 0.0210 | +0.5175 | 52 | 26.3 | 0.3498 | -0.0091 |


## Individual Experiment Details

### exp05_holdout_3src_to_ton

- **Source Datasets**: NSL-KDD + UNSW-NB15 + CICIDS2018
- **Target Held-Out**: TON-IoT
- **Train samples**: 15736
- **Test samples**: 10000

**CORAL Δ — Macro F1: -0.0196**

- **Baseline**: Acc=0.2840, F1=0.1733, Prec=0.1383, Rec=0.2840, GenGap=+0.5217, Epochs=33, Sil-DS=0.1549, Sil-Fam=-0.0405
- **Coral**: Acc=0.2936, F1=0.1537, Prec=0.2670, Rec=0.2936, GenGap=+0.4658, Epochs=33, Sil-DS=0.1624, Sil-Fam=-0.0325

### exp06_holdout_3src_to_cicids

- **Source Datasets**: NSL-KDD + UNSW-NB15 + TON-IoT
- **Target Held-Out**: CICIDS2018
- **Train samples**: 18324
- **Test samples**: 10000

**CORAL Δ — Macro F1: +0.1142**

- **Baseline**: Acc=0.0251, F1=0.0541, Prec=0.0332, Rec=0.0251, GenGap=+0.7718, Epochs=30, Sil-DS=0.1037, Sil-Fam=-0.7357
- **Coral**: Acc=0.5514, F1=0.1684, Prec=0.6809, Rec=0.5514, GenGap=-0.3876, Epochs=22, Sil-DS=0.1551, Sil-Fam=-0.6026

### exp07_holdout_3src_to_nsl

- **Source Datasets**: UNSW-NB15 + CICIDS2018 + TON-IoT
- **Target Held-Out**: NSL-KDD
- **Train samples**: 18262
- **Test samples**: 10000

**CORAL Δ — Macro F1: +0.0659**

- **Baseline**: Acc=0.0749, F1=0.0424, Prec=0.0782, Rec=0.0749, GenGap=+0.6699, Epochs=26, Sil-DS=0.0270, Sil-Fam=-0.5358
- **Coral**: Acc=0.3594, F1=0.1083, Prec=0.6593, Rec=0.3594, GenGap=-0.1951, Epochs=65, Sil-DS=0.0261, Sil-Fam=-0.7779

### exp08_holdout_3src_to_unsw

- **Source Datasets**: NSL-KDD + CICIDS2018 + TON-IoT
- **Target Held-Out**: UNSW-NB15
- **Train samples**: 16202
- **Test samples**: 10000

**CORAL Δ — Macro F1: +0.0053**

- **Baseline**: Acc=0.0197, F1=0.0113, Prec=0.0488, Rec=0.0197, GenGap=+0.7827, Epochs=29, Sil-DS=0.2975, Sil-Fam=-0.0680
- **Coral**: Acc=0.0210, F1=0.0167, Prec=0.1218, Rec=0.0210, GenGap=+0.5175, Epochs=52, Sil-DS=0.3498, Sil-Fam=-0.0091

