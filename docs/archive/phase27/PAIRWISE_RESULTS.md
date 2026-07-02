# Pairwise Transfer Results

## Experiments Run: 4/4

| Experiment | Model | Accuracy | Macro F1 | Precision | Recall | Gen Gap | Epochs | Time (s) | Sil-DS | Sil-Fam |
|-----------|------:|--------:|---------:|----------:|------:|--------:|------:|--------:|-------:|--------:|
| NSL-KDD→UNSW-NB15 (baseline) | baseline | 0.1578 | 0.0511 | 0.0357 | 0.1578 | +0.7775 | 31 | 7.8 | 0.3282 | -0.0373 |
| NSL-KDD→UNSW-NB15 (coral) | coral | 0.1593 | 0.0528 | 0.0372 | 0.1593 | +0.4596 | 38 | 12.4 | 0.3898 | -0.0429 |
| UNSW-NB15→CICIDS2018 (baseline) | baseline | 0.6365 | 0.1988 | 0.6694 | 0.6365 | +0.0351 | 28 | 9.6 | 0.1043 | -0.7365 |
| UNSW-NB15→CICIDS2018 (coral) | coral | 0.0380 | 0.0415 | 0.0154 | 0.0380 | +0.1563 | 38 | 18.0 | 0.1823 | -0.5578 |
| CICIDS2018→TON-IoT (baseline) | baseline | 0.2210 | 0.0765 | 0.0664 | 0.2210 | +0.5908 | 24 | 5.3 | 0.0468 | -0.6179 |
| CICIDS2018→TON-IoT (coral) | coral | 0.3931 | 0.2531 | 0.3475 | 0.3931 | +0.0551 | 34 | 9.3 | 0.2513 | -0.0880 |
| TON-IoT→NSL-KDD (baseline) | baseline | 0.1954 | 0.0893 | 0.4850 | 0.1954 | +0.5596 | 46 | 14.9 | 0.1312 | 0.1374 |
| TON-IoT→NSL-KDD (coral) | coral | 0.3517 | 0.1296 | 0.3427 | 0.3517 | +0.0132 | 39 | 18.1 | 0.1714 | 0.1344 |


## Individual Experiment Details

### exp01_pairwise_nsl_to_unsw

- **Source**: NSL-KDD
- **Target**: UNSW-NB15
- **Train samples**: 10068
- **Test samples**: 10000

**CORAL Δ — Macro F1: +0.0017**

- **Baseline**: Acc=0.1578, F1=0.0511, Prec=0.0357, Rec=0.1578, GenGap=+0.7775, Epochs=31, Sil-DS=0.3282, Sil-Fam=-0.0373
- **Coral**: Acc=0.1593, F1=0.0528, Prec=0.0372, Rec=0.1593, GenGap=+0.4596, Epochs=38, Sil-DS=0.3898, Sil-Fam=-0.0429

### exp02_pairwise_unsw_to_cicids

- **Source**: UNSW-NB15
- **Target**: CICIDS2018
- **Train samples**: 15442
- **Test samples**: 10000

**CORAL Δ — Macro F1: -0.1574**

- **Baseline**: Acc=0.6365, F1=0.1988, Prec=0.6694, Rec=0.6365, GenGap=+0.0351, Epochs=28, Sil-DS=0.1043, Sil-Fam=-0.7365
- **Coral**: Acc=0.0380, F1=0.0415, Prec=0.0154, Rec=0.0380, GenGap=+0.1563, Epochs=38, Sil-DS=0.1823, Sil-Fam=-0.5578

### exp03_pairwise_cicids_to_ton

- **Source**: CICIDS2018
- **Target**: TON-IoT
- **Train samples**: 8099
- **Test samples**: 10000

**CORAL Δ — Macro F1: +0.1766**

- **Baseline**: Acc=0.2210, F1=0.0765, Prec=0.0664, Rec=0.2210, GenGap=+0.5908, Epochs=24, Sil-DS=0.0468, Sil-Fam=-0.6179
- **Coral**: Acc=0.3931, F1=0.2531, Prec=0.3475, Rec=0.3931, GenGap=+0.0551, Epochs=34, Sil-DS=0.2513, Sil-Fam=-0.0880

### exp04_pairwise_ton_to_nsl

- **Source**: TON-IoT
- **Target**: NSL-KDD
- **Train samples**: 15000
- **Test samples**: 10000

**CORAL Δ — Macro F1: +0.0403**

- **Baseline**: Acc=0.1954, F1=0.0893, Prec=0.4850, Rec=0.1954, GenGap=+0.5596, Epochs=46, Sil-DS=0.1312, Sil-Fam=0.1374
- **Coral**: Acc=0.3517, F1=0.1296, Prec=0.3427, Rec=0.3517, GenGap=+0.0132, Epochs=39, Sil-DS=0.1714, Sil-Fam=0.1344

