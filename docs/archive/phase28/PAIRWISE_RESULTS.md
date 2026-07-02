# Pairwise Transfer Results (DANN)

## Experiments Run: 4/4

| Experiment | Lambda | Accuracy | Macro F1 | Precision | Recall | Gen Gap | Epochs | Time (s) | Sil-DS | Sil-Fam |
|-----------|------:|--------:|---------:|----------:|------:|--------:|------:|--------:|-------:|--------:|
| NSL-KDD→UNSW-NB15 | 0.500 | 0.1870 | 0.0645 | 0.3228 | 0.1870 | +0.0052 | 42 | 24.5 | 0.4444 | -0.0789 |
| UNSW-NB15→CICIDS2018 | 0.010 | 0.4453 | 0.1299 | 0.6534 | 0.4453 | -0.2320 | 26 | 23.0 | 0.1786 | -0.5683 |
| CICIDS2018→TON-IoT | 0.250 | 0.2563 | 0.1349 | 0.3144 | 0.2563 | +0.1440 | 31 | 15.4 | 0.1250 | -0.2345 |
| TON-IoT→NSL-KDD | 0.500 | 0.3545 | 0.1498 | 0.3297 | 0.3545 | +0.1668 | 25 | 22.7 | 0.1215 | 0.1607 |


## Individual Experiment Details

### exp01_pairwise_nsl_to_unsw

- **Source**: NSL-KDD
- **Target**: UNSW-NB15
- **λ**: 0.5
- **Train samples**: 14946
- **Test samples**: 10000
- **DANN**: Acc=0.1870, F1=0.0645, Prec=0.3228, Rec=0.1870, GenGap=+0.0052, Epochs=42, Sil-DS=0.4444, Sil-Fam=-0.0789

### exp02_pairwise_unsw_to_cicids

- **Source**: UNSW-NB15
- **Target**: CICIDS2018
- **λ**: 0.01
- **Train samples**: 23442
- **Test samples**: 10000
- **DANN**: Acc=0.4453, F1=0.1299, Prec=0.6534, Rec=0.4453, GenGap=-0.2320, Epochs=26, Sil-DS=0.1786, Sil-Fam=-0.5683

### exp03_pairwise_cicids_to_ton

- **Source**: CICIDS2018
- **Target**: TON-IoT
- **λ**: 0.25
- **Train samples**: 11837
- **Test samples**: 10000
- **DANN**: Acc=0.2563, F1=0.1349, Prec=0.3144, Rec=0.2563, GenGap=+0.1440, Epochs=31, Sil-DS=0.1250, Sil-Fam=-0.2345

### exp04_pairwise_ton_to_nsl

- **Source**: TON-IoT
- **Target**: NSL-KDD
- **λ**: 0.5
- **Train samples**: 24220
- **Test samples**: 10000
- **DANN**: Acc=0.3545, F1=0.1498, Prec=0.3297, Rec=0.3545, GenGap=+0.1668, Epochs=25, Sil-DS=0.1215, Sil-Fam=0.1607

