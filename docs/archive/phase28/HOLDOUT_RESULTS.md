# Holdout Transfer Results (DANN)

## Experiments Run: 4/4

| Experiment | Lambda | Accuracy | Macro F1 | Precision | Recall | Gen Gap | Epochs | Time (s) | Sil-DS | Sil-Fam |
|-----------|------:|--------:|---------:|----------:|------:|--------:|------:|--------:|-------:|--------:|
| 3-dataset→TON-IoT | 0.250 | 0.3239 | 0.2113 | 0.3114 | 0.3239 | -0.0545 | 66 | 116.9 | 0.0843 | -0.0236 |
| 3-dataset→CICIDS2018 | 0.500 | 0.4700 | 0.1870 | 0.6662 | 0.4700 | -0.2304 | 42 | 91.8 | 0.1538 | -0.6220 |
| 3-dataset→NSL-KDD | 0.500 | 0.3493 | 0.1099 | 0.4683 | 0.3493 | -0.1236 | 21 | 44.3 | 0.2474 | 0.1139 |
| 3-dataset→UNSW-NB15 | 0.010 | 0.1178 | 0.0617 | 0.0867 | 0.1178 | +0.1989 | 70 | 125.6 | 0.4304 | 0.0062 |


## Individual Experiment Details

### exp05_holdout_3src_to_ton

- **Source Datasets**: NSL-KDD + UNSW-NB15 + CICIDS2018
- **Target Held-Out**: TON-IoT
- **λ**: 0.25
- **Train samples**: 50225
- **Test samples**: 10000
### exp06_holdout_3src_to_cicids

- **Source Datasets**: NSL-KDD + UNSW-NB15 + TON-IoT
- **Target Held-Out**: CICIDS2018
- **λ**: 0.5
- **Train samples**: 62608
- **Test samples**: 10000
### exp07_holdout_3src_to_nsl

- **Source Datasets**: UNSW-NB15 + CICIDS2018 + TON-IoT
- **Target Held-Out**: NSL-KDD
- **λ**: 0.5
- **Train samples**: 59499
- **Test samples**: 10000
### exp08_holdout_3src_to_unsw

- **Source Datasets**: NSL-KDD + CICIDS2018 + TON-IoT
- **Target Held-Out**: UNSW-NB15
- **λ**: 0.01
- **Train samples**: 51003
- **Test samples**: 10000
