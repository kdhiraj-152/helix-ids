# Holdout Generalization Report

Each holdout experiment trains on ALL OTHER datasets and evaluates on the held-out dataset. The 4 holdouts correspond to Experiments 5-8 from the task brief.

## Held-Out Dataset: CICIDS2018

- **Source datasets**: NSL-KDD + UNSW-NB15 + TON-IoT
- **Experiment key**: `transfer_3src_to_cicids`

| Metric | Value |
|--------|------:|
| Accuracy | 0.0001 |
| Macro F1 | 0.0002 |
| Weighted F1 | 0.0002 |
| Precision | 0.0595 |
| Recall | 0.0001 |
| Train Samples | 107,993 |
| Test Samples | 10,000 |

### Per-Class Recall

| Class | Recall |
|-------|------:|
| Normal | 0.0000 |
| DoS | 0.0008 |
| Probe | 0.0000 |
| R2L | 0.0000 |
| U2R | 0.0000 |
| Generic | 0.0000 |
| Backdoor | 0.0000 |

### Confusion Matrix

```
[[   0    1   34   30 8212    0    1]
 [   0    1    0    0 1188    0    0]
 [   0    0    0    0    0    0    0]
 [   0    0    0    0  252    0    0]
 [   0    0    0    0    0    0    0]
 [   0    0    0    0  281    0    0]
 [   0    0    0    0    0    0    0]]
```

## Held-Out Dataset: NSL-KDD

- **Source datasets**: UNSW-NB15 + CICIDS2018 + TON-IoT
- **Experiment key**: `transfer_3src_to_nsl`

| Metric | Value |
|--------|------:|
| Accuracy | 0.0020 |
| Macro F1 | 0.0020 |
| Weighted F1 | 0.0008 |
| Precision | 0.0206 |
| Recall | 0.0020 |
| Train Samples | 107,994 |
| Test Samples | 10,000 |

### Per-Class Recall

| Class | Recall |
|-------|------:|
| Normal | 0.0000 |
| DoS | 0.0000 |
| Probe | 0.0043 |
| R2L | 0.0000 |
| U2R | 0.9412 |
| Generic | 0.0000 |

### Confusion Matrix

```
[[   0   11    0    0 4963  220]
 [   0    0   14    0 3543   40]
 [   0    0    4    0  859   64]
 [   0    0    0    0  258    7]
 [   0    0    0    0   16    1]
 [   0    0    0    0    0    0]]
```

## Held-Out Dataset: TON-IoT

- **Source datasets**: NSL-KDD + UNSW-NB15 + CICIDS2018
- **Experiment key**: `transfer_3src_to_ton`

| Metric | Value |
|--------|------:|
| Accuracy | 0.0290 |
| Macro F1 | 0.0239 |
| Weighted F1 | 0.0362 |
| Precision | 0.3172 |
| Recall | 0.0290 |
| Train Samples | 107,995 |
| Test Samples | 10,000 |

### Per-Class Recall

| Class | Recall |
|-------|------:|
| Normal | 0.0000 |
| DoS | 0.1102 |
| Probe | 0.0000 |
| R2L | 0.0154 |
| U2R | 0.0000 |
| Generic | 0.0000 |
| Backdoor | 0.0117 |

### Confusion Matrix

```
[[   0  651  230    0 1291    7   89]
 [   0  225    1    0 1374    7  434]
 [   0  634    0    0  232    0  168]
 [   0  100    1   44 2523    6  181]
 [   0    0    0    0    0    0    0]
 [   0    0    0    0    0    0    0]
 [   0   65    5    0 1696   15   21]]
```

## Held-Out Dataset: UNSW-NB15

- **Source datasets**: NSL-KDD + CICIDS2018 + TON-IoT
- **Experiment key**: `transfer_3src_to_unsw`

| Metric | Value |
|--------|------:|
| Accuracy | 0.0064 |
| Macro F1 | 0.0018 |
| Weighted F1 | 0.0001 |
| Precision | 0.0000 |
| Recall | 0.0064 |
| Train Samples | 107,995 |
| Test Samples | 10,000 |

### Per-Class Recall

| Class | Recall |
|-------|------:|
| Normal | 0.0000 |
| DoS | 0.0000 |
| Probe | 0.0000 |
| R2L | 0.0000 |
| U2R | 1.0000 |
| Generic | 0.0000 |
| Backdoor | 0.0000 |

### Confusion Matrix

```
[[   0    0    0    0 3147    0    0]
 [   0    0    0    0  726    0    0]
 [   0    0    0    0 1775    0    0]
 [   0    0    0    0 1891    0    0]
 [   0    0    0    0   64    0    0]
 [   0    0    0    0 2286    0    0]
 [   0    0    0    0  111    0    0]]
```
