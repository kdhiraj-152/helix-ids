# Dataset Compatibility Matrix

**Generated**: Phase 34 — Dataset Compatibility Ceiling

## 1. Oracle (Within-Dataset) Macro F1

Maximum achievable Macro F1 when training and testing on the same dataset.

| Dataset | Accuracy | Macro F1 | Precision | Recall | Samples |
|---------|--------:|---------:|----------:|-------:|-------:|
| NSL-KDD | 0.9794 | 0.8635 | 0.9794 | 0.9794 | 10,000 |
| UNSW-NB15 | 0.7944 | 0.4952 | 0.7620 | 0.7944 | 10,000 |
| CICIDS2018 | 0.9649 | 0.8623 | 0.9656 | 0.9649 | 10,000 |

## 2. Cross-Dataset Transfer Macro F1 (from Phase 26A)

Rows = training dataset, Columns = test dataset.

| Train ↓ / Test → | NSL-KDD | UNSW-NB15 | CICIDS2018 | TON-IoT |
|---|---|---|---|---|
| **NSL-KDD** **0.8635**  0.0145  —  — |
| **UNSW-NB15** —  **0.4952**  0.0189  — |
| **CICIDS2018** —  —  **0.8623**  0.0184 |
| **TON-IoT** 0.0272  —  —  **0.0000** |

## 3. Shared Class Overlap

| Class | NSL-KDD | UNSW-NB15 | CICIDS2018 | TON-IoT |
|---|---|---|---|---|
| **Normal** ✓  ✓  ✓  ✓ |
| **DoS** ✓  ✓  ✓  ✓ |
| **Probe** ✓  ✓  ✓  ✓ |
| **R2L** ✓  ✓  ✓  ✓ |
| **U2R** ✓  ✓  ✗  ✗ |
| **Generic** ✗  ✓  ✓  ✗ |
| **Backdoor** ✗  ✓  ✗  ✓ |

### Shared Across All Datasets

['Normal', 'DoS', 'Probe', 'R2L']

### Pairwise Jaccard Overlap by Class Index

| Source → Target | Jaccard |
|----------------|--------:|
| CICIDS2018 → CICIDS2018 | 1.0 |
| CICIDS2018 → NSL-KDD | 0.6667 |
| CICIDS2018 → TON-IoT | 0.6667 |
| CICIDS2018 → UNSW-NB15 | 0.7143 |
| NSL-KDD → CICIDS2018 | 0.6667 |
| NSL-KDD → NSL-KDD | 1.0 |
| NSL-KDD → TON-IoT | 0.6667 |
| NSL-KDD → UNSW-NB15 | 0.7143 |
| TON-IoT → CICIDS2018 | 0.6667 |
| TON-IoT → NSL-KDD | 0.6667 |
| TON-IoT → TON-IoT | 1.0 |
| TON-IoT → UNSW-NB15 | 0.7143 |
| UNSW-NB15 → CICIDS2018 | 0.7143 |
| UNSW-NB15 → NSL-KDD | 0.7143 |
| UNSW-NB15 → TON-IoT | 0.7143 |
| UNSW-NB15 → UNSW-NB15 | 1.0 |
