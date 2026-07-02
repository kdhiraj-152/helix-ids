# Cross-Dataset Transfer Ratio Analysis

## Definition

Transfer Ratio = Cross-Dataset Macro F1 / In-Dataset (Oracle) Macro F1

For each source-target pair, this measures how much of the within-dataset performance is preserved when transferring. Ratio < 1 means transfer degrades performance; ratio = 0 means no transfer at all.

## Transfer Ratio Matrix

Rows = training (source), Columns = test (target). Diagonal = 1.0 (always).

| Train ↓ / Test → | NSL-KDD | UNSW-NB15 | CICIDS2018 | TON-IoT |
|---|---|---|---|---|
| **NSL-KDD** 1.000  0.0168  0.0000  0.0000 |
| **UNSW-NB15** 0.0000  1.000  0.0381  0.0000 |
| **CICIDS2018** 0.0000  0.0000  1.000  0.0213 |
| **TON-IoT** 0.0000  0.0000  0.0000  1.000 |

## Summary Statistics

- **Average Transfer Ratio**: 0.0064 (0.6%)
- **Maximum Transfer Ratio**: 0.0381 (3.8%)
- **Minimum Transfer Ratio**: 0.0000 (0.0%)
- **Median Transfer Ratio**: 0.0000 (0.0%)

## Per-Pair Detail

| Source | Target | Oracle MF1 | Cross MF1 | Transfer Ratio |
|--------|--------|----------:|----------:|--------------:|
| NSL-KDD | UNSW-NB15 | 0.8635 | 0.0145 | 0.0168 |
| NSL-KDD | CICIDS2018 | 0.8635 | 0.0000 | 0.0000 |
| NSL-KDD | TON-IoT | 0.8635 | 0.0000 | 0.0000 |
| UNSW-NB15 | NSL-KDD | 0.4952 | 0.0000 | 0.0000 |
| UNSW-NB15 | CICIDS2018 | 0.4952 | 0.0189 | 0.0381 |
| UNSW-NB15 | TON-IoT | 0.4952 | 0.0000 | 0.0000 |
| CICIDS2018 | NSL-KDD | 0.8623 | 0.0000 | 0.0000 |
| CICIDS2018 | UNSW-NB15 | 0.8623 | 0.0000 | 0.0000 |
| CICIDS2018 | TON-IoT | 0.8623 | 0.0184 | 0.0213 |
| TON-IoT | NSL-KDD | 0.0000 | 0.0272 | 0.0000 |
| TON-IoT | UNSW-NB15 | 0.0000 | 0.0000 | 0.0000 |
| TON-IoT | CICIDS2018 | 0.0000 | 0.0000 | 0.0000 |

## Certification Threshold

**Average Transfer Ratio**: 0.0064 (0.6%)

**Threshold for termination**: < 0.25 (25%)

**Verdict**: **BELOW THRESHOLD** — Current public benchmarks unsuitable for cross-dataset transfer.
