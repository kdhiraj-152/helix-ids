# Phase 26A — Cross-Dataset Generalization Certification

## Summary

- **Total experiments**: 8
- **Successful experiments**: 8
- **Failed experiments**: 0
- **Best transfer score (Macro F1)**: 0.0272
  - Pair: TON-IoT → NSL-KDD
- **Worst transfer score (Macro F1)**: 0.0145
  - Pair: NSL-KDD → UNSW-NB15
- **Average transfer Macro F1**: 0.0197
- **Average holdout Macro F1**: 0.0070
- **All experiments executed successfully**: True

## Results Summary

| Experiment | Accuracy | Macro F1 | Precision | Recall |
|-----------|---------:|---------:|----------:|------:|
| exp01_nsl_to_unsw | 0.0460 | 0.0145 | 0.0035 | 0.0460 |
| exp02_unsw_to_cicids | 0.0220 | 0.0189 | 0.0537 | 0.0220 |
| exp03_cicids_to_ton | 0.0637 | 0.0184 | 0.0061 | 0.0637 |
| exp04_ton_to_nsl | 0.0843 | 0.0272 | 0.0087 | 0.0843 |
| transfer_3src_to_cicids | 0.0001 | 0.0002 | 0.0595 | 0.0001 |
| transfer_3src_to_nsl | 0.0020 | 0.0020 | 0.0206 | 0.0020 |
| transfer_3src_to_ton | 0.0290 | 0.0239 | 0.3172 | 0.0290 |
| transfer_3src_to_unsw | 0.0064 | 0.0018 | 0.0000 | 0.0064 |

## Holdout Performance Ranking

| Rank | Held-Out Dataset | Macro F1 | Accuracy |
|----:|-----------------|---------:|---------:|
| 1 | TON-IoT | 0.0239 | 0.0290 |
| 2 | NSL-KDD | 0.0020 | 0.0020 |
| 3 | UNSW-NB15 | 0.0018 | 0.0064 |
| 4 | CICIDS2018 | 0.0002 | 0.0001 |

## Recommendation

**Additional Dataset Acquisition** — The model shows very poor cross-dataset transfer (avg F1 0.020). Before attempting domain adaptation, acquire more diverse datasets covering the target distribution. Current feature space may lack the representational capacity for generalization.

## Schema Contract Audit

- Input dimension: 17 (verified)
- Binary output: 2 (verified)
- Family output: 7 (verified)
- All experiments used 17-feature harmonized data (verified)
- No dataset leakage detected (verified)
