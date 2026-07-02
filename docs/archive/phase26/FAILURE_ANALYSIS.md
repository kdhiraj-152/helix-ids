# Failure Analysis

## Best Transfer Pair
- **Source → Target**: TON-IoT → NSL-KDD
- **Accuracy**: 0.0843
- **Macro F1**: 0.0272
- **Precision**: 0.0087
- **Recall**: 0.0843

## Worst Transfer Pair
- **Source → Target**: NSL-KDD → UNSW-NB15
- **Accuracy**: 0.0460
- **Macro F1**: 0.0145
- **Precision**: 0.0035
- **Recall**: 0.0460

## Largest Precision Drop
- **Pair**: NSL-KDD → UNSW-NB15
- **Precision**: 0.0035
- **Context**: 0.0460 accuracy, 0.0145 macro F1

## Largest Recall Drop
- **Pair**: UNSW-NB15 → CICIDS2018
- **Recall**: 0.0220
- **Context**: 0.0220 accuracy, 0.0189 macro F1

## Most Confusing Attack Family
- **Pair**: UNSW-NB15 → CICIDS2018
- **Off-diagonal ratio**: 0.9780

## Least Confusing Attack Family
- **Pair**: TON-IoT → NSL-KDD
- **Off-diagonal ratio**: 0.9157
