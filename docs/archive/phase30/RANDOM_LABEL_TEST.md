# Phase 30 — Random Label Sanity Check
**Device**: mps

## Protocol
1. Train DANNHelixModel with REAL labels
2. Train identical model with PERMUTED labels (random shuffle)
3. If model achieves high performance with random labels: LEAKAGE EXISTS

## Expected baseline (random labels)
- Macro F1 ≈ 1/N_classes = 0.1429 (chance level)
- ROC-AUC ≈ 0.5 (chance level)
- Binary F1 ≈ Normal prevalence rate

## Results

| Seed | Real MF1 | Real AUC | Random MF1 | Random AUC | Leakage? |
|-----:|--------:|--------:|----------:|----------:|--------:|
| 42 | 0.5687 | 0.0000 | 0.1142 | 0.0000 | ✅ NO |
| 1337 | 0.5699 | 0.0000 | 0.1142 | 0.0000 | ✅ NO |
| 2026 | 0.5721 | 0.0000 | 0.1142 | 0.0000 | ✅ NO |

## Detailed Random Label Metrics

### Seed 42
- Accuracy: 0.6656
- Macro F1: 0.1142
- Binary F1: 0.0000
- ROC-AUC: 0.0000
### Seed 1337
- Accuracy: 0.6656
- Macro F1: 0.1142
- Binary F1: 0.0001
- ROC-AUC: 0.0000
### Seed 2026
- Accuracy: 0.6656
- Macro F1: 0.1142
- Binary F1: 0.0000
- ROC-AUC: 0.0000

## Verdict

✅ **PASS**: Random label Macro F1 (0.1142) is near chance level (~0.143). No label leakage detected.
