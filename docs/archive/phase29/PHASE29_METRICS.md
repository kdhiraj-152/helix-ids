# Phase 29 — Production Deployment Metrics

**Generated**: 2026-06-23 23:31:45 IST
**Device**: mps
**Seeds**: [42, 1337, 2026]
**Datasets**: nsl_kdd, unsw_nb15, cicids2018
**Architecture**: DANNHelixModel (256-128-64 backbone)
**Lambda (domain weight)**: 0.5
**Total parameters**: DANNHelixModel

---

## Aggregate Metrics (across 3 seeds)

| Metric | Mean | Std |
|--------|-----:|----:|
| Macro F1 | 0.5757 | 0.0033 |
| Weighted F1 | 0.8844 | — |
| Accuracy | 0.8811 | — |
| Precision (weighted) | 0.8942 | — |
| Recall (weighted) | 0.8811 | — |
| Binary F1 (Normal vs Attack) | 0.8891 | — |
| ROC-AUC (OvR) | 0.9750 | — |
| Expected Calibration Error (ECE) | 0.0059 | — |

## Per-Seed Metrics

| Seed | Macro F1 | Weighted F1 | Acc | Binary F1 | ECE | ROC-AUC |
|-----:|--------:|----------:|----:|---------:|----:|-------:|
| 42 | 0.5758 | 0.8884 | 0.8861 | 0.8949 | 0.0085 | 0.9755 |
| 1337 | 0.5724 | 0.8832 | 0.8791 | 0.8859 | 0.0046 | 0.9737 |
| 2026 | 0.5790 | 0.8816 | 0.8781 | 0.8866 | 0.0047 | 0.9757 |

## Per-Class Metrics (mean across seeds)

| Class | F1 (μ±σ) | Precision | Recall | Support |
|------|---------:|----------:|------:|-------:|
| Normal | 0.9409±0.0033 | 0.9643 | 0.9187 | 109681 |
| DoS | 0.8035±0.0099 | 0.7741 | 0.8357 | 24947 |
| Probe | 0.7142±0.0009 | 0.6923 | 0.7374 | 8951 |
| R2L | 0.6680±0.0057 | 0.5539 | 0.8430 | 9781 |
| U2R | 0.0164±0.0285 | 0.3333 | 0.0084 | 277 |
| Generic | 0.8871±0.0013 | 0.9806 | 0.8099 | 10761 |
| Backdoor | 0.0000±0.0000 | 0.0000 | 0.0000 | 375 |

### Classification Report (seed=42)

```
              precision    recall  f1-score   support

      Normal       0.96      0.93      0.94    109681
         DoS       0.78      0.85      0.81     24947
       Probe       0.70      0.73      0.71      8951
         R2L       0.58      0.81      0.67      9781
         U2R       0.00      0.00      0.00       277
     Generic       0.98      0.81      0.89     10761
    Backdoor       0.00      0.00      0.00       375

    accuracy                           0.89    164773
   macro avg       0.57      0.59      0.58    164773
weighted avg       0.90      0.89      0.89    164773

```

### Classification Report (seed=1337)

```
              precision    recall  f1-score   support

      Normal       0.97      0.91      0.94    109681
         DoS       0.77      0.85      0.81     24947
       Probe       0.69      0.74      0.71      8951
         R2L       0.54      0.85      0.66      9781
         U2R       0.00      0.00      0.00       277
     Generic       0.98      0.81      0.89     10761
    Backdoor       0.00      0.00      0.00       375

    accuracy                           0.88    164773
   macro avg       0.56      0.59      0.57    164773
weighted avg       0.89      0.88      0.88    164773

```

### Classification Report (seed=2026)

```
              precision    recall  f1-score   support

      Normal       0.96      0.92      0.94    109681
         DoS       0.78      0.81      0.79     24947
       Probe       0.69      0.74      0.71      8951
         R2L       0.55      0.87      0.67      9781
         U2R       1.00      0.03      0.05       277
     Generic       0.98      0.81      0.89     10761
    Backdoor       0.00      0.00      0.00       375

    accuracy                           0.88    164773
   macro avg       0.71      0.60      0.58    164773
weighted avg       0.89      0.88      0.88    164773

```
