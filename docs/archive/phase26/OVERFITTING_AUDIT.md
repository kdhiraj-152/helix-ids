# Overfitting Audit

For each experiment, compute `generalization_gap = train_accuracy - test_accuracy`.

A large positive gap indicates overfitting to source-domain training distribution.

## Per-Experiment Results

| Experiment | Train Accuracy | Test Accuracy | Generalization Gap | Epochs | Macro F1 |
|-----------|---------------:|--------------:|-------------------:|-------:|---------:|
| exp01_nsl_to_unsw | 0.0000 | 0.1218 | -0.1218 | 20 | 0.1068 |
| exp02_unsw_to_cicids | 0.6523 | 0.0306 | +0.6217 | 31 | 0.0196 |
| exp03_cicids_to_ton | 0.0000 | 0.1172 | -0.1172 | 20 | 0.0633 |
| exp04_ton_to_nsl | 0.0000 | 0.0076 | -0.0076 | 20 | 0.0067 |
| transfer_3src_to_cicids | 0.2423 | 0.0000 | +0.2423 | 28 | 0.0000 |
| transfer_3src_to_nsl | 0.1521 | 0.0013 | +0.1508 | 36 | 0.0004 |
| transfer_3src_to_ton | 0.1779 | 0.0095 | +0.1684 | 51 | 0.0119 |
| transfer_3src_to_unsw | 0.0078 | 0.0069 | +0.0009 | 47 | 0.0020 |

**Average generalization gap**: +0.1172

**Experiments with gap > 30%** (overfitting threshold): 1
  - exp02_unsw_to_cicids

## Interpretation

**MODERATE OVERFITTING**: Average gap is elevated but below the 30% threshold.

The model shows some degree of source-specific learning but the issue is not severe.
