# Phase 30 — Strict Domain Generalization Audit

**Device**: mps

**Protocol**: Leave-one-dataset-out cross-validation

**Epochs**: 100, **Patience**: 20


## Results


### Test on NSL_KDD

  Train: unsw_nb15, cicids2018

  Macro F1: 0.1916 ± 0.0944

  Binary F1: 0.2323 ± 0.3931

  ROC-AUC: 0.0000 ± 0.0000

  Seeds: [42, 1337, 2026]


### Test on UNSW_NB15

  Train: nsl_kdd, cicids2018

  Macro F1: 0.0627 ± 0.0238

  Binary F1: 0.7572 ± 0.0478

  ROC-AUC: 0.0000 ± 0.0000

  Seeds: [42, 1337, 2026]


### Test on CICIDS2018

  Train: nsl_kdd, unsw_nb15

  Macro F1: 0.1001 ± 0.0558

  Binary F1: 0.2624 ± 0.0107

  ROC-AUC: 0.0000 ± 0.0000

  Seeds: [42, 1337, 2026]


## Comparison


| Test Set | Macro F1 (μ±σ) | Binary F1 (μ±σ) | ROC-AUC (μ±σ) | Phase 29 (in-dist) |

|----------|:------------:|:--------------:|:-------------:|:-----------------:|
| NSL_KDD | 0.1916±0.0944 | 0.2323±0.3931 | 0.0000±0.0000 | 0.5757±0.0034 |
| UNSW_NB15 | 0.0627±0.0238 | 0.7572±0.0478 | 0.0000±0.0000 | 0.5757±0.0034 |
| CICIDS2018 | 0.1001±0.0558 | 0.2624±0.0107 | 0.0000±0.0000 | 0.5757±0.0034 |

---
*Phase 30 Audit — Domain Generalization*
