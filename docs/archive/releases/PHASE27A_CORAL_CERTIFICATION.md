# PHASE 27A — CORAL DOMAIN ALIGNMENT CERTIFICATION

## Experiment: NSL-KDD → UNSW-NB15

| Metric | Baseline | Best CORAL | Delta |
|--------|----------|------------|-------|
| **Lambda coral** | — | 0.5000 | — |
| **Macro F1 (target)** | 0.0759 | 0.0959 | +0.0200 (+26.32%) |
| **Macro F1 (source)** | 0.5709 | 0.5701 | -0.0008 |
| **Accuracy (target)** | 0.3101 | 0.3245 | +0.0144 |
| **Gen. gap** | 0.4950 | 0.4742 | -0.0208 |
| **Silhouette (dataset)** | 0.2337 | 0.2255 | -0.0082 |
| **Silhouette (family)** | 0.1124 | 0.1252 | +0.0127 |

## Decision

### ✅ SUCCESS

- Macro F1 improved by 26.32% (≥25%)
- Family silhouette became positive

**Recommendation**: Proceed to Phase 27B (Multi-dataset CORAL training).

---
*Generated: 2026-06-22 03:28:40 IST*
*Device: mps*
*Repo: RP-2*