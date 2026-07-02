# Feature Ablation Results — Phase 31

**Date:** 2026-06-24
**Experiment:** Progressive removal of top-K dataset-identifying features

## Protocol

1. Rank all 17 features by Gini importance from dataset-ID Random Forest classifier
2. Remove top-K features (K = 0, 1, 3, 5, 10)
3. Re-train dataset-ID classifier on remaining features
4. Evaluate cross-dataset Macro F1 (NSL-KDD + UNSW-NB15 → CICIDS-2018) with LogisticRegression

## Results

| Removed | Remaining | Dataset-ID Acc | Cross-Dataset Macro F1 | Cross-Dataset Binary F1 |
|---------|-----------|---------------|----------------------|------------------------|
| 0 | 17 | 1.0000 | 0.1935 | 0.0016 |
| 1 (flag) | 16 | 1.0000 | 0.0065 | 0.0187 |
| 3 (flag, src_dst_bytes_ratio, dst_src_bytes_ratio) | 14 | 1.0000 | 0.0072 | 0.0188 |
| 5 (+connection_state, protocol_service_flag) | 12 | 1.0000 | 0.0301 | 0.0374 |
| 10 (+count_x_srv_count, duration, diff_srv_rate_x_flag, src_bytes, dst_bytes) | 7 | 0.9999 | 0.1044 | 0.0000 |
| 15 (all except has_rst, traffic_direction) | 2 | 0.5761 | — | — |

## Key Observations

### Dataset Separability Persists Surprisingly

- **Removing top-10 features** (7 remaining, only 2.9% of total importance) still yields **99.99% accuracy**
- Only with **15 of 17 features removed** does accuracy drop materially (57.6%), still well above chance (33.3%)
- This confirms that dataset identity is redundantly encoded across the feature vector — no single feature or small subset is uniquely responsible

### Feature Removal Hurts Cross-Dataset Generalization

- Removing just `flag` (the single most important feature) collapses cross-dataset Macro F1 from 0.1935 to **0.0065**
- Removing top-5 features partially recovers to 0.0301 — but this is still **84% below baseline**
- Removing top-10 features improves to 0.1044 (still 46% below baseline)
- **Hypothesis:** `flag` and the byte-ratio features contain legitimate attack-signal information that is dataset-specific. Removing them eliminates both the fingerprint _and_ useful signal.

### Binary F1 Collapses Differently

- Baseline binary F1 (0.0016) is already near zero — the standard LR cannot distinguish attack vs. normal in CICIDS when trained on NSL+UNSW
- Removing top-5 features improves binary F1 to 0.0374 (still negligible)
- Removing top-10 features drops it back to 0.0000

## Plot

See `plots/dataset_id_ablation.png`

## Conclusion

Feature ablation is **not an effective strategy** for fingerprint elimination. Dataset identity is decodable from any sufficiently large subset of features, and ablating features simultaneously removes both dataset fingerprint AND attack-relevant signal, degrading cross-dataset performance.
