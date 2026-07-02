# Normalization Study — Phase 31

**Date:** 2026-06-24
**Experiment:** Evaluate whether normalization methods can remove dataset fingerprints from feature distributions

## Methods Tested

| Method | Description |
|--------|-------------|
| None | Raw harmonized features (baseline) |
| Z-Score | Global StandardScaler across all datasets |
| Robust | RobustScaler (median/IQR-based) |
| Quantile | QuantileTransformer → uniform distribution |
| Rank | Per-feature percentile ranks |
| Per-Dataset Z-Score | Independent StandardScaler per dataset |

## Results

### Dataset-ID Accuracy

All normalization methods achieve **100% dataset-ID accuracy**. No normalization technique reduces dataset separability.

### Cross-Dataset Macro F1 (NSL+UNSW → CICIDS)

| Method | Dataset-ID Acc | Macro F1 | Binary F1 |
|--------|---------------|----------|-----------|
| none | 1.0000 | 0.1914 | 0.0012 |
| zscore | 1.0000 | 0.1911 | 0.0012 |
| robust | 1.0000 | 0.1914 | 0.0012 |
| quantile | 1.0000 | 0.0418 | 0.0369 |
| rank | 1.0000 | 0.0853 | 0.0573 |
| per_dataset_zscore | 1.0000 | 0.0966 | 0.0666 |

### Analysis

| Method | Impact on Dataset-ID | Impact on Cross-Dataset MF1 |
|--------|---------------------|---------------------------|
| None | Baseline | Baseline |
| Z-Score | No change | Negligible (−0.2%) |
| Robust | No change | Negligible (±0.0%) |
| Quantile | No change | **−78.2%** (destroys signal) |
| Rank | No change | **−55.4%** (destroys signal) |
| Per-Dataset Z-Score | No change | **−49.5%** (destroys signal) |

### Key Findings

1. **No normalization method reduces dataset-ID accuracy below 100%.**
   Dataset fingerprints persist through linear transforms, quantile mapping, rank transforms, and per-dataset standardization. This confirms that fingerprinting is encoded in the **relative multi-dimensional structure** of the features — not in scale or marginal distributions.

2. **Destructive methods hurt cross-dataset performance.**
   Quantile normalization (−78%), rank normalization (−55%), and per-dataset z-score (−50%) each degrade cross-dataset Macro F1. This is consistent with the feature-level interventions: methods that "break" the dataset-specific distributional signatures also break attack-relevant signal.

3. **Linear global scaling (z-score, robust) is neutral.**
   Global z-score and robust scaling preserve cross-dataset Macro F1 (≤0.2% change). They are recommended as safe default preprocessing but do not address fingerprinting.

## Conclusion

Standard normalization techniques **cannot eliminate dataset fingerprints**. The fingerprint is not a scale artifact — it is embedded in the joint feature geometry. Specialized fingerprint-aware methods (domain-adversarial training, feature decorrelation) operating at the representation level rather than the input-feature level are required.
