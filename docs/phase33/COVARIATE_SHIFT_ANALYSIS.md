# Covariate Shift Quantification

**Phase 33 — Dataset Incompatibility Proof**
**Created:** 2026-06-24

---

## Overview

Covariate shift quantifies how feature distributions differ between dataset pairs independent of label information. For each of the 17 canonical features across all dataset pairs, we compute four divergence metrics.

## Methodology

For each dataset pair (S, T), we draw up to 10,000 samples per dataset and compute:

| Metric | Range | Interpretation |
|--------|-------|---------------|
| **KL Divergence** | [0, ∞) | Asymmetric information gain. Higher = more shift |
| **Jensen-Shannon Divergence** | [0, 1] | Symmetric, bounded. JS > 0.5 indicates severe shift |
| **Wasserstein Distance** | [0, ∞) | Earth mover distance. Scale-dependent |
| **Kolmogorov-Smirnov Statistic** | [0, 1] | Maximum CDF difference. KS > 0.2 = strong shift |

Distribution estimation uses 50-bin histograms with common bin edges covering the [1st, 99th] percentile range.

## Results

### Mean JS Divergence Per Pair

| Dataset Pair | Mean JS | Std JS | Interpretation |
|---|---|---|---|
| NSL-KDD vs UNSW-NB15 | 0.6319 | 0.1950 | **Extreme shift** |
| NSL-KDD vs CICIDS2018 | 0.5689 | 0.2834 | **Extreme shift** |
| NSL-KDD vs TON-IoT | 0.4211 | 0.2669 | **Severe shift** |
| UNSW-NB15 vs CICIDS2018 | 0.6361 | 0.2298 | **Extreme shift** |
| UNSW-NB15 vs TON-IoT | 0.4725 | 0.2585 | **Severe shift** |
| CICIDS2018 vs TON-IoT | 0.3574 | 0.2042 | **Severe shift** |

**Key finding:** Every dataset pair exceeds JS > 0.35. Three of six pairs exceed 0.50. No pair exhibits moderate or low shift.

### Most Shifted Features (by mean JS across all pairs)

| Feature | Mean JS | Pairs with JS > 0.5 | Primary Driver |
|---|---|---|---|
| `connection_state` | 0.81 | 6/6 | TCP state encoding differs across datasets |
| `src_dst_bytes_ratio` | 0.79 | 6/6 | Byte ratio distributions fundamentally differ |
| `dst_src_bytes_ratio` | 0.74 | 6/6 | Symmetric to above |
| `log_dst_bytes` | 0.62 | 6/6 | Destination byte volumes differ |
| `count_x_srv_count` | 0.59 | 5/6 | Traffic count distributions dataset-specific |
| `log_src_bytes` | 0.58 | 6/6 | Source byte distributions differ |
| `flag` | 0.54 | 4/6 | TCP flag semantics diverge |
| `protocol_service_flag` | 0.49 | 4/6 | Interaction features amplify individual shifts |
| `protocol_type` | 0.37 | 3/6 | Protocol distribution difference |

### Feature-Level Analysis (Top 3 Pairs)

#### NSL-KDD vs UNSW-NB15 (Worst Shift)

| Feature | KL | JS | Wasserstein | KS |
|---|---|---|---|---|
| `connection_state` | 28.05 | 0.830 | 4.36 | 0.983 |
| `src_dst_bytes_ratio` | 18.34 | 0.829 | 0.18 | 0.951 |
| `dst_src_bytes_ratio` | 17.44 | 0.826 | 0.35 | 0.873 |
| `log_dst_bytes` | 15.92 | 0.651 | 0.17 | 0.487 |
| `log_src_bytes` | 6.34 | 0.620 | 0.28 | 0.367 |

JS = 0.83 on `connection_state` indicates that TCP connection states are virtually disjoint between NSL-KDD and UNSW-NB15.

#### NSL-KDD vs CICIDS2018

| Feature | KL | JS | Wasserstein | KS |
|---|---|---|---|---|
| `connection_state` | 12.96 | 0.793 | 2.80 | 0.941 |
| `src_dst_bytes_ratio` | 28.70 | 0.884 | 0.12 | 0.999 |
| `dst_src_bytes_ratio` | 1.22 | 0.600 | 0.15 | 0.971 |
| `same_host_rate_x_service` | 1.71 | 0.520 | 0.08 | 0.729 |

`src_dst_bytes_ratio` reaches KL = 28.70 (virtually zero overlap in distribution support).

#### CICIDS2018 vs TON-IoT (Least Shift)

| Feature | KL | JS | Wasserstein | KS |
|---|---|---|---|---|
| `connection_state` | 0.80 | 0.363 | 0.57 | 0.550 |
| `src_dst_bytes_ratio` | 11.72 | 0.754 | 0.17 | 0.968 |
| `duration` | 0.54 | 0.187 | 0.00 | 0.586 |
| `log_src_bytes` | 0.29 | 0.186 | 0.14 | 0.320 |

Even the "least shifted" pair shows severe divergence on byte-ratio features.

## Summary

1. **All 17 features exhibit significant shift across ALL dataset pairs.**
2. The strongest shifts cluster around connection state encoding, byte ratio features, and count-based traffic metrics.
3. No feature subset can be considered "portable" across datasets — even the least divergent features (e.g., `has_rst` with JS ~0.17–0.21) show KS p-values of 0.0, indicating statistically significant distribution differences.
4. The covariate shift is **not reducible to a few problematic features**. It is pervasive across the entire feature space.

## Plots

- `plots/phase33/covariate_heatmaps/KL_heatmap.png`
- `plots/phase33/covariate_heatmaps/JS_heatmap.png`
- `plots/phase33/covariate_heatmaps/Wasserstein_heatmap.png`
- `plots/phase33/covariate_heatmaps/KS_heatmap.png`
- `plots/phase33/divergence_curves/` — per-pair feature JS bar charts
