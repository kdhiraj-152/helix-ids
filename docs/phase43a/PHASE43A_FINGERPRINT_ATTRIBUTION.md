# Phase 43A — Dataset Fingerprint Attribution Report

**Date:** 2026-06-26
**Experiment:** Phase 43A, RP-2 HELIX-IDS
**Objective:** Identify which canonical features are responsible for the near-perfect dataset identification (DOS ≈ 0.0).

## Executive Summary

Logistic Regression and Random Forest classifiers were trained to distinguish dataset origin for all 6 pairwise combinations of the 4 IDS datasets (NSL-KDD, UNSW-NB15, CICIDS2018, TON-IoT). Feature importance was extracted and aggregated to produce a global fingerprint ranking.

## Dataset-ID Classification Accuracy

| Pair | LR Accuracy | RF Accuracy | Samples (A/B) |
|------|------------|------------|--------------|
| NSL-KDD vs UNSW-NB15 | 1.0000 | 1.0000 | 107077 / 149040 |
| NSL-KDD vs CICIDS2018 | 1.0000 | 1.0000 | 107077 / 200000 |
| NSL-KDD vs TON-IoT | 1.0000 | 1.0000 | 107077 / 190474 |
| UNSW-NB15 vs CICIDS2018 | 1.0000 | 1.0000 | 149040 / 200000 |
| UNSW-NB15 vs TON-IoT | 1.0000 | 1.0000 | 149040 / 190474 |
| CICIDS2018 vs TON-IoT | 0.9999 | 1.0000 | 200000 / 190474 |
| **Mean** | **1.0000** | **1.0000** | — |

## Pairwise Feature Importance Tables

Each table shows the top-10 features ranked by contribution to dataset-origin classification for that pair.

### Pair: NSL-KDD vs UNSW-NB15

### Logistic Regression — Top 10 Features

| Rank | Feature | |Coefficient| |
| --- | --- | --- |
| 1 | flag | 8.677543 |
| 2 | protocol_type | 1.908187 |
| 3 | has_rst | 1.477606 |
| 4 | traffic_direction | 1.407891 |
| 5 | protocol_service_flag | 1.273150 |
| 6 | service_tier | 1.209309 |
| 7 | connection_state | 0.981348 |
| 8 | log_dst_bytes | 0.865950 |
| 9 | dst_bytes | 0.865950 |
| 10 | same_host_rate_x_service | 0.720691 |

### Random Forest — Top 10 Features

| Rank | Feature | Importance (Gini) |
| --- | --- | --- |
| 1 | flag | 0.347024 |
| 2 | connection_state | 0.223012 |
| 3 | src_dst_bytes_ratio | 0.150030 |
| 4 | dst_src_bytes_ratio | 0.095337 |
| 5 | count_x_srv_count | 0.058949 |
| 6 | duration | 0.037735 |
| 7 | service_tier | 0.029205 |
| 8 | diff_srv_rate_x_flag | 0.021517 |
| 9 | log_dst_bytes | 0.013838 |
| 10 | dst_bytes | 0.009206 |

---

### Pair: NSL-KDD vs CICIDS2018

### Logistic Regression — Top 10 Features

| Rank | Feature | |Coefficient| |
| --- | --- | --- |
| 1 | protocol_service_flag | 10.080936 |
| 2 | duration | 3.958620 |
| 3 | service_tier | 3.451406 |
| 4 | protocol_type | 2.854470 |
| 5 | connection_state | 2.834457 |
| 6 | flag | 2.709843 |
| 7 | log_src_bytes | 1.559031 |
| 8 | src_bytes | 1.559031 |
| 9 | traffic_direction | 1.034843 |
| 10 | diff_srv_rate_x_flag | 0.774175 |

### Random Forest — Top 10 Features

| Rank | Feature | Importance (Gini) |
| --- | --- | --- |
| 1 | dst_src_bytes_ratio | 0.219802 |
| 2 | duration | 0.182923 |
| 3 | count_x_srv_count | 0.137096 |
| 4 | connection_state | 0.133232 |
| 5 | diff_srv_rate_x_flag | 0.103196 |
| 6 | protocol_service_flag | 0.098269 |
| 7 | same_host_rate_x_service | 0.044144 |
| 8 | src_dst_bytes_ratio | 0.022054 |
| 9 | log_src_bytes | 0.018903 |
| 10 | dst_bytes | 0.015337 |

---

### Pair: NSL-KDD vs TON-IoT

### Logistic Regression — Top 10 Features

| Rank | Feature | |Coefficient| |
| --- | --- | --- |
| 1 | service_tier | 11.478098 |
| 2 | protocol_service_flag | 10.688156 |
| 3 | same_host_rate_x_service | 6.064887 |
| 4 | connection_state | 4.537920 |
| 5 | flag | 3.387806 |
| 6 | protocol_type | 1.672306 |
| 7 | log_dst_bytes | 1.446642 |
| 8 | diff_srv_rate_x_flag | 1.440317 |
| 9 | has_rst | 1.041598 |
| 10 | traffic_direction | 0.812472 |

### Random Forest — Top 10 Features

| Rank | Feature | Importance (Gini) |
| --- | --- | --- |
| 1 | duration | 0.254133 |
| 2 | diff_srv_rate_x_flag | 0.222392 |
| 3 | protocol_service_flag | 0.215729 |
| 4 | same_host_rate_x_service | 0.116358 |
| 5 | service_tier | 0.081533 |
| 6 | log_dst_bytes | 0.047619 |
| 7 | log_src_bytes | 0.037003 |
| 8 | flag | 0.006989 |
| 9 | dst_bytes | 0.006474 |
| 10 | src_bytes | 0.005664 |

---

### Pair: UNSW-NB15 vs CICIDS2018

### Logistic Regression — Top 10 Features

| Rank | Feature | |Coefficient| |
| --- | --- | --- |
| 1 | flag | 6.255874 |
| 2 | protocol_service_flag | 1.950996 |
| 3 | duration | 1.628461 |
| 4 | protocol_type | 1.422990 |
| 5 | log_src_bytes | 0.646041 |
| 6 | src_bytes | 0.646041 |
| 7 | connection_state | 0.476392 |
| 8 | traffic_direction | 0.293507 |
| 9 | log_dst_bytes | 0.231786 |
| 10 | dst_bytes | 0.231786 |

### Random Forest — Top 10 Features

| Rank | Feature | Importance (Gini) |
| --- | --- | --- |
| 1 | flag | 0.331678 |
| 2 | protocol_service_flag | 0.200796 |
| 3 | count_x_srv_count | 0.137298 |
| 4 | duration | 0.090357 |
| 5 | connection_state | 0.061030 |
| 6 | dst_src_bytes_ratio | 0.053166 |
| 7 | diff_srv_rate_x_flag | 0.041683 |
| 8 | same_host_rate_x_service | 0.027758 |
| 9 | src_bytes | 0.019804 |
| 10 | src_dst_bytes_ratio | 0.013575 |

---

### Pair: UNSW-NB15 vs TON-IoT

### Logistic Regression — Top 10 Features

| Rank | Feature | |Coefficient| |
| --- | --- | --- |
| 1 | protocol_service_flag | 6.546710 |
| 2 | flag | 6.300760 |
| 3 | service_tier | 2.558799 |
| 4 | protocol_type | 2.316535 |
| 5 | same_host_rate_x_service | 1.990451 |
| 6 | traffic_direction | 1.093875 |
| 7 | has_rst | 0.377585 |
| 8 | log_dst_bytes | 0.363016 |
| 9 | connection_state | 0.281813 |
| 10 | duration | 0.103465 |

### Random Forest — Top 10 Features

| Rank | Feature | Importance (Gini) |
| --- | --- | --- |
| 1 | flag | 0.380747 |
| 2 | connection_state | 0.183983 |
| 3 | protocol_service_flag | 0.171337 |
| 4 | same_host_rate_x_service | 0.084984 |
| 5 | duration | 0.059677 |
| 6 | log_src_bytes | 0.039109 |
| 7 | traffic_direction | 0.028978 |
| 8 | log_dst_bytes | 0.018718 |
| 9 | diff_srv_rate_x_flag | 0.017661 |
| 10 | src_bytes | 0.007295 |

---

### Pair: CICIDS2018 vs TON-IoT

### Logistic Regression — Top 10 Features

| Rank | Feature | |Coefficient| |
| --- | --- | --- |
| 1 | same_host_rate_x_service | 8.099563 |
| 2 | diff_srv_rate_x_flag | 6.806919 |
| 3 | flag | 6.630353 |
| 4 | service_tier | 6.255293 |
| 5 | log_src_bytes | 5.918000 |
| 6 | connection_state | 4.724458 |
| 7 | protocol_service_flag | 3.824363 |
| 8 | has_rst | 1.992519 |
| 9 | log_dst_bytes | 1.399568 |
| 10 | duration | 1.326100 |

### Random Forest — Top 10 Features

| Rank | Feature | Importance (Gini) |
| --- | --- | --- |
| 1 | duration | 0.228741 |
| 2 | flag | 0.209647 |
| 3 | connection_state | 0.176525 |
| 4 | service_tier | 0.119380 |
| 5 | same_host_rate_x_service | 0.069213 |
| 6 | diff_srv_rate_x_flag | 0.052037 |
| 7 | src_bytes | 0.045404 |
| 8 | dst_bytes | 0.042438 |
| 9 | count_x_srv_count | 0.019787 |
| 10 | protocol_service_flag | 0.011824 |

---

## Global Fingerprint Ranking

Features ranked by average rank position across all 6 dataset pairs. Lower average rank = more responsible for dataset identification.

### Global Ranking by Logistic Regression (avg rank)

| Global Rank | Feature | Avg Rank Position | Pairs Seen |
| --- | --- | --- | --- |
| 1 | flag | 3.00 | 6 |
| 2 | protocol_service_flag | 3.00 | 6 |
| 3 | service_tier | 5.17 | 6 |
| 4 | protocol_type | 5.67 | 6 |
| 5 | connection_state | 6.33 | 6 |
| 6 | same_host_rate_x_service | 7.00 | 6 |
| 7 | traffic_direction | 8.00 | 6 |
| 8 | duration | 8.17 | 6 |
| 9 | has_rst | 9.00 | 6 |
| 10 | log_dst_bytes | 9.17 | 6 |
| 11 | log_src_bytes | 9.33 | 6 |
| 12 | diff_srv_rate_x_flag | 9.67 | 6 |
| 13 | src_bytes | 11.67 | 6 |
| 14 | dst_bytes | 11.83 | 6 |
| 15 | count_x_srv_count | 14.33 | 6 |
| 16 | src_dst_bytes_ratio | 15.00 | 6 |
| 17 | dst_src_bytes_ratio | 16.67 | 6 |

### Global Ranking by Random Forest (avg rank)

| Global Rank | Feature | Avg Rank Position | Pairs Seen |
| --- | --- | --- | --- |
| 1 | duration | 3.17 | 6 |
| 2 | flag | 4.17 | 6 |
| 3 | connection_state | 4.67 | 6 |
| 4 | protocol_service_flag | 6.00 | 6 |
| 5 | diff_srv_rate_x_flag | 6.17 | 6 |
| 6 | same_host_rate_x_service | 6.50 | 6 |
| 7 | count_x_srv_count | 8.00 | 6 |
| 8 | service_tier | 9.50 | 6 |
| 9 | dst_src_bytes_ratio | 9.67 | 6 |
| 10 | log_src_bytes | 9.67 | 6 |
| 11 | log_dst_bytes | 9.83 | 6 |
| 12 | dst_bytes | 10.17 | 6 |
| 13 | src_dst_bytes_ratio | 10.33 | 6 |
| 14 | src_bytes | 10.33 | 6 |
| 15 | traffic_direction | 13.00 | 6 |
| 16 | has_rst | 15.67 | 6 |
| 17 | protocol_type | 16.17 | 6 |

### Consolidated Global Fingerprint Ranking

Average of LR and RF normalized rank positions. Features at the top are most responsible for dataset identification and hence for DOS failure.

| Global Rank | Feature | LR Avg Rank | RF Avg Rank | Combined Avg |
| --- | --- | --- | --- | --- |
| 1 | flag | 3.00 | 4.17 | 3.58 |
| 2 | protocol_service_flag | 3.00 | 6.00 | 4.50 |
| 3 | connection_state | 6.33 | 4.67 | 5.50 |
| 4 | duration | 8.17 | 3.17 | 5.67 |
| 5 | same_host_rate_x_service | 7.00 | 6.50 | 6.75 |
| 6 | service_tier | 5.17 | 9.50 | 7.33 |
| 7 | diff_srv_rate_x_flag | 9.67 | 6.17 | 7.92 |
| 8 | log_dst_bytes | 9.17 | 9.83 | 9.50 |
| 9 | log_src_bytes | 9.33 | 9.67 | 9.50 |
| 10 | traffic_direction | 8.00 | 13.00 | 10.50 |
| 11 | protocol_type | 5.67 | 16.17 | 10.92 |
| 12 | src_bytes | 11.67 | 10.33 | 11.00 |
| 13 | dst_bytes | 11.83 | 10.17 | 11.00 |
| 14 | count_x_srv_count | 14.33 | 8.00 | 11.17 |
| 15 | has_rst | 9.00 | 15.67 | 12.33 |
| 16 | src_dst_bytes_ratio | 15.00 | 10.33 | 12.67 |
| 17 | dst_src_bytes_ratio | 16.67 | 9.67 | 13.17 |

## Global Feature Importance Magnitude

Feature importance aggregated by mean absolute coefficient (LR) or mean Gini importance (RF) across all pairs. Higher magnitude = stronger discriminatory signal for dataset identity.

### Logistic Regression — Mean |Coefficient|

| Rank | Feature | Mean |Coefficient| |
| --- | --- | --- |
| 1 | protocol_service_flag | 5.727385 |
| 2 | flag | 5.660363 |
| 3 | service_tier | 4.172746 |
| 4 | same_host_rate_x_service | 2.948770 |
| 5 | connection_state | 2.306065 |
| 6 | protocol_type | 1.735645 |
| 7 | diff_srv_rate_x_flag | 1.585986 |
| 8 | log_src_bytes | 1.428158 |
| 9 | duration | 1.264490 |
| 10 | has_rst | 0.869174 |
| 11 | traffic_direction | 0.841970 |
| 12 | log_dst_bytes | 0.778759 |
| 13 | src_bytes | 0.470977 |
| 14 | dst_bytes | 0.317965 |
| 15 | count_x_srv_count | 0.203226 |
| 16 | src_dst_bytes_ratio | 0.094257 |
| 17 | dst_src_bytes_ratio | 0.022657 |

### Random Forest — Mean Importance (Gini)

| Rank | Feature | Mean Importance |
| --- | --- | --- |
| 1 | flag | 0.213684 |
| 2 | duration | 0.142261 |
| 3 | connection_state | 0.129812 |
| 4 | protocol_service_flag | 0.116896 |
| 5 | diff_srv_rate_x_flag | 0.076414 |
| 6 | dst_src_bytes_ratio | 0.061583 |
| 7 | count_x_srv_count | 0.059375 |
| 8 | same_host_rate_x_service | 0.057651 |
| 9 | service_tier | 0.039036 |
| 10 | src_dst_bytes_ratio | 0.031502 |
| 11 | log_src_bytes | 0.019637 |
| 12 | log_dst_bytes | 0.017235 |
| 13 | src_bytes | 0.014423 |
| 14 | dst_bytes | 0.013441 |
| 15 | traffic_direction | 0.006270 |
| 16 | protocol_type | 0.000553 |
| 17 | has_rst | 0.000227 |

## Top Fingerprint Features

The following 5 features are the most responsible for dataset identification across all pairs:

1. **flag** (LR rank=3.0, RF rank=4.2, combined=3.6)
2. **protocol_service_flag** (LR rank=3.0, RF rank=6.0, combined=4.5)
3. **connection_state** (LR rank=6.3, RF rank=4.7, combined=5.5)
4. **duration** (LR rank=8.2, RF rank=3.2, combined=5.7)
5. **same_host_rate_x_service** (LR rank=7.0, RF rank=6.5, combined=6.8)

These features encode dataset-specific distributional signatures that make dataset origin trivially identifiable.

## DOS Failure Diagnosis

The success criterion for Phase 43A is to determine whether DOS failure is driven by:

(A) **A small subset of highly discriminatory features** — if a few features dominate the global ranking, ablation can remove them.

(B) **Broad distributional differences across the entire canonical feature space** — if all features contribute meaningfully, feature-level intervention is unlikely to be effective.

- **LR top-5 concentration:** 68.4% of total |coefficient| magnitude
- **RF top-5 concentration:** 67.9% of total Gini importance

**Verdict: Mixed pattern — moderate concentration.** Top features contribute substantially, but the remaining features still encode significant dataset identity. Ablation may help but is unlikely to fully eliminate the fingerprint.

## Recommendations for Phase 43B — Feature Ablation

1. **Ablate top-5 fingerprint features** — targeted ablation may reduce identifiability, but residual signal is expected.
2. **Pair ablation with normalization** — standardize per-dataset distributions for the top features before ablation.
3. **Benchmark CORAL as fallback** — prepare a CORAL experiment in case ablation alone is insufficient.
4. **Benchmark DANN as second fallback** — if CORAL also fails, proceed to domain-adversarial training.

---

*Report generated by Phase 43A — Dataset Fingerprint Attribution*
