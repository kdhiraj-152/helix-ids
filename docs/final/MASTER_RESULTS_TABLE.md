# Master Results Table

**Project:** Helix IDS — Cross-Dataset Transfer Learning for Network Intrusion Detection
**Date:** 2026-06-24

## Notation

- **MF1** = Macro F1 (unweighted average across 7 attack families)
- **BinF1** = Binary F1 (Normal vs Attack)
- **TR** = Transfer Ratio (cross-dataset MF1 / within-dataset oracle MF1)
- **DS-ID** = Dataset-Identification Accuracy (RF classifier predicting source dataset)
- **Bold** = best value in column
- **—** = metric not applicable or not measured
- All values derived from certified phase documentation

---

## 1. Baseline Performance (Phase 26A/B)

### Pairwise Cross-Dataset Transfer — Macro F1

| Source ↓ / Target → | NSL-KDD | UNSW-NB15 | CICIDS2018 | TON-IoT |
|---------------------|--------:|----------:|-----------:|--------:|
| **NSL-KDD** | **0.8635** (oracle) | 0.0145 | — | — |
| **UNSW-NB15** | — | **0.4952** (oracle) | 0.0189 | — |
| **CICIDS2018** | — | — | **0.8623** (oracle) | 0.0184 |
| **TON-IoT** | 0.0272 | — | — | 0.0000 (oracle N/A) |

*Source: Phase 26A certification, Phase 34 oracle measurement*

### Three-Source Holdout Transfer

| Held-out Target | Phase 26A MF1 | Phase 26B MF1 | Phase 28A DANN MF1 | Phase 28C DANN μ±σ |
|-----------------|-------------:|-------------:|-------------------:|-------------------:|
| TON-IoT | 0.0239 | 0.0119 | **0.2113** | 0.1392±0.0445 |
| CICIDS2018 | 0.0002 | 0.0000 | 0.1870 | **0.1902±0.0680** |
| NSL-KDD | 0.0020 | 0.0004 | 0.1099 | **0.1549±0.0226** |
| UNSW-NB15 | 0.0018 | 0.0020 | **0.0617** | 0.0721±0.0069 |
| **Average** | **0.0070** | **0.0036** | **0.1425** | **0.1391** |

*Source: Phase 26A, 26B, 28A, 28C certifications*

---

## 2. Domain Adaptation — CORAL (Phase 27B)

| Experiment | Baseline MF1 | CORAL MF1 | Δ | Direction |
|-----------|------------:|----------:|--:|:---------:|
| NSL-KDD → UNSW-NB15 | 0.0511 | 0.0528 | +0.0017 | 🟢 |
| UNSW-NB15 → CICIDS2018 | 0.1988 | 0.0415 | −0.1574 | 🔴 |
| CICIDS2018 → TON-IoT | 0.0765 | **0.2531** | +0.1766 | 🟢 |
| TON-IoT → NSL-KDD | 0.0893 | 0.1296 | +0.0403 | 🟢 |
| 3-src → TON-IoT (holdout) | 0.1733 | 0.1537 | −0.0196 | 🔴 |
| 3-src → CICIDS (holdout) | 0.0541 | **0.1684** | +0.1142 | 🟢 |
| 3-src → NSL-KDD (holdout) | 0.0424 | 0.1083 | +0.0659 | 🟢 |
| 3-src → UNSW (holdout) | 0.0113 | 0.0167 | +0.0053 | 🟢 |

| Aggregate Metric | Value | Threshold | Result |
|-----------------|-----:|----------:|:------:|
| Avg MF1 Δ | +0.0284 (+2.84%) | ≥ +20% | ❌ FAIL |
| Avg Silhouette Δ | +0.0618 | ≤ −15% | ❌ FAIL |
| Wins/Losses | 6/2 | ≥ 5/8 | ✅ PASS |

*Source: Phase 27B certification*

---

## 3. Domain Adaptation — DANN (Phase 28A/C)

### Phase 28A (Development, Single Seed, Best λ)

| Experiment | Phase 26B MF1 | CORAL MF1 | DANN MF1 | Δ vs 26B | Δ vs CORAL |
|-----------|------------:|----------:|---------:|---------:|----------:|
| NSL-KDD → UNSW-NB15 | 0.1068 | 0.0528 | 0.0645 | −0.0423 🔴 | +0.0117 🟢 |
| UNSW-NB15 → CICIDS2018 | 0.0196 | 0.0415 | **0.1299** | +0.1103 🟢 | +0.0884 🟢 |
| CICIDS2018 → TON-IoT | 0.0633 | 0.2531 | 0.1349 | +0.0716 🟢 | −0.1182 🔴 |
| TON-IoT → NSL-KDD | 0.0067 | 0.1296 | **0.1498** | +0.1431 🟢 | +0.0202 🟢 |
| 3-src → TON-IoT | 0.0119 | 0.1537 | **0.2113** | +0.1994 🟢 | +0.0576 🟢 |
| 3-src → CICIDS | 0.0000 | 0.1684 | 0.1870 | +0.1870 🟢 | +0.0186 🟢 |
| 3-src → NSL-KDD | 0.0004 | 0.1083 | 0.1099 | +0.1095 🟢 | +0.0016 🟢 |
| 3-src → UNSW-NB15 | 0.0020 | 0.0167 | 0.0617 | +0.0597 🟢 | +0.0450 🟢 |

### Phase 28C (Production, 5 Seeds)

| Experiment | DANN μ | DANN σ | Wins/5 vs CORAL | Wins/5 vs 26B |
|-----------|------:|------:|:---------------:|:-------------:|
| NSL-KDD → UNSW-NB15 | 0.0943 | 0.0121 | 5/5 | 0/5 |
| UNSW-NB15 → CICIDS2018 | 0.1395 | 0.0026 | 5/5 | 5/5 |
| CICIDS2018 → TON-IoT | 0.1465 | 0.0587 | 0/5 | 5/5 |
| TON-IoT → NSL-KDD | 0.1425 | 0.0715 | 2/5 | 5/5 |
| 3-src → TON-IoT | 0.1392 | 0.0445 | 1/5 | 5/5 |
| 3-src → CICIDS | 0.1902 | 0.0680 | 3/5 | 5/5 |
| 3-src → NSL-KDD | 0.1549 | 0.0226 | 5/5 | 5/5 |
| 3-src → UNSW-NB15 | 0.0721 | 0.0069 | 5/5 | 5/5 |
| **Global** | **0.1349** | **0.0531** | **65.0%** | **90.0%** |

*Source: Phase 28A/C certifications*

---

## 4. Feature Ablation Results (Phase 31)

| Features Removed | Dataset-ID Accuracy | vs All-17 Δ | Avg Cross-Dataset MF1 | vs Baseline Δ |
|-----------------|-------------------:|:-----------:|:--------------------:|:------------:|
| 0 (Baseline) | **100.0%** | — | **0.099** | — |
| Top-1 | 100.0% | 0.0pp | 0.003 | −96.6% |
| Top-3 | 100.0% | 0.0pp | 0.004 | −96.3% |
| Top-5 | 100.0% | 0.0pp | 0.015 | −84.4% |
| Top-10 | 99.99% | −0.01pp | 0.053 | −46.0% |
| Top-15 (keep 2) | 57.6% | −42.4pp | — | — |

### Normalization Methods (Phase 31)

| Method | Dataset-ID Acc | Cross-Dataset MF1 Δ |
|--------|:--------------:|:-------------------:|
| None (baseline) | 100.0% | — |
| Z-score | 100.0% | baseline |
| Robust scaling | 100.0% | baseline |
| Quantile normalization | 100.0% | −78% |
| Rank normalization | 100.0% | −55% |
| Per-dataset normalization | 100.0% | baseline |

*Source: Phase 31 fingerprint analysis, feature ablation results*

---

## 5. Alternative Schemas — Cross-Dataset Metrics (Phase 32)

| Schema | Features | DS-ID Acc | Avg MF1 | Δ vs Baseline | Attack Silhouette |
|--------|:-------:|:---------:|:-------:|:-------------:|:-----------------:|
| Phase31 Baseline | 17 | 100.0% | 0.3717 | — | −0.0090 |
| Schema-A (Conservative) | 8 | 100.0% | 0.3681 | −1.0% | −0.112 |
| Schema-B (Statistical) | 8 | 100.0% | 0.3092 | −16.8% | −0.091 |
| Schema-C (Network-behavior) | 9 | 100.0% | 0.3639 | −2.1% | −0.017 |
| Schema-D (Minimal transfer) | 9 | 99.97% | 0.3098 | −16.7% | −0.083 |
| PCA-5 | 5 | 100.0% | (not reported) | — | — |
| PCA-8 | 8 | 100.0% | **0.3739** | **+0.6%** | −0.042 |
| RP-8 | 8 | 99.99% | 0.2968 | −20.2% | −0.066 |

| Criterion | Target | Best | Result |
|-----------|:-----:|:----:|:------:|
| Dataset-ID Accuracy | < 80% | 99.97% | ❌ FAIL |
| MF1 Improvement | ≥ +25% | +0.6% | ❌ FAIL |
| Attack Silhouette Improvement | Positive | −0.009 → −0.017 | ❌ FAIL |

*Source: Phase 32 schema certification*

---

## 6. Shared-Class Transfer (Phase 34)

| Source → Target | Shared Classes | Full MF1 | Shared MF1 | Improvement | Transfer Ratio |
|----------------|:--------------:|:--------:|:----------:|:-----------:|:-------------:|
| NSL-KDD → UNSW-NB15 | 5 | 0.0145 | **0.1885** | +0.1740 | 0.0168 |
| NSL-KDD → CICIDS2018 | 4 | 0.0000 | 0.1527 | +0.1527 | 0.0000 |
| NSL-KDD → TON-IoT | 0 | 0.0000 | 0.0000 | +0.0000 | 0.0000 |
| UNSW-NB15 → NSL-KDD | 5 | 0.0000 | 0.1543 | +0.1543 | 0.0000 |
| UNSW-NB15 → CICIDS2018 | 5 | 0.0189 | 0.1787 | +0.1598 | 0.0381 |
| UNSW-NB15 → TON-IoT | 0 | 0.0000 | 0.0000 | +0.0000 | 0.0000 |
| CICIDS2018 → NSL-KDD | 4 | 0.0000 | 0.1711 | +0.1711 | 0.0000 |
| CICIDS2018 → UNSW-NB15 | 5 | 0.0000 | 0.1392 | +0.1392 | 0.0000 |
| CICIDS2018 → TON-IoT | 0 | 0.0184 | 0.0000 | −0.0184 | 0.0213 |
| TON-IoT → NSL-KDD | 0 | 0.0272 | 0.0000 | −0.0272 | 0.0000 |
| TON-IoT → UNSW-NB15 | 0 | 0.0000 | 0.0000 | +0.0000 | 0.0000 |
| TON-IoT → CICIDS2018 | 0 | 0.0000 | 0.0000 | +0.0000 | 0.0000 |

| Aggregate Metric | Value |
|-----------------|------:|
| Avg shared-class improvement | +0.0755 |
| Max shared-class improvement | +0.1740 |
| Pairs improved | 6/12 |
| Pairs worsened | 2/12 |

*Source: Phase 34 shared-class results*

---

## 7. Oracle (Within-Dataset) Performance (Phase 34)

| Dataset | Accuracy | Macro F1 | Precision | Recall | Samples |
|---------|:-------:|:--------:|:---------:|:------:|:-------:|
| NSL-KDD | 0.9794 | **0.8635** | 0.9794 | 0.9794 | 10,000 |
| UNSW-NB15 | 0.7944 | 0.4952 | 0.7620 | 0.7944 | 10,000 |
| CICIDS2018 | 0.9649 | **0.8623** | 0.9656 | 0.9649 | 10,000 |
| **Average** | **0.9129** | **0.7403** | — | — | — |

*Source: Phase 34 oracle results*

---

## 8. Phase 29 Production Metrics (In-Distribution)

| Metric | μ±σ | Threshold | Status |
|--------|:---:|:--------:|:------:|
| Macro F1 | 0.5757±0.0033 | ≥ 0.12 | ✅ |
| Binary F1 | 0.8891 | ≥ 0.80 | ✅ |
| Accuracy | 0.8811 | — | (reference) |
| ROC-AUC (OvR) | 0.9750 | ≥ 0.70 | ✅ |
| ECE (calibration) | 0.0059 | < 0.05 | ✅ |
| Seed stability σ | 0.0033 | ≤ 0.03 | ✅ |
| Latency | 0.39ms | — | (reference) |
| Throughput | 639,964 samples/s | — | (reference) |

*Source: Phase 29 deployment certification*

---

## 9. Cross-Dataset Domain Divergence (Phase 33)

| Pair | JS Divergence | TVD | Proxy A-distance | KS Signif. Features | Semantic Overlap |
|------|:------------:|:---:|:----------------:|:------------------:|:----------------:|
| NSL-KDD ↔ UNSW-NB15 | 0.36 | 0.47 | 2.0 (max) | 17/17 (100%) | 0.21 |
| NSL-KDD ↔ CICIDS2018 | 0.41 | 0.76 | 2.0 (max) | 17/17 (100%) | 0.18 |
| NSL-KDD ↔ TON-IoT | 0.54 | 0.52 | 2.0 (max) | 17/17 (100%) | 0.17 |
| UNSW-NB15 ↔ CICIDS2018 | 0.38 | 0.58 | 2.0 (max) | 17/17 (100%) | 0.21 |
| UNSW-NB15 ↔ TON-IoT | 0.66 | 0.69 | 2.0 (max) | 17/17 (100%) | 0.17 |
| CICIDS2018 ↔ TON-IoT | 0.52 | 0.70 | 2.0 (max) | 17/17 (100%) | 0.17 |

*Source: Phase 33 covariate shift and label shift analyses*

---

## 10. Transfer Ceiling Estimates (Phase 34)

| Source → Target | Oracle MF1 | Cross MF1 | Transfer Entropy | Ceiling MF1 | Info Loss % |
|----------------|:----------:|:---------:|:----------------:|:-----------:|:----------:|
| CICIDS2018 → NSL-KDD | 0.8623 | 0.0000 | 0.8623 | 0.4312 | 86.2% |
| CICIDS2018 → UNSW-NB15 | 0.8623 | 0.0000 | 0.8623 | 0.4312 | 86.2% |
| NSL-KDD → CICIDS2018 | 0.8635 | 0.0000 | 0.8635 | 0.4318 | 86.4% |
| NSL-KDD → UNSW-NB15 | 0.8635 | 0.0145 | 0.8490 | 0.4318 | 84.9% |
| UNSW-NB15 → CICIDS2018 | 0.4952 | 0.0189 | 0.4763 | 0.2476 | 47.6% |
| UNSW-NB15 → NSL-KDD | 0.4952 | 0.0000 | 0.4952 | 0.2476 | 49.5% |
| **Average** | **0.7403** | **0.0056** | **0.7348** | **0.3702** | — |

---

## Summary — All Methods Compared

| Method | Avg Cross-Dataset MF1 | Best Cross MF1 | DS-ID Acc | Transfer Ratio |
|--------|:--------------------:|:--------------:|:---------:|:-------------:|
| **Baseline** (26A) | 0.0197 | 0.0272 | 100.0% | 0.0064 |
| **Baseline** (26B, 4× data) | 0.0491 | 0.1068 | 100.0% | — |
| **CORAL** (27B) | 0.1155 | 0.2531 | — | — |
| **DANN** (28A, best λ) | 0.1311 | 0.2113 | — | — |
| **DANN** (28C, prod, 5 seeds) | 0.1349 | 0.1902 | — | — |
| **Feature Ablation** (31) | 0.003–0.053 | — | 100–99.99% | — |
| **Schema-A** (32) | 0.3681 | — | 100.0% | — |
| **Schema-B** (32) | 0.3092 | — | 100.0% | — |
| **Schema-C** (32) | 0.3639 | — | 100.0% | — |
| **Schema-D** (32) | 0.3098 | — | 99.97% | — |
| **PCA-8** (32) | **0.3739** | — | 100.0% | — |
| **Shared-Class** (34) | 0.0755 (gain) | 0.1885 | — | — |
| **Oracle** (within-dataset) | **0.7403** | **0.8635** | — | **1.0** |

**Key Insight:** No method — CORAL, DANN, feature ablation, schema redesign, PCA, random projection, or shared-class filtering — achieves a cross-dataset Macro F1 above 0.38, while within-dataset performance reaches 0.86. The transfer ceiling is structural, not architectural.

---

*Generated: 2026-06-24*
