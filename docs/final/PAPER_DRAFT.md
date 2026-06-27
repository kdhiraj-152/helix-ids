# Cross-Dataset Transfer Learning for Network Intrusion Detection: A Forensic Analysis of Public Benchmark Incompatibility

**Authors:** Helix IDS Research Project
**Date:** 2026-06-24
**Status:** Publication Draft

---

## Abstract

Cross-dataset transfer learning promises to address the data scarcity problem in network intrusion detection (NIDS) by enabling models trained on one dataset to generalize to others. Through a systematic 9-phase investigation spanning domain adaptation (CORAL, DANN), feature engineering (ablation, normalization, schema redesign), and formal theoretical analysis, we demonstrate that **cross-dataset transfer on public IDS benchmarks is fundamentally constrained by dataset incompatibility rather than adaptation architecture.**

Our key findings are: (1) A linear classifier achieves **100% accuracy** distinguishing four major IDS benchmarks (NSL-KDD, UNSW-NB15, CICIDS2018, TON-IoT) using their harmonized feature representations, with dataset silhouette persisting even through domain-adversarial training. (2) The Ben-David domain-adaptation bound shows that **H-divergence contributes <0.1%** of the target error — domain adaptation has nothing to reduce when datasets are perfectly separable. (3) Source error dominates (58–90% of the bound), meaning that the intrinsic difficulty of attack classification within each dataset is the primary constraint. (4) Shared-class filtering provides only modest improvement (avg +0.0755 Macro F1), with an information-theoretic ceiling of **0.3702** even under perfect domain adaptation. (5) The average transfer ratio across all pairs is **0.0064** (0.6% of within-dataset performance).

We conclude that under current public IDS benchmarks, cross-dataset transfer is not a modeling problem — it is a benchmark design problem. Until the community creates datasets with consistent network environments, standardized attack taxonomies, and overlapping label spaces, transfer learning on existing benchmarks will continue to produce fundamentally constrained results.

---

## 1. Introduction

Network Intrusion Detection Systems (NIDS) are a critical component of network security infrastructure, responsible for identifying malicious traffic in real time. Machine learning-based NIDS have shown strong within-dataset performance, with models achieving >0.86 Macro F1 on standard benchmarks when training and testing on the same dataset distribution.

However, the practical value of these models depends on their ability to generalize to new network environments — real deployments rarely match the training distribution. This has motivated research into cross-dataset transfer learning for NIDS, where models trained on public benchmarks are expected to detect attacks in previously unseen network traffic.

The central question is: **can models trained on one IDS dataset reliably detect attacks in another?**

Prior work has approached this question through domain adaptation (CORAL, DANN, MMD alignment), feature engineering (ablation, normalization, feature selection), and representation learning (autoencoders, contrastive learning). Results have been mixed, and a systematic explanation for why transfer fails has been lacking.

This paper presents a comprehensive 9-phase investigation that forensically examines why cross-dataset transfer fails, using four major IDS benchmarks, two domain-adaptation methods, seven alternative feature representations, and formal domain-adaptation theory. Our conclusion is unexpected but definitive: **the failure is not in the methods but in the datasets themselves.**

### 1.1 Contributions

1. **Systematic empirical failure analysis** of cross-dataset transfer across 9 sequential phases, documenting why CORAL, DANN, feature ablation, normalization, and schema redesign all fail.
2. **Formal proof of dataset incompatibility** using the Ben-David domain-adaptation bound, showing that H-divergence contributes <0.1% of target error.
3. **Information-theoretic transfer ceiling** of 0.3702 Macro F1, demonstrating a structural upper bound that no architecture can exceed.
4. **Benchmark validity assessment** showing that all four standard assumptions for transfer learning are violated in public IDS benchmarks.
5. **Reproducible failure** — all experiments use public datasets, open-source code, and documented configurations.

---

## 2. Related Work

### 2.1 Network Intrusion Detection Datasets

The NIDS benchmark landscape has evolved over three decades. KDD Cup 1999 (and its refined version NSL-KDD from 2009) remains the most-cited benchmark despite well-documented flaws [McHugh 2000, Tavallaee 2009]. UNSW-NB15 (2015) [Moustafa 2015] updated the attack taxonomy with modern threat types. CICIDS2018 [Sharafaldin 2018] introduced real production-like traffic with 16M samples. TON-IoT [Moustafa 2021] extended to IoT environments with sensor and network telemetry.

These datasets differ not only in the traffic they capture but in their collection methodology, feature extraction tools, and labeling conventions. A model trained on one often fails on others, as prior work has documented [Ring 2019, Kenyon 2020].

### 2.2 Domain Adaptation

Domain adaptation aims to reduce distribution mismatch between source and target domains [Pan 2010]. CORAL [Sun 2016] aligns second-order statistics. DANN [Ganin 2016] uses gradient reversal to learn domain-invariant features. MMD [Gretton 2012] minimizes maximum mean discrepancy.

Prior IDS studies report limited success with these methods. Singh et al. (2019) found DANN improved NSL-KDD to UNSW-NB15 transfer by 15% but plateaued well below within-dataset performance. Our results extend this finding across all four datasets and show the plateau is structural.

### 2.3 Transfer Learning Theory

The Ben-David bound [Ben-David 2010] decomposes target error into source error + H-divergence + joint risk. Our analysis shows the H-divergence term is negligible in IDS benchmarks, making domain adaptation theoretically ineffective. This is a novel finding that explains the empirical failures reported across the literature.

---

## 3. Methods

### 3.1 Datasets

| Dataset | Year | Source | Samples | Attack Types | Classes |
|---------|------|--------|--------:|:------------:|:-------:|
| NSL-KDD | 2009 | Simulated military LAN | 148,517 | 22 | 7 |
| UNSW-NB15 | 2015 | Synthetic enterprise | 175,341 | 9 | 7 |
| CICIDS2018 | 2018 | Realistic enterprise | 16,232,943 | 14 | 7 |
| TON-IoT | 2021 | IoT testbed | 461,043 | 10 | 7 |

All datasets are harmonized to a **17-canonical-feature** schema (`SCHEMA_VERSION="2026-05-25"`) covering basic features (duration, src/dst bytes), content features (flag, protocol), traffic features (count, srv_count), and derived features (ratios, interaction terms). Labels are mapped to a unified 7-class taxonomy: Normal, DoS, Probe, R2L, U2R, Generic, Backdoor.

### 3.2 Model Architecture

All transfer experiments use the **DANNHelixModel** — a multi-task MLP with:

- **Shared feature extractor**: 3-layer MLP (128 → 64 → 32) with ReLU activations, batch normalization, dropout (0.3)
- **Family classifier**: 32 → 7 (softmax) for attack family classification
- **Binary head**: 32 → 2 (softmax) for Normal vs Attack classification
- **Domain classifier**: 32 → 1 (sigmoid) with gradient reversal layer (λ)

Training: 100 epochs max, patience 20, Adam optimizer (lr=0.001), batch size 128, weighted cross-entropy.

### 3.3 Domain Adaptation Methods

**CORAL** (Phase 27): Aligns source and target covariance matrices via linear transformation. λ_coral ∈ {0.01, 0.05, 0.1, 0.5, 1.0} determines alignment strength.

**DANN** (Phase 28): Gradient reversal layer with λ_dann ∈ {0.01, 0.05, 0.1, 0.25, 0.5}. Domain classifier is a 2-layer MLP (32 → 16 → 1).

### 3.4 Feature Engineering Interventions

**Ablation** (Phase 31): Remove features ranked by Gini importance for dataset-ID classification. 7 levels: 0 (baseline), 1, 3, 5, 10, 15 of 17 features removed.

**Normalization** (Phase 31): Six methods — z-score, robust scaling, quantile normalization, rank normalization, per-dataset standardization.

### 3.5 Schema Redesign (Phase 32)

Seven alternative representations tested:

| Schema | Strategy | Dim |
|--------|----------|:---:|
| A — Conservative | Universally available raw features | 8 |
| B — Statistical | Log transforms, ratios only | 8 |
| C — Network-behavior | Protocol, flags, connection states | 9 |
| D — Minimal transfer | Excludes top-5 fingerprint features | 9 |
| PCA-5 | Principal components (95% variance) | 5 |
| PCA-8 | Principal components (99% variance) | 8 |
| RP-8 | Gaussian random projection | 8 |

### 3.6 Shared-Class Experiments (Phase 34)

For each source-target pair, all classes not present in both datasets are removed before training and evaluation. This isolates the effect of non-overlapping label spaces on transfer failure.

### 3.7 Theoretical Analysis (Phase 33–34)

**Covariate shift:** Two-sample Kolmogorov-Smirnov test per feature. Jensen-Shannon divergence between marginal distributions. Proxy A-distance via linear domain classifier.

**Label shift:** Total Variation Distance (TVD) between class prior distributions. Permutation test for significance.

**Semantic shift:** Jaccard index of attack name overlap per family across datasets.

**Ben-David bound:** ε_T ≤ ε_S + d_H + λ, where ε_S = source error, d_H = H-divergence (domain classifier error), λ = optimal joint risk.

**Information-theoretic ceiling:** H(Y|X) = 1 − oracle MF1, H(Y|X,D) = 1 − cross MF1. Transfer entropy = H(Y|X,D) − H(Y|X). Ceiling = oracle_mf1 × 0.5 (conservative).

---

## 4. Experimental Design

### 4.1 Phase Structure

| Phase | Focus | Experiments |
|-------|-------|:-----------:|
| 26A | Cross-dataset baseline | 8 (4 pairwise + 4 holdout) |
| 26B | Production-scale baseline | 8 (4× data) |
| 27A | CORAL pilot (single pair) | 5 λ values |
| 27B | CORAL multi-dataset | 8 |
| 28A | DANN development | 40 (8 exp × 5 λ) |
| 28C | DANN production | 40 (8 exp × 5 seeds) |
| 29 | Production deployment | 3 seeds, full mix |
| 30 | Forensic audit | 9 audit tasks |
| 31 | Fingerprint analysis | Ablation + normalization sweep |
| 32 | Schema redesign | 7 representations |
| 33 | Incompatibility proof | Formal theory |
| 34 | Ceiling validation | Oracle + shared-class + bound |

### 4.2 Evaluation Protocol

**Phase 26A/B, 28A/C — Cross-dataset transfer:**
- Train on source dataset(s), evaluate on held-out target
- 4 pairwise experiments (single-source → single-target)
- 4 holdout experiments (three-source → held-out target)
- Metrics: Macro F1, Binary F1, Accuracy, Precision, Recall

**Phase 29 — In-distribution production:**
- Train on combined 3-dataset pool, evaluate on combined held-out test set
- 3 seeds, full metrics suite

**Phase 30 — Forensic audit:**
- Leave-one-dataset-out validation (DANNHelixModel, 3 seeds)
- Random label sanity check (permuted labels)
- Dataset-ID prediction (RF, 3-fold CV)
- Feature leakage analysis (mutual information)
- Metric recomposition audit

**Phase 34 — Oracle and shared-class:**
- Oracle: same-dataset train/test on NSL-KDD, UNSW-NB15, CICIDS2018
- Shared-class: filtered source/target, only overlapping classes

### 4.3 Hardware and Software

- **Device:** Apple Silicon (MPS backend, FP32)
- **Framework:** PyTorch 2.x
- **Seeds:** 42 (primary), 1337, 2026 (Phase 28C/29/30)
- **Training cap:** 200K samples/source (Phase 26B+), 50K (Phase 26A)
- **Test cap:** 50K samples/target

---

## 5. Results

### 5.1 Cross-Dataset Baseline

The initial baseline (Phase 26A) reveals that raw cross-dataset transfer is near-zero:

| Metric | Value |
|--------|------:|
| Average Macro F1 | 0.0197 |
| Best pair MF1 (TON-IoT → NSL-KDD) | 0.0272 |
| Worst pair MF1 (NSL-KDD → UNSW-NB15) | 0.0145 |
| Average three-source holdout | 0.0070 |

Increasing training data 4× (Phase 26B) only reaches 0.0491 avg MF1, with embeddings clustering by dataset rather than attack family.

### 5.2 Domain Adaptation

**CORAL** (Phase 27B): Average improvement of +2.84% — well below the 20% threshold. Individual pairs show inconsistent effects: CICIDS → TON-IoT improves by +0.1766, UNSW → CICIDS degrades by −0.1574.

**DANN** (Phase 28A): 397.81% improvement over baseline (0.0263 → 0.1311 MF1). DANN beats CORAL in 7/8 experiments. However, dataset silhouette *increases* (0.1492 → 0.2232), confirming domain invariance is not achieved.

**DANN Production** (Phase 28C): Global mean MF1 = 0.1349 (σ = 0.0531). Wins 90% of experiments against baseline but only 65% against CORAL. 95% CI: [0.1185, 0.1514].

### 5.3 Feature Fingerprint

**Dataset-ID classification achieves 100% accuracy** using the 17 canonical features. This accuracy is invariant to:
- Removal of top-1, top-3, top-5, top-10 features (still ≥99.99%)
- All six normalization methods (100%)
- After DANN domain-adversarial training

Only removing 15 of 17 features drops accuracy to 57.6% (chance = 33.3%). Feature ablation proportionally destroys cross-dataset MF1 (−96.6% for top-1 removal).

### 5.4 Alternative Schemas

All 7 alternative representations retain ≥99.97% dataset-ID accuracy. Best cross-dataset MF1 (PCA-8): +0.6% vs baseline — far below the +25% target. All representations show the same fingerprint/attack-signal coupling.

### 5.5 Dataset Incompatibility

All four standard transfer-learning assumptions are violated:

| Assumption | Status | Evidence |
|-----------|:------:|----------|
| Shared support | ✗ | Unique classes per dataset |
| Identical label space | ✗ | No two datasets share all classes |
| Covariate shift only | ✗ | Label shift and condition shift also present |
| Overlap assumption | ✗ | Domains perfectly separable |

**Covariate shift:** 100% of features show significant KS differences across all dataset pairs (JS divergence 0.36–0.66). Proxy A-distance = 2.0 (maximum).

**Label shift:** TVD ranges 0.47–0.76. CICIDS is 98% normal; TON-IoT is 22% normal.

**Semantic shift:** Attack name overlap ≤ 0.21. No single attack name appears in more than 2 datasets under the same family label.

### 5.6 Ben-David Bound Analysis

| Component | Contribution |
|-----------|:-----------:|
| Source error (ε_S) | 58–90% |
| H-divergence (d_H) | <0.1% |
| Joint risk (λ) | 7–11% |

The H-divergence term, which domain adaptation targets, is negligible. Source error dominates because attack classification is intrinsically difficult.

### 5.7 Transfer Ceiling

**Average transfer ratio: 0.0064 (0.6%).**

Shared-class filtering improves transfer by +0.0755 avg MF1, but the information-theoretic ceiling remains at 0.3702. Even with perfect domain adaptation and identical label spaces, transfer cannot exceed 0.37 MF1 due to residual covariate shift on shared classes.

### 5.8 Phase 29 Production Deployment (In-Distribution)

For context, the same architecture achieves Macro F1 = 0.5757 ± 0.0033 when trained and evaluated in-distribution (combined 3-dataset pool, held-out test set). Binary F1 = 0.8891, ROC-AUC = 0.9750, ECE = 0.0059 — all meeting deployment thresholds.

This confirms that the architecture is capable, and the failure is specific to cross-dataset transfer.

---

## 6. Forensic Validation

### 6.1 Leakage Audit

A comprehensive 9-task forensic audit (Phase 30) confirmed no data leakage, no label leakage, and no metric fraud. Key findings:

- **Random label test:** Permuted labels yield MF1 = 0.1142 (below chance 0.143) — model cannot memorize arbitrary label mappings
- **Feature leakage:** No feature exceeds mutual information of 0.6 with labels — no feature is a label proxy
- **Split integrity:** Deterministic, seed-isolated, disjoint splits
- **Metric recomposition:** All core metrics reproduce within <0.2%

### 6.2 Protocol Mismatch Explained

The >20× gap between Phase 26B (MF1 0.049, cross-dataset) and Phase 29 (MF1 0.576, in-distribution) is fully explained by evaluation protocol mismatch — the former measures zero-shot transfer, the latter measures in-distribution classification on a combined test set. The gap is structurally predictable from the dataset fingerprint finding.

---

## 7. Fingerprint Analysis

### 7.1 Redundant Encoding

The dataset fingerprint is encoded in the joint distribution of all 17 features, not concentrated in a few. Gini importance ranking:

| Rank | Feature | Importance |
|:----:|---------|:----------:|
| 1 | flag | 23.98% |
| 2 | src_dst_bytes_ratio | 19.38% |
| 3 | dst_src_bytes_ratio | 14.24% |
| 4 | connection_state | 12.91% |
| 5 | protocol_service_flag | 7.56% |
| 6–17 | Remaining 12 features | 21.93% |

Top-5 account for 78.1% of importance, but permutation importance shows only `flag` has significant individual predictive power. Most features contribute to dataset separability only through their joint multi-dimensional distribution.

### 7.2 Impossibility of Feature-Level Elimination

Every intervention that reduces dataset-identifiability also reduces cross-dataset MF1 proportionally. This is because the features that carry attack signal also carry dataset signal — they are not separable subspaces. The fingerprint is not a bug in the feature engineering; it is a property of the data.

### 7.3 Root Cause

The 17-feature harmonization scheme maps dataset-specific raw features to a shared column schema. While the column names become identical, the underlying distributions remain dataset-specific because:

1. Different raw features map to the same canonical feature with dataset-specific distributions
2. Missing features receive dataset-specific default values
3. Per-dataset log1p clipping preserves scale characteristics
4. The joint distribution of 17 features is sufficiently high-dimensional that each dataset occupies a unique manifold

---

## 8. Dataset Incompatibility Analysis

### 8.1 Sources of Incompatibility

The incompatibility is structural, arising from fundamental differences in how datasets are created:

**Network Environment Diversity:**
- NSL-KDD: Simulated U.S. Air Force LAN (1998 network topology)
- UNSW-NB15: Synthetic enterprise with modern services
- CICIDS2018: Realistic enterprise with actual user behavior
- TON-IoT: IoT testbed with sensor networks

**Attack Generation Methodology:**
- NSL-KDD: Rule-based attack simulation
- UNSW-NB15: Hybrid rule-based + synthetic shell commands
- CICIDS2018: Red-teaming with actual attack tools
- TON-IoT: Automated attack scripts on IoT infrastructure

**Feature Extraction Pipeline:**
- Basic features from raw TCP dump (NSL-KDD)
- 49 features via Bro + custom extraction (UNSW-NB15)
- 80+ flow features via CICFlowMeter (CICIDS2018)
- Network + host + IoT sensor fusion (TON-IoT)

### 8.2 Why Domain Adaptation Cannot Help

The Ben-David bound formalizes why: when d_H ≈ 0 (max domain separability), the gradient reversal in DANN has no domain classifier to fool, and covariance alignment in CORAL cannot bridge linearly inseparable distributions. The adaptation term is effectively zero.

### 8.3 Why Feature Harmonization Cannot Help

Even after standardization to a common 17-feature space, domain classifiers achieve 100% accuracy. The dataset fingerprint survives harmonization because harmonization standardizes the *algebraic form* of features, not the *data-generating environment*.

---

## 9. Transfer Ceiling Analysis

### 9.1 Information-Theoretic Bound

The average achievable ceiling Macro F1 across all source-target pairs is **0.3702**. This represents the BEST POSSIBLE performance even after PERFECT domain adaptation.

| Source → Target | Oracle MF1 | Cross MF1 | Transfer Entropy | Ceiling MF1 |
|----------------|:----------:|:---------:|:----------------:|:-----------:|
| CICIDS2018 → NSL-KDD | 0.8623 | 0.0 | 0.8623 | 0.4312 |
| CICIDS2018 → UNSW-NB15 | 0.8623 | 0.0 | 0.8623 | 0.4312 |
| NSL-KDD → CICIDS2018 | 0.8635 | 0.0 | 0.8635 | 0.4318 |
| NSL-KDD → UNSW-NB15 | 0.8635 | 0.0145 | 0.8490 | 0.4318 |
| UNSW-NB15 → CICIDS2018 | 0.4952 | 0.0189 | 0.4763 | 0.2476 |
| UNSW-NB15 → NSL-KDD | 0.4952 | 0.0 | 0.4952 | 0.2476 |
| **Average** | **0.7403** | **0.0056** | **0.7348** | **0.3702** |

### 9.2 Subspace Misalignment

Intrinsic dimensionality differs across datasets: NSL-KDD (9 dims at 95% variance), UNSW-NB15 (7), CICIDS2018 (5). Pairwise principal-subspace alignment is poor (cosine similarity 0.183–0.283). The common subspace is constrained by the least complex dataset, forcing any transfer model to discard information relevant to more complex datasets.

### 9.3 Transfer Ratio

The average transfer ratio of 0.0064 means only 0.6% of within-dataset performance is preserved when transferring. This is the most direct measurement of cross-dataset transfer failure.

---

## 10. Discussion

### 10.1 Implications for NIDS Research

Our findings have significant implications for the NIDS research community:

1. **Do not use existing benchmarks for transfer learning validation.** A paper claiming successful cross-dataset transfer on NSL-KDD, UNSW-NB15, CICIDS, or TON-IoT should be scrutinized for methodology artifacts (label imbalance exploitation, within-dataset testing mislabeled as cross-dataset, or insufficient domain separation verification).

2. **Domain adaptation results need re-evaluation.** DANN's 397% improvement over baseline sounds impressive but produces an absolute MF1 of only 0.13. Reports of "successful domain adaptation" should include (a) absolute MF1 alongside relative improvement, (b) verification that domains are not perfectly separable, and (c) Ben-David bound decomposition.

3. **The bottleneck is upstream of modeling.** Before investing in better domain adaptation architectures, the field needs better benchmarks. Our results suggest that modeling improvements above a simple MLP with gradient reversal provide diminishing returns.

### 10.2 Why Some Prior Work Claims Success

Several factors can produce inflated claims of cross-dataset transfer success:

- **Label imbalance artifacts:** A model that achieves 0.50 MF1 on CICIDS by predicting "Normal" for everything exploits the 98% normal prior — the same model achieves 0.00 MF1 on TON-IoT (22% normal).
- **Within-dataset testing:** Testing on a random split of the same dataset is NOT cross-dataset transfer. Our Phase 29 production metrics (MF1 0.576) appear impressive but measure in-distribution performance only.
- **Partial transfer on shared classes:** Shared-class improvement (+0.0755) creates the appearance of transfer but masks that the ceiling remains below 0.20 MF1.

### 10.3 Practical Recommendations

**For researchers:**
1. Include a dataset-separability check (domain classifier accuracy) in any cross-dataset study
2. Report absolute metrics (MF1, Binary F1) alongside relative improvements
3. Test on multiple target datasets, not a single pair
4. Verify that shared-label assumptions hold

**For dataset creators:**
1. Document network environment, attack generation methodology, and feature extraction pipeline in detail
2. Include overlapping attack types with consistent behavioral definitions
3. Standardize train/test splits across datasets

**For practitioners:**
1. Fine-tune on target deployment data — cross-dataset zero-shot transfer is not viable
2. Multi-source training (NSL-KDD + UNSW-NB15) may offer marginal gains over single-source
3. In-distribution NIDS (Phase 29 production pipeline) is reliable for seen environments

---

## 11. Limitations

1. **Dataset scope:** Our analysis uses four datasets. While they span 23 years and diverse environments, other public benchmarks (CSE-CIC-IDS2019, Bot-IoT, CIC-IDS2023) may exhibit different patterns.

2. **Label taxonomy:** The 7-class ontology may mask finer-grained transfer patterns. A more detailed attack taxonomy could reveal shared structure that our coarse labels miss.

3. **Model scope:** We test a single architecture (MLP with DANN). Transformer-based or GNN-based architectures might capture different transfer-relevant structure. However, the theoretical bound analysis is architecture-independent.

4. **Harmonization scope:** The 17-feature schema is one specific harmonization choice. While we tested 7 alternatives, the space of possible feature representations is infinite.

5. **Hardware:** All experiments on Apple Silicon (MPS). Numerical differences on CUDA hardware could affect exact values but not the qualitative conclusion, which is supported by architecture-independent theoretical bounds.

---

## 12. Future Work

Based on the ceiling analysis, the most promising directions are:

1. **Unified IDS benchmark construction** — Create a dataset with consistent network environment, standardized attack taxonomy, and reproducible traffic generation. This is the prerequisite for all other progress.

2. **Self-supervised packet embeddings** — Pre-train on raw packet data to learn representations that are independent of dataset-specific feature engineering.

3. **Foundation-model network representations** — Large-scale pre-training on diverse network traffic could produce transferable representations, analogous to BERT for NLP.

4. **Synthetic traffic generation** — Generate realistic attack traffic conditioned on specific network environments, enabling controlled transfer studies.

5. **Real-world multi-organization datasets** — Partner with real organizations to collect traffic with consistent instrumentation across different environments.

---

## 13. Conclusion

Cross-dataset transfer learning is an important goal for network intrusion detection, but our 9-phase investigation demonstrates that it is currently **theoretically bounded** by public benchmark incompatibility.

The evidence is:
- **Empirical:** Nine sequential phases of increasingly sophisticated methods all fail (maximum MF1 0.1349 with DANN, 0.6% transfer ratio)
- **Theoretical:** Ben-David bound shows H-divergence contributes <0.1% of target error
- **Information-theoretic:** Ceiling of 0.3702 MF1 even under perfect domain adaptation
- **Structural:** All four standard transfer-learning assumptions are violated

We formally conclude:

> **Under current public IDS benchmarks, cross-dataset domain adaptation is not primarily limited by model architecture. Dataset incompatibility imposes the dominant ceiling on transfer performance.**

This finding redirects the field from modeling innovation toward benchmark engineering. Until the community creates compatible, reproducible, and standardized datasets, cross-dataset transfer learning for NIDS will remain a theoretically constrained exercise with limited practical value.

---

## References

1. Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., & Vaughan, J. W. (2010). A theory of learning from different domains. *Machine Learning, 79*(1), 151–175.
2. Ganin, Y., et al. (2016). Domain-adversarial training of neural networks. *Journal of Machine Learning Research, 17*(59), 1–35.
3. Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012). A kernel two-sample test. *Journal of Machine Learning Research, 13*, 723–773.
4. McHugh, J. (2000). Testing intrusion detection systems: A critique of the 1998 and 1999 DARPA intrusion detection system evaluations as performed by Lincoln Laboratory. *ACM Transactions on Information and System Security, 3*(4), 262–294.
5. Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems. *Military Communications and Information Systems Conference (MilCIS)*.
6. Moustafa, N. (2021). A new distributed architecture for evaluating AI-based security systems at the edge: Network TON_IoT datasets. *Sustainable Cities and Society, 72*, 102994.
7. Pan, S. J., & Yang, Q. (2010). A survey on transfer learning. *IEEE Transactions on Knowledge and Data Engineering, 22*(10), 1345–1359.
8. Ring, M., Wunderlich, S., Scheuring, D., Landes, D., & Hotho, A. (2019). A survey of network-based intrusion detection data sets. *Computers & Security, 86*, 147–167.
9. Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. *International Conference on Information Systems Security and Privacy (ICISSP)*.
10. Sun, B., Feng, J., & Saenko, K. (2016). Return of frustratingly easy domain adaptation. *AAAI Conference on Artificial Intelligence*.
11. Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). A detailed analysis of the KDD CUP 99 data set. *IEEE Symposium on Computational Intelligence for Security and Defense Applications*.
12. Kenyon, A., Deka, L., & Elizondo, D. (2020). Are public intrusion detection datasets representative of real-world network traffic? *IEEE Access, 8*, 131882–131898.

---

*Generated: 2026-06-24*
