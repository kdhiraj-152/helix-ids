# Threats to Validity

**Project:** Helix IDS — Cross-Dataset Transfer Learning for Network Intrusion Detection
**Date:** 2026-06-24

---

## 1. Dataset Selection Bias

**Threat:** The four datasets used (NSL-KDD, UNSW-NB15, CICIDS2018, TON-IoT) may not be representative of all public IDS benchmarks. A different set of datasets could yield different transfer conclusions.

**Assessment:** MODERATE — Mitigated by diversity.

### Supporting Argument:
- The four datasets span 23 years of benchmark evolution (1999 KDD-derived → 2021 TON-IoT)
- They cover fundamentally different network environments: simulated military (NSL-KDD), synthetic enterprise (UNSW-NB15), real production traffic (CICIDS2018), IoT sensor networks (TON-IoT)
- They use different feature extraction tools: tcpdump-based, Bro, CICFlowMeter, custom
- They represent different attack generation methodologies: rule-based, synthetic, realistic red-teaming
- Sample sizes span three orders of magnitude: 107K (NSL-KDD) to 16M (CICIDS2018)

### Remaining Concern:
- **Older benchmarks:** NSL-KDD and UNSW-NB15 are from 2009 and 2015 respectively. Modern network environments (encrypted traffic, SDN, cloud-native) may have different transfer properties.
- **CIACA2017, Bot-IoT, CSE-CIC-IDS2019** were not included. These additional datasets could reveal different transfer patterns or confirm the ceiling with greater generality.
- **Industry datasets** are not available. The finding is strictly about *public* IDS benchmarks. Private/enterprise datasets with consistent instrumentation may support better transfer.

### Mitigation:
Phase 34's benchmark validity assessment explicitly identifies shared-support violations that are structural properties of public benchmarks. These violations are unlikely to be unique to the selected four; they arise from the benchmark design process itself (independent creation by different research groups).

---

## 2. Harmonization Bias

**Threat:** The 17-canonical-feature harmonization scheme may inadvertently introduce artifacts that inflate dataset separability or suppress transferable signal.

**Assessment:** MODERATE — Addressed by alternative schema experiments.

### Supporting Argument:
- Phase 32 tested 7 alternative representations (conservative, statistical, network-behavior, minimal-transfer, PCA-5, PCA-8, RP-8) — ALL preserved 100% dataset separability
- The fingerprints survive PCA (which removes feature interactions) and random projection (which preserves only distances)
- Harmonization follows a documented, auditable schema (`SCHEMA_VERSION="2026-05-25"`) that maps features using dataset-specific contracts (Phase 25A/C)
- Per-dataset log1p clipping preserves scale characteristics but this is a known tradeoff: aggressive normalization (quantile, rank) destroys attack signal without removing the fingerprint

### Remaining Concern:
- **Dataset-specific defaults:** Missing features (`has_rst`, `traffic_direction`, `protocol_service_flag`) receive dataset-specific default values that could encode dataset identity
- **Label mapping:** The 7-class taxonomy (Normal, DoS, Probe, R2L, U2R, Generic, Backdoor) homogenizes semantically different attacks under the same label. A richer label taxonomy might reveal transferable structure that our coarse mapping masks.
- **Feature engineering choices:** The specific log transforms and ratio features chosen for harmonization might not be optimal for transfer. However, Phase 32's comprehensive schema sweep makes alternative representations unlikely to change the conclusion.

### Mitigation:
Phase 32 directly tests the harmonization-bias hypothesis by trying 7 completely different feature representations. None succeeds, suggesting the bottleneck is not in the harmonization scheme.

---

## 3. Hardware Effects

**Threat:** All experiments run on Apple Silicon (MPS backend), which may produce different numerical behavior than CUDA (NVIDIA GPUs), potentially affecting reproducibility and generalization.

**Assessment:** LOW — Acceptable for a research conclusion.

### Supporting Argument:
- MPS uses FP32 training, identical to CUDA's FP32 path
- The MPS backend is deterministic at fixed seed (verified in Phase 30 audit)
- Phase 28C tested 5 fully independent seeds; variance (σ = 0.0531) is within normal training variance
- The Phase 30 random-label and feature-leakage audits confirm numerical correctness
- Inference latency and throughput are not central to the transfer conclusion (transfer effectiveness does not depend on inference speed)

### Remaining Concern:
- **CuDNN-specific optimizations** (Winograd convolutions, TensorCore matmul) are unavailable on MPS, but DANN Helix model is a simple MLP — no convolution paths would benefit from these optimizations
- **MPS memory architecture** limits batch size. However, sensitivity analysis at smaller batch sizes (Phase 26A vs 26B) showed the same pattern: more data → better source accuracy → same transfer failure
- **Quantization effects** were not tested. INT8/FP16 deployment on edge devices could introduce additional error but would not improve transfer

### Mitigation:
The transfer ceiling is a theoretical bound, not an empirical hardware artifact. Phase 34's information-theoretic bound is independent of hardware.

---

## 4. Training Variance

**Threat:** Reported results may be artifacts of specific hyperparameters, random seeds, or training configurations rather than stable properties of the data and model.

**Assessment:** LOW — Extensively addressed.

### Supporting Argument:
- **5-seed stability analysis** (Phase 28C): Global mean MF1 = 0.1349, σ = 0.0531 across 40 independent runs
- **Random label sanity check** (Phase 30): Random MF1 = 0.1142 (below chance 0.143), confirming no label leakage
- **Hyperparameter sensitivity** characterized: Phase 28A swept λ across 5 values (0.01–0.5) with reproducible λ-sensitivity patterns
- **Phase 32** reproduced the fingerprint finding at SEED=42 for all 7 alternative schemas
- **Phase 29** seed stability σ = 0.0033 across 3 seeds for production metrics
- All certification documents include exact hyperparameter configurations

### Remaining Concern:
- **Early stopping** (Phase 26B patience 20) may interact with learning rate scheduling in ways that favor source over target. However, Phase 30's domain generalization audit (same stopping criteria) showed consistent results.
- **Batch order randomization** was not audited. If source and target batches are not properly interleaved, DANN may converge to different equilibria. Phase 28C's seed variance (σ = 0.0531) captures this.
- **Bayesian hyperparameter optimization** was not performed. The manual sweep may not find optimal configurations. However, the Phase 33 theoretical analysis shows that hyperparameter optimization cannot address d_H ≈ 0 (max domain separability).

### Mitigation:
The primary conclusion — that benchmark incompatibility imposes a dominant ceiling — is supported by formal theoretical analysis (Ben-David bound, information-theoretic bound) that is parameter-independent. Empirical results are consistent across seeds, λ values, and experimental configurations.

---

## 5. Metric Limitations

**Threat:** The choice of Macro F1 as the primary metric may not capture all aspects of cross-dataset transfer performance and could systematically understate or overstate progress.

**Assessment:** LOW — Mitigated by multi-metric evaluation.

### Supporting Argument:
- **Five complementary metrics** tracked across all phases: Macro F1, Binary F1, Accuracy, Precision, Recall
- **Transfer Ratio** (cross MF1 / oracle MF1) normalizes by within-dataset difficulty
- **Silhouette scores** independently confirm representation quality
- **ROC-AUC** (Phase 29) confirms overall discriminability
- **ECE** (Phase 29) confirms calibration
- **Generalization gap** (train vs test accuracy) tracks overfitting

### Remaining Concern:
- **Macro F1 can be inflated by Normal class.** CICIDS (98% normal): a "always predict Normal" classifier achieves 98% accuracy but 0.50 binary F1. Our macro F1 weights all 7 classes equally, so this inflation is limited (Normal is 1/7 = 14.3% of macro F1, not 98%).
- **U2R and Backdoor have near-zero support** in some datasets (U2R: 0.17% in combined training). Their F1 ≈ 0 is expected and lowers the macro average. Alternate metrics like weighted F1 would mask this failure on minority classes.
- **Per-class metrics** should be examined for practical deployment decisions. Phase 29 reports per-class F1 (Normal 0.941, DoS 0.804, Probe 0.714, R2L 0.668, U2R 0.016, Generic 0.887, Backdoor 0.0).
- **Cohen's Kappa, Matthews Correlation Coefficient** were not tracked. These could provide additional perspective on chance-corrected performance.

### Summary

| Threat | Severity | Mitigation | Residual |
|--------|:-------:|------------|:--------:|
| Dataset selection bias | MODERATE | 4 diverse datasets, 23-year span | Additional datasets could vary results |
| Harmonization bias | MODERATE | 7 alternative schemas tested (Phase 32) | Label mapping coarseness |
| Hardware effects | LOW | MPS FP32, 5-seed analysis, deterministic | CUDA-specific optimizations untested |
| Training variance | LOW | 5 seeds, λ sweeps, random-label test, CI | Bayesian optimization not performed |
| Metric limitations | LOW | 5+ complementary metrics tracked | Minority class sensitivity, no MCC |

**Overall validity assessment:** The central conclusion — that cross-dataset IDS transfer is fundamentally bounded by benchmark incompatibility — is robust to all identified threats. The evidence is multi-faceted (theoretical bounding + 9 phases of empirical failure + independent replication via alternative schemas + shared-class experiments). The primary residual threat is dataset selection bias: the conclusion holds for the four tested benchmarks and likely generalizes to the broader public benchmark ecosystem, but cannot be assumed for private/enterprise datasets with consistent instrumentation.

---

*Generated: 2026-06-24*
