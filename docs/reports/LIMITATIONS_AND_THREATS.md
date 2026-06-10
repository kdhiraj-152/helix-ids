# Limitations and Threats to Validity

Last updated: 2026-06-09

## Internal Validity Threats

### 1. Threshold Decoupling Scope

The core contribution — per-class margin penalties in the loss function — addresses a specific failure mode where rare classes (R2L, U2R) achieve F1=0.000 due to gradient collapse. The technique works when:

- The rare class has at least some training samples (>10 per dataset)
- The rare class is semantically distinguishable from common classes
- The loss landscape has gradient signal for the rare class

**When it fails**: If a class has 0 training samples (e.g., classes 5 and 6 in the NSL-KDD split used during development), no amount of threshold tuning recovers it.

### 2. Metric Leakage

The primary metric (macro F1) is used for both:
- Early stopping / checkpoint selection
- Final reporting

This creates optimistic bias. The checkpoint that achieves the best validation macro F1 may not be the best model overall, and reporting this value overstates expected performance on unseen data.

**Mitigation**: Holdout test set is never used during training. But the validation set drives all decisions.

### 3. Hyperparameter Overfitting

Hyperparameters (focal gamma, margin tau, loss weights) were tuned on the same datasets they are evaluated on. No held-out hyperparameter tuning dataset exists.

**Risk**: Reported results may not generalize to new datasets or real-world traffic.

### 4. Single-Split Evaluation

The current pipeline uses a single train/val/test split (70/15/15). No k-fold cross-validation is implemented.

**Risk**: Results are sensitive to the specific split. A different split could yield different conclusions.

## External Validity Threats

### 5. Dataset Age

NSL-KDD is from 2009 (16+ years old). UNSW-NB15 is from 2015 (11 years old). Modern network traffic and attack patterns differ substantially:

| Factor | Academic Datasets | Real 2026 Traffic |
|--------|------------------|-------------------|
| Encryption | Minimal | Pervasive (TLS 1.3) |
| Attack volume | Fixed | Variable, adversarial |
| Protocol mix | HTTP-heavy | HTTP/2, QUIC, gRPC |
| Background noise | Clean | Noisy |

**Risk**: HELIX-IDS performance on these benchmarks may not predict real-world performance.

### 6. Single-Environment Evaluation

All experiments were conducted on a single hardware configuration (one training server). No cross-environment validation.

**Risk**: Hardware-specific results (latency, throughput) may not generalize to other configurations.

### 7. Edge Target Incompleteness

While edge deployment targets (RPi 4, RPi Zero, ESP32) are named in the architecture, no hardware benchmarks have been run on these targets. The claim of "edge-optimized" is unverified.

## Benchmark Limitations

### 8. No Baseline Comparisons

No traditional ML baselines (Random Forest, SVM, XGBoost) have been benchmarked against HELIX on the same datasets, splits, and metrics.

**Impact**: Cannot claim HELIX is better than existing methods — no evidence of improvement.

### 9. No Ablation Study

The contribution of individual components (TAM, threshold decoupling, class4 penalty, focal loss, curriculum, domain adaptation) has not been systematically ablated.

**Impact**: Cannot attribute observed performance to any specific component.

### 10. No Statistical Significance

Single-run results (or at most 3-seed consensus) without confidence intervals or significance testing.

**Impact**: Differences between configurations may be due to random variation rather than actual improvement.

## Dataset Limitations

### 11. CICIDS-2018 Coverage

CICIDS-2018 comprises 7 attack types. HELIX-IDS uses a 7-class taxonomy that does not fully capture the dataset's original label granularity. Rare attack types within CICIDS-2018 may be collapsed into broader categories, losing detection specificity.

### 12. Label Mapping Discrepancies

The 7-class taxonomy is a best-effort mapping across three independently-labeled datasets. The mapping may:
- Collapse attacks with different characteristics into the same family
- Separate attacks that belong together
- Mislabel edge cases (attacks that don't fit any family)

## Security Limitations

### 13. No Cryptographic Authenticity

Artifact provenance provides integrity (detect tampering) but not authenticity (verify creator). See `docs/SECURITY_REVIEW.md` for details.

### 14. No Adversarial Defense at Inference

While adversarial evaluation code exists (`adversarial_test.py`), there is no real-time adversarial input detection or defense during live inference.

### 15. No Access Control

The REST inference endpoint (`serve_rest.py`) has no authentication, no rate limiting, and no request logging. Any client that can reach the endpoint can query the model arbitrarily.

## Provenance Limitations

### 16. Self-Verified Manifests

Artifact manifests are created by the training pipeline and verified by the same system. There is no independent notary or witness.

**Consequence**: A compromised training pipeline can produce valid manifests for malicious artifacts.

### 17. No External Timestamp Authority

Timestamps in manifests use local system time. Temporal ordering is only as reliable as the local clock.

## Summary of Critical Gaps

| Gap | Impact | Priority |
|-----|--------|----------|
| No baseline comparisons | Cannot claim improvement over existing methods | **CRITICAL** |
| No ablation study | Cannot attribute performance to components | **CRITICAL** |
| No cross-validation | Results may be split-dependent | **HIGH** |
| No statistical significance | Results may be noise | **HIGH** |
| No edge benchmarks | "Edge-optimized" claim unverified | **HIGH** |
| No access control on inference | Deployment security risk | **HIGH** |
| Datasets 11-16 years old | Real-world relevance uncertain | **MEDIUM** |
| No cryptographic authenticity | Provenance can be forged | **MEDIUM** |
| No Docker container | Environment not fully reproducible | **MEDIUM** |
