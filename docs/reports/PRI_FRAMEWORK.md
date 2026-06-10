# Production Readiness Index (PRI) Framework

> A systematic scorecard for evaluating production readiness of HELIX-IDS models.

Last updated: 2026-06-09

## Overview

The PRI Framework defines a quantitative assessment of a trained model's readiness for production deployment. It produces a single 0–100 score across 6 dimensions.

## Scoring Dimensions

### D1: Detection Quality (35 points max)

| Component | Points | Threshold | Weight |
|-----------|--------|-----------|--------|
| Macro F1 | 10 | >0.85 → 10 pts; 0.75–0.85 → 7 pts; 0.60–0.75 → 3 pts | 0.20 |
| Binary F1 | 5 | >0.95 → 5 pts; 0.90–0.95 → 3 pts | 0.10 |
| Threat-weighted F1 | 10 | >0.80 → 10 pts; 0.70–0.80 → 6 pts | 0.20 |
| Class 4 (U2R) F1 | 10 | >0.70 → 10 pts; 0.50–0.70 → 6 pts | 0.20 |

**Formula**: D1 = 10*(macro_ok) + 5*(binary_ok) + 10*(threat_ok) + 10*(u2r_ok)

### D2: Operational Performance (25 points max)

| Component | Points | Threshold | Weight |
|-----------|--------|-----------|--------|
| Latency (p95) | 10 | <100ms → 10 pts; 100–500ms → 5 pts | 0.20 |
| Throughput | 10 | >100 req/s → 10 pts; >10 req/s → 5 pts | 0.20 |
| Memory (RSS) | 5 | <1 GB → 5 pts; <4 GB → 2 pts | 0.10 |

**Formula**: D2 = 10*(latency_ok) + 10*(throughput_ok) + 5*(memory_ok)

### D3: Robustness (15 points max)

| Component | Points | Threshold | Weight |
|-----------|--------|-----------|--------|
| Adversarial robustness | 8 | Accuracy drop <10% under FGSM → 8 pts | 0.16 |
| Schema drift resistance | 4 | Schema guard active → 4 pts | 0.08 |
| Input validation | 3 | NaN/Inf detection → 3 pts | 0.06 |

**Formula**: D3 = 8*(adv_ok) + 4*(schema_guard_ok) + 3*(input_val_ok)

### D4: Provenance & Governance (15 points max)

| Component | Points | Threshold | Weight |
|-----------|--------|-----------|--------|
| Artifact manifest present | 5 | Embedded manifest → 5 pts | 0.10 |
| Provenance chain intact | 5 | Full chain verified → 5 pts | 0.10 |
| Contract compliance | 5 | Schema contract passed → 5 pts | 0.10 |

**Formula**: D4 = 5*(manifest) + 5*(chain) + 5*(contract)

### D5: Reproducibility (10 points max)

| Component | Points | Threshold | Weight |
|-----------|--------|-----------|--------|
| Deterministic training | 4 | Seed recorded → 4 pts | 0.08 |
| Dataset fingerprint | 3 | Fingerprint in manifest → 3 pts | 0.06 |
| Training config captured | 3 | Config hash in manifest → 3 pts | 0.06 |

**Formula**: D5 = 4*(seed) + 3*(fingerprint) + 3*(config)

### Total PRI Score

```
PRI = min(100, D1 + D2 + D3 + D4 + D5)
```

### Interpretation

| Score | Label | Action |
|-------|-------|--------|
| 90–100 | Production Ready | Deploy with standard monitoring |
| 75–89 | Staging Ready | Deploy to staging, monitor 7 days |
| 50–74 | Development | Cannot deploy; address gaps |
| <50 | Experimental | Not suitable for production use |

## Status: NOT IMPLEMENTED

The PRI scoring framework is defined above but **has not been implemented in code** and **no model has been scored**. Implementation requires:

1. Add `compute_pri_score()` to `utils/metrics.py` (the function exists as a stub in calculate_pri_score)
2. Create a `scripts/operations/assess_pri.py` entrypoint
3. Run benchmarks to generate the required inputs

## Limitations

1. **Subjective thresholds**: The scoring thresholds (e.g., "Macro F1 > 0.85") are based on internal standards and may not generalize.
2. **No weighting calibration**: Weights are equal across all dimensions, which may not reflect real-world priorities.
3. **Single-point evaluation**: PRI is computed at a point in time and does not account for model degradation over time.
4. **Hardware-dependent**: D2 scores are specific to the deployment target — a model scores differently on server vs. edge.
5. **No security score**: The framework does not include a security posture dimension.
6. **Partial data required**: Several inputs require benchmarks that have not been run.
