# Priority 3: External Validation Plan — Unseen Benchmark

**Requirement:** Test cross-dataset transfer on a dataset never touched during development.

## Current State

Phase 53 attempted external validation on IoT-23 and Kyoto 2006+ but **fell back to held-out partitions** because neither external dataset was actually downloaded. The Phase 53 results (0.45–0.61 zero-shot MF1) measure within-distribution heldout splits, NOT true external generalization.

---

## Three External Candidates

### 1. IoT-23 (highest priority)

| Aspect | Details |
|--------|---------|
| Source | CTU University, Stratosphere Lab |
| Size | ~21 labeled scenarios, ~140M flows |
| Difference from core 6 | IoT traffic (not general enterprise/IDS), Zeek/Bro conn logs (not CICFlowMeter) |
| Availability | `https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/` (verified accessible) |
| Small tarball | `iot_23_datasets_small.tar.gz` — 6 scenarios, suitable for initial test |
| Features | Conn.log columns (ts, uid, id.orig_h, id.resp_h, id.orig_p, id.resp_p, proto, service, duration, orig_bytes, resp_bytes, conn_state, local_orig, local_resp, missed_bytes, history, orig_pkts, orig_ip_bytes, resp_pkts, resp_ip_bytes, tunnel_parents, label, detailed_label) — needs mapping to 17 canonical features |

**Prediction:** Transfer MF1 will be < 0.10 (consistent with Phase 44c baseline on all IDS pairs). If it's higher, the feature harmonization (not the encoder) may be better at capturing IoT-relevant statistics.

### 2. UGR'16

| Aspect | Details |
|--------|---------|
| Source | University of Granada |
| Size | ~5B flows over 5 months |
| Difference from core 6 | Real ISP backbone traffic (not testbed/simulated), 2016 (older patterns) |
| Availability | `https://nesg.ugr.es/nesg-ugr16/` — verified accessible |
| Challenge | Labeled only for specific attack windows; mostly unlabeled background traffic |

**Best use:** Background drift validation rather than transfer MF1. Tests whether representations are stable across years (2016 real traffic → 2018+ testbed traffic).

### 3. Edge-IIoTset

| Aspect | Details |
|--------|---------|
| Source | Kaggle, 2022 |
| Size | ~221K samples |
| Difference from core 6 | Industrial IoT + cloud/edge scenarios |
| Availability | Requires Kaggle authentication — accessible but less straightforward |

---

## Protocol

For each external dataset D_external:

```
1. Train best SupCon encoder on all 6 core datasets (multi-source)
2. Freeze encoder
3. Train linear probe on source latents
4. Evaluate on D_external latents (zero-shot)
5. Report: transfer MF1, transfer binary MF1, transfer ratio (transfer/within-dataset)
6. Compare with: Phase 44c baseline (CE), Phase 50 SupCon, Phase 56 no-BN
```

### What Constitutes Success

| Transfer MF1 | Interpretation |
|---|---|
| ≥ 0.25 | **Strong generalization** — encoder captures truly transferable features |
| 0.10–0.25 | **Moderate** — comparable to best within-family transfer (Phase 44c: 0.09) |
| < 0.10 | **Bottleneck confirmed** — 17-feature P(Y\|X) bottleneck is IDS-wide, not just these 6 datasets |

### What We Learn

- If external transfer MF1 ≥ 0.15: **The 6-dataset limitation matters.** The encoder can generalize, just not across these datasets.
- If external transfer MF1 < 0.10: **The bottleneck is deeper than dataset-specific bias.** It's structural in the 17-feature representation.
- If IoT-23 > core-6 transfer: **The encoder captures IoT-relevant features better than enterprise-network ones**, which would be an important architectural finding.
