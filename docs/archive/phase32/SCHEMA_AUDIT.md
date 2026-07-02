# Schema Audit — Phase 32

**Date:** 2026-06-24
**Experiment:** Phase 32, RP-2 HELIX-IDS
**Status:** Complete

## Summary

Audit of all 17 canonical features mapping each back to its raw source columns across four public IDS datasets (NSL-KDD, UNSW-NB15, CICIDS-2018, TON-IoT). Each feature is classified by transfer-risk level based on how consistently it is sourced and processed across datasets.

**Transfer-risk distribution:**

| Risk Level | Count | Features |
|-----------|-------|----------|
| Very High | 6 | flag, connection_state, same_host_rate_x_service, diff_srv_rate_x_flag, count_x_srv_count, protocol_service_flag |
| High | 3 | src_dst_bytes_ratio, dst_src_bytes_ratio, service_tier |
| Medium | 3 | src_bytes, dst_bytes, traffic_direction |
| Low | 5 | protocol_type, has_rst, log_src_bytes, log_dst_bytes, duration |

## Feature-by-Feature Audit

### 1. protocol_type (Low Risk)

| Dataset | Raw Source | Processing |
|---------|-----------|------------|
| NSL-KDD | `protocol_type` | Categorical: tcp→0, udp→1, icmp→2 |
| UNSW-NB15 | `proto` | Same encoding |
| CICIDS-2018 | `Protocol` | Same encoding |
| TON-IoT | `proto` | Same encoding |

**Assessment:** Universally available. 3–4 distinct values. Direct categorical mapping with no dataset-specific logic. **Transfer-safe.**

---

### 2. connection_state (Very High Risk)

| Dataset | Raw Source | Processing |
|---------|-----------|------------|
| NSL-KDD | Derived from `flag` via `NSL_STATE_MAP` | REJ/S0 state → canonical INT→...→State 9 encoding |
| UNSW-NB15 | Derived from `state` via `UNSW_STATE_MAP` ± frozen/ambiguous handling | Same canonical encoding, but with frozen weights for rare states |
| CICIDS-2018 | Derived from SYN/RST/ACK/FIN count columns | Completely different: flag-bit counting → state |
| TON-IoT | Derived from `conn_state` via `TON_IOT_STATE_MAP` | State → canonical |

**Assessment:** Each dataset uses an entirely different method to derive connection state. NSL derives from text flags, UNSW from named state column with frozen-rare-state logic, CICIDS from flag counters, TON from a separate conn_state column. The canonical 9-state encoding hides massive dataset-specific derivation differences. **Dataset-specific.**

---

### 3. traffic_direction (Medium Risk)

| Dataset | Raw Source | Processing |
|---------|-----------|------------|
| NSL-KDD | `src_bytes > 1.5 × dst_bytes` | → outbound/inbound/balanced |
| UNSW-NB15 | `sbytes > 1.5 × dbytes` | Same threshold |
| CICIDS-2018 | `TotLen Fwd Pkts > 1.5 × TotLen Bwd Pkts` | Same threshold, different byte column |
| TON-IoT | `src_bytes > 1.5 × dst_bytes` | Same threshold |

**Assessment:** Derived from byte-ratio threshold. Concept is universal, but CICIDS uses aggregated forward/backward packet lengths (TotLen) while others use per-flow src/dst bytes. The threshold choice (1.5×) is a harmonization design decision. **Transfer-risk.**

---

### 4. has_rst (Low Risk)

| Dataset | Raw Source | Processing |
|---------|-----------|------------|
| NSL-KDD | `flag` text contains `rst|rej` | Binary indicator |
| UNSW-NB15 | `state` text contains `rst|rej` | Binary indicator |
| CICIDS-2018 | `RST Flag Cnt > 0` | Binary from separate flag counter column |
| TON-IoT | `conn_state` text contains `rst|rej` | Binary indicator |

**Assessment:** Simple binary feature. Concept is consistent (RST present/absent). Extraction path differs per dataset but yields functionally equivalent semantics. **Transfer-safe.**

---

### 5. log_src_bytes (Low Risk)

| Dataset | Raw Source | Processing |
|---------|-----------|------------|
| NSL-KDD | `src_bytes` | `abs(x) → log(1 + x)` |
| UNSW-NB15 | `sbytes` | Same |
| CICIDS-2018 | `TotLen Fwd Pkts` | Same |
| TON-IoT | `src_bytes` | Same |

**Assessment:** Standard log transform of source byte count. All datasets have this column (name varianation only). Log is a reversible, monotonic transform that does not introduce dataset-specific structure. **Transfer-safe.**

---

### 6. log_dst_bytes (Low Risk)

| Dataset | Raw Source | Processing |
|---------|-----------|------------|
| NSL-KDD | `dst_bytes` | `abs(x) → log(1 + x)` |
| UNSW-NB15 | `dbytes` | Same |
| CICIDS-2018 | `TotLen Bwd Pkts` | Same |
| TON-IoT | `dst_bytes` | Same |

**Assessment:** Identical situation to log_src_bytes. **Transfer-safe.**

---

### 7. src_dst_bytes_ratio (High Risk)

| Dataset | Raw Source | Processing |
|---------|-----------|------------|
| All | `src_bytes / max(dst_bytes, 1)` | Ratio construction |

**Assessment:** Phase 31 Gini importance rank: **#2** (0.1938). Ratio features encode byte-asymmetry patterns that are fundamentally dataset-specific. NSL-KDD has many one-sided connections (simulated attacks where only src_bytes is non-zero), while CICIDS has balanced bidirectional flows. This distributional difference creates a strong dataset signature. **Dataset-specific.**

---

### 8. dst_src_bytes_ratio (High Risk)

| Dataset | Raw Source | Processing |
|---------|-----------|------------|
| All | `dst_bytes / max(src_bytes, 1)` | Ratio construction |

**Assessment:** Phase 31 Gini importance rank: **#3** (0.1424). Reciprocal of src_dst_bytes_ratio; same dataset-specific byte-asymmetry problem. **Dataset-specific.**

---

### 9. same_host_rate_x_service (Very High Risk)

| Dataset | Raw Source | Processing |
|---------|-----------|------------|
| NSL-KDD | `same_srv_rate` or `dst_host_same_srv_rate` | `rate × (service_signal + 1)` |
| UNSW-NB15 | `same_srv_rate` or synthetic fallback | Same, synthetic fallback if rate missing |
| CICIDS-2018 | Synthetic fallback only | `src_bytes / max(src_bytes + dst_bytes, 1)` as proxy |
| TON-IoT | Synthetic fallback | Same proxy |

**Assessment:** When the rate column is missing (CICIDS, TON-IoT), a synthetic fallback based on byte ratios is injected. This creates **dataset-specific artifacts** — datasets with the raw rate column use actual connection-pattern statistics while datasets without use a byte-proxy that has different distributional properties. **Dataset-specific.**

---

### 10. diff_srv_rate_x_flag (Very High Risk)

| Dataset | Raw Source | Processing |
|---------|-----------|------------|
| NSL-KDD | `diff_srv_rate` and `flag` | `rate × (flag_signal + 1)` |
| UNSW-NB15 | `diff_srv_rate` and `state` | Same, flag encoding differs |
| CICIDS-2018 | Synthetic byte-difference proxy | Uses flag encoding that derives from unrelated columns |
| TON-IoT | Synthetic | Same byte proxy |

**Assessment:** Combines two dataset-identifying sources: (1) diff_srv_rate availability per dataset, and (2) flag encoding that varies per dataset's flag column. CICIDS has neither diff_srv_rate nor a native flag text column, forcing synthetic construction. **Dataset-specific.**

---

### 11. count_x_srv_count (Very High Risk)

| Dataset | Raw Source | Processing |
|---------|-----------|------------|
| NSL-KDD | `count` and `srv_count` | `count × srv_count` |
| UNSW-NB15 | `count` and `srv_count` | Same |
| CICIDS-2018 | Synthetic: `duration + src_bytes/max(dst_bytes, 1)` | Completely different computation |
| TON-IoT | `src_pkts × dst_pkts` (packet counts substitute) | Different semantics |

**Assessment:** NSL-KDD and UNSW have native count/srv_count columns representing connection-level statistics. CICIDS has neither, forcing a duration+byte-proxy synthetic. TON-IoT uses packet counts as substitute. Three different computational paths → three different signatures. **Dataset-specific.**

---

### 12. protocol_service_flag (Very High Risk)

| Dataset | Raw Source | Processing |
|---------|-----------|------------|
| All | Derived from protocol, service_tier, and flag | Triple interaction: `(p+1) × (s+1) × (f+1)` |

**Assessment:** Phase 31 Gini importance rank: **#5** (0.0756). Combines three dataset-specific encoding sources into one multiplicative interaction. Each of the three components has dataset-specific derivation, and the interaction amplifies these differences. **Dataset-specific.**

---

### 13. src_bytes (Medium Risk)

| Dataset | Raw Source | Processing |
|---------|-----------|------------|
| NSL-KDD | `src_bytes` | `abs(x)` + clip(−1e9, 1e9) |
| UNSW-NB15 | `sbytes` | Same |
| CICIDS-2018 | `TotLen Fwd Pkts` | Named differently: aggregated forward packet length |
| TON-IoT | `src_bytes` | Same |

**Assessment:** Universally available but CICIDS column measures a different concept: TotLen Fwd Pkts aggregates all forward-propagated packet lengths across a flow, while NSL/UNSW/TON src_bytes is per-record source bytes. This introduces systematic offset. **Transfer-risk.**

---

### 14. dst_bytes (Medium Risk)

| Dataset | Raw Source | Processing |
|---------|-----------|------------|
| NSL-KDD | `dst_bytes` | `abs(x)` + clip(−1e9, 1e9) |
| UNSW-NB15 | `dbytes` | Same |
| CICIDS-2018 | `TotLen Bwd Pkts` | Aggregated backward packet length |
| TON-IoT | `dst_bytes` | Same |

**Assessment:** Same as src_bytes — CICIDS naming/normalization difference. **Transfer-risk.**

---

### 15. service_tier (High Risk)

| Dataset | Raw Source | Processing |
|---------|-----------|------------|
| NSL-KDD | `service` column | Regex → tier: none(0), web, dns, ftp, mail, auth, other |
| UNSW-NB15 | `service` column | Same regex → same tiers |
| CICIDS-2018 | `Dst Port` → IANA port range | Port-based service classification |
| TON-IoT | `service` column | Same regex → same tiers |

**Assessment:** NSL-KDD, UNSW, and TON-IoT all derive service_tier from a named `service` column with shared regex logic. CICIDS has no named `service` column — it must use port ranges, which produces a different distribution (many higher port numbers mapped to `other`). **Dataset-specific.**

---

### 16. duration (Low Risk)

| Dataset | Raw Source | Processing |
|---------|-----------|------------|
| NSL-KDD | `duration` | Numeric coercion, NaN→0 |
| UNSW-NB15 | `dur` | Same |
| CICIDS-2018 | `Flow Duration` | Same |
| TON-IoT | `duration` | Same |

**Assessment:** Universally available. Measured similarly across all datasets (connection duration). **Transfer-safe.**

---

### 17. flag (Very High Risk)

| Dataset | Raw Source | Processing |
|---------|-----------|------------|
| NSL-KDD | `flag` (state text: REJ, S0, SF, ...) | `FLAG_MAP` 22-entry encoding |
| UNSW-NB15 | `state` (named column: FIN, CON, REQ, ...) | Same `FLAG_MAP` |
| CICIDS-2018 | Derived from SYN/RST/ACK/FIN flag count columns | Symbolic flag construction from counts → FLAG_MAP |
| TON-IoT | `conn_state` (SIN, SP, SAD, ...) | Same `FLAG_MAP` |

**Assessment:** Phase 31 Gini importance rank: **#1** (0.2398). The fundamental problem: while the encoding mapping is shared (FLAG_MAP), the underlying distribution of flags across datasets is radically different. NSL-KDD has many REJ and S0 states from simulated attacks; CICIDS computes flag states from normalized flag counters; UNSW has its own state naming convention; TON-IoT has its own conn_state naming. **Dataset-specific.**

## Classification Summary

| Feature | Transfer Risk | Root Cause |
|---------|--------------|------------|
| protocol_type | Transfer-safe | Universal 3–4 value encoding |
| has_rst | Transfer-safe | Simple binary, concept consistent |
| log_src_bytes | Transfer-safe | Log transform of raw byte count |
| log_dst_bytes | Transfer-safe | Log transform of raw byte count |
| duration | Transfer-safe | Universal column, direct measurement |
| src_bytes | Transfer-risk | CICIDS column name/concept variant |
| dst_bytes | Transfer-risk | CICIDS column name/concept variant |
| traffic_direction | Transfer-risk | Byte-ratio derived; data-dependent |
| src_dst_bytes_ratio | Dataset-specific | Ratio amplifies byte distribution differences |
| dst_src_bytes_ratio | Dataset-specific | Ratio amplifies byte distribution differences |
| service_tier | Dataset-specific | Port-based vs name-based derivation |
| connection_state | Dataset-specific | Per-dataset state machine logic |
| diff_srv_rate_x_flag | Dataset-specific | Synthetic fallback when column missing |
| same_host_rate_x_service | Dataset-specific | Synthetic fallback when column missing |
| count_x_srv_count | Dataset-specific | Synthetic fallback when column missing |
| protocol_service_flag | Dataset-specific | Triple interaction of dataset-specific encodings |
| flag | Dataset-specific | Per-dataset state distribution |

## Conclusion

**5 of 17 features are transfer-safe** (29%). The remaining 12 features have varying degrees of dataset-specific processing that encodes dataset identity into the canonical representation. The highest-risk features are the interaction terms (same_host_rate_x_service, diff_srv_rate_x_flag, count_x_srv_count, protocol_service_flag) and the connection behavior features (flag, connection_state), all of which depend on per-dataset column availability and dataset-specific fallback logic.

Per Phase 32 experimental results, **no combination or transformation of these 17 features reduces dataset-ID accuracy below 100%**, confirming that the fingerprint is embedded in the joint distribution of all features, not any individual feature or small subset.
