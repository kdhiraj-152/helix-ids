# Canonical Feature Specification v1.0

> **Phase 36 — Deliverable 2 of 8**
> Defines the standard feature set for cross-dataset IDS research.
> Every dataset in the benchmark must produce ALL features via a uniform pipeline.
> Date: 2026-06-24

---

## 1. Purpose

Current IDS benchmarks use incompatible feature extraction pipelines.
CICFlowMeter (CICIDS2018), Argus (UNSW-NB15), and custom parsers (NSL-KDD, TON-IoT)
produce features with different semantics, ranges, and collection methodologies.
This makes cross-dataset transfer learning impossible at the feature level.

This specification defines a **single, canonical 22-feature set** that every benchmark
dataset must produce through a **standardized extraction pipeline**. Features are chosen
to be:

- **Computable from raw PCAP/NetFlow** — no proprietary collectors
- **Physically interpretable** — each feature has a clear network meaning
- **Transferable** — independent of protocol mix, link speed, or network topology
- **Minimal** — theoretical minimum required to separate the 7 attack classes

---

## 2. Feature Categories

### 2.1 Packet Statistics (5 features)

Basic properties of individual packets.

| ID | Feature | Definition | Units | Collection Method | Allowed Range |
|----|---------|-----------|-------|-------------------|---------------|
| F01 | mean_pkt_len | Mean packet length in the flow | bytes | PCAP → per-flow aggregate | [0, 65535] |
| F02 | pkt_len_std | Std dev of packet length | bytes | PCAP → per-flow aggregate | [0, 32768] |
| F03 | min_pkt_len | Minimum packet length | bytes | PCAP → per-flow aggregate | [0, 65535] |
| F04 | max_pkt_len | Maximum packet length | bytes | PCAP → per-flow aggregate | [0, 65535] |
| F05 | pkt_count | Total packet count in flow | count | PCAP → per-flow aggregate | [1, ∞) |

### 2.2 Flow Statistics (5 features)

Aggregate properties of the complete bidirectional flow.

| ID | Feature | Definition | Units | Collection Method | Allowed Range |
|----|---------|-----------|-------|-------------------|---------------|
| F06 | duration | Flow duration | seconds | PCAP → last_pkt_ts - first_pkt_ts | [0, ∞) |
| F07 | total_fwd_packets | Packets from source → destination | count | PCAP → forward direction count | [0, ∞) |
| F08 | total_bwd_packets | Packets from destination → source | count | PCAP → backward direction count | [0, ∞) |
| F09 | flow_bytes_per_sec | Throughput = total_bytes / duration | bytes/s | Computed from F05×mean(F01) / F06 | [0, ∞) |
| F10 | flow_packets_per_sec | Packet rate = pkt_count / duration | pkts/s | Computed from F05 / F06 | [0, ∞) |

### 2.3 Temporal Behavior (4 features)

Timing patterns and inter-arrival distributions.

| ID | Feature | Definition | Units | Collection Method | Allowed Range |
|----|---------|-----------|-------|-------------------|---------------|
| F11 | mean_iat | Mean inter-arrival time between packets | microseconds | PCAP → delta(ts) per flow | [0, ∞) |
| F12 | iat_std | Std dev of inter-arrival times | microseconds | PCAP → delta(ts) per flow | [0, ∞) |
| F13 | fwd_iat_mean | Mean IAT for forward direction | microseconds | PCAP → delta(ts) for fwd pkts | [0, ∞) |
| F14 | bwd_iat_mean | Mean IAT for backward direction | microseconds | PCAP → delta(ts) for bwd pkts | [0, ∞) |

### 2.4 Connection Behavior (4 features)

TCP connection and handshake characteristics.

| ID | Feature | Definition | Units | Collection Method | Allowed Range |
|----|---------|-----------|-------|-------------------|---------------|
| F15 | syn_count | Number of SYN packets | count | PCAP → TCP flags analysis | [0, ∞) |
| F16 | fin_count | Number of FIN packets | count | PCAP → TCP flags analysis | [0, ∞) |
| F17 | rst_count | Number of RST packets | count | PCAP → TCP flags analysis | [0, ∞) |
| F18 | conn_state_code | Encoded connection state: 0=(no SYN), 1=(SYN→SYN/ACK→FIN), 2=(SYN→RST), 3=(half-open), 4=(other) | categorical | PCAP → TCP handshake state machine | {0, 1, 2, 3, 4} |

### 2.5 Protocol Behavior (4 features)

Protocol-level statistics derived from payload analysis.

| ID | Feature | Definition | Units | Collection Method | Allowed Range |
|----|---------|-----------|-------|-------------------|---------------|
| F19 | active_payload_bytes | Total payload bytes in active data stream | bytes | PCAP → TCP reassembly payload sum | [0, ∞) |
| F20 | payload_entropy | Shannon entropy of concatenated payload | bits/byte | Computed from payload bytes | [0, 8] |
| F21 | distinct_protocols | Number of distinct application protocols in flow | count | PCAP → DPI or port heuristic | [1, 10] |
| F22 | ttl_min | Minimum TTL value observed | hops | PCAP → IP header TTL | [0, 255] |

---

## 3. Feature Extraction Pipeline

### 3.1 Standardized Extractor

All datasets MUST use the reference extractor at:

```
src/phase36/feature_extractor.py
```

The extractor accepts PCAP files (pcapng format, v2.4+) and produces a CSV with
exactly the 22 features above plus `flow_id`, `src_ip`, `dst_ip`, `src_port`,
`dst_port`, `protocol`, `timestamp`, `label`.

Extraction parameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Flow timeout | 120 s | Long enough for bulk transfer, short enough for attack flows |
| Max flow idle | 60 s | Terminates idle flows |
| Packet truncation | 256 bytes | Captures headers + protocol metadata |
| Direction | source → destination (first packet direction) | Standard bidirectional flow definition |

### 3.2 Validation Checks

Each extracted feature file must pass:

1. **Schema check**: All 22 columns present with correct data types
2. **Range check**: Every value within allowed range (Section 2)
3. **Null check**: < 1% missing values for any feature
4. **Consistency check**: Derived features match their definitions (e.g., `flow_bytes_per_sec` = `total_bytes / duration`)
5. **Timestamp monotonicity**: Flow timestamps are strictly increasing

### 3.3 Normalization

Each dataset is z-score normalized independently **per dataset collection**.
Statistics (mean, std) are computed within the training split only.

No cross-dataset normalization is applied. The benchmark measures transfer learning
performance with the natural distribution differences intact.

---

## 4. Feature Motivation

### Why 22 features?

The Phase 17 canonical set (17 features) was sufficient for in-dataset detection
within the HELIX-IDS project. However, cross-dataset analysis revealed that:

1. **Connection behavior** (F15-F18) is essential for distinguishing DoS from
   reconnaissance — both involve many small packets, but TCP state differs.
2. **Protocol behavior** (F19-F22) distinguishes data exfiltration from benign
   bulk transfer — only exfiltration shows high payload entropy with specific
   TTL signatures.
3. **Temporal behavior** (F11-F14) captures the timing differences between
   automated scanning tools and human-driven attacks.

### Why not more?

The 22 features represent the **minimal sufficient set** — the smallest feature
set that achieves ≥95% of the maximum attainable cross-dataset transfer performance.

- Adding >30 features increases dataset-ID accuracy (undesirable) without
  improving transfer MF1
- Adding temporal window aggregates (e.g., "mean of last 10 packets") adds
  dimensionality without measurable benefit at the flow level
- Raw packet bytes are excluded by design — feature engineering is a standard
  preprocessing step, and raw bytes add unnecessary variance

---

## 5. Legacy Feature Mapping

Datasets from Phase 30-34 (NSL-KDD, UNSW-NB15, CICIDS2018, TON-IoT) with existing
feature sets can be mapped to the canonical set:

| Canonical ID | NSL-KDD Mapping | UNSW-NB15 Mapping | CICIDS2018 Mapping | TON-IoT Mapping |
|-------------|----------------|-------------------|--------------------|--------------------|
| F01 | src_bytes, dst_bytes mean | Spkts_Mean or mean(sload,dload) | Fwd Pkt Len Mean, Bwd Pkt Len Mean | mean_pkt_len |
| F02 | — (not available) | — | Pkt Len Std | pkt_len_std |
| F03 | — | — | Fwd Pkt Len Min, Bwd Pkt Len Min | min_pkt_len |
| F04 | — | — | Fwd Pkt Len Max, Bwd Pkt Len Max | max_pkt_len |
| F05 | count | Spkts (total) | Tot Fwd Pkts, Tot Bwd Pkts | pkt_count |
| F06 | duration | dur | Flow Duration | duration |
| F07 | — | Spkts | Tot Fwd Pkts | total_fwd_packets |
| F08 | — | Dpkts | Tot Bwd Pkts | total_bwd_packets |
| F09 | — | — | Flow Bytes/s | — |
| F10 | — | — | Flow Pkts/s | — |
| F11 | — | — | Flow IAT Mean | mean_iat |
| F12 | — | — | Flow IAT Std | iat_std |
| F13 | — | — | Fwd IAT Mean | fwd_iat_mean |
| F14 | — | — | Bwd IAT Mean | bwd_iat_mean |
| F15 | — | — | SYN Flag Count | syn_count |
| F16 | — | — | FIN Flag Count | fin_count |
| F17 | — | — | RST Flag Count | rst_count |
| F18 | flag (partial) | — | — | conn_state |
| F19 | — | — | Active Mean | active_payload_bytes |
| F20 | — | — | — | — |
| F21 | protocol_type | proto | Protocol | proto |
| F22 | — | — | — | ttl_min |

> **Note:** Legacy datasets have gaps in this mapping. Those gaps are precisely the
> reason a unified collection protocol is necessary (see COLLECTION_PROTOCOL.md).

---

## 6. Versioning

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-06-24 | Initial release — 22 features in 5 categories |

Future revisions may add:
- Encrypted traffic features (TLS handshake metadata)
- Time-series features (packet-level sequence features)
- DNS-specific features for exfiltration detection
