# Dataset Collection Protocol v1.0

> **Phase 36 — Deliverable 3 of 8**
> Standardized procedure for IDS benchmark data collection.
> Every dataset in the benchmark MUST follow this protocol exactly.
> Date: 2026-06-24

---

## 1. Purpose

The four existing IDS benchmarks violated transfer learning assumptions partly because
their collection methodologies were incompatible:

- **NSL-KDD**: 1998 DARPA simulation, synthetic traffic, no real captures
- **UNSW-NB15**: IXIA PerfectStorm tool, proprietary traffic mix
- **CICIDS2018**: AWS testbed, custom B-Profile traffic generation
- **TON-IoT**: IoT/IIoT testbed, telemetry dataset

This protocol defines a **single standardized collection procedure** that eliminates
methodological variance as a confound in cross-dataset transfer research.

---

## 2. Network Topology

### 2.1 Reference Topology

```
                        ┌──────────────┐
                        │  Internet     │
                        │  (BGP peer)   │
                        └──────┬───────┘
                               │ 1 Gbps
                     ┌─────────┴─────────┐
                     │  Border Router     │
                     │  (pfSense 2.7+)   │
                     └─────────┬─────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
    ┌─────┴─────┐     ┌───────┴───────┐    ┌───────┴───────┐
    │   DMZ     │     │ Internal Net  │    │   IOT Net    │
    │  Servers  │     │  (192.168.x)  │    │  (10.0.0.x)  │
    │           │     │               │    │               │
    │ • Web     │     │ • Workstations│    │ • Sensors     │
    │ • DNS     │     │ • Printers    │    │ • Actuators   │
    │ • Mail    │     │ • File server │    │ • Gateway     │
    │ • DB      │     │ • Admin       │    │               │
    └───────────┘     └───────────────┘    └───────────────┘
```

### 2.2 Component Specifications

| Component | Specification | Count |
|-----------|--------------|-------|
| Border router | pfSense 2.7+, 4 NICs, 1 Gbps | 1 |
| Web server | Ubuntu 22.04, Apache 2.4 + Nginx 1.24 | 2 |
| DNS server | Ubuntu 22.04, BIND 9.18 | 1 |
| Mail server | Ubuntu 22.04, Postfix 3.6 + Dovecot 2.3 | 1 |
| Database server | Ubuntu 22.04, PostgreSQL 15 + MySQL 8 | 2 |
| Workstations | Windows 11 Pro (×2), Ubuntu 22.04 Desktop (×2) | 4 |
| File server | FreeBSD 13, Samba 4.17 | 1 |
| Admin workstation | Windows 11, Wireshark, PuTTY, admin tools | 1 |
| IoT sensors | Raspberry Pi 4 running MQTT broker + sensor sim | 4 |
| IoT actuators | ESP32 running Modbus TCP slaves | 4 |
| IoT gateway | Raspberry Pi 4, network bridge + protocol converter | 1 |
| Attack machine | Kali Linux 2024.1, all tool suites installed | 2 |
| Monitoring host | Ubuntu 22.04, tcpdump + Wireshark + Zeek | 1 |

### 2.3 Network Segmentation

Three subnets (VLAN-separated):

| VLAN | Subnet | Purpose | Hosts |
|------|--------|---------|-------|
| 10 | 10.0.10.0/24 | DMZ | Public-facing servers |
| 20 | 192.168.20.0/24 | Internal | Workstations, printers, file servers |
| 30 | 10.0.30.0/24 | IoT | Sensors, actuators, gateway |

Inter-VLAN routing is permitted only through the border router with default-drop rules.

---

## 3. Traffic Sources

### 3.1 Benign Traffic Generation

Benign traffic MUST be generated using the **Phase 36 Traffic Generator**:

```
scripts/phase36/generate_benign.py
```

Traffic mix:

| Traffic Type | % of Total | Description | Tools |
|-------------|-----------|-------------|-------|
| Web browsing | 35% | HTTP/HTTPS browsing with varied user agents | Selenium + Chrome/Firefox |
| Email | 10% | SMTP/IMAP/POP3 client traffic | Thunderbird automated |
| File transfer | 10% | SMB, FTP, SFTP transfers | smbclient, lftp |
| DNS | 10% | Queries to diverse domains | dnsperf, custom clients |
| Database | 10% | PostgreSQL/MySQL queries | pgbench, mysqlslap |
| IoT telemetry | 10% | MQTT publish/subscribe | Eclipse Paho clients |
| Remote access | 5% | SSH, RDP sessions | Paramiko, FreeRDP |
| VoIP | 5% | SIP + RTP calls | PJSIP, sipgrep |
| Software updates | 3% | apt/HTTP downloads | wget, curl automation |
| Background noise | 2% | ARP, NTP, DHCP, ICMP | Background protocols |

### 3.2 Attack Traffic Injection

Attack traffic MUST be generated using the **Phase 36 Attack Injector**:

```
scripts/phase36/inject_attacks.py
```

Attack injection schedule:

| Attack Class | Tools | Duration per Run | Instances per Run | Timing |
|-------------|-------|-----------------|-------------------|--------|
| Reconnaissance | nmap, masscan, zmap, nikto, gobuster | 5-30 minutes | 3-5 | Interleaved with benign |
| Denial of Service | hping3, slowloris, GoldenEye, HOIC, LOIC, metasploit | 2-15 minutes | 2-3 | Target: DMZ servers |
| Initial Access | hydra, john, sqlmap, metasploit, searchsploit | 10-60 minutes | 3-5 | Target: all subnets |
| Privilege Escalation | linpeas, peass-ng, CVE exploits | 5-30 minutes | 2-3 | Post-exploitation only |
| Lateral Movement | crackmapexec, impacket, psexec, ssh pivoting | 10-45 minutes | 2-3 | Internal net only |
| Exfiltration | nc, scp, dns-tunneling, icmp-tunneling | 5-20 minutes | 2 | Target: internal to internet |

---

## 4. Capture Configuration

### 4.1 Capture Points

Traffic is captured at four vantage points simultaneously:

| Capture ID | Location | Interface | Data Volume/day (est.) |
|-----------|----------|-----------|----------------------|
| C1 | Border router — WAN side | eth0 (mirror port) | ~50 GB |
| C2 | DMZ switch — SPAN port | eth1 | ~30 GB |
| C3 | Internal switch — SPAN port | eth2 | ~40 GB |
| C4 | IoT gateway — inline tap | eth3 | ~10 GB |

### 4.2 Capture Tool Configuration

```
Tool: tcpdump v4.99+
Flags: -i <interface> -s 256 -w <output_file>.pcap -C 1024 -W 100
Filter: none (full packet capture, truncated to 256 bytes)
Rotation: every 1024 MB or 60 minutes, whichever comes first
Storage: RAID-10 NVMe array, minimum 10 TB usable
```

### 4.3 Capture Duration

| Phase | Duration | Benign:Attack Ratio |
|-------|----------|--------------------|
| Calibration (no attacks) | 24 hours | 100:0 |
| Week 1 — Reconnaissance | 7 days | 85:15 |
| Week 2 — DoS | 7 days | 80:20 |
| Week 3 — Initial Access | 7 days | 85:15 |
| Week 4 — Privilege Escalation | 7 days | 90:10 |
| Week 5 — Lateral Movement | 7 days | 85:15 |
| Week 6 — Exfiltration | 7 days | 85:15 |
| Week 7 — Mixed (all classes) | 7 days | 75:25 |
| **Total** | **50 days** | **~85:15 average** |

### 4.4 Repetitions

The full 50-day capture cycle MUST be repeated at least **3 times** in the same
testbed, with the same topology, to establish variance bounds.

Each repetition may vary:
- Attack start times (randomized within daily windows)
- Attack targets (randomized across available hosts)
- Benign user behavior seeds (randomized browsing patterns)

---

## 5. Labeling Methodology

### 5.1 Ground Truth Collection

Label provenance is the most critical quality requirement.

| Attack Ground Truth | Method | Timestamp Precision |
|--------------------|--------|---------------------|
| Attack start/end | Scripted injector logs timestamps ±100ms | ±1 second |
| Attack parameters | JSON log of all injection parameters | Per-run |
| Target IP:Port | Logged in injector manifest | Exact |
| Attack class | Level-1 + Level-2 ID from ontology | Exact |

Ground truth is stored in structured format:

```json
{
  "attack_run_id": "R01-RECON-003",
  "attack_class": "Reconnaissance",
  "attack_subclass": "nmap-syn-scan",
  "target_hosts": ["192.168.20.12", "192.168.20.13"],
  "source_host": "10.0.10.100",
  "start_time": "2026-07-15T14:30:00.000Z",
  "end_time": "2026-07-15T15:15:00.000Z",
  "tool": "nmap 7.94",
  "parameters": {"scan_type": "SYN", "ports": "1-10000", "flags": "-sS -T4"}
}
```

### 5.2 Flow Labeling

Flows are labeled using exact timestamp matching:

```
For each flow F with [first_pkt_ts, last_pkt_ts]:
  For each attack run R with [start_time, end_time]:
    If F overlaps R by ≥ 0.5 seconds:
      Label F with R.attack_class
  If no attack run matches:
    Label F as Benign
```

Edge cases:
- **Partial overlap**: If a flow overlaps multiple attack runs, it inherits the
  majority-overlap class. If no majority, the earliest attack class prevails.
- **Mixed flows**: If a single flow contains both attack and benign packets (rare),
  the entire flow inherits the attack label.
- **Benign periods**: Flows during no-attack windows are unconditionally Benign.

### 5.3 Quality Assurance

| Check | Threshold | Action |
|-------|-----------|--------|
| Label consistency (inter-rater) | Cohen's κ ≥ 0.95 | Re-label if below |
| Temporal precision | ±1 second of injector log | Adjust injector clock |
| Label balance (per capture week) | Each attack class ≥ 2% of total | Extend capture for under-represented classes |
| Benign label contamination | < 0.1% attack flows mislabeled as benign | Manual audit of flagged flows |

---

## 6. Dataset Packaging

### 6.1 Release Artifacts

Each collection produces:

```
phase36_v<version>_<collection_run>/
├── raw_pcap/
│   ├── week1_c1_001.pcap
│   ├── week1_c2_001.pcap
│   └── ...
├── flows/
│   ├── week1_flows.csv.gz
│   └── ...
├── features/
│   ├── week1_features.csv.gz
│   └── ...
├── labels/
│   ├── ground_truth.json
│   ├── label_manifest.csv
│   └── flow_labels.csv
├── metadata/
│   ├── topology.yaml
│   ├── inventory.yaml
│   ├── capture_config.yaml
│   ├── attack_manifest.json
│   └── collection_log.txt
└── checksums/
    └── sha256sums.txt
```

### 6.2 Required Metadata

Each collection must document:

1. Testbed hardware specifications (CPU, RAM, NIC models, switch models)
2. Software versions (OS kernel, libraries, all tools)
3. Network baseline (average throughput, latency, packet loss)
4. Timestamp offset between capture hosts (NTP synchronization status)
5. Known anomalies or capture interruptions
6. Any deviations from this protocol

---

## 7. Reproducibility

The entire testbed is defined as Infrastructure-as-Code:

```
rpository: https://github.com/helix-ids/phase36-testbed
terraform/   — Network topology (virtual or physical switch configs)
ansible/     — Host provisioning and configuration
docker/      — Containerized service definitions
```

Any researcher with sufficient hardware can reproduce the exact testbed configuration
from these definitions. Virtualized testbed (VirtualBox/Vagrant) configuration is also
provided for small-scale validation.

---

## 8. Validation

Before any dataset release, the collection must be validated:

1. **Protocol compliance**: All 22 canonical features extractable from raw PCAP
2. **Label mapping**: All attack injections map to exactly one Level-1 class
3. **Class coverage**: All 7 Level-1 classes present in the labeled output
4. **Temporal continuity**: No gaps > 1 second in capture coverage
5. **Packet truncation verification**: Verify 256-byte truncation doesn't lose attack signatures
6. **Flow completion**: Verify flow timeout settings capture complete attack flows

---

## 9. Versioning

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-06-24 | Initial release — full 50-day capture protocol with 4 vantage points |
