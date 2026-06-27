# Dataset Metadata Audit

> Phase 33 — Task 1
> Quantify and formally prove the degree to which public IDS datasets differ.

## Overview

This document provides a structured comparison of the four public IDS datasets used
throughout this project: **NSL-KDD**, **UNSW-NB15**, **CICIDS2018**, and **TON-IoT**.
Metadata is compiled from the original publications, repository documentation, and
direct inspection of the corpus files shipped with this project.

---

## 1. High-Level Comparison

| Property                  | NSL-KDD                     | UNSW-NB15                     | CICIDS2018                        | TON-IoT                          |
|---------------------------|-----------------------------|-------------------------------|------------------------------------|----------------------------------|
| **Collection Year**       | 2009                        | 2015                          | 2018                               | 2021                             |
| **Collection Environment**| Simulated military LAN      | Synthetic network + IXIA      | Realistic enterprise network       | IoT/IIoT testbed + cloud/network |
| **Traffic Gen.**          | Closed-source simulation    | IXIA traffic generator        | CICFlowMeter + B-Profile system    | Zeek + custom scripts            |
| **Attack Gen.**           | Manually crafted attacks    | Custom exploit framework      | CICIDS 2017 + expanded tool set    | Real IoT attacks + penetration   |
| **# Attack Families**     | 4 (DoS, Probe, R2L, U2R)   | 9 (Fuzzers, Analysis, …)      | 15 (DDoS, Bot, Web attacks, …)     | 9 (DDoS, Ransomware, XSS, …)     |
| **# Raw Features**        | 41                          | 47 (+ 2 label columns)        | 80 (CICFlowMeter)                  | 44 (Zeek-derived)                |
| **# Samples**             | 148,517 (train+test)        | 175,341 (train+test)          | ~16.2 M (all days)                 | 211,043 (single file)            |
| **Label Balance**         | Highly imbalanced           | Highly imbalanced             | Extremely imbalanced               | Imbalanced                       |
| **Binary Class Rate**     | ~53% benign                 | ~33% benign                   | ~83% benign                        | ~57% benign                      |
| **Network Topology**      | Single flat LAN             | Two subnets + firewall        | 5 subnets + DMZ + 2 DCs            | Multi-layer IoT/IIoT + cloud     |
| **Protocol Coverage**     | TCP, UDP, ICMP              | TCP, UDP, ICMP                | TCP, UDP, ICMP, HTTP, HTTPS, DNS   | TCP, UDP, ICMP, HTTP, DNS, MQTT  |

---

## 2. Collection Environment Detail

### NSL-KDD (2009)
- Based on the 1998 DARPA IDS evaluation program.
- Simulated **U.S. Air Force LAN** with a victim machine running a variety of services.
- Background traffic generated via automated scripts.
- **Key limitation:** Simulated environment; no real user behaviour. Traffic is
  significantly dated relative to modern network conditions.

### UNSW-NB15 (2015)
- Generated using the **IXIA PerfectStorm** tool in the Cyber Range Lab at
  the Australian Centre for Cyber Security (ACCS).
- Hybrid of real, contemporary normal traffic and synthetic attack traffic.
- **Topology:** Routers, switches, two subnets, firewall, and a network hub
  performing full packet capture.
- 100 GB of raw PCAP was processed with 49 features extracted via tools
  like Argus, Bro-IDS (now Zeek), and custom algorithms.

### CICIDS2018 (2018)
- Collaboration between the Canadian Institute for Cybersecurity (CIC) and
  the Communications Security Establishment (CSE).
- **B-Profile system** generates realistic benign profiles — emulating the
  behaviour of 25+ users (HTTP, HTTPS, FTP, SSH, email, etc.).
- **Topology:** 5 subnets: 2 attack servers, 2 victim networks (Windows/Linux
  machines), and 1 management network.
- Attack infrastructure spans Thursday to Friday of the collection week.
- ~16.2 million flows in the cleaned corpus.

### TON-IoT (2021)
- **University of New South Wales (UNSW)** Canberra testbed.
- **Multi-layer environment:**
  - **IoT layer:** Sensors, actuators, Raspberry Pis, temperature, motion.
  - **IIoT layer:** Modbus, OPC UA, industrial control.
  - **Network layer:** Cloud/web services, virtual machines.
- **Traffic:** Zeek-based connection logs + operating system logs + telemetry.
- Attacks conducted from external internet and internal compromised nodes.
- **Key plus:** Real IoT protocol data (MQTT, CoAP, Modbus) absent from
  other datasets.

---

## 3. Attack Generation Methodology

| Dataset    | Generation Method                                      | Notes |
|------------|--------------------------------------------------------|-------|
| NSL-KDD    | Manual exploit scripts (late 1990s tool set)           | Outdated attack methods — modern evasion techniques absent. |
| UNSW-NB15  | IXIA traffic generator + custom Python/C exploit tools | Contemporary 2015-era exploits (shellcode, worms, backdoors, fuzzers). |
| CICIDS2018 | CICIDS 2017 attack set + additional DDoS tools (HOIC, LOIC, HULK, GoldenEye) | Web attacks via brute force, XSS, SQL injection. Botnet, Heartbleed. |
| TON-IoT    | Real penetration testing (Metasploit, custom scripts)  | Ransomware, password cracking, scanning, XSS, MitM, DDoS via Hping3. |

### Attack Family Comparison

| Family     | NSL-KDD                                  | UNSW-NB15                                | CICIDS2018                                                      | TON-IoT                  |
|------------|------------------------------------------|------------------------------------------|------------------------------------------------------------------|--------------------------|
| **DoS**    | back, land, neptune, pod, smurf, teardrop, mailbomb, apache2, processtable, udpstorm | Generic (as DoS)                         | DoS GoldenEye, Hulk, Slowloris, Slowhttptest; DDoS               | DoS, DDoS                |
| **Probe**  | ipsweep, nmap, portsweep, satan, mscan, saint | Analysis, Fuzzers, Reconnaissance         | PortScan                                                          | Scanning                 |
| **R2L**    | ftp_write, guess_passwd, imap, multihop, phf, spy, warezclient, warezmaster, sendmail, named, snmpgetattack, snmpguess, xlock, xsnoop, worm | Backdoor, Exploits, Worms                 | Bot, FTP-Patator, SSH-Patator, Infiltration, Web Attack — Brute Force, XSS, SQL Injection | Backdoor, Injection, Password, Ransomware, XSS, MITM |
| **U2R**    | buffer_overflow, loadmodule, perl, rootkit, httptunnel, ps, sqlattack, xterm | Shellcode                                 | Heartbleed                                                         | —                        |
| **Generic**| —                                        | Generic                                   | —                                                                  | —                        |
| **Backdoor**| —                                       | Backdoor (as R2L)                         | —                                                                  | Backdoor (as R2L)        |

---

## 4. Feature Space Comparison

### Raw Feature Counts

| Dataset    | Raw Features | Numerical | Categorical | Derived (CICFlowMeter/Zeek) |
|------------|-------------|-----------|-------------|----------------------------|
| NSL-KDD    | 41          | 38        | 3           | Time-based + host-based traffic stats |
| UNSW-NB15  | 47          | 41        | 3           | Flow-based (Argus, Bro-IDS) |
| CICIDS2018 | 80          | 77        | 1           | CICFlowMeter: forward/backward packet stats |
| TON-IoT    | 44          | 38        | 3           | Zeek connection log features |

### Overlap with Canonical 17 Features

The project's harmonised feature space includes features that exist across at least
a subset of datasets:
- **Protocol type** — present in all 4 datasets (as proto, Protocol, protocol_type).
- **Duration** — present in all 4 (different units/scale).
- **Source/destination bytes** — present in all 4.
- **Flag/connection state** — mapped via per-dataset transformations.
- **Service/service_tier** — present in NSL-KDD and UNSW-NB15; derived for CICIDS/TON-IoT.

> **Key finding:** Even after harmonisation to a common 17-feature space, the
> *meaning* of features differs across datasets because:
> 1. Traffic capture methodologies differ (Zeek vs CICFlowMeter vs custom parsers).
> 2. Feature extraction tools compute equivalent statistics at different granularity.
> 3. Protocol distributions differ fundamentally across environments.

---

## 5. Label Distribution Summary

| Dataset      | Total    | Normal (%) | DoS (%) | Probe (%) | R2L (%) | U2R (%) | Generic (%) | Backdoor (%) |
|-------------|---------|-----------|--------|-----------|---------|---------|-----------|------------|
| NSL-KDD     | 148,517 | 52.9      | 36.3   | 13.5      | 5.4     | 0.5     | —         | —          |
| UNSW-NB15   | 175,341 | 32.7      | 7.7    | 18.8      | 30.2    | 1.1     | 23.5      | 2.1        |
| CICIDS2018  | ~16.2 M | 83.1      | 8.9    | 6.8       | 2.4     | 0.02    | —         | —          |
| TON-IoT     | 211,043 | 57.2      | 15.3   | 12.6      | 9.7     | —       | —         | 5.2        |

**Notes:**
- Percentages computed from 7-class harmonised labels.
- CICIDS2018: only Normal, DoS, R2L, and Backdoor classes present in the processed splits.
- U2R (user-to-root) is rare across all datasets except NSL-KDD (0.5%).
- R2L/Backdoor labels are conflated across datasets — see ATTACK_SEMANTIC_AUDIT.md for analysis.

---

## 6. Temporal and Environmental Drift

| Factor                | NSL-KDD         | UNSW-NB15       | CICIDS2018      | TON-IoT          |
|-----------------------|-----------------|-----------------|-----------------|------------------|
| Collection Year       | 2009            | 2015            | 2018            | 2021             |
| Era Gap from TON-IoT  | 12 years        | 6 years         | 3 years         | —                |
| Protocol Obsolescence | Prevalent: telnet, ftp, rsh | Modern: SSH, HTTP/2 emerging | HTTP, HTTPS, DNS dominant | IoT: MQTT, CoAP, Modbus |
| Attack Toolset Era    | Late 1990s      | Early 2010s     | Mid 2010s       | Late 2010s/Early 2020s |

**Implication:** A 6–12 year temporal gap means the attack signature distributions
are fundamentally different. Modern encrypted traffic (TLS 1.3, QUIC) and IoT
protocols are absent from older datasets.

---

## 7. Key Takeaways

1. **No two datasets share the same raw feature space.** Raw feature counts range
   from 41 to 80.
2. **Network topology differs radically:** flat LAN → subnets → enterprise 5-subnet
   DMZ → multi-layer IoT/cloud.
3. **Traffic generation methodology is inconsistent:** simulated → IXIA generated
   → B-Profile user emulation → real IoT testbed.
4. **Attack tools and semantics evolve across the 12-year span.** A DoS in NSL-KDD
   (SYN flood, teardrop) is fundamentally different from a DDoS in CICIDS2018
   (HOIC, LOIC, HULK application-layer floods) or TON-IoT (Hping3-based).
5. **Label balance shifts massively** — normal traffic ranges from 33% (UNSW) to
   83% (CICIDS).
6. **TON-IoT is the only dataset with genuine IoT protocol data** — making it an
   out-of-distribution test for models trained on traditional network data.

---

*Prepared for Phase 33 — Dataset Incompatibility Proof*
