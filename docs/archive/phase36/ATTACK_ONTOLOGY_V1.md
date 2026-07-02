# Unified IDS Attack Ontology v1.0

> **Phase 36 — Deliverable 1 of 8**
> Canonical attack hierarchy for cross-dataset IDS research.
> Defines the label space required for transfer learning.
> Date: 2026-06-24

---

## 1. Purpose

A unified attack ontology is the prerequisite for meaningful cross-dataset transfer
learning. Without a shared label space, transfer learning reduces to dataset identification
(detect which dataset a sample came from), not generalization across attack behaviors.

This ontology defines a **7-class Level-1 hierarchy** and maps every attack label from
four public IDS benchmark datasets onto it. Datasets that cannot be mapped onto this
ontology are excluded from the benchmark, ensuring **label-space overlap** — one of the
four assumptions that Phase 33 proved was violated.

---

## 2. Level 1: Unified Attack Hierarchy

| ID | Class | Description |
|----|-------|-------------|
| 0 | **Benign** | Normal, non-malicious network traffic. No attack behavior. |
| 1 | **Reconnaissance** | Information gathering: port scans, ping sweeps, service enumeration, vulnerability probing. Goal is to map the target network, not compromise it. |
| 2 | **Denial of Service** | Resource exhaustion attacks that degrade, disrupt, or deny legitimate access to services, bandwidth, or computational resources. |
| 3 | **Initial Access** | Techniques that achieve a foothold in the target network: exploitation of software vulnerabilities, credential theft, injection attacks, brute force. Entry point for further compromise. |
| 4 | **Privilege Escalation** | Post-exploitation techniques that obtain higher privileges: local privilege escalation, rootkit installation, U2R attacks. |
| 5 | **Lateral Movement** | Movement through the network from the initial foothold to other hosts: remote service exploitation, internal scanning, pass-the-hash, SMB exploitation. |
| 6 | **Exfiltration** | Unauthorized data extraction: data theft, backdoor triggered outbound transfer, covert channels, command and control (C2) beaconing. |

### Design Rationale

- **7 classes balanced between expressiveness and learnability.** Fewer than 5 classes
  collapse too much behavioral diversity; more than 10 create sparse training signals
  for rare attacks.
- **Mirrors MITRE ATT&CK Enterprise v13 at the tactic level**, ensuring compatibility
  with operational cybersecurity frameworks.
- **Every attack type from the source datasets maps to exactly one Level-1 class**,
  eliminating the ambiguous or "other" categories that plague existing benchmarks.

---

## 3. Level 2: Dataset-to-Ontology Mapping

### 3.1 NSL-KDD

| Original Label | Level-2 ID | Level-1 Class | Notes |
|---------------|-----------|---------------|-------|
| normal | 0 | Benign | — |
| back | DoS-back | Denial of Service | Apache/PHP back — HTTP DoS |
| land | DoS-land | Denial of Service | Land attack — TCP SYN spoof |
| neptune | DoS-neptune | Denial of Service | SYN flood |
| pod | DoS-pod | Denial of Service | Ping of death |
| smurf | DoS-smurf | Denial of Service | ICMP smurf amplification |
| teardrop | DoS-teardrop | Denial of Service | Fragmented packet overlap |
| apache2 | DoS-apache2 | Denial of Service | Apache HTTP DoS (test set) |
| mailbomb | DoS-mailbomb | Denial of Service | Mail server flood (test set) |
| processtable | DoS-processtable | Denial of Service | Process table exhaustion (test set) |
| udpstorm | DoS-udpstorm | Denial of Service | UDP storm amplification (test set) |
| ipsweep | Recon-ipsweep | Reconnaissance | IP sweep scan |
| nmap | Recon-nmap | Reconnaissance | Nmap port scan |
| portsweep | Recon-portsweep | Reconnaissance | Port sweep |
| satan | Recon-satan | Reconnaissance | SATAN vulnerability scanner |
| mscan | Recon-mscan | Reconnaissance | Mscan scanner (test set) |
| saint | Recon-saint | Reconnaissance | Saint scanner (test set) |
| ftp_write | IA-ftp_write | Initial Access | FTP anonymous write + rsh login |
| guess_passwd | IA-guess_passwd | Initial Access | Password guessing |
| imap | IA-imap | Initial Access | IMAP buffer overflow exploit |
| multihop | IA-multihop | Initial Access | Multi-hop login (rlogin chain) |
| phf | IA-phf | Initial Access | PHP CGI buffer overflow |
| spy | IA-spy | Initial Access | Spy — covert channel monitoring |
| warezclient | IA-warezclient | Initial Access | Warez client — unauthorized download via FTP |
| warezmaster | IA-warezmaster | Initial Access | Warez master — unauthorized upload via FTP |
| buffer_overflow | PE-buffer_overflow | Privilege Escalation | Local buffer overflow |
| loadmodule | PE-loadmodule | Privilege Escalation | Loadmodule — privilege escalation via LD_PRELOAD |
| perl | PE-perl | Privilege Escalation | Perl privilege escalation |
| rootkit | PE-rootkit | Privilege Escalation | Rootkit installation |
| xterm | PE-xterm | Privilege Escalation | Xterm privilege escalation (test set) |
| ps | PE-ps | Privilege Escalation | Ps — process table manipulation (test set) |
| sqlattack | PE-sqlattack | Privilege Escalation | SQL server buffer overflow (test set) |
| snmpgetattack | IA-snmpgetattack | Initial Access | SNMP community string guessing (test set) |
| named | IA-named | Initial Access | DNS named buffer overflow (test set) |
| sendmail | IA-sendmail | Initial Access | Sendmail buffer overflow (test set) |
| worm | LM-worm | Lateral Movement | Worm propagation (test set) |
| xlock | LM-xlock | Lateral Movement | Xlock screen lock + password steal (test set) |
| xsnoop | LM-xsnoop | Lateral Movement | Xsnoop — X11 session hijack (test set) |

### 3.2 UNSW-NB15

| Original Label | Level-2 ID | Level-1 Class | Notes |
|---------------|-----------|---------------|-------|
| Normal | 0 | Benign | — |
| Analysis | IA-analysis | Initial Access | HTML/script-based attacks: port scan tool traffic, spam probe |
| Backdoor | IA-backdoor | Initial Access | Unauthorized remote access trojans |
| DoS | DoS-unsw | Denial of Service | Various DoS attacks |
| Exploits | IA-exploits | Initial Access | Direct exploitation of software vulnerabilities |
| Fuzzers | Recon-fuzzers | Reconnaissance | Automated input fuzzing for crash detection |
| Generic | DoS-generic | Denial of Service | Cryptographic hash collision attacks |
| Reconnaissance | Recon-unsw | Reconnaissance | Network probing and scanning |
| Shellcode | IA-shellcode | Initial Access | Malicious code injection |
| Worms | LM-worms | Lateral Movement | Self-propagating malware |

### 3.3 CICIDS2018

| Original Label | Level-2 ID | Level-1 Class | Notes |
|---------------|-----------|---------------|-------|
| Benign | 0 | Benign | — |
| Bot | LM-bot | Lateral Movement | Botnet C2 communication |
| Brute Force — FTP | IA-brute_ftp | Initial Access | FTP credential brute force |
| Brute Force — SSH | IA-brute_ssh | Initial Access | SSH credential brute force |
| DoS attacks — GoldenEye | DoS-goldeneye | Denial of Service | HTTP GoldenEye DoS |
| DoS attacks — Hulk | DoS-hulk | Denial of Service | HTTP Hulk DoS |
| DoS attacks — SlowHTTPTest | Dos-slowhttp | Denial of Service | Slow HTTP attack |
| DoS attacks — Slowloris | DoS-slowloris | Denial of Service | Slowloris attack |
| DDOS attack — HOIC | DoS-hoic | Denial of Service | HOIC DDoS |
| DDOS attack — LOIC | DoS-loic | Denial of Service | LOIC DDoS |
| Heartbleed | IA-heartbleed | Initial Access | OpenSSL Heartbleed exploit |
| Infiltration | LM-infiltration | Lateral Movement | Internal network pivot |
| SQL Injection | IA-sqli | Initial Access | SQL injection attack |
| SSH-BruteForce | IA-brute_ssh2 | Initial Access | SSH brute force (alternate label) |

### 3.4 TON-IoT

| Original Label | Level-2 ID | Level-1 Class | Notes |
|---------------|-----------|---------------|-------|
| normal | 0 | Benign | — |
| backdoor | IA-ton_backdoor | Initial Access | Backdoor trojan |
| ddos | DoS-ton_ddos | Denial of Service | Distributed Denial of Service |
| dos | DoS-ton_dos | Denial of Service | Direct Denial of Service |
| injection | IA-ton_injection | Initial Access | SQL / command injection |
| mitm | LM-ton_mitm | Lateral Movement | Man-in-the-middle attack |
| password | IA-ton_password | Initial Access | Password cracking / brute force |
| ransomware | Exf-ton_ransomware | Exfiltration | Ransomware file encryption + exfiltration |
| scanning | Recon-ton_scanning | Reconnaissance | Network scanning / probing |
| xss | IA-ton_xss | Initial Access | Cross-site scripting |
| **Bruteforce** | IA-ton_bruteforce | Initial Access | Brute force via protocols |
| **DoS** | DoS-ton_dos2 | Denial of Service | Additional DoS type |
| **DDoS** | DoS-ton_ddos2 | Denial of Service | Additional DDoS type |

> **Note:** TON-IoT labels marked **bold** use the exact same name as others but represent
> different attack implementations. The Level-2 ID disambiguates them.

---

## 4. Cross-Dataset Label Distribution Summary

| Level-1 Class | NSL-KDD | UNSW-NB15 | CICIDS2018 | TON-IoT |
|--------------|:-------:|:---------:|:----------:|:-------:|
| Benign | ✓ | ✓ | ✓ | ✓ |
| Reconnaissance | ✓ | ✓ | ✗ | ✓ |
| Denial of Service | ✓ | ✓ | ✓ | ✓ |
| Initial Access | ✓ | ✓ | ✓ | ✓ |
| Privilege Escalation | ✓ | ✗ | ✗ | ✗ |
| Lateral Movement | ✓ | ✓ | ✓ | ✓ |
| Exfiltration | ✗ | ✗ | ✗ | ✓ |

### Observations

1. **Privilege Escalation** appears only in NSL-KDD. No other dataset captures this
   attack category explicitly. This is a known gap; future collection efforts must
   include privilege escalation scenarios.

2. **Exfiltration** appears only in TON-IoT (ransomware). This reflects the IoT focus
   of TON-IoT, where ransomware is a primary threat. Future collections should include
   data exfiltration across all tiers.

3. **Reconnaissance** is absent from CICIDS2018 as a labeled category, though
   scanning activity undoubtedly occurred in the testbed. This is a labeling artifact.

4. **No dataset covers all 7 classes.** Only by combining all four datasets does
   the full hierarchy emerge. This is the fundamental motivation for the unified
   benchmark.

---

## 5. Machine-Readable Format

The ontology is available in machine-readable JSON format at:

```
benchmarks/phase36/attack_ontology_v1.json
```

Schema:

```json
{
  "ontology_version": "1.0.0",
  "classes": [
    {"id": 0, "name": "Benign", "description": "..."},
    {"id": 1, "name": "Reconnaissance", "description": "..."},
    {"id": 2, "name": "Denial of Service", "description": "..."},
    {"id": 3, "name": "Initial Access", "description": "..."},
    {"id": 4, "name": "Privilege Escalation", "description": "..."},
    {"id": 5, "name": "Lateral Movement", "description": "..."},
    {"id": 6, "name": "Exfiltration", "description": "..."}
  ],
  "dataset_mappings": {
    "NSL_KDD": [
      {"original_label": "neptune", "level_2_id": "DoS-neptune", "level_1": 2}
    ],
    ...
  }
}
```

---

## 6. Versioning

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-06-24 | Initial release — 7-level hierarchy, 4 dataset mappings |

Future versions may add:
- Additional dataset mappings (CICIDS2017, CSE-CIC-IDS2019)
- Sub-class refinement for uneven class distributions
- Temporal evolution of attack signatures

---

## 7. Mapping Rules and Exclusions

### Inclusion Criteria

A dataset label is included in the ontology if:

1. The label describes a reproducible attack behavior (not a statistical artifact)
2. The attack can be generated in a controlled testbed
3. The label has a clear mapping to exactly one Level-1 class

### Exclusion Criteria

The following are excluded:

- **Statistical-only labels** (e.g., "background", "unknown") with no defined attack behavior
- **Highly dataset-specific labels** that cannot be reproduced outside their original environment
- **Merged categories** ("attacks" generic) that collapse multiple distinct behaviors
- **Flow-level aggregates** where individual attack behavior is not identifiable

---

## 8. References

1. MITRE ATT&CK Enterprise v13 — Tactic definitions
2. NSL-KDD Dataset Documentation — Tavallaee et al. (2009)
3. UNSW-NB15 Dataset — Moustafa & Slay (2015)
4. CICIDS2018 Dataset — Sharafaldin et al. (2018)
5. TON-IoT Dataset — Moustafa et al. (2020)
6. Phase 33 — Incompatibility Analysis (docs/phase33/INCOMPATIBILITY_PROOF.md)
7. Phase 34 — Transfer Ceiling Certification (docs/releases/PHASE34_TRANSFER_CEILING_CERTIFICATION.md)
