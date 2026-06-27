# Attack Family Ontology Mapping

Unified attack ontology across all 4 IDS benchmark datasets.

## Canonical 7-Class Taxonomy

| Index | Family | Threat Severity |
|:-----:|--------|:---------------:|
| 0 | **Normal** | Benign |
| 1 | **DoS** (Denial of Service) | High |
| 2 | **Probe** (Reconnaissance) | Medium |
| 3 | **R2L** (Remote-to-Local) | High |
| 4 | **U2R** (User-to-Root) | Critical |
| 5 | **Generic** | Medium |
| 6 | **Backdoor** | Critical |

## Class Presence Matrix

✓ = class present in dataset, ✗ = class absent

| Class | NSL-KDD | UNSW-NB15 | CICIDS2018 | TON-IoT |
|------|---|---|---|---|
| **Normal** ✓  ✓  ✓  ✓ |
| **DoS** ✓  ✓  ✓  ✓ |
| **Probe** ✓  ✓  ✓  ✓ |
| **R2L** ✓  ✓  ✓  ✓ |
| **U2R** ✓  ✓  ✗  ✗ |
| **Generic** ✗  ✓  ✓  ✗ |
| **Backdoor** ✗  ✓  ✗  ✓ |

## Shared Classes Across All Datasets

Only ['Normal', 'DoS', 'Probe', 'R2L'] are present in ALL 4 datasets.

Classes **U2R**, **Generic**, and **Backdoor** are dataset-specific:
- U2R: Present only in NSL-KDD, UNSW-NB15, CICIDS2018 (absent from TON-IoT)
- Generic: Present only in UNSW-NB15 and CICIDS2018
- Backdoor: Present only in UNSW-NB15 and TON-IoT

## Pairwise Jaccard Overlap

| Source → Target | Jaccard Index |
|----------------|:-------------:|
| CICIDS2018 → CICIDS2018 | 1.0 |
| CICIDS2018 → NSL-KDD | 0.6667 |
| CICIDS2018 → TON-IoT | 0.6667 |
| CICIDS2018 → UNSW-NB15 | 0.7143 |
| NSL-KDD → CICIDS2018 | 0.6667 |
| NSL-KDD → NSL-KDD | 1.0 |
| NSL-KDD → TON-IoT | 0.6667 |
| NSL-KDD → UNSW-NB15 | 0.7143 |
| TON-IoT → CICIDS2018 | 0.6667 |
| TON-IoT → NSL-KDD | 0.6667 |
| TON-IoT → TON-IoT | 1.0 |
| TON-IoT → UNSW-NB15 | 0.7143 |
| UNSW-NB15 → CICIDS2018 | 0.7143 |
| UNSW-NB15 → NSL-KDD | 0.7143 |
| UNSW-NB15 → TON-IoT | 0.7143 |
| UNSW-NB15 → UNSW-NB15 | 1.0 |

## Raw Attack Label Overlap (Per Family)

### Normal
- **NSL-KDD**: normal
- **UNSW-NB15**: normal
- **CICIDS2018**: benign
- **TON-IoT**: normal
- **Shared attack names**: None — attack names are dataset-specific

### DoS
- **NSL-KDD**: dos
- **UNSW-NB15**: dos
- **CICIDS2018**: ddos, ddos attack-hoic, ddos attack-loic-udp, ddos attacks-loic-http, dos, dos attacks-goldeneye, dos attacks-hulk, dos attacks-slowhttptest, dos attacks-slowloris
- **TON-IoT**: ddos, dos
- **Shared attack names**: dos

### Probe
- **NSL-KDD**: probe
- **UNSW-NB15**: analysis, fuzzers, reconnaissance
- **CICIDS2018**: portscan
- **TON-IoT**: scanning
- **Shared attack names**: None — attack names are dataset-specific

### R2L
- **NSL-KDD**: r2l
- **UNSW-NB15**: exploits
- **CICIDS2018**: brute force, brute force -web, brute force -xss, ftp-bruteforce, ftp-patator, sql injection, ssh-bruteforce, ssh-patator
- **TON-IoT**: injection, password, xss
- **Shared attack names**: None — attack names are dataset-specific

### U2R
- **NSL-KDD**: u2r
- **UNSW-NB15**: shellcode
- **CICIDS2018**: (none)
- **TON-IoT**: (none)
- **Shared attack names**: None — attack names are dataset-specific

### Generic
- **NSL-KDD**: (none)
- **UNSW-NB15**: generic
- **CICIDS2018**: bot, infilteration, infiltration
- **TON-IoT**: (none)
- **Shared attack names**: None — attack names are dataset-specific

### Backdoor
- **NSL-KDD**: (none)
- **UNSW-NB15**: backdoor, backdoors, worms
- **CICIDS2018**: (none)
- **TON-IoT**: backdoor, mitm, ransomware
- **Shared attack names**: backdoor
