# Attack Semantic Mapping — Taxonomy Audit

**Phase 33 — Dataset Incompatibility Proof**
**Created:** 2026-06-24

---

## Overview

Each IDS dataset uses its own attack naming convention and family taxonomy. Even when labels are mapped to a common 5-class schema (Normal, DoS, Probe, R2L, U2R), the underlying attack **behaviors** mapped to the same label differ substantially. This document quantifies the semantic mismatch.

## Methodology

We use the canonical mapping from each dataset's native attack names to the unified 5-class system. Per dataset family, we compute:

- **Jaccard Overlap**: Shared attack type names vs total unique attack type names per family
- **Family Coverage**: Which families are present in which dataset
- **Overall Semantic Overlap**: Mean per-family Jaccard across all families

## Family Coverage

| Family | NSL-KDD | UNSW-NB15 | CICIDS2018 | TON-IoT |
|---|---|---|---|---|
| Normal | ✓ | ✓ | ✓ | ✓ |
| DoS | ✓ | ✓ | ✓ | ✓ |
| Probe | ✓ | ✓ | ✓ | ✓ |
| R2L | ✓ | ✓ | ✓ | ✓ |
| U2R | ✓ | ✓ | ✗ | ✗ |

**UC4 (U2R) missing from CICIDS2018 and TON-IoT.** This means any cross-dataset transfer involving these datasets loses an entire attack category.

## Attack Name Statistics

| Dataset | Attack Names | Families Covered |
|---|---|---|
| NSL-KDD | 40 distinct types | 5/5 |
| UNSW-NB15 | 10 distinct types | 5/5 |
| CICIDS2018 | 15 distinct types | 4/5 |
| TON-IoT | 9 distinct types | 4/5 |

## Pairwise Semantic Overlap

| Pair | Family Jaccard | Overall Overlap | Interpretation |
|---|---|---|---|
| NSL-KDD vs UNSW-NB15 | 1.0 | 0.207 | Full family coverage, but names don't match |
| NSL-KDD vs CICIDS2018 | 0.8 | 0.145 | CICIDS lacks U2R |
| NSL-KDD vs TON-IoT | 0.8 | 0.115 | TON-IoT lacks U2R |
| UNSW-NB15 vs CICIDS2018 | 0.8 | 0.078 | Wildly different attack naming |
| UNSW-NB15 vs TON-IoT | 0.8 | 0.071 | Near-zero name overlap |
| CICIDS2018 vs TON-IoT | 1.0 | 0.160 | Both lack U2R, but names differ |

**Key finding: The overall semantic overlap ranges from 0.07 to 0.21 (on a [0,1] scale).** This means that even after mapping to a common label space, the actual attack types aggregated under each label differ almost completely between datasets.

### Per-Family Overlap Detail

#### DoS Family

| Pair | Overlap | NSL-KDD Names | Other Dataset Names |
|---|---|---|---|
| NSL-KDD vs UNSW-NB15 | 0.000 | apache2, back, land, mailbomb, neptune, pod, processtable, smurf, teardrop, udpstorm | generic |
| NSL-KDD vs CICIDS2018 | 0.000 | 10 DoS types | ddos, dos goldeneye, dos hulk, dos slowhttptest, dos slowloris |
| NSL-KDD vs TON-IoT | 0.000 | 10 DoS types | ddos, dos |
| UNSW-NB15 vs CICIDS2018 | 0.000 | generic | ddos, dos goldeneye, dos hulk, ... |
| UNSW-NB15 vs TON-IoT | 0.000 | generic | ddos, dos |
| CICIDS2018 vs TON-IoT | 0.286 | ddos, dos goldeneye, dos hulk, ... | ddos, dos |

**Zero overlap in 4/6 pairs.** The attack types mapped to "DoS" are completely different across datasets. The only partial overlap is CICIDS↔TON-IoT sharing "ddos" as a label.

#### Probe Family

| Pair | Overlap | NSL-KDD Names | Other Dataset Names |
|---|---|---|---|
| NSL-KDD vs UNSW-NB15 | 0.000 | ipsweep, mscan, nmap, portsweep, saint, satan | analysis, fuzzers, reconnaissance |
| NSL-KDD vs CICIDS2018 | 0.000 | 6 Probe types | portscan |
| NSL-KDD vs TON-IoT | 0.000 | 6 Probe types | scanning |
| UNSW-NB15 vs CICIDS2018 | 0.000 | analysis, fuzzers, reconnaissance | portscan |
| UNSW-NB15 vs TON-IoT | 0.000 | analysis, fuzzers, reconnaissance | scanning |
| CICIDS2018 vs TON-IoT | 0.000 | portscan | scanning |

**Zero overlap in all pairs.** Every dataset uses completely different naming for its Probe/Scanning attacks.

#### R2L Family

| Pair | Overlap | NSL-KDD Names | Other Dataset Names |
|---|---|---|---|
| NSL-KDD vs UNSW-NB15 | 0.000 | ftp_write, guess_passwd, imap, multihop, named, phf, ... | backdoor, exploits, worms |
| NSL-KDD vs TON-IoT | 0.000 | 11 R2L types | backdoor, injection, mitm, ransomware, xss |
| CICIDS2018 vs TON-IoT | 0.000 | bot, ftp-patator, heartbleed, infiltration, ssh-patator, ... | backdoor, injection, mitm, ransomware, xss |

**Zero overlap.** Even when the same conceptual attack type exists (e.g., "backdoor" in both UNSW and TON-IoT), datasets map it to different unified labels.

## Semantic Mismatch Examples

| Attack Behavior | NSL-KDD Label | UNSW-NB15 Label | CICIDS Label | TON-IoT Label |
|---|---|---|---|---|
| Backdoor | R2L (xlock, xsnoop) | R2L (backdoor) | — | R2L (backdoor) |
| Generic (unknown attack) | — | DoS (generic) | — | — |
| Reconnaissance/Scanning | Probe | Probe | Probe | Probe |
| Brute force | R2L | — | R2L | — |
| XSS | — | — | R2L | R2L |

Even when attack names overlap conceptually (e.g., "backdoor" appears in both UNSW and TON-IoT labels), the mapping decisions may differ at the granularity of what constitutes each family.

## Summary

1. **Semantic overlap is critically low** — the maximum overlap score (0.21 for NSL-KDD vs UNSW-NB15) is due entirely to family coverage, not shared attack type names.
2. **Zero overlap in Probe across all pairs.** The concept of "probing" or "scanning" is partitioned differently in every dataset.
3. **DoS is mapped inconsistently.** Generic/DDoS/DoS distinctions are dataset-specific. CICIDS2018 distinguishes 6 DoS subtypes (GoldenEye, Hulk, Slowloris, etc.) while UNSW-NB15 lumps everything into "generic" or "dos".
4. **U2R is absent from CICIDS2018 and TON-IoT**, making U2R detection incomparable.
5. **Label harmonization merges semantically distinct attacks** (e.g., "backdoor" with "exploits" into R2L), creating a coarse label space where a single unified class encompasses qualitatively different attack behaviors.

## Plots

- `plots/phase33/semantic_overlap/semantic_overlap_matrix.png`
