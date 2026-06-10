# Paper Readiness Audit

> Assessment of HELIX-IDS readiness for academic publication venues.

Last updated: 2026-06-09

## Venue Requirements

### IEEE S&P / Oakland

| Requirement | Status | Evidence |
|------------|--------|----------|
| Novel contribution | PARTIAL | Threshold decoupling is novel; not formally proven |
| Rigorous evaluation | MISSING | No baselines, no ablations, no significance |
| Reproducibility | MISSING | No container, no automated reproduction |
| Related work coverage | PARTIAL | ~300 words, needs expansion |
| Results with full tables | PARTIAL | 16 tables in manuscript, no raw outputs |

**Verdict**: NOT READY — requires experimental evidence

### ACM CCS

| Requirement | Status | Evidence |
|------------|--------|----------|
| System description | PARTIAL | Architecture documented, deployment incomplete |
| Threat model | PARTIAL | Security review written, deployment threats not validated |
| Evaluation | MISSING | No hardware evaluation |
| Artifact evaluation | MISSING | No AEC submission ready |

**Verdict**: NOT READY — requires evaluation + artifacts

### USENIX Security

| Requirement | Status | Evidence |
|------------|--------|----------|
| Real-world relevance | PARTIAL | Datasets are academic, not real |
| Impact measurement | MISSING | No deployment study |
| Reproducibility | MISSING | Not reproducible by third party |

**Verdict**: NOT READY

### NDSS

| Requirement | Status | Evidence |
|------------|--------|----------|
| Network security focus | ✓ MATCH | NIDS application |
| Experimental rigor | MISSING | No baselines, no significance |
| Practical deployment | MISSING | No real deployment study |

**Verdict**: NOT READY

### RAID

| Requirement | Status | Evidence |
|------------|--------|----------|
| Intrusion detection focus | ✓ MATCH | Core application |
| Reproducibility | PARTIAL | Governance framework is strong; experiments are not |
| Practical relevance | PARTIAL | Edge deployment focus is relevant |

**Verdict**: PARTIALLY READY — needs experiments

### ACSAC

| Requirement | Status | Evidence |
|------------|--------|----------|
| Applied contribution | ✓ MATCH | Production readiness focus |
| System building | ✓ MATCH | Full system implemented |
| Evaluation | MISSING | No benchmarks run |

**Verdict**: NOT READY — no evaluation results

## Paper Section — Evidence Mapping

| Paper Section | Evidence Source | Missing | Blocker? |
|--------------|----------------|---------|----------|
| Abstract | Manuscript draft | Claims untested | YES |
| Introduction | Manuscript draft | Problem formalization | NO (cosmetic) |
| Related Work | Manuscript ($\sim$300 words) | Systematic literature review | PARTIAL |
| Methodology | Manuscript (Sections III-V), code | Formal proofs | PARTIAL |
| System Design | ARCHITECTURE.md, code | None | NO |
| PRI Framework | None | **No specification, no implementation** | **YES** |
| Experimental Setup | None | **Hardware, seeds, significance** | **YES** |
| Datasets | Code, processing scripts | **Class distributions, statistics** | PARTIAL |
| Hardware Evaluation | None | **No benchmarks run** | **YES** |
| Results | Manuscript (16 tables) | **No raw outputs, no reproduction** | PARTIAL |
| Ablation Studies | None | **No ablation experiments** | **YES** |
| Discussion | Manuscript | Practical implications | PARTIAL |
| Limitations | This document | Formal documentation | NO (now exists) |
| Reproducibility | This document | **Container, automation** | PARTIAL |

## Claim Verification

Every claim in the manuscript must have supporting evidence. The following claims are at risk:

| Claim | Evidence Status | Required |
|-------|----------------|----------|
| "HELIX achieves state-of-the-art detection" | **NO EVIDENCE** | Baseline comparisons |
| "Per-class margin penalties recover rare-class F1" | **NO EVIDENCE** | Ablation study |
| "Threshold decoupling prevents false negatives" | PARTIAL (code exists) | Controlled experiment |
| "HELIX is edge-optimized" | **NO EVIDENCE** | Hardware benchmarks |
| "Domain adaptation enables cross-dataset generalization" | **NO EVIDENCE** | DA evaluation |
| "Governance framework ensures reproducibility" | PARTIAL (docs) | AEC reproduction |
| "PRI framework evaluates production readiness" | **NO EVIDENCE** | **Framework not implemented** |

## Blocker Summary

| Blocker | Severity | Resolution | Effort |
|---------|----------|-----------|--------|
| No baseline comparisons | CRITICAL | Run RF, SVM, MLP baselines | 2-3 days |
| No ablation study | CRITICAL | Run ablations (6+ configs) | 3-5 days |
| No hardware benchmarks | CRITICAL | Run latency/throughput tests | 2-3 days |
| No experimental setup doc | HIGH | Document (exists now) | 0 (done) |
| No statistical significance | HIGH | Multi-seed runs + CI | 2-3 days |
| PRI not implemented | HIGH | Implement scoring function | 1 day |
| Missing dataset statistics | HIGH | Extract from data | 1 day |
| Container not available | MEDIUM | Build Dockerfile | 1 day |
| Related work thin | MEDIUM | Literature expansion | 2-3 days |
| Manuscript formatting | LOW | IEEE template | 0.5 day |

## Estimated Time to Submission Readiness

| Venue | Current | With 2 weeks | With 4 weeks |
|-------|---------|-------------|-------------|
| IEEE S&P | 0/10 | 2/10 | 4/10 |
| ACM CCS | 1/10 | 4/10 | 6/10 |
| RAID | 3/10 | 6/10 | 8/10 |
| ACSAC | 2/10 | 5/10 | 7/10 |

**Most realistic target**: RAID or ACSAC with 4 weeks of focused experimentation.
