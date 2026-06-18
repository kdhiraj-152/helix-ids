# Production Blockers

## Gate: Production Certification — 24-Hour Soak

This document catalogs all findings that must be resolved BEFORE the 24-hour production soak can certify runtime behavior as correct. Findings are ordered by severity within each gate.

---

## GATE 1: Security Integrity (CRITICAL for Production)

### B-01: `eval()` in Benchmark Entry Point
| | |
|---|---|
| **Severity** | HIGH |
| **File** | `scripts/evaluation/benchmark_e2e.py:53` |
| **Description** | `eval()` of parsed JSON architecture string enables arbitrary code execution |
| **Status** | UNRESOLVED |
| **Gate** | Must fix before soak — a crafted benchmark config can execute arbitrary Python |

### B-02: `weights_only=False` in Transfer Learning
| | |
|---|---|
| **Severity** | HIGH |
| **File** | `src/helix_ids/models/adaptation/transfer_learning.py:1185` |
| **Description** | Pickle deserialization allows arbitrary code execution during model loading |
| **Status** | UNRESOLVED |
| **Gate** | Must fix before soak — loading a compromised checkpoint is RCE |

### B-03: `assert` Statements Disabled Under `-O`
| | |
|---|---|
| **Severity** | HIGH |
| **Files** | `trainer_facade.py:133-218`, `deploy.py:147-230` (40+ sites) |
| **Description** | Critical guards silently removed when Python runs with `-O` |
| **Status** | UNRESOLVED |
| **Gate** | Must fix before production deployment with `python -O` |

### B-04: `pickle.load` in Tests
| | |
|---|---|
| **Severity** | MEDIUM |
| **Files** | `tests/test_callbacks.py:339`, `tests/test_fault_injection.py:306`, `tests/test_export_quantization_deployment.py:243` |
| **Description** | Test code uses unsafe deserialization patterns |
| **Status** | UNRESOLVED |
| **Gate** | Should fix — establishes bad patterns that migrate to production |

---

## GATE 2: Operational Reliability

### B-05: Zero Retry Patterns
| | |
|---|---|
| **Severity** | HIGH |
| **Scope** | Entire codebase |
| **Description** | No retry/backoff for any network or I/O operation |
| **Status** | UNRESOLVED |
| **Gate** | Blocking for soak — transient failures will cause flaky certification |

### B-06: Only 3 `finally` Blocks
| | |
|---|---|
| **Severity** | HIGH |
| **Scope** | Entire codebase (57K LOC) |
| **Description** | Resource cleanup not guaranteed on exception paths |
| **Status** | UNRESOLVED |
| **Gate** | Blocking for soak — resource leaks degrade over 24 hours |

### B-07: No Concurrency Safety
| | |
|---|---|
| **Severity** | HIGH |
| **Scope** | `scripts/operations/serve_rest.py` |
| **Description** | Multi-worker REST server has no shared-state protection |
| **Status** | UNRESOLVED |
| **Gate** | Blocking for soak — race conditions cause non-deterministic behavior |

### B-08: No External Alerting
| | |
|---|---|
| **Severity** | HIGH |
| **Scope** | Entire monitoring system |
| **Description** | No way to notify operators of production incidents |
| **Status** | UNRESOLVED |
| **Gate** | Blocking for production — must have notification channel before go-live |

---

## GATE 3: Testing Completeness

### B-09: No E2E Pipeline Test
| | |
|---|---|
| **Severity** | HIGH |
| **Scope** | Missing integration test coverage |
| **Description** | No test exercises the full pipeline end-to-end |
| **Status** | UNRESOLVED |
| **Gate** | Should fix before soak — no automated validation of pipeline integrity |

### B-10: No Property-Based Tests
| | |
|---|---|
| **Severity** | MEDIUM |
| **Scope** | Metrics, losses, data transformations |
| **Description** | Critical invariants untested for edge cases |
| **Status** | UNRESOLVED |
| **Gate** | Should fix before soak — metric correctness is foundational |

### B-11: No Negative Tests
| | |
|---|---|
| **Severity** | MEDIUM |
| **Scope** | Error handling paths |
| **Description** | Defensive code is entirely untested |
| **Status** | UNRESOLVED |
| **Gate** | Should fix before soak — error handling could be buggy |

---

## GATE 4: Architecture Stability

### B-12: Cross-Layer Coupling
| | |
|---|---|
| **Severity** | MEDIUM |
| **Files** | `governance/orchestrator.py` imports models; `data/fingerprinting.py` imports governance |
| **Description** | Core package has layer violations between subpackages |
| **Status** | UNRESOLVED |
| **Gate** | Not blocking for soak but increases change-risk during maintenance |

### B-13: `ENGINEERED_FEATURE_NAMES` Violation
| | |
|---|---|
| **Severity** | MEDIUM |
| **Files** | `src/helix_ids/contracts/schema_contract.py` imports from `scripts/training/data/feature_engineering.py` |
| **Description** | Core package depends on training scripts — wrong direction |
| **Status** | UNRESOLVED |
| **Gate** | Not blocking for soak, but should be resolved in Phase 24 |

---

## GATE 5: Dependency Hygiene

### B-14: 18 Outdated Packages
| | |
|---|---|
| **Severity** | LOW |
| **Scope** | All lockfiles |
| **Description** | 18 packages have newer versions available; `chardet` has a major version gap (5.2 → 7.4) |
| **Status** | UNRESOLVED |
| **Gate** | Not blocking for soak, but update known-affected packages |

### B-15: No Automated CVE Scanning
| | |
|---|---|
| **Severity** | MEDIUM |
| **Scope** | CI pipeline |
| **Description** | No `pip-audit` or `safety` step in CI |
| **Status** | UNRESOLVED |
| **Gate** | Should fix before production deployment |

---

## Production Readiness Decision Matrix

| Gate | Condition | Status |
|------|-----------|--------|
| **G1: Security** | 0 unknown HIGH+ risks | ❌ FAIL — 3 HIGH, 1 MEDIUM |
| **G2: Operations** | Retries + cleanup + concurrency + alerts | ❌ FAIL — 4 HIGH gaps |
| **G3: Testing** | E2E test + property tests + negative tests | ❌ FAIL — 1 HIGH, 2 MEDIUM |
| **G4: Architecture** | Layer violations resolved | ⚠️ PASS — 2 MEDIUM, not blocking |
| **G5: Dependencies** | CVEs scanned, versions current | ⚠️ PASS — 1 MEDIUM, not blocking |

## VERDICT

**NO-GO for production certification without remediation.**

The soak run would certify runtime behavior while 12+ high-severity issues remain unresolved. The recommended sequence is:

1. Remediate B-01 through B-11 (all High severity)
2. Run the full test suite + new E2E test
3. Run the 24-hour soak
4. Final production certification

**Estimated remediation time: 5-8 person-days for all blocking items.**
