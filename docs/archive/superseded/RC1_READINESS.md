# RC1 Readiness Audit — HELIX-IDS

> Release Candidate 1 assessment for HELIX-IDS.
> Based on Phase 19 Architecture Freeze deliverables, Technical Debt Register (21 items),
> Production Readiness audit (12 categories), Reproducibility scorecard, and CI/CD inventory.

**Date:** 2026-06-16
**Status:** `rc1-readiness-audit`
**Classifier (pyproject.toml):** `Development Status :: 3 - Alpha`

---

## Table of Contents

1. [Known Issues](#1-known-issues)
2. [Open Technical Debt](#2-open-technical-debt)
3. [Production Blockers](#3-production-blockers)
4. [Performance Risks](#4-performance-risks)
5. [Maintenance Risks](#5-maintenance-risks)
6. [Final Verdict](#6-final-verdict)

---

## 1. Known Issues

Issues are drawn from the Technical Debt Register (TDR), Production Readiness audit
warnings, and Phase 19 architecture freeze gaps. Each is cross-referenced to source.

### 1.1 Architecture Freeze Violations / Boundary Gaps

| ID / Source | Issue | Severity | Status |
|---|---|---|---|
| TDR-001 / Phase 19 §7 | `ENGINEERED_FEATURE_NAMES` defined in both `src/helix_ids/data/feature_harmonization.py` and `scripts/training/train_helix_ids_full.py`. The `scripts/training` copy is stale; all importers should use the `src` copy. | MEDIUM | Open — Phase 20 |
| TDR-016 / Phase 19 §6 | Pre-commit check framework not installed. Developers can commit boundary violations locally without the architecture freeze gate running. | INFO | Open — Phase 23 |
| Phase 19 §8.3 | Architecture freeze exceptions are possible for bug fixes, security patches, perf optimizations, and new delegates. No standing exceptions at this time. | — | Stable |

### 1.2 Production Readiness Warnings (4 Categories)

The Production Readiness audit scores **8 PASS / 4 WARNING / 0 FAIL** overall.
All four WARNING categories represent known issues for RC1:

| Category | Score | Key Issues |
|---|---|---|
| **Typing** | WARNING | Core library (`src/helix_ids/`) has good coverage; `scripts/` have fewer annotations. Some packages missing `__all__` exports. |
| **Tests** | WARNING | No performance regression tests (FAIL). No property-based/fuzzing tests (FAIL). Coverage at 69.9% — just 4.9% above the 65% gate. |
| **Logging** | WARNING | Mix of `print()`, `logging.info()`, `stdout.write()`. No JSON log format (FAIL). No request-scoped correlation IDs (FAIL). |
| **Failure Recovery** | WARNING | No auto-restart on crash (WARNING). No circuit breaker for downstream service failures (FAIL). |

### 1.3 Reproducibility Gaps

From the Reproducibility scorecard:

| Gap | Status | Impact |
|---|---|---|
| No Dockerfile for containerized reproduction | MISSING | Third parties cannot reproduce results in an identical environment |
| No automated reproduction validation script | MISSING | No script exists to verify that a reproduction attempt succeeded |
| Hardware configuration not recorded automatically | MISSING | Cannot audit whether training runs used equivalent hardware |
| Single-seed default; multi-seed results not published | MISSING | Statistical significance of results is unconfirmed |
| Cross-platform testing (macOS/Ubuntu only; Windows untested) | MISSING | Windows deployment or development is unsupported |
| CICIDS-2018 acquisition requires manual symlink | PARTIAL | Full automation of the CICIDS-2018 pipeline is blocked by external storage requirements |

### 1.4 Dataset & Data Pipeline Issues

| Source | Issue | Severity |
|---|---|---|
| Production Readiness §11 | Dataset versioning tracked by name, not content hash | WARNING |
| Production Readiness §11 | No automated drift detection pipeline | WARNING |

### 1.5 Configuration & Deployment Issues

| Source | Issue | Severity |
|---|---|---|
| Production Readiness §9 | No environment-variable config loading (`.env` support) — FAIL | WARNING |
| Production Readiness §9 | No config schema versioning/compatibility logic | WARNING |

---

## 2. Open Technical Debt

All 21 registered items from `docs/architecture/TECHNICAL_DEBT_REGISTER.md`.
Estimated total effort: **17–26 person-days**.

### 2.1 Top Items by Severity

#### HIGH (1 item)

| ID | Item | Effort | Phase | Rationale |
|---|---|---|---|---|
| **TDR-002** | Partial delegation anti-pattern — 17 methods in HelixFullTrainer mix delegate calls with inline orchestration | 3–5 days | Phase 21 | Creates maintenance coupling and test-surface duplication. Trainer still carries phase-management logic that should live in delegate objects. |

#### MEDIUM (8 items)

| ID | Item | Effort | Phase | Rationale |
|---|---|---|---|---|
| **TDR-001** | `ENGINEERED_FEATURE_NAMES` duplication between `src/` and `scripts/training/` | 1–2 hours | Phase 20 | Violates single-source-of-truth. Stale copy in training file risks divergence. |
| **TDR-003** | 9 loss functions still inline in trainer despite `LossRegistry` existing | 2–3 days | Phase 20 | Prevents independent testing and reuse of individual loss components. |
| **TDR-004** | No frozen requirements lockfile | 1 day | Phase 20 | Builds can produce different environments depending on when deps were resolved. |
| **TDR-006** | No structured JSON logging | 1–2 days | Phase 20 | Log aggregation pipelines (ELK, Loki, Datadog) cannot parse training metrics. |
| **TDR-007** | No performance regression tests | 2–3 days | Phase 21 | Training throughput, inference latency, and memory usage regressions can go undetected. |
| **TDR-008** | No circuit breaker for inference service | 1 day | Phase 22 | Sustained downstream failures cascade without protection. *(Note: listed as LOW severity in item body, MEDIUM in summary table — assess at MEDIUM for production contexts.)* |
| **TDR-011** | No checkpoint garbage collection | 1 day | Phase 21 | Checkpoints accumulate indefinitely with no `max_checkpoints` or TTL-based cleanup. |
| — | No config lockfile (overlaps TDR-004) | — | Phase 20 | Covered by TDR-004 above. |

#### LOW (9 items)

| ID | Item | Effort | Phase |
|---|---|---|---|
| TDR-005 | Trainer `__init__` at 392 LOC | 1–2 days | Phase 22 |
| TDR-008 | (if assessed as LOW) No circuit breaker | 1 day | Phase 22 |
| TDR-009 | No ONNX export path | 2–3 days | Phase 23 |
| TDR-010 | Pre-commit hooks not configured | 0.5 day | Phase 22 |
| TDR-012 | No hypothesis/property-based tests | 2–3 days | Phase 23 |
| TDR-013 | Module-level helpers in `train_helix_ids_full.py` | 1 day | Phase 22 |
| TDR-014 | `setup_logging` duplicated across two files | 1 day | Phase 22 |
| TDR-015 | 30 full-delegation wrappers not removed | 2 hours | Phase 21 |
| TDR-016 | Pre-commit check framework not installed | 1 hour | Phase 23 |

#### INFO (3 items)

| ID | Item | Effort | Phase |
|---|---|---|---|
| TDR-017 | No `.env` / environment-variable config loading | 1 day | Phase 23 |
| TDR-018 | No auto-restart for deployed service | 1 day | Phase 23 |

### 2.2 Effort Breakdown

| Horizon | Items | Person-Days |
|---|---|---|
| Immediate (Phase 20) | TDR-001, TDR-003, TDR-004, TDR-006 | 5–8 |
| Medium-term (Phase 21) | TDR-002, TDR-007, TDR-011, TDR-015 | 7–10 |
| Long-term (Phase 22+) | TDR-005, TDR-008, TDR-009, TDR-010, TDR-012, TDR-013, TDR-014, TDR-016, TDR-017, TDR-018 | 5–8 |
| **Total** | **21 items** | **17–26 person-days** |

---

## 3. Production Blockers

These items **prevent** a production deployment of HELIX-IDS. They represent
either (a) gaps that cause the system to fail in production scenarios or
(b) missing capabilities that production operations require.

### 3.1 Blocker: No Frozen Requirements Lockfile (TDR-004)

**Risk:** Without `requirements-lock.txt`, two deployments built from the same
`pyproject.toml` at different times may resolve different dependency versions.
A dependency that introduces a breaking change or vulnerability between
resolution events will pass CI and reach production.

**Severity:** MEDIUM (escalates to HIGH for production deployment)

**Mitigation:** Generate lockfile (1 day effort, Phase 20).

### 3.2 Blocker: No Structured / JSON Logging (TDR-006)

**Risk:** Production monitoring and alerting depend on log aggregation.
Without structured JSON logging, log ingestion pipelines (ELK, Loki, Datadog,
Splunk) cannot parse training metrics or operational diagnostics automatically.
Incident response is manual and slow.

**Severity:** MEDIUM (escalates to HIGH for production operations)

**Mitigation:** Add structured logging utility, migrate step-level diagnostics
to JSON format (1–2 days, Phase 20).

### 3.3 Blocker: No Auto-Restart for Deployed Service (TDR-018)

**Risk:** The REST inference server (`serve_rest.py`) runs as a single process.
If it crashes (OOM, unhandled exception, dependency fault), it stays down until
manually restarted. This creates a hard outage window of unknown duration.

**Severity:** INFO (escalates to HIGH for production deployment)

**Mitigation:** Add systemd service or supervisord config (1 day, Phase 23).

### 3.4 Blocker: No Circuit Breaker (TDR-008)

**Risk:** The inference service has health checks but no circuit breaker for
degraded downstream services. If a dependency (e.g., database, model registry)
fails, the inference service will continue to attempt connections, accumulate
backpressure, and eventually exhaust resources — cascading the failure.

**Severity:** LOW–MEDIUM (escalates to HIGH for production)

**Mitigation:** Add stateful circuit breaker in `monitoring.py` (1 day, Phase 22).

### 3.5 Blocker: No Environment-Variable Config Loading (TDR-017)

**Risk:** Production deployments require environment-specific configuration
(`DATABASE_URL`, `REDIS_HOST`, `API_KEYS`). The system loads all config from
YAML files only, with no `.env` support or environment-variable override
mechanism. This makes containerized / cloud deployment difficult.

**Severity:** INFO (escalates to MEDIUM for production)

**Mitigation:** Add `python-dotenv` support in `config_parser.py` (1 day,
Phase 23).

### 3.6 Summary: Blocker Impact

| Blocker | Production Component | Min. Effort | Required Before Production |
|---|---|---|---|
| No frozen lockfile | All deployments | 1 day | Yes |
| No structured logging | All operations | 1–2 days | Yes |
| No auto-restart | REST inference server | 1 day | Yes |
| No circuit breaker | REST inference server | 1 day | Yes |
| No env-var config loading | Container/cloud deployments | 1 day | Yes |
| **Total** | | **5–6 days** | |

---

## 4. Performance Risks

### 4.1 No Performance Regression Tests (TDR-007)

**Status:** FAIL in Production Readiness audit (Test category).

**Risk:** There are no benchmarks for training throughput, inference latency,
or memory usage. Performance regressions — whether from code changes,
dependency upgrades, or model architecture changes — cannot be detected
automatically. The system may silently degrade in production.

**Mitigation:** Add `pytest-benchmark` harness for inference path, track in CI
(2–3 days, Phase 21).

### 4.2 No Property-Based / Fuzzing Tests (TDR-012)

**Status:** FAIL in Production Readiness audit (Test category).

**Risk:** Edge cases in input-space combinations (feature vectors, schema
validation, loss computation) are not fuzzed. Unusual but valid inputs could
trigger slow paths or excessive memory allocation in production.

**Mitigation:** Add `hypothesis` tests for `schema_contract.validate()`, loss
functions, and model forward pass (2–3 days, Phase 23).

### 4.3 No ONNX Export Path (TDR-009)

**Status:** FAIL in Production Readiness audit (Model Export Path category).

**Risk:** Only PyTorch native export is available. This limits deployment
targets to Python-based runtimes. Edge deployments (RPi, ESP32) and
non-Python production runtimes cannot use optimized ONNX Runtime, TensorRT,
or ONNX-compatible inference servers. Inference latency on constrained
hardware may be higher than necessary.

**Mitigation:** Add `torch.onnx.export` path, validate on inference runtime
(2–3 days, Phase 23).

### 4.4 Coverage Margin Is Thin

**Metric:** Line coverage is 69.9%, with a gate of ≥65%.

**Risk:** A margin of only 4.9% above the minimum gate provides no safety
buffer. A single release that adds untested code paths (or adjusts coverage
measurement) could push the system below the threshold. This is especially
risky given that branch coverage is not measured at all.

**Current status:** PASS (within margin)

### 4.5 No Checkpoint Garbage Collection (TDR-011)

**Risk:** Checkpoints accumulate indefinitely. During long training runs or
continuous training pipelines, disk usage grows without bound. A production
training job with default settings could fill available storage and crash.

**Mitigation:** Add checkpoint rotation with `max_checkpoints` or TTL-based
cleanup in `_save_checkpoint_if_needed` (1 day, Phase 21).

---

## 5. Maintenance Risks

### 5.1 CI/CD Gaps

| Risk | Source | Detail |
|---|---|---|
| No pre-commit hooks | TDR-010 / Production Readiness §2 | Developers can commit lint/format/type violations locally. CI catches them, but the feedback loop is slower. |
| No pre-push architecture freeze gate | TDR-016 | Developers can commit boundary violations locally without running architecture tests. |
| No `pip audit` / `safety` check in CI | Production Readiness §10 | Vulnerable dependency versions are not automatically detected beyond Dependabot (which only scans GitHub advisories, not PyPI). |
| No performance benchmark CI step | TDR-007 | Performance regressions pass CI silently. |

### 5.2 Documentation Risks

| Risk | Source | Detail |
|---|---|---|
| Reproducibility not automatable | Reproducibility scorecard | No Dockerfile, no automated reproduction validation script. A third party cannot verify results without manual setup. |
| No deployment runbook | Implicit | The `scripts/deployment/` directory exists but no deployment operations guide is documented. |
| Classifier is "Alpha" | `pyproject.toml` | The project is still classified as `Development Status :: 3 - Alpha`. This signals to users that the system is not yet stable. |

### 5.3 Dependency Management Risks

| Risk | Source | Detail |
|---|---|---|
| No lockfile | TDR-004 | Build reproducibility is not guaranteed across time or machines. |
| Version ranges only | `pyproject.toml` | Dependencies like `torch>=2.0.0`, `numpy>=1.21.0` use open-ended ranges. Minor version bumps could introduce breaking changes. |
| Pre-commit not installed | TDR-010 | `pre-commit>=3.0.0` is listed as an optional dev dependency but no `.pre-commit-config.yaml` exists. |

### 5.4 Code Quality Risks

| Risk | Source | Detail |
|---|---|---|
| Partial delegation anti-pattern | TDR-002 | 17 methods in the trainer mix delegation with inline orchestration. Complicates maintenance and testing. |
| Inline loss functions | TDR-003 | 9 loss functions defined in the trainer instead of registered in `LossRegistry`. |
| `setup_logging` duplicated | TDR-014 | Two copies of `setup_logging` with different behavior exist (`train_helix_ids_full.py` and `orchestration/run_orchestrator.py`). |
| Module-level helpers in trainer file | TDR-013 | `MultiTaskNumpyDataset`, `ClassBalancedIndexSampler`, `FrozenIndexSampler` defined at module level in the trainer file instead of in `data/` subpackage. |

---

## 6. Final Verdict

### Assessment Summary

| Dimension | Score | Key Evidence |
|---|---|---|
| Architecture Integrity | **PASS** | 0 cycles, 0 reverse deps, 24/24 architecture tests pass. Phase 19 freeze certified GO. |
| Production Readiness | **PASS with WARNINGS** | 8/12 PASS, 4/12 WARNING. No FAIL categories. |
| Technical Debt | **21 open items** | 1 HIGH, 8 MEDIUM, 9 LOW, 3 INFO. 0 CRITICAL. Estimated 17–26 person-days. |
| Reproducibility | **PARTIAL** | Deterministic training works; no Dockerfile, no automated validation script. |
| CI/CD | **FUNCTIONAL** | 7 GitHub Actions workflows active. Architecture lockdown, quality gates, dependabot, CodeQL, SLSA provenance. |
| Performance Testing | **MISSING** | No regression benchmarks. No property-based tests. |
| Deployment Readiness | **NOT PRODUCTION-READY** | No auto-restart, no circuit breaker, no structured logging, no lockfile, no env-var config. |

### Production Blockers Checklist

| Requirement | Status | Minimum Resolution |
|---|---|---|
| Frozen requirements lockfile | ❌ NOT DONE | 1 day (Phase 20) |
| Structured JSON logging | ❌ NOT DONE | 1–2 days (Phase 20) |
| Auto-restart for inference server | ❌ NOT DONE | 1 day (Phase 23) |
| Circuit breaker for downstream services | ❌ NOT DONE | 1 day (Phase 22) |
| Env-var config loading for containerization | ❌ NOT DONE | 1 day (Phase 23) |
| Performance regression tests | ❌ NOT DONE | 2–3 days (Phase 21) |

### Verdict

> ## ⚠️ READY WITH WARNINGS — NOT READY FOR PRODUCTION

**HELIX-IDS RC1 is architecturally sound** (zero reverse deps, zero cycles,
all freeze gates passing) and ready for release-candidate testing by
integrators and evaluators. The Phase 19 architecture freeze has been
successfully certified with all 7 deliverables produced.

**However, HELIX-IDS RC1 is NOT READY for production deployment.**
Five production blockers (lockfile, structured logging, auto-restart,
circuit breaker, env-var config) must be resolved before any production
workload can be accepted. These blockers represent an estimated **5–6 days**
of additional work. The `pyproject.toml` classifier (`Development Status :: 3 - Alpha`)
correctly reflects this stage.

**Recommended next actions for RC1 → RC2:**

1. **Phase 20 (5–8 person-days):** Resolve TDR-001 (feature names dup), TDR-003
   (inline losses), TDR-004 (lockfile), TDR-006 (structured logging).
2. **Add `requirements-lock.txt`:** Highest priority — blocks all production
   deployment and reproducible CI.
3. **Add structured JSON logging:** Required for any operational monitoring.
4. **Post-RC1:** Begin Phase 21 (TDR-002 delegation refactor, TDR-007 perf
   benchmarks, TDR-011 checkpoint GC) in parallel with RC1 evaluation.
5. **Classifier bump to Beta:** When all 5 production blockers are resolved
   and ≥1 Phase 20 sprint is complete.

---

*End of RC1 Readiness Audit. Evidence sources:*
- `docs/architecture/TECHNICAL_DEBT_REGISTER.md` (21 items)
- `docs/architecture/PHASE19_ARCHITECTURE_FREEZE.md` (GO certification)
- `docs/architecture/PRODUCTION_READINESS.md` (12 categories)
- `docs/architecture/FINAL_METRICS.md` (snapshot data)
- `docs/reproducibility/REPRODUCIBILITY.md` (scorecard)
- `tests/architecture/test_architecture_lockdown.py` (D1 deliverable)
- `.github/workflows/architecture.yml` (D2 deliverable)
- `.github/workflows/quality.yml` (D2 deliverable)
- `pyproject.toml` (version, classifiers, tool config)
- `README.md` (project overview)
- `scripts/ci/` (18 CI scripts)
