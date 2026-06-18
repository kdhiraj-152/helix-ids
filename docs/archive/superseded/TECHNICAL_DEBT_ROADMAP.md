# Technical Debt Roadmap — Ranked Execution Plan

> Transforms the 21-item Technical Debt Register into a prioritized,
> phased execution roadmap with assigned owners, effort estimates,
> and impact assessments.
>
> **Source:** [`TECHNICAL_DEBT_REGISTER.md`](TECHNICAL_DEBT_REGISTER.md) (18 numbered + 3 tracked items)
> **Context:** [`PHASE19_ARCHITECTURE_FREEZE.md`](PHASE19_ARCHITECTURE_FREEZE.md),
> [`RC1_READINESS.md`](../releases/RC1_READINESS.md),
> [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md),
> [`FINAL_METRICS.md`](FINAL_METRICS.md)
>
> **Last updated:** Phase 19 Architecture Freeze + RC1 Audit

---

## Ranking Methodology

Items are ranked by a composite score weighing:
1. **Severity** (CRITICAL > HIGH > MEDIUM > LOW > INFO) — from the debt register
2. **Production blocker status** — items that prevent production deployment
3. **Impact** — blast radius of the defect if left unfixed
4. **Effort-to-value ratio** — quick wins with high impact rank higher
5. **Dependency order** — items that unlock or simplify later work go first

---

## 1. Phase 20 Sprint — Now (Highest Priority)

| Priority | ID | Title | Severity | Effort (pd) | Impact | Owner | Target Phase |
|---------:|:---|:------|:---------|:------------|:-------|:------|:-------------|
| 1 | TDR-004 | No Frozen Requirements Lockfile | MEDIUM | 1 | **HIGH** — production blocker; builds produce irreproducible environments | CI/DevOps | Phase 20 |
| 2 | TDR-006 | No Structured JSON Logging | MEDIUM | 1–2 | **HIGH** — production blocker; log aggregation (ELK/Loki/Datadog) cannot parse training metrics | Operations | Phase 20 |
| 3 | TDR-001 | ENGINEERED_FEATURE_NAMES Duplication | MEDIUM | 0.25 | **MEDIUM** — violates single-source-of-truth; stale copy in training file risks silent divergence | Domain (src/helix_ids) | Phase 20 |
| 4 | TDR-003 | Loss Logic Still Inline (9 functions) | MEDIUM | 2–3 | **MEDIUM** — prevents independent testing and reuse of individual loss components | Training Pipeline | Phase 20 |

**Phase 20 rationale:** These four items are quick-to-medium wins (4.25–6.25 person-days total) that address the most critical production blockers and architecture invariants. TDR-004 and TDR-006 are production blockers per the RC1 audit. TDR-001 is a 2-hour fix that closes an architecture boundary violation. TDR-003 unlocks cleaner testing for the Phase 21 delegation refactor.

---

## 2. Phase 21 Sprint — Near-Term

| Priority | ID | Title | Severity | Effort (pd) | Impact | Owner | Target Phase |
|---------:|:---|:------|:---------|:------------|:-------|:------|:-------------|
| 5 | TDR-002 | Partial Delegation Anti-Pattern (17 wrappers) | **HIGH** | 3–5 | **HIGH** — creates maintenance coupling and test-surface duplication; the only HIGH-severity item in the register | Training Pipeline | Phase 21 |
| 6 | TDR-007 | No Performance Regression Tests | MEDIUM | 2–3 | **MEDIUM** — training throughput, inference latency, and memory regressions go undetected | QA | Phase 21 |
| 7 | TDR-011 | No Checkpoint Garbage Collection | LOW | 1 | **MEDIUM** — checkpoints accumulate indefinitely; disk exhaustion risk in long training runs | Training Pipeline | Phase 21 |
| 8 | TDR-015 | 30 Full-Delegation Wrappers Not Removed | LOW | 0.25 | **LOW** — ~200 LOC of boilerplate; quick cleanup grouped with delegation refactor | Training Pipeline | Phase 21 |

**Phase 21 rationale:** TDR-002 is the single HIGH-severity item and the most technically significant refactor. TDR-015 is grouped here because it's the same domain (delegation cleanup). TDR-007 closes a FAIL in the Production Readiness audit's Test category. TDR-011 prevents disk-full incidents during production training.

---

## 3. Phase 22 — Medium-Term

| Priority | ID | Title | Severity | Effort (pd) | Impact | Owner | Target Phase |
|---------:|:---|:------|:---------|:------------|:-------|:------|:-------------|
| 9 | TDR-008 | No Circuit Breaker for Inference Service | LOW | 1 | **HIGH** — production blocker; sustained downstream failures cascade without protection | Operations | Phase 22 |
| 10 | TDR-005 | Trainer `__init__` at 392 LOC | LOW | 1–2 | **LOW** — dense initialization method; hard to test in isolation | Training Pipeline | Phase 22 |
| 11 | TDR-010 | Pre-commit Hooks Not Configured | LOW | 0.5 | **MEDIUM** — no local gate for lint/format/type checks; slower developer feedback loop | DevOps | Phase 22 |
| 12 | TDR-013 | Module-Level Helpers in Trainer File | LOW | 1 | **LOW** — clutter in `train_helix_ids_full.py`; extract to `data/` subpackage | Training Pipeline | Phase 22 |
| 13 | TDR-014 | `setup_logging` Duplicated | LOW | 1 | **MEDIUM** — two copies with different behavior; behavioral inconsistency risk | Training Pipeline | Phase 22 |

**Phase 22 rationale:** Mix of code-quality improvements and one production blocker (TDR-008 — circuit breaker). TDR-005, TDR-013, and TDR-014 are straightforward extraction/cleanup tasks. TDR-010 improves the developer experience. Total: 4.5–5.5 person-days.

---

## 4. Phase 23+ — Future / Stretch

| Priority | ID | Title | Severity | Effort (pd) | Impact | Owner | Target Phase |
|---------:|:---|:------|:---------|:------------|:-------|:------|:-------------|
| 14 | TDR-009 | No ONNX Export Path | LOW | 2–3 | **MEDIUM** — limits deployment to Python runtimes; blocks edge/mobile/ONNX RT targets | Model Export (MLOps) | Phase 23 |
| 15 | TDR-012 | No Hypothesis/Property-Based Tests | LOW | 2–3 | **MEDIUM** — edge cases in schema validation, loss computation, and forward pass not fuzzed | QA | Phase 23 |
| 16 | TDR-017 | No `.env` / Environment-Variable Config Loading | INFO | 1 | **MEDIUM** — production blocker for containerized/cloud deployments; no env-var override mechanism | Configuration (Domain) | Phase 23 |
| 17 | TDR-018 | No Auto-Restart for Deployed Service | INFO | 1 | **HIGH** — production blocker; single-process REST server stays down on crash | Operations | Phase 23 |
| 18 | TDR-016 | Pre-commit Check Framework Not Installed | INFO | 0.125 | **LOW** — no pre-push gate for architecture freeze tests; minor process gap | DevOps | Phase 23 |

**Phase 23+ rationale:** These items are stretch goals that do not block RC1 or RC2. TDR-009 and TDR-012 expand the testing and deployment envelope. TDR-017 and TDR-018 are production blockers for cloud/containerized deployments but are acceptable for on-prem/controlled deployments. TDR-016 is a <1-hour process improvement.

---

## 5. Effort Summary

| Phase | Items | Effort (person-days) |
|:------|:------|:--------------------|
| **Phase 20** — Now | 4 (TDR-001, TDR-003, TDR-004, TDR-006) | 4.25–6.25 |
| **Phase 21** — Near-term | 4 (TDR-002, TDR-007, TDR-011, TDR-015) | 6.25–9.25 |
| **Phase 22** — Medium-term | 5 (TDR-005, TDR-008, TDR-010, TDR-013, TDR-014) | 4.5–5.5 |
| **Phase 23+** — Future | 5 (TDR-009, TDR-012, TDR-016, TDR-017, TDR-018) | 6.125–8.125 |
| **Total (all 18 numbered items)** | **18** | **21.125–29.125** |

> **Note:** The register tracks 21 items (1 HIGH, 8 MEDIUM, 9 LOW, 3 INFO). Eighteen are individually numbered (TDR-001 through TDR-018); three additional items are implicitly tracked via the Production Readiness audit gaps (config schema versioning, dataset content-hash pinning, drift detection). These three are covered by the production-readiness lifecycle and are not included in this ranked plan.

---

## 6. Roadmap Summary Table

| Rank | ID | Item | Effort | Impact | Owner | Phase |
|-----:|:---|:-----|:------:|:------:|:------|:-----:|
| 1 | TDR-004 | Frozen requirements lockfile | 1 pd | HIGH | CI/DevOps | **20** |
| 2 | TDR-006 | Structured JSON logging | 1–2 pd | HIGH | Operations | **20** |
| 3 | TDR-001 | ENGINEERED_FEATURE_NAMES dedup | 0.25 pd | MEDIUM | Domain | **20** |
| 4 | TDR-003 | Inline losses → LossRegistry | 2–3 pd | MEDIUM | Training Pipeline | **20** |
| 5 | TDR-002 | Partial delegation anti-pattern | 3–5 pd | HIGH | Training Pipeline | **21** |
| 6 | TDR-007 | Performance regression tests | 2–3 pd | MEDIUM | QA | **21** |
| 7 | TDR-011 | Checkpoint garbage collection | 1 pd | MEDIUM | Training Pipeline | **21** |
| 8 | TDR-015 | Remove 30 full-delegation wrappers | 0.25 pd | LOW | Training Pipeline | **21** |
| 9 | TDR-008 | Circuit breaker for inference service | 1 pd | HIGH | Operations | **22** |
| 10 | TDR-005 | Split trainer `__init__` | 1–2 pd | LOW | Training Pipeline | **22** |
| 11 | TDR-010 | Pre-commit hooks configuration | 0.5 pd | MEDIUM | DevOps | **22** |
| 12 | TDR-013 | Extract module-level helpers | 1 pd | LOW | Training Pipeline | **22** |
| 13 | TDR-014 | Deduplicate `setup_logging` | 1 pd | MEDIUM | Training Pipeline | **22** |
| 14 | TDR-009 | ONNX export path | 2–3 pd | MEDIUM | Model Export | **23** |
| 15 | TDR-012 | Hypothesis/property-based tests | 2–3 pd | MEDIUM | QA | **23** |
| 16 | TDR-017 | `.env` / env-var config loading | 1 pd | MEDIUM | Configuration | **23** |
| 17 | TDR-018 | Auto-restart for deployed service | 1 pd | HIGH | Operations | **23** |
| 18 | TDR-016 | Pre-commit check framework hook | 0.125 pd | LOW | DevOps | **23** |

---

## 7. Key Dependencies & Sequencing Notes

- **TDR-003 (inline losses) should precede TDR-002 (delegation refactor):** Moving loss functions into `LossRegistry` simplifies the trainer and reduces the surface area that the Phase 21 delegation refactor needs to touch.
- **TDR-004 (lockfile) and TDR-006 (structured logging) are zero-dependency:** They can be executed in parallel and should be started first.
- **TDR-015 (remove wrappers) is gated on TDR-002:** The 30 full-delegation wrappers should be removed as part of the broader delegation cleanup; doing them separately would cause churn.
- **TDR-016 (pre-commit hook) is gated on TDR-010 (pre-commit config):** The hook cannot be installed until `.pre-commit-config.yaml` exists.

---

## 8. Production Blocker Coverage

| Production Blocker (from RC1 audit) | TDR ID | Phase | Effort |
|:-------------------------------------|:-------|:-----|:------:|
| Frozen requirements lockfile | TDR-004 | 20 | 1 pd |
| Structured JSON logging | TDR-006 | 20 | 1–2 pd |
| Circuit breaker for downstream services | TDR-008 | 22 | 1 pd |
| Auto-restart for inference server | TDR-018 | 23 | 1 pd |
| Env-var config loading for containerization | TDR-017 | 23 | 1 pd |

All five production blockers from the RC1 readiness audit are mapped to specific TDR items with assigned phases and owners.

---

*End of Technical Debt Roadmap. For the full register with detailed descriptions, see [`TECHNICAL_DEBT_REGISTER.md`](TECHNICAL_DEBT_REGISTER.md).*
