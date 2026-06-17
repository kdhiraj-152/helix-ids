# Technical Debt Register — Phase 19 Architecture Freeze

All remaining architectural, operational, and code-quality issues catalogued
with severity, ownership, and recommended remediation phase.

---

## Severity Levels

| Level | Meaning | Response |
|---|---|---|
| **CRITICAL** | Blocks deployment or breaks architecture invariants | Fix immediately |
| **HIGH** | Risk of production failure or significant maintenance burden | Fix within 1 sprint |
| **MEDIUM** | Degrades developer experience or code quality | Fix within 2-3 sprints |
| **LOW** | Minor inconvenience or cosmetic | Fix when convenient |
| **INFO** | Monitoring item, no immediate action | Track for future |

---

## Register

### TDR-001 — ENGINEERED_FEATURE_NAMES Duplication

| Field | Value |
|---|---|
| **Severity** | MEDIUM |
| **Owner** | src/helix_ids |
| **Impact** | Single source of truth violated. Defined in both `src/helix_ids/data/feature_harmonization.py` and `scripts/training/train_helix_ids_full.py`. The scripts/training copy is stale; all importers should use the src copy. |
| **Effort** | 1-2 hours: remove from training file, update import in run_orchestrator |
| **Recommended Phase** | Phase 20 |

---

### TDR-002 — Partial Delegation Anti-Pattern (17 wrappers)

| Field | Value |
|---|---|
| **Severity** | HIGH |
| **Owner** | HelixFullTrainer |
| **Impact** | 17 methods mix delegate calls with inline orchestration. The trainer still carries phase-management logic that should live in delegate objects. This creates maintenance coupling and test surface duplication. |
| **Effort** | 3-5 days: extract inline logic into PhaseManager, Evaluator, EarlyStoppingManager |
| **Recommended Phase** | Phase 21 |

---

### TDR-003 — Loss Logic Still Inline (9 functions)

| Field | Value |
|---|---|
| **Severity** | MEDIUM |
| **Owner** | HelixFullTrainer / LossRegistry |
| **Impact** | `LossRegistry` exists but the trainer still defines loss functions inline. Prevents independent testing and reuse of individual loss components. |
| **Effort** | 2-3 days: register all 9 inline loss functions in LossRegistry, replace `_apply_loss_regularizations` with registry dispatch |
| **Recommended Phase** | Phase 20 |

---

### TDR-004 — No Frozen Requirements Lockfile

| Field | Value |
|---|---|
| **Severity** | MEDIUM |
| **Owner** | CI/DevOps |
| **Impact** | `pyproject.toml` has version ranges but no `requirements-lock.txt`. Builds can produce different environments depending on when deps were resolved. |
| **Effort** | 1 day: generate lockfile, add CI validation step, update Dockerfile |
| **Recommended Phase** | Phase 20 |

---

### TDR-005 — Trainer __init__ at 392 LOC

| Field | Value |
|---|---|
| **Severity** | LOW |
| **Owner** | HelixFullTrainer |
| **Impact** | Configuration + state init in a single method creates dense dependency on exact attribute layout. Hard to test in isolation. |
| **Effort** | 1-2 days: split into `_init_config()`, `_init_model()`, `_init_data()`, `_init_optimizer()` sub-methods |
| **Recommended Phase** | Phase 22 |

---

### TDR-006 — No Structured JSON Logging

| Field | Value |
|---|---|
| **Severity** | MEDIUM |
| **Owner** | Operations |
| **Impact** | Mix of `print()`, `logging.info()`, `stdout.write()`. No JSON format means log aggregation pipelines (ELK, Loki, Datadog) can't parse training metrics automatically. |
| **Effort** | 1-2 days: add structured logging utility, migrate step-level diagnostics to JSON format |
| **Recommended Phase** | Phase 20 |

---

### TDR-007 — No Performance Regression Tests

| Field | Value |
|---|---|
| **Severity** | MEDIUM |
| **Owner** | CI/QA |
| **Impact** | No benchmarks for training throughput, inference latency, or memory usage. Regressions can go undetected. |
| **Effort** | 2-3 days: add `pytest-benchmark` harness for inference path, track in CI |
| **Recommended Phase** | Phase 21 |

---

### TDR-008 — No Circuit Breaker for Inference Service

| Field | Value |
|---|---|
| **Severity** | LOW |
| **Owner** | Operations |
| **Impact** | `serve_rest.py` has health checks but no circuit breaker for degraded downstream services. Sustained failures cascade. |
| **Effort** | 1 day: add stateful circuit breaker in `monitoring.py` |
| **Recommended Phase** | Phase 22 |

---

### TDR-009 — No ONNX Export Path

| Field | Value |
|---|---|
| **Severity** | LOW |
| **Owner** | Model Export |
| **Impact** | Only PyTorch native export. Limits deployment targets (mobile, edge, non-Python runtimes). |
| **Effort** | 2-3 days: add `torch.onnx.export` path, validate on inference runtime |
| **Recommended Phase** | Phase 23 |

---

### TDR-010 — Pre-commit Hooks Not Configured

| Field | Value |
|---|---|
| **Severity** | LOW |
| **Owner** | DevOps |
| **Impact** | No local gate for lint/format/type checks before commit. CI catches issues but developer feedback loop is slower. |
| **Effort** | Half day: add `.pre-commit-config.yaml` with ruff, mypy, yamllint |
| **Recommended Phase** | Phase 22 |

---

### TDR-011 — No Checkpoint Garbage Collection

| Field | Value |
|---|---|
| **Severity** | LOW |
| **Owner** | Training Pipeline |
| **Impact** | Checkpoints accumulate indefinitely; no `max_checkpoints` or TTL-based cleanup. |
| **Effort** | 1 day: add checkpoint rotation in `_save_checkpoint_if_needed` |
| **Recommended Phase** | Phase 21 |

---

### TDR-012 — No Hypothesis/Property-Based Tests

| Field | Value |
|---|---|
| **Severity** | LOW |
| **Owner** | Testing |
| **Impact** | No fuzz testing for schema validation, loss computation, or model forward pass. Edge cases in input-space combinations may be missed. |
| **Effort** | 2-3 days: add `hypothesis` tests for `schema_contract.validate()`, loss functions, model forward pass |
| **Recommended Phase** | Phase 23 |

---

### TDR-013 — Module-Level Helpers in train_helix_ids_full.py

| Field | Value |
|---|---|
| **Severity** | LOW |
| **Owner** | Training Pipeline |
| **Impact** | `MultiTaskNumpyDataset`, `ClassBalancedIndexSampler`, `FrozenIndexSampler` defined at module level in the trainer file instead of in `data/` subpackage. Clutters the trainer file. |
| **Effort** | 1 day: extract to `scripts/training/data/` |
| **Recommended Phase** | Phase 22 |

---

### TDR-014 — setup_logging Duplicated

| Field | Value |
|---|---|
| **Severity** | LOW |
| **Owner** | Training Pipeline |
| **Impact** | Two copies of `setup_logging`: one in `train_helix_ids_full.py`, one variant in `orchestration/run_orchestrator.py`. Different behavior. |
| **Effort** | 1 day: extract to shared module, unify behavior |
| **Recommended Phase** | Phase 22 |

---

### TDR-015 — 30 Full-Delegation Wrappers Not Removed

| Field | Value |
|---|---|
| **Severity** | LOW |
| **Owner** | HelixFullTrainer |
| **Impact** | 30 thin pass-through wrappers add ~200 LOC of pure boilerplate. Measure: every wrapper is 1-3 lines calling `self._xxx.yyy()`. |
| **Effort** | 2 hours: replace with direct calls in all callers, then remove wrappers |
| **Recommended Phase** | Phase 21 |

---

### TDR-016 — Pre-commit check framework not installed

| Field | Value |
|---|---|
| **Severity** | INFO |
| **Owner** | DevOps |
| **Impact** | No pre-push gate for architecture freeze tests. Developers can commit boundary violations locally. |
| **Effort** | 1 hour: add pre-commit hook for `tests/architecture/` |
| **Recommended Phase** | Phase 23 |

---

### TDR-017 — No .env / Environment-Variable Config Loading

| Field | Value |
|---|---|
| **Severity** | INFO |
| **Owner** | Configuration |
| **Impact** | All config comes from YAML files. No support for environment-variable overrides (e.g. `DATABASE_URL`, `REDIS_HOST`) which are standard for production deployments. |
| **Effort** | 1 day: add `python-dotenv` support in `config_parser.py` |
| **Recommended Phase** | Phase 23 |

---

### TDR-018 — No Auto-Restart for Deployed Service

| Field | Value |
|---|---|
| **Severity** | INFO |
| **Owner** | Operations |
| **Impact** | `serve_rest.py` runs as a single process. No supervisor/process manager configured for auto-restart on crash. |
| **Effort** | 1 day: add systemd service definition or supervisord config |
| **Recommended Phase** | Phase 23 |

---

## Summary

| Severity | Count |
|---|---|
| CRITICAL | 0 |
| HIGH | 1 |
| MEDIUM | 8 |
| LOW | 9 |
| INFO | 3 |
| **Total** | **21** |

### Priority Action Items (HIGH + MEDIUM)

| ID | Severity | Item | Phase |
|---|---|---|---|
| TDR-002 | HIGH | Partial delegation anti-pattern (17 wrappers) | Phase 21 |
| TDR-001 | MEDIUM | ENGINEERED_FEATURE_NAMES duplication | Phase 20 |
| TDR-003 | MEDIUM | Inline loss functions (9) | Phase 20 |
| TDR-004 | MEDIUM | No lockfile | Phase 20 |
| TDR-006 | MEDIUM | No structured logging | Phase 20 |
| TDR-007 | MEDIUM | No performance tests | Phase 21 |
| TDR-008 | MEDIUM | No circuit breaker | Phase 22 |
| TDR-011 | MEDIUM | No checkpoint GC | Phase 21 |

---

## Estimated Total Effort

| Category | Person-Days |
|---|---|
| Immediate (Phase 20) | 5-8 |
| Medium-term (Phase 21) | 7-10 |
| Long-term (Phase 22+) | 5-8 |
| **Total** | **17-26 person-days** |
