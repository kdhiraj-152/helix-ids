# Technical Debt Rebaseline

## Scoring Methodology

- **CRITICAL**: Prevents production deployment or creates severe risk
- **HIGH**: Significantly impacts reliability, security, or maintainability
- **MEDIUM**: Reducing debt has high ROI at moderate cost
- **LOW**: Nice-to-have cleanup, minimal risk

---

## TD-001: Monolithic Training Files

| Field | Value |
|-------|-------|
| **ID** | TD-001 |
| **Severity** | HIGH |
| **Category** | Architecture |
| **Impact** | MAINTENANCE |
| **Description** | 17 files exceed 1000 LOC. `train_helix_ids_full.py` (4,606 LOC) is the largest — contains 73 functions and 3 classes. `trainer_facade.py` (2,818 LOC) contains a 1,929-LOC `HelixFullTrainer` class with 32 methods. |
| **Evidence** | AST analysis: `scripts/training/train_helix_ids_full.py`, `scripts/training/core/trainer_facade.py`, `src/helix_ids/data/multi_dataset_loader.py` |
| **Remediation** | Extract responsibility clusters into dedicated modules: (1) data validation from trainer, (2) evaluation from trainer, (3) governance from orchestrator, (4) data loading into separate loader classes. Target: no file > 1,500 LOC. |
| **Effort** | 3-5 days |
| **Phase** | 24 |

## TD-002: No Retry Policies

| Field | Value |
|-------|-------|
| **ID** | TD-002 |
| **Severity** | HIGH |
| **Category** | Reliability |
| **Impact** | OPERATIONS |
| **Description** | Zero retry patterns anywhere in the codebase. All network I/O, file operations, and API calls are single-shot with no retry logic. |
| **Evidence** | Global search: zero occurrences of `retry`, `backoff`, `max_retries` |
| **Remediation** | Add `tenacity` as a dependency. Add retry decorators to: model download, dataset loading, checkpoint save/load, API calls (serve_rest, staging_gate, traffic_expansion_guard). |
| **Effort** | 4-6 hours |

## TD-003: Only 3 `finally` Blocks

| Field | Value |
|-------|-------|
| **ID** | TD-003 |
| **Severity** | HIGH |
| **Category** | Reliability |
| **Impact** | OPERATIONS |
| **Description** | Across 57K LOC of production code, only 3 `finally:` blocks exist. Resource cleanup is not guaranteed on exception paths. |
| **Evidence** | `structured_logger.py:98`, `visualize_helix_demo.py:265`, `load_test.py:409` |
| **Remediation** | Audit all 30+ `try:` blocks. Add `finally:` clauses for GPU memory cleanup, file handle release, and temporary file deletion. |
| **Effort** | 3-4 hours |

## TD-004: 24 Broad `except Exception` Catches

| Field | Value |
|-------|-------|
| **ID** | TD-004 |
| **Severity** | MEDIUM |
| **Category** | Reliability |
| **Impact** | MAINTAINABILITY |
| **Description** | ~24 occurrences of bare `except Exception` in production code. Masks critical failures like MemoryError and KeyboardInterrupt. |
| **Evidence** | `feature_io.py`, `loader_core.py`, `augmentation.py`, `export.py` |
| **Remediation** | Replace with specific exception types. Add logging in catch blocks to surface suppressed errors. |
| **Effort** | 3-4 hours |

## TD-005: 17 Files > 1000 LOC

| Field | Value |
|-------|-------|
| **ID** | TD-005 |
| **Severity** | MEDIUM |
| **Category** | Architecture |
| **Impact** | MAINTAINABILITY |
| **Description** | 17 files exceed the conventional 1000-LOC maintainability threshold. |
| **Evidence** | AST analysis across all 290 Python files |
| **Remediation** | Extract functions > 100 LOC to dedicated modules. Target: no file > 1,500 LOC, gradually reduce toward 1,000. |
| **Effort** | 5-8 days (distributed across multiple phases) |

## TD-006: 12 God Classes > 500 LOC

| Field | Value |
|-------|-------|
| **ID** | TD-006 |
| **Severity** | MEDIUM |
| **Category** | Architecture |
| **Impact** | TESTABILITY |
| **Description** | 12 classes exceed 500 LOC. `HelixFullTrainer` (1,929 LOC) and `MultiDatasetLoader` (1,443 LOC) are the largest. |
| **Evidence** | AST analysis |
| **Remediation** | Apply SRP decomposition. Extract validation, evaluation, and calibration from HelixFullTrainer. Extract dataset-specific logic from MultiDatasetLoader. |
| **Effort** | 4-6 days |

## TD-007: 58 God Functions > 100 LOC

| Field | Value |
|-------|-------|
| **ID** | TD-007 |
| **Severity** | MEDIUM |
| **Category** | Code Quality |
| **Impact** | MAINTAINABILITY |
| **Description** | 58 functions exceed 100 LOC. `run_orchestration` (900 LOC) and `HelixFullTrainer.__init__` (392 LOC) are the largest. |
| **Evidence** | AST analysis |
| **Remediation** | Break down orchestration functions into sub-functions with single responsibilities. Use composition over sequential logic. |
| **Effort** | 3-5 days |

## TD-008: 474 Line-Too-Long Violations

| Field | Value |
|-------|-------|
| **ID** | TD-008 |
| **Severity** | LOW |
| **Category** | Code Quality |
| **Impact** | READABILITY |
| **Description** | 474 E501 violations (line-too-long). 27 C901 complex-structure violations. 2 E741 ambiguous-variable-name. |
| **Evidence** | `ruff check src/ scripts/ --select C901,E --statistics` |
| **Remediation** | Fix in batches. Start with C901 complexity violations (highest cognitive load). Then fix E501 where lines exceed 120 chars. |
| **Effort** | 4-8 hours |

## TD-009: Missing Test Categorization

| Field | Value |
|-------|-------|
| **ID** | TD-009 |
| **Severity** | MEDIUM |
| **Category** | Testing |
| **Impact** | CI TIME |
| **Description** | Zero `@pytest.mark.unit/integration/e2e/slow/timeout` markers. CI runs all 2,531 tests every time with no ability to tier. |
| **Evidence** | Global search across all test files |
| **Remediation** | Add markers to all test functions. Configure CI to: (1) run unit tests on every push, (2) run integration tests nightly, (3) run e2e tests pre-release. |
| **Effort** | 4-6 hours |

## TD-010: Conftest Single Bottleneck

| Field | Value |
|-------|-------|
| **ID** | TD-010 |
| **Severity** | LOW |
| **Category** | Testing |
| **Impact** | MAINTAINABILITY |
| **Description** | Single root `conftest.py` (217 LOC) with all global fixtures. No subdirectory conftest files. |
| **Evidence** | `tests/conftest.py` only |
| **Remediation** | Split fixtures into subdirectory conftest files per test domain. |
| **Effort** | 2-3 hours |

## TD-011: Production Code Uses `assert`

| Field | Value |
|-------|-------|
| **ID** | TD-011 |
| **Severity** | HIGH |
| **Category** | Security/Reliability |
| **Impact** | PRODUCTION INTEGRITY |
| **Description** | ~40 production `assert` statements that are silently removed under `python -O`. |
| **Evidence** | `trainer_facade.py:133-218`, `deploy.py:147-230` |
| **Remediation** | Replace with explicit `if/raise ValueError(...)` |
| **Effort** | 2-3 hours |

## TD-012: `eval()` in Benchmark Code

| Field | Value |
|-------|-------|
| **ID** | TD-012 |
| **Severity** | HIGH |
| **Category** | Security |
| **Impact** | PRODUCTION INTEGRITY |
| **Description** | `eval(card["architecture"])` in benchmark entry point allows arbitrary code execution. |
| **Evidence** | `scripts/evaluation/benchmark_e2e.py:53` |
| **Remediation** | Replace with `ast.literal_eval()` |
| **Effort** | 30 minutes |

## TD-013: `weights_only=False` in Transfer Learning

| Field | Value |
|-------|-------|
| **ID** | TD-013 |
| **Severity** | HIGH |
| **Category** | Security |
| **Impact** | PRODUCTION INTEGRITY |
| **Description** | `torch.load(f, weights_only=False)` allows pickle-based RCE during model loading. |
| **Evidence** | `transfer_learning.py:1185` |
| **Remediation** | Change to `weights_only=True`. Verify checkpoint loading still works. |
| **Effort** | 1-2 hours |

## TD-014: No E2E Integration Test

| Field | Value |
|-------|-------|
| **ID** | TD-014 |
| **Severity** | HIGH |
| **Category** | Testing |
| **Impact** | RELEASE CONFIDENCE |
| **Description** | No end-to-end pipeline test exercising data → training → eval → export → deploy. |
| **Evidence** | Test inventory analysis |
| **Remediation** | Add smoke E2E test with tiny dataset and minimal training loop. |
| **Effort** | 4-6 hours |

## TD-015: No External Alerting

| Field | Value |
|-------|-------|
| **ID** | TD-015 |
| **Severity** | HIGH |
| **Category** | Operations |
| **Impact** | INCIDENT RESPONSE |
| **Description** | Zero external notification integrations (Slack, PagerDuty, webhook, email). All alerts stay in-memory. |
| **Evidence** | Global search for notification patterns |
| **Remediation** | Add notification service interface + Prometheus Alertmanager. |
| **Effort** | 5-8 hours |

## TD-016: `session_logs/` in Repository

| Field | Value |
|-------|-------|
| **ID** | TD-016 |
| **Severity** | LOW |
| **Category** | Operations |
| **Impact** | REPOSITORY HYGIENE |
| **Description** | Chat session logs stored in repo root. Contains credential setup patterns. |
| **Evidence** | `session_logs/` directory |
| **Remediation** | Add to `.gitignore`. Remove existing logs from git tracking. |
| **Effort** | 10 minutes |

## TD-017: Empty Result Directories

| Field | Value |
|-------|-------|
| **ID** | TD-017 |
| **Severity** | LOW |
| **Category** | Operations |
| **Impact** | REPOSITORY HYGIENE |
| **Description** | 3 empty directories: `results/gates`, `results/manifests`, `results/metrics` |
| **Evidence** | `find . -type d -empty` |
| **Remediation** | Remove empty directories or add `.gitkeep` if placeholder needed. |
| **Effort** | 10 minutes |

## TD-018: Lockfiles Generated by Different Tools

| Field | Value |
|-------|-------|
| **ID** | TD-018 |
| **Severity** | LOW |
| **Category** | Dependencies |
| **Impact** | BUILD REPRODUCIBILITY |
| **Description** | `requirements-lock.txt` generated by `uv`, `requirements-all-lock.txt` and `requirements-dev-lock.txt` generated by `pip-compile`. Different tools may produce different resolutions. |
| **Evidence** | Lockfile headers |
| **Remediation** | Standardize on one tool (prefer `uv` for speed). Re-generate all lockfiles with the same tool. |
| **Effort** | 2 hours |

## TD-019: Unpinned Requirements.in

| Field | Value |
|-------|-------|
| **ID** | TD-019 |
| **Severity** | LOW |
| **Category** | Dependencies |
| **Impact** | BUILD REPRODUCIBILITY |
| **Description** | All 7 runtime deps in `requirements.in` use `>=` ranges (e.g., `torch>=2.0.0`). While lockfiles pin exact versions, the source `.in` file would produce different results on a fresh `pip-compile` run. |
| **Evidence** | `cat requirements.in` |
| **Remediation** | Pin minimum versions more tightly based on tested ranges. Add upper bounds where known. |
| **Effort** | 1 hour |
