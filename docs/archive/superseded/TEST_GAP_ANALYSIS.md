# Test Gap Analysis

## Test Inventory

| Metric | Value |
|--------|-------|
| Total tests collected | 2,531 |
| Test files | 72 |
| Test LOC | 44,208 |
| Passing tests | 2,512 |
| Skipped tests | 16 |
| Failing tests | 0 |
| Conftest files | 1 (root only) |
| Root conftest LOC | 217 |

## Test Distribution by Subdirectory

| Directory | Test Files | Coverage |
|-----------|------------|----------|
| `tests/` (root) | ~45 | Broad unit coverage |
| `tests/training/` | 18 | Training pipeline |
| `tests/test_governance/` | 12 | Governance contracts |
| `tests/test_operations/` | 8 | Operations/runtime |
| `tests/architecture/` | 6 | Architecture invariants |
| `tests/test_data/` | 5 | Data pipeline |
| `tests/test_models/` | 3 | Model components |
| `tests/test_training/` | 2 | Training integration |
| `tests/operations/` | 3 | Operations |
| `tests/test_utils/` | 1 | Utilities |
| `tests/config/` | 1 | Configuration |
| `tests/fixtures/` | 0 (fixture files only) | |

## Finding T-01: No Test Categorization Markers

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Category** | Testing |
| **Evidence** | Zero `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.e2e`, `@pytest.mark.slow`, `@pytest.mark.flaky`, or `@pytest.mark.timeout` markers exist in any test file |
| **Risk** | Cannot run targeted test subsets. CI runs all 2,531 tests every time. Slow/flaky tests cannot be quarantined. No ability to run "smoke tests" (unit only) or "full regression" (all). |
| **Remediation** | Add `@pytest.mark.unit`, `@pytest.mark.integration`, and `@pytest.mark.e2e` markers to all tests. Add `@pytest.mark.slow` to tests > 5 seconds. Configure CI to run unit tests first, then integration/e2e separately. |
| **Effort** | 4-6 hours |
| **Status** | UNRESOLVED |

## Finding T-02: No Property-Based Tests (Except One File)

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Category** | Testing |
| **Evidence** | Only 1 test file (`tests/test_data/test_diagnostic_contract.py`) uses Hypothesis `@given`. Zero tests use `property_test`, `PropertyTest`, or `prop_test` naming. Zero property-based tests for metrics, loss functions, or data transformations. |
| **Risk** | Critical invariants (metric monotonicity, loss bounds, feature transformation consistency) are only tested with example-based tests. Edge cases and corner states are missed. |
| **Remediation** | Add property-based tests for: (1) loss functions — assert loss ≥ 0 for all inputs, (2) metrics — assert macro_f1 ∈ [0,1] for all label distributions, (3) feature engineering — assert output shape matches input, (4) schema contracts — assert all feature names present in data. |
| **Effort** | 6-10 hours |
| **Status** | UNRESOLVED |

## Finding T-03: No Cooldown/Timeout Markers

| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **Category** | Testing |
| **Evidence** | Zero `@pytest.mark.timeout` markers. Zero timeout configuration in `pyproject.toml` pytest config. |
| **Risk** | A single hung test blocks the entire CI pipeline for the default timeout (often 30+ minutes). No early detection of infinite loops or deadlocks. |
| **Remediation** | Add `pytest-timeout` plugin. Set a global timeout of 300s in pyproject.toml. Add per-test timeouts for known long-running tests. |
| **Effort** | 1 hour |
| **Status** | UNRESOLVED |

## Finding T-04: Single Conftest Bottleneck

| Field | Value |
|-------|-------|
| **Severity** | LOW |
| **Category** | Testing |
| **Evidence** | Single `tests/conftest.py` (217 LOC). No conftest files in any subdirectory. |
| **Risk** | All fixtures are global, creating implicit test coupling. A fixture change in the root conftest affects all tests. Cannot scope fixtures to specific subdirectories. |
| **Remediation** | Split fixtures into subdirectory-level conftest files. Move data-loading fixtures to `tests/test_data/conftest.py`. Move training fixtures to `tests/training/conftest.py`. Keep only shared fixtures (CLI helpers, type aliases) in root conftest. |
| **Effort** | 2-3 hours |
| **Status** | UNRESOLVED |

## Finding T-05: No Integration Test for End-to-End Pipeline

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Category** | Testing |
| **Evidence** | No test file exercises the full pipeline from data loading → training → evaluation → export → deployment. `benchmark_e2e.py` is a benchmark script, not a test. The architecture tests verify import contracts, not runtime behavior. |
| **Risk** | Integration issues between pipeline stages are only discovered during actual runs or in production. Breaking changes to data contracts between stages go undetected. |
| **Remediation** | Add a smoke E2E test that: (1) loads a tiny dataset, (2) runs a minimal training loop (1-3 epochs), (3) evaluates, (4) exports to ONNX, (5) deploys via serve_rest and sends a test request. Mark as `@pytest.mark.e2e` and `@pytest.mark.slow`. |
| **Effort** | 4-6 hours |
| **Status** | UNRESOLVED |

## Finding T-06: No Negative Tests

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Category** | Testing |
| **Evidence** | No test files explicitly test error handling paths (invalid inputs, corrupted data, missing files, wrong shapes, unexpected types). |
| **Risk** | Error handling code is untested. Defensive guards (`if x is None: raise ...`) may themselves be buggy. Production failures in edge cases go undetected. |
| **Remediation** | Add negative tests for: (1) corrupted or truncated data files, (2) wrong feature dimensions, (3) missing label columns, (4) NaN/inf input values, (5) empty datasets, (6) unsupported model variants. Use `pytest.raises()` to verify expected exceptions. |
| **Effort** | 4-6 hours |
| **Status** | UNRESOLVED |

## Finding T-07: Mutation Testing Well-Structured

| Field | Value |
|-------|-------|
| **Severity** | INFO (Positive) |
| **Category** | Testing |
| **Evidence** | 15 cosmic-ray configs with 5 mutation operators each. Operators: NumberReplacer, ReplaceComparisonOperator (Eq_NotEq, Gt_Lt), ReplaceAndWithOr, AddNot. Local distributor with 120s timeout. 7 modules previously completed (3,673 mutants, 100% killed). |
| **Risk** | N/A — this is a strength. Mutation testing is well-configured with targeted configs per module. |
| **Remediation** | Continue expanding coverage to remaining modules. Ensure mutation tests run in CI periodically. |
| **Effort** | Ongoing |
| **Status** | ACCEPTABLE |

## Test Coverage Estimation

Based on test file count and test collection:

| Layer | Estimated Coverage | Assessment |
|-------|-------------------|------------|
| Data loading | ⚠️ Partial | Loaders tested not all edge cases |
| Feature engineering | ✅ Good | Feature harmonization + engineering tests |
| Models/SRP | ✅ Good | Forward pass, loss, architecture |
| Training loop | ⚠️ Partial | Trainer logic, not multi-epoch orchestration |
| Evaluation | ❌ Minimal | No eval-specific test files |
| Export | ✅ Good | ONNX export + quantization tested |
| Deployment | ❌ Minimal | No deploy-specific tests |
| Governance | ✅ Excellent | 12 test files for governance contracts |
| Operations | ✅ Good | Runtime + inference tested |
| Architecture | ✅ Good | Import invariants, boundary enforcement |

## Test Scorecard

| Check | Status | Note |
|-------|--------|------|
| Unit test markers | ❌ FAIL | Missing across all 2,531 tests |
| Integration test markers | ❌ FAIL | Missing |
| E2E test markers | ❌ FAIL | Missing |
| Slow test markers | ❌ FAIL | Missing |
| Timeout markers | ❌ FAIL | Missing |
| Property-based tests | ⚠️ WARNING | Only 1 file uses Hypothesis |
| Negative tests | ❌ FAIL | None found |
| E2E pipeline test | ❌ FAIL | No end-to-end runtime test |
| Conftest organization | ⚠️ WARNING | Single 217 LOC bottleneck |
| Mutation testing | ✅ PASS | 15 configs, 100% killed |
