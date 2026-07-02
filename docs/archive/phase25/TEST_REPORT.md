# Test Report — Phase 25C

Generated: 2026-06-21

**Tool:** pytest (tests/test_data/ + tests/test_operations/)

## Summary

**172 passed** in 11.51s

## Test Breakdown

| Test suite | Count | Result |
|------------|-------|--------|
| `test_diagnostic_contract` | 6 | PASS |
| `test_phase1_harmonization` | 35 | PASS |
| `test_root_cause_reducer` | 32 | PASS |
| `test_unified_loader` | 5 | PASS |
| `test_unsw_learnability_contract` | 3 | PASS |
| `test_baseline_freeze` | 1 | PASS |
| `test_deployment_manifest_injection` | 6 | PASS |
| `test_inference_runtime` | 13 | PASS |
| `test_monitoring` | 3 | PASS |
| `test_serve_rest_metrics` | 1 | PASS |
| `test_staging_gate_check` | 3 | PASS |
| `test_structured_logger` | 57 | PASS |
| `test_traffic_expansion_guard` | 3 | PASS |

## TON-IoT Related Tests

The `test_phase1_harmonization` suite (35 tests) covers multi-dataset loading including TON-IoT harmonization. All pass.

## Verdict

**PASS** — All tests pass, including TON-IoT harmonization tests.
