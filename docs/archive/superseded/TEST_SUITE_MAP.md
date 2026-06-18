# Phase 23 — Test Suite Map

> Generated: 2026-06-18
> Inventory of all test suites by type, location, and purpose.

---

## Test Suite Overview

| Test Type | Count | Location(s) | CI Stage |
|-----------|-------|-------------|----------|
| Unit | 50+ | `tests/`, `tests/test_*`, `tests/training/` | `ci.yml` |
| Integration | 10+ | `tests/test_operations/`, `tests/training/` | `ci.yml` |
| Architecture | 6 | `tests/architecture/` | `architecture.yml` |
| Governance | 10 | `tests/test_governance/` | `architecture.yml` |
| Property-based | 1 | `tests/` (`test_property_based.py`) | `ci.yml` |
| Fuzz | 1 | `tests/` (`test_fuzz.py`) | `nightly.yml` |
| Chaos | 1 | `tests/` (`test_checkpoint_chaos.py`) | `nightly.yml` |
| Fault injection | 1 | `tests/` (`test_fault_injection.py`) | `nightly.yml` |
| Memory leak | 1 | `tests/` (`test_memory_leak_detection.py`) | `nightly.yml` |
| Load | 1 | `scripts/benchmarks/load_test.py` | Manual |
| Soak | 3 | `scripts/benchmarks/soak_*.py` | Manual (Phase 25) |
| **Total** | **~85+** | | |

---

## Detailed Test Inventory

### Unit Tests (`tests/` root — 50+ files)

| File | Type | What It Tests |
|------|------|---------------|
| `test_adversarial_robustness.py` | Unit | Adversarial training robustness |
| `test_benchmark_formalization.py` | Unit | Benchmark config formalization |
| `test_benchmark_output_validator.py` | Unit | Benchmark output validation |
| `test_callbacks.py` | Unit | ModelCheckpoint callback |
| `test_check_performance_regression.py` | Unit | Performance regression gate |
| `test_checkpoint_contracts.py` | Unit | Checkpoint contract invariants |
| `test_classifier.py` | Unit | Classifier model |
| `test_combined_da.py` | Unit | Combined domain adaptation |
| `test_coral_loss.py` | Unit | CORAL loss |
| `test_critical_pipeline_invariants.py` | Unit | Pipeline invariant checks |
| `test_data_integrity_guards.py` | Unit | Data integrity guards |
| `test_data_loading.py` | Unit | Data loading |
| `test_dataset_corruption.py` | Unit | Dataset corruption handling |
| `test_e2e_smoke.py` | Unit/Smoke | End-to-end smoke |
| `test_entropy_deployment_validation.py` | Unit | Entropy deployment validation |
| `test_export.py` | Unit | Export utilities |
| `test_export_contract.py` | Unit | Export contract verification |
| `test_export_quantization_deployment.py` | Unit | Quantization export |
| `test_export_unit.py` | Unit | Export unit tests |
| `test_fault_injection.py` | Fault/Chaos | System fault tolerance |
| `test_feature_engineering.py` | Unit | Feature engineering |
| `test_feature_harmonization.py` | Unit | Feature harmonization |
| `test_fn_tracker.py` | Unit | FN tracker |
| `test_fuzz.py` | Fuzz | Fuzz testing |
| `test_helix_ids_unit.py` | Unit | Helix IDS model |
| `test_label_aware_da.py` | Unit | Label-aware DA |
| `test_lifecycle_verifier.py` | Unit | Lifecycle verifier |
| `test_loss_unit.py` | Unit | Loss functions |
| `test_memory_leak_detection.py` | Memory Leak | Memory leak detection |
| `test_mmd_loss.py` | Unit | MMD loss |
| `test_model_inference.py` | Unit | Model inference |
| `test_per_class_metrics.py` | Unit | Per-class metrics |
| `test_prepare_canonical_artifacts.py` | Unit | Canonical artifact preparation |
| `test_preprocessing.py` | Unit | Preprocessing pipeline |
| `test_property_based.py` | Property-based | Hypothesis property tests |
| `test_provenance.py` | Unit | Provenance tracking |
| `test_regression_taxonomy_contracts.py` | Unit | Regression taxonomy |
| `test_regression_threat_weights.py` | Unit | Threat-weighted metrics |
| `test_runtime_invariants.py` | Unit | Runtime invariants |
| `test_runtime_monitoring_hardening.py` | Unit | Monitoring hardening |
| `test_schema_contract.py` | Unit | Schema contract |
| `test_schema_registry_validation.py` | Unit | Schema registry validation |
| `test_training_direct_adaptation_eval.py` | Unit/Integration | Direct adaptation eval |
| `test_training_pipeline_v2.py` | Integration | Training pipeline |
| `test_transfer_learning_da_schedule.py` | Unit | Transfer learning DA schedule |
| `test_validation_artifacts.py` | Unit | Validation artifacts |
| `test_validation_calibrator.py` | Unit | Validation calibrator |
| `test_validation_evaluator.py` | Unit | Validation evaluator |

### Architecture Tests (`tests/architecture/` — 6 files)

| File | What It Verifies |
|------|------------------|
| `test_architecture_freeze.py` | Architecture has not changed since freeze |
| `test_architecture_lockdown.py` | Package import patterns are locked |
| `test_dependency_cycles.py` | No circular dependencies |
| `test_dependency_lockdown.py` | Dependency graph matches reference |
| `test_no_reverse_dependencies.py` | No `scripts`→`src` reverse imports |
| `test_trainer_boundary.py` | Trainer LOC/method limits |

### Governance Tests (`tests/test_governance/` — 10 files)

| File | What It Verifies |
|------|------------------|
| `test_ast_validator.py` | AST-based policy enforcement |
| `test_enforcement_completeness.py` | All rules have enforcement |
| `test_fingerprinting.py` | Fingerprinting correctness |
| `test_integration_enforcement.py` | Integration enforcement |
| `test_legacy_policy.py` | Legacy artifact handling |
| `test_metrics_contract.py` | Metrics contract validation |
| `test_nested_schema_validation.py` | Nested schema validation |
| `test_orchestrator_runtime.py` | Governance orchestrator runtime |
| `test_promotion.py` | Promotion gate logic |
| `test_promotion_parser.py` | Promotion log parsing |
| `test_run_registry.py` | Run registry |
| `test_validate_schema_registry.py` | Schema registry validation |

### Operations Tests (`tests/test_operations/` — 8 files)

| File | What It Verifies |
|------|------------------|
| `test_baseline_freeze.py` | Baseline freeze logic |
| `test_deployment_manifest_injection.py` | Deployment manifest |
| `test_inference_runtime.py` | Inference runtime |
| `test_monitoring.py` | Monitoring system |
| `test_serve_rest_metrics.py` | REST serving metrics |
| `test_staging_gate_check.py` | Staging gate |
| `test_structured_logger.py` | Structured logging |
| `test_traffic_expansion_guard.py` | Traffic expansion guard |

### Data Tests (`tests/test_data/` — 5 files)

| File | What It Verifies |
|------|------------------|
| `test_diagnostic_contract.py` | Diagnostic contract |
| `test_phase1_harmonization.py` | Phase 1 harmonization |
| `test_root_cause_reducer.py` | Root cause reducer |
| `test_unified_loader.py` | Unified data loader |
| `test_unsw_learnability_contract.py` | UNSW learnability contract |

### Model Tests (`tests/test_models/` — 3 files)

| File | What It Verifies |
|------|------------------|
| `test_helix_full.py` | HelixFull model |
| `test_helix_ids.py` | HelixIDS model |
| `test_loss.py` | Loss functions |

### Utility Tests (`tests/test_utils/` — 1 file)

| File | What It Verifies |
|------|------------------|
| `test_metrics.py` | Metrics computation |

### Training Component Tests (`tests/training/` — 9 files)

| File | What It Verifies |
|------|------------------|
| `test_checkpoint_recovery.py` | Checkpoint recovery manager |
| `test_evaluation_orchestrator.py` | Eval orchestration |
| `test_execution_batch_processor.py` | Batch processing |
| `test_execution_epoch_runner.py` | Epoch runner |
| `test_execution_orchestrator.py` | Training orchestration |
| `test_execution_warmup_manager.py` | Warmup manager |
| `test_recovery_manager.py` | Recovery manager |
| `test_trainer_facade.py` | Trainer facade |
| `test_trainer_state.py` | Trainer state |
| `test_validation_orchestrator.py` | Validation orchestration |

### Extracted Component Tests (`tests/test_training/` — 9 files)

| File | What It Verifies |
|------|------------------|
| `test_extracted_data_components.py` | Extracted data components |
| `test_extracted_diagnostics.py` | Extracted diagnostics |
| `test_extracted_evaluation.py` | Extracted evaluation |
| `test_extracted_governance.py` | Extracted governance |
| `test_extracted_losses.py` | Extracted losses |
| `test_extracted_orchestration.py` | Extracted orchestration |
| `test_extracted_phase_orchestration.py` | Extracted phase orchestration |
| `test_extracted_representation.py` | Extracted representation |
| `test_extracted_scheduler.py` | Extracted scheduler |

### Other Tests

| File | Location | What It Verifies |
|------|----------|------------------|
| `test_environment_loader.py` | `tests/config/` | Config environment loading |
| `test_checkpoint_chaos.py` | `tests/` | Checkpoint chaos testing |
| `test_circuit_breaker.py` | `tests/operations/` | Circuit breaker |
| `test_restart_manager.py` | `tests/operations/` | Restart manager |
| `test_structured_logging.py` | `tests/operations/` | Structured logging |

---

## Test Grouping Assessment

### Strengths
- Architecture, governance, operations, data, and model tests are well-grouped into subdirectories.
- Training tests have clear separation between component tests (`tests/training/`) and extracted component tests (`tests/test_training/`).
- CI workflow alignment is mostly correct.

### Weaknesses
- **50+ tests at `tests/` root** — These are a mix of unit, integration, fuzz, chaos, and property-based tests with no organizational structure.
- **Poor separation of test types** — Chaos, fuzz, memory leak, and fault injection tests are at the same level as unit tests, making it hard to run targeted CI stages.
- **Naming collision risk** — Many tests at root level could logically live in subdirectories but don't.

### Recommendation

Migrate root-level tests into subdirectories:

```
tests/
├── unit/                    # Pure unit tests (fast, no I/O)
├── integration/             # Integration tests (loader, pipeline)
├── stability/               # Chaos, fuzz, fault injection, memory leak
└── smoke/                   # E2E smoke tests
```

This is a medium-effort restructuring (~2-3 hours for migration + import fixup). The `pytest` configuration would need `--ignore` or updated `testpaths`.

### CI Stage Alignment

| CI Workflow | Tests Included | Current fitness |
|-------------|---------------|-----------------|
| `ci.yml` (fast) | Unit + integration (~60 tests) | Good fit |
| `quality.yml` (medium) | Contract + governance | Good fit |
| `architecture.yml` | Architecture + dependency checks | Good fit |
| `nightly.yml` (slow) | Fuzz, chaos, memory leak, fault injection | Good fit — keep these separate |
| Manual | Load, soak | Good fit |
