# Dependency Graph

Generated: Phase 19 Architecture Freeze

## Summary

- **Total nodes:** 256
- **Total edges:** 590
- **Cycle detection:** No cycles detected

## Package-Level Import Map (src/helix_ids)

| Source Package | Target Package | Files |
|---|---|---|
| src.helix_ids.config | src.helix_ids.config.platform_loader | 1 |
| src.helix_ids.contracts | src.helix_ids.contracts.immutable_constants | 1 |
| src.helix_ids.contracts | src.helix_ids.contracts.schema_contract | 1 |
| src.helix_ids.data | src.helix_ids.contracts.attack_taxonomy | 2 |
| src.helix_ids.data | src.helix_ids.governance | 1 |
| src.helix_ids.governance | src.helix_ids | 2 |
| src.helix_ids.governance | src.helix_ids.contracts | 2 |
| src.helix_ids.governance | src.helix_ids.contracts.schema_contract | 1 |
| src.helix_ids.governance | src.helix_ids.governance.fingerprinting | 1 |
| src.helix_ids.governance | src.helix_ids.governance.parameters | 2 |
| src.helix_ids.governance | src.helix_ids.governance.provenance | 1 |
| src.helix_ids.models | src.helix_ids | 1 |
| src.helix_ids.models | src.helix_ids.contracts.attack_taxonomy | 1 |
| src.helix_ids.models | src.helix_ids_full | 1 |
| src.helix_ids.models.adaptation | src.helix_ids.contracts.schema_contract | 1 |
| src.helix_ids.models.adaptation | src.helix_ids.data.unified_loader | 1 |
| src.helix_ids.models.adaptation | src.helix_ids.governance | 1 |
| src.helix_ids.models.adaptation | src.helix_ids.governance.provenance | 1 |
| src.helix_ids.models.adaptation | src.helix_ids.models.adaptation.combined_da | 1 |
| src.helix_ids.models.adaptation | src.helix_ids.models.adaptation.coral_loss | 1 |
| src.helix_ids.models.adaptation | src.helix_ids.models.adaptation.dann | 1 |
| src.helix_ids.models.adaptation | src.helix_ids.models.adaptation.label_aware_da | 1 |
| src.helix_ids.models.adaptation | src.helix_ids.models.adaptation.mmd_loss | 1 |
| src.helix_ids.models.adaptation | src.helix_ids.utils.export | 1 |
| src.helix_ids.operations | src.helix_ids.contracts | 1 |
| src.helix_ids.operations | src.helix_ids.contracts.schema_contract | 2 |
| src.helix_ids.operations | src.helix_ids.governance | 2 |
| src.helix_ids.operations | src.helix_ids.governance.provenance | 1 |
| src.helix_ids.operations | src.helix_ids.models.full | 1 |
| src.helix_ids.operations | src.helix_ids.utils.export | 1 |
| src.helix_ids.utils | src.helix_ids | 1 |
| src.helix_ids.utils | src.helix_ids.contracts | 1 |
| src.helix_ids.utils | src.helix_ids.contracts.attack_taxonomy | 2 |
| src.helix_ids.utils | src.helix_ids.contracts.schema_contract | 1 |
| src.helix_ids.utils | src.helix_ids.governance | 1 |
| src.helix_ids.utils | src.helix_ids.governance.parameters | 1 |
| src.helix_ids.utils | src.helix_ids.governance.provenance | 1 |

## Cross-Boundary Imports (scripts -> src)

| Source File | Target Module |
|---|---|
| scripts/data/process_cicids.py | helix_ids.data.feature_harmonization |
| scripts/data/process_cicids.py | helix_ids.data.learnability_contract |
| scripts/data/process_cicids.py | helix_ids.data.multi_dataset_loader |
| scripts/data/process_cicids.py | helix_ids.data.unified_loader |
| scripts/data/process_nsl_kdd.py | helix_ids.data.feature_harmonization |
| scripts/data/process_nsl_kdd.py | helix_ids.data.learnability_contract |
| scripts/data/process_nsl_kdd.py | helix_ids.data.multi_dataset_loader |
| scripts/data/process_unsw_nb15.py | helix_ids.data.feature_harmonization |
| scripts/data/process_unsw_nb15.py | helix_ids.data.learnability_contract |
| scripts/data/process_unsw_nb15.py | helix_ids.data.multi_dataset_loader |
| scripts/evaluation/benchmark_e2e.py | helix_ids.governance.determinism |
| scripts/evaluation/benchmark_e2e.py | helix_ids.governance.entrypoint |
| scripts/evaluation/benchmark_e2e.py | helix_ids.governance.parameters |
| scripts/evaluation/benchmark_e2e.py | helix_ids.governance.promotion |
| scripts/evaluation/benchmark_e2e.py | helix_ids.governance.run_registry |
| scripts/evaluation/benchmark_e2e.py | helix_ids.utils.metrics |
| scripts/evaluation/benchmarks.py | helix_ids.contracts |
| scripts/evaluation/benchmarks.py | helix_ids.data.feature_harmonization |
| scripts/evaluation/benchmarks.py | helix_ids.governance |
| scripts/evaluation/benchmarks.py | helix_ids.governance.ast_validator |
| scripts/evaluation/benchmarks.py | helix_ids.governance.lifecycle_verifier |
| scripts/evaluation/benchmarks.py | helix_ids.governance.parameters |
| scripts/evaluation/benchmarks.py | helix_ids.governance.provenance |
| scripts/evaluation/holdout_evaluation.py | helix_ids.governance.determinism |
| scripts/evaluation/holdout_evaluation.py | helix_ids.governance.entrypoint |
| scripts/evaluation/holdout_evaluation.py | helix_ids.governance.parameters |
| scripts/evaluation/holdout_evaluation.py | helix_ids.governance.promotion |
| scripts/evaluation/holdout_evaluation.py | helix_ids.governance.run_registry |
| scripts/evaluation/holdout_evaluation.py | helix_ids.utils.metrics |
| scripts/evaluation/test_phase3_smoke.py | helix_ids.config.helix_full_config |
| scripts/evaluation/test_phase3_smoke.py | helix_ids.data.feature_harmonization |
| scripts/evaluation/test_phase3_smoke.py | helix_ids.models.full |
| scripts/operations/export_inference_bundle.py | helix_ids.operations.inference_runtime |
| scripts/operations/freeze_baseline.py | helix_ids.operations.baseline_freeze |
| scripts/operations/serve_rest.py | helix_ids.operations.inference_runtime |
| scripts/operations/stress_validate_baseline.py | helix_ids.operations.inference_runtime |
| scripts/operations/stress_validate_baseline.py | helix_ids.operations.monitoring |
| scripts/smoke_save_checkpoint.py | helix_ids.utils.callbacks |
| scripts/training/adversarial_training.py | helix_ids.contracts.schema_contract |
| scripts/training/adversarial_training.py | helix_ids.governance |
| scripts/training/adversarial_training.py | helix_ids.utils.export |
| scripts/training/core/trainer_factory.py | helix_ids.config.helix_full_config |
| scripts/training/core/trainer_factory.py | helix_ids.models.full |
| scripts/training/core/trainer_state.py | helix_ids.config.helix_full_config |
| scripts/training/core/trainer_state.py | helix_ids.models.full |
| scripts/training/diagnostics/cluster_analyzer.py | helix_ids.data.datasets |
| scripts/training/diagnostics/rep_diagnostics.py | helix_ids.data.geometric_representation_fixes |
| scripts/training/evaluation/evaluation_orchestrator.py | helix_ids.models.full |
| scripts/training/evaluation/evaluator.py | helix_ids.models.full |
| scripts/training/governance/orchestrator.py | helix_ids.models.full |
| scripts/training/orchestration/__init__.py | helix_ids.config.helix_full_config |
| scripts/training/orchestration/config_parser.py | helix_ids.config.helix_full_config |
| scripts/training/orchestration/governance_pipeline.py | helix_ids.governance.promotion |
| scripts/training/orchestration/governance_pipeline.py | helix_ids.governance.run_registry |
| scripts/training/orchestration/run_orchestrator.py | helix_ids.data.feature_harmonization |
| scripts/training/orchestration/run_orchestrator.py | helix_ids.data.learnability_contract |
| scripts/training/orchestration/run_orchestrator.py | helix_ids.governance.determinism |
| scripts/training/orchestration/run_orchestrator.py | helix_ids.models.full |
| scripts/training/orchestration/run_orchestrator.py | helix_ids.data.learnability_contract |
| scripts/training/orchestration/run_orchestrator.py | helix_ids.data.feature_harmonization |
| scripts/training/prepare_canonical_artifacts.py | helix_ids.contracts |
| scripts/training/prepare_canonical_artifacts.py | helix_ids.data.feature_harmonization |
| scripts/training/prepare_canonical_artifacts.py | helix_ids.data.multi_dataset_loader |
| scripts/training/train_edge_models.py | helix_ids.contracts.schema_contract |
| scripts/training/train_edge_models.py | helix_ids.governance |
| scripts/training/train_edge_models.py | helix_ids.utils.export |
| scripts/training/train_helix_ids_full.py | helix_ids.config.helix_full_config |
| scripts/training/train_helix_ids_full.py | helix_ids.contracts |
| scripts/training/train_helix_ids_full.py | helix_ids.data.learnability_contract |
| scripts/training/train_helix_ids_full.py | helix_ids.governance |
| scripts/training/train_helix_ids_full.py | helix_ids.governance.determinism |
| scripts/training/train_helix_ids_full.py | helix_ids.governance.entrypoint |
| scripts/training/train_helix_ids_full.py | helix_ids.governance.parameters |
| scripts/training/train_helix_ids_full.py | helix_ids.models.full |
| scripts/training/train_helix_ids_full.py | helix_ids.utils.export |
| scripts/training/train_helix_ids_full.py | helix_ids.utils.metrics |
| scripts/training/train_helix_ids_full.py | helix_ids.data.feature_harmonization |
| scripts/training/train_multidataset.py | helix_ids.contracts.schema_contract |
| scripts/training/train_multidataset.py | helix_ids.governance |
| scripts/training/train_multidataset.py | helix_ids.governance.determinism |
| scripts/training/train_multidataset.py | helix_ids.governance.entrypoint |
| scripts/training/train_multidataset.py | helix_ids.governance.parameters |
| scripts/training/train_multidataset.py | helix_ids.governance.promotion |
| scripts/training/train_multidataset.py | helix_ids.governance.run_registry |
| scripts/training/train_multidataset.py | helix_ids.utils.export |
| scripts/training/train_multidataset.py | helix_ids.utils.metrics |
| scripts/training/train_multidataset.py | helix_ids.utils.metrics |
| scripts/training/train_unified_rebalanced.py | helix_ids.data.loader_core |
| scripts/training/train_unified_rebalanced.py | helix_ids.models.adaptation.transfer_learning |
| scripts/training/train_unsw_only.py | helix_ids.contracts.schema_contract |
| scripts/training/train_unsw_only.py | helix_ids.governance |
| scripts/training/train_unsw_only.py | helix_ids.utils.export |
| scripts/training/train_unsw_only.py | helix_ids.config.helix_full_config |
| scripts/training/train_unsw_only.py | helix_ids.data.feature_harmonization |
| scripts/training/train_unsw_only.py | helix_ids.data.multi_dataset_loader |
| scripts/training/train_unsw_only.py | helix_ids.models.full |
| scripts/training/validation/calibrator.py | helix_ids.models.full |
| scripts/training/validation/calibrator.py | helix_ids.utils.metrics |
| scripts/training/validation/evaluator.py | helix_ids.models.full |

## Ownership Domains

- **Core Library (src.helix_ids)**: 67 files
- **Training Pipeline (scripts.training)**: 57 files
- **Operations (scripts.operations)**: 7 files
- **Deployment (scripts.deployment)**: 1 files
- **Evaluation (scripts.evaluation)**: 4 files
- **Data Processing (scripts.data)**: 4 files
- **CI (scripts.ci)**: 13 files
- **Tests**: 100 files

## Reverse Dependencies

**RESULT:** 0 reverse dependencies (src -> scripts). Invariant holds.
