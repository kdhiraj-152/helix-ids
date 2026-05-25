# HELIX Forensic Canonicalization Audit

This document is an evidence-first forensic audit of the repository. It is built from executable code paths, artifacts, checkpoints, metrics, logs, and tests. It does not rely on README claims or design intent.

Canonical decision: the only acceptable canonical feature schema is the 19-feature harmonized contract defined in [src/helix_ids/data/feature_harmonization.py](src/helix_ids/data/feature_harmonization.py). Everything else is treated as deprecated or invalid.

---

## SECTION 1 - Executive Reality Check

### What the project actually is
- A multi-dataset intrusion detection research prototype with multiple overlapping generations of data pipelines, model architectures, and deployment stories. The only fully wired training/inference path uses the HelixIDS-Full model with 17 audited invariant features and a 7-class family head, as defined in [src/helix_ids/models/helix_ids_full.py](src/helix_ids/models/helix_ids_full.py) and trained via [scripts/training/train_helix_ids_full.py](scripts/training/train_helix_ids_full.py).
- The canonical 19-feature contract exists in [src/helix_ids/data/feature_harmonization.py](src/helix_ids/data/feature_harmonization.py) but is not actually respected by the processed artifacts or checkpoints.

### What actually works
- The helix_full training script can produce checkpoints and evaluation artifacts, but only a single seed (1337) passes post-training macro-F1 guards; other seeds fail the governance gate with run_exit_code=1 ([results/helix_full/training_results_seed42.json](results/helix_full/training_results_seed42.json), [results/helix_full/training_results_seed2026.json](results/helix_full/training_results_seed2026.json), [results/helix_full/training_results_seed1337.json](results/helix_full/training_results_seed1337.json)).
- The inference runtime can load helix_full checkpoints and run predictions ([src/helix_ids/operations/inference_runtime.py](src/helix_ids/operations/inference_runtime.py)).

### What partially works
- Calibration artifacts are generated, but calibration and thresholding do not change metrics or outputs in stored artifacts. Example: UNSW calibration artifacts show identical macro-F1 and identical confusion matrices across calibrated and ablation modes ([results/helix_full/calibration/unsw_nb15_calibration_governance_seed1337.json](results/helix_full/calibration/unsw_nb15_calibration_governance_seed1337.json)).
- Runtime override logic exists, but controlled A-D ablations show zero override activation and no metric change on the UNSW test set ([results/helix_full/calibration/runtime_ablation_unsw_nb15_seed1337.json](results/helix_full/calibration/runtime_ablation_unsw_nb15_seed1337.json)).

### What is fake or theoretical
- The multi-stage ESP32 -> RPi -> Server pipeline expects 41-feature inputs and a 5-class model but is not connected to the helix_full training path ([src/helix_ids/pipeline/multi_stage.py](src/helix_ids/pipeline/multi_stage.py)).
- The unified deployment script expects 32-feature models and specific model files that do not exist ([scripts/deployment/deploy.py](scripts/deployment/deploy.py)).
- The export infrastructure describes ONNX and edge deployment variants with 41 input features and 5 classes, but no current export artifacts match those assumptions ([src/helix_ids/utils/export.py](src/helix_ids/utils/export.py), [config/helix_config.yaml](config/helix_config.yaml)).

### What is disconnected
- The canonical 19-feature contract is not used by the processed artifacts. Actual processed features are 17 and include synthetic engineered features not present in the canonical 19 feature list ([data/processed/multi_dataset_v1/feature_columns.npy](data/processed/multi_dataset_v1/feature_columns.npy)).
- The deployment and edge training scripts use 32 features from a separate engineered pipeline that is not produced by the current canonical artifact builder ([scripts/training/train_edge_models.py](scripts/training/train_edge_models.py)).

### Current maturity level
- Category: Advanced research prototype undergoing architectural convergence.
- Not a production IDS.
- Not a deployable edge platform.
- Not publication-ready from a reproducibility or scientific validity standpoint.

### Biggest blockers (with evidence)
1. Schema drift: canonical 19-feature contract exists, but processed artifacts and checkpoints use 17 features and additional engineered columns. Evidence in [data/processed/multi_dataset_v1/feature_columns.npy](data/processed/multi_dataset_v1/feature_columns.npy), [src/helix_ids/data/feature_harmonization.py](src/helix_ids/data/feature_harmonization.py), and checkpoint metadata from [models/helix_full/helix_full_unsw_nb15_best.pt](models/helix_full/helix_full_unsw_nb15_best.pt).
2. Calibration and override ineffectiveness: A-D ablations show identical macro-F1, identical confusion matrices, and zero override rates ([results/helix_full/calibration/runtime_ablation_unsw_nb15_seed1337.json](results/helix_full/calibration/runtime_ablation_unsw_nb15_seed1337.json)).
3. Deployment artifacts are missing: expected model files for production/rpi/esp32 are absent and the export bundle path does not exist. Evidence in [results/helix_full/deployment_artifact_audit.json](results/helix_full/deployment_artifact_audit.json).
4. Export parity mismatch: the shipped TorchScript artifact does not match the current checkpoint outputs (max abs diff on family logits ~3.8e5) as measured against [artifacts/releases/helix_demo_packaging/helix_ids_v1_0.torchscript.pt](artifacts/releases/helix_demo_packaging/helix_ids_v1_0.torchscript.pt).

---

## SECTION 2 - Beginner Explanation

- Machine learning is a way to let computers learn patterns from data instead of writing hand-coded rules. Here, the goal is to classify network traffic as normal or attack types.
- An intrusion detection system (IDS) monitors network traffic and tries to detect malicious behavior. This project builds a machine-learning IDS using datasets of labeled network flows.
- Datasets are large collections of network flow records with labels like "Normal", "DoS", or "Probe". Examples in this repo include NSL-KDD, UNSW-NB15, and CICIDS.
- Classification means assigning a label to an input. The model outputs labels for traffic, such as an attack family.
- Neural networks are models that learn patterns by adjusting weights in layers. This repo uses multi-layer perceptrons (MLPs).
- Training means feeding labeled examples to the model so it learns. Evaluation means checking how well it predicts on held-out data.
- Inference is running the trained model on new traffic to make predictions.
- Calibration is a post-training adjustment to make prediction confidence better match reality. This project attempts temperature scaling and a class-specific threshold.
- Governance here means enforcing gates on training quality (for example minimum macro-F1) and monitoring runtime signals (override rates, entropy).
- HELIX is the name of this IDS system, intended to be a multi-stage and multi-dataset model with governance controls.

---

## SECTION 3 - Complete Repository Map

### High-level directory map
- [src/helix_ids](src/helix_ids): core library code (data, models, operations, governance)
- [scripts](scripts): training, evaluation, deployment, data processing
- [data](data): raw and processed datasets
- [models](models): checkpoints and deployment model artifacts
- [results](results): evaluation, calibration, and governance outputs
- [artifacts](artifacts): runtime logs and release packaging
- [tests](tests): unit tests and pipeline checks

### Subsystem inventory
- Data preprocessing and harmonization: [src/helix_ids/data](src/helix_ids/data)
- Model architecture: [src/helix_ids/models/helix_ids_full.py](src/helix_ids/models/helix_ids_full.py)
- Training entrypoints: [scripts/training/train_helix_ids_full.py](scripts/training/train_helix_ids_full.py), [scripts/training/train_multidataset_v2_fixed.py](scripts/training/train_multidataset_v2_fixed.py)
- Inference runtime: [src/helix_ids/operations/inference_runtime.py](src/helix_ids/operations/inference_runtime.py)
- REST serving and metrics: [scripts/operations/serve_rest.py](scripts/operations/serve_rest.py)
- Governance gates: [src/helix_ids/governance](src/helix_ids/governance), [scripts/operations/staging_gate_check.py](scripts/operations/staging_gate_check.py)
- Deployment scripts: [scripts/deployment/deploy.py](scripts/deployment/deploy.py), [src/helix_ids/pipeline/multi_stage.py](src/helix_ids/pipeline/multi_stage.py)

### Dependency graph (functional)
- Training pipeline: MultiDatasetLoader -> processed artifacts -> HelixIDSFull training -> evaluation -> calibration artifacts -> governance summary.
- Inference pipeline: checkpoint -> HelixInferenceRuntime -> REST API -> metrics/gates.
- Deployment pipeline: deploy.py or multi_stage.py -> model files and scalers (missing).

### Execution graphs (real paths)
- Real training path: [scripts/training/train_helix_ids_full.py](scripts/training/train_helix_ids_full.py) uses artifacts from [scripts/training/prepare_canonical_artifacts.py](scripts/training/prepare_canonical_artifacts.py) which uses [src/helix_ids/data/multi_dataset_loader.py](src/helix_ids/data/multi_dataset_loader.py).
- Real inference path: [scripts/operations/serve_rest.py](scripts/operations/serve_rest.py) -> [src/helix_ids/operations/inference_runtime.py](src/helix_ids/operations/inference_runtime.py).

### Disconnected or dead paths
- Multi-stage pipeline expects 41-feature inputs and a 5-class model; no training path produces this today ([src/helix_ids/pipeline/multi_stage.py](src/helix_ids/pipeline/multi_stage.py)).
- deploy.py assumes 32-feature models stored in models/production and edge directories; those model files do not exist ([scripts/deployment/deploy.py](scripts/deployment/deploy.py)).
- config/helix_config.yaml defines an attention and multi-stage architecture that is not implemented in active training code ([config/helix_config.yaml](config/helix_config.yaml)).

---

## SECTION 4 - Dataset Forensics

### Datasets present
- NSL-KDD, UNSW-NB15, CICIDS-2018 configured in [src/helix_ids/data/dataset_config.py](src/helix_ids/data/dataset_config.py).
- Raw feature counts: NSL-KDD 41, UNSW-NB15 47, CICIDS 78-79 ([src/helix_ids/data/dataset_config.py](src/helix_ids/data/dataset_config.py)).

### Processed artifacts
- Processed artifacts live under [data/processed/multi_dataset_v1](data/processed/multi_dataset_v1) and include X/y splits and feature_columns.npy.
- Actual feature_columns.npy contains 17 engineered features (not the canonical 19) with additional engineered columns such as connection_state and traffic_direction ([data/processed/multi_dataset_v1/feature_columns.npy](data/processed/multi_dataset_v1/feature_columns.npy)).

### Canonical 19-feature contract vs actual 17-feature artifact
- Canonical 19 features are defined in [src/helix_ids/data/feature_harmonization.py](src/helix_ids/data/feature_harmonization.py).
- Actual feature_columns.npy length is 17 and contains non-canonical engineered features.
- Missing canonical features from the current processed artifact include wrong_fragment, urgent, count, srv_count, serror_rate, srv_serror_rate, rerror_rate, srv_rerror_rate, same_srv_rate, diff_srv_rate, dst_host_count, dst_host_srv_count, dst_host_same_srv_rate, dst_host_diff_srv_rate.
- Extra engineered features not in the canonical 19 include connection_state, traffic_direction, has_rst, log_src_bytes, log_dst_bytes, src_dst_bytes_ratio, dst_src_bytes_ratio, same_host_rate_x_service, diff_srv_rate_x_flag, count_x_srv_count, protocol_service_flag, service_tier.

### Conflicting feature schemas
- 19 canonical features: [src/helix_ids/data/feature_harmonization.py](src/helix_ids/data/feature_harmonization.py).
- 17 features in processed artifacts: [data/processed/multi_dataset_v1/feature_columns.npy](data/processed/multi_dataset_v1/feature_columns.npy).
- 32 engineered features in unified_features.json: [data/processed/unified_features.json](data/processed/unified_features.json).
- 41 feature assumption in config and pipeline scripts: [config/helix_config.yaml](config/helix_config.yaml), [src/helix_ids/pipeline/multi_stage.py](src/helix_ids/pipeline/multi_stage.py).
- 32 feature assumption in deployment and edge training: [scripts/deployment/deploy.py](scripts/deployment/deploy.py), [scripts/training/train_edge_models.py](scripts/training/train_edge_models.py).

### Label space
- 7-class family taxonomy defined in [src/helix_ids/data/feature_harmonization.py](src/helix_ids/data/feature_harmonization.py) and used by HelixIDSFull.
- Unified 5-class mapping exists in [src/helix_ids/data/label_mapping.py](src/helix_ids/data/label_mapping.py).
- Some scripts use 5-class or binary labels without explicit alignment to the 7-class runtime.

---

## SECTION 5 - Model Architecture Forensics

### HelixIDSFull (actual model used in training/inference)
- Defined in [src/helix_ids/models/helix_ids_full.py](src/helix_ids/models/helix_ids_full.py).
- Input dim: 17 audited invariant features (explicit in HelixFullConfig).
- Backbone: 4-layer MLP with hidden dims (512, 384, 256, 128) and batch norm + dropout.
- Heads: binary head (2 classes), family head (7 classes). Family head uses a projection + whitening before final logits.

### Mathematical flow (simplified)
Given input x in R^17:
- h = MLP_backbone(x)
- binary_logits = Wb h + bb
- family_features = Projection(h) and whitened
- family_logits = Wf family_features + bf

Loss used in training (from [src/helix_ids/models/helix_ids_full.py](src/helix_ids/models/helix_ids_full.py)):
- Multi-task loss = lambda_binary * CE(binary) + lambda_family * CE(family) + optional margin and penalties.

### Architecture contradictions
- The inference runtime infers input_dim from checkpoint weights, not from canonical schema or feature contracts ([src/helix_ids/operations/inference_runtime.py](src/helix_ids/operations/inference_runtime.py)).
- The export utilities in [src/helix_ids/utils/export.py](src/helix_ids/utils/export.py) assume 41 input features and 5 classes, conflicting with HelixIDSFull.

---

## SECTION 6 - Training Pipeline Forensics

### Entry points
- Canonical training: [scripts/training/train_helix_ids_full.py](scripts/training/train_helix_ids_full.py).
- Alternative pipeline: [scripts/training/train_multidataset_v2_fixed.py](scripts/training/train_multidataset_v2_fixed.py) uses 32 engineered features and a different data loader.

### Data flow
- Processed artifacts are built via [scripts/training/prepare_canonical_artifacts.py](scripts/training/prepare_canonical_artifacts.py), which calls MultiDatasetLoader in [src/helix_ids/data/multi_dataset_loader.py](src/helix_ids/data/multi_dataset_loader.py).
- MultiDatasetLoader selects features by intersection and mutual information, leading to a 17-feature engineered set (feature_columns.npy).
- Learnability contract validation uses a different schema hash function in [src/helix_ids/data/learnability_contract.py](src/helix_ids/data/learnability_contract.py) than the canonical feature_harmonization hash.

### Seed stability evidence
- Seed 1337 passes with macro_f1=0.432 on UNSW and returns exit code 0 ([results/helix_full/eval_results_seed1337.json](results/helix_full/eval_results_seed1337.json)).
- Seeds 42 and 2026 fail post-training macro-F1 guard (exit code 1) ([results/helix_full/training_results_seed42.json](results/helix_full/training_results_seed42.json), [results/helix_full/training_results_seed2026.json](results/helix_full/training_results_seed2026.json)).
- Only one seed is valid, so reproducibility gates cannot be met in practice.

### Observed instability signals
- Family entropy is near zero in eval results (6.6e-12), indicating saturated logits and near-deterministic predictions ([results/helix_full/eval_results_seed1337.json](results/helix_full/eval_results_seed1337.json)).
- Minority recall for class 4 is near zero in standard evaluation (family_minority_recall_min ~0.0098 in eval results) ([results/helix_full/eval_results_seed1337.json](results/helix_full/eval_results_seed1337.json)).

---

## SECTION 7 - Calibration + Override Validity

### Calibration artifacts (from training script)
- Temperature scaling and class-4 thresholding are computed in [scripts/training/train_helix_ids_full.py](scripts/training/train_helix_ids_full.py).
- Calibration artifacts for UNSW show identical macro-F1 and confusion matrices across calibrated, no-temperature, and no-threshold ablations, indicating no effect ([results/helix_full/calibration/unsw_nb15_calibration_governance_seed1337.json](results/helix_full/calibration/unsw_nb15_calibration_governance_seed1337.json)).

### Runtime A-D ablations (controlled)
Runtime ablations were executed on UNSW test set using the actual checkpoint and calibration settings:
- A: raw logits only
- B: calibration only (temperature + threshold)
- C: overrides only (runtime overrides, temperature=1.0)
- D: calibration + overrides (temperature only + overrides)

Results are identical across A-D:
- macro_f1 = 0.26246
- class4_precision = 0.03148
- class4_recall = 1.0
- confusion matrices identical
- override rates = 0.0
Evidence: [results/helix_full/calibration/runtime_ablation_unsw_nb15_seed1337.json](results/helix_full/calibration/runtime_ablation_unsw_nb15_seed1337.json).

### Interpretation
- Calibration layers are mathematically disconnected from evaluation outcomes.
- Runtime override logic did not activate on the test set (override rate 0.0).
- Temperature scaling does not change predictions in practice (logits likely saturated).

---

## SECTION 8 - Governance Forensics

### Runtime gate logic
- REST runtime emits metrics for override rate and degraded state ([scripts/operations/serve_rest.py](scripts/operations/serve_rest.py)).
- Promotion gate blocks when override_rate > 0.02 or degraded_state == 1 ([scripts/operations/staging_gate_check.py](scripts/operations/staging_gate_check.py)).
- Traffic expansion guard halts traffic if degraded_state == 1 ([scripts/operations/traffic_expansion_guard.py](scripts/operations/traffic_expansion_guard.py)).

### Governance vs correctness
- Gates are driven by override_rate and entropy, not by correctness.
- Override rate can be zero even when macro-F1 is low (runtime ablation results), so gates do not detect failure.
- Promotion consensus requires 3 seeds and low variance, but only one seed passes, so consensus is unreachable in current state ([src/helix_ids/governance/promotion.py](src/helix_ids/governance/promotion.py)).

---

## SECTION 9 - Deployment Forensics

### Deployment scripts
- [scripts/deployment/deploy.py](scripts/deployment/deploy.py) expects 32-feature models and model files under models/production, models/rpi_4, models/rpi_zero, models/esp32.
- Those model files are missing; only scalers and feature_names.json exist.
- Evidence of missing files: [results/helix_full/deployment_artifact_audit.json](results/helix_full/deployment_artifact_audit.json).

### Export pipeline mismatch
- Export script expects output in artifacts/releases/helix_ids_v1.0/packaging ([scripts/operations/export_inference_bundle.py](scripts/operations/export_inference_bundle.py)).
- Actual artifact exists only at artifacts/releases/helix_demo_packaging/helix_ids_v1_0.torchscript.pt.
- ONNX and service_contract.json are missing (audit report).

### Export parity failure
- TorchScript artifact outputs differ dramatically from the current checkpoint outputs (max abs diff on family logits ~4.7e5), indicating mismatch or stale artifact.
- Evidence: [results/helix_full/export_parity_audit.json](results/helix_full/export_parity_audit.json).

---

## SECTION 10 - Testing + Engineering Quality

### Tests present
- Export/quantization tests: [tests/test_export_quantization_deployment.py](tests/test_export_quantization_deployment.py).
- Pipeline and operations tests: [tests/test_operations](tests/test_operations).

### Gaps
- No tests enforce schema consistency between feature_harmonization canonical 19 features and processed artifacts.
- No tests verify inference runtime uses calibration outputs (it does not).
- No tests validate deploy.py paths or multi_stage pipeline against real artifacts.
- No tests compare export artifacts with current checkpoints.

---

## SECTION 11 - Canonical System Design (Required Convergence)

### Canonical feature schema
- MUST be the 19-feature harmonized contract in [src/helix_ids/data/feature_harmonization.py](src/helix_ids/data/feature_harmonization.py).
- All processed artifacts must be rebuilt to match the 19 feature order.

### Canonical label space
- The canonical label space is the 7-class attack family mapping defined in [src/helix_ids/data/feature_harmonization.py](src/helix_ids/data/feature_harmonization.py).
- All training, evaluation, and runtime must be aligned to this 7-class space.

### Canonical training path
- Only [scripts/training/train_helix_ids_full.py](scripts/training/train_helix_ids_full.py) should remain, using artifacts built by [scripts/training/prepare_canonical_artifacts.py](scripts/training/prepare_canonical_artifacts.py).

### Canonical inference path
- Only [src/helix_ids/operations/inference_runtime.py](src/helix_ids/operations/inference_runtime.py) + [scripts/operations/serve_rest.py](scripts/operations/serve_rest.py) should be used.
- Runtime must validate schema hash and feature order length at startup.

### Canonical deployment path
- One deployment path based on the actual trained HelixIDSFull model and a verified export bundle.
- Edge and multi-stage deployments are invalid until they are rebuilt on top of the canonical 19-feature contract.

---

## SECTION 12 - What Must Be Deleted

These components are incompatible with the canonical 19-feature contract and/or have no valid artifacts:
- [scripts/deployment/deploy.py](scripts/deployment/deploy.py) (32-feature model assumption, missing model files).
- [src/helix_ids/pipeline/multi_stage.py](src/helix_ids/pipeline/multi_stage.py) (41-feature assumptions, no matching training pipeline).
- [scripts/training/train_edge_models.py](scripts/training/train_edge_models.py) (32-feature edge pipeline detached from canonical artifacts).
- [config/helix_config.yaml](config/helix_config.yaml) (41-feature, attention, TFLite, and multi-stage claims not implemented).
- [src/helix_ids/utils/export.py](src/helix_ids/utils/export.py) (41-feature, 5-class export assumptions not aligned with HelixIDSFull).

---

## SECTION 13 - What Must Be Rebuilt

- Canonical schema validation at runtime: enforce 19-feature order and schema hash before inference.
- Processed artifacts must be regenerated from raw datasets with strict 19-feature harmonization.
- Calibration pipeline must be wired into runtime (or removed). Currently calibration is mathematically disconnected from runtime.
- Export pipeline must be rebuilt to generate a verified TorchScript and ONNX pair tied to the exact checkpoint and schema hash.
- Reproducibility must be enforced: at least 3 seeds passing governance gates with variance constraints.

---

## SECTION 14 - Publication Readiness

- Not publishable.
- Lacks reproducibility (only one seed passes).
- Calibration results show no functional impact.
- Multiple contradictory feature schemas make experimental claims ambiguous.

---

## SECTION 15 - Deployment Readiness

- Not deployable.
- No valid production or edge model artifacts exist.
- Runtime gates do not correlate with correctness (override rate can be zero while macro-F1 is low).
- Export artifacts do not match current checkpoints.

---

## SECTION 16 - Final Verdict

### What category is this project really?
- Advanced research prototype undergoing architectural convergence.

### Single biggest blocker
- Schema incoherence: canonical 19-feature contract exists but real artifacts and checkpoints use a 17-feature engineered schema.

### Next immediate step
- Freeze architecture and rebuild processed artifacts and checkpoints strictly under the 19-feature contract.

### Timeline
- 24 hours: delete invalid deployment paths, remove non-canonical feature pipelines, regenerate canonical artifacts.
- 7 days: retrain HelixIDSFull on canonical 19 features; pass 3-seed reproducibility gates.
- 30 days: validate calibration and runtime overrides with real effect; export verified TorchScript/ONNX bundle.
- 90 days: only then consider deployment or publication.

---

## Canonical Roadmap

1. Enforce 19-feature schema in preprocessing, checkpoint metadata, and runtime validation.
2. Retrain and evaluate using fixed seeds and reproducibility gates.
3. Validate calibration and overrides with measurable metric deltas.
4. Produce one verified export bundle (TorchScript + ONNX + service contract).
5. Only then reintroduce any deployment path.

---

## Technical Debt Register

- Schema drift across 19 vs 17 vs 32 vs 41 features.
- Calibration pipeline does not influence runtime or evaluation.
- Deployment scripts assume missing artifacts.
- Export artifacts are stale or mismatched.
- Multiple training pipelines with conflicting assumptions.

---

## Final System Maturity Table

| Area | Status | Evidence |
| --- | --- | --- |
| Schema consistency | Failed | [data/processed/multi_dataset_v1/feature_columns.npy](data/processed/multi_dataset_v1/feature_columns.npy) vs [src/helix_ids/data/feature_harmonization.py](src/helix_ids/data/feature_harmonization.py) |
| Training reproducibility | Failed | [results/helix_full/training_results_seed42.json](results/helix_full/training_results_seed42.json), [results/helix_full/training_results_seed2026.json](results/helix_full/training_results_seed2026.json) |
| Calibration validity | Failed | [results/helix_full/calibration/unsw_nb15_calibration_governance_seed1337.json](results/helix_full/calibration/unsw_nb15_calibration_governance_seed1337.json) |
| Override validity | Failed | [results/helix_full/calibration/runtime_ablation_unsw_nb15_seed1337.json](results/helix_full/calibration/runtime_ablation_unsw_nb15_seed1337.json) |
| Deployment readiness | Failed | [results/helix_full/deployment_artifact_audit.json](results/helix_full/deployment_artifact_audit.json) |
| Export parity | Failed | [results/helix_full/export_parity_audit.json](results/helix_full/export_parity_audit.json) |

---

End of audit.
