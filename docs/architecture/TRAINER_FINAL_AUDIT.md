# Trainer Final Audit — HelixFullTrainer & TrainerFacade

Generated: Phase 19 Architecture Freeze

---

## 1. Overview

| Component | LOC | Methods |
|---|---|---|
| HelixFullTrainer (class) | 1,929 | 93 |
| train_helix_ids_full.py (file total) | 4,607 | — |
| TrainerFacade | 180 | 20 |
| Extracted subpackages (9 groups, 33 files) | 7,325 | — |

**Delegation ratio:** 47/93 total external methods invoked from the trainer are
re-routed through extracted delegates (51% external delegation by method
count).

---

## 2. Method Inventory — HelixFullTrainer

### 2.1 Methods Remaining (93 total)

| # | Method | LOC | Category | Owner |
|---|---|---|---|---|
| 1 | `__init__` | 392 | Init & Setup | Trainer |
| 2 | `_should_exit_representation_curriculum` | 18 | Scheduler | Partial → PhaseManager |
| 3 | `_stabilize_centroids` | 7 | Representation | Delegate → CentroidManager |
| 4 | `configure_family_controls` | 33 | Init & Setup | Trainer |
| 5 | `configure_structure_recovery` | 32 | Init & Setup | Trainer |
| 6 | `_set_phase_trainability` | 31 | Phase | Trainer |
| 7 | `_set_phase_lr_scales` | 16 | Phase | Trainer |
| 8 | `_is_representation_window_step` | 6 | Scheduler | Delegate → PhaseManager |
| 9 | `_prepare_representation_features` | 3 | Diagnostics | Delegate → ClusterAnalyzer |
| 10 | `_compute_grad_l2_norm` | 9 | Training Loop | Trainer |
| 11 | `_scale_backbone_gradients` | 8 | Training Loop | Trainer |
| 12 | `_intra_class_variance_clamp_loss` | 11 | Loss | Trainer |
| 13 | `_compute_class_centroids` | 7 | Diagnostics | Delegate → ClusterAnalyzer |
| 14 | `_compute_batch_class_centroids_for_loss` | 6 | Loss | Trainer |
| 15 | `_update_running_rep_centroids` | 7 | Representation | Delegate → CentroidManager |
| 16 | `_freeze_epoch_centroid_snapshot` | 3 | Representation | Delegate → CentroidManager |
| 17 | `_update_centroids_from_epoch_buffer` | 7 | Representation | Delegate → CentroidManager |
| 18 | `_global_centroid_guided_losses` | 12 | Loss | Trainer |
| 19 | `_critical_pair_centroid_push_loss` | 13 | Loss | Trainer |
| 20 | `_current_geometry_ratio_threshold` | 6 | Diagnostics | Delegate → GeometryAnalyzer |
| 21 | `_build_representation_snapshot_id` | 14 | Diagnostics | Delegate → RepDiagnostics |
| 22 | `_maybe_activate_joint_finetune_phase` | 14 | Scheduler | Partial → PhaseManager |
| 23 | `_rebalance_representation_batch` | 14 | Representation | Delegate → RepresentationCoordinator |
| 24 | `_critical_pair_key` | 3 | Diagnostics | Delegate → GeometryAnalyzer |
| 25 | `_has_critical_collision_pairs` | 3 | Diagnostics | Delegate → GeometryAnalyzer |
| 26 | `_apply_emergency_label_merge` | 14 | Diagnostics | Trainer |
| 27 | `_enforce_geometry_integrity` | 11 | Diagnostics | Delegate → GeometryAnalyzer |
| 28 | `_collect_normalized_embeddings` | 8 | Diagnostics | Delegate → RepDiagnostics |
| 29 | `_embed_feature_matrix` | 8 | Diagnostics | Delegate → ClusterAnalyzer |
| 30 | `_assign_labels_from_centers` | 7 | Diagnostics | Delegate → ClusterAnalyzer |
| 31 | `_fit_embedding_clusters` | 8 | Diagnostics | Delegate → ClusterAnalyzer |
| 32 | `_build_cluster_label_bridge` | 11 | Diagnostics | Delegate → ClusterAnalyzer |
| 33 | `_apply_cluster_relabels_to_datasets` | 11 | Diagnostics | Delegate → ClusterAnalyzer |
| 34 | `_nearest_center_accuracy` | 9 | Diagnostics | Delegate → RepDiagnostics |
| 35 | `_build_class_centers` | 8 | Diagnostics | Delegate → ClusterAnalyzer |
| 36 | `_compute_inter_and_intra_distances` | 18 | Diagnostics | Delegate → GeometryAnalyzer |
| 37 | `_estimate_local_density_diagnostics` | 14 | Diagnostics | Delegate → GeometryAnalyzer |
| 38 | `_compute_center_pair_diagnostics` | 7 | Diagnostics | Delegate → RepDiagnostics |
| 39 | `_compute_representation_diagnostics` | 25 | Diagnostics | Delegate → RepDiagnostics |
| 40 | `_run_representation_diagnostics` | 20 | Diagnostics | Delegate → RepDiagnostics |
| 41 | `_active_balance_strategy` | 5 | Training Loop | Trainer |
| 42 | `_set_epoch_loss_strategy` | 6 | Training Loop | Trainer |
| 43 | `_apply_family_logit_controls` | 30 | Training Loop | Trainer |
| 44 | `_ensure_energy_win_rate_ema` | 11 | Training Loop | Trainer |
| 45 | `_update_energy_win_rate_ema` | 14 | Training Loop | Trainer |
| 46 | `_compute_energy_emergence_bias` | 12 | Training Loop | Trainer |
| 47 | `_reseed_epoch_generators` | 19 | Training Loop | Trainer |
| 48 | `_compute_f1_stats_from_confusion` | 6 | Monitoring | Trainer |
| 49 | `_get_learning_rate` | 6 | Scheduler | Delegate → LRScheduler |
| 50 | `_set_learning_rate` | 8 | Scheduler | Delegate → LRScheduler |
| 51 | `_set_backbone_freeze_state` | 21 | Training Loop | Trainer |
| 52 | `_current_learning_rate` | 6 | Monitoring | Trainer |
| 53 | `_apply_loss_regularizations` | 18 | Loss | Trainer (partial) |
| 54 | `_update_train_batch_stats` | 24 | Monitoring | Trainer |
| 55 | `_log_step10_diagnostics` | 17 | Monitoring | Trainer |
| 56 | `_log_batch_progress` | 28 | Monitoring | Trainer |
| 57 | `_apply_cluster_relabeling` | 5 | Diagnostics | Trainer |
| 58 | `_as_python_int` | 5 | Utility | Trainer |
| 59 | `_collect_class_to_indices` | 15 | Utility | Trainer |
| 60 | `_resolve_warmup_active_class_ids` | 17 | Warmup | Trainer |
| 61 | `_build_warmup_batch_tensors` | 20 | Warmup | Trainer |
| 62 | `_run_epoch0_forced_coverage_warmup` | 24 | Warmup | Trainer |
| 63 | `_resolve_batch_active_family_class_ids` | 14 | Training Loop | Trainer |
| 64 | `_stabilize_batch_family_logits` | 39 | Training Loop | Trainer |
| 65 | `_compute_tail_focal_loss` | 6 | Loss | Trainer |
| 66 | `_compute_representation_energy_objective` | 24 | Loss | Trainer |
| 67 | `_log_energy_gap_diag_if_needed` | 10 | Monitoring | Trainer |
| 68 | `_apply_entropy_floor_regularizer` | 12 | Loss | Trainer |
| 69 | `_backpropagate_train_batch_loss` | 29 | Training Loop | Trainer |
| 70 | `_compute_loss_with_optional_energy` | 23 | Loss | Trainer |
| 71 | `_apply_optional_non_representation_regularizations` | 20 | Loss | Trainer |
| 72 | `_maybe_store_representation_chunks` | 15 | Representation | Trainer |
| 73 | `_process_train_batch` | 49 | Training Loop | Trainer (core) |
| 74 | `_check_backbone_freeze_state` | 9 | Scheduler | Delegate → FreezeManager |
| 75 | `_check_family_class_coverage` | 10 | Monitoring | Trainer |
| 76 | `_check_step_coverage` | 28 | Monitoring | Trainer |
| 77 | `_handle_representation_phase_logic` | 38 | Phase | Trainer |
| 78 | `train_epoch` | 69 | Training Loop | Trainer (core) |
| 79 | `_log_epoch_completion` | 44 | Monitoring | Trainer |
| 80 | `_apply_eval_class4_logit_shift` | 10 | Evaluation | Delegate → Evaluator |
| 81 | `_apply_inference_prediction_floor` | 13 | Evaluation | Trainer |
| 82 | `_evaluate_loader` | 12 | Evaluation | Delegate → Evaluator |
| 83 | `validate` | 11 | Evaluation | Partial → Evaluator |
| 84 | `_process_test_batch` | 18 | Evaluation | Delegate → Evaluator |
| 85 | `_evaluate_test_loader` | 11 | Evaluation | Delegate → Evaluator |
| 86 | `evaluate_per_dataset` | 11 | Evaluation | Partial → Evaluator |
| 87 | `fit` | 50 | Entry Point | Trainer (core) |
| 88 | `_post_training_macro_floor` | 8 | Evaluation | Trainer |
| 89 | `_is_smoke_mode` | 9 | Scheduler | Delegate → EarlyStoppingManager |
| 90 | `_hard_stop_reason` | 15 | Scheduler | Partial → EarlyStoppingManager |
| 91 | `_update_early_stopping` | 12 | Scheduler | Partial → EarlyStoppingManager |
| 92 | `_save_checkpoint_if_needed` | 19 | Checkpoint | Trainer |
| 93 | `_log_per_dataset_results` | 8 | Monitoring | Trainer |

### 2.2 Category Breakdown

| Category | Methods | % of Total |
|---|---|---|
| Training Loop (core) | 18 | 19% |
| Monitoring/Logging | 12 | 13% |
| Diagnostics (delegated) | 11 | 12% |
| Loss regularization | 10 | 11% |
| Init & Setup | 4 | 4% |
| Scheduler (delegated/partial) | 8 | 9% |
| Evaluation | 7 | 8% |
| Utility | 3 | 3% |
| Warmup | 3 | 3% |
| Representation (delegated) | 4 | 4% |
| Phase | 3 | 3% |
| Checkpoint | 1 | 1% |
| Entry Point | 1 | 1% |

---

## 3. TrainerFacade — Full Inventory

| Property/Method | LOC | Type | Target |
|---|---|---|---|
| `__init__` | 22 | Constructor | Creates 18 lazy proxies |
| `build` | 56 | Method | Builds all dependencies via factory |
| `phase_manager` | 3 | Property | Lazy init → PhaseManager |
| `early_stopping_manager` | 3 | Property | Lazy init → EarlyStoppingManager |
| `freeze_manager` | 3 | Property | Lazy init → FreezeManager |
| `lr_scheduler` | 3 | Property | Lazy init → LRScheduler |
| `evaluation_orchestrator` | 3 | Property | Lazy init → EvaluationOrchestrator |
| `validation_orchestrator` | 3 | Property | Lazy init → ValidationOrchestrator |
| `geometry_analyzer` | 3 | Property | Lazy init → GeometryAnalyzer |
| `cluster_analyzer` | 3 | Property | Lazy init → ClusterAnalyzer |
| `rep_diagnostics` | 3 | Property | Lazy init → RepDiagnostics |
| `centroid_manager` | 3 | Property | Lazy init → CentroidManager |
| `phase_orchestrator` | 3 | Property | Lazy init → PhaseOrchestrator |
| `representation_coordinator` | 3 | Property | Lazy init → RepresentationCoordinator |
| `loss_registry` | 3 | Property | Lazy init → LossRegistry |
| `batch_processor` | 3 | Property | Lazy init → BatchProcessor |
| `warmup_manager` | 3 | Property | Lazy init → WarmupManager |
| `epoch_runner` | 3 | Property | Lazy init → EpochRunner |
| `training_orchestrator` | 3 | Property | Lazy init → TrainingOrchestrator |
| `recovery_manager` | 3 | Property | Lazy init → RecoveryManager |

**Total:** 20 methods/properties, 180 LOC. All lazy-init properties are 3-line
pass-throughs. `build()` contains the factory wiring.

---

## 4. Delegation Analysis

### 4.1 Full Delegation (30 wrappers)

These are pure pass-throughs that could be replaced with direct calls to
`self._xxx.yyy()`:

| Group | Wrappers | Delegate Target |
|---|---|---|
| Diagnostics (ClusterAnalyzer) | 8 | `_cluster_analyzer.*()` |
| Diagnostics (GeometryAnalyzer) | 6 | `_geometry_analyzer.*()` |
| Diagnostics (RepDiagnostics) | 6 | `_rep_diagnostics.*()` |
| Representation (CentroidManager) | 4 | `_centroid_manager.*()` |
| Representation (RepresentationCoordinator) | 1 | `_representation_coordinator.*()` |
| Evaluation (Evaluator) | 4 | `_evaluator.*()` |
| Scheduler (LRScheduler) | 2 | `_lr_scheduler.*()` |
| Scheduler (FreezeManager) | 1 | `_freeze_manager.*()` |
| Scheduler (EarlyStoppingManager) | 1 | `_early_stopping_manager.*()` |
| Scheduler (PhaseManager) | 1 | `_phase_manager.*()` |

**Removal candidates:** All 30 full-delegation wrappers are technically
removable. Removing them, however, would change the trainer's public API
(which is referenced by scripts/orchestration/run_orchestrator.py and tests).
**Recommendation:** Mark as `REMOVAL CANDIDATE` with docstring noting they
exist for API stability only. Remove in Phase 20+ when callers are updated.

### 4.2 Partial Delegation (17 wrappers)

These mix delegate calls with inline orchestration:

| Wrapper | LOC | Delegate | Inline Logic |
|---|---|---|---|
| `_should_exit_representation_curriculum` | 18 | PhaseManager | Condition checking |
| `_maybe_activate_joint_finetune_phase` | 14 | PhaseManager | Transition logic |
| `_handle_representation_phase_logic` | 38 | — | Phase orchestration |
| `_update_early_stopping` | 12 | EarlyStoppingManager | Metric collection |
| `_hard_stop_reason` | 15 | EarlyStoppingManager | Reason formatting |
| `validate` | 11 | Evaluator | Metric aggregation |
| `evaluate_per_dataset` | 11 | Evaluator | Per-dataset routing |
| `_apply_loss_regularizations` | 18 | LossRegistry | Loss assembly |

**Recommendation:** Extract inline logic into delegate objects. This is the
remaining architectural debt from Phase 13B.

### 4.3 True Trainer-Owned Responsibilities (46 methods remaining inline)

These are methods that genuinely belong to the trainer because they represent
training-loop orchestration, batch processing, and phase management:

1. `__init__` (392 LOC) — Configuration and state initialization
2. `fit` (50 LOC) — Top-level training entry point
3. `train_epoch` (69 LOC) — Per-epoch training loop
4. `_process_train_batch` (49 LOC) — Single-batch training step
5. `_backpropagate_train_batch_loss` (29 LOC) — Loss backpropagation
6. `_compute_loss_with_optional_energy` (23 LOC) — Loss computation dispatch
7. `_apply_loss_regularizations` (18 LOC) — Loss assembly
8. `_stabilize_batch_family_logits` (39 LOC) — Logit stabilization
9. `_apply_family_logit_controls` (30 LOC) — Family-aware logit correction
10. `_check_step_coverage` (28 LOC) — Step coverage validation
11. `_check_family_class_coverage` (10 LOC) — Class coverage
12. `_handle_representation_phase_logic` (38 LOC) — Representation phase
13. `_set_phase_trainability` (31 LOC) — Phase parameter management
14. `_set_phase_lr_scales` (16 LOC) — Per-phase LR scaling
15. `configure_family_controls` (33 LOC) — Family control config
16. `configure_structure_recovery` (32 LOC) — Recovery configuration
17. `_save_checkpoint_if_needed` (19 LOC) — Checkpoint persistence
18. Loss methods (9 functions, ~120 LOC) — Various loss computations

---

## 5. Removal Candidates

| Method | Type | Reason | Effort |
|---|---|---|---|
| All 30 full-delegation wrappers | FULL | Pure pass-through | Low |
| `_as_python_int` | Utility | Python built-in exists | Trivial |
| `_compute_f1_stats_from_confusion` | Monitoring | Could be utility function | Low |
| `_current_learning_rate` | Monitoring | Trivial property | Trivial |
| 9 inline loss methods | Loss | Should go to LossRegistry | Medium |

---

## 6. Justification for Remaining Inline Methods

| Reason | Count | Examples |
|---|---|---|
| True orchestration that belongs in the trainer | 25 | `fit`, `train_epoch`, `_process_train_batch` |
| Complex phase/scheduling logic partially delegated | 10 | `_handle_representation_phase_logic` |
| Loss functions awaiting registry migration | 9 | `_compute_tail_focal_loss` |
| Monitoring tightly coupled to trainer state | 9 | `_log_batch_progress`, `_update_train_batch_stats` |
| Utility helpers used exclusively by trainer | 3 | `_as_python_int`, `_collect_class_to_indices` |

---

## 7. Summary Metrics

| Metric | Value | Status |
|---|---|---|
| HelixFullTrainer methods | 93 | ↓ from 109 (Phase 13B) |
| HelixFullTrainer class LOC | 1,929 | ↓ from 2,525 (Phase 13B) |
| train_helix_ids_full.py LOC | 4,607 | ↓ from 4,920 (Phase 13B) |
| TrainerFacade methods | 20 | Stable |
| Full delegation wrappers | 30 | Removal candidates |
| Partial delegation wrappers | 17 | Need extraction |
| True trainer-owned methods | 46 | Justified |
| Extracted subpackages | 9 groups, 33 files | Stable |
| Decomposition ratio | 51% delegated | Healthy |
