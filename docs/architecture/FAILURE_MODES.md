# HELIX-IDS Runtime Failure Modes

> **Purpose:** Inventory of known runtime failure modes, detection mechanisms,
> recovery strategies, and severity assessments based on actual codebase evidence.
>
> **Scope:** Training pipeline, inference runtime, monitoring, staging gates,
> data loading, and governance subsystems.
>
> **Last updated:** 2026-06-16

---

## 1. Dataset Failures (Missing Files, Load Errors, Corrupt Data)

| Field | Value |
|---|---|
| Detection | `FileNotFoundError` raised by `MultiDatasetLoader.load_nslkdd()` / `load_unsw()` / `load_cicids()` when all candidate paths are exhausted. `SchemaDriftError` (AssertionError subclass) in `feature_harmonization.py` for structural schema mismatches. `ValueError` for label-space mismatches (unresolved attack categories). `validate_no_nan_inf()` catches NaN/inf in harmonized output. |
| Recovery | Sequential fallback path traversal — each dataset loader tries 3–4 candidate paths before raising. CICIDS loader caches cleaned combined CSV to `processed/` for repeatability. `load_cicids()` returns `None` (graceful degradation) if no CICIDS data found. |
| Severity | **HIGH** — blocks training entirely when all paths fail; no automatic retry mechanism at the loader level. |

**Evidence locations:**

- `src/helix_ids/data/multi_dataset_loader.py`:234 — NSL-KDD fallback paths (`paths = [raw_train, processed_dir/nsl-kdd_cleaned.csv, ...]`)
- `src/helix_ids/data/multi_dataset_loader.py`:258–273 — UNSW fallback paths
- `src/helix_ids/data/multi_dataset_loader.py`:275–341 — CICIDS multi-path search with 8+ candidate directories
- `src/helix_ids/data/multi_dataset_loader.py`:439, 480 — label-space `ValueError` for NSL-KDD and CICIDS
- `src/helix_ids/data/feature_harmonization.py`:252–258 — `validate_no_nan_inf()` post-harmonization assertion
- `src/helix_ids/data/feature_harmonization.py`:310, 349–351 — `SchemaDriftError` on feature-order mismatch
- `src/helix_ids/data/loader_core.py`:406 — `FileNotFoundError` for generic data loading

---

## 2. OOM (Out of Memory)

| Field | Value |
|---|---|
| Detection | Not explicitly caught. Relies on Python `MemoryError` or CUDA `torch.cuda.OutOfMemoryError` propagating as unhandled exceptions. |
| Recovery | **No explicit recovery.** Batch sizes are configured per variant in `helix_config.yaml` (nano=256, lite=128, full=64) as implicit OOM prevention. Gradient checkpointing is not present. |
| Severity | **CRITICAL** — no detection or recovery path exists. OOM at any point crashes the process. |

**Evidence locations:**

- `config/helix_config.yaml`:35 — `batch_size: 256` (nano), `batch_size: 128` (lite), `batch_size: 64` (full)
- No `try/except` for `MemoryError` or `OutOfMemoryError` anywhere in the training or inference codebases.

---

## 3. NaN Propagation

| Field | Value |
|---|---|
| Detection | `torch.isfinite()` check in `_assert_output_parity()` on inference outputs. Logit saturation warnings in `EpochRunner._log_epoch_completion()` when `train_logit_max > 10.0` or `train_logit_min < -10.0`. NaN/inf detected post-harmonization by `validate_no_nan_inf()`. |
| Recovery | **Prevention rather than recovery.** Logits clamped to [-10.0, 10.0] in `BatchProcessor._stabilize_batch_family_logits()`. Gradient clipping via `max_grad_norm` config (default 1.0). NaN → 0.0 substitution in preprocessing (`np.nan_to_num`). Inf → 1e6/-1e6 substitution. Temperature clamping at 1e-6. |
| Severity | **HIGH** — detected but no active repair mechanism; clamping only masks limited-range NaN ingress; unchecked NaN in loss computation would still crash. |

**Evidence locations:**

- `src/helix_ids/operations/inference_runtime.py`:269–270 — `torch.isfinite()` output parity check
- `src/helix_ids/operations/inference_runtime.py`:270 — `"output[{index}] contains non-finite values"` error
- `scripts/training/execution/batch_processor.py`:242 — `torch.clamp(controlled_logits, -10.0, 10.0)`
- `scripts/training/execution/batch_processor.py`:256–257 — gradient clipping via `clip_grad_norm_`
- `scripts/training/execution/epoch_runner.py`:437–444 — logit saturation warnings
- `src/helix_ids/data/preprocessing.py`:190–203 — NaN imputation (median/mean/zero) and `np.nan_to_num`
- `src/helix_ids/data/preprocessing.py`:203 — `np.nan_to_num(x_arr, nan=0.0, posinf=1e6, neginf=-1e6)`
- `src/helix_ids/data/feature_harmonization.py`:252–258 — `validate_no_nan_inf()` assertion
- `src/helix_ids/data/loader_core.py`:460–465 — inf-to-NaN conversion then `nan_to_num`
- `src/helix_ids/config/helix_full_config.py`:45 — `max_grad_norm: float = 1.0`

---

## 4. Corrupt Checkpoints

| Field | Value |
|---|---|
| Detection | Multi-layer contract validation: `_validate_checkpoint_contract()` checks 9 required metadata keys (schema_version, feature_order, schema_hash, contract_version, feature_order_hash, input_dim, binary_output_dim, family_output_dim). Sidecar file validation (`.contract.json`, `.feature_order.json`, `.schema_hash.txt`). `assert_runtime_contract()` cross-checks all fields. `verify_ingress_artifact()` validates manifests, feature-order hashes, and ingress metadata. `ArtifactManifestError` on hash mismatch. |
| Recovery | **Hard failure.** `strict=True` in `load_state_dict()` prevents silent loading. Missing sidecar files raise `RuntimeError`. Manifest hash mismatches raise `RuntimeError`. No automatic repair or fallback to alternative checkpoints. |
| Severity | **CRITICAL** — exhaustive detection, but corrupt checkpoints are unrecoverable at runtime. |

**Evidence locations:**

- `src/helix_ids/operations/inference_runtime.py`:135–178 — `_validate_checkpoint_contract()` with 9-field validation
- `src/helix_ids/operations/inference_runtime.py**:447–494 — checkpoint loading with sidecar validation
- `src/helix_ids/operations/inference_runtime.py**:166–173 — contract version and feature-order hash checks
- `src/helix_ids/operations/inference_runtime.py**:462 — `load_state_dict(state_dict, strict=True)`
- `src/helix_ids/governance/provenance.py**:346–397 — `checkpoint_manifest_payload()` and embedded manifest reader
- `src/helix_ids/governance/lifecycle_verifier.py**:138–182 — `_write_checkpoint()` / `_reload_checkpoint()` with `verify_contract_integrity()`
- `src/helix_ids/governance/lifecycle_verifier.py**:358 — `ArtifactManifestError` for feature-order hash mismatch
- `src/helix_ids/governance/lifecycle_verifier.py**:432 — schema hash sidecar mismatch detection

---

## 5. Missing Config

| Field | Value |
|---|---|
| Detection | `FileNotFoundError` when `platform_configs.yaml` is missing. `KeyError` when platform not found in config YAML. `ValueError` for unknown scale methods or activation types. `AttributeError` on missing config fields (implicit). |
| Recovery | Dataclass defaults for many config objects: `PreprocessingConfig()` with `scale_method="standard"`, `handle_missing="median"`. `InferenceConfig()` with `fixed_temperature=1.0`, `prediction_floor=1e-6`. `DEFAULT_GOVERNANCE_POLICY` as fallback. `helix_config.yaml` provides variant-specific defaults. |
| Severity | **MEDIUM** — many configs have safe defaults, but missing critical config (checkpoint path, model variant) causes hard failure. |

**Evidence locations:**

- `src/helix_ids/config/platform_loader.py`:41–45 — `FileNotFoundError` when config file missing
- `src/helix_ids/data/preprocessing.py`:44 — `config or PreprocessingConfig()` default fallback
- `src/helix_ids/operations/inference_runtime.py`:36–58 — `InferenceConfig` with all field defaults
- `src/helix_ids/governance/orchestrator.py`:69 — `policy: GovernancePolicy = DEFAULT_GOVERNANCE_POLICY`
- `src/helix_ids/data/preprocessing.py`:74 — `ValueError` for unknown scale method
- `src/helix_ids/models/helix_ids_full.py`:87 — `ValueError` for unknown activation
- `config/helix_config.yaml` — complete variant-specific configuration tree

---

## 6. Invalid Labels

| Field | Value |
|---|---|
| Detection | `ValueError("unsw_nb15 label-space mismatch")` in `_normalize_unsw_label_series()`. `ValueError("nsl_kdd label-space mismatch")` and `ValueError("cicids label-space mismatch")` in harmonization. `RuntimeError("Hard-stop integrity guard triggered: invalid_family_labels_in_eval_")` in `Evaluator`. `RuntimeError("Hard-stop integrity guard triggered: invalid_family_labels_in_test")`. `RuntimeError("Hard-stop integrity guard triggered: batch_diversity_violation_lt2")` in `EpochRunner` (fewer than 2 unique classes in a batch). |
| Recovery | **Hard-stop with no correction.** `disable_integrity_hard_stops` flag exists to bypass some checks during development. Label mapping dictionaries (`NSLKDD_TO_7CLASS`, `UNSW_TO_7CLASS`, `CICIDS_TO_7CLASS`) with fallback maps for common aliases. `fillna("UNKNOWN")` in categorical encoding. |
| Severity | **CRITICAL** — hard-stop terminates training on any invalid label detected; no automatic label correction. |

**Evidence locations:**

- `src/helix_ids/data/multi_dataset_loader.py`:198 — `ValueError` for UNSW non-numeric labels
- `src/helix_ids/data/multi_dataset_loader.py`:205 — `ValueError` for UNSW unresolved labels
- `src/helix_ids/data/multi_dataset_loader.py**:368 — `ValueError` for CICIDS missing label column
- `src/helix_ids/data/multi_dataset_loader.py**:439 — `ValueError` for NSL-KDD unresolved labels
- `src/helix_ids/data/multi_dataset_loader.py**:480 — `ValueError` for CICIDS unresolved labels
- `scripts/training/evaluation/evaluator.py`:352–355 — eval invalid label hard-stop
- `scripts/training/evaluation/evaluator.py`:603–605 — test invalid label hard-stop
- `scripts/training/execution/epoch_runner.py`:156–160 — batch diversity hard-stop
- `scripts/training/train_helix_ids_full.py`:2095 — `ValueError` for absent class-4
- `src/helix_ids/data/preprocessing.py**:177 — `fillna("UNKNOWN")` in categorical encoding

---

## 7. Device Mismatch (CPU/GPU)

| Field | Value |
|---|---|
| Detection | Not explicitly detected. Implicit `RuntimeError` from PyTorch when tensors on different devices interact. |
| Recovery | **Prevention by design.** Explicit `device` parameter throughout the stack: `HelixInferenceRuntime(device=...)`, `EpochRunner(device=...)`. Tensors explicitly moved with `.to(self._device, non_blocking=True)`. Model moved with `.to(self.device)`. Checkpoint loaded with `map_location="cpu"` for safe CPU-side deserialization. |
| Severity | **MEDIUM** — well prevented by design, but an unhandled mismatch between trainer device and checkpoint device would crash. |

**Evidence locations:**

- `scripts/training/execution/epoch_runner.py`:149–151 — `x.to(self._device, non_blocking=True)`
- `src/helix_ids/operations/inference_runtime.py**:443 — `self.device = torch.device(device)`
- `src/helix_ids/operations/inference_runtime.py**:463 — `self.model.to(self.device)`
- `src/helix_ids/operations/inference_runtime.py**:447 — `map_location="cpu"` in `torch.load()`
- `scripts/operations/serve_rest.py**:234 — `--device` CLI argument (default `"cpu"`)
- `scripts/training/execution/epoch_runner.py**:30 — `device: torch.device` parameter

---

## 8. Network Timeouts

| Field | Value |
|---|---|
| Detection | `TimeoutError` caught in `staging_gate_check.py` (URL fetch to Prometheus metrics endpoint). `TimeoutError` caught in `traffic_expansion_guard.py`. `TimeoutError` caught in `visualize_helix_demo.py`. Stage-level timeout contracts in `GovernanceOrchestrator` (preload/presplit/pretrain/intrain/posteval/prepromote). |
| Recovery | **Graceful failure.** Staging gate returns exit code 1 with `BLOCKED` status, `degraded_state=1`, `override_rate=nan`. Governance orchestrator raises `GateTimeoutError` with structured failure event logged to `FailureMemory`. Polling with configurable timeout (default 15s for Prometheus). |
| Severity | **MEDIUM** — well handled with clean error reporting; degraded/blocked status prevents incorrect promotions. |

**Evidence locations:**

- `scripts/operations/staging_gate_check.py`:30 — `urlopen(url, timeout=15)`
- `scripts/operations/staging_gate_check.py**:39 — `except (URLError, HTTPError, TimeoutError, OSError)`
- `scripts/operations/staging_gate_check.py**:40–44 — BLOCKED status output
- `scripts/operations/traffic_expansion_guard.py`:19 — `urlopen(url, timeout=15)`
- `scripts/operations/traffic_expansion_guard.py`:53 — timeout exception handler
- `src/helix_ids/governance/orchestrator.py`:184–346 — stage timeout contracts (preload/presplit/pretrain/intrain/posteval/prepromote)
- `src/helix_ids/governance/parameters.py`:10 — `StageTimeouts` dataclass

---

## 9. Disk Full

| Field | Value |
|---|---|
| Detection | Not explicitly detected. Generic `OSError` caught in network operations and benchmarks, but not for write failures. No disk-quota check or low-disk-space warning exists in the codebase. |
| Recovery | **None.** No explicit handling. `OSError` on write would propagate as an unhandled exception. |
| Severity | **HIGH** — no detection or recovery; a full disk during checkpoint saving or data caching would cause silent data loss or process crash. |

**Evidence locations:**

- No `shutil.disk_usage()`, `psutil.disk_usage()`, or equivalent checks found.
- `OSError` only caught in: `staging_gate_check.py`:39, `traffic_expansion_guard.py`:53, `benchmarks.py`:195 — all for network or subprocess failures, not disk writes.
- Checkpoint writes in `lifecycle_verifier.py`:138 and inference runtime:447 have no disk-space checks.

---

## 10. CUDA Errors

| Field | Value |
|---|---|
| Detection | Not explicitly caught. CUDA errors (`cuda.OutOfMemoryError`, `cuda.RuntimeError`, `cuda.CudaError`) propagate as unhandled exceptions. CUDA availability is checked only for ONNX export (`CUDAExecutionProvider` in `export.py`). |
| Recovery | **None.** No `try/except` for CUDA errors in training or inference. Default device is `"cpu"` in `serve_rest.py` — CPU fallback is config-driven, not an automatic fallback on CUDA failure. No gradient checkpointing or memory profiling to proactively avoid OOM. |
| Severity | **CRITICAL** — no detection, no recovery, no automatic CPU fallback on CUDA failure. |

**Evidence locations:**

- `scripts/operations/serve_rest.py`:234 — `--device` default `"cpu"`
- `src/helix_ids/utils/export.py**:616–618, 846–847 — `CUDAExecutionProvider` availability check (ONNX only)
- No `torch.cuda.is_available()` guards in training loops
- No `except (torch.cuda.OutOfMemoryError, ...)` handlers anywhere

---

## 11. Dataloader Worker Crashes

| Field | Value |
|---|---|
| Detection | Not explicitly caught. A worker crash in `torch.utils.data.DataLoader` with `num_workers > 0` propagates as a generic `RuntimeError` with `"DataLoader worker (pid=X) is killed by signal"`. |
| Recovery | **Mitigated by default config.** Most training scripts use `num_workers=0` (in-process loading), which avoids multiprocessing crashes entirely. `seed_worker()` function available for deterministic seeding when workers > 0. `reseed_dataloader_generator()` for reproducible shuffling. No automatic worker restart logic. |
| Severity | **MEDIUM** — mitigated by `num_workers=0` default, but no recovery path for systems configured with >0 workers. |

**Evidence locations:**

- `scripts/training/train_multidataset_v2_fixed.py`:804–812 — `num_workers=0`
- `scripts/evaluation/holdout_evaluation_v2.py`:67, 74 — `num_workers=0`
- `src/helix_ids/adaptation/online_finetune.py**:100 — `num_workers=0`
- `src/helix_ids/config/helix_full_config.py**:75 — `num_workers: int = 2` (configurable default)
- `src/helix_ids/governance/determinism.py**:59–63 — `seed_worker()` function
- `src/helix_ids/governance/determinism.py**:66 — `reseed_dataloader_generator()`
- `scripts/evaluation/test_phase3_smoke.py**:78 — `num_workers=0`

---

## Summary Table

| # | Failure Mode | Detection | Recovery | Severity |
|---|---|---|---|---|
| 1 | Dataset failures | `FileNotFoundError`, `SchemaDriftError`, `ValueError` | Fallback path traversal, graceful None return | **HIGH** |
| 2 | OOM | Unhandled `MemoryError`/`OutOfMemoryError` | None (batch-size config only) | **CRITICAL** |
| 3 | NaN propagation | `torch.isfinite()` checks, logit saturation warnings | Clamp [-10,10], gradient clipping, NaN→0 | **HIGH** |
| 4 | Corrupt checkpoints | 9-field contract validation, sidecar hash checks | Hard failure (no repair) | **CRITICAL** |
| 5 | Missing config | `FileNotFoundError`, `KeyError`, `ValueError` | Dataclass defaults, default policies | **MEDIUM** |
| 6 | Invalid labels | `ValueError` in loaders, hard-stop `RuntimeError` | Hard-stop (no correction); `disable_integrity_hard_stops` flag | **CRITICAL** |
| 7 | Device mismatch | Implicit PyTorch `RuntimeError` | Prevention via explicit device parameters | **MEDIUM** |
| 8 | Network timeouts | `TimeoutError` in gate scripts, governance stage timers | BLOCKED status, exit code 1, gate timeout error | **MEDIUM** |
| 9 | Disk full | Unhandled `OSError` on write | None | **HIGH** |
| 10 | CUDA errors | Unhandled CUDA exceptions | None (CPU default is config-based only) | **CRITICAL** |
| 11 | Dataloader worker crashes | Unhandled `RuntimeError` | Mitigated by `num_workers=0` default | **MEDIUM** |

---

## Risk Assessment

### Critical Severity (4 failure modes)
- **OOM**, **Corrupt checkpoints**, **Invalid labels**, **CUDA errors** — all lack runtime recovery mechanisms and cause process-terminating failures. OOM and CUDA errors have no detection at all. These represent the highest operational risk for production deployments.

### High Severity (3 failure modes)
- **Dataset failures**, **NaN propagation**, **Disk full** — detected (or partially detected) but recovery is limited or absent. Disk full is particularly concerning as it can silently corrupt checkpoint saves.

### Medium Severity (4 failure modes)
- **Missing config**, **Device mismatch**, **Network timeouts**, **Dataloader worker crashes** — either well-mitigated by design defaults or handled gracefully with clean error reporting.

### Key Gaps
1. **No OOM/CUDA error handling** anywhere in the codebase — critical for GPU deployments.
2. **No disk-full detection** — dangerous for long training runs and checkpoint-heavy workflows.
3. **No automatic checkpoint repair or fallback** — corrupt checkpoints are total losses.
4. **NaN recovery is limited** — clamping prevents symptom spread but doesn't address root cause or repair corrupted gradients.
