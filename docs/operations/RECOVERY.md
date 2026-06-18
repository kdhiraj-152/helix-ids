# Recovery

> Last updated: 2026-06-18  
> Runtime failure modes, detection mechanisms, and recovery strategies.

## Failure Mode Inventory

### 1. Dataset Failures

| Aspect | Details |
|--------|---------|
| Detection | `FileNotFoundError` from MultiDatasetLoader after exhausting candidate paths; `SchemaDriftError` for feature-order mismatches; `ValueError` for label-space mismatches; `validate_no_nan_inf()` catches NaN/inf |
| Recovery | Sequential fallback path traversal (3-4 candidate paths per dataset); CICIDS caches cleaned CSV; `load_cicids()` returns `None` for graceful degradation |
| Severity | **HIGH** — blocks training when all paths fail |

### 2. OOM (Out of Memory)

| Aspect | Details |
|--------|---------|
| Detection | Not explicitly caught — relies on Python `MemoryError` or CUDA OOM |
| Recovery | No explicit recovery. Batch sizes configured per variant as implicit prevention (nano=256, lite=128, full=64). No gradient checkpointing |
| Severity | **CRITICAL** — no detection or recovery path |

### 3. NaN Propagation

| Aspect | Details |
|--------|---------|
| Detection | `torch.isfinite()` check on inference outputs; logit saturation warnings when `|logit| > 10.0`; NaN/inf detected post-harmonization |
| Recovery | Prevention: logits clamped to [-10.0, 10.0]; gradient clipping (default 1.0); NaN→0.0 substitution; Inf→1e6/-1e6; temperature clamping at 1e-6 |
| Severity | **HIGH** — detected but no active repair; clamping masks limited ingress only |

### 4. Coverage Override Rate Spike

| Aspect | Details |
|--------|---------|
| Detection | `helix_coverage_override_rate` metric; `helix_degraded_state` auto-flip at 0.02 |
| Recovery | `traffic_expansion_guard.py` halts traffic increase; `staging_gate_check.py` blocks deployment |
| Severity | **HIGH** — detected, guarded, automated |

### 5. Checkpoint Corruption

| Aspect | Details |
|--------|---------|
| Detection | Post-write verification via `verify_export_artifact()`; contract integrity via `verify_contract_integrity()`; manifest verification via `verify_artifact_manifest()` |
| Recovery | Atomic writes (temp file + `os.replace`); corruption detected at verification step, artifact rejected |
| Severity | **MEDIUM** — detected but requires re-run to regenerate |

### 6. Thread/Process Crash

| Aspect | Details |
|--------|---------|
| Detection | Process exit; liveness check failure at `/health` |
| Recovery | No automatic restart. Supervised deployments expected to restart the process externally |
| Severity | **MEDIUM** — depends on external supervision |

## Recovery Manager

`RecoveryManager` (`src/helix_ids/operations/recovery_manager.py`) handles structured phase recovery during training:

- **Phase recovery** — Resets all phase flags (representation, head, joint-finetune)
- **Representation state** — Resets centroids, EMA state, snapshot IDs, geometry thresholds
- **Window pattern** — Applies curriculum schedule, cluster relabel settings
- **Centroid manager** — Resets EMA state, frozen centroids

Tested by `test_recovery_manager.py` (15 tests covering core settings, phase steps, multipliers, energy settings, edge cases).

## Certification Matrix

| Recovery Type | Status | Detail |
|--------------|--------|--------|
| Save — model state | ✅ **CERTIFIED** | Full contract + manifest + sidecars + post-write verification |
| Save — optimizer state | ❌ NOT IMPLEMENTED | Not captured in checkpoint |
| Save — scheduler state | ❌ NOT IMPLEMENTED | Phase orchestration is stateless |
| Save — RNG state | ❌ NOT IMPLEMENTED | Determinism seeded at start only |
| Save — metrics | ✅ **CERTIFIED** | Persisted atomically per seed |
| Resume integrity | ❌ NOT IMPLEMENTED | Training always starts from epoch 0 |
| Phase recovery | ✅ **CERTIFIED** | Full state reset via RecoveryManager |
| Governance metadata | ✅ **CERTIFIED** | Full provenance chain preserved |
| Data loader state | ❌ NOT IMPLEMENTED | Not saved or restorable |
