# HELIX-IDS Reproducibility Audit

> **Audit date:** 2026-06-16  
> **Scope:** All reproducibility mechanisms claimed or implemented for HELIX-IDS training pipeline  
> **Methodology:** Source code evidence traced for each mechanism; classification based on actual implementation, not documentation

## Classification Legend

| Label | Meaning |
|-------|---------|
| **PASS** | Mechanism correctly implemented and actively used in the pipeline |
| **WARNING** | Mechanism present but has caveats, incomplete coverage, or known limitations |
| **FAIL** | Mechanism absent, broken, or misimplemented |

---

## Audit Table

| # | Item | Mechanism | Classification | Evidence |
|---|------|-----------|----------------|----------|
| 1 | **Random seed** | `random.seed(seed)` | ✅ PASS | `src/helix_ids/governance/determinism.py` line 37 — called inside `set_global_determinism()`. Invoked at `scripts/training/train_helix_ids_full.py` line 4473: `set_global_determinism(args.seed)`. Also seeded in worker processes via `seed_worker()` at line 63. |
| 2 | **NumPy seed** | `np.random.seed(seed)` | ✅ PASS | `src/helix_ids/governance/determinism.py` line 38 — main seed via `set_global_determinism()`. Additionally reseeded per-worker via `seed_worker()` at line 62 with a derived `worker_seed`. Also uses `np.random.default_rng(seed)` at `train_helix_ids_full.py` line 221 and `run_orchestrator.py` line 494 for specific sampling tasks. |
| 3 | **Torch seed** | `torch.manual_seed(seed)` | ✅ PASS | `src/helix_ids/governance/determinism.py` line 40 — called unconditionally in `set_global_determinism()`. Also CUDA variants at lines 42-43 guarded by `torch.cuda.is_available()`. |
| 4 | **CUDA deterministic** | `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic = True`, `cudnn.benchmark = False` | ✅ PASS | `src/helix_ids/governance/determinism.py` lines 46-48 — all three flags set unconditionally (safe on non-CUDA backends). CUDA manual seeds at lines 41-43 with `is_available()` guard. State captured in `DeterminismState` dataclass (lines 14-30). |
| 5 | **Dataloader seed_worker** | `worker_init_fn=seed_worker` | ✅ PASS | `src/helix_ids/governance/determinism.py` lines 59-63 — `seed_worker()` defined to seed `np.random` and `random` per-worker with `(torch.initial_seed() + (worker_id + 1) * 1009) % 2**32`. Used in `scripts/training/orchestration/run_orchestrator.py` at lines 566 (train), 623 (val), 637 (test). Also used in legacy `scripts/training/train_multidataset_v2_fixed.py`. |
| 6 | **Dataloader generator reseeding** | `torch.Generator().manual_seed()` + `reseed_dataloader_generator()` per epoch | ⚠️ WARNING | Initial generators seeded at `run_orchestrator.py` lines 475-477: `train_loader_generator = torch.Generator().manual_seed(args.seed)`, `val_loader_generator = torch.Generator().manual_seed(args.seed + 1)`, `test_loader_generator = torch.Generator().manual_seed(args.seed + 2)`. Per-epoch reseeding via `HelixFullTrainer._reseed_epoch_generators()` at `train_helix_ids_full.py` lines 3490-3507. **Caveat:** When `forced_sampler_mode` is `"interleaved_rr"` or `"weighted_random_sampler"`, the DataLoader `generator=` is explicitly set to `None` (`run_orchestrator.py` lines 567-571), making the per-epoch `reseed_dataloader_generator()` a no-op (function returns early when generator is `None` at `determinism.py` line 68-69). The `WeightedRandomSampler` itself receives the seeded generator (line 552), but it is not re-seeded per epoch — it draws from its initial state each epoch. Partial coverage. |
| 7 | **PYTHONHASHSEED** | `os.environ["PYTHONHASHSEED"] = str(seed)` | ⚠️ WARNING | `src/helix_ids/governance/determinism.py` line 35 — set in `set_global_determinism()`. **Known limitation:** On CPython, `PYTHONHASHSEED` only takes effect **before interpreter startup**; setting it via `os.environ` after Python has started has **no effect** on `str`/`bytes` hash randomization for `dict`/`set`. The value is captured in `DeterminismState.python_hash_seed` (line 22) but is not actually enforced. The `os.environ` set is still useful for subprocesses spawned after this point. |
| 8 | **Checkpoint save determinism** | `torch.save()` with deterministic metadata | ✅ PASS | `scripts/training/train_helix_ids_full.py` `_write_checkpoint_artifact()` at lines 538-568: saves with `model_state_dict` (detached, CPU, cloned), `schema_hash`, `feature_order_hash`, `ARTIFACT_MANIFEST_KEY`. Sidecar files written: `.contract.json`, `.feature_order.json`, `.schema_hash.txt` (lines 524-535). Manifest via `build_export_manifest()` + `checkpoint_manifest_payload()`. Intermediate checkpoints at `_save_checkpoint_if_needed()` (lines 4408-4427). Best model state tracked as deep copy (line 4403). **Minor:** No explicit pickle protocol pinning (e.g. `pickle_protocol=2`) — Python version differences may affect byte-exact reproducibility of the `.pt` file. |
| 9 | **Checkpoint load determinism** | `torch.load()` with model reconstruction | ⚠️ WARNING | `scripts/training/governance/orchestrator.py` line 179: `artifact = torch.load(model_path, map_location="cpu", weights_only=True)` — `weights_only=True` is good for security. Model is reconstructed from config (lines 181-184) rather than from a saved architecture. No explicit CUDA seed reset or `torch.manual_seed()` call after loading before running evaluation/inference. No deterministic `map_location` strategy documented — `"cpu"` is hardcoded. |
| 10 | **Config fingerprinting** | SHA-256 hashing of configuration and schema | ✅ PASS | `src/helix_ids/governance/fingerprinting.py`: `build_run_fingerprint()` (lines 58-75) — hashes `dataset_hashes`, `mapping_version`, `schema_hash`, `model_config_hash`, `commit_sha` into canonical JSON → SHA-256. `canonical_json_hash()` (lines 17-20) uses `json.dumps(sort_keys=True, separators=(",", ":"))`. `build_dataset_manifest_hash()` (lines 23-36) hashes file contents. `src/helix_ids/contracts/schema_contract.py`: `compute_schema_hash()` (lines 52-68) — SHA-256 of canonical JSON payload. `src/helix_ids/data/learnability_contract.py`: `compute_schema_hash()` (lines 314-320) — SHA-256 of feature columns + transformations. `src/helix_ids/governance/run_registry.py`: fingerprint stored on every registry record (lines 134, 241, 408) and validated for consistency across runs (lines 148-173). |

---

## Summary

### PASS (7/10)

| Mechanism | Status | Notes |
|-----------|--------|-------|
| Random seed | ✅ PASS | `random.seed()` in `set_global_determinism` + `seed_worker` |
| NumPy seed | ✅ PASS | `np.random.seed()` in both global + per-worker |
| Torch seed | ✅ PASS | `torch.manual_seed()` + CUDA variants |
| CUDA deterministic | ✅ PASS | `use_deterministic_algorithms`, `cudnn.deterministic`, `cudnn.benchmark` |
| Dataloader seed_worker | ✅ PASS | `worker_init_fn=seed_worker` wired on all loaders |
| Checkpoint save determinism | ✅ PASS | `torch.save` with metadata, sidecars, manifest |
| Config fingerprinting | ✅ PASS | SHA-256 canonical JSON across schema, dataset, model config |

### WARNING (3/10)

| Mechanism | Status | Issue |
|-----------|--------|-------|
| Dataloader generator reseeding | ⚠️ WARNING | Per-epoch reseeding is a no-op when `forced_sampler_mode` is `interleaved_rr`/`weighted_random_sampler` because DataLoader `generator=` is `None` |
| PYTHONHASHSEED | ⚠️ WARNING | Set in-process after startup — has no effect on CPython hash randomization for the current process |
| Checkpoint load determinism | ⚠️ WARNING | No seed reset after load; model reconstructed from config, not from saved architecture; pickle protocol not pinned |

### FAIL (0/10)

All ten mechanisms have at least a partial implementation. No mechanism is completely absent or broken.

---

## Verdict

**PASS** with **3 warnings** requiring attention before the pipeline can be considered fully reproducible.

The core seed-enforcement infrastructure (`set_global_determinism`, `seed_worker`, `reseed_dataloader_generator`) is well-designed and correctly implemented. The three warnings represent edge cases and operational gaps rather than fundamental design flaws.

### Recommended Remediations

1. **PYTHONHASHSEED enforcement** — Document that `PYTHONHASHSEED` must be set **before** launching the Python process (e.g., via `env PYTHONHASHSEED=42 python scripts/...`). Add a startup warning if the current value doesn't match the configured seed.

2. **Generator reseeding coverage** — When using `WeightedRandomSampler` with a seeded generator, the sampler's internal state should be re-seeded per epoch, or the DataLoader `generator=` should be propagated so `reseed_dataloader_generator()` works. Alternatively, document which sampler modes guarantee per-epoch determinism.

3. **Checkpoint load determinism** — Call `set_global_determinism(seed)` after loading a checkpoint before running evaluation. Pin `pickle_protocol` in `torch.save` calls for byte-identical checkpoints across Python versions.

---

*Generated by REPRODUCIBILITY_AUDIT.md — source-code traced audit for HELIX-IDS*
