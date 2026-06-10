# Phase 4A — Reproducibility Gap Analysis

**Date:** 2026-06-03
**Purpose:** Document current reproducibility guarantees and gaps. No new functionality.

---

## Reproducibility Guarantee Matrix

| Feature | Status | Implementation | Gap / Limitation |
|---------|--------|---------------|-----------------|
| **Random seed control** | Implemented | `helix_ids.governance.parameters.py` — `StageTimeouts.pretrain_seconds`, seed CLI arg | Single-seed runs fail promotion (`E-T3-SINGLE-SEED-INVALID`) — requires `min_seed_runs=3` |
| **Torch deterministic flags** | Implemented | `scripts/training/train_helix_ids_full.py` — `torch.use_deterministic_algorithms`, `cudnn_deterministic`, `cudnn_benchmark` | Platform-dependent — CUDA-only deterministic guarantee |
| **Python hash seed** | Implemented | `PYTHONHASHSEED` set via env var in governance entrypoint | Not validated in result reproducibility_metadata |
| **Data split reproducibility** | Implemented | `helix_ids.data.multi_dataset_loader` — deterministic split with seed | Split hash tracked in manifest |
| **Feature harmonization determinism** | Implemented | `helix_ids.data.feature_harmonization` — canonical ordering (sort_keys, separators) | No separate feature-order hash; relies on schema_hash |
| **Manifest reproducibility** | Implemented | `scripts/evaluation/benchmarks.RunSpec.run_id()` — deterministic ID from hash of config/dataset/governance | Deterministic across pass order (tested in `test_benchmark_manifest_expansion_is_deterministic`) |
| **Result lineage completeness** | Implemented | `result_schema_governance.md` §3 — all required lineage fields enforced by `validate_benchmark_outputs.py` | None |
| **Reproducibility metadata in result** | Implemented | `result_schema_governance.md` §4 — seed + determinism object | `torch_deterministic_algorithms` value not independently verified |
| **Multi-seed consensus** | Implemented | Promotion contract in `lifecycle_verifier.py` — `min_seed_runs=3`, `inter_seed_macro_f1_variance` threshold | Single-seed training passes all gates but fails promotion |
| **Multi-run variance tracking** | Implemented | `inter_seed_macro_f1_variance`, `reproducibility_delta` in `prepromote` stages | Computed but no formal upper-bound threshold defined |
| **Training reproducibility report** | Partially implemented | `results/gates/` artifacts — prometheus metrics, eval results | No formal reproducibility certification document |
| **Checkpoint reproducibility** | Implemented | Provenance sidecar (`artifact_sha256`) — verified on load | No separate training reproducibility claim |
| **Hash algorithm stability** | Implemented | SHA-256 throughout; canonical JSON encoding (sorted keys, compact separators) | Hash algorithm change requires migration (documented in `hash_authority.md`) |
| **Frozen data pipeline** | Partially implemented | Dataset hashes (raw, processed, split, primary) in manifest | Processed hash includes all pipeline steps; individual step hashes not tracked separately |
| **Frozen code snapshot** | Implemented | `git_commit` in result lineage + AST validator enforcing no `__import__`, `eval`, `exec` | Git state only; no frozen pip/conda environment spec |
| **Frozen dependencies** | Missing | — | No `requirements.lock` or equivalent frozen env snapshot in lineage |
| **Provenance-locked run** | Partially implemented | Fingerprinting pipeline + run registry | No cryptographic provenance chain (blockchain/notarization) |
| **Reproducibility CI gate** | Implemented | `validate_benchmark_outputs.py` checks manifest result completeness | No dedicated reproducibility test proving same seed → same output |
| **Dataset artifact freeze** | Partially implemented | Raw, processed, split hashes tracked; no artifact pinning service | Hash confirms content but not location immutability |
| **Training log reproducibility** | Missing | — | Logs are written but not hashed into provenance chain |

---

## Summary

| Category | Count |
|----------|-------|
| Implemented | 13 |
| Partially implemented | 5 |
| Missing | 2 |
| **Total** | **20** |

### Partially Implemented Details

1. **Training reproducibility report** — Promotes/reports metrics but no formal reproducibility certificate
2. **Frozen data pipeline** — Hashes confirm content; individual pipeline step hashes not tracked
3. **Provenance-locked run** — Fingerprinting + registry but no cryptographic notary
4. **Dataset artifact freeze** — Content hash confirms immutability; location pinning not enforced
5. **Reproducibility metadata** — Field presence enforced; determinism flag values not independently verified

### Missing Details

1. **Frozen dependencies** — No pip/conda lock file in provenance chain
2. **Training log reproducibility** — Logs not incorporated into provenance hash

---

## Future Provenance-Locking Roadmap

See `docs/governance/ADR-001-governance-philosophy.md` for formal ADRs.