# Active System Overview

**Last updated:** 2026-06-20  
**Purpose:** New agent orientation to the production system. You should understand how HELIX-IDS works within 2 minutes.

## Active Model

**HelixIDS-Full** — Multi-task MLP backbone with threat-weighted loss.

- Source: `src/helix_ids/models/helix_ids_full.py`
- Wrapper: `src/helix_ids/models/full.py` (re-exports for convenience)
- Edge variants (Nano, Lite) are generated via quantization from the same training pipeline

## Active Training Entrypoint

**`scripts/training/train_helix_ids_full.py`** — Sole production training pipeline.

```bash
PYTHONPATH=src python scripts/training/train_helix_ids_full.py \
    --config config/experiments/smoke.yaml \
    --output /path/to/output \
    --device cpu
```

## Active Inference Entrypoint

**`scripts/operations/serve_rest.py`** — REST inference server.

```bash
PYTHONPATH=src python scripts/operations/serve_rest.py \
    --checkpoint models/helix_full/helix_full_nsl_kdd_best.pt \
    --host 127.0.0.1 --port 8080 --device cpu
```

## Active Loss Implementation

Self-contained in `src/helix_ids/models/helix_ids_full.py` (class `MultiTaskLoss`, line 187).
The legacy `MultiTaskLoss` in `src/helix_ids/models/loss.py` is a separate implementation used only by the superseded helix_ids model.

## Active Adaptation Path

**`src/helix_ids/data/feature_harmonization.py`** — Cross-dataset feature mapping to 17 canonical features.

There is NO active domain-adaptation layer during training. The production pipeline uses the learnability contract system (`src/helix_ids/data/learnability_contract.py`) to validate data quality before training.

## Active Artifact Generation Path

- Checkpoints: `models/helix_full/` (governed, with provenance manifests)
- Quantized models: `models/quantized/`
- Benchmark results: `results/benchmarks/`
- Training/eval results: JSON per seed in `results/`

## Active Governance Path

- Entrypoint: `src/helix_ids/governance/entrypoint.py` — Non-bypassable governance wrapping
- Determinism: `src/helix_ids/governance/determinism.py`
- Lifecycle: `src/helix_ids/governance/lifecycle_verifier.py`
- Promotion: `src/helix_ids/governance/promotion.py` — Multi-seed consensus
- Staging gate: `scripts/operations/staging_gate_check.py`

## Active Deployment Gate

- **Coverage override rate** ≤ 0.02
- **Degraded state** == 0
- **Provenance manifest** verified
- On failure: automatic rollback (not alert — rollback)

## Archived Subsystems (not part of production)

| Subsystem | Why Archived | Current Location |
|-----------|-------------|------------------|
| `src/helix_ids/adaptation/` | Superseded by `data/feature_harmonization.py` | `archive/phase24a/src/helix_ids/adaptation/` |
| `src/helix_ids/models/helix_ids.py` | Superseded by `helix_ids_full.py` | Still in source (Phase 24B second-pass candidate) |
| `src/helix_ids/models/classifier.py` | Legacy hierarchical classifier | Still in source (Phase 24B second-pass candidate) |
| `src/helix_ids/models/attention.py` | Legacy temporal attention module | Still in source (Phase 24B second-pass candidate) |
| `src/helix_ids/models/loss.py` | Legacy loss functions (superseded by self-contained `MultiTaskLoss` in `helix_ids_full.py`) | Still in source (Phase 24B second-pass candidate) |
| `src/helix_ids/models/adaptation/` | Legacy domain adaptation framework (3,393 lines) | Still in source (Phase 24B candidate) |
| `scripts/training/train_multidataset.py` | Legacy multi-dataset training (supplanted by `train_helix_ids_full.py`) | Still in source (Phase 24B candidate) |
| `src/helix_ids/data/data_audit.py` | Not used in production pipelines | `archive/phase24a/src/helix_ids/data/data_audit.py` |
| `scripts/training/train_unified_rebalanced.py` | Legacy direct adaptation runner | `archive/phase24a/scripts/training/train_unified_rebalanced.py` |

## Future Fine-Tuning Location

New training scripts, models, or fine-tuning pipelines should be placed in:

- Model: `src/helix_ids/models/` (new file, `v2_*` or `tuned_*` pattern)
- Training: `scripts/training/` (new file, `train_*_v2.py` pattern)
- Configuration: `config/experiments/` (new YAML config)

## Quick Command Reference

```bash
# Train
PYTHONPATH=src python scripts/training/train_helix_ids_full.py --config config/experiments/smoke.yaml --output /tmp/test_output --device cpu

# Serve
PYTHONPATH=src python scripts/operations/serve_rest.py --checkpoint models/helix_full/helix_full_nsl_kdd_best.pt --host 127.0.0.1 --port 8080 --device cpu

# Stage gate check
PYTHONPATH=src python scripts/operations/staging_gate_check.py

# Run tests
pytest -q

# Run benchmarks
PYTHONPATH=src python scripts/evaluation/benchmarks.py
```
