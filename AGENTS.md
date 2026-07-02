# AGENTS.md — Master Context for AI Coding Agents

> Domain: Network Intrusion Detection · Framework: PyTorch · Codebase: ~7.5K LOC (core), ~12K LOC (scripts+tests)
> Last updated: 2026-06-28

---

## 1. Project Identity

**HELIX-IDS**: Hierarchical Edge-optimized Lightweight Intrusion eXpert — a production-ready network intrusion detection system (IDS) targeting resource-constrained edge devices (ESP32, Raspberry Pi Zero/4).

**Primary mission**: Solve the minority class suppression problem in NIDS — where rare but dangerous attacks (R2L, U2R) achieve F1=0.000 with standard cross-entropy — while maintaining edge deployability (<30KB models for ESP32).

**Paper**: IEEE-format manuscript at `docs/manuscript/HELIX_submission_ready.md`

---

## 2. Architecture Overview

### High-Level Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                      Operational Scripts                         │
│  scripts/training/   scripts/operations/   scripts/evaluation/  │
│  scripts/deployment/  scripts/benchmarks/   scripts/data/       │
│  scripts/ci/         scripts/analysis/                          │
├─────────────────────────────────────────────────────────────────┤
│                     Core Package (src/helix_ids/)                │
│  models/     data/     training/     operations/    governance/  │
│  contracts/  metrics/  config/       utils/                     │
├─────────────────────────────────────────────────────────────────┤
│                    Data & Artifact Layer                          │
│  config/     data/     models/     results/     benchmarks/      │
│  artifacts/  checkpoints/                                        │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Constraint
**NO reverse dependencies**: `src/helix_ids/` must NEVER import from `scripts/`. This is enforced as a CI gate (`tests/architecture/test_no_reverse_dependencies.py`). Scripts import from `src/helix_ids`, never the reverse.

### Input/Output Contract
- **Input**: 17 harmonized flow features (defined in `CANONICAL_FEATURE_ORDER`, `src/helix_ids/contracts/schema_contract.py`)
- **Binary output**: 2-class (Normal=0, Attack=1)
- **Family output**: 7-class (Normal=0, DoS=1, Probe=2, R2L=3, U2R=4, Generic=5, Backdoor=6)
- **Schema version**: `2026-05-25` — any change requires hash recomputation

---

## 3. Complete Module Map

### src/helix_ids/ — Core Package (the package)

| Module | File | Purpose |
|--------|------|---------|
| **models** | `helix_ids_full.py` | HelixIDSFull model: 4-layer MLP backbone (512→384→256→256), binary + family heads |
| | `classifier.py` | Hierarchical classifier (binary → family → fine-grained 23-class) |
| | `attention.py` | Temporal Attention Module (TAM) variants: Nano (2 heads/32), Lite (4/48), Full (4/64) |
| | `loss.py` | ThreatAwareFocalLoss, CalibrationLoss, MultiTaskLoss with curriculum learning |
| | `coral_loss.py` | CORAL (CORrelation ALignment) for domain adaptation |
| | `full.py` | Canonical re-export alias for HelixIDSFull, HelixFullConfig, MultiTaskLoss |
| **data** | `feature_harmonization.py` | **THE critical file** — cross-dataset feature engineering, 41 raw → 17 canonical features |
| | `unified_loader.py` | Multi-dataset loading: NSL-KDD, UNSW-NB15, CICIDS-2017/2018, TON-IoT, Bot-IoT |
| | `augmentation.py` | AttackAwareAugmentation: SMOTE-like oversampling for minority classes |
| | `learnability_contract.py` | Schema hash + data quality validation contract |
| **contracts** | `schema_contract.py` | **IMMUTABLE** — CANONICAL_FEATURE_ORDER (17 features), SCHEMA_VERSION, SCHEMA_HASH |
| | `attack_taxonomy.py` | Single authoritative source for 5-class/7-class attack family definitions + mappings |
| | `immutable_constants.py` | CONTRACT_VERSION, EXPORTER_API_VERSION, FEATURE_ORDER_HASH, MANIFEST_VERSION |
| | `diagnostic_contract.py` | Decision transition enforcement (DECISION_MODES, DECISION_TRANSITIONS) |
| **governance** | `gate_orchestrator.py` | GateOrchestrator — stage-based training control (preload→presplit→pretrain→intrain→posteval→prepromote) |
| | `entrypoint.py` | `@governed_entrypoint` decorator — forces ALL entrypoints through governance gates |
| | `provenance.py` | Artifact manifests, SHA256 verification, deployment manifest, sidecar files |
| | `parameters.py` | GovernancePolicy: StageTimeouts, TrainingAbortPolicy, BootstrapPolicy, DriftPolicy, PromotionPolicy |
| | `fingerprinting.py` | Canonical JSON hashing |
| | `run_registry.py` | RunRegistry — cross-run reproducibility tracking |
| | `determinism.py` | Reproducibility: `reseed_dataloader_generator`, `set_global_determinism` |
| **operations** | `inference_runtime.py` | **THE deployment runtime** — loads checkpoint, runs inference with configurable overrides |
| | `monitoring.py` | Prometheus metrics, coverage override tracking |
| | `circuit_breaker.py` | Circuit breaker pattern for degraded state |
| **training** | `helix_trainer.py` | HelixFullTrainer — training orchestration (if extracted) |
| | `coral_loss.py` | Domain adaptation loss (CORAL) |
| **metrics** | `metrics.py` | ModelMetrics, PRI (Production Readiness Index), bootstrap CI, evaluate() |
| **config** | `helix_full_config.py` | TrainingConfig dataclass parsed from YAML |
| **utils** | `export.py` | ONNX/TorchScript export with manifest embedding, artifact verification |

### scripts/ — Operational Scripts

| Directory | Script | Purpose |
|-----------|--------|---------|
| **training/** | `train_helix_ids_full.py` (4,605 LOC) | **THE main training pipeline** — the largest file. Multi-dataset, multi-task, governed training |
| | `core/` | RecoveryManager, TrainerFacade, TrainerFactory, TrainerState |
| | `data/` | DatasetBuilder, MultiTaskNumpyDataset, Samplers, Validators |
| | `evaluation/` | Evaluator, EvaluationOrchestrator |
| | `_constants.py` | Training-layer feature engineering constants (ENGINEERED_FEATURE_NAMES) |
| **operations/** | `serve_rest.py` | FastAPI REST server with Prometheus metrics, class-margin overrides |
| | `staging_gate_check.py` | Prometheus-based promotion gate: checks override_rate ≤ 0.02 |
| **deployment/** | Deployment pipeline entrypoints | |
| **evaluation/** | Validation and benchmark pipelines | |
| **benchmarks/** | `benchmark_pipeline.py` | Performance/load/soak testing |
| **ci/** | CI validators: trainer_size_check, reverse_deps_check, cycle_check, enforce_skip_governance, mutation analysis |

### config/

| File | Purpose |
|------|---------|
| `helix_config.yaml` | Model variants (nano/lite/full), curriculum schedule, evaluation targets, export formats |
| `mutation/cosmic-ray-*.toml` | Mutation testing configs |

### tests/ — 110+ test files

| Directory | Focus |
|-----------|-------|
| `tests/architecture/` | **Critical** — architecture freeze, dependency cycles, reverse deps, trainer boundary |
| `tests/test_governance/` | AST validation, fingerprinting, orchestrator runtime, promotion, run registry |
| `tests/test_operations/` | Inference runtime, serve_rest metrics, monitoring, staging gate |
| `tests/test_models/` | HelixFull, HelixIDS, Loss functions |
| `tests/test_data/` | Phase1 harmonization, unified loader, learnability contracts |
| `tests/training/` | Trainer extraction, checkpoint recovery, orchestration components |
| `tests/test_utils/` | Metrics |
| Root tests | 50+ standalone test files covering every module |

---

## 4. Data Flow (End-to-End)

### Training Pipeline
```
Raw Datasets (NSL-KDD, UNSW-NB15, CICIDS, TON-IoT, Bot-IoT)
    │
    ▼
Feature Harmonization (41 raw → 17 canonical features)
    • feature_harmonization.py — THE critical transformation
    • Log transforms, ratio features, interaction features
    • Schema hash verifies feature order integrity
    │
    ▼
Unified Loader (load per dataset, unify to 17-feature schema)
    │
    ▼
Harmonized Dataset → MultiTaskNumpyDataset
    │
    ▼
Training (HelixFullTrainer)
    • Multi-task learning: binary (Normal/Attack) + family (7-class)
    • ThreatAwareFocalLoss with curriculum learning
    • Governance gates: preload → presplit → pretrain → intrain → posteval → prepromote
    • Deterministic seeds, reproducibility tracking
    │
    ▼
Checkpoint (.pt) → Manifest embedded
    │
    ▼
Export (ONNX / TorchScript / TFLite / C header)
    • Provenance manifest sidecar
    • SHA256 verification
```

### Inference Pipeline
```
Checkpoint (.pt) → HelixInferenceRuntime
    • Model auto-configuration from state_dict
    • Schema contract validation
    • Optional: coverage floor, class-margin overrides
    │
    ▼
REST API (FastAPI via serve_rest.py)
    • POST /predict (single/batch)
    • GET /health (schema_hash, contract_version)
    • GET /metrics (Prometheus format)
    │
    ▼
Monitoring & Gates
    • Coverage override tracking (threshold: 2% = degraded)
    • Class-margin adaptive thresholds for minority classes
    • Staging gate check before promotion
```

---

## 5. Model Architecture Details

### HelixIDSFull (production model)
```
Input: 17 features
  │
  ├─ Backbone: 4-layer MLP (512→384→256→256)
  │   • Linear + BatchNorm + ReLU + Dropout (0.3, 0.3, 0.25, 0.2)
  │
  ├── Binary Head: Linear(256→2)  →  Normal vs Attack
  │
  └── Family Head: Linear(256→7)  →  7-class attack families
```

### Attack Classification Taxonomy
- **5-class** (training): Normal, DoS, Probe, R2L, U2R
- **7-class** (production export): Normal, DoS, Probe, R2L, U2R, Generic, Backdoor
- **Threat weights** (training, conservative): Normal=1.0, DoS=1.2, Probe=1.5, R2L=3.0, U2R=4.0

### Model Variants (from config/helix_config.yaml)

| Variant | Target | Params | Size | Latency |
|---------|--------|--------|------|---------|
| Nano | ESP32 (520KB SRAM) | ~2.6K (TAM) | <30KB | <1ms |
| Lite | RPi Zero (512MB) | ~7.4K (TAM) | <200KB | <5ms |
| Full | RPi 4 (4GB) | ~500K | <2MB | <10ms |

### Loss Functions (Multi-task Curriculum)
```
Epochs 1-10:   α=1.0 binary only (CE warmup)
Epochs 11-30:  α=0.5 binary + β=0.5 family (focal loss)
Epochs 31-50:  α=0.3, β=0.4, γ=0.3 (add fine-grained)
Epochs 51+:    α=0.2, β=0.3, γ=0.3, δ=0.2 calibration
```

---

## 6. Governance System

The governance system wraps ALL training/inference entrypoints with deterministic gates.

### Stage Sequence
1. **preload** — Validate environment, schema, dependencies
2. **presplit** — Verify data splits, stratification
3. **pretrain** — Model initialization, determinism, seed
4. **intrain** — Training progress, abort conditions (low entropy, gradient dominance, no improvement)
5. **posteval** — Post-training evaluation, metrics
6. **prepromote** — Reproducibility check, run registry

### Key Policies
- **Bootstrap**: 2000 replicates, 95% CI, min CI95 lower bound = 0.50
- **Drift**: 20-run baseline window, max abs z-score = 2.5
- **Promotion**: 3 seed runs, max inter-seed variance = 0.01
- **Smoke profile**: `HELIX_GOV_POLICY_PROFILE=smoke` relaxes gates for fast CI

### Artifact Provenance
Every checkpoint/ONNX export carries:
- SHA256 hash of artifact
- Contract metadata (schema_version, schema_hash, feature_order_hash)
- Git commit, branch, dirty state
- Training timestamp, torch version
- Sidecar manifest file (`manifest.json`) + deployment manifest

---

## 7. CI/CD Pipeline

| Workflow | Trigger | Scope |
|----------|---------|-------|
| `ci.yml` | Push to dev, PR to main/dev | Ruff, mypy, pytest (fast subset), lockfile sync |
| `architecture.yml` | Push to dev, PR to main | Trainer size limit (≤2000 LOC), reverse deps, cycle check, architecture tests |
| `quality.yml` | Push to dev, PR to main | Full pytest with coverage ≥65%, benchmark regression, dependency audit |
| `nightly.yml` | Weekly Monday | CodeQL SAST, cross-python tests (3.9/3.10/3.11), mutation testing, assertion audit |
| `release.yml` | Release trigger | Full deployment pipeline |
| `dependency-review.yml` | PR | Supply chain security |

**Gate criteria** (from operations perspective):
- Coverage override rate > 2% → degraded state → block staging promotion
- Served by `scripts/operations/staging_gate_check.py`

---

## 8. Critical Invariants (DO NOT BREAK)

### 🔴 CRITICAL — Schema Contract
- **CANONICAL_INPUT_DIM = 17** — every model, every checkpoint, every runtime
- **CANONICAL_FAMILY_CLASSES = 7** — family head output
- **CANONICAL_BINARY_CLASSES = 2** — binary head output
- **SCHEMA_VERSION = "2026-05-25"** — ties to hash
- These are defined in `src/helix_ids/contracts/schema_contract.py` and enforced at:
  - Training: `assert_runtime_contract()` in trainer
  - Inference: `_validate_checkpoint_contract()` in inference_runtime
  - Export: manifest hash verification

### 🟡 HIGH — Architectural Boundaries
- `src/helix_ids/` must NEVER import from `scripts/` (enforced by CI)
- `train_helix_ids_full.py` must stay ≤ 2000 LOC (enforced in `architecture.yml` via trainer_size_check.py) — currently at 4,605 LOC, this is a known breach
- `AGENTS.md` specifies mission constraints: **no new features, no new scripts, no broad refactors** — minimal targeted fixes only

### 🟢 MEDIUM — Data/Feature Pipeline
- Feature harmonization order MUST match `CANONICAL_FEATURE_ORDER` exactly
- Training uses 17 features; legacy code paths may reference 41 — ALWAYS prefer the 17-feature canonical contract
- Dataset-specific attack label mappings are in `attack_taxonomy.py` — do NOT define parallel mappings in consumer modules

---

## 9. Common Pitfalls for AI Agents

### 1. Feature Dimension Confusion
The codebase has legacy references to 32 and 41 features. The **canonical input dim is 17**. When modifying any data pipeline:
- `feature_harmonization.py` produces 17 features
- Model config defaults to `input_dim=17`
- If you see 32 or 41, you're looking at legacy/dead code or a bug

### 2. Training Script is HUGE
`scripts/training/train_helix_ids_full.py` is 4,605 LOC — the largest file. It keeps growing despite extraction efforts. The CI gate limits it to 2000 but this is currently breached. Making targeted edits in this file requires careful navigation.

### 3. Lockfile Synchronization
`requirements-lock.txt` must be synchronized with `requirements.in`. CI enforces this. Use `uv pip compile` to regenerate when adding deps.

### 4. MPS Memory Contention (macOS)
On Apple Silicon: run inference soak AFTER training completes, not concurrently. MPS memory contention prevents parallel soaks.

### 5. Smoke Profile for Quick Testing
Set `HELIX_GOV_POLICY_PROFILE=smoke` to bypass strict governance gates during local development:
```bash
HELIX_GOV_POLICY_PROFILE=smoke python scripts/training/train_helix_ids_full.py --config config/helix_config.yaml
```

### 6. PYTHONPATH
Always set `PYTHONPATH=src` before running scripts, e.g.:
```bash
PYTHONPATH=src python scripts/training/train_helix_ids_full.py ...
```

### 7. Checkpoint Contract Validation
When loading checkpoints, `inference_runtime.py` enforces strict contract validation. If you get a `ValueError` about input_dim mismatch, the checkpoint was trained with a different feature schema.

---

## 10. Development Workflow

### Quick Start
```bash
# Setup
python3 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements-lock.txt

# Run tests
PYTHONPATH=src pytest -q

# Train model
PYTHONPATH=src python scripts/training/train_helix_ids_full.py \
  --config config/helix_config.yaml \
  --output models/helix_full

# Serve model
PYTHONPATH=src python scripts/operations/serve_rest.py \
  --checkpoint models/helix_full/helix_full_nsl_kdd_best.pt

# Run CI locally
ruff check src scripts tests
mypy src
pytest -q --ignore=tests/architecture --ignore=tests/training
```

### Validation Commands
```bash
pytest -q                                          # primary
pytest tests/operations -q                         # operations subset
ruff check src scripts tests                       # lint
mypy src                                            # type check (src only)
PYTHONPATH=src python .github/scripts/reverse_deps_check.py  # architecture
```

### Code Review Workflow (code-review-graph)
1. `mcp_code-review-g_get_minimal_context_tool`
2. `mcp_code-review-g_build_or_update_graph_tool(full_rebuild=true, postprocess="minimal")`
3. `mcp_code-review-g_get_review_context_tool`

---

## 11. Datasets Used

| Dataset | Classes | Training Samples | Variants Harmonized |
|---------|---------|-----------------|---------------------|
| NSL-KDD | 5-class + 23 fine | ~125K | KDDTrain+, KDDTest+ |
| UNSW-NB15 | 10 raw → 5-class | ~2.5M | Full training/test |
| CIC-IDS-2017 | 15 raw → 7-class | ~2.8M | Day-wise CSVs |
| CIC-IDS-2018 | 15 raw → 7-class | ~16M | Day-wise CSVs |
| TON-IoT | 10 raw → 5-class | ~461K | Train/test splits |
| Bot-IoT | 5 raw → 5-class | ~73M | Subsampled |

All dataset → 5-class mapping is in `src/helix_ids/contracts/attack_taxonomy.py`.

---

## 12. Research Context

The project has undergone extensive research phases (documented in `docs/archive/phase31/` through `docs/archive/phase43/`). Key findings:

- **Phase 32/44C**: 0/30 cross-dataset transfer directions achieved MF1≥0.25 baseline without adaptation
- **CORAL**: 15/30 directions improved, mean ΔMF1=+0.025 (Outcome B)
- **Phase 52**: P(Y|X) bottleneck confirmed. SupCon MF1=0.719 (+282% improvement)
- **dim=32 optimal**: temperature, loss-weight, and noise parameters all robust
- **10% data suffices**: for representation learning gains

These are research outputs, not the curriculum for current development. The active task is formalization — making the existing pipeline stable, tested, and production-ready.

---

## 13. File Counts & Stats

- **Total tracked files**: ~550 (git)
- **Python files**: ~315
- **Markdown docs**: ~170 (archived research phases in `docs/archive/`)
- **Core package (src/helix_ids/)**: ~40 files over 10 modules
- **Scripts**: ~85 files across 11 subdirectories
- **Tests**: ~110 files over 8 test directories
- **Config**: ~15 config files in `config/` + mutation testing configs
- **Main training script**: 4,605 LOC (breaches 2,000 LOC CI gate)

### Documentation Structure (docs/)

| Directory | Contents |
|-----------|----------|
| `docs/architecture/` | System architecture, data flow, decisions, governance |
| `docs/development/` | Coding standards, testing guide, release process, contributing |
| `docs/operations/` | Deployment, monitoring, recovery, soak testing |
| `docs/api/` | API reference |
| `docs/manuscript/` | IEEE paper drafts (submission-ready + variant) |
| `docs/changelog/` | Changelog |
| `docs/figures/` | Architecture diagrams (PNG) |
| `docs/archive/` | **Research phase documentation** (phase4 through phase43, final results, releases, red-team reports) |

---

## 14. External References

- **Attack taxonomy**: See `src/helix_ids/contracts/attack_taxonomy.py` for ALL dataset label mappings
- **Architecture**: `docs/architecture/SYSTEM_ARCHITECTURE.md`
- **Testing**: `docs/development/TESTING.md`
- **Deployment**: `docs/operations/DEPLOYMENT.md`
- **Governance**: `docs/architecture/GOVERNANCE.md`
- **API Reference**: `docs/api/API_REFERENCE.md`
- **Coding standards**: `docs/development/CODING_STANDARDS.md`
- **Paper draft**: `docs/manuscript/HELIX_submission_ready.md`
