# HELIX-IDS

Network intrusion detection is broken.

Academic benchmarks have been gamed for two decades. Real deployments ship with 90%+ false positive rates because the lab data is clean and the real world is not. And the systems that _do_ work require server-class hardware at every network tap — which is why most networks don't have detection at all, they have alerting after the fact.

HELIX-IDS was built to break that pattern.

Not by inventing a miracle algorithm. By being honest about what edge deployment requires and building from that constraint up.

## The problem this solves

Every network intrusion dataset in existence is:

1. **Small.** NSL-KDD has 25k training samples. That's two minutes of traffic on a moderately busy office link.
2. **Ancient.** The original KDD Cup 99 dataset is from 1999. UNSW-NB15 (2015) is still considered "modern."
3. **Imbalanced.** R2L and U2R attacks — the ones that do the real damage — show up in single-digit percentages. Standard loss functions ignore them.
4. **Stale the moment you deploy.** Train on CICIDS-2018 and deploy on a 2025 network and your feature distributions have already shifted.

HELIX doesn't solve problems 1-4. It acknowledges they exist and works _despite_ them.

## What it actually does

HELIX ingests network flow features from standard formats (NSL-KDD, UNSW-NB15, CICIDS-2017/2018) and classifies each flow as benign or one of several attack families — DoS, Probe, R2L, U2R, and dataset-specific subtypes.

It uses a neural network with three specific design decisions that matter:

**Temporal attention.** Not all network features are equally important, and which ones matter changes depending on the attack. The attention mechanism learns which features to weight when, rather than assuming a fixed importance for each feature across all traffic types.

**Domain adaptation.** The gap between datasets is not noise — it's the primary source of deployment failure. HELIX trains a shared representation that generalizes across datasets so you can train on NSL-KDD and deploy on a live network without the performance cliff that naive transfer produces.

**Threat-aware multi-task loss.** R2L and U2R attacks are infrequent but dangerous. Rather than treat them as "class 4" and watch standard loss functions wash their signal out, HELIX uses a threat-weighted multi-task loss that amplifies rare-class gradients. This is not free — it comes at a small cost to benign accuracy — but the tradeoff is explicit and measurable.

## What it runs on

| Tier | Device | What works |
|------|--------|------------|
| Server | x86/GPU | Full training, inference, all 7 attack classes |
| Edge | Raspberry Pi 4/Zero | Optimized inference (quantized), reduced feature set |
| Micro | ESP32 | Minimal binary classification (normal vs attack), 16KB weights |

The micro tier is the point. A $6 microcontroller that can flag malicious traffic inline is useful in places where nobody runs a server — an IoT sensor mesh, a router, a satellite link. The server tier exists to make the edge models possible.

## What it does NOT do

This is the honest part.

HELIX does **not** detect zero-day attacks. It classifies against known attack families. An unknown attack will produce a low-confidence prediction that triggers the coverage override gate — which is a safety mechanism, not a detection capability.

HELIX does **not** process full packet captures. It works on flow-level features extracted upstream. If your pipeline cannot produce CICFlowMeter-style feature vectors, HELIX cannot help you.

HELIX does **not** run in real time on ESP32. Inference takes ~150ms per sample on an ESP32-S3. That's fast enough for periodic scanning but not inline packet-by-packet inspection.

HELIX does **not** have a threat intelligence feed, a SIEM integration, or a dashboard. It produces predictions and Prometheus metrics. You wire the rest.

These are not gaps. They are scope boundaries. The system does what it says and says what it does.

## Why provenance matters

Every checkpoint, every training run, every dataset transform produces a SHA-256 manifest. This is not compliance theater — it exists because the paper pipeline must be independently verifiable. If I claim a macro F1 of 0.87 on UNSW-NB15, there is a hash chain from that number back to the exact dataset split, preprocessing config, model weights, and random seed that produced it. You can reproduce it, audit it, or falsify it.

Three seeds minimum are required before a checkpoint can be promoted to a baseline. Single-seed runs are not deployable. This is not negotiable.

## The production architecture

The current production system is built around a single certified model:

**HelixIDS-Full** — Multi-task neural network with MLP backbone, threat-weighted multi-task loss, and deployment gate certification. Trained via the unified pipeline in `scripts/training/train_helix_ids_full.py` using the learnability contract system and canonical feature harmonization.

All edge variants (Nano, Lite) are generated from the same training pipeline through quantization — there is no separate codebase for each tier.

## Deployment truth

The staging gate is a hard choke point. Before any model reaches production, it must pass:

- Coverage override rate ≤ 0.02 (model is not guessing too often)
- Degraded state == 0 (runtime monitors are healthy)
- Verified provenance manifest (the checkpoint is who it says it is)

If the gate fails, traffic is rolled back. Not flagged, not alerted — rolled back. The system will not serve a model it cannot justify.

## The hard constraints you should know

- **Input dimension is 17 features.** The system is built around 17 canonical features derived from the intersection of NSL-KDD, UNSW-NB15, and CICIDS-2017/2018. If your feature space is different, you need a mapping layer.
- **Training requires PYTHONPATH=src.** The package is not installed as a pip package. This is intentional for the paper pipeline.
- **Training is multi-seed.** Single runs produce research checkpoints. Three-seed consensus produces deployable baselines. This doubles training cost by design.
- **The architecture is frozen.** No new features, no refactors, no new scripts. The repository is in formalization mode — everything exists to support reproducible publication. The next capability arrives in HELIX v2, not by expanding v1.

## Where to go from here

Everything else is in `docs/`. For the operating details — how to train, evaluate, deploy, tune, benchmark, or extend — that is where to look.

### Quick links

| You want… | Go to |
|-----------|-------|
| System architecture | `docs/architecture/SYSTEM_ARCHITECTURE.md` |
| Active system overview (for new contributors) | `docs/ACTIVE_SYSTEM.md` |
| How to train and deploy | `docs/operations/DEPLOYMENT.md` |
| All test types and CI gates | `docs/development/TESTING.md` |
| API reference | `docs/api/API_REFERENCE.md` |
| Governance and ADRs | `docs/architecture/GOVERNANCE.md` |
| Changelog | `docs/changelog/CHANGELOG.md` |
| AI agent guidance | `AGENTS.md` (repo root) — start here |

### Archived Components

Historical implementations that are **not** part of the active production system are preserved under `archive/phase24a/`. These include:

- `src/helix_ids/adaptation/` — Early cross-dataset adaptation (superseded by `data/feature_harmonization.py`)
- `src/helix_ids/data/data_audit.py` — Dataset quality auditing (not used in production pipelines)
- `scripts/training/train_unified_rebalanced.py` — Direct adaptation training (production uses `train_helix_ids_full.py`)
- `scripts/training/train_unsw_only.py` — UNSW-only training (deleted — zero production references)

See `archive/phase24a/README.md` for details.

### Current state

- **2,500+ tests** across unit, integration, property, mutation, chaos, and fault injection
- **8,479 mutants killed**, 100% mutation score (cosmic-ray, 7 modules)
- **≥70% coverage** enforced at CI gate
- **RC3 certified** — C1–C4 all pass
- **SLSA provenance** generated on every release
- **Digest-pinned containers**, CycloneDX SBOM

### Manuscript

The paper is at `docs/manuscript/HELIX_submission_ready.md`. Figures are in `docs/figures/`.

---

HELIX-IDS is a research system built for the purpose of demonstrating that effective network intrusion detection does not require expensive infrastructure. It is not a commercial product. It is not SOC-2 certified. It is not a replacement for your existing SIEM.

It is an honest attempt to solve a hard problem on hardware that costs single-digit dollars, with every number auditable, every tradeoff documented, and every failure mode understood.

That is the pitch. No marketing. Just the work.

## Technical Quick Reference (for AI agents)

### Essential Commands
```bash
# Setup
python3 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements-lock.txt

# Always set PYTHONPATH before running
export PYTHONPATH=src

# Tests
pytest -q
pytest tests/operations -q          # operations subset
pytest tests/architecture -q        # architecture invariants
pytest tests/test_models -q         # model tests
pytest tests/test_data -q           # data pipeline tests

# Lint & type
ruff check src scripts tests
mypy src

# Smoke training (fast, bypasses strict governance)
HELIX_GOV_POLICY_PROFILE=smoke python scripts/training/train_helix_ids_full.py --config config/helix_config.yaml

# Serve
python scripts/operations/serve_rest.py --checkpoint models/helix_full/helix_full_nsl_kdd_best.pt

# Export (ONNX/TorchScript)
python scripts/deployment/export_model.py --checkpoint <path> --format onnx
```

### Critical Architecture Invariants
| Invariant | Value | Location |
|-----------|-------|----------|
| Input features | 17 (CANONICAL_INPUT_DIM) | schema_contract.py |
| Binary classes | 2 (Normal, Attack) | schema_contract.py |
| Family classes | 7 (5 + Generic + Backdoor) | schema_contract.py |
| Schema version | "2026-05-25" | schema_contract.py |
| Contract version | "1.0.0" | immutable_constants.py |

### File Layout (Key Files)
```
AGENTS.md                          ← Master context for Hermes (start here)

src/helix_ids/
  contracts/schema_contract.py     ← THE immutable schema
  contracts/attack_taxonomy.py     ← Attack family definitions
  data/feature_harmonization.py    ← 41→17 feature engineering
  models/helix_ids_full.py         ← HelixIDSFull model
  operations/inference_runtime.py  ← Production inference
  governance/entrypoint.py         ← Governance gate decorator
  governance/gate_orchestrator.py  ← Stage-based training control
  governance/provenance.py         ← Artifact manifests & SHA256

scripts/
  training/train_helix_ids_full.py ← Main training pipeline (4,605 LOC)
  operations/serve_rest.py         ← FastAPI REST server
  operations/staging_gate_check.py ← Prometheus promotion gate

config/
  helix_config.yaml                ← Model & training configuration

docs/
  architecture/SYSTEM_ARCHITECTURE.md
  development/TESTING.md
  operations/DEPLOYMENT.md
  manuscript/HELIX_submission_ready.md  ← Paper draft
```

### Governance Gate Sequence
```
preload → presplit → pretrain → intrain → posteval → prepromote
```
Use `HELIX_GOV_POLICY_PROFILE=smoke` to bypass during development.
