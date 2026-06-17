# HELIX-IDS Configuration Governance

> **Document**: Configuration entry point inventory, source-of-truth verification,
> schema validation audit, and default safety assessment.
>
> **Scope**: All paths through which runtime behaviour can be influenced —
> CLI arguments, config files, environment variables, Python dataclass defaults,
> and in-code immutable constants.

---

## 1. Override Priority Order

Configuration resolution follows a strict precedence chain.
**Higher priority overrides lower priority at every decision point:**

```
   CLI Arguments           (highest priority)
        ↓
   Environment Variables   (read explicitly at point-of-use)
        ↓
   YAML Config Files       (loaded at startup, merged into dataclass)
        ↓
   Python Dataclass Defaults  (compiled-time default values)
        ↓
   In-Code Immutable Constants (lowest priority, truly immutable)
```

> **Note**: The system does NOT use a unified override fabric. Each layer is
> read independently at different call sites. The priority chain holds
> *conceptually* but is not enforced by a single config resolver.

---

## 2. Config File Entry Points

| # | File | Purpose | Schema Validated | Source of Truth | Default Safe |
|---|------|---------|-----------------|-----------------|-------------|
| 1 | `config/helix_config.yaml` | Model architecture variants (nano/lite/full), training hyperparameters, curriculum schedule, export formats, evaluation targets | **NO** — loaded via `yaml.safe_load()` with no jsonschema validation | Shared — partially duplicated in `helix_full_config.py` dataclass | ⚠️ **WARNING** — unused fields exist (`input_features: 41` vs canonical 17); no cross-validation with schema contract |
| 2 | `config/training.yaml` | Unified Transformer training config (model type, DA, TTA, TENT, synthetic data, threat weights) | **NO** — loaded ad-hoc by various scripts; no schema | Standalone — not consumed by the governed training pipeline | ⚠️ **WARNING** — many fields (`dann_weight`, `use_synthetic`, `threat_weights`) are aspirational and not wired into the actual trainer |
| 3 | `config/platform_configs.yaml` | Per-platform deployment constraints (ESP32, RPi Zero, RPi 4, Server) | **NO** — loaded by `platform_loader.py` via `yaml.safe_load()` only | Single source of truth for platform configs | ✅ **PASS** — has `get()` fallbacks for optional keys (e.g. `max_latency_ms` falls back from `max_latency_us`) |
| 4 | `config/schema_registry.yaml` | Schema lifecycle registry (manifest, benchmark-result) | **N/A** — **NOT CONSUMED** by any code | **ORPHANED** — zero references in `src/` | ❌ **FAIL** — file is defined but never imported or validated anywhere |

### 2.1 Config File Details

#### `config/helix_config.yaml` (243 lines)
- **Loaded by**: Not directly loaded by current code. The `helix_config.yaml` path is passed as `--config` default to the argparse parser but the `parse_config()` in `config_parser.py` never actually reads it — it only stores `args.config`.
- **Contents**: Defines `variants` (nano/lite/full), `common` settings, `curriculum` schedule, `export` formats, `evaluation` targets.
- **Discrepancy**: `helix_config.yaml` declares `input_features: 41` (NSL-KDD default) while the canonical schema contract (`schema_contract.py`) mandates `CANONICAL_INPUT_DIM = 17`.
- **Discrepancy**: `n_classes_family: 4` in YAML vs `CANONICAL_FAMILY_CLASSES = 7` in the contract.

#### `config/training.yaml` (103 lines)
- **Loaded by**: Some scripts reference it, but the governed `train_helix_ids_full.py` pipeline reads its config almost entirely from CLI args.
- **Contents**: `model_type: transformer`, threat weights, domain adaptation params, TTA/TENT, synthetic data.
- **Status**: Many settings are aspirational — the model type used is `HelixIDSFull` (not a generic transformer), and several advanced features (DANN, MMD, CORAL, TENT) are not wired into the `TrainingConfig` dataclass.

#### `config/platform_configs.yaml` (79 lines)
- **Loaded by**: `helix_ids/config/platform_loader.py` — correctly reads `platforms.{name}` and maps to `PlatformConfig` dataclass.
- **Safe defaults**: Uses `dict.get()` with unit-conversion fallbacks (e.g. `max_latency_ms` ÷ 1000 from `max_latency_us`; `max_size_kb` × 1024 from `max_size_mb`).
- **No schema validation**: No jsonschema or pydantic model validates the YAML structure before access.

#### `config/schema_registry.yaml` (20 lines)
- **Status**: **ORPHANED**. Zero code references to `schema_registry` anywhere in `src/`. The `jsonschema` dependency is declared in `pyproject.toml` but never wired to registry validation. The schema contract in `schema_contract.py` uses its own hardcoded constants and in-code assertions, not this registry.

---

## 3. CLI Argument Entry Points

### 3.1 Top-Level CLI (`helix-ids` command)

**Source**: `src/helix_ids/cli.py` (69 lines)
**Entry point**: `helix-ids = helix_ids.cli:main` (from `pyproject.toml`)
**Framework**: `argparse`

| Subcommand | Delegates To | Has Its Own Args? |
|------------|-------------|-------------------|
| `train` | `scripts.train_multidataset_v2_fixed` | Yes — via `argparse` in train script |
| `adversarial` | `scripts.adversarial_training_v2` | Yes |
| `holdout_eval` | `scripts.holdout_evaluation_v2` | Yes |
| `benchmark` | `scripts.benchmark_e2e_v2_fixed` | Yes |
| `deploy` | `scripts.deploy` | Yes |
| `download_data` | `scripts.download_datasets` | Yes |
| `train_edge` | `scripts.train_edge_models` | Yes |

> **Note**: The top-level CLI passes no arguments to subcommands — it uses
> `runpy.run_module()` which inherits `sys.argv`. Each subcommand re-parses
> its own arguments independently. There is no unified argument namespace.

### 3.2 Training CLI (`scripts/training/train_helix_ids_full.py`)

**Source**: `scripts/training/orchestration/config_parser.py` (745 lines)
**Invoked via**: `python scripts/training/train_helix_ids_full.py --config ... --output ...`

| Argument | Type | Default | Schema Constraint | Default Safe |
|----------|------|---------|-------------------|-------------|
| `--config` | str | `config/helix_config.yaml` | — | ⚠️ **WARNING**: default path is set but file is never actually read by `parse_config()` |
| `--output` | str | `models/helix_full` | — | ✅ |
| `--device` | str | `mps` | — | ✅ Matches M4 MPS target |
| `--batch-size` | int | `256` | `>= 1` (validated) | ✅ |
| `--eval-batch-size` | int | `None` (defaults to `--batch-size`) | — | ✅ |
| `--epochs` | int | `150` | — | ✅ |
| `--learning-rate` | float | `1.5e-4` | — | ✅ |
| `--warmup-epochs` | int | `2` | — | ✅ |
| `--grad-clip` | float | `1.0` | — | ✅ |
| `--log-interval` | int | `50` | — | ✅ |
| `--class-balance-strategy` | str | `focal` | choices: `none, weighted_ce, sqrt_weighted_ce, focal` | ⚠️ **WARNING**: default is `focal` but `TrainingConfig` default is `"weighted_ce"` |
| `--focal-gamma` | float | `1.2` | — | ⚠️ **WARNING**: differs from `TrainingConfig` default `0.0` |
| `--label-smoothing` | float | `0.0` | — | ✅ But differs from `TrainingConfig` default `0.1` |
| `--sampler-mode` | str | `interleaved_rr` | choices: `interleaved_rr, weighted_random_sampler` | ✅ |
| `--min-class4-samples` | int | `2000` | — | ✅ |
| `--class4-per-batch-min` | int | `2` | — | ✅ |
| `--family-margin-loss-weight` | float | `None` (auto: 0.15/0.1) | — | ✅ |
| `--family-class4-logit-penalty-weight` | float | `0.0` | — | ✅ |
| `--family-feature-separation-weight` | float | `0.0` | — | ✅ |
| `--family-class4-target-scale` | float | `1.0` | — | ✅ |
| `--enable-logit-adjustment` | bool | `False` | BooleanOptionalAction | ⚠️ **WARNING**: differs from `TrainingConfig` default `True` |
| `--logit-temp` | float | `1.0` | — | ⚠️ **WARNING**: differs from `TrainingConfig` default `1.5` |
| `--logit-adjustment-tau` | float | `None` | — | ✅ Overrides `--logit-temp` when set |
| `--min-class-prob-eps` | float | `0.0` | — | Same as `TrainingConfig` default |
| `--entropy-regularization` | float | `0.02` | — | ✅ |
| `--seed` | int | `os.environ.get("HELIX_SEED", 42)` | — | ✅ |
| `--disable-early-stopping` | flag | `False` | — | ✅ |
| `--calibration-mode` | str | `internal_on` | choices: `internal_on, internal_off` | ✅ |
| `--max-temperature` | float | `5.0` | `> 0` (validated) | ✅ |
| `--class4-logit-shift` | float | `0.0` | `0.0 <= x <= 5.0` (validated) | ✅ |
| `--multi-seed-governance` | bool | `False` | BooleanOptionalAction | ✅ |
| `--multi-seeds` | str | `42,1337,2026` | — | ✅ |
| `--dataset` | str | `None` | choices: `nsl_kdd, unsw_nb15, cicids` | ✅ |
| `--holdout-dataset` | str | `cicids` | choices: `nsl_kdd, unsw_nb15, cicids` | ✅ |
| `--precomputed-splits-dir` | str | `data/processed/multi_dataset_v1` | — | ✅ |
| `--force-recompute-splits` | flag | `False` | — | ✅ |
| `--snapshot-mode` | str | `strict` | choices: `strict, research_override` | ✅ |
| `--allow-unfrozen-snapshot` | bool | `False` | BooleanOptionalAction | ✅ Alias for `--snapshot-mode research_override` |
| `--ab-mode` | bool | `True` | BooleanOptionalAction | ✅ |
| `--ab-track` | str | `objective` | choices: `feature, objective` | ✅ |
| `--ab-change-id` | str | `baseline` | — | ✅ |
| `--ab-baseline` | str | `None` | — | ✅ |
| `--ab-require-baseline` | bool | `True` | BooleanOptionalAction | ✅ |
| `--cluster-objective` | str | `kmeans` | choices: `kmeans, gmm, spectral` | ✅ |
| `--cluster-spectral-affinity` | str | `nearest_neighbors` | choices: `nearest_neighbors, rbf` | ✅ |

> **Key Finding**: Several CLI defaults **diverge** from their corresponding
> `TrainingConfig` dataclass defaults (see §7). The parser in `config_parser.py`
> completely redefines defaults rather than reading from `TrainingConfig`.

---

## 4. Environment Variable Entry Points

All HELIX-IDS environment variables use the `HELIX_` prefix convention.
A few non-prefixed variables are also used.
Variables are read ad-hoc via `os.environ.get()` / `os.getenv()` at point of use — 
there is no central env-var registry or validation layer.

| Variable | Default | Used In | Purpose | Safe |
|----------|---------|---------|---------|------|
| `HELIX_SEED` | `42` | `config_parser.py`, `train_helix_ids_full.py`, `train_multidataset_v2_fixed.py` | Global random seed for deterministic execution | ✅ |
| `HELIX_STRICT_MISSING` | (unset → disabled) | `train_helix_ids_full.py` | Enable strict missing-data assertion mode | ✅ |
| `STRICT_MISSING` | (unset → disabled) | `train_helix_ids_full.py` | Non-prefixed alias for strict missing mode | ⚠️ **WARNING**: no `HELIX_` prefix |
| `HELIX_GOV_POLICY_PROFILE` | `""` (empty → full policy) | `entrypoint.py`, `train_helix_ids_full.py` | Governance profile: `"smoke"` relaxes CI bounds | ✅ |
| `HELIX_GOV_STAGE_SEQUENCE` | `""` (→ `DEFAULT_STAGE_SEQUENCE`) | `entrypoint.py` | Comma-separated override of governance stage sequence | ✅ |
| `HELIX_RUN_REGISTRY` | `results/gates/run_registry.jsonl` | `entrypoint.py`, `train_multidataset_v2_fixed.py` | Path to run registry JSONL for drift detection | ✅ |
| `HELIX_RUN_ID` | `{entrypoint_id}-local` | `entrypoint.py` | Unique run identifier | ✅ |
| `HELIX_GATE_EVENTS` | `results/gates/gate_events.jsonl` | `entrypoint.py` | Path to gate events log | ✅ |
| `HELIX_FAILURE_MEMORY` | `results/gates/failure_memory.jsonl` | `entrypoint.py` | Path to failure memory log | ✅ |
| `HELIX_PARENT_RUN_ID` | (unset → `None`) | `entrypoint.py`, `train_multidataset_v2_fixed.py` | Run lineage: parent run ID | ✅ |
| `HELIX_FINGERPRINT` | (unset → `None`) | `entrypoint.py`, `train_multidataset_v2_fixed.py` | Run fingerprint for lineage | ✅ |
| `HELIX_DATASET_HASHES` | (unset → `"unknown"`) | `train_multidataset_v2_fixed.py` | Dataset hash for lineage tracking | ✅ |
| `HELIX_SCHEMA_HASH` | (unset → `"unknown"`) | `train_multidataset_v2_fixed.py` | Schema hash for lineage tracking | ✅ |
| `HELIX_MAPPING_VERSION` | (unset → `"unknown"`) | `train_multidataset_v2_fixed.py` | Feature mapping version | ✅ |
| `HELIX_ALLOW_LEGACY_MANIFEST` | `""` | `provenance.py`, `lifecycle_verifier.py` | Allow legacy manifest ingress | ⚠️ **WARNING**: gated — blocked in production |
| `HELIX_ALLOW_LEGACY_ARTIFACTS` | `""` | `parameters.py`, `provenance.py` | Allow legacy artifact ingress | ⚠️ **WARNING**: gated — blocked in production |
| `HELIX_PROVENANCE_TELEMETRY` | `results/provenance/provenance_events.jsonl` | `provenance.py` | Path to provenance event log | ✅ |
| `HELIX_RUNTIME_ENV` | (unset → dev) | `parameters.py` | Production detection (`"prod"`, `"production"`) | ✅ |
| `HELIX_ENV` | (unset → dev) | `parameters.py` | Production detection (alias) | ✅ |
| `HELIX_DEPLOY_ENV` | (unset → dev) | `parameters.py` | Production detection (alias) | ✅ |
| `HELIX_LOADER_CACHE_DIR` | (unset → default cache) | `loader_core.py` | Override data loader cache directory | ✅ |
| `HELIX_SKIP_LARGE_CICIDS` | `"0"` | `feature_io.py` | Skip loading large CICIDS raw dump | ✅ |
| `HELIX_DEBUG_LENIENT` | `"0"` | `feature_harmonization.py` | Enable debug-only lenient pipeline mode | ✅ |
| `PYTHONHASHSEED` | (set dynamically) | `determinism.py` | Python hash seed for determinism (set from `HELIX_SEED`) | ✅ |

**Environment Variable Count**: **25** documented variables (2 non-prefixed).

---

## 5. Python Dataclass / In-Code Default Entry Points

These are compiled-time defaults that define the "last-resort" configuration
when no CLI, env, or config-file override is provided.

### 5.1 `TrainingConfig` (`helix_full_config.py`)

| Field | Default | Notes |
|-------|---------|-------|
| `input_dim` | `17` | Matches `CANONICAL_INPUT_DIM` ✅ |
| `hidden_dims` | `(256, 192, 128, 64)` | — |
| `dropout_rates` | `(0.3, 0.3, 0.25, 0.2)` | — |
| `use_batch_norm` | `True` | — |
| `activation` | `"relu"` | — |
| `lambda_binary` | `1.0` | — |
| `lambda_family` | `0.8` | — |
| `batch_size` | `256` | Matches CLI default ✅ |
| `learning_rate` | `1e-3` | **Differs from CLI** (1.5e-4) ⚠️ |
| `weight_decay` | `1e-4` | — |
| `epochs` | `150` | Matches CLI default ✅ |
| `warmup_epochs` | `2` | Matches CLI default ✅ |
| `device` | `"mps"` | Matches CLI default ✅ |
| `class_balance_strategy` | `"weighted_ce"` | **Differs from CLI** ("focal") ⚠️ |
| `focal_gamma` | `0.0` | **Differs from CLI** (1.2) ⚠️ |
| `label_smoothing` | `0.1` | **Differs from CLI** (0.0) ⚠️ |
| `enable_logit_adjustment` | `True` | **Differs from CLI** (False) ⚠️ |
| `logit_temp` | `1.5` | **Differs from CLI** (1.0) ⚠️ |
| `checkpoint_dir` | `Path("checkpoints/helix_full")` | — |
| `early_stopping_patience` | `15` | — |
| `use_class_weights` | `True` | — |

### 5.2 `DataConfig` (`helix_full_config.py`)

| Field | Default |
|-------|---------|
| `data_dir` | `Path("data/processed")` |
| `split_file` | `Path("data/processed/splits.pkl")` |
| `feature_mappings_file` | `Path("data/processed/feature_mappings.json")` |
| `use_per_dataset_normalization` | `True` |

### 5.3 `EvaluationConfig` (`helix_full_config.py`)

| Field | Default |
|-------|---------|
| `metrics` | `["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "cm"]` |
| `per_dataset_eval` | `True` |
| `quantization_target_drop` | `0.02` |
| `results_dir` | `Path("results/helix_full")` |
| `save_predictions` | `True` |
| `save_model_card` | `True` |

### 5.4 `PlatformConfig` (`platform_loader.py`)

| Field | Default | Notes |
|-------|---------|-------|
| No class-level defaults | All fields populated from `platform_configs.yaml` | Uses `dict.get()` fallbacks for unit conversion ✅ |

### 5.5 Governance Parameters (`governance/parameters.py`) — Frozen Dataclasses

| Config Object | Key Fields | Safe |
|---------------|-----------|------|
| `StageTimeouts` | `preload_seconds=900`, `presplit_seconds=1200`, `pretrain_seconds=1200`, `intrain_seconds=21600`, `posteval_seconds=1800`, `prepromote_seconds=600` | ✅ |
| `TrainingAbortPolicy` | `low_entropy_threshold=0.30`, `epochs_without_improvement=12`, etc. | ✅ |
| `BootstrapPolicy` | `n_replicates=2000`, `max_ci_width=0.05`, `min_ci95_lower_bound=0.50` | ✅ |
| `DriftPolicy` | `baseline_window_runs=20`, `max_abs_macro_f1_drift=0.05` | ✅ |
| `DatasetIdentityLeakagePolicy` | `model_name="multinomial_logistic_regression"`, `max_balanced_accuracy=0.90` | ✅ |
| `PromotionPolicy` | `min_seed_runs=3`, `max_inter_seed_macro_f1_variance=0.01` | ✅ |

### 5.6 Immutable Constants (`contracts/immutable_constants.py`, `contracts/schema_contract.py`)

| Constant | Value | Notes |
|----------|-------|-------|
| `CANONICAL_INPUT_DIM` | `17` | Audited invariant flow features |
| `CANONICAL_BINARY_CLASSES` | `2` | Normal vs Attack |
| `CANONICAL_FAMILY_CLASSES` | `7` | Attack family count |
| `SCHEMA_VERSION` | `"2026-05-25"` | — |
| `CONTRACT_VERSION` | `"2.1"` | From `immutable_constants.py` |
| `MANIFEST_VERSION` | `"1.0.0"` | From `immutable_constants.py` |

---

## 6. Schema Validation Audit

| Artifact | Has Schema? | Schema Format | Validated At Runtime? |
|----------|------------|---------------|-----------------------|
| `schema_contract.py` constants | ✅ Built-in | In-code assertions (`assert_runtime_contract()`) | ✅ In `train_helix_ids_full.py` and data pipelines |
| `helix_config.yaml` | ❌ | None | ❌ Not validated |
| `training.yaml` | ❌ | None | ❌ Not validated |
| `platform_configs.yaml` | ❌ | None | ❌ Only `KeyError` if platform missing |
| `schema_registry.yaml` | ❌ | None (self-describing) | ❌ **Not consumed at all** |
| CLI arguments | ⚠️ Partial | Manual `if` checks in `parse_config()` | ✅ Basic range validation (batch_size>=1, max_temperature>0, class4_logit_shift 0-5) |

### 6.1 Is `schema_registry.yaml` Actually Used?

**No.** The file `config/schema_registry.yaml` is defined with a lifecyle
registry for `manifest_schema` and `benchmark_result_schema` but:

- Zero references to `schema_registry` exist in `src/` or `scripts/`.
- The `jsonschema>=4.0.0` dependency in `pyproject.toml` is declared but
  never imported for config validation.
- Schema enforcement happens entirely through in-code assertions in
  `schema_contract.py`, which relies on hardcoded Python constants, not
  the YAML registry.

**Verdict: ❌ FAIL — ORPHANED FILE**

---

## 7. Source-of-Truth Conflicts

The most significant governance finding is the divergence between
`TrainingConfig` dataclass defaults and CLI argparse defaults in
`config_parser.py`. Both claim authority:

| Parameter | `TrainingConfig` Default | CLI Argparse Default | Conflict? |
|-----------|-------------------------|---------------------|-----------|
| `learning_rate` | `1e-3` | `1.5e-4` | ❌ **DIVERGENT** |
| `class_balance_strategy` | `"weighted_ce"` | `"focal"` | ❌ **DIVERGENT** |
| `focal_gamma` | `0.0` | `1.2` | ❌ **DIVERGENT** |
| `label_smoothing` | `0.1` | `0.0` | ❌ **DIVERGENT** |
| `enable_logit_adjustment` | `True` | `False` | ❌ **DIVERGENT** |
| `logit_temp` | `1.5` | `1.0` | ❌ **DIVERGENT** |
| `batch_size` | `256` | `256` | ✅ Match |
| `epochs` | `150` | `150` | ✅ Match |
| `warmup_epochs` | `2` | `2` | ✅ Match |
| `device` | `"mps"` | `"mps"` | ✅ Match |
| `min_class_prob_eps` | `0.0` | `0.0` | ✅ Match |

> **Impact**: Depending on which code path constructs a `TrainingConfig`
> (direct instantiation vs `parse_config()`), the effective defaults differ.
> The CLI path (via `parse_config()`) overrides the dataclass defaults at
> parse time by constructing `TrainingConfig(...)` with CLI values, so CLI
> wins in practice — but the divergence creates maintenance risk.

---

## 8. Default Safety Assessment

### ✅ PASS Categories

| Category | Finding |
|----------|---------|
| Immutable constants | ✅ All constants in `schema_contract.py` and `immutable_constants.py` are frozen strings/integers with no runtime side effects |
| Governance parameters | ✅ All `@dataclass(frozen=True)` — `StageTimeouts`, `TrainingAbortPolicy`, `BootstrapPolicy`, `DriftPolicy`, `DatasetIdentityLeakagePolicy`, `PromotionPolicy` |
| CLI range validation | ✅ Batch size ≥ 1, max temperature > 0, class4-logit-shift 0..5 validated in `parse_config()` |
| Platform config fallbacks | ✅ `platform_loader.py` uses `.get()` with unit-conversion fallbacks |
| Production mode detection | ✅ `is_production_runtime()` checks 3 env vars; `allow_legacy_artifacts()` asserts not in production |
| Determinism | ✅ `HELIX_SEED` always set; `set_global_determinism()` called before training |
| Training defaults | ✅ epochs=150, batch_size=256, device=mps are safe for target M4 hardware |

### ⚠️ WARNING Categories

| Category | Issue |
|----------|-------|
| CLI vs dataclass default divergence | 6 parameters differ between `TrainingConfig` and argparse defaults (§7) |
| `--config` default unused | CLI default `config/helix_config.yaml` is stored in `args.config` but never actually loaded by `parse_config()` |
| `helix_config.yaml` stale | Declares `input_features: 41` and `n_classes_family: 4` which conflict with canonical constants (17, 7) |
| `training.yaml` aspirational | Contains DANN/MMD/CORAL/TENT/synthetic data settings not wired into the active training loop |
| Non-prefixed env var | `STRICT_MISSING` without `HELIX_` prefix (alias for `HELIX_STRICT_MISSING`) |
| `TrainingConfig` not frozen | Mutable dataclass — `_apply_disable_early_stopping()` mutates `early_stopping_patience` after construction |

### ❌ FAIL Categories

| Category | Issue |
|----------|-------|
| `schema_registry.yaml` orphaned | File exists but is **never loaded or referenced** by any code. Zero schema validation via `jsonschema`. |
| No config-wide schema validation | None of the YAML config files (`helix_config.yaml`, `training.yaml`, `platform_configs.yaml`) are validated against a jsonschema or pydantic model. Only manual assertion checks exist. |
| No unified override fabric | CLI, env, config files, and dataclass defaults are read independently at different call sites. No single resolver enforces priority ordering. |
| Redundant config surface | At least 3 overlapping config representations: `TrainingConfig` dataclass, CLI argparse defaults, `helix_config.yaml` — with partial duplication. |

---

## 9. Summary Inventory Table

| Entry Point | Type | Source of Truth | Schema Validated | Default Safe |
|------------|------|-----------------|-----------------|-------------|
| `cli.py` subcommands | CLI | `helix_ids/cli.py` | ❌ | ✅ |
| `config_parser.py` args (~38) | CLI | `config_parser.py` | ⚠️ Partial (range checks) | ⚠️ Divergent from dataclass |
| `config/helix_config.yaml` | File | YAML | ❌ | ⚠️ Stale values |
| `config/training.yaml` | File | YAML | ❌ | ⚠️ Aspirational only |
| `config/platform_configs.yaml` | File | YAML | ❌ | ✅ Fallbacks present |
| `config/schema_registry.yaml` | File | YAML | ❌ Not consumed | ❌ **FAIL — orphaned** |
| `Helix*` env vars (22 prefixed) | Env | Ad-hoc `os.getenv()` | ❌ | ✅ Most with defaults |
| `STRICT_MISSING` (unprefixed) | Env | Ad-hoc | ❌ | ⚠️ Non-standard naming |
| `TrainingConfig` dataclass | Defaults | `helix_full_config.py` | ❌ | ⚠️ Mutable, conflicts with CLI |
| `DataConfig` / `EvaluationConfig` | Defaults | `helix_full_config.py` | ❌ | ✅ |
| Governance frozen dataclasses | Defaults | `parameters.py` | ❌ | ✅ Frozen, safe defaults |
| `schema_contract.py` constants | Immutable | `schema_contract.py` | ✅ Self-validating | ✅ |
| `immutable_constants.py` | Immutable | `immutable_constants.py` | ✅ SHA-256 anchored | ✅ |

---

## 10. Recommendations

| Priority | Recommendation |
|----------|---------------|
| **P0** | Either wire `schema_registry.yaml` into `schema_contract.py` validation, or remove the orphaned file |
| **P0** | Reconcile 6 CLI↔dataclass default divergences to establish a single source of truth for training parameters |
| **P1** | Unify config resolution: create a central `ConfigResolver` that enforces CLI > env > file > defaults ordering |
| **P1** | Validate all YAML configs against jsonschema or pydantic models at load time |
| **P2** | Freeze `TrainingConfig` (`@dataclass(frozen=True)`) to prevent post-construction mutation |
| **P2** | Centralize env-var documentation and add a runtime env-var registry with validation |
| **P3** | Remove unused/aspirational fields from `config/training.yaml` or wire them into the actual trainer |
| **P3** | Add `HELIX_` prefix to `STRICT_MISSING` for naming consistency |

---

## 11. Overall Verdict

| Category | Result |
|----------|--------|
| **CLI args** | ⚠️ **WARNING** — 6 defaults diverge from dataclass; partial validation only |
| **Config files** | ❌ **FAIL** — `schema_registry.yaml` orphaned; no schema validation on any YAML; stale values in `helix_config.yaml` |
| **Environment variables** | ⚠️ **WARNING** — well-prefixed but no central registry; one non-prefixed alias |
| **Dataclass defaults** | ⚠️ **WARNING** — `TrainingConfig` is mutable and conflicts with CLI values |
| **Immutable constants** | ✅ **PASS** — frozen, self-validating, SHA-256 anchored |
| **Governance parameters** | ✅ **PASS** — all frozen dataclasses with safe defaults |
| **Override priority** | ❌ **FAIL** — no unified resolver enforces CLI > env > file > defaults |
| **Single source of truth** | ❌ **FAIL** — 3 overlapping config representations for training |
| **schema_registry.yaml usage** | ❌ **FAIL** — declared but never consumed |
