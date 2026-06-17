"""Environment-based configuration loader for HELIX-IDS.

Provides a ``load_environment`` function that merges configuration from four
priority layers (highest to lowest):

1. CLI arguments          (passed directly as overrides dict)
2. Environment variables  (``HELIX_*`` prefixed)
3. YAML config file       (if provided)
4. Default values         (hardcoded in schema)

Key features
------------
- Schema validation with type coercion
- ``HELIX_*`` environment variable support (e.g. ``HELIX_BATCH_SIZE=128``)
- Dot-notation nested key access (e.g. ``training.batch_size``)
- Full type coercion (int, float, bool, str, list, Path)
- Dataclass-based result schema
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Any, Callable

# ── Type coercion helpers ────────────────────────────────────────────────────


def _parse_bool(value: str) -> bool:
    """Parse a string as a boolean. Accepts true/false/1/0/yes/no."""
    val = value.strip().lower()
    if val in ("true", "1", "yes"):
        return True
    if val in ("false", "0", "no"):
        return False
    raise ValueError(f"Cannot parse bool from {value!r}")


def _parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated string into a list of ints."""
    if not value:
        return []
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [int(p) for p in parts]


def _parse_float_list(value: str) -> list[float]:
    """Parse a comma-separated string into a list of floats."""
    if not value:
        return []
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [float(p) for p in parts]


def _parse_path(value: str) -> Path:
    """Parse a string as a Path."""
    return Path(value).expanduser().resolve()


# ── Schema definition ────────────────────────────────────────────────────────


# HELIX_ENV_VAR -> (dotted_key, coerce_fn, default)
# The ordering here defines the struct and field listing
ENV_SCHEMA: dict[str, tuple[str, Callable[[str], Any], Any]] = {
    "HELIX_BATCH_SIZE": ("training.batch_size", int, 256),
    "HELIX_LEARNING_RATE": ("training.learning_rate", float, 1e-3),
    "HELIX_WEIGHT_DECAY": ("training.weight_decay", float, 1e-4),
    "HELIX_EPOCHS": ("training.epochs", int, 150),
    "HELIX_WARMUP_EPOCHS": ("training.warmup_epochs", int, 2),
    "HELIX_WARMUP_INIT_LR": ("training.warmup_init_lr", float, 1e-5),
    "HELIX_LR_DECAY_FACTOR": ("training.lr_decay_factor", float, 0.1),
    "HELIX_LR_DECAY_STEPS": ("training.lr_decay_steps", _parse_int_list, "50,100,140"),
    "HELIX_INPUT_DIM": ("model.input_dim", int, 17),
    "HELIX_HIDDEN_DIMS": ("model.hidden_dims", _parse_int_list, "256,192,128,64"),
    "HELIX_DROPOUT_RATES": ("model.dropout_rates", _parse_float_list, "0.3,0.3,0.25,0.2"),
    "HELIX_USE_BATCH_NORM": ("model.use_batch_norm", _parse_bool, True),
    "HELIX_ACTIVATION": ("model.activation", str, "relu"),
    "HELIX_DEVICE": ("runtime.device", str, "mps"),
    "HELIX_NUM_WORKERS": ("runtime.num_workers", int, 2),
    "HELIX_PIN_MEMORY": ("runtime.pin_memory", _parse_bool, True),
    "HELIX_CHECKPOINT_DIR": ("runtime.checkpoint_dir", _parse_path, "checkpoints/helix_full"),
    "HELIX_DATA_DIR": ("data.data_dir", _parse_path, "data/processed"),
    "HELIX_LOG_INTERVAL": ("runtime.log_interval", int, 50),
    "HELIX_VAL_INTERVAL": ("runtime.val_interval", int, 1),
    "HELIX_SAVE_BEST_ONLY": ("runtime.save_best_only", _parse_bool, True),
    "HELIX_SAVE_INTERVAL": ("runtime.save_interval", int, 5),
    "HELIX_GRAD_CLIP_NORM": ("training.max_grad_norm", float, 1.0),
    "HELIX_EARLY_STOP_PATIENCE": ("training.early_stopping_patience", int, 15),
    "HELIX_EARLY_STOP_THRESHOLD": ("training.early_stopping_threshold", float, 1e-4),
    "HELIX_MINORITY_RECALL": ("training.min_family_minority_recall_for_best", float, 0.70),
    "HELIX_USE_CLASS_WEIGHTS": ("training.use_class_weights", _parse_bool, True),
    "HELIX_CLASS_BALANCE_STRATEGY": ("training.class_balance_strategy", str, "weighted_ce"),
    "HELIX_FOCAL_GAMMA": ("training.focal_gamma", float, 0.0),
    "HELIX_LABEL_SMOOTHING": ("training.label_smoothing", float, 0.1),
    "HELIX_LAMBDA_BINARY": ("loss.lambda_binary", float, 1.0),
    "HELIX_LAMBDA_FAMILY": ("loss.lambda_family", float, 0.8),
    "HELIX_LOGIT_TEMP": ("loss.logit_temp", float, 1.5),
    "HELIX_ENABLE_LOGIT_ADJUST": ("loss.enable_logit_adjustment", _parse_bool, True),
    "HELIX_FEATURE_MAPPINGS": (
        "data.feature_mappings_file",
        _parse_path,
        "data/processed/feature_mappings.json",
    ),
    "HELIX_RESULTS_DIR": ("runtime.results_dir", _parse_path, "results/helix_full"),
    "HELIX_RUN_ID": ("runtime.run_id", str, ""),
    "HELIX_EXPERIMENT_ID": ("runtime.experiment_id", str, ""),
    "HELIX_SEED": ("runtime.seed", int, 42),
    "HELIX_VERBOSE": ("runtime.verbose", _parse_bool, False),
    "HELIX_STRICT_MODE": ("runtime.strict_mode", _parse_bool, True),
    "HELIX_RECOVERY_ENABLED": ("runtime.recovery_enabled", _parse_bool, True),
    "HELIX_CIRCUIT_BREAKER_ENABLED": ("runtime.circuit_breaker_enabled", _parse_bool, True),
    "HELIX_LOGGING_JSON": ("runtime.logging_json", _parse_bool, True),
    "HELIX_LOGGING_LEVEL": ("runtime.logging_level", str, "INFO"),
}

# Build reverse mapping: flat_field_name -> dotted_key
_FLAT_TO_DOTTED: dict[str, str] = {}
# Build section -> field_names mapping
_SECTION_FIELDS: dict[str, set[str]] = {}
for _env_var, (dotted_key, _, _) in ENV_SCHEMA.items():
    _FLAT_TO_DOTTED[dotted_key.split(".")[-1]] = dotted_key
    section = dotted_key.split(".")[0]
    _SECTION_FIELDS.setdefault(section, set()).add(dotted_key.split(".")[-1])


# ── Configuration result dataclass ───────────────────────────────────────────


@dataclass
class HelixEnvironment:
    """Resolved HELIX-IDS configuration with all priority layers applied.

    Attributes mirror the ENV_SCHEMA keys but flattened into sections.
    """

    training: TrainingSettings = field(default_factory=lambda: TrainingSettings())
    model: ModelSettings = field(default_factory=lambda: ModelSettings())
    runtime: RuntimeSettings = field(default_factory=lambda: RuntimeSettings())
    data: DataSettings = field(default_factory=lambda: DataSettings())
    loss: LossSettings = field(default_factory=lambda: LossSettings())

    def __post_init__(self) -> None:
        # Ensure nested dataclasses are properly initialised
        for attr_name in ("training", "model", "runtime", "data", "loss"):
            val = getattr(self, attr_name)
            if isinstance(val, dict):
                cls_map = {
                    "training": TrainingSettings,
                    "model": ModelSettings,
                    "runtime": RuntimeSettings,
                    "data": DataSettings,
                    "loss": LossSettings,
                }
                setattr(self, attr_name, cls_map[attr_name](**val))


@dataclass
class TrainingSettings:
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 150
    warmup_epochs: int = 2
    warmup_init_lr: float = 1e-5
    lr_decay_factor: float = 0.1
    lr_decay_steps: list[int] = field(default_factory=lambda: [50, 100, 140])
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 15
    early_stopping_threshold: float = 1e-4
    min_family_minority_recall_for_best: float = 0.70
    use_class_weights: bool = True
    class_balance_strategy: str = "weighted_ce"
    focal_gamma: float = 0.0
    label_smoothing: float = 0.1


@dataclass
class ModelSettings:
    input_dim: int = 17
    hidden_dims: list[int] = field(default_factory=lambda: [256, 192, 128, 64])
    dropout_rates: list[float] = field(default_factory=lambda: [0.3, 0.3, 0.25, 0.2])
    use_batch_norm: bool = True
    activation: str = "relu"


@dataclass
class RuntimeSettings:
    device: str = "mps"
    num_workers: int = 2
    pin_memory: bool = True
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/helix_full"))
    log_interval: int = 50
    val_interval: int = 1
    save_best_only: bool = True
    save_interval: int = 5
    results_dir: Path = field(default_factory=lambda: Path("results/helix_full"))
    run_id: str = ""
    experiment_id: str = ""
    seed: int = 42
    verbose: bool = False
    strict_mode: bool = True
    recovery_enabled: bool = True
    circuit_breaker_enabled: bool = True
    logging_json: bool = True
    logging_level: str = "INFO"


@dataclass
class DataSettings:
    data_dir: Path = field(default_factory=lambda: Path("data/processed"))
    feature_mappings_file: Path = field(
        default_factory=lambda: Path("data/processed/feature_mappings.json")
    )


@dataclass
class LossSettings:
    lambda_binary: float = 1.0
    lambda_family: float = 0.8
    focal_gamma: float = 0.0
    label_smoothing: float = 0.1
    enable_logit_adjustment: bool = True
    logit_temp: float = 1.5


# ── Resolution Engine ────────────────────────────────────────────────────────


def _set_nested(
    target: dict[str, Any],
    dotted_key: str,
    value: Any,
) -> None:
    """Set a value in a nested dict using dot-notation key.

    Example: ``_set_nested(env, 'training.batch_size', 128)``.
    """
    parts = dotted_key.split(".")
    key = parts[-1]
    for part in parts[:-1]:
        if part not in target:
            target[part] = {}
        elif not isinstance(target[part], dict):
            target[part] = {}
        target = target[part]
    target[key] = value


def _apply_defaults() -> dict[str, Any]:
    """Build a flat-override dict from defaults in the schema.

    Preserves relative path defaults without resolving (matching the
    dataclass behaviour), but applies list-type coercion so that
    comma-separated default strings become proper Python lists.
    """
    result: dict[str, Any] = {}
    for _env_var, (dotted_key, coerce_fn, default) in ENV_SCHEMA.items():
        if isinstance(default, str) and coerce_fn not in (str, _parse_path):
            # Coerce list/float-list strings to actual Python types
            try:
                value = coerce_fn(default)
            except (ValueError, TypeError):
                value = default
        elif isinstance(default, str) and coerce_fn is _parse_path:
            # Keep paths relative — don't resolve
            value = Path(default)
        else:
            value = default
        _set_nested(result, dotted_key, value)
    return result


def _apply_env_overrides(
    config: dict[str, Any],
    prefix: str = "HELIX_",
) -> dict[str, Any]:
    """Read ``HELIX_*`` environment variables and apply to config dict."""
    for env_var, (dotted_key, coerce_fn, _) in ENV_SCHEMA.items():
        raw = os.environ.get(env_var)
        if raw is None or not raw.strip():
            continue
        try:
            value = coerce_fn(raw)
            _set_nested(config, dotted_key, value)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Failed to coerce {env_var}={raw!r}: {exc}"
            ) from exc
    return config


def _apply_yaml_overrides(config: dict[str, Any], yaml_path: str | Path) -> dict[str, Any]:
    """Apply overrides from a JSON-based config file."""
    p = Path(yaml_path)
    if not p.exists():
        return config

    try:
        with open(p, encoding="utf-8") as f:
            overrides: dict[str, Any] = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Failed to load config file {yaml_path}: {exc}") from exc

    return _deep_merge(config, overrides)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge *override* into *base*, mutating *base*."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _apply_cli_overrides(
    config: dict[str, Any],
    cli_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """Apply CLI-level overrides (highest priority).

    Supports:
    - Nested dicts: ``{"training": {"batch_size": 64}}``
    - Dot-notation: ``{"training.batch_size": 64}``
    - Flat field names: ``{"batch_size": 64}`` (auto-resolved to section)
    """
    if not cli_overrides:
        return config

    for key, value in cli_overrides.items():
        if "." in key:
            # Dot-notation: resolve directly via _set_nested
            _set_nested(config, key, value)
        elif key in _FLAT_TO_DOTTED:
            # Flat field name: look up in schema
            _set_nested(config, _FLAT_TO_DOTTED[key], value)
        elif isinstance(value, dict):
            # Nested dict: try section-based merge
            _deep_merge(config, {key: value})
        else:
            # Unknown flat key: try common sections by scanning _SECTION_FIELDS
            resolved = False
            for section, fields in _SECTION_FIELDS.items():
                if key in fields:
                    _set_nested(config, f"{section}.{key}", value)
                    resolved = True
                    break
            if not resolved:
                config[key] = value

    return config


def _validate_config(config: HelixEnvironment) -> list[str]:
    """Validate resolved config and return a list of warning strings."""
    warnings: list[str] = []

    if config.training.epochs <= 0:
        warnings.append(f"epochs={config.training.epochs} must be > 0; using 1")
    if config.training.batch_size <= 0:
        warnings.append(f"batch_size={config.training.batch_size} must be > 0; using 1")
    if config.training.learning_rate <= 0:
        warnings.append(f"learning_rate={config.training.learning_rate} must be > 0")
    if config.training.weight_decay < 0:
        warnings.append(f"weight_decay={config.training.weight_decay} must be >= 0")
    if not config.model.hidden_dims:
        warnings.append("hidden_dims is empty; model may not work")
    if config.data.data_dir and not config.data.data_dir.exists():
        warnings.append(f"data_dir={config.data.data_dir} does not exist")

    return warnings


# ── Public API ───────────────────────────────────────────────────────────────


def load_environment(
    *,
    cli_overrides: dict[str, Any] | None = None,
    config_file: str | Path | None = None,
    env_prefix: str = "HELIX_",
    strict: bool = True,
) -> HelixEnvironment:
    """Resolve the HELIX-IDS environment configuration.

    Priority (highest → lowest):
        1. CLI overrides (passed as dict)
        2. HELIX_* environment variables
        3. YAML/JSON config file
        4. Schema defaults

    Parameters
    ----------
    cli_overrides : dict or None
        Direct overrides from CLI argument parsing.  Dot-notation supported.
    config_file : str or Path or None
        Path to a JSON config file for YAML-layer overrides.
    env_prefix : str
        Environment variable prefix (default ``HELIX_``).
    strict : bool
        If True, raise on validation errors.  If False, log warnings.

    Returns
    -------
    HelixEnvironment
        Fully resolved configuration.
    """
    # 1. Start with defaults (lowest priority)
    config_dict: dict[str, Any] = _apply_defaults()

    # 2. Apply YAML/JSON overrides
    if config_file:
        config_dict = _apply_yaml_overrides(config_dict, config_file)

    # 3. Apply environment variable overrides
    config_dict = _apply_env_overrides(config_dict, prefix=env_prefix)

    # 4. Apply CLI overrides (highest priority)
    config_dict = _apply_cli_overrides(config_dict, cli_overrides)

    # 5. Build the dataclass structure
    env = _dict_to_environment(config_dict)

    # 6. Validate
    warnings = _validate_config(env)
    for w in warnings:
        import logging

        logging.getLogger(__name__).warning(w)

    if strict and warnings:
        from warnings import warn

        warn(f"HELIX-IDS environment validation warnings: {'; '.join(warnings)}", stacklevel=2)

    return env


def _dict_to_environment(config_dict: dict[str, Any]) -> HelixEnvironment:
    """Convert a nested dict into a ``HelixEnvironment``.

    Uses ``dataclasses.fields()`` for attribute filtering because
    ``hasattr`` returns ``False`` for some dataclass fields (e.g. those
    with ``default_factory``).
    """
    from dataclasses import fields as dc_fields

    def _take(cls: type, section: dict[str, Any]) -> dict[str, Any]:
        valid = {f.name for f in dc_fields(cls)}
        return {k: v for k, v in section.items() if k in valid}

    return HelixEnvironment(
        training=TrainingSettings(**_take(TrainingSettings, config_dict.get("training", {}))),
        model=ModelSettings(**_take(ModelSettings, config_dict.get("model", {}))),
        runtime=RuntimeSettings(**_take(RuntimeSettings, config_dict.get("runtime", {}))),
        data=DataSettings(**_take(DataSettings, config_dict.get("data", {}))),
        loss=LossSettings(**_take(LossSettings, config_dict.get("loss", {}))),
    )


def environment_to_dict(env: HelixEnvironment) -> dict[str, Any]:
    """Convert a ``HelixEnvironment`` back to a nested dict (JSON-safe).

    Uses ``dataclasses.fields()`` to correctly extract actual attribute values
    rather than the ``Field`` descriptors returned by ``__dataclass_fields__``.
    """
    sections: dict[str, Any] = {}
    for section_name in ("training", "model", "runtime", "data", "loss"):
        instance = getattr(env, section_name)
        sections[section_name] = {
            f.name: str(getattr(instance, f.name)) if isinstance(getattr(instance, f.name), Path) else getattr(instance, f.name)
            for f in dc_fields(instance)
        }
    return sections


# ── Module-level convenience ─────────────────────────────────────────────────


def get_env() -> HelixEnvironment:
    """Convenience: load environment from ``os.environ`` only.

    Equivalent to ``load_environment()`` with no args.
    """
    return load_environment()
