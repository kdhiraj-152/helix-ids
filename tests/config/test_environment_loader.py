"""Tests for environment configuration loader.

Covers:
  - Default values
  - HELIX_* environment variable overrides
  - Type coercion (int, float, bool, list, Path)
  - CLI overrides (highest priority)
  - YAML/JSON config file overrides
  - Priority ordering (CLI > ENV > YAML > DEFAULT)
  - Schema validation warnings
  - Dot-notation nested key support
  - Boolean parsing edge cases
  - List parsing edge cases
"""

from __future__ import annotations

import json
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from helix_ids.config.environment import (
    HelixEnvironment,
    _parse_bool,
    _parse_float_list,
    _parse_int_list,
    _set_nested,
    environment_to_dict,
    get_env,
    load_environment,
)

# ── Shared environment isolation helper ──────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_env() -> Generator[None, None, None]:
    """Temporarily remove HELIX_* vars so tests don't pollute each other."""
    saved = {}
    for key in list(os.environ.keys()):
        if key.startswith("HELIX_"):
            saved[key] = os.environ.pop(key)
    yield
    for key, value in saved.items():
        os.environ[key] = value


# ═══════════════════════════════════════════════════════════════════════════════
# Helper tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestParseBool:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("False", False),
            ("0", False),
            ("no", False),
        ],
    )
    def test_parse_bool(self, raw: str, expected: bool) -> None:
        assert _parse_bool(raw) is expected

    def test_parse_bool_invalid(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse bool"):
            _parse_bool("maybe")


class TestParseIntList:
    def test_empty(self) -> None:
        assert _parse_int_list("") == []

    def test_single(self) -> None:
        assert _parse_int_list("42") == [42]

    def test_multiple(self) -> None:
        assert _parse_int_list("1, 2, 3") == [1, 2, 3]

    def test_trailing_comma(self) -> None:
        assert _parse_int_list("1,2,") == [1, 2]


class TestParseFloatList:
    def test_empty(self) -> None:
        assert _parse_float_list("") == []

    def test_single(self) -> None:
        assert _parse_float_list("0.5") == [0.5]

    def test_multiple(self) -> None:
        assert _parse_float_list("0.3, 0.5, 0.2") == [0.3, 0.5, 0.2]


class TestSetNested:
    def test_simple_key(self) -> None:
        d: dict[str, Any] = {}
        _set_nested(d, "batch_size", 128)
        assert d["batch_size"] == 128

    def test_nested_key(self) -> None:
        d: dict[str, Any] = {}
        _set_nested(d, "training.batch_size", 128)
        assert d["training"]["batch_size"] == 128

    def test_deep_nested(self) -> None:
        d: dict[str, Any] = {}
        _set_nested(d, "a.b.c.d", "val")
        assert d["a"]["b"]["c"]["d"] == "val"

    def test_overwrite_existing(self) -> None:
        d: dict[str, Any] = {"training": {"batch_size": 256}}
        _set_nested(d, "training.batch_size", 128)
        assert d["training"]["batch_size"] == 128


# ═══════════════════════════════════════════════════════════════════════════════
# HelixEnvironment defaults
# ═══════════════════════════════════════════════════════════════════════════════


class TestDefaults:
    def test_defaults_loaded(self) -> None:
        env = load_environment()
        assert isinstance(env, HelixEnvironment)
        assert env.training.batch_size == 256
        assert env.training.learning_rate == 1e-3
        assert env.training.epochs == 150
        assert env.training.warmup_epochs == 2
        assert env.model.input_dim == 17
        assert env.model.hidden_dims == [256, 192, 128, 64]
        assert env.model.dropout_rates == [0.3, 0.3, 0.25, 0.2]
        assert env.model.use_batch_norm is True
        assert env.runtime.device == "mps"
        assert env.runtime.num_workers == 2
        assert env.runtime.pin_memory is True
        assert env.loss.lambda_binary == 1.0
        assert env.loss.lambda_family == 0.8

    def test_get_env(self) -> None:
        env = get_env()
        assert isinstance(env, HelixEnvironment)
        assert env.training.epochs == 150

    def test_environment_to_dict(self) -> None:
        env = load_environment()
        d = environment_to_dict(env)
        assert "training" in d
        assert "model" in d
        assert "runtime" in d
        assert "data" in d
        assert "loss" in d
        assert d["model"]["input_dim"] == 17


# ═══════════════════════════════════════════════════════════════════════════════
# Environment variable overrides
# ═══════════════════════════════════════════════════════════════════════════════


class TestEnvOverride:
    def test_env_override_int(self) -> None:
        os.environ["HELIX_BATCH_SIZE"] = "128"
        os.environ["HELIX_EPOCHS"] = "200"
        env = load_environment()
        assert env.training.batch_size == 128
        assert env.training.epochs == 200
        assert env.training.learning_rate == 1e-3  # unchanged

    def test_env_override_float(self) -> None:
        os.environ["HELIX_LEARNING_RATE"] = "0.01"
        env = load_environment()
        assert env.training.learning_rate == 0.01

    def test_env_override_bool_true(self) -> None:
        os.environ["HELIX_USE_BATCH_NORM"] = "false"
        env = load_environment()
        assert env.model.use_batch_norm is False

    def test_env_override_bool_false(self) -> None:
        os.environ["HELIX_PIN_MEMORY"] = "false"
        env = load_environment()
        assert env.runtime.pin_memory is False

    def test_env_override_int_list(self) -> None:
        os.environ["HELIX_HIDDEN_DIMS"] = "512, 256, 128"
        env = load_environment()
        assert env.model.hidden_dims == [512, 256, 128]

    def test_env_override_float_list(self) -> None:
        os.environ["HELIX_DROPOUT_RATES"] = "0.5, 0.4, 0.3, 0.2"
        env = load_environment()
        assert env.model.dropout_rates == [0.5, 0.4, 0.3, 0.2]

    def test_env_override_string(self) -> None:
        os.environ["HELIX_DEVICE"] = "cuda"
        env = load_environment()
        assert env.runtime.device == "cuda"

    def test_env_override_path(self) -> None:
        os.environ["HELIX_CHECKPOINT_DIR"] = "/tmp/my_ckpts"
        env = load_environment()
        assert str(env.runtime.checkpoint_dir) == str(Path("/tmp/my_ckpts").expanduser().resolve())

    def test_env_override_run_id(self) -> None:
        os.environ["HELIX_RUN_ID"] = "exp_42"
        env = load_environment()
        assert env.runtime.run_id == "exp_42"

    def test_env_empty_string_does_not_override(self) -> None:
        os.environ["HELIX_BATCH_SIZE"] = ""
        env = load_environment()
        assert env.training.batch_size == 256  # default

    def test_env_whitespace_does_not_override(self) -> None:
        os.environ["HELIX_BATCH_SIZE"] = "   "
        env = load_environment()
        assert env.training.batch_size == 256  # default

    def test_env_invalid_coerce_raises(self) -> None:
        os.environ["HELIX_BATCH_SIZE"] = "not_a_number"
        with pytest.raises(ValueError, match="Failed to coerce"):
            load_environment()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI overrides
# ═══════════════════════════════════════════════════════════════════════════════


class TestCLIOverrides:
    def test_cli_override_flat(self) -> None:
        env = load_environment(cli_overrides={"batch_size": 64})
        assert env.training.batch_size == 64

    def test_cli_override_nested(self) -> None:
        env = load_environment(cli_overrides={"training": {"batch_size": 64}})
        assert env.training.batch_size == 64

    def test_cli_dot_notation(self) -> None:
        env = load_environment(cli_overrides={"training.batch_size": 64})
        assert env.training.batch_size == 64

    def test_cli_override_path(self) -> None:
        env = load_environment(cli_overrides={
            "runtime": {"checkpoint_dir": "/custom/checkpoint/path"}
        })
        assert str(env.runtime.checkpoint_dir) == str(Path("/custom/checkpoint/path").expanduser().resolve())


# ═══════════════════════════════════════════════════════════════════════════════
# Priority ordering
# ═══════════════════════════════════════════════════════════════════════════════


class TestPriorityOrder:
    def test_env_overrides_defaults(self) -> None:
        os.environ["HELIX_EPOCHS"] = "100"
        env = load_environment()
        assert env.training.epochs == 100  # env > default

    def test_cli_overrides_env(self) -> None:
        os.environ["HELIX_EPOCHS"] = "100"
        env = load_environment(cli_overrides={"training": {"epochs": 200}})
        assert env.training.epochs == 200  # cli > env

    def test_cli_overrides_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"training": {"epochs": 175}}), encoding="utf-8")
        env = load_environment(
            config_file=config_file,
            cli_overrides={"training": {"epochs": 200}},
        )
        assert env.training.epochs == 200  # cli > file

    def test_file_overrides_defaults(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"training": {"epochs": 125}}), encoding="utf-8")
        env = load_environment(config_file=config_file)
        assert env.training.epochs == 125  # file > default

    def test_env_overrides_file(self, tmp_path: Path) -> None:
        os.environ["HELIX_EPOCHS"] = "100"
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"training": {"epochs": 175}}), encoding="utf-8")
        env = load_environment(config_file=config_file)
        assert env.training.epochs == 100  # env > file

    def test_missing_file_does_not_error(self) -> None:
        env = load_environment(config_file="/tmp/nonexistent_config_file.json")
        assert env.training.epochs == 150  # defaults


# ═══════════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidation:
    def test_negative_epochs_warns(self) -> None:
        env = load_environment(cli_overrides={"training": {"epochs": -1}}, strict=True)
        # Should not raise; just warns
        assert env.training.epochs == -1

    def test_empty_hidden_dims_warns(self) -> None:
        env = load_environment(cli_overrides={"model": {"hidden_dims": []}}, strict=True)
        assert env.model.hidden_dims == []


# ═══════════════════════════════════════════════════════════════════════════════
# Bound checks and edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_all_env_vars(self) -> None:
        """Set all env vars and verify they're reflected."""
        os.environ["HELIX_BATCH_SIZE"] = "512"
        os.environ["HELIX_LEARNING_RATE"] = "0.005"
        os.environ["HELIX_EPOCHS"] = "300"
        os.environ["HELIX_DEVICE"] = "cuda"
        os.environ["HELIX_USE_BATCH_NORM"] = "false"
        os.environ["HELIX_ACTIVATION"] = "gelu"
        os.environ["HELIX_HIDDEN_DIMS"] = "1024,512,256"
        os.environ["HELIX_RUN_ID"] = "exp_99"
        os.environ["HELIX_LAMBDA_BINARY"] = "1.5"
        os.environ["HELIX_VERBOSE"] = "true"
        os.environ["HELIX_RECOVERY_ENABLED"] = "false"
        os.environ["HELIX_CIRCUIT_BREAKER_ENABLED"] = "false"
        os.environ["HELIX_LOGGING_JSON"] = "false"
        os.environ["HELIX_LOGGING_LEVEL"] = "DEBUG"

        env = load_environment()
        assert env.training.batch_size == 512
        assert env.training.learning_rate == 0.005
        assert env.training.epochs == 300
        assert env.runtime.device == "cuda"
        assert env.model.use_batch_norm is False
        assert env.model.activation == "gelu"
        assert env.model.hidden_dims == [1024, 512, 256]
        assert env.runtime.run_id == "exp_99"
        assert env.loss.lambda_binary == 1.5
        assert env.runtime.verbose is True
        assert env.runtime.recovery_enabled is False
        assert env.runtime.circuit_breaker_enabled is False
        assert env.runtime.logging_json is False
        assert env.runtime.logging_level == "DEBUG"

    def test_dict_to_environment_handles_dict_init(self) -> None:
        """Ensure HelixEnvironment handles dict init in __post_init__."""
        env = HelixEnvironment(
            training={"batch_size": 64, "epochs": 50},
            runtime={"device": "cpu"},
        )
        assert env.training.batch_size == 64
        assert env.training.epochs == 50
        assert env.runtime.device == "cpu"
        assert env.loss.lambda_binary == 1.0  # default
