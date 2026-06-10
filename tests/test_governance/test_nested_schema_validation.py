"""Tests for nested schema validation in validate_benchmark_outputs.py.

Phase 4B Objective 2 — covers:
  - config_hashes value non-emptiness (any present key must be non-empty string)
  - dataset_hashes sub-key completeness (raw, processed, split, primary)
  - non-empty value enforcement
  - malformed object handling
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ci.validate_benchmark_outputs import (
    _check_config_hashes,
    _check_dataset_hashes,
)

# ---------------------------------------------------------------------------
# _check_config_hashes — valid and invalid cases
# ---------------------------------------------------------------------------

def _payload(config_hashes: dict | None) -> dict:
    return {"config_hashes": config_hashes} if config_hashes is not None else {}


class TestCheckConfigHashes:
    """Validate _check_config_hashes: any present key must be a non-empty string."""

    def test_valid_all_keys_present_and_non_empty(self):
        # Use real schema key names; any non-empty values pass.
        payload = _payload({
            "helix_config": "h",
            "training_config": "t",
            "platform_configs": "pconf",
        })
        checks, failed = _check_config_hashes(payload)
        fails = [c for c in checks if c["status"] == "fail"]
        assert not failed, f"Expected pass, got: {fails}"
        assert len([c for c in checks if c["status"] == "pass"]) == 3

    def test_valid_single_key(self):
        payload = _payload({"helix_config": "h"})
        checks, failed = _check_config_hashes(payload)
        assert not failed
        assert len([c for c in checks if c["status"] == "pass"]) == 1

    def test_missing_entire_config_hashes(self):
        payload = {}
        checks, failed = _check_config_hashes(payload)
        assert failed
        assert any("missing or not a mapping" in c["message"] for c in checks)

    def test_not_a_mapping(self):
        payload = {"config_hashes": "not-a-dict"}
        checks, failed = _check_config_hashes(payload)
        assert failed
        assert any("missing or not a mapping" in c["message"] for c in checks)

    def test_empty_mapping_fails(self):
        payload = _payload({})
        checks, failed = _check_config_hashes(payload)
        assert failed
        assert any("empty" in c["message"] for c in checks)

    def test_empty_string_value_fails(self):
        payload = _payload({"helix_config": "", "training_config": "t"})
        checks, failed = _check_config_hashes(payload)
        assert failed
        assert any(
            "'helix_config'" in c["message"] and "empty" in c["message"]
            for c in checks
        )

    def test_whitespace_only_value_fails(self):
        payload = _payload({"helix_config": "   ", "training_config": "t"})
        checks, failed = _check_config_hashes(payload)
        assert failed
        assert any("empty" in c["message"] for c in checks)

    def test_none_value_fails(self):
        payload = _payload({"helix_config": None})
        checks, failed = _check_config_hashes(payload)
        assert failed
        assert any("'helix_config'" in c["message"] and "empty" in c["message"] for c in checks)

    def test_extra_keys_ignored(self):
        payload = _payload({"helix_config": "h", "training_config": "t", "extra": "ignored"})
        checks, failed = _check_config_hashes(payload)
        assert not failed

    def test_empty_value_in_single_key_fails(self):
        payload = _payload({"helix_config": ""})
        checks, failed = _check_config_hashes(payload)
        assert failed
        fail_msgs = [c["message"] for c in checks if c["status"] == "fail"]
        assert len(fail_msgs) == 1  # only the one key, and it's empty


# ---------------------------------------------------------------------------
# _check_dataset_hashes — valid and invalid cases
# ---------------------------------------------------------------------------

def _ds_payload(dataset_hashes: dict | None, dataset_hash_primary: str = "primary-hash") -> dict:
    base = {"dataset_hashes": dataset_hashes, "dataset_hash_primary": dataset_hash_primary} if dataset_hashes is not None else {"dataset_hash_primary": dataset_hash_primary}
    return base


class TestCheckDatasetHashes:
    def test_valid_all_four_present_and_non_empty(self):
        payload = _ds_payload(
            {"raw": "raw-h", "processed": "proc-h", "split": "split-h", "primary": "primary-h"},
            dataset_hash_primary="primary-h",
        )
        checks, failed = _check_dataset_hashes(payload)
        fails = [c for c in checks if c["status"] == "fail"]
        assert not failed, f"Expected pass, got: {fails}"

    def test_missing_entire_dataset_hashes(self):
        payload = {"dataset_hash_primary": "primary"}
        checks, failed = _check_dataset_hashes(payload)
        assert failed
        assert any("missing or not a mapping" in c["message"] for c in checks)

    def test_dataset_hashes_not_a_mapping(self):
        payload = {"dataset_hashes": "not-a-dict", "dataset_hash_primary": "primary"}
        checks, failed = _check_dataset_hashes(payload)
        assert failed

    def test_missing_raw_subkey(self):
        payload = _ds_payload({"processed": "p", "split": "s", "primary": "pr"})
        checks, failed = _check_dataset_hashes(payload)
        assert failed
        assert any("'raw'" in c["message"] for c in checks if c["status"] == "fail")

    def test_missing_processed_subkey(self):
        payload = _ds_payload({"raw": "r", "split": "s", "primary": "pr"})
        checks, failed = _check_dataset_hashes(payload)
        assert failed
        assert any("'processed'" in c["message"] for c in checks if c["status"] == "fail")

    def test_missing_split_subkey(self):
        payload = _ds_payload({"raw": "r", "processed": "p", "primary": "pr"})
        checks, failed = _check_dataset_hashes(payload)
        assert failed
        assert any("'split'" in c["message"] for c in checks if c["status"] == "fail")

    def test_missing_primary_subkey(self):
        payload = _ds_payload({"raw": "r", "processed": "p", "split": "s"})
        checks, failed = _check_dataset_hashes(payload)
        assert failed
        assert any("'primary'" in c["message"] for c in checks if c["status"] == "fail")

    def test_empty_string_value_fails(self):
        payload = _ds_payload({"raw": "", "processed": "p", "split": "s", "primary": "pr"})
        checks, failed = _check_dataset_hashes(payload)
        assert failed
        assert any("'raw'" in c["message"] and "empty" in c["message"] for c in checks)

    def test_none_value_fails(self):
        payload = _ds_payload({"raw": None, "processed": "p", "split": "s", "primary": "pr"})
        checks, failed = _check_dataset_hashes(payload)
        assert failed

    def test_primary_mismatch_with_dataset_hash_primary_fails(self):
        payload = _ds_payload(
            {"raw": "r", "processed": "p", "split": "s", "primary": "wrong-primary"},
            dataset_hash_primary="correct-primary",
        )
        checks, failed = _check_dataset_hashes(payload)
        assert failed
        assert any("mismatch" in c["message"].lower() for c in checks)

    def test_primary_match_passes(self):
        payload = _ds_payload(
            {"raw": "r", "processed": "p", "split": "s", "primary": "correct-primary"},
            dataset_hash_primary="correct-primary",
        )
        checks, failed = _check_dataset_hashes(payload)
        fails = [c for c in checks if c["status"] == "fail"]
        assert not fails

    def test_extra_keys_ignored(self):
        payload = _ds_payload(
            {"raw": "r", "processed": "p", "split": "s", "primary": "pr", "extra": "ignored"},
            dataset_hash_primary="pr",
        )
        checks, failed = _check_dataset_hashes(payload)
        assert not failed

    def test_partial_missing_multiple(self):
        payload = _ds_payload({"raw": ""})  # empty raw + missing others
        checks, failed = _check_dataset_hashes(payload)
        assert failed
        fail_msgs = [c["message"] for c in checks if c["status"] == "fail"]
        assert len(fail_msgs) >= 3  # processed, split, primary missing; raw empty
