"""Tests for canonical governance fingerprinting utilities."""

from pathlib import Path

import pandas as pd

from helix_ids.governance.fingerprinting import (
    build_dataset_manifest_hash,
    build_run_fingerprint,
    build_schema_hash_from_frame,
)


def test_dataset_manifest_hash_stable(tmp_path: Path):
    file_a = tmp_path / "a.csv"
    file_b = tmp_path / "b.csv"
    file_a.write_text("x,y\n1,2\n", encoding="utf-8")
    file_b.write_text("x,y\n3,4\n", encoding="utf-8")

    hash_1 = build_dataset_manifest_hash([file_a, file_b])
    hash_2 = build_dataset_manifest_hash([file_b, file_a])

    assert hash_1 == hash_2


def test_schema_hash_changes_on_label_vocab_change():
    frame = pd.DataFrame({"duration": [1.0, 2.0], "src_bytes": [10, 11]})

    hash_a = build_schema_hash_from_frame(frame, ["Normal", "DoS"])
    hash_b = build_schema_hash_from_frame(frame, ["Normal", "DoS", "Probe"])

    assert hash_a != hash_b


def test_run_fingerprint_changes_with_commit_sha():
    base_args = {
        "dataset_hashes": {"nsl": "abc", "unsw": "def"},
        "mapping_version": "mapping-v1",
        "schema_hash": "schema-1",
        "model_config_hash": "config-1",
    }

    fingerprint_a = build_run_fingerprint(commit_sha="111", **base_args)
    fingerprint_b = build_run_fingerprint(commit_sha="222", **base_args)

    assert fingerprint_a != fingerprint_b
