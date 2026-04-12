"""Canonical hashing and fingerprint generation helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_hash(payload: dict[str, Any]) -> str:
    """Return SHA256 hash of sorted canonical JSON bytes."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def build_dataset_manifest_hash(files: list[Path]) -> str:
    """Build canonical dataset hash from file metadata and contents."""
    manifest_entries: list[dict[str, Any]] = []
    for file_path in sorted(files, key=lambda p: str(p)):
        file_bytes = file_path.read_bytes()
        manifest_entries.append(
            {
                "path": str(file_path),
                "size": file_path.stat().st_size,
                "sha256": _sha256_bytes(file_bytes),
            }
        )

    return canonical_json_hash({"files": manifest_entries})


def build_schema_hash_from_frame(df: pd.DataFrame, label_vocabulary: list[str]) -> str:
    """Build schema hash from deterministic feature and label schema description."""
    features = [
        {
            "name": col,
            "dtype": str(df[col].dtype),
            "nullable": bool(df[col].isnull().any()),
        }
        for col in df.columns
    ]
    payload = {
        "features": sorted(features, key=lambda item: item["name"]),
        "label_vocabulary": sorted(label_vocabulary),
    }
    return canonical_json_hash(payload)


def build_run_fingerprint(
    *,
    dataset_hashes: dict[str, str],
    mapping_version: str,
    schema_hash: str,
    model_config_hash: str,
    commit_sha: str,
) -> str:
    """Build deterministic run fingerprint in fixed field order."""
    payload = {
        "dataset_hashes": {k: dataset_hashes[k] for k in sorted(dataset_hashes)},
        "mapping_version": mapping_version,
        "schema_hash": schema_hash,
        "model_config_hash": model_config_hash,
        "commit_sha": commit_sha,
        "fingerprint_spec_version": 1,
    }
    return canonical_json_hash(payload)
