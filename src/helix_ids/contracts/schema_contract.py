"""Immutable runtime schema contract for HELIX-IDS."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

CANONICAL_FEATURE_ORDER = [
    "protocol_type",
    "connection_state",
    "traffic_direction",
    "has_rst",
    "log_src_bytes",
    "log_dst_bytes",
    "src_dst_bytes_ratio",
    "dst_src_bytes_ratio",
    "same_host_rate_x_service",
    "diff_srv_rate_x_flag",
    "count_x_srv_count",
    "protocol_service_flag",
    "src_bytes",
    "dst_bytes",
    "service_tier",
    "duration",
    "flag",
]
CANONICAL_INPUT_DIM = 17
CANONICAL_BINARY_CLASSES = 2
CANONICAL_FAMILY_CLASSES = 7
SCHEMA_VERSION = "2026-05-25"


def _canonical_schema_payload(
    *,
    schema_version: str = SCHEMA_VERSION,
    feature_order: Sequence[str] = CANONICAL_FEATURE_ORDER,
    input_dim: int = CANONICAL_INPUT_DIM,
    binary_output_dim: int = CANONICAL_BINARY_CLASSES,
    family_output_dim: int = CANONICAL_FAMILY_CLASSES,
) -> dict[str, Any]:
    return {
        "schema_version": str(schema_version),
        "feature_order": [str(feature) for feature in feature_order],
        "input_dim": int(input_dim),
        "binary_output_dim": int(binary_output_dim),
        "family_output_dim": int(family_output_dim),
    }


def compute_schema_hash(
    *,
    schema_version: str = SCHEMA_VERSION,
    feature_order: Sequence[str] = CANONICAL_FEATURE_ORDER,
    input_dim: int = CANONICAL_INPUT_DIM,
    binary_output_dim: int = CANONICAL_BINARY_CLASSES,
    family_output_dim: int = CANONICAL_FAMILY_CLASSES,
) -> str:
    payload = _canonical_schema_payload(
        schema_version=schema_version,
        feature_order=feature_order,
        input_dim=input_dim,
        binary_output_dim=binary_output_dim,
        family_output_dim=family_output_dim,
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


SCHEMA_HASH = compute_schema_hash()


def validate_feature_order(feature_order: Sequence[str], *, context: str = "feature order") -> None:
    actual = [str(feature) for feature in feature_order]
    expected = list(CANONICAL_FEATURE_ORDER)
    if actual != expected:
        raise AssertionError(f"{context} must exactly match the canonical feature order")


def assert_runtime_contract(
    *,
    schema_version: str,
    schema_hash: str,
    feature_order: Sequence[str],
    input_dim: int,
    binary_output_dim: int,
    family_output_dim: int,
    context: str = "runtime contract",
) -> None:
    validate_feature_order(feature_order, context=context)

    actual_input_dim = int(input_dim)
    actual_binary_output_dim = int(binary_output_dim)
    actual_family_output_dim = int(family_output_dim)
    actual_schema_version = str(schema_version)
    actual_schema_hash = str(schema_hash)

    if actual_input_dim != CANONICAL_INPUT_DIM:
        raise AssertionError(
            f"{context} input_dim mismatch: expected {CANONICAL_INPUT_DIM}, got {actual_input_dim}"
        )
    if actual_binary_output_dim != CANONICAL_BINARY_CLASSES:
        raise AssertionError(
            f"{context} binary_output_dim mismatch: expected {CANONICAL_BINARY_CLASSES}, got {actual_binary_output_dim}"
        )
    if actual_family_output_dim != CANONICAL_FAMILY_CLASSES:
        raise AssertionError(
            f"{context} family_output_dim mismatch: expected {CANONICAL_FAMILY_CLASSES}, got {actual_family_output_dim}"
        )
    if actual_schema_version != SCHEMA_VERSION:
        raise AssertionError(
            f"{context} schema_version mismatch: expected {SCHEMA_VERSION}, got {actual_schema_version}"
        )

    expected_hash = compute_schema_hash(
        schema_version=actual_schema_version,
        feature_order=feature_order,
        input_dim=actual_input_dim,
        binary_output_dim=actual_binary_output_dim,
        family_output_dim=actual_family_output_dim,
    )
    if actual_schema_hash != expected_hash:
        raise AssertionError(
            f"{context} schema_hash mismatch: expected {expected_hash}, got {actual_schema_hash}"
        )


def runtime_contract_payload() -> dict[str, Any]:
    from helix_ids.contracts.immutable_constants import CONTRACT_VERSION, FEATURE_ORDER_HASH
    return {
        "schema_version": SCHEMA_VERSION,
        "schema_hash": SCHEMA_HASH,
        "contract_version": CONTRACT_VERSION,
        "feature_order_hash": FEATURE_ORDER_HASH,
        "feature_order": list(CANONICAL_FEATURE_ORDER),
        "input_dim": CANONICAL_INPUT_DIM,
        "binary_output_dim": CANONICAL_BINARY_CLASSES,
        "family_output_dim": CANONICAL_FAMILY_CLASSES,
    }
