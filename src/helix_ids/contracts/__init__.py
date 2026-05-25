"""Shared contract schemas and validation utilities for HELIX-IDS."""

from .schema_contract import (
    CANONICAL_BINARY_CLASSES,
    CANONICAL_FEATURE_ORDER,
    CANONICAL_FAMILY_CLASSES,
    CANONICAL_INPUT_DIM,
    SCHEMA_HASH,
    SCHEMA_VERSION,
    assert_runtime_contract,
    compute_schema_hash,
    runtime_contract_payload,
    validate_feature_order,
)

from .diagnostic_contract import (
    CONTRACT_VERSION,
    DECISION_TRANSITIONS,
    DECISION_MODES,
    DiagnosticContract,
    enforce_decision_transition,
    migrate_contract_payload,
    validate_diagnostic_contract,
)

__all__ = [
    "CANONICAL_FEATURE_ORDER",
    "CANONICAL_INPUT_DIM",
    "CANONICAL_BINARY_CLASSES",
    "CANONICAL_FAMILY_CLASSES",
    "SCHEMA_VERSION",
    "SCHEMA_HASH",
    "compute_schema_hash",
    "validate_feature_order",
    "assert_runtime_contract",
    "runtime_contract_payload",
    "CONTRACT_VERSION",
    "DECISION_TRANSITIONS",
    "DECISION_MODES",
    "DiagnosticContract",
    "enforce_decision_transition",
    "migrate_contract_payload",
    "validate_diagnostic_contract",
]
