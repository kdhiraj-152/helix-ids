"""Shared contract schemas and validation utilities for HELIX-IDS."""

from .diagnostic_contract import (
    DECISION_MODES,
    DECISION_TRANSITIONS,
    DiagnosticContract,
    enforce_decision_transition,
    migrate_contract_payload,
    validate_diagnostic_contract,
)
from .immutable_constants import (
    CONTRACT_VERSION,
    EXPORTER_API_VERSION,
    FEATURE_ORDER_HASH,
    MANIFEST_VERSION,
    SCHEMA_HASH,
)
from .schema_contract import (
    CANONICAL_BINARY_CLASSES,
    CANONICAL_FAMILY_CLASSES,
    CANONICAL_FEATURE_ORDER,
    CANONICAL_INPUT_DIM,
    SCHEMA_VERSION,
    assert_runtime_contract,
    compute_schema_hash,
    runtime_contract_payload,
    validate_feature_order,
)

__all__ = [
    "CANONICAL_FEATURE_ORDER",
    "CANONICAL_INPUT_DIM",
    "CANONICAL_BINARY_CLASSES",
    "CANONICAL_FAMILY_CLASSES",
    "SCHEMA_VERSION",
    "SCHEMA_HASH",
    "FEATURE_ORDER_HASH",
    "compute_schema_hash",
    "validate_feature_order",
    "assert_runtime_contract",
    "runtime_contract_payload",
    "CONTRACT_VERSION",
    "EXPORTER_API_VERSION",
    "MANIFEST_VERSION",
    "DECISION_TRANSITIONS",
    "DECISION_MODES",
    "DiagnosticContract",
    "enforce_decision_transition",
    "migrate_contract_payload",
    "validate_diagnostic_contract",
]
