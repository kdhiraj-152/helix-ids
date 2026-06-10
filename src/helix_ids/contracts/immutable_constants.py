"""Immutable governance constants for HELIX-IDS."""

from __future__ import annotations

import hashlib
import json

from helix_ids.contracts.schema_contract import CANONICAL_FEATURE_ORDER, SCHEMA_HASH

CONTRACT_VERSION = "2.1"
EXPORTER_API_VERSION = "1.0.0"
MANIFEST_VERSION = "1.0.0"
FEATURE_ORDER_HASH = hashlib.sha256(
    json.dumps({"feature_order": [str(feature) for feature in CANONICAL_FEATURE_ORDER]}, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()

__all__ = [
    "CONTRACT_VERSION",
    "EXPORTER_API_VERSION",
    "FEATURE_ORDER_HASH",
    "MANIFEST_VERSION",
    "SCHEMA_HASH",
]
