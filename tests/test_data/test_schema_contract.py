from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from helix_ids.data.feature_harmonization import (
    FEATURE_ORDER,
    SchemaDriftError,
    compute_schema_hash,
    enforce_feature_order,
)

EXPECTED_FEATURE_ORDER = [
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


def test_feature_order_is_exact_and_stable() -> None:
    assert len(FEATURE_ORDER) == 17
    assert list(FEATURE_ORDER) == EXPECTED_FEATURE_ORDER



def test_schema_hash_is_deterministic() -> None:
    frame = pd.DataFrame(
        np.arange(34, dtype=np.float32).reshape(2, 17),
        columns=FEATURE_ORDER,
    )

    first = compute_schema_hash(frame)
    second = compute_schema_hash(frame.copy())

    assert first == second



def test_enforce_feature_order_rejects_reordered_columns() -> None:
    frame = pd.DataFrame(
        np.arange(34, dtype=np.float32).reshape(2, 17),
        columns=list(reversed(FEATURE_ORDER)),
    )

    with pytest.raises(SchemaDriftError):
        enforce_feature_order(frame, FEATURE_ORDER, context="schema-contract")
