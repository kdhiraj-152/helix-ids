"""
Training-layer constants isolated from core domain package.

These constants are specific to training-script feature-engineering logic
and should not live in src/helix_ids/.
"""

from __future__ import annotations

ENGINEERED_FEATURE_NAMES: frozenset[str] = frozenset(
    {
        "log_src_bytes",
        "log_dst_bytes",
        "src_dst_bytes_ratio",
        "dst_src_bytes_ratio",
        "same_host_rate_x_service",
        "diff_srv_rate_x_flag",
        "count_x_srv_count",
        "protocol_service_flag",
    }
)
