"""Invariant feature harmonization for multi-dataset IDS training.

This module intentionally defines a compact, behavior-level representation
shared by NSL-KDD, UNSW-NB15, and CICIDS. It avoids dataset-specific
synthetic fills and removes non-transferable proxy features.
"""

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..contracts import ATTACK_TAXONOMY_7CLASS, CONTRACT_VERSION
from ..contracts.schema_contract import (
    CANONICAL_FEATURE_ORDER,
    SCHEMA_VERSION,
    assert_runtime_contract,
    validate_feature_order,
)
from ..contracts.schema_contract import (
    compute_schema_hash as _compute_schema_hash_contract,
)


def compute_schema_hash(*args, **kwargs) -> str:
    """Compatibility wrapper for computing schema hashes.

    Supported call patterns:
    - `compute_schema_hash()` -> canonical schema hash for this repo
    - `compute_schema_hash(df_or_sequence)` -> compute hash for DataFrame/sequence
    - `compute_schema_hash(**contract_kwargs)` -> forwarded to canonical compute
    """
    # No args/kwargs -> canonical
    if not args and not kwargs:
        return _compute_schema_hash_contract(
            schema_version=SCHEMA_VERSION,
            feature_order=list(FEATURE_ORDER),
            input_dim=len(FEATURE_ORDER),
        )

    # Single positional DataFrame or sequence
    if len(args) == 1 and not kwargs:
        obj = args[0]
        if hasattr(obj, "columns"):
            feature_order = list(obj.columns)
            shape = getattr(obj, "shape", None)
            if isinstance(shape, tuple) and len(shape) > 1 and shape[1] is not None:
                input_dim = int(shape[1])
            else:
                input_dim = len(feature_order)
            return _compute_schema_hash_contract(
                schema_version=SCHEMA_VERSION,
                feature_order=feature_order,
                input_dim=input_dim,
            )
        if isinstance(obj, (list, tuple)):
            feature_order = list(obj)
            input_dim = len(feature_order)
            return _compute_schema_hash_contract(
                schema_version=SCHEMA_VERSION,
                feature_order=feature_order,
                input_dim=input_dim,
            )

    # Fallback: pass kwargs through to canonical implementation
    return _compute_schema_hash_contract(**kwargs)

# ============================================================================
# Strict Common Feature Space (17 features)
#
# These are the engineered, invariant features produced by the repository's
# multi-dataset pipeline (see `multi_dataset_loader._augment_geometry_expansion_features`).
# This canonical list is the runtime contract for HelixIDS-Full.
# ============================================================================

COMMON_FEATURES_METADATA = {
    "protocol_type": {"type": "categorical", "description": "transport protocol"},
    "connection_state": {"type": "categorical", "description": "derived connection state"},
    "traffic_direction": {"type": "categorical", "description": "traffic direction proxy"},
    "has_rst": {"type": "numeric", "description": "RST flag indicator"},
    "log_src_bytes": {"type": "numeric", "description": "log(1+src_bytes)"},
    "log_dst_bytes": {"type": "numeric", "description": "log(1+dst_bytes)"},
    "src_dst_bytes_ratio": {"type": "numeric", "description": "src / (dst + 1)"},
    "dst_src_bytes_ratio": {"type": "numeric", "description": "dst / (src + 1)"},
    "same_host_rate_x_service": {"type": "numeric", "description": "interaction: same host rate x service"},
    "diff_srv_rate_x_flag": {"type": "numeric", "description": "interaction: diff_srv_rate x flag"},
    "count_x_srv_count": {"type": "numeric", "description": "interaction: count x srv_count"},
    "protocol_service_flag": {"type": "numeric", "description": "interaction: protocol x service x flag"},
    "src_bytes": {"type": "numeric", "description": "raw source bytes"},
    "dst_bytes": {"type": "numeric", "description": "raw destination bytes"},
    "service_tier": {"type": "categorical", "description": "derived service tier"},
    "duration": {"type": "numeric", "description": "connection duration"},
    "flag": {"type": "categorical", "description": "connection flag/state"},
}

COMMON_FEATURES = list(COMMON_FEATURES_METADATA.keys())
if len(COMMON_FEATURES) != 17:
    raise AssertionError(f"Expected 17 common features, got {len(COMMON_FEATURES)}")
FEATURE_ORDER = list(CANONICAL_FEATURE_ORDER)

LEAKAGE_PRONE_FEATURES: set[str] = set()
INVARIANT_FEATURES = COMMON_FEATURES.copy()

PROTOCOL_MAP = {"tcp": 0, "udp": 1, "icmp": 2}

# Shared flag/state encoding across datasets.
FLAG_MAP = {
    "sf": 0,
    "s0": 1,
    "s1": 2,
    "s2": 3,
    "s3": 4,
    "rej": 5,
    "rsto": 6,
    "rstr": 7,
    "rstos0": 8,
    "sh": 9,
    "oth": 10,
    "con": 11,
    "int": 12,
    "fin": 13,
    "req": 14,
    "rst": 15,
    "eco": 16,
    "no": 17,
    "par": 18,
    "urn": 19,
    "acc": 20,
    "clo": 21,
}

REQUIRED_KEYS = {"protocol", "mapping", "version"}
PIPELINE_MODES = {"strict", "lenient"}
ARTIFACT_REQUIRED_KEYS = [
    "model",
    "schema_version",
    "contract_version",
    "schema_hash",
    "feature_order",
    "input_dim",
    "binary_output_dim",
    "family_output_dim",
]

SCHEMA: dict[str, type] = {
    "protocol_type": int,
    "connection_state": int,
    "traffic_direction": int,
    "has_rst": int,
    "log_src_bytes": float,
    "log_dst_bytes": float,
    "src_dst_bytes_ratio": float,
    "dst_src_bytes_ratio": float,
    "same_host_rate_x_service": float,
    "diff_srv_rate_x_flag": float,
    "count_x_srv_count": float,
    "protocol_service_flag": float,
    "src_bytes": float,
    "dst_bytes": float,
    "service_tier": int,
    "duration": float,
    "flag": int,
}

CANONICAL_CONNECTION_STATES = ("S0", "S1", "SF", "REJ", "RST", "FIN", "CON", "INT", "OTH")
CONNECTION_STATE_MAP = {state.lower(): idx for idx, state in enumerate(CANONICAL_CONNECTION_STATES)}
TRAFFIC_DIRECTION_MAP = {"outbound": 0, "balanced": 1, "inbound": 2}
SERVICE_TIER_MAP = {
    "none": 0,
    "web": 1,
    "dns": 2,
    "ftp": 3,
    "mail": 4,
    "auth": 5,
    "other": 6,
}
CICIDS_FWD_BYTES_COL = "TotLen Fwd Pkts"
CICIDS_BWD_BYTES_COL = "TotLen Bwd Pkts"
CICIDS_RST_FLAG_COL = "RST Flag Cnt"

NSL_STATE_MAP = {
    "s0": "S0",
    "s1": "S1",
    "sf": "SF",
    "rej": "REJ",
    "rsto": "RST",
    "rstr": "RST",
    "rstos0": "RST",
    "fin": "FIN",
    "con": "CON",
    "int": "INT",
    "oth": "OTH",
}

UNSW_STATE_MAP = {
    "fin": "FIN",
    "int": "INT",
    "con": "CON",
    "rst": "RST",
    "req": "S1",
    "acc": "S1",
    "clo": "FIN",
    "no": "S0",
    "par": "OTH",
    "eco": "OTH",
    "urn": "OTH",
}
UNSW_FROZEN_STATE_MAP = {
    "fin": "FIN",
    "int": "INT",
    "con": "CON",
}
UNSW_AMBIGUOUS_STATES = frozenset({"req", "acc", "clo", "no", "par", "eco", "urn"})
UNSW_RARE_STATE_FREQ_THRESHOLD = 0.005


def validate_mapping(m: Mapping[str, object]) -> None:
    for k in REQUIRED_KEYS:
        assert k in m, f"Missing required key: {k}"


def coerce_numeric_strict(col: pd.Series) -> pd.Series:
    return pd.to_numeric(col, errors="raise")


def sanitize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).copy()
    num = num.replace([np.inf, -np.inf], np.nan)
    num = num.fillna(0.0)
    df = df.copy()
    df[num.columns] = num
    return df


def validate_no_nan_inf(df: pd.DataFrame) -> None:
    invalid_cols = list(df.columns[df.isnull().any()])
    numeric = df.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=np.float64, copy=False)).all():
        invalid_cols.extend([str(col) for col in numeric.columns[~np.isfinite(numeric.to_numpy(dtype=np.float64, copy=False)).all(axis=0)]])
    if invalid_cols:
        raise AssertionError(f"NaN/inf detected in input; invalid_cols={sorted(set(invalid_cols))}")


def normalize_column_name(name: str) -> str:
    """Normalize a feature name for resilient matching across datasets."""
    return str(name).strip().lower().replace("_", " ")


def normalize_per_dataset(X):
    """Normalize a single dataset tensor with z-score scaling."""

    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True)

    std[std < 1e-6] = 1.0
    return (X - mean) / std


# ============================================================================
# Attack Taxonomy: 7-class mapping
# Imported from helix_ids.contracts.attack_taxonomy — see that module for
# definitions: ATTACK_TAXONOMY_7CLASS, NSLKDD_TO_7CLASS, UNSW_TO_7CLASS,
# CICIDS_TO_7CLASS, CICIDS2018_TO_7CLASS.
# ============================================================================


# ============================================================================
# Dataset Raw-Column Contracts
# ============================================================================


@dataclass
class FeatureMapping:
    """Raw column aliases required to construct the invariant feature space."""

    dataset_name: str
    original_features: list[str]
    common_features: list[str]
    feature_mapping: Mapping[str, str | list[str]]

    def to_dict(self):
        payload = asdict(self)
        contract = {
            "protocol": str(payload.get("protocol", "v1")),
            "version": SCHEMA_VERSION,
            "mapping": payload,
        }
        contract["protocol"] = contract.get("protocol", "v1")
        validate_mapping(contract)
        return contract


class SchemaDriftError(AssertionError):
    def __init__(
        self,
        expected: Sequence[str] | str,
        actual: Sequence[str] | str,
        *,
        context: str = "feature order",
        missing: Sequence[str] | None = None,
        extra: Sequence[str] | None = None,
    ):
        self.expected = expected
        self.actual = actual

        if missing is not None:
            self.missing = list(missing)
        elif isinstance(expected, str) or isinstance(actual, str):
            self.missing = []
        else:
            self.missing = list(set(expected) - set(actual))

        if extra is not None:
            self.extra = list(extra)
        elif isinstance(expected, str) or isinstance(actual, str):
            self.extra = []
        else:
            self.extra = list(set(actual) - set(expected))

        self.order_mismatch = bool(expected != actual)

        message = (
            f"Feature schema drift detected ({context}); expected={expected}; actual={actual}; "
            f"missing={self.missing}; extra={self.extra}"
        )
        if self.order_mismatch:
            message += "; order_mismatch=True"
        super().__init__(message)


def enforce_feature_order(df: pd.DataFrame, feature_order: Sequence[str], *, context: str = "feature order") -> pd.DataFrame:
    actual_columns = list(df.columns)
    if actual_columns != list(feature_order):
        raise SchemaDriftError(feature_order, actual_columns, context=context)
    return df.loc[:, list(feature_order)].copy()


def _validate_artifact_keys(artifact: Mapping[str, Any]) -> None:
    for key in ARTIFACT_REQUIRED_KEYS:
        assert key in artifact, f"Missing required artifact key: {key}"


def load_artifact(
    path: Path | str,
    df: pd.DataFrame,
    *,
    mapping: FeatureMapping | None = None,
    label_col: str = "attack_type",
) -> tuple[dict[str, Any], pd.DataFrame]:
    artifact_path = Path(path)
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    from helix_ids.governance import verify_ingress_artifact

    _validate_artifact_keys(artifact)
    _deployment_manifest_candidate = artifact_path.parent / "deployment.manifest.json"
    _deployment_manifest = _deployment_manifest_candidate if _deployment_manifest_candidate.exists() else None
    verify_ingress_artifact(
        artifact_path,
        kind="checkpoint",
        contract=artifact,
        embedded_manifest=artifact.get("artifact_manifest"),
        deployment_manifest=_deployment_manifest,
        sidecars={
            "contract": artifact_path.with_suffix(artifact_path.suffix + ".contract.json"),
            "feature_order": artifact_path.with_suffix(artifact_path.suffix + ".feature_order.json"),
            "schema_hash": artifact_path.with_suffix(artifact_path.suffix + ".schema_hash.txt"),
        },
    )
    artifact_contract_version = str(artifact["contract_version"])
    if artifact_contract_version != CONTRACT_VERSION:
        raise AssertionError(
            f"Artifact contract version mismatch: expected {CONTRACT_VERSION}, got {artifact_contract_version}"
        )
    assert_runtime_contract(
        schema_version=str(artifact["schema_version"]),
        schema_hash=str(artifact["schema_hash"]),
        feature_order=[str(col) for col in artifact["feature_order"]],
        input_dim=int(artifact["input_dim"]),
        binary_output_dim=int(artifact["binary_output_dim"]),
        family_output_dim=int(artifact["family_output_dim"]),
        context="artifact contract",
    )

    expected_order = [str(col) for col in artifact["feature_order"]]
    expected_hash = str(artifact["schema_hash"])

    if mapping is None:
        features = enforce_feature_order(df, expected_order, context="artifact feature order")
    else:
        harmonized = harmonize_features(df, mapping, label_col=label_col, mode="strict")
        features = enforce_feature_order(harmonized, expected_order, context=f"artifact/{mapping.dataset_name}")

    features = features.astype(np.float32, copy=False)
    current_hash = compute_schema_hash(feature_order=expected_order)
    if current_hash != expected_hash:
        raise SchemaDriftError(
            expected_hash,
            current_hash,
            context="artifact schema hash",
            missing=list(set(expected_order) - set(features.columns)),
            extra=list(set(features.columns) - set(expected_order)),
        )
    return dict(artifact["model"]), features


def create_nslkdd_mapping() -> FeatureMapping:
    return FeatureMapping(
        dataset_name="nsl_kdd",
        original_features=[],
        common_features=COMMON_FEATURES,
        feature_mapping={
            "duration": ["duration"],
            "protocol_type": ["protocol_type"],
            "protocol": ["protocol_type"],
            "src_bytes": ["src_bytes"],
            "dst_bytes": ["dst_bytes"],
            "flag": ["flag"],
            "state": ["flag"],
            "wrong_fragment": ["wrong_fragment"],
            "urgent": ["urgent"],
            "count": ["count"],
            "srv_count": ["srv_count"],
            "serror_rate": ["serror_rate"],
            "srv_serror_rate": ["srv_serror_rate"],
            "rerror_rate": ["rerror_rate"],
            "srv_rerror_rate": ["srv_rerror_rate"],
            "same_srv_rate": ["same_srv_rate"],
            "diff_srv_rate": ["diff_srv_rate"],
            "dst_host_count": ["dst_host_count"],
            "dst_host_srv_count": ["dst_host_srv_count"],
            "dst_host_same_srv_rate": ["dst_host_same_srv_rate"],
            "dst_host_diff_srv_rate": ["dst_host_diff_srv_rate"],
            "connection_state": ["connection_state"],
            "traffic_direction": ["traffic_direction"],
            "has_rst": ["has_rst"],
            "service_tier": ["service_tier"],
        },
    )


def create_unsw_mapping() -> FeatureMapping:
    return FeatureMapping(
        dataset_name="unsw_nb15",
        original_features=[],
        common_features=COMMON_FEATURES,
        feature_mapping={
            "duration": ["dur", "duration"],
            "protocol_type": ["proto", "protocol_type"],
            "protocol": ["proto", "protocol_type"],
            "src_bytes": ["sbytes", "src_bytes"],
            "dst_bytes": ["dbytes", "dst_bytes"],
            "flag": ["state", "flag"],
            "state": ["state", "flag"],
            "wrong_fragment": ["wrong_fragment"],
            "urgent": ["urgent"],
            "count": ["count"],
            "srv_count": ["srv_count"],
            "serror_rate": ["serror_rate"],
            "srv_serror_rate": ["srv_serror_rate"],
            "rerror_rate": ["rerror_rate"],
            "srv_rerror_rate": ["srv_rerror_rate"],
            "same_srv_rate": ["same_srv_rate"],
            "diff_srv_rate": ["diff_srv_rate"],
            "dst_host_count": ["dst_host_count"],
            "dst_host_srv_count": ["dst_host_srv_count"],
            "dst_host_same_srv_rate": ["dst_host_same_srv_rate"],
            "dst_host_diff_srv_rate": ["dst_host_diff_srv_rate"],
            "connection_state": ["connection_state"],
            "traffic_direction": ["traffic_direction"],
            "has_rst": ["has_rst"],
            "service_tier": ["service_tier"],
        },
    )


def create_cicids_mapping() -> FeatureMapping:
    return FeatureMapping(
        dataset_name="cicids",
        original_features=[],
        common_features=COMMON_FEATURES,
        feature_mapping={
            "duration": ["Flow Duration", "duration"],
            "protocol_type": ["Protocol", "protocol_type"],
            "protocol": ["Protocol", "protocol_type"],
            "src_bytes": [CICIDS_FWD_BYTES_COL, "Total Length of Fwd Packets", "src_bytes"],
            "dst_bytes": [CICIDS_BWD_BYTES_COL, "Total Length of Bwd Packets", "dst_bytes"],
            "flag": ["flag"],
            "state": ["flag"],
            "syn_count": ["SYN Flag Cnt", "syn_count"],
            "rst_count": [CICIDS_RST_FLAG_COL, "rst_count"],
            "wrong_fragment": ["wrong_fragment"],
            "urgent": ["urgent"],
            "count": ["count"],
            "srv_count": ["srv_count"],
            "serror_rate": ["serror_rate"],
            "srv_serror_rate": ["srv_serror_rate"],
            "rerror_rate": ["rerror_rate"],
            "srv_rerror_rate": ["srv_rerror_rate"],
            "same_srv_rate": ["same_srv_rate"],
            "diff_srv_rate": ["diff_srv_rate"],
            "dst_host_count": ["dst_host_count"],
            "dst_host_srv_count": ["dst_host_srv_count"],
            "dst_host_same_srv_rate": ["dst_host_same_srv_rate"],
            "dst_host_diff_srv_rate": ["dst_host_diff_srv_rate"],
            "connection_state": ["connection_state"],
            "traffic_direction": ["traffic_direction"],
            "has_rst": ["has_rst"],
            "service_tier": ["service_tier"],
        },
    )


# ============================================================================
# Invariant Feature Extraction
# ============================================================================


def _find_column(
    df: pd.DataFrame,
    normalized_columns: Mapping[str, str],
    candidates: Sequence[str],
) -> str | None:
    _ = df
    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in normalized_columns:
            return normalized_columns[key]
    return None


def _require_numeric(
    df: pd.DataFrame,
    normalized_columns: Mapping[str, str],
    candidates: Sequence[str],
    *,
    dataset_name: str,
    field_name: str,
) -> pd.Series:
    col = _find_column(df, normalized_columns, candidates)
    if col is None:
        raise ValueError(
            f"{dataset_name}: required raw column missing for '{field_name}' "
            f"(aliases={list(candidates)})"
        )
    return pd.to_numeric(df[col], errors="coerce")


def _optional_numeric(
    df: pd.DataFrame,
    normalized_columns: Mapping[str, str],
    candidates: Sequence[str],
) -> pd.Series | None:
    col = _find_column(df, normalized_columns, candidates)
    if col is None:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def _require_text(
    df: pd.DataFrame,
    normalized_columns: Mapping[str, str],
    candidates: Sequence[str],
    *,
    dataset_name: str,
    field_name: str,
) -> pd.Series:
    col = _find_column(df, normalized_columns, candidates)
    if col is None:
        raise ValueError(
            f"{dataset_name}: required raw column missing for '{field_name}' "
            f"(aliases={list(candidates)})"
        )
    return df[col].astype(str).str.strip()


def _protocol_one_hot(protocol_raw: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    numeric = pd.to_numeric(protocol_raw, errors="coerce")
    text = protocol_raw.astype(str).str.lower().str.strip()

    is_tcp = ((numeric == 6) | text.str.contains("tcp", regex=False)).astype(np.float32)
    is_udp = ((numeric == 17) | text.str.contains("udp", regex=False)).astype(np.float32)
    is_icmp = (
        (numeric == 1)
        | text.str.contains("icmp", regex=False)
        | text.str.contains("igmp", regex=False)
    ).astype(np.float32)
    is_other = (1.0 - np.clip(is_tcp + is_udp + is_icmp, 0.0, 1.0)).astype(np.float32)

    return is_tcp, is_udp, is_icmp, is_other


def _state_indicators(
    *,
    dataset_name: str,
    state_raw: pd.Series | None,
    syn_count: pd.Series | None,
    rst_count: pd.Series | None,
    total_bytes: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    if dataset_name == "cicids":
        if syn_count is None or rst_count is None:
            raise ValueError(
                "cicids: required raw columns missing for state indicators "
                "(SYN Flag Cnt and RST Flag Cnt)"
            )
        syn = syn_count.fillna(0.0)
        rst = rst_count.fillna(0.0)
        err = ((rst > 0.0) | ((syn <= 0.0) & (total_bytes > 0.0))).astype(np.float32)
        reset_retrans = ((rst > 0.0) & (syn > 0.0)).astype(np.float32)
        return err, reset_retrans

    if state_raw is None:
        raise ValueError(f"{dataset_name}: state column is required for non-CICIDS datasets")

    state = state_raw.astype(str).str.lower().str.strip()
    err = state.str.contains(r"rej|rst|s0|s1|s2|s3|sh|err|fail|int|req", regex=True).astype(np.float32)
    reset_retrans = state.str.contains(r"rst|rej|ret|rtr", regex=True).astype(np.float32)
    return err, reset_retrans


def _optional_average(
    df: pd.DataFrame,
    normalized_columns: Mapping[str, str],
    candidates: Sequence[str],
) -> pd.Series | None:
    series_list: list[pd.Series] = []
    for candidate in candidates:
        col = _find_column(df, normalized_columns, [candidate])
        if col is not None:
            series_list.append(pd.to_numeric(df[col], errors="coerce"))
    if not series_list:
        return None
    stacked = pd.concat(series_list, axis=1)
    return stacked.mean(axis=1)


def _series_or_default(series: pd.Series | None, index: pd.Index, default: float = 0.0) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=np.float64)
    return pd.to_numeric(series, errors="coerce").fillna(default).astype(np.float64)


def _nonnegative_proxy(series: pd.Series | None, index: pd.Index, default: float = 0.0) -> pd.Series:
    values = _series_or_default(series, index=index, default=default)
    # Some preprocessed exports contain z-scored count-like fields; abs keeps magnitude signal.
    if (values < 0).any():
        values = values.abs()
    return values.clip(lower=0.0)


def _port_rarity(port_like: pd.Series) -> pd.Series:
    normalized = pd.to_numeric(port_like, errors="coerce").fillna(-1).astype(np.int64)
    frequency = normalized.value_counts(normalize=True)
    rarity = 1.0 - normalized.map(frequency).fillna(0.0)
    return rarity.astype(np.float64)


def _encode_protocol(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip().str.lower()
    encoded = pd.Series(3, index=series.index, dtype=np.int64)

    non_na_numeric = numeric.dropna()
    if not non_na_numeric.empty and non_na_numeric.isin([0, 1, 2]).all():
        encoded[numeric == 0] = PROTOCOL_MAP["tcp"]
        encoded[numeric == 1] = PROTOCOL_MAP["udp"]
        encoded[numeric == 2] = PROTOCOL_MAP["icmp"]
        return encoded

    encoded[(numeric == 6) | (text == "tcp") | (text == "0")] = PROTOCOL_MAP["tcp"]
    encoded[(numeric == 17) | (text == "udp")] = PROTOCOL_MAP["udp"]
    encoded[(numeric == 1) | (text == "icmp") | (text == "2")] = PROTOCOL_MAP["icmp"]
    return encoded


def _encode_flag(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map(FLAG_MAP)
    unresolved_mask = mapped.isna()
    if unresolved_mask.any():
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_mask = unresolved_mask & numeric.notna()
        if numeric_mask.any():
            as_int = numeric.loc[numeric_mask].astype(np.int64)
            in_range = as_int.between(0, len(FLAG_MAP) - 1)
            mapped.loc[as_int.index[in_range]] = as_int.loc[in_range]

    unknown = sorted(set(text[mapped.isna()].unique().tolist()))
    if unknown:
        raise AssertionError(
            "Unknown flag/state values encountered: "
            f"{unknown[:10]}{'...' if len(unknown) > 10 else ''}"
        )
    return mapped.astype(np.int64)


def _encode_connection_state(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map(CONNECTION_STATE_MAP)
    unknown = sorted(set(text[mapped.isna()].unique().tolist()))
    if unknown:
        raise AssertionError(
            "Unknown connection_state values encountered: "
            f"{unknown[:10]}{'...' if len(unknown) > 10 else ''}"
        )
    return mapped.astype(np.int64)


def _encode_traffic_direction(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map(TRAFFIC_DIRECTION_MAP)
    unknown = sorted(set(text[mapped.isna()].unique().tolist()))
    if unknown:
        raise AssertionError(
            "Unknown traffic_direction values encountered: "
            f"{unknown[:10]}{'...' if len(unknown) > 10 else ''}"
        )
    return mapped.astype(np.int64)


def _encode_service_tier(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map(SERVICE_TIER_MAP)
    unknown = sorted(set(text[mapped.isna()].unique().tolist()))
    if unknown:
        raise AssertionError(
            "Unknown service_tier values encountered: "
            f"{unknown[:10]}{'...' if len(unknown) > 10 else ''}"
        )
    return mapped.astype(np.int64)


def _derive_service_tier(df: pd.DataFrame, dataset_name: str) -> pd.Series:
    normalized_columns = {normalize_column_name(col): col for col in df.columns}

    if dataset_name in {"nsl_kdd", "unsw_nb15"}:
        service_col = _find_column(df, normalized_columns, ["service"])
        if service_col is None:
            raise AssertionError(f"{dataset_name} missing service required for service_tier")

        service = df[service_col].astype(str).str.strip().str.lower()
        tier = pd.Series("other", index=df.index, dtype="object")
        tier[service.isin({"-", "", "none"})] = "none"
        tier[service.str.contains(r"http|https|www", regex=True)] = "web"
        tier[service.str.contains(r"^domain$|dns", regex=True)] = "dns"
        tier[service.str.contains(r"ftp", regex=True)] = "ftp"
        tier[service.str.contains(r"smtp|pop3|imap|mail", regex=True)] = "mail"
        tier[service.str.contains(r"ssh|telnet|auth", regex=True)] = "auth"
        return tier

    if dataset_name == "cicids":
        dst_port_col = _find_column(df, normalized_columns, ["Dst Port", "Destination Port", "dst_port"])
        if dst_port_col is None:
            return pd.Series("other", index=df.index, dtype="object")

        dst_port = pd.to_numeric(df[dst_port_col], errors="raise").fillna(-1).astype(np.int64)
        tier = pd.Series("other", index=df.index, dtype="object")
        tier[dst_port.isin({80, 443, 8080, 8443})] = "web"
        tier[dst_port.isin({53})] = "dns"
        tier[dst_port.isin({20, 21})] = "ftp"
        tier[dst_port.isin({25, 110, 143, 465, 587, 993, 995})] = "mail"
        tier[dst_port.isin({22, 23})] = "auth"
        return tier

    raise AssertionError(f"Unsupported dataset for service_tier derivation: {dataset_name}")


def _derive_traffic_direction(df: pd.DataFrame, dataset_name: str) -> pd.Series:
    normalized_columns = {normalize_column_name(col): col for col in df.columns}

    if dataset_name == "nsl_kdd":
        src = _require_numeric(
            df,
            normalized_columns,
            ["src_bytes"],
            dataset_name=dataset_name,
            field_name="src_bytes",
        )
        dst = _require_numeric(
            df,
            normalized_columns,
            ["dst_bytes"],
            dataset_name=dataset_name,
            field_name="dst_bytes",
        )
    elif dataset_name == "unsw_nb15":
        src = _require_numeric(
            df,
            normalized_columns,
            ["sbytes", "src_bytes"],
            dataset_name=dataset_name,
            field_name="sbytes/src_bytes",
        )
        dst = _require_numeric(
            df,
            normalized_columns,
            ["dbytes", "dst_bytes"],
            dataset_name=dataset_name,
            field_name="dbytes/dst_bytes",
        )
    elif dataset_name == "cicids":
        src = _require_numeric(
            df,
            normalized_columns,
            [CICIDS_FWD_BYTES_COL, "Total Length of Fwd Packets", "src_bytes"],
            dataset_name=dataset_name,
            field_name=CICIDS_FWD_BYTES_COL,
        )
        dst = _require_numeric(
            df,
            normalized_columns,
            [CICIDS_BWD_BYTES_COL, "Total Length of Bwd Packets", "dst_bytes"],
            dataset_name=dataset_name,
            field_name=CICIDS_BWD_BYTES_COL,
        )
    else:
        raise AssertionError(f"Unsupported dataset for traffic_direction derivation: {dataset_name}")

    src = pd.to_numeric(src, errors="raise").fillna(0.0).astype(np.float64).clip(lower=0.0)
    dst = pd.to_numeric(dst, errors="raise").fillna(0.0).astype(np.float64).clip(lower=0.0)

    direction = pd.Series("balanced", index=df.index, dtype="object")
    direction[src > (dst * 1.5)] = "outbound"
    direction[dst > (src * 1.5)] = "inbound"
    return direction


def _derive_has_rst(df: pd.DataFrame, dataset_name: str) -> pd.Series:
    normalized_columns = {normalize_column_name(col): col for col in df.columns}

    if dataset_name in {"nsl_kdd", "unsw_nb15"}:
        raw_col = _find_column(df, normalized_columns, ["flag", "state"])
        if raw_col is None:
            raise AssertionError(f"{dataset_name} missing flag/state required for has_rst")
        text = df[raw_col].astype(str).str.strip().str.lower()
        return text.str.contains(r"rst|rej", regex=True).astype(np.int64)

    if dataset_name == "cicids":
        rst_col = _find_column(df, normalized_columns, [CICIDS_RST_FLAG_COL, "rst_count"])
        if rst_col is None:
            raise AssertionError("cicids missing RST Flag Cnt required for has_rst")
        rst = pd.to_numeric(df[rst_col], errors="raise").fillna(0.0).astype(np.float64)
        return (rst > 0).astype(np.int64)

    raise AssertionError(f"Unsupported dataset for has_rst derivation: {dataset_name}")


def _map_unsw_connection_state(raw_state: pd.Series) -> pd.Series:
    state = raw_state.map(UNSW_STATE_MAP).fillna("OTH")

    # Rare ambiguous UNSW states inject label noise; collapse them to OTH.
    state_freq = raw_state.value_counts(normalize=True, dropna=False)
    rare_ambiguous = raw_state.isin(UNSW_AMBIGUOUS_STATES) & (
        raw_state.map(state_freq).fillna(0.0) < UNSW_RARE_STATE_FREQ_THRESHOLD
    )
    state.loc[rare_ambiguous] = "OTH"

    # Keep canonical UNSW states stable regardless of corpus frequency.
    for raw_token, canonical in UNSW_FROZEN_STATE_MAP.items():
        state.loc[raw_state == raw_token] = canonical

    return state


def _derive_connection_state(df: pd.DataFrame, dataset_name: str) -> pd.Series:
    normalized_columns = {normalize_column_name(col): col for col in df.columns}

    if dataset_name == "nsl_kdd":
        raw_col = _find_column(df, normalized_columns, ["flag", "state"])
        if raw_col is None:
            raise AssertionError("nsl_kdd missing flag/state required for connection_state")
        state = (
            df[raw_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(NSL_STATE_MAP)
            .fillna("OTH")
        )
        return state

    if dataset_name == "unsw_nb15":
        raw_col = _find_column(df, normalized_columns, ["state", "flag"])
        if raw_col is None:
            raise AssertionError("unsw_nb15 missing state/flag required for connection_state")
        raw_state = df[raw_col].astype(str).str.strip().str.lower()
        return _map_unsw_connection_state(raw_state)

    if dataset_name == "cicids":
        syn_col = _find_column(df, normalized_columns, ["SYN Flag Cnt", "syn_count"])
        rst_col = _find_column(df, normalized_columns, [CICIDS_RST_FLAG_COL, "rst_count"])
        ack_col = _find_column(df, normalized_columns, ["ACK Flag Cnt", "ack_count"])
        fin_col = _find_column(df, normalized_columns, ["FIN Flag Cnt", "fin_count"])

        if syn_col is None or rst_col is None:
            raise AssertionError("cicids missing SYN/RST flag columns required for connection_state")

        syn = pd.to_numeric(df[syn_col], errors="raise").astype(np.float64)
        rst = pd.to_numeric(df[rst_col], errors="raise").astype(np.float64)
        ack = (
            pd.to_numeric(df[ack_col], errors="raise").astype(np.float64)
            if ack_col is not None
            else pd.Series(0.0, index=df.index)
        )
        fin = (
            pd.to_numeric(df[fin_col], errors="raise").astype(np.float64)
            if fin_col is not None
            else pd.Series(0.0, index=df.index)
        )

        state = pd.Series("OTH", index=df.index, dtype="object")
        state[(syn > 0) & (ack <= 0)] = "S0"
        state[(syn > 0) & (ack > 0)] = "S1"
        state[fin > 0] = "FIN"
        state[rst > 0] = "RST"
        return state

    raise AssertionError(f"Unsupported dataset for connection_state derivation: {dataset_name}")


def _default_common_feature_series(feature: str, index: pd.Index) -> pd.Series:
    if feature == "flag":
        return pd.Series("SF", index=index, dtype="object")
    if feature == "protocol_type":
        return pd.Series("tcp", index=index, dtype="object")
    if feature == "traffic_direction":
        return pd.Series("balanced", index=index, dtype="object")
    if feature == "has_rst":
        return pd.Series(0, index=index, dtype=np.int64)
    if feature == "service_tier":
        return pd.Series("other", index=index, dtype="object")
    return pd.Series(0.0, index=index, dtype=np.float64)


def select_common_features(df: pd.DataFrame, dataset_name: str, aliases: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """Select only explicit semantically aligned features in deterministic order."""
    normalized_columns = {normalize_column_name(col): col for col in df.columns}
    selected: dict[str, pd.Series] = {}
    missing: list[str] = []

    for feature in COMMON_FEATURES:
        candidates = aliases.get(feature, [feature])
        col = _find_column(df, normalized_columns, candidates)
        if col is None:
            missing.append(feature)
            continue
        selected[feature] = df[col]

    if missing:
        raise AssertionError(f"{dataset_name} is missing required canonical features: {missing}")
    selected_df = pd.DataFrame(selected, index=df.index)
    return selected_df


def _augment_geometry_expansion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Materialize canonical interaction features deterministically."""
    out = df.copy()

    def _numeric_series(column_name: str) -> pd.Series:
        if column_name in out.columns:
            series = pd.to_numeric(out[column_name], errors="coerce")
        else:
            series = pd.Series(np.zeros(len(out), dtype=np.float32), index=out.index)
        return series.fillna(0.0)

    duration_signal = np.abs(_numeric_series("duration"))
    protocol_signal = _numeric_series("protocol_type")
    connection_signal = _numeric_series("connection_state")
    traffic_signal = _numeric_series("traffic_direction")
    service_signal = _numeric_series("service_tier")

    src_bytes = np.abs(_numeric_series("src_bytes"))
    dst_bytes = np.abs(_numeric_series("dst_bytes"))
    out["log_src_bytes"] = np.log1p(src_bytes)
    out["log_dst_bytes"] = np.log1p(dst_bytes)
    out["src_dst_bytes_ratio"] = src_bytes / (dst_bytes + 1.0)
    out["dst_src_bytes_ratio"] = dst_bytes / (src_bytes + 1.0)

    if "count" in out.columns:
        count_vals = np.abs(_numeric_series("count"))
    else:
        count_vals = duration_signal + src_bytes / (dst_bytes + 1.0)

    if "srv_count" in out.columns:
        srv_count_vals = np.abs(_numeric_series("srv_count"))
    else:
        srv_count_vals = (service_signal + 1.0) * (traffic_signal + 1.0)
    out["count_x_srv_count"] = count_vals * srv_count_vals

    same_host_rate_series = None
    for candidate in ("same_host_rate", "same_srv_rate", "dst_host_same_srv_rate"):
        if candidate in out.columns:
            same_host_rate_series = np.abs(_numeric_series(candidate))
            break
    if same_host_rate_series is None:
        same_host_rate_series = src_bytes / (src_bytes + dst_bytes + 1.0)

    out["same_host_rate_x_service"] = same_host_rate_series * (service_signal + 1.0)

    if "diff_srv_rate" in out.columns:
        diff_srv_rate_series = np.abs(_numeric_series("diff_srv_rate"))
    else:
        diff_srv_rate_series = np.abs(src_bytes - dst_bytes) / (src_bytes + dst_bytes + 1.0)

    if "flag" in out.columns:
        flag_signal = _numeric_series("flag")
    else:
        out["flag"] = _default_common_feature_series("flag", out.index)
        flag_signal = _numeric_series("has_rst") + connection_signal
    out["diff_srv_rate_x_flag"] = diff_srv_rate_series * (flag_signal + 1.0)

    out["protocol_service_flag"] = (
        (protocol_signal + 1.0) * (service_signal + 1.0) * (flag_signal + 1.0)
    )

    return out


def harmonize_features(  # NOSONAR
    df: pd.DataFrame,
    mapping: FeatureMapping,
    label_col: str = "attack_type",
    mode: str = "strict",
) -> pd.DataFrame:
    """Strictly select semantically matched common features in fixed order."""
    if mode not in PIPELINE_MODES:
        raise ValueError(f"Unsupported pipeline mode: {mode}")
    if mode == "lenient" and os.environ.get("HELIX_DEBUG_LENIENT", "0") != "1":
        raise AssertionError(
            "Lenient mode is debug-only. Set HELIX_DEBUG_LENIENT=1 to enable it explicitly."
        )
    working_df = df.copy()
    if mode == "strict":
        validate_no_nan_inf(working_df)
    else:
        working_df = sanitize_numeric(working_df)

    # Normalized column map used for label lookup and optional checks.
    normalized_columns = {normalize_column_name(col): col for col in working_df.columns}

    working_df["connection_state"] = _derive_connection_state(working_df, mapping.dataset_name)
    working_df["traffic_direction"] = _derive_traffic_direction(working_df, mapping.dataset_name)
    working_df["has_rst"] = _derive_has_rst(working_df, mapping.dataset_name)
    working_df["service_tier"] = _derive_service_tier(working_df, mapping.dataset_name)
    working_df = _augment_geometry_expansion_features(working_df)

    harmonized = select_common_features(working_df, mapping.dataset_name, mapping.feature_mapping)
    validate_feature_order(harmonized.columns, context=f"{mapping.dataset_name} harmonized features")

    if "protocol_type" in harmonized.columns:
        harmonized["protocol_type"] = _encode_protocol(harmonized["protocol_type"])
    if "flag" in harmonized.columns:
        harmonized["flag"] = _encode_flag(harmonized["flag"])
    if "connection_state" in harmonized.columns:
        harmonized["connection_state"] = _encode_connection_state(harmonized["connection_state"])
    if "traffic_direction" in harmonized.columns:
        harmonized["traffic_direction"] = _encode_traffic_direction(harmonized["traffic_direction"])
    if "service_tier" in harmonized.columns:
        harmonized["service_tier"] = _encode_service_tier(harmonized["service_tier"])

    for feature in FEATURE_ORDER:
        if feature in {"protocol_type", "flag", "connection_state", "traffic_direction", "has_rst", "service_tier"}:
            continue
        harmonized[feature] = coerce_numeric_strict(harmonized[feature]).astype(np.float64)

    for feature, feature_type in SCHEMA.items():
        if feature not in harmonized.columns:
            continue
        harmonized[feature] = coerce_numeric_strict(harmonized[feature]).astype(feature_type, errors="raise")

    for feature in FEATURE_ORDER:
        if feature not in {"protocol_type", "flag", "connection_state", "traffic_direction", "has_rst", "service_tier"}:
            harmonized[feature] = harmonized[feature].clip(lower=-1e9, upper=1e9)

    harmonized = enforce_feature_order(
        harmonized[FEATURE_ORDER],
        FEATURE_ORDER,
        context=f"{mapping.dataset_name} harmonized features",
    )

    values = harmonized[FEATURE_ORDER].to_numpy(dtype=np.float64, copy=False)
    validate_no_nan_inf(harmonized[FEATURE_ORDER])
    if np.abs(values).max() >= 1e6:
        # Keep as a diagnostic signal only; per-dataset pipelines can legitimately
        # contain high-magnitude raw counters without cross-dataset normalization.
        pass

    validate_mapping(mapping.to_dict())

    # Use canonical compute_schema_hash (compat wrapper) to produce the
    # canonical hash for the repository-feature-order.
    schema_hash = compute_schema_hash()

    normalized_label_col = normalize_column_name(label_col)
    if normalized_label_col in normalized_columns:
        harmonized["label"] = working_df[normalized_columns[normalized_label_col]]
    else:
        for alias in ("label", "__label__", "class", "attack_type", "attack cat", "attack_cat"):
            normalized_alias = normalize_column_name(alias)
            if normalized_alias in normalized_columns:
                harmonized["label"] = working_df[normalized_columns[normalized_alias]]
                break

    harmonized.attrs.update(
        {
            "source": mapping.dataset_name.upper(),
            "schema_version": SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "schema_hash": schema_hash,
            "feature_order": list(FEATURE_ORDER),
            "pipeline_mode": mode,
        }
    )

    return harmonized


def save_feature_mappings(output_dir: Path):
    """Save raw-contract mappings and invariant feature metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mappings = {
        "nsl_kdd": create_nslkdd_mapping().to_dict(),
        "unsw_nb15": create_unsw_mapping().to_dict(),
        "cicids": create_cicids_mapping().to_dict(),
        "common_features": COMMON_FEATURES,
        "attack_taxonomy_7class": ATTACK_TAXONOMY_7CLASS,
    }

    output_file = output_dir / "feature_mappings_multi_dataset.json"
    with open(output_file, "w") as f:
        json.dump(mappings, f, indent=2)

    print(f"Saved feature mappings to {output_file}")


def family_label_to_binary(family_label: int) -> int:
    """Convert 7-class family label to binary (Normal vs Attack)."""
    return 0 if family_label == 0 else 1


def labels_to_multi_task(family_labels):
    """Return binary and family labels for multi-task training."""
    family_labels = np.asarray(family_labels)
    binary_labels = np.where(family_labels == 0, 0, 1)
    return binary_labels, family_labels


if __name__ == "__main__":
    print(f"Common features ({len(COMMON_FEATURES)}): {COMMON_FEATURES}")
    print(f"Attack taxonomy 7-class: {ATTACK_TAXONOMY_7CLASS}")
