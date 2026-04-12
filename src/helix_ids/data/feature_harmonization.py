"""Invariant feature harmonization for multi-dataset IDS training.

This module intentionally defines a compact, behavior-level representation
shared by NSL-KDD, UNSW-NB15, and CICIDS. It avoids dataset-specific
synthetic fills and removes non-transferable proxy features.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

# ============================================================================
# Common Invariant Feature Space (15 features)
# ============================================================================

COMMON_FEATURES_METADATA = {
    "duration_log": {"type": "numeric", "description": "log1p(duration_seconds)"},
    "total_bytes_log": {"type": "numeric", "description": "log1p(src_bytes + dst_bytes)"},
    "bytes_forward_ratio": {"type": "numeric", "description": "src_bytes / (src_bytes + dst_bytes + eps)"},
    "bytes_asymmetry": {"type": "numeric", "description": "(src_bytes - dst_bytes) / (src_bytes + dst_bytes + eps)"},
    "byte_direction_ratio": {
        "type": "numeric",
        "description": "tanh(log1p(src_bytes) - log1p(dst_bytes))",
    },
    "proto_tcp": {"type": "binary", "description": "protocol family indicator"},
    "proto_udp": {"type": "binary", "description": "protocol family indicator"},
    "proto_icmp": {"type": "binary", "description": "protocol family indicator"},
    "proto_other": {"type": "binary", "description": "protocol family indicator"},
    "state_error_indicator": {"type": "binary", "description": "connection/state error indicator"},
    "state_reset_retrans_indicator": {"type": "binary", "description": "reset/retrans indicator"},
    "rst_fraction": {
        "type": "numeric",
        "description": "fraction of reset-like outcomes over handshake attempts",
    },
    "handshake_completion_rate": {
        "type": "numeric",
        "description": "proxy for successful connection establishment",
    },
    "iat_coefficient_of_variation": {
        "type": "numeric",
        "description": "inter-arrival-time variability proxy",
    },
    "unique_dst_ports_per_window": {
        "type": "numeric",
        "description": "destination-port interaction diversity proxy",
    },
}

COMMON_FEATURES = list(COMMON_FEATURES_METADATA.keys())
assert len(COMMON_FEATURES) == 15, f"Expected 15 common features, got {len(COMMON_FEATURES)}"

LEAKAGE_PRONE_FEATURES: set[str] = set()
INVARIANT_FEATURES = COMMON_FEATURES.copy()


def normalize_column_name(name: str) -> str:
    """Normalize a feature name for resilient matching across datasets."""
    return str(name).strip().lower().replace("_", " ")


# ============================================================================
# Attack Taxonomy: 7-class mapping
# ============================================================================

ATTACK_TAXONOMY_7CLASS = {
    0: "Normal",
    1: "DoS",
    2: "Probe",
    3: "R2L",
    4: "U2R",
    5: "Generic",
    6: "Backdoor",
}

NSLKDD_TO_7CLASS = {
    "Normal": 0,
    "DoS": 1,
    "Probe": 2,
    "R2L": 3,
    "U2R": 4,
}

UNSW_TO_7CLASS = {
    "Normal": 0,
    "Analysis": 2,
    "Backdoors": 6,
    "DoS": 1,
    "Exploits": 3,
    "Fuzzers": 2,
    "Generic": 5,
    "Reconnaissance": 2,
    "Shellcode": 4,
    "Worms": 6,
}

CICIDS_TO_7CLASS = {
    "BENIGN": 0,
    "DDoS": 1,
    "DoS GoldenEye": 1,
    "DoS Hulk": 1,
    "DoS slowloris": 1,
    "DoS Slowhttptest": 1,
    "PortScan": 2,
    "Bot": 6,
    "Infiltration": 6,
    "Web Attack - Brute Force": 3,
    "Web Attack - XSS": 3,
    "Web Attack - Sql Injection": 3,
    "FTP-Patator": 3,
    "SSH-Patator": 3,
    "Heartbleed": 4,
}

CICIDS2018_TO_7CLASS = {
    "BENIGN": 0,
    "DDoS": 1,
    "DoS": 1,
    "PortScan": 2,
    "Bot": 6,
    "Infiltration": 6,
    "Brute Force": 3,
    "SQL Injection": 3,
    "SSH-Patator": 3,
    "FTP-Patator": 3,
}


# ============================================================================
# Dataset Raw-Column Contracts
# ============================================================================


@dataclass
class FeatureMapping:
    """Raw column aliases required to construct the invariant feature space."""

    dataset_name: str
    original_features: List[str]
    common_features: List[str]
    feature_mapping: Mapping[str, Union[str, List[str]]]

    def to_dict(self):
        return asdict(self)


def create_nslkdd_mapping() -> FeatureMapping:
    return FeatureMapping(
        dataset_name="nsl_kdd",
        original_features=[],
        common_features=COMMON_FEATURES,
        feature_mapping={
            "duration": ["duration"],
            "src_bytes": ["src_bytes"],
            "dst_bytes": ["dst_bytes"],
            "protocol": ["protocol_type"],
            "service": ["service"],
            "state": ["flag"],
            "syn_count": [],
            "rst_count": [],
            "ack_count": [],
            "rerror_rate": ["rerror_rate", "srv_rerror_rate", "dst_host_rerror_rate"],
            "serror_rate": ["serror_rate", "srv_serror_rate", "dst_host_serror_rate"],
            "count": ["count"],
            "srv_count": ["srv_count"],
            "diff_srv_rate": ["diff_srv_rate"],
            "dst_port": [],
            "iat_mean": [],
            "iat_std": [],
            "iat_max": [],
            "iat_min": [],
        },
    )


def create_unsw_mapping() -> FeatureMapping:
    return FeatureMapping(
        dataset_name="unsw_nb15",
        original_features=[],
        common_features=COMMON_FEATURES,
        feature_mapping={
            "duration": ["dur"],
            "src_bytes": ["sbytes"],
            "dst_bytes": ["dbytes"],
            "protocol": ["proto"],
            "service": ["service"],
            "state": ["state"],
            "syn_count": [],
            "rst_count": [],
            "ack_count": [],
            "rerror_rate": [],
            "serror_rate": [],
            "count": ["ct_src_ltm"],
            "srv_count": ["ct_srv_src"],
            "diff_srv_rate": [],
            "dst_port": ["dsport"],
            "iat_mean": ["Sintpkt", "Dintpkt"],
            "iat_std": ["Sjit", "Djit"],
            "iat_max": [],
            "iat_min": [],
            "ct_src_ltm": ["ct_src_ltm"],
            "ct_src_dport_ltm": ["ct_src_dport_ltm"],
        },
    )


def create_cicids_mapping() -> FeatureMapping:
    return FeatureMapping(
        dataset_name="cicids",
        original_features=[],
        common_features=COMMON_FEATURES,
        feature_mapping={
            "duration": ["Flow Duration"],
            "src_bytes": ["TotLen Fwd Pkts", "Total Length of Fwd Packets"],
            "dst_bytes": ["TotLen Bwd Pkts", "Total Length of Bwd Packets"],
            "protocol": ["Protocol"],
            "service": [],
            "state": [],
            "syn_count": ["SYN Flag Cnt"],
            "rst_count": ["RST Flag Cnt"],
            "ack_count": ["ACK Flag Cnt"],
            "rerror_rate": [],
            "serror_rate": [],
            "count": ["Tot Fwd Pkts"],
            "srv_count": [],
            "diff_srv_rate": [],
            "dst_port": ["Dst Port"],
            "iat_mean": ["Flow IAT Mean", "Fwd IAT Mean", "Bwd IAT Mean"],
            "iat_std": [],
            "iat_max": ["Fwd IAT Max", "Bwd IAT Max"],
            "iat_min": ["Fwd IAT Min", "Bwd IAT Min"],
            "ct_src_ltm": [],
            "ct_src_dport_ltm": [],
        },
    )


# ============================================================================
# Invariant Feature Extraction
# ============================================================================


def _find_column(
    df: pd.DataFrame,
    normalized_columns: Mapping[str, str],
    candidates: Sequence[str],
) -> Optional[str]:
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
) -> Optional[pd.Series]:
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
    state_raw: Optional[pd.Series],
    syn_count: Optional[pd.Series],
    rst_count: Optional[pd.Series],
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
) -> Optional[pd.Series]:
    series_list: list[pd.Series] = []
    for candidate in candidates:
        col = _find_column(df, normalized_columns, [candidate])
        if col is not None:
            series_list.append(pd.to_numeric(df[col], errors="coerce"))
    if not series_list:
        return None
    stacked = pd.concat(series_list, axis=1)
    return stacked.mean(axis=1)


def _series_or_default(series: Optional[pd.Series], index: pd.Index, default: float = 0.0) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=np.float64)
    return pd.to_numeric(series, errors="coerce").fillna(default).astype(np.float64)


def _port_rarity(port_like: pd.Series) -> pd.Series:
    normalized = pd.to_numeric(port_like, errors="coerce").fillna(-1).astype(np.int64)
    frequency = normalized.value_counts(normalize=True)
    rarity = 1.0 - normalized.map(frequency).fillna(0.0)
    return rarity.astype(np.float64)


def harmonize_features(
    df: pd.DataFrame,
    mapping: FeatureMapping,
    label_col: str = "attack_type",
) -> pd.DataFrame:
    """Construct invariant behavior-level features and return harmonized DataFrame."""
    normalized_columns = {normalize_column_name(col): col for col in df.columns}
    dataset_name = mapping.dataset_name

    duration = _require_numeric(
        df,
        normalized_columns,
        mapping.feature_mapping.get("duration", []),
        dataset_name=dataset_name,
        field_name="duration",
    )
    src_bytes = _require_numeric(
        df,
        normalized_columns,
        mapping.feature_mapping.get("src_bytes", []),
        dataset_name=dataset_name,
        field_name="src_bytes",
    )
    dst_bytes = _require_numeric(
        df,
        normalized_columns,
        mapping.feature_mapping.get("dst_bytes", []),
        dataset_name=dataset_name,
        field_name="dst_bytes",
    )
    protocol_raw = _require_text(
        df,
        normalized_columns,
        mapping.feature_mapping.get("protocol", []),
        dataset_name=dataset_name,
        field_name="protocol",
    )

    state_col = _find_column(df, normalized_columns, mapping.feature_mapping.get("state", []))
    state_raw = df[state_col].astype(str).str.strip() if state_col is not None else None
    syn_count = _optional_numeric(df, normalized_columns, mapping.feature_mapping.get("syn_count", []))
    rst_count = _optional_numeric(df, normalized_columns, mapping.feature_mapping.get("rst_count", []))
    ack_count = _optional_numeric(df, normalized_columns, mapping.feature_mapping.get("ack_count", []))
    rerror_rate = _optional_average(
        df,
        normalized_columns,
        mapping.feature_mapping.get("rerror_rate", []),
    )
    serror_rate = _optional_average(
        df,
        normalized_columns,
        mapping.feature_mapping.get("serror_rate", []),
    )
    count = _optional_average(df, normalized_columns, mapping.feature_mapping.get("count", []))
    srv_count = _optional_average(df, normalized_columns, mapping.feature_mapping.get("srv_count", []))
    diff_srv_rate = _optional_average(
        df,
        normalized_columns,
        mapping.feature_mapping.get("diff_srv_rate", []),
    )
    dst_port = _optional_numeric(df, normalized_columns, mapping.feature_mapping.get("dst_port", []))
    iat_mean = _optional_average(df, normalized_columns, mapping.feature_mapping.get("iat_mean", []))
    iat_std = _optional_average(df, normalized_columns, mapping.feature_mapping.get("iat_std", []))
    iat_max = _optional_average(df, normalized_columns, mapping.feature_mapping.get("iat_max", []))
    iat_min = _optional_average(df, normalized_columns, mapping.feature_mapping.get("iat_min", []))
    ct_src_ltm = _optional_average(df, normalized_columns, mapping.feature_mapping.get("ct_src_ltm", []))
    ct_src_dport_ltm = _optional_average(
        df,
        normalized_columns,
        mapping.feature_mapping.get("ct_src_dport_ltm", []),
    )
    service_col = _find_column(df, normalized_columns, mapping.feature_mapping.get("service", []))
    service_raw = df[service_col].astype(str).str.strip() if service_col is not None else None

    duration_num = duration.astype(np.float64)
    if dataset_name == "cicids":
        duration_num = duration_num / 1_000_000.0
    duration_num = duration_num.clip(lower=0.0)

    src = src_bytes.astype(np.float64).clip(lower=0.0)
    dst = dst_bytes.astype(np.float64).clip(lower=0.0)
    total_bytes = src + dst
    denom = total_bytes + 1e-6
    byte_direction_ratio = np.tanh(np.log1p(src) - np.log1p(dst))

    proto_tcp, proto_udp, proto_icmp, proto_other = _protocol_one_hot(protocol_raw)
    state_err, state_reset = _state_indicators(
        dataset_name=dataset_name,
        state_raw=state_raw,
        syn_count=syn_count,
        rst_count=rst_count,
        total_bytes=total_bytes,
    )

    rst_fraction = state_reset.astype(np.float64)
    handshake_completion_rate = (1.0 - state_err.astype(np.float64)).clip(0.0, 1.0)

    if dataset_name == "cicids":
        syn = _series_or_default(syn_count, df.index)
        rst = _series_or_default(rst_count, df.index)
        ack = _series_or_default(ack_count, df.index)
        rst_fraction = (rst / (syn + rst + 1e-6)).clip(0.0, 1.0)
        handshake_completion_rate = (ack / (syn + ack + rst + 1e-6)).clip(0.0, 1.0)
    elif dataset_name == "nsl_kdd":
        rst_rate = _series_or_default(rerror_rate, df.index, default=np.nan)
        syn_err_rate = _series_or_default(serror_rate, df.index, default=np.nan)
        rst_fraction = rst_rate.fillna(state_reset.astype(np.float64)).clip(0.0, 1.0)
        handshake_completion_rate = (1.0 - np.maximum(rst_rate, syn_err_rate)).fillna(
            1.0 - state_err.astype(np.float64)
        )
        handshake_completion_rate = handshake_completion_rate.clip(0.0, 1.0)

    if iat_mean is not None and iat_std is not None:
        iat_coefficient_of_variation = (iat_std.abs() / (iat_mean.abs() + 1e-6)).clip(0.0, 10.0)
    elif iat_mean is not None and iat_max is not None and iat_min is not None:
        iat_spread = (iat_max - iat_min).abs()
        iat_coefficient_of_variation = (iat_spread / (iat_mean.abs() + 1e-6)).clip(0.0, 10.0)
    else:
        diff_rate = _series_or_default(diff_srv_rate, df.index)
        c_all = _series_or_default(count, df.index)
        srv_all = _series_or_default(srv_count, df.index)
        variability_proxy = (c_all - srv_all).abs() / (c_all + srv_all + 1e-6)
        iat_coefficient_of_variation = (diff_rate + variability_proxy).clip(0.0, 2.0)

    if ct_src_ltm is not None and ct_src_dport_ltm is not None:
        unique_dst_ports_per_window = (ct_src_ltm / (ct_src_dport_ltm + 1e-6)).clip(0.0, 10.0)
    elif dst_port is not None:
        unique_dst_ports_per_window = _port_rarity(dst_port)
    elif service_raw is not None:
        service_frequency = service_raw.str.lower().value_counts(normalize=True)
        unique_dst_ports_per_window = (1.0 - service_raw.str.lower().map(service_frequency).fillna(0.0)).astype(
            np.float64
        )
    else:
        unique_dst_ports_per_window = pd.Series(0.0, index=df.index, dtype=np.float64)

    harmonized = pd.DataFrame(
        {
            "duration_log": np.log1p(duration_num),
            "total_bytes_log": np.log1p(total_bytes),
            "bytes_forward_ratio": (src / denom).clip(0.0, 1.0),
            "bytes_asymmetry": ((src - dst) / denom).clip(-1.0, 1.0),
            "byte_direction_ratio": byte_direction_ratio,
            "proto_tcp": proto_tcp,
            "proto_udp": proto_udp,
            "proto_icmp": proto_icmp,
            "proto_other": proto_other,
            "state_error_indicator": state_err,
            "state_reset_retrans_indicator": state_reset,
            "rst_fraction": rst_fraction,
            "handshake_completion_rate": handshake_completion_rate,
            "iat_coefficient_of_variation": iat_coefficient_of_variation,
            "unique_dst_ports_per_window": unique_dst_ports_per_window,
        },
        index=df.index,
    )

    normalized_label_col = normalize_column_name(label_col)
    if normalized_label_col in normalized_columns:
        harmonized["label"] = df[normalized_columns[normalized_label_col]]
    else:
        for alias in ("label", "__label__", "class", "attack_type", "attack cat", "attack_cat"):
            normalized_alias = normalize_column_name(alias)
            if normalized_alias in normalized_columns:
                harmonized["label"] = df[normalized_columns[normalized_alias]]
                break

    harmonized[COMMON_FEATURES] = harmonized[COMMON_FEATURES].replace([np.inf, -np.inf], np.nan)
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
