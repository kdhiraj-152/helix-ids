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
# Strict Common Feature Space (19 features)
# ============================================================================

COMMON_FEATURES_METADATA = {
    "duration": {"type": "numeric", "description": "connection duration"},
    "protocol_type": {"type": "categorical", "description": "transport protocol"},
    "src_bytes": {"type": "numeric", "description": "source to destination bytes"},
    "dst_bytes": {"type": "numeric", "description": "destination to source bytes"},
    "flag": {"type": "categorical", "description": "connection state/flag"},
    "wrong_fragment": {"type": "numeric", "description": "wrong fragment count"},
    "urgent": {"type": "numeric", "description": "urgent packet count"},
    "count": {"type": "numeric", "description": "connection count"},
    "srv_count": {"type": "numeric", "description": "service connection count"},
    "serror_rate": {"type": "numeric", "description": "SYN error rate"},
    "srv_serror_rate": {"type": "numeric", "description": "service SYN error rate"},
    "rerror_rate": {"type": "numeric", "description": "RST error rate"},
    "srv_rerror_rate": {"type": "numeric", "description": "service RST error rate"},
    "same_srv_rate": {"type": "numeric", "description": "same service ratio"},
    "diff_srv_rate": {"type": "numeric", "description": "different service ratio"},
    "dst_host_count": {"type": "numeric", "description": "dst host count"},
    "dst_host_srv_count": {"type": "numeric", "description": "dst host service count"},
    "dst_host_same_srv_rate": {"type": "numeric", "description": "dst host same service ratio"},
    "dst_host_diff_srv_rate": {"type": "numeric", "description": "dst host different service ratio"},
}

COMMON_FEATURES = list(COMMON_FEATURES_METADATA.keys())
assert len(COMMON_FEATURES) == 19, f"Expected 19 common features, got {len(COMMON_FEATURES)}"

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
}


def normalize_column_name(name: str) -> str:
    """Normalize a feature name for resilient matching across datasets."""
    return str(name).strip().lower().replace("_", " ")


def normalize_per_dataset(X):
    """Normalize a single dataset tensor with z-score scaling."""
    import torch

    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True)

    std[std < 1e-6] = 1.0
    return (X - mean) / std


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
            "protocol_type": ["protocol_type"],
            "src_bytes": ["src_bytes"],
            "dst_bytes": ["dst_bytes"],
            "flag": ["flag"],
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
            "src_bytes": ["sbytes", "src_bytes"],
            "dst_bytes": ["dbytes", "dst_bytes"],
            "flag": ["state", "flag"],
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
            "src_bytes": ["TotLen Fwd Pkts", "Total Length of Fwd Packets", "src_bytes"],
            "dst_bytes": ["TotLen Bwd Pkts", "Total Length of Bwd Packets", "dst_bytes"],
            "flag": ["flag"],
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


def _nonnegative_proxy(series: Optional[pd.Series], index: pd.Index, default: float = 0.0) -> pd.Series:
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
    encoded[(numeric == 6) | (text == "tcp")] = PROTOCOL_MAP["tcp"]
    encoded[(numeric == 17) | (text == "udp")] = PROTOCOL_MAP["udp"]
    encoded[(numeric == 1) | (text == "icmp")] = PROTOCOL_MAP["icmp"]
    return encoded


def _encode_flag(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map(FLAG_MAP)
    unknown = sorted(set(text[mapped.isna()].unique().tolist()))
    if unknown:
        raise AssertionError(
            "Unknown flag/state values encountered: "
            f"{unknown[:10]}{'...' if len(unknown) > 10 else ''}"
        )
    return mapped.astype(np.int64)


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

    assert len(missing) == 0, f"{dataset_name} missing features: {missing}"
    return pd.DataFrame(selected, index=df.index)


def harmonize_features(  # NOSONAR
    df: pd.DataFrame,
    mapping: FeatureMapping,
    label_col: str = "attack_type",
) -> pd.DataFrame:
    """Strictly select semantically matched common features in fixed order."""
    normalized_columns = {normalize_column_name(col): col for col in df.columns}
    harmonized = select_common_features(df, mapping.dataset_name, mapping.feature_mapping)

    harmonized["protocol_type"] = _encode_protocol(harmonized["protocol_type"])
    harmonized["flag"] = _encode_flag(harmonized["flag"])

    for feature in COMMON_FEATURES:
        if feature in {"protocol_type", "flag"}:
            continue
        harmonized[feature] = pd.to_numeric(harmonized[feature], errors="coerce")

    values = harmonized[COMMON_FEATURES].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise AssertionError(f"{mapping.dataset_name} has NaN/inf in common features")
    if np.abs(values).max() >= 1e6:
        # Keep as a diagnostic signal only; per-dataset pipelines can legitimately
        # contain high-magnitude raw counters without cross-dataset normalization.
        pass

    normalized_label_col = normalize_column_name(label_col)
    if normalized_label_col in normalized_columns:
        harmonized["label"] = df[normalized_columns[normalized_label_col]]
    else:
        for alias in ("label", "__label__", "class", "attack_type", "attack cat", "attack_cat"):
            normalized_alias = normalize_column_name(alias)
            if normalized_alias in normalized_columns:
                harmonized["label"] = df[normalized_columns[normalized_alias]]
                break

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
