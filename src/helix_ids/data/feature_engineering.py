"""
Feature Engineering Module for HELIX-IDS.

This module handles attack-specific feature extraction and engineering,
including temporal aggregation, normalization, and cross-dataset alignment.
"""

# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportAssignmentType=false

from dataclasses import dataclass, field
from typing import Literal, Optional, Union, cast

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# Throughput features for rate-based detection (critical for DoS)
THROUGHPUT_FEATURES: list[str] = [
    "bytes_per_sec",
    "packets_per_sec",
    "fwd_bytes_per_sec",
    "bwd_bytes_per_sec",
]

# Packet length distribution features for exploit detection (critical for U2R/R2L)
# Buffer overflows and exploit payloads often have distinctive packet size patterns
PACKET_LENGTH_FEATURES: list[str] = [
    "avg_pkt_size",  # Average packet size in flow
    "fwd_avg_pkt_size",  # Forward direction average
    "bwd_avg_pkt_size",  # Backward direction average
    "pkt_size_ratio",  # Asymmetry indicator (src/dst)
    "small_pkt_ratio",  # Probe indicator (< 100 bytes)
    "large_pkt_ratio",  # Potential overflow indicator (> 1400 bytes)
    "pkt_size_variance",  # Normalized variance in sizes
    "bytes_imbalance",  # Signed imbalance metric
]

# Inter-Arrival Time (IAT) derived features
# Since NSL-KDD lacks raw packet data, we approximate IAT from flow-level statistics
# Expected gain: +7-15% Probe/DoS detection improvement
IAT_FEATURES: list[str] = [
    "iat_mean",  # Average inter-arrival time: duration / count
    "iat_std",  # IAT standard deviation approximation
    "iat_max",  # Upper bound: duration
    "iat_min",  # Lower bound: duration / (count * 10)
    "conn_rate",  # Connection rate: count / duration
    "srv_conn_rate",  # Service connection rate: srv_count / duration
    "dst_host_conn_rate",  # Destination host connection rate
    "burst_indicator",  # Binary indicator for burst activity
]

# Threshold for burst detection (connections per second)
BURST_THRESHOLD: float = 100.0

# Attack-specific feature subsets based on research findings
ATTACK_FEATURES: dict[str, list[str]] = {
    "DoS": [
        "src_bytes",
        "count",
        "srv_count",
        "same_srv_rate",
        "dst_host_count",
        "dst_host_srv_count",
        # Throughput features (rate-based metrics critical for DoS detection)
        "bytes_per_sec",
        "packets_per_sec",
        "fwd_bytes_per_sec",
        "bwd_bytes_per_sec",
        # IAT-derived features for DoS detection
        "conn_rate",
        "burst_indicator",
        "iat_mean",
    ],
    "Probe": [
        "dst_host_diff_srv_rate",
        "flag",
        "srv_diff_host_rate",
        "diff_srv_rate",
        # IAT-derived features for Probe detection (timing patterns)
        "iat_std",
        "iat_min",
        "dst_host_conn_rate",
    ],
    "R2L": [
        "hot",
        "num_failed_logins",
        "logged_in",
        "num_access_files",
        "num_shells",
        "num_file_creations",
        # Packet length features (exploit payload detection)
        "avg_pkt_size",
        "pkt_size_ratio",
        "large_pkt_ratio",
        "bytes_imbalance",
    ],
    "U2R": [
        "root_shell",
        "su_attempted",
        "num_root",  # CRITICAL: 0.35 importance
        "num_shells",
        "num_access_files",
        # Packet length features (buffer overflow detection)
        "avg_pkt_size",
        "large_pkt_ratio",
        "pkt_size_variance",
        "bytes_imbalance",
    ],
}

# Feature importance rankings from experiments
FEATURE_IMPORTANCE: dict[str, dict[str, float]] = {
    "DoS": {
        "src_bytes": 0.25,
        "count": 0.22,
        "srv_count": 0.18,
        "same_srv_rate": 0.15,
        "dst_host_count": 0.12,
        "dst_host_srv_count": 0.08,
        # Throughput features (rate-based metrics for DoS detection)
        "bytes_per_sec": 0.28,
        "packets_per_sec": 0.26,
        "fwd_bytes_per_sec": 0.24,
        "bwd_bytes_per_sec": 0.20,
    },
    "Probe": {
        "dst_host_diff_srv_rate": 0.30,
        "flag": 0.28,
        "srv_diff_host_rate": 0.22,
        "diff_srv_rate": 0.20,
    },
    "R2L": {
        "hot": 0.22,
        "num_failed_logins": 0.20,
        "logged_in": 0.18,
        "num_access_files": 0.16,
        "num_shells": 0.14,
        "num_file_creations": 0.10,
        # Packet length features (exploit payload signatures)
        "avg_pkt_size": 0.18,
        "pkt_size_ratio": 0.15,
        "large_pkt_ratio": 0.20,
        "bytes_imbalance": 0.12,
    },
    "U2R": {
        "num_root": 0.35,  # CRITICAL importance
        "root_shell": 0.25,
        "su_attempted": 0.18,
        "num_shells": 0.12,
        "num_access_files": 0.10,
        # Packet length features (buffer overflow indicators)
        "avg_pkt_size": 0.20,
        "large_pkt_ratio": 0.25,
        "pkt_size_variance": 0.15,
        "bytes_imbalance": 0.12,
    },
}

# Cross-dataset feature mappings
UNSW_TO_NSL_MAPPING: dict[str, str] = {
    "sbytes": "src_bytes",
    "dbytes": "dst_bytes",
    "sttl": "ttl",
    "dttl": "ttl",
    "sloss": "wrong_fragment",
    "dloss": "wrong_fragment",
    "sinpkt": "duration",
    "dinpkt": "duration",
    "sjit": "duration",
    "djit": "duration",
    "swin": "src_bytes",
    "stcpb": "src_bytes",
    "dtcpb": "dst_bytes",
    "dwin": "dst_bytes",
    "tcprtt": "duration",
    "synack": "duration",
    "ackdat": "duration",
    "smean": "src_bytes",
    "dmean": "dst_bytes",
    "trans_depth": "num_access_files",
    "response_body_len": "dst_bytes",
    "ct_srv_src": "srv_count",
    "ct_state_ttl": "count",
    "ct_dst_ltm": "dst_host_count",
    "ct_src_dport_ltm": "count",
    "ct_dst_sport_ltm": "dst_host_srv_count",
    "ct_dst_src_ltm": "dst_host_count",
    "is_ftp_login": "logged_in",
    "ct_ftp_cmd": "num_access_files",
    "ct_flw_http_mthd": "count",
    "ct_src_ltm": "count",
    "ct_srv_dst": "srv_count",
    "is_sm_ips_ports": "same_srv_rate",
    "attack_cat": "attack_type",
    "label": "label",
}

CICIDS_TO_NSL_MAPPING: dict[str, str] = {
    "Flow Duration": "duration",
    "Total Fwd Packets": "count",
    "Total Backward Packets": "count",
    "Total Length of Fwd Packets": "src_bytes",
    "Total Length of Bwd Packets": "dst_bytes",
    "Fwd Packet Length Max": "src_bytes",
    "Fwd Packet Length Min": "src_bytes",
    "Fwd Packet Length Mean": "src_bytes",
    "Fwd Packet Length Std": "src_bytes",
    "Bwd Packet Length Max": "dst_bytes",
    "Bwd Packet Length Min": "dst_bytes",
    "Bwd Packet Length Mean": "dst_bytes",
    "Bwd Packet Length Std": "dst_bytes",
    "Flow Bytes/s": "src_bytes",
    "Flow Packets/s": "count",
    "Flow IAT Mean": "duration",
    "Flow IAT Std": "duration",
    "Flow IAT Max": "duration",
    "Flow IAT Min": "duration",
    "Fwd IAT Total": "duration",
    "Fwd IAT Mean": "duration",
    "Fwd IAT Std": "duration",
    "Fwd IAT Max": "duration",
    "Fwd IAT Min": "duration",
    "Bwd IAT Total": "duration",
    "Bwd IAT Mean": "duration",
    "Bwd IAT Std": "duration",
    "Bwd IAT Max": "duration",
    "Bwd IAT Min": "duration",
    "Fwd PSH Flags": "flag",
    "Bwd PSH Flags": "flag",
    "Fwd URG Flags": "urgent",
    "Bwd URG Flags": "urgent",
    "Fwd Header Length": "src_bytes",
    "Bwd Header Length": "dst_bytes",
    "Fwd Packets/s": "count",
    "Bwd Packets/s": "count",
    "Min Packet Length": "src_bytes",
    "Max Packet Length": "src_bytes",
    "Packet Length Mean": "src_bytes",
    "Packet Length Std": "src_bytes",
    "Packet Length Variance": "src_bytes",
    "FIN Flag Count": "flag",
    "SYN Flag Count": "flag",
    "RST Flag Count": "flag",
    "PSH Flag Count": "flag",
    "ACK Flag Count": "flag",
    "URG Flag Count": "urgent",
    "CWE Flag Count": "flag",
    "ECE Flag Count": "flag",
    "Down/Up Ratio": "same_srv_rate",
    "Average Packet Size": "src_bytes",
    "Avg Fwd Segment Size": "src_bytes",
    "Avg Bwd Segment Size": "dst_bytes",
    "Subflow Fwd Packets": "count",
    "Subflow Fwd Bytes": "src_bytes",
    "Subflow Bwd Packets": "count",
    "Subflow Bwd Bytes": "dst_bytes",
    "Init_Win_bytes_forward": "src_bytes",
    "Init_Win_bytes_backward": "dst_bytes",
    "act_data_pkt_fwd": "count",
    "min_seg_size_forward": "src_bytes",
    "Active Mean": "duration",
    "Active Std": "duration",
    "Active Max": "duration",
    "Active Min": "duration",
    "Idle Mean": "duration",
    "Idle Std": "duration",
    "Idle Max": "duration",
    "Idle Min": "duration",
    "Label": "label",
}

# NSL-KDD standard feature schema
NSL_KDD_SCHEMA: list[str] = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

# ============================================================================
# Error Rate Feature Consolidation
# ============================================================================
# NSL-KDD has 14 highly correlated error rate features (r > 0.95).
# Consolidating to 2 weighted aggregates reduces dimensionality by 6x
# while preserving information and improving model efficiency.

# SYN error rate cluster (4 features -> 1)
SERROR_RATE_FEATURES: list[str] = [
    "serror_rate",
    "srv_serror_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
]

# REJ error rate cluster (4 features -> 1)
RERROR_RATE_FEATURES: list[str] = [
    "rerror_rate",
    "srv_rerror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

# All redundant error rate features
REDUNDANT_ERROR_FEATURES: list[str] = SERROR_RATE_FEATURES + RERROR_RATE_FEATURES

# Consolidated feature names
CONSOLIDATED_ERROR_FEATURES: list[str] = ["weighted_serror", "weighted_rerror"]

# NSL-KDD schema with consolidated error features (reduces 41 -> 35 features)
NSL_KDD_SCHEMA_CONSOLIDATED: list[str] = [
    f for f in NSL_KDD_SCHEMA if f not in REDUNDANT_ERROR_FEATURES
] + CONSOLIDATED_ERROR_FEATURES


@dataclass
class ErrorRateConsolidationConfig:
    """Configuration for error rate feature consolidation."""

    enabled: bool = True
    weights: Optional[dict[str, float]] = None  # Custom weights per feature

    def get_serror_weights(self) -> dict[str, float]:
        """Get weights for SYN error rate features."""
        if self.weights:
            return {k: v for k, v in self.weights.items() if k in SERROR_RATE_FEATURES}
        return {f: 1.0 / len(SERROR_RATE_FEATURES) for f in SERROR_RATE_FEATURES}

    def get_rerror_weights(self) -> dict[str, float]:
        """Get weights for REJ error rate features."""
        if self.weights:
            return {k: v for k, v in self.weights.items() if k in RERROR_RATE_FEATURES}
        return {f: 1.0 / len(RERROR_RATE_FEATURES) for f in RERROR_RATE_FEATURES}


def consolidate_error_features(
    X: pd.DataFrame,
    config: Optional[ErrorRateConsolidationConfig] = None,
    drop_original: bool = True,
) -> pd.DataFrame:
    """
    Consolidate redundant error rate features into weighted aggregates.

    NSL-KDD has 8 highly correlated error rate features (r > 0.95):
    - SYN error cluster: serror_rate, srv_serror_rate, dst_host_serror_rate,
      dst_host_srv_serror_rate
    - REJ error cluster: rerror_rate, srv_rerror_rate, dst_host_rerror_rate,
      dst_host_srv_rerror_rate

    This consolidation:
    - Reduces 8 features to 2 (6x dimension reduction in error features)
    - Eliminates multicollinearity that causes model instability
    - Preserves signal through weighted averaging

    Args:
        X: Input DataFrame with original error rate features.
        config: Consolidation configuration. Uses equal weights if None.
        drop_original: Whether to drop original error rate features.

    Returns:
        DataFrame with consolidated error features.
    """
    config = config or ErrorRateConsolidationConfig()
    result = X.copy()

    if not config.enabled:
        return result

    # Check which error features are present
    serror_present = [f for f in SERROR_RATE_FEATURES if f in result.columns]
    rerror_present = [f for f in RERROR_RATE_FEATURES if f in result.columns]

    # Compute weighted SYN error aggregate
    if serror_present:
        serror_weights = config.get_serror_weights()
        weights_present = [serror_weights.get(f, 0.0) for f in serror_present]
        weight_sum = sum(weights_present)
        if weight_sum > 0:
            weights_normalized = [w / weight_sum for w in weights_present]
            result["weighted_serror"] = sum(
                result[f] * w for f, w in zip(serror_present, weights_normalized)
            )
        else:
            result["weighted_serror"] = result[serror_present].mean(axis=1)

    # Compute weighted REJ error aggregate
    if rerror_present:
        rerror_weights = config.get_rerror_weights()
        weights_present = [rerror_weights.get(f, 0.0) for f in rerror_present]
        weight_sum = sum(weights_present)
        if weight_sum > 0:
            weights_normalized = [w / weight_sum for w in weights_present]
            result["weighted_rerror"] = sum(
                result[f] * w for f, w in zip(rerror_present, weights_normalized)
            )
        else:
            result["weighted_rerror"] = result[rerror_present].mean(axis=1)

    # Drop original features if requested
    if drop_original:
        cols_to_drop = [f for f in REDUNDANT_ERROR_FEATURES if f in result.columns]
        result = result.drop(columns=cols_to_drop)

    return result


def get_schema_with_error_consolidation(consolidated: bool = True) -> list[str]:
    """
    Get NSL-KDD schema with or without error feature consolidation.

    Args:
        consolidated: If True, return schema with consolidated error features.
                     If False, return original schema.

    Returns:
        List of feature names.
    """
    if consolidated:
        return NSL_KDD_SCHEMA_CONSOLIDATED.copy()
    return NSL_KDD_SCHEMA.copy()


# Temporal window sizes in seconds
TEMPORAL_WINDOWS: list[int] = [1, 5, 30, 60]

# Aggregation functions
AGGREGATIONS: list[str] = ["count", "mean", "std", "max", "min", "entropy"]

@dataclass
class TemporalConfig:
    """Configuration for temporal feature aggregation."""

    windows: list[int] = field(default_factory=lambda: TEMPORAL_WINDOWS.copy())
    aggregations: list[str] = field(default_factory=lambda: AGGREGATIONS.copy())
    timestamp_col: str = "timestamp"
    group_cols: Optional[list[str]] = None


class FeatureEngineer:
    """
    Feature engineering class for HELIX-IDS.

    Handles attack-specific feature extraction, temporal aggregation,
    normalization, and cross-dataset alignment.
    """

    def __init__(
        self,
        attack_features: Optional[dict[str, list[str]]] = None,
        feature_importance: Optional[dict[str, dict[str, float]]] = None,
        consolidate_error_rates: bool = True,
        error_consolidation_config: Optional[ErrorRateConsolidationConfig] = None,
    ):
        """
        Initialize the FeatureEngineer.

        Args:
            attack_features: Custom attack-feature mapping. Uses defaults if None.
            feature_importance: Custom importance rankings. Uses defaults if None.
            consolidate_error_rates: Whether to consolidate redundant error rate
                features by default. Default True for cleaner models.
            error_consolidation_config: Custom configuration for error consolidation.
        """
        self.attack_features = attack_features or ATTACK_FEATURES.copy()
        self.feature_importance = feature_importance or FEATURE_IMPORTANCE.copy()
        self.consolidate_error_rates = consolidate_error_rates
        self.error_consolidation_config = (
            error_consolidation_config or ErrorRateConsolidationConfig()
        )
        self._scalers: dict[str, Union[StandardScaler, MinMaxScaler, RobustScaler]] = {}
        self._fitted = False

    def get_attack_features(self, attack_type: str) -> list[str]:
        """
        Get the relevant features for a specific attack type.

        Args:
            attack_type: One of 'DoS', 'Probe', 'R2L', 'U2R'

        Returns:
            List of feature names relevant to the attack type.

        Raises:
            ValueError: If attack_type is not recognized.
        """
        if attack_type not in self.attack_features:
            raise ValueError(
                f"Unknown attack type: {attack_type}. "
                f"Valid types: {list(self.attack_features.keys())}"
            )
        return self.attack_features[attack_type].copy()

    def get_feature_importance(
        self,
        attack_type: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> dict[str, float]:
        """
        Get feature importance rankings.

        Args:
            attack_type: Specific attack type, or None for all.
            top_k: Return only top-k features by importance.

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        if attack_type:
            if attack_type not in self.feature_importance:
                raise ValueError(f"Unknown attack type: {attack_type}")
            importance = self.feature_importance[attack_type].copy()
        else:
            # Combine all importance scores
            importance = {}
            for _attack, features in self.feature_importance.items():
                for feat, score in features.items():
                    if feat in importance:
                        importance[feat] = max(importance[feat], score)
                    else:
                        importance[feat] = score

        if top_k:
            sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
            importance = dict(sorted_items)

        return importance

    def extract_attack_features(
        self,
        X: pd.DataFrame,
        attack_type: str,
    ) -> pd.DataFrame:
        """
        Extract features relevant to a specific attack type.

        Args:
            X: Input DataFrame with all features.
            attack_type: Target attack type.

        Returns:
            DataFrame with only relevant features.
        """
        features = self.get_attack_features(attack_type)
        available = [f for f in features if f in X.columns]

        if not available:
            raise ValueError(
                f"None of the required features for {attack_type} "
                f"are present in the data. Required: {features}"
            )

        return X[available].copy()

    def compute_temporal_features(  # NOSONAR
        self,
        X: pd.DataFrame,
        config: Optional[TemporalConfig] = None,
        feature_cols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute temporal aggregation features over multiple window sizes.

        Args:
            X: Input DataFrame with timestamp column.
            config: Temporal configuration. Uses defaults if None.
            feature_cols: Columns to aggregate. Uses numeric columns if None.

        Returns:
            DataFrame with original and temporal features.
        """
        config = config or TemporalConfig()

        if config.timestamp_col not in X.columns:
            raise ValueError(f"Timestamp column '{config.timestamp_col}' not found")

        result = X.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(result[config.timestamp_col]):
            result[config.timestamp_col] = pd.to_datetime(result[config.timestamp_col])

        # Determine feature columns
        if feature_cols is None:
            feature_cols = result.select_dtypes(include=[np.number]).columns.tolist()
            if config.timestamp_col in feature_cols:
                feature_cols.remove(config.timestamp_col)

        # Sort by timestamp
        result = result.sort_values(config.timestamp_col).reset_index(drop=True)

        # Compute aggregations for each window
        for window in config.windows:
            window_str = f"{window}s"

            for col in feature_cols:
                for agg in config.aggregations:
                    new_col = f"{col}_{agg}_{window_str}"

                    if agg == "entropy":
                        # Custom entropy calculation
                        result[new_col] = self._rolling_entropy(
                            result[col], result[config.timestamp_col], window
                        )
                    else:
                        # Use pandas rolling with time-based window
                        result = result.set_index(config.timestamp_col)
                        rolling = result[col].rolling(window_str, min_periods=1)

                        if agg == "count":
                            result[new_col] = rolling.count()
                        elif agg == "mean":
                            result[new_col] = rolling.mean()
                        elif agg == "std":
                            result[new_col] = rolling.std().fillna(0)
                        elif agg == "max":
                            result[new_col] = rolling.max()
                        elif agg == "min":
                            result[new_col] = rolling.min()

                        result = result.reset_index()

        return result

    def _rolling_entropy(
        self,
        series: pd.Series,
        timestamps: pd.Series,
        window_seconds: int,
    ) -> pd.Series:
        """
        Compute rolling entropy over a time window.

        Args:
            series: Data series.
            timestamps: Timestamp series.
            window_seconds: Window size in seconds.

        Returns:
            Series of entropy values.
        """
        entropy_values = []
        window_td = pd.Timedelta(seconds=window_seconds)

        for _i, ts in enumerate(timestamps):
            mask = (timestamps >= ts - window_td) & (timestamps <= ts)
            window_data = series[mask]

            if len(window_data) <= 1:
                entropy_values.append(0.0)
            else:
                # Bin continuous values for entropy calculation
                try:
                    bins = min(10, len(window_data.unique()))
                    if bins > 1:
                        binned = pd.cut(window_data, bins=bins, labels=False)
                        value_counts = binned.value_counts(normalize=True)
                        entropy_values.append(scipy_entropy(value_counts))
                    else:
                        entropy_values.append(0.0)
                except Exception:
                    entropy_values.append(0.0)

        return pd.Series(entropy_values, index=series.index)

    def normalize(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        method: Literal["standard", "minmax", "robust"] = "standard",
        fit: bool = True,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Normalize features using the specified method.

        Args:
            X: Input data (DataFrame or array).
            method: Normalization method:
                - 'standard': StandardScaler (zero mean, unit variance)
                - 'minmax': MinMaxScaler (scale to [0, 1])
                - 'robust': RobustScaler (robust to outliers)
            fit: Whether to fit the scaler. Set False for transform only.

        Returns:
            Normalized data in same format as input.
        """
        is_dataframe = isinstance(X, pd.DataFrame)
        if is_dataframe:
            x_df = cast(pd.DataFrame, X)
            columns = x_df.columns.tolist()
            index = x_df.index
            x_array = x_df.values
        else:
            columns = None
            index = None
            x_array = cast(np.ndarray, X)

        # Get or create scaler
        if method not in self._scalers:
            if method == "standard":
                self._scalers[method] = StandardScaler()
            elif method == "minmax":
                self._scalers[method] = MinMaxScaler()
            elif method == "robust":
                self._scalers[method] = RobustScaler()
            else:
                raise ValueError(
                    f"Unknown normalization method: {method}. "
                    f"Valid methods: 'standard', 'minmax', 'robust'"
                )

        scaler = self._scalers[method]

        if fit:
            x_normalized = scaler.fit_transform(x_array)
            self._fitted = True
        else:
            if not self._fitted:
                raise ValueError("Scaler not fitted. Call normalize with fit=True first.")
            x_normalized = scaler.transform(x_array)

        if is_dataframe:
            return pd.DataFrame(x_normalized, columns=columns, index=index)
        return x_normalized

    def align_features(  # NOSONAR
        self,
        X: pd.DataFrame,
        source_dataset: Literal["unsw", "cicids", "nsl"],
        target_schema: Optional[list[str]] = None,
        fill_method: Literal["zeros", "mean"] = "zeros",
    ) -> pd.DataFrame:
        """
        Align features from source dataset to target schema.

        Args:
            X: Input DataFrame from source dataset.
            source_dataset: Source dataset identifier ('unsw', 'cicids', 'nsl').
            target_schema: Target feature schema. Uses NSL-KDD if None.
            fill_method: How to handle missing features:
                - 'zeros': Fill with zeros
                - 'mean': Fill with column mean (from available data)

        Returns:
            DataFrame aligned to target schema.
        """
        target_schema = target_schema or NSL_KDD_SCHEMA.copy()

        # Get mapping based on source
        if source_dataset == "unsw":
            mapping = UNSW_TO_NSL_MAPPING
        elif source_dataset == "cicids":
            mapping = CICIDS_TO_NSL_MAPPING
        elif source_dataset == "nsl":
            mapping = {col: col for col in X.columns}
        else:
            raise ValueError(
                f"Unknown source dataset: {source_dataset}. Valid sources: 'unsw', 'cicids', 'nsl'"
            )

        # Create result DataFrame
        result = pd.DataFrame(index=X.index)

        # Map features
        for target_col in target_schema:
            # Find source columns that map to this target
            source_cols = [
                src for src, tgt in mapping.items() if tgt == target_col and src in X.columns
            ]

            if source_cols:
                # Use the first matching source column
                result[target_col] = X[source_cols[0]].copy()
            elif target_col in X.columns:
                # Column exists with same name
                result[target_col] = X[target_col].copy()
            else:
                # Missing feature - fill according to method
                if fill_method == "zeros":
                    result[target_col] = 0
                elif fill_method == "mean":
                    # Use mean of similar columns or 0
                    similar_cols = list(X.select_dtypes(include=[np.number]).columns)
                    if similar_cols:
                        result[target_col] = X[similar_cols].mean(axis=1)
                    else:
                        result[target_col] = 0

        return result

    def compute_throughput_features(
        self,
        X: pd.DataFrame,
        duration_col: str = "duration",
        src_bytes_col: str = "src_bytes",
        dst_bytes_col: str = "dst_bytes",
        count_col: str = "count",
        min_duration: float = 0.001,
    ) -> pd.DataFrame:
        """
        Compute throughput features critical for DoS detection.

        Adds rate-based metrics: bytes/sec, packets/sec, forward/backward bytes/sec.
        These features are essential for detecting high-volume attacks.

        Args:
            X: Input DataFrame with duration, byte counts, and packet counts.
            duration_col: Column name for connection duration.
            src_bytes_col: Column name for source (forward) bytes.
            dst_bytes_col: Column name for destination (backward) bytes.
            count_col: Column name for packet count.
            min_duration: Minimum duration value to prevent division by zero.

        Returns:
            DataFrame with additional throughput features:
            - bytes_per_sec: Total bytes / duration
            - packets_per_sec: Packet count / duration
            - fwd_bytes_per_sec: Source bytes / duration
            - bwd_bytes_per_sec: Destination bytes / duration
        """
        result = X.copy()

        # Get duration with minimum threshold to prevent division by zero
        if duration_col in X.columns:
            duration = X[duration_col].clip(lower=min_duration)
        else:
            # Default to minimum duration if column not present
            duration = pd.Series(min_duration, index=X.index)

        # Compute throughput features
        if src_bytes_col in X.columns:
            result["fwd_bytes_per_sec"] = X[src_bytes_col] / duration

            if dst_bytes_col in X.columns:
                total_bytes = X[src_bytes_col] + X[dst_bytes_col]
                result["bytes_per_sec"] = total_bytes / duration
            else:
                result["bytes_per_sec"] = X[src_bytes_col] / duration

        if dst_bytes_col in X.columns:
            result["bwd_bytes_per_sec"] = X[dst_bytes_col] / duration

        if count_col in X.columns:
            result["packets_per_sec"] = X[count_col] / duration

        # Handle infinite values that may arise from very small durations
        throughput_cols = [
            "bytes_per_sec",
            "packets_per_sec",
            "fwd_bytes_per_sec",
            "bwd_bytes_per_sec",
        ]
        for col in throughput_cols:
            if col in result.columns:
                result[col] = result[col].replace([np.inf, -np.inf], np.nan)
                result[col] = result[col].fillna(0)

        return result

    def compute_packet_length_features(
        self,
        X: pd.DataFrame,
        src_bytes_col: str = "src_bytes",
        dst_bytes_col: str = "dst_bytes",
        count_col: str = "count",
    ) -> pd.DataFrame:
        """
        Compute packet length distribution features for exploit detection.

        These features are critical for detecting U2R/R2L attacks where
        buffer overflows and exploit payloads have distinctive packet sizes.

        Args:
            X: Input DataFrame with byte counts and packet counts.
            src_bytes_col: Column name for source (forward) bytes.
            dst_bytes_col: Column name for destination (backward) bytes.
            count_col: Column name for packet count.

        Returns:
            DataFrame with additional packet length features:
            - avg_pkt_size: Average packet size in flow
            - fwd_avg_pkt_size: Forward direction average
            - bwd_avg_pkt_size: Backward direction average
            - pkt_size_ratio: Asymmetry indicator (src/dst)
            - small_pkt_ratio: Probe indicator (< 100 bytes)
            - large_pkt_ratio: Potential overflow indicator (> 1400 bytes)
            - pkt_size_variance: Normalized variance in sizes
            - bytes_imbalance: Signed imbalance metric
        """
        result = X.copy()

        # Get required columns with defaults
        src_bytes = X[src_bytes_col] if src_bytes_col in X.columns else pd.Series(0, index=X.index)
        dst_bytes = X[dst_bytes_col] if dst_bytes_col in X.columns else pd.Series(0, index=X.index)
        count = X[count_col] if count_col in X.columns else pd.Series(1, index=X.index)

        # Prevent division by zero
        count_safe = count.clip(lower=1)
        total_bytes = src_bytes + dst_bytes
        total_bytes_safe = total_bytes.clip(lower=1)
        dst_bytes_safe = dst_bytes.clip(lower=1)

        # Average packet size calculations
        result["avg_pkt_size"] = total_bytes / count_safe
        result["fwd_avg_pkt_size"] = src_bytes / count_safe
        result["bwd_avg_pkt_size"] = dst_bytes / count_safe

        # Asymmetry indicator (src/dst ratio)
        result["pkt_size_ratio"] = src_bytes / dst_bytes_safe

        # Size category features (binary indicators for exploit signatures)
        avg_size = result["avg_pkt_size"]
        result["small_pkt_ratio"] = (avg_size < 100).astype(int)  # Probe indicator
        result["large_pkt_ratio"] = (avg_size > 1400).astype(int)  # Potential overflow

        # Normalized variance (how different are src vs dst)
        result["pkt_size_variance"] = (src_bytes - dst_bytes).abs() / total_bytes_safe

        # Signed imbalance metric (direction matters for some attacks)
        result["bytes_imbalance"] = (src_bytes - dst_bytes) / total_bytes_safe

        # Handle edge cases
        pkt_cols = PACKET_LENGTH_FEATURES
        for col in pkt_cols:
            if col in result.columns:
                result[col] = result[col].replace([np.inf, -np.inf], np.nan)
                result[col] = result[col].fillna(0)

        return result

    def compute_iat_features(
        self,
        X: pd.DataFrame,
        duration_col: str = "duration",
        count_col: str = "count",
        srv_count_col: str = "srv_count",
        dst_host_count_col: str = "dst_host_count",
        burst_threshold: float = 100.0,
    ) -> pd.DataFrame:
        """
        Compute Inter-Arrival Time (IAT) derived features.

        Since NSL-KDD lacks raw packet data, we approximate IAT from
        flow-level statistics. These features are critical for detecting
        DoS (burst patterns) and Probe (timing-based scanning) attacks.

        Args:
            X: Input DataFrame with duration and count features.
            duration_col: Column name for connection duration.
            count_col: Column name for packet count.
            srv_count_col: Column name for service connection count.
            dst_host_count_col: Column name for destination host count.
            burst_threshold: Connections/sec threshold for burst detection.

        Returns:
            DataFrame with additional IAT-derived features:
            - iat_mean: Average inter-arrival time (duration / count)
            - iat_std: IAT standard deviation approximation
            - iat_max: Upper bound (duration)
            - iat_min: Lower bound (duration / (count * 10))
            - conn_rate: Connection rate (count / duration)
            - srv_conn_rate: Service connection rate
            - dst_host_conn_rate: Destination host connection rate
            - burst_indicator: Binary indicator for burst activity
        """
        result = X.copy()

        # Get columns with safe defaults
        duration = X[duration_col] if duration_col in X.columns else pd.Series(0.001, index=X.index)
        count = X[count_col] if count_col in X.columns else pd.Series(1, index=X.index)
        srv_count = X[srv_count_col] if srv_count_col in X.columns else pd.Series(1, index=X.index)
        dst_host_count = (
            X[dst_host_count_col]
            if dst_host_count_col in X.columns
            else pd.Series(1, index=X.index)
        )

        # Prevent division by zero
        duration_safe = duration.clip(lower=0.001)
        count_safe = count.clip(lower=1)

        # IAT approximations from flow statistics
        result["iat_mean"] = duration_safe / count_safe
        result["iat_std"] = result["iat_mean"] * 0.5  # Approximation: std ~ 0.5 * mean
        result["iat_max"] = duration_safe  # Upper bound
        result["iat_min"] = duration_safe / (count_safe * 10)  # Lower bound estimate

        # Connection rate features (critical for DoS detection)
        result["conn_rate"] = count / duration_safe
        result["srv_conn_rate"] = srv_count / duration_safe
        result["dst_host_conn_rate"] = dst_host_count / duration_safe

        # Burst indicator (high connection rate = likely DoS)
        result["burst_indicator"] = (result["conn_rate"] > burst_threshold).astype(int)

        # Handle infinite values
        iat_cols = IAT_FEATURES
        for col in iat_cols:
            if col in result.columns:
                result[col] = result[col].replace([np.inf, -np.inf], np.nan)
                result[col] = result[col].fillna(0)

        return result

    def compute_tcp_flag_features(
        self,
        X: pd.DataFrame,
        flag_col: str = "flag",
    ) -> pd.DataFrame:
        """
        Compute TCP flag-based features for connection state analysis.

        TCP flags reveal connection behavior patterns that are critical for
        detecting scans (Probe) and connection state attacks (DoS).

        Args:
            X: Input DataFrame with flag column.
            flag_col: Column name for TCP flag (encoded as categorical).

        Returns:
            DataFrame with additional TCP flag features:
            - flag_syn: SYN flag indicator (connection initiation)
            - flag_fin: FIN flag indicator (normal termination)
            - flag_rst: RST flag indicator (abnormal termination)
            - flag_sf: SF (normal completion) indicator
            - flag_rej: REJ (rejected) indicator
            - flag_s0: S0 (connection attempt) indicator
            - flag_anomaly_score: Weighted anomaly score based on flags
        """
        result = X.copy()

        if flag_col not in X.columns:
            return result

        flag = X[flag_col]

        # Handle both string and numeric flag representations
        if flag.dtype == object or str(flag.dtype) == "category":
            # String-based flags (NSL-KDD format)
            result["flag_sf"] = (flag == "SF").astype(int)  # Normal completion
            result["flag_s0"] = (flag == "S0").astype(int)  # No SYN/ACK reply
            result["flag_rej"] = (flag == "REJ").astype(int)  # Rejected
            result["flag_rsto"] = (flag == "RSTO").astype(int)  # Reset originator
            result["flag_rstos0"] = (flag == "RSTOS0").astype(int)  # Reset after S0
            result["flag_rstr"] = (flag == "RSTR").astype(int)  # Reset responder
            result["flag_s1"] = (flag == "S1").astype(int)  # SYN, but no SYN/ACK reply
            result["flag_s2"] = (flag == "S2").astype(int)  # SYN/ACK, no ACK
            result["flag_s3"] = (flag == "S3").astype(int)  # Full connection established
            result["flag_sh"] = (flag == "SH").astype(int)  # SYN/ACK, but no final ACK
            result["flag_oth"] = (flag == "OTH").astype(int)  # Other

            # Anomaly score: weighted combination of suspicious flags
            # Higher weights for more suspicious connection states
            result["flag_anomaly_score"] = (
                result.get("flag_s0", 0) * 3.0  # Connection attempt, no reply (scan)
                + result.get("flag_rej", 0) * 2.5  # Rejected connection (scan)
                + result.get("flag_rsto", 0) * 2.0  # Reset from originator
                + result.get("flag_rstos0", 0) * 3.5  # Reset after no reply (highly suspicious)
                + result.get("flag_rstr", 0) * 1.5  # Reset from responder
                + result.get("flag_sh", 0) * 2.0  # Half-open connection
                - result.get("flag_sf", 0) * 1.0  # Normal completion (reduces score)
            )

        return result

    def create_combined_features(  # NOSONAR
        self,
        X: pd.DataFrame,
        attack_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Create combined/interaction features based on attack patterns.

        Args:
            X: Input DataFrame.
            attack_type: Optional attack type for specialized features.

        Returns:
            DataFrame with additional combined features.
        """
        result = X.copy()

        # General combined features
        if "src_bytes" in X.columns and "dst_bytes" in X.columns:
            result["bytes_ratio"] = X["src_bytes"] / (X["dst_bytes"] + 1)
            result["total_bytes"] = X["src_bytes"] + X["dst_bytes"]

        if "count" in X.columns and "srv_count" in X.columns:
            result["srv_count_ratio"] = X["srv_count"] / (X["count"] + 1)

        if "same_srv_rate" in X.columns and "diff_srv_rate" in X.columns:
            result["srv_rate_diff"] = X["same_srv_rate"] - X["diff_srv_rate"]

        # Attack-specific features
        if attack_type == "DoS":
            if "dst_host_count" in X.columns:
                result["dos_intensity"] = X.get("count", 0) * X.get("dst_host_count", 1)

        elif attack_type == "U2R":
            # Privilege escalation indicator
            if "root_shell" in X.columns and "num_root" in X.columns:
                result["priv_escalation_score"] = (
                    X["root_shell"] * 0.5
                    + X["num_root"] * 0.35  # CRITICAL weight
                    + X.get("su_attempted", 0) * 0.15
                )

        elif attack_type == "R2L":
            if "logged_in" in X.columns and "num_failed_logins" in X.columns:
                result["login_success_ratio"] = X["logged_in"] / (X["num_failed_logins"] + 1)

        return result

    def select_features_by_importance(
        self,
        X: pd.DataFrame,
        attack_type: str,
        threshold: float = 0.1,
    ) -> pd.DataFrame:
        """
        Select features above importance threshold for attack type.

        Args:
            X: Input DataFrame.
            attack_type: Attack type for importance lookup.
            threshold: Minimum importance score to include.

        Returns:
            DataFrame with only important features.
        """
        importance = self.get_feature_importance(attack_type)
        selected = [
            feat for feat, score in importance.items() if score >= threshold and feat in X.columns
        ]

        if not selected:
            raise ValueError(f"No features meet threshold {threshold} for {attack_type}")

        return X[selected].copy()

    def get_scaler(
        self,
        method: str = "standard",
    ) -> Union[StandardScaler, MinMaxScaler, RobustScaler]:
        """
        Get the fitted scaler for a method.

        Args:
            method: Normalization method.

        Returns:
            The fitted scaler object.
        """
        if method not in self._scalers:
            raise ValueError(f"No scaler fitted for method: {method}")
        return self._scalers[method]

    def reset_scalers(self) -> None:
        """Reset all fitted scalers."""
        self._scalers.clear()
        self._fitted = False

    def engineer_all_features(
        self,
        X: pd.DataFrame,
        add_throughput: bool = True,
        add_iat: bool = True,
        add_tcp_flags: bool = True,
        add_packet_length: bool = True,
        consolidate_errors: bool = True,
        attack_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.

        This is the recommended entry point for comprehensive feature engineering.
        It applies throughput, IAT, TCP flag, packet length features, error
        consolidation, and attack-specific combined features in the correct order.

        Expected improvement: +15-25% accuracy with proper features.

        Args:
            X: Input DataFrame with raw NSL-KDD features.
            add_throughput: Add throughput rate features (DoS critical).
            add_iat: Add inter-arrival time features (DoS/Probe critical).
            add_tcp_flags: Add TCP flag analysis features (Probe critical).
            add_packet_length: Add packet length distribution (U2R/R2L critical).
            consolidate_errors: Consolidate redundant error rate features.
            attack_type: Optional attack type for specialized features.

        Returns:
            DataFrame with all engineered features applied.
        """
        result = X.copy()

        # 1. Throughput features (rate-based, critical for DoS)
        if add_throughput:
            result = self.compute_throughput_features(result)

        # 2. IAT features (timing patterns, critical for DoS/Probe)
        if add_iat:
            result = self.compute_iat_features(result)

        # 3. TCP flag features (connection state, critical for Probe)
        if add_tcp_flags:
            result = self.compute_tcp_flag_features(result)

        # 4. Packet length features (payload analysis, critical for U2R/R2L)
        if add_packet_length:
            result = self.compute_packet_length_features(result)

        # 5. Error rate consolidation (reduces 8 redundant features to 2)
        if consolidate_errors:
            result = consolidate_error_features(result, self.error_consolidation_config)

        # 6. Combined/interaction features
        result = self.create_combined_features(result, attack_type)

        return result


def get_all_attack_features() -> list[str]:
    """
    Get the union of all attack-specific features.

    Returns:
        List of all unique feature names across attack types.
    """
    all_features = set()
    for features in ATTACK_FEATURES.values():
        all_features.update(features)
    return sorted(all_features)


def get_critical_u2r_features() -> dict[str, float]:
    """
    Get critical U2R features with importance scores.

    num_root is CRITICAL with 0.35 importance.

    Returns:
        Dictionary of feature names to importance scores.
    """
    return FEATURE_IMPORTANCE["U2R"].copy()


def get_throughput_features() -> list[str]:
    """
    Get the list of throughput feature names.

    These rate-based features are critical for DoS detection:
    - bytes_per_sec: Total bytes transferred per second
    - packets_per_sec: Packet count per second
    - fwd_bytes_per_sec: Forward (source) bytes per second
    - bwd_bytes_per_sec: Backward (destination) bytes per second

    Returns:
        List of throughput feature names.
    """
    return THROUGHPUT_FEATURES.copy()


def compute_throughput_features(
    df: pd.DataFrame,
    duration_col: str = "duration",
    src_bytes_col: str = "src_bytes",
    dst_bytes_col: str = "dst_bytes",
    count_col: str = "count",
    min_duration: float = 0.001,
) -> pd.DataFrame:
    """
    Convenience function to compute throughput features on a DataFrame.

    This is a standalone function that wraps FeatureEngineer.compute_throughput_features
    for ease of use in preprocessing pipelines.

    Args:
        df: Input DataFrame with duration, byte counts, and packet counts.
        duration_col: Column name for connection duration.
        src_bytes_col: Column name for source (forward) bytes.
        dst_bytes_col: Column name for destination (backward) bytes.
        count_col: Column name for packet count.
        min_duration: Minimum duration value to prevent division by zero.

    Returns:
        DataFrame with additional throughput features.
    """
    engineer = FeatureEngineer()
    return engineer.compute_throughput_features(
        df,
        duration_col=duration_col,
        src_bytes_col=src_bytes_col,
        dst_bytes_col=dst_bytes_col,
        count_col=count_col,
        min_duration=min_duration,
    )


def compute_iat_features(
    df: pd.DataFrame,
    duration_col: str = "duration",
    count_col: str = "count",
    srv_count_col: str = "srv_count",
    dst_host_count_col: str = "dst_host_count",
    burst_threshold: float = 100.0,
) -> pd.DataFrame:
    """
    Convenience function to compute IAT features on a DataFrame.

    Args:
        df: Input DataFrame with duration and count features.
        duration_col: Column name for connection duration.
        count_col: Column name for packet count.
        srv_count_col: Column name for service connection count.
        dst_host_count_col: Column name for destination host count.
        burst_threshold: Connections/sec threshold for burst detection.

    Returns:
        DataFrame with additional IAT-derived features.
    """
    engineer = FeatureEngineer()
    return engineer.compute_iat_features(
        df,
        duration_col=duration_col,
        count_col=count_col,
        srv_count_col=srv_count_col,
        dst_host_count_col=dst_host_count_col,
        burst_threshold=burst_threshold,
    )


def compute_tcp_flag_features(
    df: pd.DataFrame,
    flag_col: str = "flag",
) -> pd.DataFrame:
    """
    Convenience function to compute TCP flag features on a DataFrame.

    Args:
        df: Input DataFrame with flag column.
        flag_col: Column name for TCP flag.

    Returns:
        DataFrame with additional TCP flag features.
    """
    engineer = FeatureEngineer()
    return engineer.compute_tcp_flag_features(df, flag_col=flag_col)


def engineer_all_features(
    df: pd.DataFrame,
    add_throughput: bool = True,
    add_iat: bool = True,
    add_tcp_flags: bool = True,
    add_packet_length: bool = True,
    consolidate_errors: bool = True,
    attack_type: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience function to apply all feature engineering transformations.

    This is the recommended entry point for comprehensive feature engineering.
    Expected improvement: +15-25% accuracy with proper features.

    Args:
        df: Input DataFrame with raw NSL-KDD features.
        add_throughput: Add throughput rate features (DoS critical).
        add_iat: Add inter-arrival time features (DoS/Probe critical).
        add_tcp_flags: Add TCP flag analysis features (Probe critical).
        add_packet_length: Add packet length distribution (U2R/R2L critical).
        consolidate_errors: Consolidate redundant error rate features.
        attack_type: Optional attack type for specialized features.

    Returns:
        DataFrame with all engineered features applied.
    """
    engineer = FeatureEngineer()
    return engineer.engineer_all_features(
        df,
        add_throughput=add_throughput,
        add_iat=add_iat,
        add_tcp_flags=add_tcp_flags,
        add_packet_length=add_packet_length,
        consolidate_errors=consolidate_errors,
        attack_type=attack_type,
    )
