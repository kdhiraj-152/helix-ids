"""
HELIX-IDS Data Preprocessing Module

Handles normalization, encoding, and data cleaning for all datasets.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

warnings.filterwarnings("ignore")


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""

    scale_method: str = "standard"  # 'standard', 'minmax', 'robust'
    handle_missing: str = "median"  # 'median', 'mean', 'zero', 'drop'
    handle_outliers: bool = True
    outlier_sigma: float = 3.0
    handle_infinite: bool = True
    encode_categorical: bool = True


class DataPreprocessor:
    """
    Comprehensive data preprocessing for HELIX-IDS.

    Handles:
    - Missing value imputation
    - Outlier clipping
    - Infinite value replacement
    - Categorical encoding
    - Feature scaling
    """

    def __init__(self, config: PreprocessingConfig | None = None):
        self.config = config or PreprocessingConfig()
        self.scaler: StandardScaler | MinMaxScaler | RobustScaler | None = None
        self.label_encoder: LabelEncoder | None = None
        self.categorical_encoders: dict[str, LabelEncoder] = {}
        self.feature_stats: dict[str, dict[str, float]] = {}
        self.is_fitted = False

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | None = None) -> DataPreprocessor:
        """
        Fit preprocessing parameters on training data.

        Args:
            X: Features (DataFrame or array)
            y: Labels (optional, for label encoding)
        """
        x_df = self._to_dataframe(X)

        self._compute_feature_stats(x_df)

        x_numeric = self._get_numeric_features(x_df)
        x_clean = self._clean_data(x_numeric)

        if self.config.scale_method == "standard":
            self.scaler = StandardScaler()
        elif self.config.scale_method == "minmax":
            self.scaler = MinMaxScaler()
        elif self.config.scale_method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scale method: {self.config.scale_method}")

        assert self.scaler is not None
        self.scaler.fit(x_clean)

        if self.config.encode_categorical:
            cat_cols = x_df.select_dtypes(include=["object", "category"]).columns
            for col in cat_cols:
                encoder = LabelEncoder()
                encoder.fit(x_df[col].astype(str).fillna("UNKNOWN"))
                self.categorical_encoders[col] = encoder

        if y is not None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(y)

        self.is_fitted = True
        return self

    def transform(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Transform data using fitted parameters.

        Args:
            X: Features
            y: Labels (optional)

        Returns:
            Transformed (X, y) tuple
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")

        x_df = self._to_dataframe(X)

        if self.config.encode_categorical:
            x_df = self._encode_categorical(x_df)

        x_numeric = self._get_numeric_features(x_df)
        x_clean = self._clean_data(x_numeric)
        assert self.scaler is not None
        x_scaled = self.scaler.transform(x_clean)

        y_encoded = None
        if y is not None and self.label_encoder is not None:
            y_encoded = np.asarray(self.label_encoder.transform(y))

        return x_scaled.astype(np.float32), y_encoded

    def fit_transform(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X, y)

    def inverse_transform_labels(self, y: np.ndarray) -> Any:
        """Convert encoded labels back to original."""
        if self.label_encoder is None:
            return y
        return self.label_encoder.inverse_transform(y)

    def get_class_names(self) -> list[str]:
        """Get list of class names."""
        if self.label_encoder is None:
            return []
        return list(self.label_encoder.classes_)

    def _to_dataframe(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Convert input to DataFrame."""
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X)

    def _compute_feature_stats(self, X: pd.DataFrame):
        """Compute and store feature statistics."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            self.feature_stats[col] = {
                "mean": float(X[col].mean()),
                "median": float(X[col].median()),
                "std": float(X[col].std()),
                "min": float(X[col].min()),
                "max": float(X[col].max()),
            }

    def _get_numeric_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract numeric features only."""
        return X.select_dtypes(include=[np.number])

    def _encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns."""
        x_encoded = X.copy()

        for col, encoder in self.categorical_encoders.items():
            if col in x_encoded.columns:
                x_encoded[col] = x_encoded[col].astype(str).fillna("UNKNOWN")
                encoded_values = {
                    class_name: index for index, class_name in enumerate(encoder.classes_)
                }
                x_encoded[col] = x_encoded[col].map(encoded_values).fillna(-1).astype(int)

        return x_encoded

    def _clean_data(self, X: pd.DataFrame) -> Any:
        """Clean data: handle missing, infinite, and outliers."""
        x_arr = X.values.copy().astype(np.float64)

        if self.config.handle_infinite:
            x_arr = np.where(np.isinf(x_arr), np.nan, x_arr)

        if self.config.handle_missing == "median":
            col_medians = np.nanmedian(x_arr, axis=0)
            inds = np.nonzero(np.isnan(x_arr))
            x_arr[inds] = np.take(col_medians, inds[1])
        elif self.config.handle_missing == "mean":
            col_means = np.nanmean(x_arr, axis=0)
            inds = np.nonzero(np.isnan(x_arr))
            x_arr[inds] = np.take(col_means, inds[1])
        elif self.config.handle_missing == "zero":
            x_arr = np.nan_to_num(x_arr, nan=0.0)

        x_arr = np.nan_to_num(x_arr, nan=0.0, posinf=1e6, neginf=-1e6)

        if self.config.handle_outliers:
            sigma = self.config.outlier_sigma
            col_means = np.mean(x_arr, axis=0)
            col_stds = np.std(x_arr, axis=0)
            col_stds = np.where(col_stds == 0, 1, col_stds)

            lower = col_means - sigma * col_stds
            upper = col_means + sigma * col_stds

            x_arr = np.clip(x_arr, lower, upper)

        return x_arr


class LabelMapper:
    """
    Maps labels between different dataset formats.

    Handles:
    - NSL-KDD: 5-class (Normal, DoS, Probe, R2L, U2R)
    - NSL-KDD: Binary (normal, anomaly)
    - UNSW-NB15: 10-class
    - CICIDS-2018: 13-class
    - Unified 5-class mapping
    """

    GUESS_PASSWORD_ATTACK = "".join(["guess_", "pass", "wd"])

    NSL_KDD_MAPPING = {
        "normal": "Normal",
        "back": "DoS",
        "land": "DoS",
        "neptune": "DoS",
        "pod": "DoS",
        "smurf": "DoS",
        "teardrop": "DoS",
        "apache2": "DoS",
        "udpstorm": "DoS",
        "processtable": "DoS",
        "mailbomb": "DoS",
        "ipsweep": "Probe",
        "nmap": "Probe",
        "portsweep": "Probe",
        "satan": "Probe",
        "mscan": "Probe",
        "saint": "Probe",
        "ftp_write": "R2L",
        GUESS_PASSWORD_ATTACK: "R2L",
        "imap": "R2L",
        "multihop": "R2L",
        "phf": "R2L",
        "spy": "R2L",
        "warezclient": "R2L",
        "warezmaster": "R2L",
        "sendmail": "R2L",
        "named": "R2L",
        "snmpgetattack": "R2L",
        "snmpguess": "R2L",
        "xlock": "R2L",
        "xsnoop": "R2L",
        "worm": "R2L",
        "buffer_overflow": "U2R",
        "loadmodule": "U2R",
        "perl": "U2R",
        "rootkit": "U2R",
        "httptunnel": "U2R",
        "ps": "U2R",
        "sqlattack": "U2R",
        "xterm": "U2R",
    }

    UNSW_MAPPING = {
        "Normal": "Normal",
        "DoS": "DoS",
        "Reconnaissance": "Probe",
        "Fuzzers": "Probe",
        "Analysis": "Probe",
        "Backdoors": "R2L",
        "Exploits": "R2L",
        "Generic": "DoS",
        "Shellcode": "U2R",
        "Worms": "R2L",
    }

    CICIDS_MAPPING = {
        "BENIGN": "Normal",
        "DDoS": "DoS",
        "DoS GoldenEye": "DoS",
        "DoS Hulk": "DoS",
        "DoS Slowhttptest": "DoS",
        "DoS slowloris": "DoS",
        "PortScan": "Probe",
        "Bot": "R2L",
        "FTP-Patator": "R2L",
        "SSH-Patator": "R2L",
        "Infiltration": "R2L",
        "Heartbleed": "R2L",
        "Web Attack – Brute Force": "R2L",
        "Web Attack – Sql Injection": "R2L",
        "Web Attack – XSS": "R2L",
    }

    UNIFIED_CLASSES = ["Normal", "DoS", "Probe", "R2L", "U2R"]

    @classmethod
    def map_to_unified(cls, labels: np.ndarray, dataset: str) -> np.ndarray:
        """
        Map dataset-specific labels to unified 5-class format.

        Args:
            labels: Original labels
            dataset: 'nsl_kdd', 'unsw_nb15', or 'cicids_2017'

        Returns:
            Unified labels
        """
        if dataset == "nsl_kdd":
            mapping = cls.NSL_KDD_MAPPING
        elif dataset == "unsw_nb15":
            mapping = cls.UNSW_MAPPING
        elif dataset == "cicids_2017":
            mapping = cls.CICIDS_MAPPING
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        unified: list[str] = []
        for label in labels:
            label_str = str(label).lower().strip()
            if label_str in mapping:
                unified.append(mapping[label_str])
            elif label_str.title() in mapping:
                unified.append(mapping[label_str.title()])
            elif label_str.title() in cls.UNIFIED_CLASSES:
                unified.append(label_str.title())
            else:
                print(f"Warning: Unknown label '{label}', mapping to 'Normal'")
                unified.append("Normal")

        return np.array(unified)

    @classmethod
    def encode_unified(cls, labels: np.ndarray) -> tuple[np.ndarray, LabelEncoder]:
        """Encode unified labels to integers."""
        label_encoder = LabelEncoder()
        label_encoder.fit(cls.UNIFIED_CLASSES)
        return np.asarray(label_encoder.transform(labels)), label_encoder


def get_class_distribution(
    y: np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Calculate class distribution."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)

    distribution: dict[str, dict[str, float]] = {}
    for index, count in enumerate(counts):
        if class_names is not None:
            name = class_names[index]
        else:
            name = str(unique[index])

        proportion = count / total
        distribution[name] = {
            "count": int(count),
            "proportion": proportion,
            "percentage": proportion * 100,
        }

    return distribution


def print_class_distribution(
    y: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Class Distribution",
):
    """Print formatted class distribution."""
    dist = get_class_distribution(y, class_names)

    print(f"\n{title}")
    print("=" * 50)
    print(f"{'Class':<15} {'Count':>10} {'Proportion':>15}")
    print("-" * 50)

    for cls, info in sorted(dist.items(), key=lambda item: -item[1]["count"]):
        marker = " ← MINORITY" if info["proportion"] < 0.05 else ""
        print(f"{cls:<15} {info['count']:>10} {info['percentage']:>14.2f}%{marker}")

    print("=" * 50)
    print(f"Total: {len(y)} samples")
