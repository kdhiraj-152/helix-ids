"""
Core UnifiedDataLoader implementation for HELIX-IDS datasets.

Refactored from the 1230-line monolithic unified_loader.py.
Coordinates feature I/O, label mapping, and dataset configuration.
"""

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .dataset_config import DATASET_CONFIGS, DatasetConfig
from .feature_io import harmonize_cicids_frames, load_all_files, try_load_split
from .label_mapping import encode_labels, log_class_distribution, map_labels

logger = logging.getLogger(__name__)


class UnifiedDataLoader:
    """
    Refactored unified data loader. Loads, cleans, and standardizes formats
    across all supported datasets (NSL-KDD, UNSW-NB15, CICIDS-2018).
    """

    def __init__(
        self,
        label_mode: str = "unified_5class",
        scale_features: bool = True,
        handle_missing: bool = True,
        handle_outliers: bool = True,
        outlier_sigma: float = 3.0,
        verbose: bool = False,
    ):
        self.label_mode = label_mode
        self.scale_features = scale_features
        self.handle_missing = handle_missing
        self.handle_outliers = handle_outliers
        self.outlier_sigma = outlier_sigma
        self.verbose = verbose

        self._scaler: Optional[StandardScaler] = None
        self._label_encoder: Optional[LabelEncoder] = None
        self._feature_encoders: dict[str, LabelEncoder] = {}
        self._feature_stats: dict[str, np.ndarray] = {}

    def reset(self):
        """Reset all fitted transformers."""
        self._scaler = None
        self._label_encoder = None
        self._feature_encoders.clear()
        self._feature_stats.clear()

    def get_splits(
        self,
        dataset_name: str,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Load and split dataset into Train/Val/Test."""
        X, y, _ = self.load(dataset_name)

        # Base train / test split
        x_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Train / val split
        val_ratio = val_size / (1.0 - test_size)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val,
            y_train_val,
            test_size=val_ratio,
            random_state=random_state,
            stratify=y_train_val,
        )

        return {
            "train": (x_train, y_train),
            "val": (x_val, y_val),
            "test": (X_test, y_test),
        }

    def load(
        self, dataset_name: str, split: Optional[str] = None, fit: bool = True
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Load a standalone dataset.
        Returns: `(X_encoded, y_encoded, class_names)`
        """
        dataset_key = dataset_name.lower().replace("_", "-")
        if dataset_key not in DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_CONFIGS.keys())}"
            )
        config = DATASET_CONFIGS[dataset_key]

        if self.verbose:
            logger.info(f"Loading '{config.name}' (split: {split or 'all'})")

        df = self._load_dataframes(config, split)
        x_df, y_raw = self._extract_features_labels(df, config)

        y_mapped = map_labels(y_raw, config, label_mode=self.label_mode)
        y_encoded, class_names, encoder = encode_labels(
            y_mapped,
            config,
            label_mode=self.label_mode,
            encoder=self._label_encoder if not fit else None,
        )
        if fit:
            self._label_encoder = encoder

        if self.verbose:
            log_class_distribution(y_encoded, class_names)

        x_processed = self._preprocess_features(x_df, fit=fit)

        if self.verbose:
            logger.info(
                f"Loaded {config.name}: {x_processed.shape[0]} samples, "
                f"{x_processed.shape[1]} features, {len(class_names)} classes."
            )

        return x_processed, y_encoded, class_names

    def _encode_categorical_features(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Encode categorical features using label encoders."""
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                X[col] = X[col].fillna("__MISSING__").astype(str)
                le.fit(X[col])
                self._feature_encoders[col] = le
            if col in self._feature_encoders:
                le = self._feature_encoders[col]
                X[col] = X[col].fillna("__MISSING__").astype(str)
                X[col] = X[col].apply(
                    lambda v, e=le: e.transform([v])[0] if v in e.classes_ else -1
                )
        return X

    def _handle_missing_values(self, x_arr: np.ndarray, fit: bool) -> np.ndarray:
        """Handle missing values by imputing with medians."""
        if self.handle_missing:
            if fit:
                self._feature_stats["median"] = np.nanmedian(x_arr, axis=0)
            medians = self._feature_stats.get("median", np.nanmedian(x_arr, axis=0))
            for i in range(x_arr.shape[1]):
                mask = np.isnan(x_arr[:, i])
                if mask.any():
                    x_arr[mask, i] = medians[i] if not np.isnan(medians[i]) else 0
        return x_arr

    def _handle_outliers(self, x_arr: np.ndarray, fit: bool) -> np.ndarray:
        """Handle outliers by clipping to sigma bounds."""
        if self.handle_outliers:
            if fit:
                self._feature_stats["mean"] = np.nanmean(x_arr, axis=0)
                self._feature_stats["std"] = np.nanstd(x_arr, axis=0)
            means = self._feature_stats.get("mean", np.nanmean(x_arr, axis=0))
            stds = self._feature_stats.get("std", np.nanstd(x_arr, axis=0))
            stds = np.where(stds == 0, 1, stds)
            lower, upper = means - self.outlier_sigma * stds, means + self.outlier_sigma * stds
            x_arr = np.clip(x_arr, lower, upper)
        return x_arr

    # -----------------------------------------------------------------------
    # Private Implementation
    # -----------------------------------------------------------------------

    def _load_dataframes(self, config: DatasetConfig, split: Optional[str]) -> pd.DataFrame:
        if split:
            df = try_load_split(config, split)
            if df is not None:
                return df
            logger.warning(f"Split '{split}' not found. Loading all files.")

        dfs = load_all_files(config, self.verbose)
        if not dfs:
            raise FileNotFoundError(f"No data files found for {config.name} in {config.paths}")

        if config.name.startswith("CICIDS-"):
            dfs = harmonize_cicids_frames(dfs)

        return pd.concat(dfs, ignore_index=True)

    def _extract_features_labels(
        self, df: pd.DataFrame, config: DatasetConfig
    ) -> tuple[pd.DataFrame, np.ndarray]:
        label_col = next(
            (
                c
                for c in [
                    config.label_column,
                    "label",
                    "class",
                    "Label",
                    "Class",
                    "attack_cat",
                    "attack_type",
                ]
                if c in df.columns
            ),
            df.columns[-1],
        )

        # Remove header artifacts (e.g. repeated column names in data)
        header_rows = (
            df[label_col].astype(str).str.strip().str.lower().eq(str(label_col).strip().lower())
        )
        if header_rows.any():
            df = df.loc[~header_rows].copy()

        y = df[label_col].values
        drop_cols = [
            c for c in config.drop_columns + sorted({label_col, "label"}) if c in df.columns
        ]
        return df.drop(columns=drop_cols), y

    def _preprocess_features(self, X: Union[pd.DataFrame, np.ndarray], fit: bool) -> np.ndarray:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        X = self._encode_categorical_features(X, fit)

        x_arr = X.to_numpy(dtype=np.float32)
        x_arr = np.where(np.isinf(x_arr), np.nan, x_arr)

        x_arr = self._handle_missing_values(x_arr, fit)
        x_arr = self._handle_outliers(x_arr, fit)

        x_arr = np.nan_to_num(x_arr, nan=0.0, posinf=1e6, neginf=-1e6)

        if self.scale_features:
            if fit:
                self._scaler = StandardScaler()
                x_arr = self._scaler.fit_transform(x_arr)
            elif self._scaler:
                x_arr = self._scaler.transform(x_arr)

        return np.asarray(x_arr, dtype=np.float32)


# Convenience wrappers
def load_dataset(
    dataset_name: str, split: Optional[str] = None, **kwargs
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    return UnifiedDataLoader(**kwargs).load(dataset_name, split=split)


def get_dataset_splits(
    dataset_name: str, test_size: float = 0.15, val_size: float = 0.15, **kwargs
) -> dict:
    return UnifiedDataLoader(**kwargs).get_splits(dataset_name, test_size, val_size)


def list_available_datasets() -> dict[str, DatasetConfig]:
    seen, unique = set(), {}
    for name, config in DATASET_CONFIGS.items():
        if config.name not in seen:
            unique[name] = config
            seen.add(config.name)
    return unique
