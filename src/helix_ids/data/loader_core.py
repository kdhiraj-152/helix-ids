"""
Core UnifiedDataLoader implementation for HELIX-IDS datasets.

Refactored from the 1230-line monolithic unified_loader.py.
Coordinates feature I/O, label mapping, and dataset configuration.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional, Union, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .dataset_config import DATASET_CONFIGS, DatasetConfig
from .feature_io import harmonize_cicids_frames, load_all_files, try_load_split
from .geometric_representation_fixes import GeometricRepresentationFixer
from .label_mapping import encode_labels, log_class_distribution, map_labels

logger = logging.getLogger(__name__)


DOMAIN_NAME_TO_ID: dict[str, int] = {
    "nsl-kdd": 0,
    "unsw-nb15": 1,
    "cicids-2017": 2,
    "cicids-2018": 2,
}


class UnifiedDataLoader:
    """
    Refactored unified data loader. Loads, cleans, and standardizes formats
    across all supported datasets (NSL-KDD, UNSW-NB15, CICIDS-2018).
    """

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        label_mode: str = "unified_5class",
        scale_features: bool = True,
        handle_missing: bool = True,
        handle_outliers: bool = True,
        outlier_sigma: float = 3.0,
        verbose: bool = False,
    ):
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.label_mode = label_mode
        self.scale_features = scale_features
        self.handle_missing = handle_missing
        self.handle_outliers = handle_outliers
        self.outlier_sigma = outlier_sigma
        self.verbose = verbose

        self._scaler: Optional[StandardScaler] = None
        self._label_encoder: Optional[LabelEncoder] = None
        self._feature_encoders: dict[str, LabelEncoder] = {}
        self._feature_stats: dict[str, Any] = {}

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
        include_domain_id: bool = False,
    ) -> dict[str, tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Load and split dataset into Train/Val/Test."""
        if include_domain_id:
            X, y, _class_names, domain_ids = self.load_with_domain_ids(
                dataset_name,
                return_class_names=True,
            )
        else:
            X, y, _class_names = self.load(dataset_name, return_class_names=True)

        # Base train / test split
        if include_domain_id:
            x_train_val, X_test, y_train_val, y_test, d_train_val, d_test = train_test_split(
                X,
                y,
                domain_ids,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
        else:
            x_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

        # Train / val split
        val_ratio = val_size / (1.0 - test_size)
        if include_domain_id:
            x_train, x_val, y_train, y_val, d_train, d_val = train_test_split(
                x_train_val,
                y_train_val,
                d_train_val,
                test_size=val_ratio,
                random_state=random_state,
                stratify=y_train_val,
            )
        else:
            x_train, x_val, y_train, y_val = train_test_split(
                x_train_val,
                y_train_val,
                test_size=val_ratio,
                random_state=random_state,
                stratify=y_train_val,
            )

        if include_domain_id:
            return {
                "train": (
                    np.asarray(x_train, dtype=np.float32),
                    np.asarray(y_train, dtype=np.int64),
                    np.asarray(d_train, dtype=np.int64),
                ),
                "val": (
                    np.asarray(x_val, dtype=np.float32),
                    np.asarray(y_val, dtype=np.int64),
                    np.asarray(d_val, dtype=np.int64),
                ),
                "test": (
                    np.asarray(X_test, dtype=np.float32),
                    np.asarray(y_test, dtype=np.int64),
                    np.asarray(d_test, dtype=np.int64),
                ),
            }

        return {
            "train": (x_train, y_train),
            "val": (x_val, y_val),
            "test": (X_test, y_test),
        }

    def load(
        self,
        dataset_name: str,
        split: Optional[str] = None,
        fit: bool = True,
        return_class_names: bool = False,
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
        is_cicids = config.name.startswith("CICIDS-")

        if self.verbose:
            logger.info(f"Loading '{config.name}' (split: {split or 'all'})")

        cache_key = self._build_cache_key(dataset_key, split)
        cached = self._load_cicids_cache_with_trace(is_cicids, cache_key)
        if cached is not None:
            return cached

        df = self._load_dataframes(config, split)
        self._trace_cicids_stage(is_cicids, "L2: file read done")
        self._trace_cicids_stage(is_cicids, "L3: start preprocessing")
        x_df, y_raw = self._extract_features_labels(df, config)

        y_mapped = map_labels(y_raw, config, label_mode=self.label_mode)
        self._trace_cicids_stage(is_cicids, "L4: preprocessing done")
        self._trace_cicids_stage(is_cicids, "L5: start encoding")
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
        self._trace_cicids_stage(is_cicids, "L6: encoding done")

        if is_cicids:
            self._save_preprocessed_cache(cache_key, x_processed, y_encoded, class_names)

        if self.verbose:
            logger.info(
                f"Loaded {config.name}: {x_processed.shape[0]} samples, "
                f"{x_processed.shape[1]} features, {len(class_names)} classes."
            )

        if not return_class_names and self.verbose:
            logger.debug("return_class_names=False is deprecated; class_names are always returned")

        return x_processed, y_encoded, class_names

    def _trace_cicids_stage(self, is_cicids: bool, marker: str) -> None:
        """Emit deterministic stage markers for CICIDS loader diagnostics."""
        if is_cicids:
            print(marker, flush=True)

    def _load_cicids_cache_with_trace(
        self,
        is_cicids: bool,
        cache_key: str,
    ) -> Optional[tuple[np.ndarray, np.ndarray, list[str]]]:
        """Try CICIDS preprocessed cache while preserving stage markers."""
        self._trace_cicids_stage(is_cicids, "L1: start file read")
        if not is_cicids:
            return None

        cached = self._try_load_preprocessed_cache(cache_key)
        if cached is None:
            return None

        self._trace_cicids_stage(is_cicids, "L2: file read done (preprocessed cache hit)")
        self._trace_cicids_stage(is_cicids, "L3: start preprocessing")
        self._trace_cicids_stage(is_cicids, "L4: preprocessing done (cache)")
        self._trace_cicids_stage(is_cicids, "L5: start encoding")
        self._trace_cicids_stage(is_cicids, "L6: encoding done (cache)")
        return cached

    def load_with_domain_ids(
        self,
        dataset_name: str,
        split: Optional[str] = None,
        fit: bool = True,
        return_class_names: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
        """Load dataset and return per-sample domain ids.

        Returns: `(X_encoded, y_encoded, class_names, domain_ids)`
        """
        X, y, class_names = self.load(
            dataset_name,
            split=split,
            fit=fit,
            return_class_names=return_class_names,
        )
        dataset_key = dataset_name.lower().replace("_", "-")
        if dataset_key not in DOMAIN_NAME_TO_ID:
            raise ValueError(f"No domain_id mapping for dataset '{dataset_name}' ({dataset_key}).")

        domain_ids = np.full(
            shape=(X.shape[0],),
            fill_value=DOMAIN_NAME_TO_ID[dataset_key],
            dtype=np.int64,
        )
        return X, y, class_names, domain_ids

    def _apply_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply geometric representation Fix 1: add interaction and distributional features."""
        fixer = GeometricRepresentationFixer()
        try:
            x_enhanced = fixer.add_interaction_features(X, minimal_set=True)
            if logger.isEnabledFor(logging.DEBUG):
                n_new = len(x_enhanced.columns) - len(X.columns)
                logger.debug(f"Added {n_new} interaction features during loading")
            return x_enhanced
        except Exception as exc:
            logger.warning(f"Failed to add interaction features: {exc}; continuing with raw features")
            return X

    def _encode_categorical_features(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Encode categorical features using label encoders."""
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in categorical_cols:
            series = X[col].fillna("__MISSING__").astype(str)
            if fit:
                le = LabelEncoder()
                le.fit(series)
                self._feature_encoders[col] = le
            if col in self._feature_encoders:
                le = self._feature_encoders[col]
                mapping = {label: idx for idx, label in enumerate(le.classes_)}
                X[col] = series.map(mapping).fillna(-1).astype(np.int64)
        return X

    def _cache_root(self) -> Path:
        """Return cache root used for preprocessed loader artifacts."""
        override = os.environ.get("HELIX_LOADER_CACHE_DIR")
        if override:
            return Path(override)
        return Path("data/processed/loader_cache")

    def _build_cache_key(self, dataset_key: str, split: Optional[str]) -> str:
        """Build deterministic cache key scoped to load configuration."""
        split_part = split or "all"
        return (
            f"{dataset_key}__{split_part}__"
            f"lm-{self.label_mode}__"
            f"scale-{int(self.scale_features)}__"
            f"miss-{int(self.handle_missing)}__"
            f"out-{int(self.handle_outliers)}__"
            f"sigma-{self.outlier_sigma:.2f}"
        )

    def _cache_path(self, cache_key: str) -> Path:
        """Return full cache file path for a cache key."""
        return self._cache_root() / f"{cache_key}.npz"

    def _try_load_preprocessed_cache(
        self,
        cache_key: str,
    ) -> Optional[tuple[np.ndarray, np.ndarray, list[str]]]:
        """Load preprocessed CICIDS cache when available."""
        cache_path = self._cache_path(cache_key)
        if not cache_path.exists():
            return None

        try:
            payload = np.load(cache_path, allow_pickle=True)
            X = np.asarray(payload["X"], dtype=np.float32)
            y = np.asarray(payload["y"], dtype=np.int64)
            class_names = [str(name) for name in payload["class_names"].tolist()]
            return X, y, class_names
        except Exception as exc:
            logger.warning("Failed to load preprocessed cache %s: %s", cache_path, exc)
            return None

    def _save_preprocessed_cache(
        self,
        cache_key: str,
        X: np.ndarray,
        y: np.ndarray,
        class_names: list[str],
    ) -> None:
        """Persist preprocessed CICIDS arrays for fast subsequent loads."""
        cache_path = self._cache_path(cache_key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            np.savez_compressed(
                cache_path,
                X=np.asarray(X, dtype=np.float32),
                y=np.asarray(y, dtype=np.int64),
                class_names=np.asarray(class_names, dtype=object),
            )
        except Exception as exc:
            logger.warning("Failed to write preprocessed cache %s: %s", cache_path, exc)

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
            means = cast(np.ndarray, self._feature_stats.get("mean", np.nanmean(x_arr, axis=0)))
            stds = cast(np.ndarray, self._feature_stats.get("std", np.nanstd(x_arr, axis=0)))
            stds = np.where(stds == 0, 1, stds)
            lower, upper = means - self.outlier_sigma * stds, means + self.outlier_sigma * stds
            x_arr = np.clip(x_arr, lower, upper)
        return x_arr

    # -----------------------------------------------------------------------
    # Private Implementation
    # -----------------------------------------------------------------------

    def _load_dataframes(self, config: DatasetConfig, split: Optional[str]) -> pd.DataFrame:
        if split is None and config.name.startswith("CICIDS-"):
            preprocessed_candidates = [
                Path("data/processed/cicids2018_cleaned.csv"),
                Path("data/processed/cicids-2018_cleaned.csv"),
            ]
            for preprocessed_path in preprocessed_candidates:
                if preprocessed_path.exists():
                    logger.info(
                        "Using preprocessed CICIDS fast-path: %s",
                        preprocessed_path,
                    )
                    return pd.read_csv(preprocessed_path, low_memory=False)

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

        # Fix 1: Add interaction and distributional features
        X = self._apply_interaction_features(X)

        categorical_cols = set(X.select_dtypes(include=["object", "category"]).columns.tolist())

        # Encode categorical before converting to numpy
        X = self._encode_categorical_features(X, fit)
        feature_names = [str(col) for col in X.columns.tolist()]

        x_arr = X.to_numpy(dtype=np.float32)
        x_arr = np.where(np.isinf(x_arr), np.nan, x_arr)

        x_arr = self._handle_missing_values(x_arr, fit)
        x_arr = self._handle_outliers(x_arr, fit)

        x_arr = np.nan_to_num(x_arr, nan=0.0, posinf=1e6, neginf=-1e6)

        # Fix 2: Apply split normalization (continuous vs categorical)
        if self.scale_features:
            if fit:
                fixer = GeometricRepresentationFixer()
                x_arr, norm_stats = fixer.split_normalization(
                    x_arr,
                    feature_names=feature_names,
                    categorical_cols=categorical_cols,
                    fit=True,
                )
                self._feature_stats["split_norm_stats"] = norm_stats
            elif "split_norm_stats" in self._feature_stats and self._scaler:
                # For inference, apply the same continuous-only normalization
                norm_stats = self._feature_stats.get("split_norm_stats", {})
                if norm_stats and norm_stats.get("scaler"):
                    continuous_idx = norm_stats.get("continuous_indices", [])
                    if continuous_idx:
                        x_arr[:, continuous_idx] = norm_stats["scaler"].transform(x_arr[:, continuous_idx])

        return np.asarray(x_arr, dtype=np.float32)


# Convenience wrappers
def load_dataset(
    dataset_name: str,
    split: Optional[str] = None,
    return_domain_id: bool = False,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, list[str]] | tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    loader = UnifiedDataLoader(**kwargs)
    if return_domain_id:
        return loader.load_with_domain_ids(
            dataset_name,
            split=split,
            return_class_names=True,
        )
    return loader.load(
        dataset_name,
        split=split,
        return_class_names=True,
    )


def get_dataset_splits(
    dataset_name: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    include_domain_id: bool = False,
    **kwargs,
) -> dict:
    return UnifiedDataLoader(**kwargs).get_splits(
        dataset_name,
        test_size,
        val_size,
        include_domain_id=include_domain_id,
    )


def list_available_datasets() -> dict[str, DatasetConfig]:
    seen, unique = set(), {}
    for name, config in DATASET_CONFIGS.items():
        if config.name not in seen:
            unique[name] = config
            seen.add(config.name)
    return unique
