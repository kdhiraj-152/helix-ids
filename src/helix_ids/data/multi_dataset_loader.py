"""
Multi-Dataset Loader for HelixIDS-Full.

Loads NSL-KDD, UNSW-NB15, and CICIDS datasets, harmonizes them to a common
feature space, and prepares leakage-aware train/val/test splits.

Returns train-only-fitted preprocessing artifacts and split tensors.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .learnability_contract import (
    build_meta,
    compute_contract_metrics,
    compute_schema_hash,
    compute_stage_diagnostics,
    freeze_snapshot_if_valid,
    load_reference_profile_bundle,
    write_meta,
    write_reference_profile,
)

from .feature_harmonization import (
    CICIDS2018_TO_7CLASS,
    CICIDS_TO_7CLASS,
    COMMON_FEATURES,
    INVARIANT_FEATURES,
    LEAKAGE_PRONE_FEATURES,
    NSLKDD_TO_7CLASS,
    UNSW_TO_7CLASS,
    create_cicids_mapping,
    create_nslkdd_mapping,
    create_unsw_mapping,
    harmonize_features,
    normalize_column_name,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Multi-Dataset Loader
# ============================================================================


class MultiDatasetLoader:
    """Loads and harmonizes all 3 IDS datasets for unified training."""

    TRAIN_FILE = "train.csv"

    def __init__(self, project_root: Optional[Path] = None, random_state: int = 42):
        """
        Initialize loader.

        Args:
            project_root: Root path to RP-2 project (auto-detected if None)
            random_state: For reproducible splits
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent.parent  # src/helix_ids/data -> RP-2

        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data"
        self.processed_dir = self.data_dir / "processed"
        self.random_state = random_state

        logger.info(f"Initialized MultiDatasetLoader with project_root={self.project_root}")

    def load_nslkdd(self) -> pd.DataFrame:
        """Load NSL-KDD dataset."""
        # Try multiple paths
        paths = [
            self.processed_dir / "nsl-kdd_cleaned.csv",
            self.data_dir / "nsl_kdd" / self.TRAIN_FILE,
            self.data_dir / "nsl_kdd_5class" / self.TRAIN_FILE,
        ]

        for path in paths:
            if path.exists():
                logger.info(f"Loading NSL-KDD from {path}")
                df = pd.read_csv(path, low_memory=False)
                # Add attack_type if not present
                if "attack_type" not in df.columns and "label" in df.columns:
                    df.rename(columns={"label": "attack_type"}, inplace=True)
                return df

        raise FileNotFoundError(f"NSL-KDD not found in any of {paths}")

    def load_unsw(self) -> pd.DataFrame:
        """Load UNSW-NB15 dataset."""
        paths = [
            self.processed_dir / "unsw-nb15_cleaned.csv",
            self.data_dir / "unsw_nb15" / self.TRAIN_FILE,
        ]

        for path in paths:
            if path.exists():
                logger.info(f"Loading UNSW-NB15 from {path}")
                df = pd.read_csv(path, low_memory=False)
                # Ensure attack_cat column exists
                if "attack_cat" in df.columns and "attack_type" not in df.columns:
                    df.rename(columns={"attack_cat": "attack_type"}, inplace=True)
                return df

        raise FileNotFoundError(f"UNSW-NB15 not found in any of {paths}")

    def load_cicids(self, year: int = 2017) -> Optional[pd.DataFrame]:
        """Load CICIDS dataset, preferring full CICIDS-2018 day-wise files."""
        # Fast path: pre-cleaned combined file
        cached_paths = [
            self.processed_dir / "cicids2018_cleaned.csv",
            self.processed_dir / "cicids_cleaned.csv",
        ]
        for cached in cached_paths:
            if cached.exists():
                logger.info(f"Loading cached CICIDS from {cached}")
                return self._clean_cicids_frame(pd.read_csv(cached, low_memory=False))

        # Directory detection for CICIDS 2018 upload (10 day-wise files)
        search_dirs = [
            self.project_root / "CICDS2018",
            self.project_root / "cicds.2018",
            self.project_root / "cicds2018",
            self.data_dir / "cicids2018",
            self.data_dir / "CICDS2018",
            self.project_root / "archive-2",
            self.project_root / "archive",
        ]

        daywise_files: list[Path] = []
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            csv_files = sorted(search_dir.glob("*.csv"))
            if csv_files:
                daywise_files = csv_files
                logger.info(f"Found {len(daywise_files)} CICIDS CSV files in {search_dir}")
                break

        if not daywise_files:
            logger.warning(f"CICIDS dataset not found (year={year})")
            return None

        frames: list[pd.DataFrame] = []
        required_columns = self._required_cicids_columns()
        for csv_path in daywise_files:
            logger.info(f"Reading CICIDS file: {csv_path.name}")
            df_part = pd.read_csv(
                csv_path,
                low_memory=False,
                usecols=lambda c: normalize_column_name(c) in required_columns,
            )
            df_part = self._clean_cicids_frame(df_part)
            frames.append(df_part)

        combined = pd.concat(frames, ignore_index=True)
        logger.info(f"Combined CICIDS shape: {combined.shape}")

        # Cache cleaned combined file for repeatability and speed
        cache_output = self.processed_dir / "cicids2018_cleaned.csv"
        cache_output.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(cache_output, index=False)
        logger.info(f"Saved cleaned CICIDS cache to {cache_output}")

        return combined

    def _required_cicids_columns(self) -> set[str]:
        """Build normalized required column set for CICIDS feature extraction."""
        mapping = create_cicids_mapping()
        required = {"label", "attack type", "attack_type"}
        for value in mapping.feature_mapping.values():
            candidates = value if isinstance(value, list) else [value]
            for candidate in candidates:
                required.add(normalize_column_name(candidate))
        return required

    def _clean_cicids_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize CICIDS columns/labels and coerce numerics without imputation."""
        df = df.copy()

        # Fix column inconsistencies (leading/trailing spaces and odd spacing)
        df.columns = [str(col).strip() for col in df.columns]

        normalized_to_original = {normalize_column_name(col): col for col in df.columns}
        label_col = normalized_to_original.get("label")
        if label_col is None:
            label_col = normalized_to_original.get("attack type") or normalized_to_original.get(
                "attack_type"
            )
        if label_col is None:
            raise ValueError("CICIDS frame has no label column after normalization")

        # Standardize label column name and values
        df.rename(columns={label_col: "attack_type"}, inplace=True)
        df["attack_type"] = (
            df["attack_type"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        )

        # Replace infinities and preserve NaNs for post-split train-only imputation.
        numeric_cols = [col for col in df.columns if col != "attack_type"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        return df

    def harmonize_nslkdd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Harmonize NSL-KDD to the common invariant feature space."""
        mapping = create_nslkdd_mapping()
        harmonized = harmonize_features(df, mapping)

        # Map attack types to 7-class
        if "label" in harmonized.columns:
            # Support both 5-class labels (e.g. DoS/Probe/R2L/U2R) and binary
            # variants (e.g. normal/anomaly) seen in some cleaned NSL-KDD exports.
            normalized_labels = harmonized["label"].astype(str).str.strip().str.lower()
            label_map = {str(key).strip().lower(): value for key, value in NSLKDD_TO_7CLASS.items()}
            label_map.update(
                {
                    "normal": 0,
                    "benign": 0,
                    "anomaly": 1,
                    "attack": 1,
                    "malicious": 1,
                }
            )
            harmonized["label"] = normalized_labels.map(label_map).fillna(0).astype(int)

        return harmonized

    def harmonize_unsw(self, df: pd.DataFrame) -> pd.DataFrame:
        """Harmonize UNSW-NB15 to the common invariant feature space."""
        mapping = create_unsw_mapping()
        harmonized = harmonize_features(df, mapping)

        # Map attack types to 7-class
        if "label" in harmonized.columns:
            harmonized["label"] = harmonized["label"].map(UNSW_TO_7CLASS).fillna(0).astype(int)

        return harmonized

    def harmonize_cicids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Harmonize CICIDS to the common invariant feature space."""
        mapping = create_cicids_mapping()
        harmonized = harmonize_features(df, mapping)

        # Map attack types to 7-class
        if "label" in harmonized.columns:
            normalized_labels = (
                harmonized["label"]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .str.upper()
            )
            label_map = {
                **{str(k).upper(): v for k, v in CICIDS_TO_7CLASS.items()},
                **{str(k).upper(): v for k, v in CICIDS2018_TO_7CLASS.items()},
            }
            harmonized["label"] = normalized_labels.map(label_map).fillna(0).astype(int)

        return harmonized

    def _safe_stratify(self, y: np.ndarray, dataset_name: str) -> np.ndarray:
        """Return stratify labels only when every class has >=2 samples.

        CRITICAL: This must never return None. Stratification is mandatory to ensure
        both classes are represented in train/val/test splits.
        """
        class_counts = pd.Series(y).value_counts()
        if class_counts.min() < 2:
            logger.error(
                f"{dataset_name}: Class imbalance too extreme for stratification!\n"
                f"  Class distribution: {class_counts.to_dict()}\n"
                f"  Min class count: {class_counts.min()} < 2\n"
                f"  This indicates corrupted data or label mismapping."
            )
            raise ValueError(
                f"{dataset_name}: Cannot stratify - dataset has < 2 samples in some class. "
                f"Check label mapping and data integrity."
            )
        return y

    def _compute_class_weights(self, y: np.ndarray) -> np.ndarray:
        """Compute inverse-frequency class weights for imbalance-aware training."""
        counts = np.bincount(y.astype(int))
        counts = np.where(counts == 0, 1, counts)
        weights = counts.sum() / (len(counts) * counts)
        return np.asarray(weights, dtype=np.float32)

    def _entity_group_series(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Return entity grouping key when real flow/session identity is available."""
        normalized_to_original = {normalize_column_name(col): col for col in df.columns}
        candidate_keys = [
            "flow_id",
            "flow id",
            "session",
            "session_id",
            "connection_id",
            "src_ip",
            "src ip",
        ]
        for key in candidate_keys:
            normalized_key = normalize_column_name(key)
            if normalized_key in normalized_to_original:
                col = normalized_to_original[normalized_key]
                return df[col].astype(str)
        return None

    def _get_audited_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Select invariant features and remove known shortcut-prone proxies."""
        available = [feature for feature in COMMON_FEATURES if feature in df.columns]
        selected = [feature for feature in available if feature in INVARIANT_FEATURES]
        removed = [feature for feature in available if feature not in selected]
        if removed:
            logger.info(
                "Feature audit dropped leakage-prone columns: "
                f"{sorted(set(removed) & LEAKAGE_PRONE_FEATURES)}"
            )
        if not selected:
            raise ValueError("No invariant feature columns available after audit")
        return selected

    def load_and_harmonize_all(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and harmonize all 3 datasets.

        Returns:
            (nsl_kdd_df, unsw_df, cicids_df) - each with common features + label
        """
        logger.info("Loading NSL-KDD...")
        nsl_kdd = self.load_nslkdd()
        nsl_kdd = self.harmonize_nslkdd(nsl_kdd)
        logger.info(f"  → NSL-KDD: {len(nsl_kdd)} samples, shape {nsl_kdd.shape}")

        logger.info("Loading UNSW-NB15...")
        unsw = self.load_unsw()
        unsw = self.harmonize_unsw(unsw)
        logger.info(f"  → UNSW-NB15: {len(unsw)} samples, shape {unsw.shape}")

        cicids = None
        logger.info("Loading CICIDS...")
        cicids_raw = self.load_cicids()
        if cicids_raw is not None:
            cicids = self.harmonize_cicids(cicids_raw)
            logger.info(f"  → CICIDS: {len(cicids)} samples, shape {cicids.shape}")
        else:
            logger.warning("  → CICIDS not available, continuing with 2 datasets")

        return nsl_kdd, unsw, cicids

    def create_splits(  # NOSONAR
        self,
        dfs: list[pd.DataFrame],
        test_size: float = 0.15,
        val_size: float = 0.15,
        holdout_dataset: Optional[str] = "cicids",
    ) -> dict[str, np.ndarray]:
        """
        Create train/val/test splits with split-first preprocessing.

        Args:
            dfs: List of harmonized DataFrames
            test_size: Fraction for test set
            val_size: Fraction for validation set (of remaining after train/test split)

        Returns:
            Dict with combined and per-dataset split arrays.
        """
        dataset_names = ["nsl_kdd", "unsw_nb15", "cicids"]
        named_dfs = [(dataset_names[idx], df) for idx, df in enumerate(dfs) if df is not None]
        if not named_dfs:
            raise ValueError("No datasets provided for split creation")

        entity_available = any(self._entity_group_series(df) is not None for _, df in named_dfs)
        active_names = {name for name, _ in named_dfs}
        holdout_name: Optional[str] = None
        if not entity_available and len(named_dfs) >= 3:
            candidate = holdout_dataset if holdout_dataset in active_names else named_dfs[-1][0]
            holdout_name = candidate
            logger.warning(
                "No entity keys detected (flow_id/src_ip/session). "
                f"Enforcing dataset-level holdout on '{holdout_name}'."
            )

        raw_splits: dict[str, dict[str, np.ndarray]] = {}
        selected_feature_columns: Optional[list[str]] = None
        val_ratio = val_size / (1 - test_size)

        for dataset_name, df in named_dfs:
            logger.info(f"Creating splits for {dataset_name}...")
            feature_columns = self._get_audited_feature_columns(df)
            if selected_feature_columns is None:
                selected_feature_columns = feature_columns

            x_all = df[feature_columns].to_numpy(dtype=np.float32, copy=False)
            y_all = df["label"].to_numpy(dtype=np.int64, copy=False)

            if holdout_name == dataset_name:
                raw_splits[dataset_name] = {
                    "X_train": np.empty((0, x_all.shape[1]), dtype=np.float32),
                    "y_train": np.empty((0,), dtype=np.int64),
                    "X_val": np.empty((0, x_all.shape[1]), dtype=np.float32),
                    "y_val": np.empty((0,), dtype=np.int64),
                    "X_test": x_all,
                    "y_test": y_all,
                }
                logger.info(
                    f"  {dataset_name}: assigned as holdout test-only split ({x_all.shape[0]:,} rows)"
                )
                continue

            groups = self._entity_group_series(df)
            if groups is not None:
                group_values = groups.to_numpy()
                outer_split = GroupShuffleSplit(
                    n_splits=1,
                    test_size=test_size,
                    random_state=self.random_state,
                )
                train_idx, test_idx = next(outer_split.split(x_all, y_all, groups=group_values))
                x_train_val = x_all[train_idx]
                y_train_val = y_all[train_idx]
                x_test = x_all[test_idx]
                y_test = y_all[test_idx]

                inner_groups = group_values[train_idx]
                inner_split = GroupShuffleSplit(
                    n_splits=1,
                    test_size=val_ratio,
                    random_state=self.random_state,
                )
                sub_train_idx, val_idx = next(
                    inner_split.split(x_train_val, y_train_val, groups=inner_groups)
                )
                x_train = x_train_val[sub_train_idx]
                y_train = y_train_val[sub_train_idx]
                x_val = x_train_val[val_idx]
                y_val = y_train_val[val_idx]
            else:
                stratify_outer = self._safe_stratify(y_all, f"{dataset_name}-train-test")
                x_train_val, x_test, y_train_val, y_test = train_test_split(
                    x_all,
                    y_all,
                    test_size=test_size,
                    random_state=self.random_state,
                    stratify=stratify_outer,
                )
                stratify_inner = self._safe_stratify(y_train_val, f"{dataset_name}-train-val")
                x_train, x_val, y_train, y_val = train_test_split(
                    x_train_val,
                    y_train_val,
                    test_size=val_ratio,
                    random_state=self.random_state,
                    stratify=stratify_inner,
                )

            raw_splits[dataset_name] = {
                "X_train": x_train,
                "y_train": y_train,
                "X_val": x_val,
                "y_val": y_val,
                "X_test": x_test,
                "y_test": y_test,
            }

            logger.info(
                f"  {dataset_name}: train {x_train.shape[0]:,}, "
                f"val {x_val.shape[0]:,}, test {x_test.shape[0]:,}"
            )

        if selected_feature_columns is None:
            raise ValueError("Unable to determine feature columns for preprocessing")

        train_blocks = [parts["X_train"] for parts in raw_splits.values() if parts["X_train"].size > 0]
        if not train_blocks:
            raise ValueError("No training samples available after split construction")

        def _transform(x_raw: np.ndarray) -> np.ndarray:
            if x_raw.size == 0:
                return np.empty((0, len(selected_feature_columns)), dtype=np.float32)
            return np.nan_to_num(
                np.asarray(x_raw, dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        transformed_splits: dict[str, np.ndarray] = {}
        combined_train_x: list[np.ndarray] = []
        combined_train_y: list[np.ndarray] = []
        combined_val_x: list[np.ndarray] = []
        combined_val_y: list[np.ndarray] = []

        for dataset_name, parts in raw_splits.items():
            x_train = _transform(parts["X_train"])
            x_val = _transform(parts["X_val"])
            x_test = _transform(parts["X_test"])

            y_train = parts["y_train"].astype(np.int64, copy=False)
            y_val = parts["y_val"].astype(np.int64, copy=False)
            y_test = parts["y_test"].astype(np.int64, copy=False)

            if dataset_name == "unsw_nb15" and x_train.shape[0] > 0:
                pre_transform = np.nan_to_num(
                    np.asarray(parts["X_train"], dtype=np.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                stage_snapshots = {
                    "pre_transform": pre_transform,
                    "split_then_nan_to_num": x_train,
                }
                stage_diag = compute_stage_diagnostics(
                    stage_snapshots=stage_snapshots,
                    y_train=y_train,
                    feature_names=[str(col) for col in selected_feature_columns],
                    random_seed=self.random_state,
                )
                transition = stage_diag["transitions"].get("pre_transform->split_then_nan_to_num", {})
                f1_ratio = float(transition.get("f1_ratio", 1.0))
                shrink_ratio = float(transition.get("centroid_shrinkage_ratio", 1.0))
                logger.info(
                    "UNSW stage transition pre_transform->split_then_nan_to_num "
                    "f1_ratio=%.4f centroid_shrinkage_ratio=%.4f",
                    f1_ratio,
                    shrink_ratio,
                )
                if f1_ratio < 0.90:
                    raise RuntimeError(
                        "Fail-fast learnability guard: macro-F1 dropped >10% after transformation "
                        "'split_then_nan_to_num'"
                    )
                if shrink_ratio < 0.30:
                    raise RuntimeError(
                        "Fail-fast scaling guard: centroid distance shrinkage ratio below 0.30 "
                        "after 'split_then_nan_to_num'"
                    )
                transformed_splits["diagnostic_unsw_pre_transform"] = pre_transform

            transformed_splits[f"X_train_{dataset_name}"] = x_train
            transformed_splits[f"y_train_{dataset_name}"] = y_train
            transformed_splits[f"X_val_{dataset_name}"] = x_val
            transformed_splits[f"y_val_{dataset_name}"] = y_val
            transformed_splits[f"X_test_{dataset_name}"] = x_test
            transformed_splits[f"y_test_{dataset_name}"] = y_test

            if x_train.size > 0:
                combined_train_x.append(x_train)
                combined_train_y.append(y_train)
            if x_val.size > 0:
                combined_val_x.append(x_val)
                combined_val_y.append(y_val)

        x_train_combined = np.vstack(combined_train_x)
        y_train_combined = np.hstack(combined_train_y)
        x_val_combined = (
            np.vstack(combined_val_x)
            if combined_val_x
            else np.empty((0, len(selected_feature_columns)), dtype=np.float32)
        )
        y_val_combined = (
            np.hstack(combined_val_y) if combined_val_y else np.empty((0,), dtype=np.int64)
        )

        train_rng = np.random.default_rng(self.random_state)
        train_order = train_rng.permutation(x_train_combined.shape[0])
        x_train_combined = x_train_combined[train_order]
        y_train_combined = y_train_combined[train_order]

        if x_val_combined.shape[0] > 0:
            val_rng = np.random.default_rng(self.random_state + 1)
            val_order = val_rng.permutation(x_val_combined.shape[0])
            x_val_combined = x_val_combined[val_order]
            y_val_combined = y_val_combined[val_order]

        splits: dict[str, np.ndarray] = {
            "X_train": x_train_combined,
            "y_train": y_train_combined,
            "X_val": x_val_combined,
            "y_val": y_val_combined,
            "train_class_weights": self._compute_class_weights(y_train_combined),
            "feature_columns": np.array(selected_feature_columns, dtype=object),
        }
        splits.update(transformed_splits)

        logger.info(
            "Final combined splits: "
            f"train {x_train_combined.shape[0]:,}, val {x_val_combined.shape[0]:,}"
        )

        return splits

    def save_processed_data(self, output_dir: Optional[Path] = None):
        """Save processed datasets to disk for checkpointing."""
        if output_dir is None:
            output_dir = self.processed_dir / "multi_dataset_v1"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load all datasets
        nsl_kdd, unsw, cicids = self.load_and_harmonize_all()

        # Create splits
        splits = self.create_splits(
            [nsl_kdd, unsw, cicids] if cicids is not None else [nsl_kdd, unsw]
        )

        # Save splits as numpy arrays
        for key, arr in splits.items():
            if key.startswith(("X_", "y_")):
                np.save(output_dir / f"{key}.npy", arr)
                logger.info(f"Saved {key} -> {output_dir / f'{key}.npy'}")

        feature_columns = np.asarray(splits.get("feature_columns", np.array([], dtype=object)))
        if feature_columns.size == 0:
            raise RuntimeError("Missing feature_columns in split artifact; cannot validate schema")
        np.save(output_dir / "feature_columns.npy", feature_columns)

        x_unsw = splits.get("X_train_unsw_nb15")
        y_unsw = splits.get("y_train_unsw_nb15")
        if x_unsw is None or y_unsw is None:
            raise RuntimeError("Missing UNSW train splits required for learnability contract")

        transformations = ["split_then_nan_to_num"]
        feature_names = [str(col) for col in feature_columns.tolist()]
        schema_hash = compute_schema_hash(
            feature_columns=feature_names,
            transformations=transformations,
        )
        unsw_mapping = create_unsw_mapping().feature_mapping
        feature_lineage = {
            f"f_{idx}": ",".join(unsw_mapping.get(feature_name, [feature_name]))
            for idx, feature_name in enumerate(feature_names)
        }
        raw_unsw_train = np.nan_to_num(
            np.asarray(splits.get("diagnostic_unsw_pre_transform", x_unsw), dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        metrics = compute_contract_metrics(
            x_train=np.asarray(x_unsw, dtype=np.float32),
            y_train=np.asarray(y_unsw, dtype=np.int64),
            dataset="unsw_nb15",
            schema_hash=schema_hash,
            feature_names=feature_names,
            feature_lineage=feature_lineage,
            stage_snapshots={
                "pre_transform": raw_unsw_train,
                "split_then_nan_to_num": np.asarray(x_unsw, dtype=np.float32),
            },
            random_seed=self.random_state,
        )
        profile_bundle = load_reference_profile_bundle(
            artifact_dir=output_dir,
            dataset_signature="unsw",
        )
        metrics["reference_profile"] = profile_bundle["profile"]
        metrics["expected_reference_profile_version"] = profile_bundle["profile"].get("version")
        meta = build_meta(metrics)
        write_meta(meta, artifact_dir=output_dir)
        profile_bundle["payload"]["reference_profiles"][profile_bundle["profile_key"]] = meta["reference_profile"]
        write_reference_profile(profile_bundle["payload"], artifact_dir=output_dir)
        meta = freeze_snapshot_if_valid(artifact_dir=output_dir)
        meta_path = output_dir / "meta.json"
        logger.info(f"Saved UNSW learnability contract -> {meta_path}")

        if not bool(meta.get("validated", False)):
            raise RuntimeError(
                "UNSW learnability contract failed during preprocessing. "
                f"violations={meta.get('violations', {})}"
            )

        logger.info(f"✅ Processed datasets saved to {output_dir}")

        return splits

    def _downsample_majority_class(
        self,
        x: np.ndarray,
        y: np.ndarray,
        max_majority_ratio: float = 0.90,
        random_state: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Downsample majority class to achieve balanced representation.
        
        Args:
            x: Feature array (n_samples, n_features)
            y: Label array (n_samples,)
            max_majority_ratio: Target max ratio for majority class (e.g., 0.90 = 90%)
            random_state: Random seed for reproducibility
            
        Returns:
            (x_balanced, y_balanced): Downsampled feature and label arrays
        """
        if random_state is None:
            random_state = self.random_state
            
        rng = np.random.default_rng(random_state)
        unique_classes, counts = np.unique(y, return_counts=True)
        
        if len(unique_classes) <= 1:
            return x, y
            
        # Identify majority and minority classes
        majority_class = unique_classes[np.argmax(counts)]
        target_ratio = max_majority_ratio
        total_samples = len(y)
        
        # Calculate how many majority samples we should keep
        majority_mask = y == majority_class
        majority_count = np.sum(majority_mask)
        minority_count = total_samples - majority_count
        
        # We want: majority_count_new / total_new = target_ratio
        # So: majority_count_new / (majority_count_new + minority_count) = target_ratio
        # Solving: majority_count_new = target_ratio * minority_count / (1 - target_ratio)
        target_majority_count = int(target_ratio * minority_count / (1.0 - target_ratio))
        target_majority_count = max(1, min(target_majority_count, majority_count))
        
        # Randomly select which majority samples to keep
        majority_indices = np.nonzero(majority_mask)[0]
        selected_majority_indices = rng.choice(
            majority_indices,
            size=target_majority_count,
            replace=False,
        )
        
        # Combine with all minority samples
        minority_indices = np.nonzero(~majority_mask)[0]
        selected_indices = np.concatenate([selected_majority_indices, minority_indices])
        selected_indices = np.sort(selected_indices)  # Maintain order
        
        return x[selected_indices], y[selected_indices]

    def _fingerprint_rows(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Create deterministic fingerprints for rows to detect duplicates.
        
        Uses a combination of feature hash and label to create a unique identifier.
        
        Args:
            x: Feature array (n_samples, n_features)
            y: Label array (n_samples,)
            
        Returns:
            fingerprints: Hash array of shape (n_samples,)
        """
        # Round to 4 decimal places for numerical stability
        x_rounded = np.round(x, decimals=4)
        
        # Create fingerprints by concatenating rounded features with label
        fingerprints = []
        for i in range(len(x)):
            # Convert row to tuple for hashing
            row_tuple = tuple(x_rounded[i]) + (int(y[i]),)
            # Use Python's hash for a simple fingerprint
            fp = hash(row_tuple)
            fingerprints.append(fp)
            
        return np.array(fingerprints, dtype=np.int64)

    def _remove_cross_split_overlap(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        dataset_name: str = "unknown",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Remove duplicate rows between train/val/test splits.
        
        Ensures that no same row appears in multiple splits (data leakage prevention).
        
        Args:
            x_train, y_train: Training set
            x_val, y_val: Validation set
            x_test, y_test: Test set
            dataset_name: Name for logging
            
        Returns:
            (x_train, y_train, x_val_clean, y_val_clean, x_test_clean, y_test_clean):
                Training set unchanged, validation and test cleaned
        """
        train_fp = self._fingerprint_rows(x_train, y_train)
        val_fp = self._fingerprint_rows(x_val, y_val)
        test_fp = self._fingerprint_rows(x_test, y_test)
        
        # Find indices in val that don't exist in train
        val_mask = ~np.isin(val_fp, train_fp)
        x_val_clean = x_val[val_mask]
        y_val_clean = y_val[val_mask]
        
        # Find indices in test that don't exist in train or val
        test_mask = ~(np.isin(test_fp, train_fp) | np.isin(test_fp, val_fp[val_mask]))
        x_test_clean = x_test[test_mask]
        y_test_clean = y_test[test_mask]
        
        # Log statistics
        logger.info(
            f"{dataset_name}: Removed {len(y_val) - len(y_val_clean)} val duplicates, "
            f"{len(y_test) - len(y_test_clean)} test duplicates"
        )
        
        return x_train, y_train, x_val_clean, y_val_clean, x_test_clean, y_test_clean

    def _build_group_keys(
        self,
        x: np.ndarray,
        y: np.ndarray,
        dataset_code: int = 0,
    ) -> np.ndarray:
        """Build group/session keys using coarse fingerprinting.
        
        Rows with similar features are grouped together (coarse fingerprint).
        Useful for GroupShuffleSplit to prevent inter-group leakage.
        
        Args:
            x: Feature array (n_samples, n_features)
            y: Label array (n_samples,)
            dataset_code: Integer code for the dataset (0=NSL-KDD, 1=UNSW, 2=CICIDS)
            
        Returns:
            group_keys: Array of group identifiers (n_samples,)
        """
        # Coarse fingerprint: round to fewer decimal places
        x_coarse = np.round(x, decimals=1)
        
        group_keys = []
        for i in range(len(x)):
            row_tuple = tuple(x_coarse[i]) + (int(y[i]),) + (int(dataset_code),)
            group_key = hash(row_tuple)
            group_keys.append(group_key)
            
        return np.array(group_keys, dtype=np.int64)


# ============================================================================
# Utility Functions
# ============================================================================


def load_processed_splits(data_dir: Path) -> dict[str, np.ndarray]:
    """Load pre-processed splits from disk."""
    splits = {}
    for npy_file in data_dir.glob("*.npy"):
        key = npy_file.stem
        splits[key] = np.load(npy_file)
    return splits


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    loader = MultiDatasetLoader()
    nsl_kdd, unsw, cicids = loader.load_and_harmonize_all()

    print("\n=== Harmonized Datasets ===")
    print(f"NSL-KDD: {nsl_kdd.shape}")
    print(f"  Columns: {list(nsl_kdd.columns)}")
    print(f"UNSW-NB15: {unsw.shape}")
    print(f"  Columns: {list(unsw.columns)}")
    if cicids is not None:
        print(f"CICIDS: {cicids.shape}")
        print(f"  Columns: {list(cicids.columns)}")

    print("\n=== Creating Splits ===")
    splits = loader.create_splits(
        [nsl_kdd, unsw, cicids] if cicids is not None else [nsl_kdd, unsw]
    )
    for key in sorted(splits.keys()):
        print(f"{key}: {splits[key].shape}")
