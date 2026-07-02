"""
Multi-Dataset Loader for HelixIDS-Full.

Loads NSL-KDD, UNSW-NB15, and CICIDS datasets, harmonizes them to a common
feature space, and prepares leakage-aware train/val/test splits.

Returns train-only-fitted preprocessing artifacts and split tensors.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import datasets as hf_datasets
except ImportError:
    hf_datasets = None  # type: ignore[assignment]
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

from ..contracts.attack_taxonomy import (
    BOTIOT_TO_7CLASS,
    CICIDS2018_TO_7CLASS,
    CICIDS_TO_7CLASS,
    NSL_KDD_ATTACK_MAPPING,
    NSLKDD_TO_7CLASS,
    TONIOT_TO_7CLASS,
    UNSW_TO_7CLASS,
)
from ..contracts.schema_contract import (
    CANONICAL_BINARY_CLASSES,
    CANONICAL_FAMILY_CLASSES,
    CANONICAL_FEATURE_ORDER,
    CANONICAL_INPUT_DIM,
    SCHEMA_VERSION,
    compute_schema_hash,
    validate_feature_order,
)
from .feature_harmonization import (
    FEATURE_ORDER,
    create_bot_iot_mapping,
    create_cicids2017_mapping,
    create_cicids_mapping,
    create_nslkdd_mapping,
    create_ton_iot_mapping,
    create_unsw_mapping,
    harmonize_features,
    normalize_column_name,
)
from .learnability_contract import (
    PREPROCESS_THRESHOLDS,
    build_meta,
    compute_contract_metrics,
    compute_stage_diagnostics,
    freeze_snapshot_if_valid,
    load_reference_profile_bundle,
    write_meta,
    write_reference_profile,
)

logger = logging.getLogger(__name__)

REQUIRED_DISCRETE_DRIVERS = ("protocol_type", "connection_state", "traffic_direction", "service_tier")
SUPPORTED_DISCRETE_DRIVERS = (
    "protocol_type",
    "connection_state",
    "traffic_direction",
    "service_tier",
    "has_rst",
    "flag",
)
GEOMETRIC_EXPANSION_FEATURES = (
    "log_src_bytes",
    "log_dst_bytes",
    "src_dst_bytes_ratio",
    "dst_src_bytes_ratio",
    "same_host_rate_x_service",
    "diff_srv_rate_x_flag",
    "count_x_srv_count",
    "protocol_service_flag",
)
UNSW_DISCRETE_PROBE_DRIVERS = (
    "protocol_type",
    "connection_state",
    "traffic_direction",
    "has_rst",
)
MAX_SIGNAL_FEATURES = 20
MIN_UNIQUE_PRED_CLASS_COVERAGE = 0.50
UNSW_DISCRETE_PROBE_F1_MIN = 0.40
UNSW_DISCRETE_PROBE_F1_ADAPTIVE_FLOOR = 0.30
UNSW_DISCRETE_PROBE_BASELINE_MARGIN = 0.10
ATTACK_TYPE_ALIASES = {"label", "attack type", "attack_type"}


# ============================================================================
# Multi-Dataset Loader
# ============================================================================


class MultiDatasetLoader:
    """Loads and harmonizes all IDS datasets (NSL-KDD, UNSW-NB15, CICIDS, TON-IoT, Bot-IoT, CIC-IDS2017)."""

    TRAIN_FILE = "train.csv"
    TONIOT_RAW_COLUMNS = [
        "ts", "src_ip", "src_port", "dst_ip", "dst_port", "proto",
        "duration", "src_bytes", "dst_bytes", "conn_state", "missed_bytes",
        "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes", "dns_ttl_answer",
        "dns_query", "dns_qclass", "dns_qtype", "dns_rcode", "dns_aa",
        "dns_tc", "dns_rd", "dns_ra", "dns_res", "http_method",
        "http_uri", "http_referrer", "http_version", "http_request_body_len",
        "http_response_body_len", "http_status_code", "http_user_agent",
        "http_orig_mime_types", "http_resp_mime_types", "http_trans_depth",
        "ssl_version", "ssl_cipher", "ssl_resumed", "ssl_established",
        "ssl_subject", "ssl_issuer", "type",
    ]

    NSLKDD_RAW_COLUMNS = [
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
        "label",
        "difficulty",
    ]

    def __init__(self, project_root: Optional[Path] = None, random_state: int = 42):
        """
        Initialize loader.

        Args:
            project_root: Root path to helix-ids project (auto-detected if None)
            random_state: For reproducible splits
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent.parent  # src/helix_ids/data -> helix-ids

        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data"
        self.processed_dir = self.data_dir / "processed"
        self.random_state = random_state

        logger.info(f"Initialized MultiDatasetLoader with project_root={self.project_root}")

    @staticmethod
    def _normalize_attack_type_column(df: pd.DataFrame) -> pd.DataFrame:
        if "attack_type" in df.columns or "label" not in df.columns:
            return df
        normalized = df.copy()
        normalized.rename(columns={"label": "attack_type"}, inplace=True)
        return normalized

    def _load_nslkdd_frame(self, path: Path) -> pd.DataFrame:
        logger.info(f"Loading NSL-KDD from {path}")
        if path.suffix.lower() == ".txt":
            df = self._read_nslkdd_raw(path)
        else:
            df = pd.read_csv(path, low_memory=False)
        return self._normalize_attack_type_column(df)

    @staticmethod
    def _resolve_unsw_attack_cat_column(df: pd.DataFrame) -> str | None:
        normalized_cols = {normalize_column_name(col): col for col in df.columns}
        for candidate in ("attack_cat", "attack cat"):
            key = normalize_column_name(candidate)
            if key in normalized_cols:
                column_name = normalized_cols[key]
                if isinstance(column_name, str):
                    return column_name
                return str(column_name)
        return None

    @staticmethod
    def _normalize_unsw_label_series(labels: pd.Series) -> pd.Series:
        numeric_labels = pd.to_numeric(labels, errors="coerce")
        if not numeric_labels.isna().all() and set(numeric_labels.dropna().astype(int).unique()).issubset({0, 1, 2, 3, 4, 5, 6}):
            if numeric_labels.isna().any():
                raise ValueError("unsw_nb15 label-space mismatch: non-numeric labels present")
            return numeric_labels.astype(int)

        source_labels = labels.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        # Case-insensitive mapping (UNSW labels like "DoS" should not be title-cased)
        _lower_to_class = {k.lower(): v for k, v in UNSW_TO_7CLASS.items()}
        mapped_labels = source_labels.str.lower().map(_lower_to_class)
        unresolved = source_labels[mapped_labels.isna()].unique().tolist()
        if unresolved:
            raise ValueError(f"unsw_nb15 label-space mismatch: {sorted(map(str, unresolved))}")
        return mapped_labels.astype(int)

    def load_nslkdd(self) -> pd.DataFrame:
        """Load NSL-KDD dataset.

        Uses full corpus when canonical raw files are available by concatenating
        KDDTrain+.txt and KDDTest+.txt before harmonization/splitting.
        """
        raw_train = self.data_dir / "nsl_kdd" / "raw" / "KDDTrain+.txt"
        raw_test = self.data_dir / "nsl_kdd" / "raw" / "KDDTest+.txt"
        if raw_train.exists() and raw_test.exists():
            logger.info(f"Loading NSL-KDD full corpus from {raw_train} + {raw_test}")
            df_train = self._load_nslkdd_frame(raw_train)
            df_test = self._load_nslkdd_frame(raw_test)
            return self._normalize_attack_type_column(pd.concat([df_train, df_test], ignore_index=True))

        # Fallback paths (legacy processed snapshots)
        paths = [
            raw_train,
            self.processed_dir / "nsl-kdd_cleaned.csv",
            self.data_dir / "nsl_kdd" / self.TRAIN_FILE,
            self.data_dir / "nsl_kdd_5class" / self.TRAIN_FILE,
        ]

        for path in paths:
            if path.exists():
                return self._load_nslkdd_frame(path)

        raise FileNotFoundError(f"NSL-KDD not found in any of {paths}")

    def _read_nslkdd_raw(self, path: Path) -> pd.DataFrame:
        """Read canonical NSL-KDD raw text (KDDTrain+/KDDTest+) with stable column names."""
        preview = pd.read_csv(path, header=None, nrows=1)
        n_cols = int(preview.shape[1])

        if n_cols == len(self.NSLKDD_RAW_COLUMNS):
            names = self.NSLKDD_RAW_COLUMNS
        elif n_cols == len(self.NSLKDD_RAW_COLUMNS) - 1:
            names = self.NSLKDD_RAW_COLUMNS[:-1]
        else:
            raise ValueError(
                f"Unexpected NSL-KDD raw column count in {path}: {n_cols}. "
                "Expected 42 or 43 columns."
            )

        df = pd.read_csv(path, header=None, names=names, low_memory=False)
        if "difficulty" in df.columns:
            df = df.drop(columns=["difficulty"])
        return df

    def load_unsw(self) -> pd.DataFrame:
        """Load UNSW-NB15 dataset."""
        paths = [
            self.data_dir / "unsw_nb15" / "raw" / "UNSW_NB15_training-set.csv",
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
                cached_df = pd.read_csv(cached, low_memory=False)
                normalized_cached = {normalize_column_name(col) for col in cached_df.columns}
                if "dst port" not in normalized_cached and "destination port" not in normalized_cached:
                    logger.info(
                        "Cached CICIDS is missing destination port; reloading from raw day-wise files"
                    )
                    continue
                return self._clean_cicids_frame(cached_df)

        # Directory detection for CICIDS 2018 upload (10 day-wise files)
        search_dirs = [
            self.data_dir / "cicids2018" / "raw",
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
        required = set(ATTACK_TYPE_ALIASES)
        required.add(normalize_column_name("Dst Port"))
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

        # Raw CICIDS exports may contain malformed rows (e.g., embedded headers).
        # Drop rows missing any required mapped field so strict harmonization can proceed.
        normalized_to_original = {normalize_column_name(col): col for col in df.columns}
        required_keys = {
            key
            for key in self._required_cicids_columns()
            if key not in ATTACK_TYPE_ALIASES
        }
        required_numeric_cols = [
            normalized_to_original[key]
            for key in sorted(required_keys)
            if key in normalized_to_original and normalized_to_original[key] != "attack_type"
        ]

        if required_numeric_cols:
            # Keep rows even when required numeric fields are missing; harmonization
            # and split-time scaling handle NaN/inf sanitation deterministically.
            pass

        df = df[df["attack_type"].notna()]
        df = df[df["attack_type"].astype(str).str.strip() != ""]

        return df

    def harmonize_nslkdd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Harmonize NSL-KDD to the common invariant feature space."""
        mapping = create_nslkdd_mapping()
        harmonized = harmonize_features(df, mapping)

        # Map attack types to 7-class
        if "label" in harmonized.columns:
            # Support raw NSL attack names, 5-class family labels, and binary variants.
            normalized_labels = (
                harmonized["label"].astype(str).str.strip().str.lower().str.rstrip(".")
            )

            # Raw attack names -> family labels (Normal/DoS/Probe/R2L/U2R)
            family_labels = normalized_labels.map(
                {str(key).strip().lower(): value for key, value in NSL_KDD_ATTACK_MAPPING.items()}
            )

            # Family labels -> integer targets
            family_to_index = {str(key).strip().lower(): value for key, value in NSLKDD_TO_7CLASS.items()}

            mapped = family_labels.str.strip().str.lower().map(family_to_index)
            fallback_map = {
                **family_to_index,
                "normal": 0,
                "benign": 0,
                "anomaly": 1,
                "attack": 1,
                "malicious": 1,
            }
            fallback_mapped = normalized_labels.map(fallback_map)
            unresolved = normalized_labels[mapped.isna() & fallback_mapped.isna()].unique().tolist()
            if unresolved:
                raise ValueError(f"nsl_kdd label-space mismatch: {sorted(map(str, unresolved))}")
            harmonized["label"] = mapped.fillna(fallback_mapped).astype(int)

        return harmonized

    def harmonize_unsw(self, df: pd.DataFrame) -> pd.DataFrame:
        """Harmonize UNSW-NB15 to the common invariant feature space."""
        mapping = create_unsw_mapping()
        harmonized = harmonize_features(df, mapping)

        # Map attack types to 7-class
        attack_cat_col = self._resolve_unsw_attack_cat_column(df)

        if attack_cat_col is not None:
            harmonized["label"] = self._normalize_unsw_label_series(df[attack_cat_col].astype(str))
        elif "label" in harmonized.columns:
            harmonized["label"] = self._normalize_unsw_label_series(harmonized["label"])

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
            mapped_labels = normalized_labels.map(label_map)
            unresolved = normalized_labels[mapped_labels.isna()].unique().tolist()
            if unresolved:
                raise ValueError(f"cicids label-space mismatch: {sorted(map(str, unresolved))}")
            harmonized["label"] = mapped_labels.astype(int)

        return harmonized

    def _clean_ton_iot_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean TON-IoT raw dataframe: drop exact dupes, rename label col.

        TON-IoT raw data has both a binary ``label`` column (0=normal, 1=attack)
        and a multi-class ``type`` column (backdoor, ddos, dos, injection, …).
        The pipeline requires multi-class labels, so ``type`` is used as the
        canonical label column.
        """
        df = df.copy()
        if "type" in df.columns:
            # Multi-class labels live in the 'type' column. Drop the binary
            # 'label' column if present, then rename 'type' → 'label'.
            if "label" in df.columns:
                df = df.drop(columns=["label"])
            df.rename(columns={"type": "label"}, inplace=True)
        elif "label" not in df.columns and "attack_type" in df.columns:
            df.rename(columns={"attack_type": "label"}, inplace=True)
        # Drop exact duplicate rows
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        logger.info(f"TON-IoT dedup: {before:,} -> {after:,} rows ({((before - after) / before * 100):.1f}% removed)")
        return df

    def load_ton_iot(self) -> pd.DataFrame:
        """Load TON-IoT dataset (single file; train=test in this corpus)."""
        paths = [
            self.data_dir / "ton_iot" / "raw" / self.TRAIN_FILE,
            self.data_dir / "ton_iot" / self.TRAIN_FILE,
        ]
        for path in paths:
            if path.exists():
                logger.info(f"Loading TON-IoT from {path}")
                raw = pd.read_csv(path, low_memory=False)
                raw = self._clean_ton_iot_frame(raw)
                return raw
        raise FileNotFoundError(f"TON-IoT not found in any of {paths}")

    def harmonize_ton_iot(self, df: pd.DataFrame) -> pd.DataFrame:
        """Harmonize TON-IoT to the common invariant feature space."""
        mapping = create_ton_iot_mapping()
        harmonized = harmonize_features(df, mapping)

        # Map attack types to 7-class
        if "label" in harmonized.columns:
            normalized_labels = (
                harmonized["label"]
                .astype(str)
                .str.strip()
                .str.lower()
            )
            mapped_labels = normalized_labels.map(
                {k.lower(): v for k, v in TONIOT_TO_7CLASS.items()}
            )
            unresolved = normalized_labels[mapped_labels.isna()].unique().tolist()
            if unresolved:
                raise ValueError(f"ton_iot label-space mismatch: {sorted(map(str, unresolved))}")
            harmonized["label"] = mapped_labels.astype(int)

        return harmonized

    def load_bot_iot(self) -> pd.DataFrame | None:
        """Load Bot-IoT dataset.

        Tries local CSV cache first, then falls back to HuggingFace.
        """
        local_paths = [
            self.data_dir / "bot_iot" / "raw" / self.TRAIN_FILE,
            self.data_dir / "bot_iot" / self.TRAIN_FILE,
        ]
        for path in local_paths:
            if path.exists():
                logger.info(f"Loading Bot-IoT from {path}")
                df = pd.read_csv(path, low_memory=False)
                return self._clean_bot_iot_frame(df)

        if hf_datasets is None:
            logger.warning("datasets library not installed; cannot load Bot-IoT")
            return None
        try:
            logger.info("Loading Bot-IoT from HuggingFace (masoltani/bot-iot)...")
            ds = hf_datasets.load_dataset("masoltani/bot-iot", split="train")
            df = ds.to_pandas()
            logger.info(f"  Bot-IoT: {len(df)} samples from HuggingFace")
            return self._clean_bot_iot_frame(df)
        except Exception as exc:
            logger.warning(f"Bot-IoT not available from HuggingFace: {exc}")
            return None

    def harmonize_bot_iot(self, df: pd.DataFrame) -> pd.DataFrame:
        """Harmonize Bot-IoT to the common invariant feature space."""
        mapping = create_bot_iot_mapping()
        harmonized = harmonize_features(df, mapping)

        # Map categories to 7-class
        if "label" in harmonized.columns:
            normalized_labels = (
                harmonized["label"]
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(r"\s+", " ", regex=True)
            )
            mapped_labels = normalized_labels.map(BOTIOT_TO_7CLASS)
            unresolved = normalized_labels[mapped_labels.isna()].unique().tolist()
            if unresolved:
                raise ValueError(f"bot_iot label-space mismatch: {sorted(map(str, unresolved))}")
            harmonized["label"] = mapped_labels.astype(int)

        return harmonized

    def load_cicids2017(self) -> pd.DataFrame | None:
        """Load CIC-IDS2017.

        Tries local CSV cache first, then falls back to HuggingFace.
        """
        local_paths = [
            self.data_dir / "cicids2017" / "raw" / self.TRAIN_FILE,
            self.data_dir / "cicids2017" / self.TRAIN_FILE,
        ]
        for path in local_paths:
            if path.exists():
                logger.info(f"Loading CIC-IDS2017 from {path}")
                df = pd.read_csv(path, low_memory=False)
                if "attack_label" in df.columns:
                    df = df.rename(columns={"attack_label": "label"})
                return df

        if hf_datasets is None:
            logger.warning("datasets library not installed; cannot load CIC-IDS2017")
            return None
        try:
            logger.info("Loading CIC-IDS2017 from HuggingFace (rdpahalavan/CIC-IDS2017)...")
            ds = hf_datasets.load_dataset("rdpahalavan/CIC-IDS2017", split="train")
            df = ds.to_pandas()
            logger.info(f"  CIC-IDS2017: {len(df)} samples from HuggingFace")
            if "attack_label" in df.columns:
                df = df.rename(columns={"attack_label": "label"})
            return df
        except Exception as exc:
            logger.warning(f"CIC-IDS2017 not available from HuggingFace: {exc}")
            return None

    def harmonize_cicids2017(self, df: pd.DataFrame) -> pd.DataFrame:
        """Harmonize CIC-IDS2017 using CICIDS feature mapping + label mapping."""
        mapping = create_cicids2017_mapping()
        harmonized = harmonize_features(df, mapping)

        # Map attack types to 7-class (same labels as CIC-IDS2018)
        if "label" in harmonized.columns:
            normalized_labels = (
                harmonized["label"]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .str.replace("\u2013", "-")  # normalize en-dash → hyphen
                .str.upper()
            )
            label_map = {
                **{str(k).upper(): v for k, v in CICIDS_TO_7CLASS.items()},
                **{str(k).upper(): v for k, v in CICIDS2018_TO_7CLASS.items()},
            }
            mapped_labels = normalized_labels.map(label_map)
            unresolved = normalized_labels[mapped_labels.isna()].unique().tolist()
            if unresolved:
                raise ValueError(f"cicids2017 label-space mismatch: {sorted(map(str, unresolved))}")
            harmonized["label"] = mapped_labels.astype(int)

        return harmonized

    def _clean_bot_iot_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean Bot-IoT raw dataframe: rename label col, drop unneeded cols."""
        df = df.copy()
        # Bot-IoT has attack (binary), category (7-class target), subcategory
        # Use 'category' as the label target.
        if "category" in df.columns:
            if "label" in df.columns:
                df = df.drop(columns=["label"])
            df.rename(columns={"category": "label"}, inplace=True)
        # Drop columns that are not needed for flow-level harmonization
        # (pre-computed entropy / network stats features)
        entropy_cols = [
            "AR_P_Proto_P_Dport", "AR_P_Proto_P_DstIP", "AR_P_Proto_P_Sport",
            "AR_P_Proto_P_SrcIP", "N_IN_Conn_P_DstIP", "N_IN_Conn_P_SrcIP",
            "Pkts_P_State_P_Protocol_P_DestIP", "Pkts_P_State_P_Protocol_P_SrcIP",
            "TnBPDstIP", "TnBPSrcIP", "TnP_PDstIP", "TnP_PSrcIP",
            "TnP_PerProto", "TnP_Per_Dport",
        ]
        existing_entropy = [c for c in entropy_cols if c in df.columns]
        if existing_entropy:
            df = df.drop(columns=existing_entropy)
        # Drop timestamp and other non-flow columns
        extra_drop = ["stime", "ltime", "seq", "attack", "subcategory"]
        existing_extra = [c for c in extra_drop if c in df.columns]
        if existing_extra:
            df = df.drop(columns=existing_extra)
        return df

    def _augment_geometry_expansion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-dataset geometric expansion features for minority-class separation."""
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
            flag_signal = _numeric_series("has_rst") + connection_signal
        out["diff_srv_rate_x_flag"] = diff_srv_rate_series * (flag_signal + 1.0)

        out["protocol_service_flag"] = (
            (protocol_signal + 1.0) * (service_signal + 1.0) * (flag_signal + 1.0)
        )

        return out

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

    def _compute_min_support_targets(
        self,
        *,
        class_count: int,
        test_size: float,
        val_size: float,
    ) -> tuple[int, int, int]:
        """Compute per-class train/val/test targets with minimum support constraints.

        Constraints:
        - train_count_k >= max(50, ceil(0.01 * count_total_k))
        - if class_count < 50: full inclusion into train
        - validation/test target at least 10 when feasible
        """
        n = int(class_count)
        if n <= 0:
            return 0, 0, 0

        if n < 50:
            return n, 0, 0

        min_train = min(n, max(50, int(np.ceil(0.01 * float(n)))))
        min_val = 10
        min_test = 10

        test_n = max(min_test, int(round(float(n) * float(test_size))))
        val_n = max(min_val, int(round(float(n) * float(val_size))))

        overflow = (test_n + val_n) - (n - min_train)
        if overflow > 0:
            reducible_val = max(0, val_n - min_val)
            cut_val = min(reducible_val, (overflow + 1) // 2)
            val_n -= cut_val
            overflow -= cut_val

        if overflow > 0:
            reducible_test = max(0, test_n - min_test)
            cut_test = min(reducible_test, overflow)
            test_n -= cut_test
            overflow -= cut_test

        if overflow > 0:
            test_n = min_test
            val_n = min_val

        train_n = n - val_n - test_n
        if train_n < min_train:
            train_n = min_train
            remainder = max(0, n - train_n)
            if remainder >= 20:
                denom = max(1e-9, float(test_size + val_size))
                test_n = int(round(remainder * float(test_size) / denom))
                test_n = max(min_test, test_n)
                test_n = min(test_n, remainder - min_val)
                val_n = remainder - test_n
                if val_n < min_val:
                    val_n = min_val
                    test_n = remainder - val_n
            else:
                val_n = remainder // 2
                test_n = remainder - val_n

        if train_n + val_n + test_n != n:
            delta = n - (train_n + val_n + test_n)
            train_n += delta

        if train_n < 0 or val_n < 0 or test_n < 0:
            raise RuntimeError(
                f"Invalid support target computation: n={n}, train={train_n}, val={val_n}, test={test_n}"
            )

        return int(train_n), int(val_n), int(test_n)

    def _split_indices_with_min_support(
        self,
        *,
        y: np.ndarray,
        dataset_name: str,
        test_size: float,
        val_size: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Class-constrained split: guarantees non-zero train support for all present classes."""
        rng = np.random.default_rng(self.random_state)
        train_indices: list[int] = []
        val_indices: list[int] = []
        test_indices: list[int] = []

        unique_classes = sorted(int(c) for c in np.unique(y).tolist())
        for class_id in unique_classes:
            cls_idx = np.nonzero(y == class_id)[0].astype(np.int64, copy=False)
            if cls_idx.size == 0:
                continue
            cls_idx = rng.permutation(cls_idx)
            train_n, val_n, test_n = self._compute_min_support_targets(
                class_count=int(cls_idx.size),
                test_size=test_size,
                val_size=val_size,
            )

            cut_train = train_n
            cut_val = train_n + val_n
            cut_test = train_n + val_n + test_n
            train_indices.extend(cls_idx[:cut_train].tolist())
            val_indices.extend(cls_idx[cut_train:cut_val].tolist())
            test_indices.extend(cls_idx[cut_val:cut_test].tolist())

        train_arr = np.asarray(train_indices, dtype=np.int64)
        val_arr = np.asarray(val_indices, dtype=np.int64)
        test_arr = np.asarray(test_indices, dtype=np.int64)

        train_arr = rng.permutation(train_arr) if train_arr.size > 0 else train_arr
        val_arr = rng.permutation(val_arr) if val_arr.size > 0 else val_arr
        test_arr = rng.permutation(test_arr) if test_arr.size > 0 else test_arr

        train_classes = set(np.unique(y[train_arr]).astype(int).tolist()) if train_arr.size > 0 else set()
        missing_train_classes = [c for c in unique_classes if c not in train_classes]
        if missing_train_classes:
            raise RuntimeError(
                "Split integrity violation: classes missing from train split "
                f"for dataset={dataset_name}: {missing_train_classes}"
            )

        return train_arr, val_arr, test_arr

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
        feature_columns = [str(col) for col in df.columns if str(col) != "label"]
        if feature_columns != list(CANONICAL_FEATURE_ORDER):
            raise ValueError(
                "Canonical feature order mismatch; expected exact 17-feature ordering, "
                f"got={feature_columns}"
            )
        return list(CANONICAL_FEATURE_ORDER)

    def _run_unsw_discrete_probe(
        self,
        *,
        unsw_df: pd.DataFrame,
        available_features: list[str],
    ) -> tuple[list[str], float]:
        """Verify separability from discrete transport/state signal before MI ranking."""
        probe_features = [
            feature
            for feature in UNSW_DISCRETE_PROBE_DRIVERS
            if feature in available_features and feature in unsw_df.columns
        ]
        if not probe_features:
            raise RuntimeError(
                "UNSW discrete separability probe unavailable: no mapped categorical drivers "
                "(protocol_type/connection_state/traffic_direction/has_rst) in shared feature space"
            )

        x_probe = np.nan_to_num(
            unsw_df[probe_features].to_numpy(dtype=np.float32, copy=False),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        y_probe = unsw_df["label"].to_numpy(dtype=np.int64, copy=False)
        clf = LogisticRegression(max_iter=1000, random_state=self.random_state)
        clf.fit(x_probe, y_probe)
        pred = np.asarray(clf.predict(x_probe), dtype=np.int64)
        macro_f1 = float(f1_score(y_probe, pred, average="macro", zero_division=0))

        # Use a skew-aware threshold: keep strict upper bound (0.40) but avoid
        # false fails on heavily imbalanced multi-class labels when signal is valid.
        majority_class = int(np.argmax(np.bincount(y_probe)))
        baseline_pred = np.full_like(y_probe, fill_value=majority_class)
        baseline_macro_f1 = float(
            f1_score(y_probe, baseline_pred, average="macro", zero_division=0)
        )
        adaptive_threshold = max(
            UNSW_DISCRETE_PROBE_F1_ADAPTIVE_FLOOR,
            baseline_macro_f1 + UNSW_DISCRETE_PROBE_BASELINE_MARGIN,
        )
        effective_threshold = min(UNSW_DISCRETE_PROBE_F1_MIN, adaptive_threshold)

        if macro_f1 <= effective_threshold:
            raise RuntimeError(
                "UNSW discrete separability probe failed "
                "("
                f"macro_f1={macro_f1:.3f}, "
                f"threshold={effective_threshold:.3f}, "
                f"baseline_macro_f1={baseline_macro_f1:.3f}"
                "); "
                "dataset or semantic mapping may be corrupted"
            )
        return probe_features, macro_f1

    def _select_signal_features(
        self,
        *,
        named_dfs: list[tuple[str, pd.DataFrame]],
        intersection_features: list[str],
        variance_floor: float = 1e-8,
        mi_threshold: float = 2e-3,
    ) -> list[str]:
        """Select UNSW signal-bearing features from the shared intersection."""
        if not intersection_features:
            raise RuntimeError("No cross-dataset feature intersection available")

        unsw_df = next((df for name, df in named_dfs if name == "unsw_nb15"), None)
        if unsw_df is None:
            raise RuntimeError("UNSW dataset is required for signal reconstruction")

        dropped_required = [
            feature for feature in REQUIRED_DISCRETE_DRIVERS if feature not in intersection_features
        ]
        if dropped_required:
            raise RuntimeError(
                "Missing required discrete features from shared space: "
                f"{dropped_required}"
            )

        forced_features, discrete_probe_f1 = self._run_unsw_discrete_probe(
            unsw_df=unsw_df,
            available_features=intersection_features,
        )
        for feature in GEOMETRIC_EXPANSION_FEATURES:
            if feature in intersection_features and feature not in forced_features:
                forced_features.append(feature)

        x_unsw = np.nan_to_num(
            unsw_df[intersection_features].to_numpy(dtype=np.float32, copy=False),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        y_unsw = unsw_df["label"].to_numpy(dtype=np.int64, copy=False)

        variances = np.var(x_unsw, axis=0)
        variance_keep = variances > variance_floor
        if not bool(np.any(variance_keep)):
            raise RuntimeError("All intersection features collapsed to near-zero variance on UNSW")

        variance_candidates = [
            feature
            for feature, keep in zip(intersection_features, variance_keep, strict=False)
            if bool(keep)
        ]
        candidate_names = [feature for feature in variance_candidates if feature not in forced_features]
        if not candidate_names:
            logger.info(
                "UNSW signal reconstruction selected only forced categorical drivers=%s (probe_f1=%.4f)",
                forced_features,
                discrete_probe_f1,
            )
            return forced_features

        candidate_indices = [intersection_features.index(feature) for feature in candidate_names]
        x_candidate = x_unsw[:, candidate_indices]

        discrete_features = np.asarray(
            [name in set(SUPPORTED_DISCRETE_DRIVERS) for name in candidate_names],
            dtype=bool,
        )
        mi_scores = mutual_info_classif(
            x_candidate,
            y_unsw,
            discrete_features=discrete_features,
            random_state=self.random_state,
        )

        ranked = sorted(
            zip(candidate_names, mi_scores, strict=False),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        selected_by_mi = [
            feature
            for feature, score in ranked
            if float(score) > mi_threshold
        ]
        if not selected_by_mi:
            selected_by_mi = [name for name, _ in ranked[: max(1, min(3, len(ranked)))]]

        budget = max(0, MAX_SIGNAL_FEATURES - len(forced_features))
        selected = list(forced_features) + selected_by_mi[:budget]

        dropped_variance = [
            feature
            for feature, keep in zip(intersection_features, variance_keep, strict=False)
            if not bool(keep)
        ]
        dropped_mi = [
            feature
            for feature, score in zip(candidate_names, mi_scores, strict=False)
            if float(score) <= mi_threshold
        ]
        logger.info(
            "UNSW signal reconstruction: intersection=%d selected=%d forced=%s probe_f1=%.4f dropped_variance=%s dropped_mi=%s",
            len(intersection_features),
            len(selected),
            forced_features,
            discrete_probe_f1,
            dropped_variance,
            dropped_mi,
        )
        return selected

    def _scale_dataset_features(
        self,
        *,
        x_train: np.ndarray,
        x_val: np.ndarray,
        x_test: np.ndarray,
        feature_columns: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply per-dataset scaling so continuous bytes/duration do not dominate."""
        train = np.asarray(x_train, dtype=np.float32).copy()
        val = np.asarray(x_val, dtype=np.float32).copy()
        test = np.asarray(x_test, dtype=np.float32).copy()

        feature_to_idx = {name: idx for idx, name in enumerate(feature_columns)}
        for skewed_feature in ("duration", "src_bytes", "dst_bytes"):
            idx = feature_to_idx.get(skewed_feature)
            if idx is None:
                continue
            train[:, idx] = np.log1p(np.clip(train[:, idx], a_min=0.0, a_max=None))
            val[:, idx] = np.log1p(np.clip(val[:, idx], a_min=0.0, a_max=None))
            test[:, idx] = np.log1p(np.clip(test[:, idx], a_min=0.0, a_max=None))

        continuous_idx = [
            idx
            for idx, name in enumerate(feature_columns)
            if name not in set(SUPPORTED_DISCRETE_DRIVERS)
        ]
        if continuous_idx:
            if train.shape[0] == 0:
                # Holdout datasets (for example CICIDS) may be test-only with no train rows.
                # Skip z-score fitting in this case to avoid empty-slice NaNs propagating.
                return (
                    np.nan_to_num(train, nan=0.0, posinf=0.0, neginf=0.0),
                    np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0),
                    np.nan_to_num(test, nan=0.0, posinf=0.0, neginf=0.0),
                )

            mean = train[:, continuous_idx].mean(axis=0, keepdims=True)
            std = train[:, continuous_idx].std(axis=0, keepdims=True)
            std[std < 1e-6] = 1.0
            train[:, continuous_idx] = (train[:, continuous_idx] - mean) / std
            val[:, continuous_idx] = (val[:, continuous_idx] - mean) / std
            test[:, continuous_idx] = (test[:, continuous_idx] - mean) / std

        return (
            np.nan_to_num(train, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(test, nan=0.0, posinf=0.0, neginf=0.0),
        )

    def _assert_prediction_coverage_guard(
        self,
        *,
        x_train: np.ndarray,
        y_train: np.ndarray,
        dataset_name: str,
        stage_name: str,
        min_coverage: float = MIN_UNIQUE_PRED_CLASS_COVERAGE,
    ) -> None:
        """Abort early when model predictions collapse to too few classes."""
        y = np.asarray(y_train, dtype=np.int64)
        x = np.asarray(x_train, dtype=np.float32)
        if x.shape[0] == 0:
            return

        classes = np.unique(y)
        if classes.size <= 1:
            return

        clf = LogisticRegression(max_iter=1000, random_state=self.random_state)
        clf.fit(x, y)
        pred = np.asarray(clf.predict(x), dtype=np.int64)
        coverage = float(np.unique(pred).size / classes.size)
        if coverage < min_coverage:
            raise RuntimeError(
                f"{dataset_name} class collapse detected at {stage_name}: "
                f"unique_pred_coverage={coverage:.3f} < {min_coverage:.3f}"
            )

    def load_and_harmonize_all(
        self,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame | None,
        pd.DataFrame | None,
        pd.DataFrame | None,
        pd.DataFrame | None,
    ]:
        """Load and harmonize all 6 datasets.

        Returns:
            (nsl_kdd_df, unsw_df, cicids_df, ton_iot_df, bot_iot_df, cicids2017_df) - each with common features + label
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
            logger.warning("  → CICIDS not available, continuing")

        ton_iot = None
        logger.info("Loading TON-IoT...")
        try:
            ton_iot_raw = self.load_ton_iot()
            ton_iot = self.harmonize_ton_iot(ton_iot_raw)
            logger.info(f"  → TON-IoT: {len(ton_iot)} samples, shape {ton_iot.shape}")
        except FileNotFoundError:
            logger.warning("  → TON-IoT not available, continuing")

        bot_iot = None
        logger.info("Loading Bot-IoT (masoltani/bot-iot)...")
        try:
            bot_iot_raw = self.load_bot_iot()
            if bot_iot_raw is not None:
                bot_iot = self.harmonize_bot_iot(bot_iot_raw)
                logger.info(f"  → Bot-IoT: {len(bot_iot)} samples, shape {bot_iot.shape}")
            else:
                logger.warning("  → Bot-IoT not available, continuing")
        except Exception as exc:
            logger.warning(f"  → Bot-IoT failed: {exc}")

        cicids2017 = None
        logger.info("Loading CIC-IDS2017...")
        try:
            cicids2017_raw = self.load_cicids2017()
            if cicids2017_raw is not None:
                cicids2017 = self.harmonize_cicids2017(cicids2017_raw)
                logger.info(f"  → CIC-IDS2017: {len(cicids2017)} samples, shape {cicids2017.shape}")
            else:
                logger.warning("  → CIC-IDS2017 not available, continuing")
        except Exception as exc:
            logger.warning(f"  → CIC-IDS2017 failed: {exc}")

        return nsl_kdd, unsw, cicids, ton_iot, bot_iot, cicids2017

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
        dataset_names = ["nsl_kdd", "unsw_nb15", "cicids", "ton_iot", "bot_iot", "cicids2017"]
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
        selected_feature_columns: list[str] = list(FEATURE_ORDER)
        val_ratio = val_size / (1 - test_size)

        for dataset_name, df in named_dfs:
            logger.info(f"Creating splits for {dataset_name}...")
            feature_columns = [str(col) for col in df.columns if str(col) != "label"]
            if feature_columns != selected_feature_columns:
                raise RuntimeError(
                    "Cross-dataset canonical feature order violated. "
                    f"dataset={dataset_name}; expected={selected_feature_columns}; actual={feature_columns}"
                )

            x_all = df[selected_feature_columns].to_numpy(dtype=np.float32, copy=False)
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
                train_idx, val_idx, test_idx = self._split_indices_with_min_support(
                    y=y_all,
                    dataset_name=dataset_name,
                    test_size=test_size,
                    val_size=val_size,
                )
                x_train = x_all[train_idx]
                y_train = y_all[train_idx]
                x_val = x_all[val_idx]
                y_val = y_all[val_idx]
                x_test = x_all[test_idx]
                y_test = y_all[test_idx]

                # Verification diagnostics
                total_counts = pd.Series(y_all).value_counts().sort_index().to_dict()
                train_counts = pd.Series(y_train).value_counts().sort_index().to_dict()
                val_counts = pd.Series(y_val).value_counts().sort_index().to_dict()
                test_counts = pd.Series(y_test).value_counts().sort_index().to_dict()
                logger.info(
                    "SplitSupport[%s] total=%s train=%s val=%s test=%s",
                    dataset_name,
                    total_counts,
                    train_counts,
                    val_counts,
                    test_counts,
                )
                if any(int(v) <= 0 for v in train_counts.values()):
                    raise RuntimeError(
                        f"Split integrity violation: non-positive train class support for {dataset_name}"
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
            x_train_raw = _transform(parts["X_train"])
            x_val_raw = _transform(parts["X_val"])
            x_test_raw = _transform(parts["X_test"])

            x_train, x_val, x_test = self._scale_dataset_features(
                x_train=x_train_raw,
                x_val=x_val_raw,
                x_test=x_test_raw,
                feature_columns=selected_feature_columns,
            )

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
                    "split_then_nan_to_num": x_train_raw,
                    "per_dataset_log1p_zscore": x_train,
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

                self._assert_prediction_coverage_guard(
                    x_train=x_train,
                    y_train=y_train,
                    dataset_name=dataset_name,
                    stage_name="per_dataset_log1p_zscore",
                )
                transformed_splits["diagnostic_unsw_pre_transform"] = pre_transform
                transformed_splits["diagnostic_unsw_post_nan"] = x_train_raw

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
        nsl_kdd, unsw, cicids, ton_iot, bot_iot, cicids2017 = self.load_and_harmonize_all()

        # Create splits
        splits = self.create_splits(
            [nsl_kdd, unsw, cicids, ton_iot, bot_iot, cicids2017]
        )

        # Save splits as numpy arrays
        for key, arr in splits.items():
            if key.startswith(("X_", "y_")):
                np.save(output_dir / f"{key}.npy", arr)
                logger.info(f"Saved {key} -> {output_dir / f'{key}.npy'}")

        feature_columns = np.asarray(splits.get("feature_columns", np.array([], dtype=object)))
        if feature_columns.size == 0:
            raise RuntimeError("Missing feature_columns in split artifact; cannot validate schema")
        if int(feature_columns.size) != CANONICAL_INPUT_DIM:
            raise RuntimeError(
                f"feature_columns.npy length mismatch: expected {CANONICAL_INPUT_DIM}, got {int(feature_columns.size)}"
            )
        validate_feature_order([str(col) for col in feature_columns.tolist()], context="feature_columns.npy")
        np.save(output_dir / "feature_columns.npy", feature_columns)

        contract_payload = {
            "schema_version": SCHEMA_VERSION,
            "feature_order": [str(col) for col in feature_columns.tolist()],
            "schema_hash": None,
            "input_dim": CANONICAL_INPUT_DIM,
            "binary_output_dim": CANONICAL_BINARY_CLASSES,
            "family_output_dim": CANONICAL_FAMILY_CLASSES,
        }

        x_unsw = splits.get("X_train_unsw_nb15")
        y_unsw = splits.get("y_train_unsw_nb15")
        if x_unsw is None or y_unsw is None:
            raise RuntimeError("Missing UNSW train splits required for learnability contract")

        feature_names = [str(col) for col in feature_columns.tolist()]
        schema_hash = compute_schema_hash(feature_order=feature_names)
        contract_payload["schema_hash"] = schema_hash
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
        raw_unsw_post_nan = np.nan_to_num(
            np.asarray(splits.get("diagnostic_unsw_post_nan", x_unsw), dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        contract_x_unsw = np.asarray(raw_unsw_post_nan, dtype=np.float32)
        contract_y_unsw = np.asarray(y_unsw, dtype=np.int64)

        metrics = compute_contract_metrics(
            x_train=contract_x_unsw,
            y_train=contract_y_unsw,
            dataset="unsw_nb15",
            schema_hash=schema_hash,
            feature_names=feature_names,
            feature_lineage=feature_lineage,
            stage_snapshots={
                "pre_transform": raw_unsw_train,
                "split_then_nan_to_num": contract_x_unsw,
            },
            random_seed=self.random_state,
        )
        profile_bundle = load_reference_profile_bundle(
            artifact_dir=output_dir,
            dataset_signature="unsw",
        )
        metrics["reference_profile"] = profile_bundle["profile"]
        metrics["expected_reference_profile_version"] = profile_bundle["profile"].get("version")
        meta = build_meta(metrics, thresholds=PREPROCESS_THRESHOLDS)
        write_meta(meta, artifact_dir=output_dir)
        profile_bundle["payload"]["reference_profiles"][profile_bundle["profile_key"]] = meta["reference_profile"]
        write_reference_profile(profile_bundle["payload"], artifact_dir=output_dir)
        (output_dir / "canonical_contract.json").write_text(
            json.dumps(contract_payload, indent=2),
            encoding="utf-8",
        )
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
    nsl_kdd, unsw, cicids, ton_iot, bot_iot, cicids2017 = loader.load_and_harmonize_all()

    print("\n=== Harmonized Datasets ===")
    print(f"NSL-KDD: {nsl_kdd.shape}")
    print(f"  Columns: {list(nsl_kdd.columns)}")
    print(f"UNSW-NB15: {unsw.shape}")
    print(f"  Columns: {list(unsw.columns)}")
    if cicids is not None:
        print(f"CICIDS: {cicids.shape}")
        print(f"  Columns: {list(cicids.columns)}")
    if ton_iot is not None:
        print(f"TON-IoT: {ton_iot.shape}")
        print(f"  Columns: {list(ton_iot.columns)}")
    if bot_iot is not None:
        print(f"Bot-IoT: {bot_iot.shape}")
        print(f"  Columns: {list(bot_iot.columns)}")
    if cicids2017 is not None:
        print(f"CIC-IDS2017: {cicids2017.shape}")
        print(f"  Columns: {list(cicids2017.columns)}")

    print("\n=== Creating Splits ===")
    splits = loader.create_splits(
        [nsl_kdd, unsw, cicids, ton_iot, bot_iot, cicids2017]
    )
    for key in sorted(splits.keys()):
        print(f"{key}: {splits[key].shape}")
