"""
File I/O utilities for HELIX-IDS datasets.

Extracted from unified_loader.py (was >1230 lines).
Responsible solely for reading CSV / ARFF / TXT files from disk.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from scipy.io import arff

from .dataset_config import (
    CICIDS_EXCLUDE_COLUMNS,
    NSL_KDD_FEATURE_NAMES,
    DatasetConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def load_file(filepath: Path) -> Optional[pd.DataFrame]:
    """Load a single data file (auto-detect CSV / ARFF / TXT)."""
    suffix = filepath.suffix.lower()
    if suffix == ".arff":
        return _load_arff(filepath)
    elif suffix == ".csv":
        return _load_csv(filepath)
    elif suffix == ".txt":
        return _load_txt(filepath)
    logger.warning(f"Unknown file format: {filepath}")
    return None


def find_data_files(base_path: Path) -> list[Path]:
    """Return all CSV / ARFF / TXT files under *base_path*, excluding metadata."""
    files: list[Path] = []
    for pattern in ("*.csv", "*.arff", "*.txt"):
        files.extend(base_path.glob(pattern))

    exclude = {"feature", "metadata", "readme", "list_event"}
    files = [f for f in files if not any(p in f.name.lower() for p in exclude)]
    return sorted(files)


def try_load_split(
    config: DatasetConfig,
    split: str,
) -> Optional[pd.DataFrame]:
    """Try to load a named split (``train`` / ``test``) from pre-defined paths."""
    for base_path in config.paths:
        df = _try_load_split_from_base(base_path, config, split)
        if df is not None:
            return df
    return None


def load_all_files(config: DatasetConfig, verbose: bool = True) -> list[pd.DataFrame]:
    """Load every matching file from all configured paths."""
    dfs: list[pd.DataFrame] = []
    for base_path in config.paths:
        dfs.extend(_load_files_from_path(base_path, verbose))
    return dfs


def _load_files_from_path(base_path: Path, verbose: bool) -> list[pd.DataFrame]:
    """Load all data files from a single path."""
    dfs: list[pd.DataFrame] = []
    if not base_path.exists():
        return dfs
    for filepath in find_data_files(base_path):
        try:
            df = load_file(filepath)
            if df is not None and len(df) > 0:
                dfs.append(df)
                if verbose:
                    logger.info(f"  Loaded {len(df)} rows from {filepath.name}")
        except Exception as e:
            logger.warning(f"  Failed to load {filepath}: {e}")
    return dfs


def harmonize_cicids_frames(dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Align CICIDS DataFrames to their shared columns after dropping identifiers."""
    sanitized = [_sanitize_cicids_frame(df) for df in dfs]
    if not sanitized:
        return sanitized

    common_cols = set(sanitized[0].columns)
    for df in sanitized[1:]:
        common_cols &= set(df.columns)
    if not common_cols:
        return sanitized

    ordered = [c for c in sanitized[0].columns if c in common_cols]
    return [df.loc[:, ordered].copy() for df in sanitized]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_arff(filepath: Path) -> pd.DataFrame:
    data, _ = arff.loadarff(filepath)
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
    return df


def _load_csv(filepath: Path) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(filepath, low_memory=False, encoding=enc)
            df.columns = df.columns.str.strip()
            return df
        except UnicodeDecodeError:
            continue
        except Exception:
            break
    # Fallback: auto-detect separator
    df = pd.read_csv(filepath, low_memory=False, sep=None, engine="python")
    df.columns = df.columns.str.strip()
    return df


def _load_txt(filepath: Path) -> Optional[pd.DataFrame]:
    """Load NSL-KDD style TXT file (41 features + attack_type + difficulty)."""
    columns = NSL_KDD_FEATURE_NAMES + ["attack_type", "difficulty"]
    try:
        return pd.read_csv(filepath, header=None, names=columns)
    except Exception as e:
        logger.warning(f"Failed to load TXT file {filepath}: {e}")
        return None


def _sanitize_cicids_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    drop_cols = [c for c in CICIDS_EXCLUDE_COLUMNS if c in cleaned.columns]
    if drop_cols:
        cleaned = cleaned.drop(columns=drop_cols)
    return cleaned


def _try_load_split_from_base(
    base_path: Path,
    config: DatasetConfig,
    split: str,
) -> Optional[pd.DataFrame]:
    if not base_path.exists():
        return None
    for pattern in _build_split_patterns(config, split):
        matches = list(base_path.glob(pattern))
        if matches:
            try:
                return load_file(matches[0])
            except Exception:
                continue
    return None


def _build_split_patterns(config: DatasetConfig, split: str) -> list[str]:
    if config.name == "NSL-KDD":
        return [
            f"KDD{'Train' if split == 'train' else 'Test'}+.txt",
            f"{split}.csv",
            f"{split}.arff",
        ]
    return [
        f"{split}.csv",
        f"{split}.arff",
        f"*{split}*.csv",
        f"*_{split}*.csv",
        f"*-{split}*.csv",
        f"UNSW_NB15_{'training' if split == 'train' else 'testing'}-set.csv",
    ]
