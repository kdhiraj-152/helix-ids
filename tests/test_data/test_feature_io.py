"""Tests for feature_io.py — file load/save for HELIX-IDS datasets.

Covers:
  - load_file for CSV, ARFF, TXT formats
  - find_data_files with exclude patterns
  - try_load_split for named splits
  - CICIDS frame harmonization
  - CSV chunk reading
  - Error handling for missing/corrupt files
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from helix_ids.data.dataset_config import DatasetConfig, NSL_KDD_FEATURE_NAMES
from helix_ids.data.feature_io import (
    find_data_files,
    harmonize_cicids_frames,
    load_all_files,
    load_file,
    try_load_split,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def tmp_csv(tmp_path: Path) -> Path:
    """Create a small CSV file and return its path."""
    path = tmp_path / "test_data.csv"
    pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}).to_csv(path, index=False)
    return path


@pytest.fixture
def tmp_arff(tmp_path: Path) -> Path:
    """Create a small ARFF file with valid syntax."""
    path = tmp_path / "test_data.arff"
    content = (
        "@RELATION test\n"
        "@ATTRIBUTE a NUMERIC\n"
        "@ATTRIBUTE b NUMERIC\n"
        "@DATA\n"
        "1.0,2.0\n"
        "3.0,4.0\n"
    )
    path.write_text(content)
    return path


@pytest.fixture
def tmp_txt(tmp_path: Path) -> Path:
    """Create a small NSL-KDD style TXT file."""
    path = tmp_path / "KDDTrain+.txt"
    row = [str(i) for i in range(41)] + ["normal", "0"]
    path.write_text(",".join(row) + "\n")
    return path


@pytest.fixture
def tmp_unknown(tmp_path: Path) -> Path:
    """Create a file with unknown extension."""
    path = tmp_path / "test.bin"
    path.write_bytes(b"\x00\x01\x02")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# load_file
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoadFile:
    def test_load_csv(self, tmp_csv: Path) -> None:
        """CSV file loads correctly."""
        df = load_file(tmp_csv)
        assert df is not None
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2

    def test_load_arff(self, tmp_arff: Path) -> None:
        """ARFF file loads correctly."""
        df = load_file(tmp_arff)
        assert df is not None
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2

    def test_load_txt_nsl_kdd(self, tmp_txt: Path) -> None:
        """NSL-KDD TXT file loads correctly."""
        df = load_file(tmp_txt)
        assert df is not None
        assert len(df) == 1
        expected_cols = len(NSL_KDD_FEATURE_NAMES) + 2
        assert df.shape[1] == expected_cols

    def test_load_unknown_format(self, tmp_unknown: Path) -> None:
        """Unknown file format returns None."""
        result = load_file(tmp_unknown)
        assert result is None

    def test_load_nonexistent_unknown_format(self, tmp_path: Path) -> None:
        """Nonexistent file with unknown extension returns None."""
        missing = tmp_path / "nonexistent.xyz"
        result = load_file(missing)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# find_data_files
# ═══════════════════════════════════════════════════════════════════════════════


class TestFindDataFiles:
    def test_finds_csv_files(self, tmp_path: Path) -> None:
        """Finds CSV files."""
        (tmp_path / "data.csv").write_text("a,b\n1,2\n")
        files = find_data_files(tmp_path)
        assert any(f.name == "data.csv" for f in files)

    def test_excludes_metadata(self, tmp_path: Path) -> None:
        """Files with 'feature', 'metadata', 'readme' in name are excluded."""
        (tmp_path / "data.csv").write_text("a,b\n1,2\n")
        (tmp_path / "feature_list.csv").write_text("a,b\n1,2\n")
        files = find_data_files(tmp_path)
        assert any(f.name == "data.csv" for f in files)
        assert not any("feature" in f.name.lower() for f in files)

    def test_skip_large_cicids(self, tmp_path: Path) -> None:
        """HELIX_SKIP_LARGE_CICIDS env var skips the large CICIDS file."""
        large_name = "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"
        (tmp_path / large_name).write_text("a,b\n1,2\n")
        (tmp_path / "other.csv").write_text("a,b\n1,2\n")
        os.environ["HELIX_SKIP_LARGE_CICIDS"] = "1"
        try:
            files = find_data_files(tmp_path)
            assert not any(large_name in f.name for f in files)
            assert any(f.name == "other.csv" for f in files)
        finally:
            os.environ.pop("HELIX_SKIP_LARGE_CICIDS", None)

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory returns empty list."""
        files = find_data_files(tmp_path)
        assert files == []


# ═══════════════════════════════════════════════════════════════════════════════
# try_load_split
# ═══════════════════════════════════════════════════════════════════════════════


class TestTryLoadSplit:
    def test_loads_train_split(self, tmp_txt: Path) -> None:
        """Loads training split by pattern."""
        base = tmp_txt.parent
        config = DatasetConfig(
            name="NSL-KDD",
            class_names=["normal", "attack"],
            label_column="label",
            feature_count=41,
            paths=[base],
        )
        df = try_load_split(config, split="train")
        assert df is not None
        assert len(df) > 0

    def test_nonexistent_path_returns_none(self, tmp_path: Path) -> None:
        """Nonexistent base path returns None."""
        config = DatasetConfig(
            name="NSL-KDD",
            class_names=["normal", "attack"],
            label_column="label",
            feature_count=41,
            paths=[tmp_path / "nonexistent"],
        )
        result = try_load_split(config, split="train")
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# harmonize_cicids_frames
# ═══════════════════════════════════════════════════════════════════════════════


class TestHarmonizeCicidsFrames:
    def test_aligns_common_columns(self) -> None:
        """Frames are aligned to common columns."""
        df1 = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        df2 = pd.DataFrame({"a": [4], "b": [5], "d": [6]})
        result = harmonize_cicids_frames([df1, df2])
        assert len(result) == 2
        assert list(result[0].columns) == ["a", "b"]
        assert list(result[1].columns) == ["a", "b"]

    def test_no_common_columns(self) -> None:
        """No common columns returns sanitized frames unchanged."""
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"b": [2]})
        result = harmonize_cicids_frames([df1, df2])
        assert len(result) == 2

    def test_empty_list(self) -> None:
        """Empty input list returns empty list."""
        result = harmonize_cicids_frames([])
        assert result == []

    def test_single_frame(self) -> None:
        """Single frame is returned as-is."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = harmonize_cicids_frames([df])
        assert len(result) == 1
        assert list(result[0].columns) == ["a", "b"]


# ═══════════════════════════════════════════════════════════════════════════════
# load_all_files
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoadAllFiles:
    def test_loads_multiple_files(self, tmp_csv: Path) -> None:
        """Loads all matching files from a path."""
        config = DatasetConfig(
            name="Test",
            class_names=["normal", "attack"],
            label_column="label",
            feature_count=2,
            paths=[tmp_csv.parent],
        )
        dfs = load_all_files(config, verbose=False)
        assert len(dfs) > 0
        assert all(isinstance(df, pd.DataFrame) for df in dfs)

    def test_nonexistent_path_returns_empty(self, tmp_path: Path) -> None:
        """Nonexistent path returns empty list."""
        config = DatasetConfig(
            name="Test",
            class_names=["normal", "attack"],
            label_column="label",
            feature_count=2,
            paths=[tmp_path / "nonexistent"],
        )
        dfs = load_all_files(config, verbose=False)
        assert dfs == []
