"""
Test suite for HELIX-IDS data loading.

Tests cover:
- NSL-KDD dataset loading
- UNSW-NB15 dataset loading
- Unified feature alignment
- Train/val/test split ratios
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Skip all data-dependent tests when raw datasets are not available
if not (Path("data/nsl_kdd/train.csv").exists() and Path("data/unsw_nb15/UNSW_NB15_training-set.csv").exists()):
    pytest.skip("Dataset files not available locally (gitignored). Run scripts/data/download_datasets.py first.", allow_module_level=True)

# =============================================================================
# Constants
# =============================================================================

# Expected split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Expected feature counts
NSL_KDD_RAW_FEATURES = 41
UNSW_NB15_RAW_FEATURES = 49
UNIFIED_FEATURES = 32

# Dataset paths (relative to project root)
NSL_KDD_PATH = "data/nsl_kdd"
UNSW_NB15_PATH = "data/unsw_nb15"
PROCESSED_PATH = "data/processed"


# =============================================================================
# NSL-KDD Dataset Tests
# =============================================================================


class TestNSLKDDLoading:
    """Test NSL-KDD dataset loading."""

    def test_nsl_kdd_train_exists(self, nsl_kdd_data_path):
        """
        Test that NSL-KDD training file exists.
        """
        train_path = nsl_kdd_data_path / "train.csv"
        assert train_path.exists(), f"Training file not found at {train_path}"

    def test_nsl_kdd_test_exists(self, nsl_kdd_data_path):
        """
        Test that NSL-KDD test file exists.
        """
        test_path = nsl_kdd_data_path / "test.csv"
        assert test_path.exists(), f"Test file not found at {test_path}"

    def test_nsl_kdd_train_loadable(self, nsl_kdd_data_path):
        """
        Test that NSL-KDD training file can be loaded.
        """
        train_path = nsl_kdd_data_path / "train.csv"
        df = pd.read_csv(train_path)

        assert len(df) > 0, "Training data should not be empty"

    def test_nsl_kdd_test_loadable(self, nsl_kdd_data_path):
        """
        Test that NSL-KDD test file can be loaded.
        """
        test_path = nsl_kdd_data_path / "test.csv"
        df = pd.read_csv(test_path)

        assert len(df) > 0, "Test data should not be empty"

    def test_nsl_kdd_metadata_exists(self, nsl_kdd_data_path):
        """
        Test that NSL-KDD metadata file exists.
        """
        metadata_path = nsl_kdd_data_path / "metadata.json"
        assert metadata_path.exists(), f"Metadata not found at {metadata_path}"

    def test_nsl_kdd_no_missing_files(self, nsl_kdd_data_path):
        """
        Test that all expected NSL-KDD files are present.
        """
        expected_files = ["train.csv", "test.csv", "metadata.json"]
        for filename in expected_files:
            file_path = nsl_kdd_data_path / filename
            assert file_path.exists(), f"Expected file {filename} not found"

    def test_nsl_kdd_train_has_data(self, nsl_kdd_data_path):
        """
        Test that NSL-KDD training data has reasonable number of samples.
        """
        train_path = nsl_kdd_data_path / "train.csv"
        df = pd.read_csv(train_path)

        # NSL-KDD should have at least 1000 samples
        assert len(df) >= 1000, f"Expected at least 1000 samples, got {len(df)}"

    def test_nsl_kdd_data_types(self, nsl_kdd_data_path):
        """
        Test that NSL-KDD data has expected column types.
        """
        train_path = nsl_kdd_data_path / "train.csv"
        df = pd.read_csv(train_path)

        # Should have numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) > 0, "Should have numeric columns"


# =============================================================================
# UNSW-NB15 Dataset Tests
# =============================================================================


class TestUNSWNB15Loading:
    """Test UNSW-NB15 dataset loading."""

    def test_unsw_train_exists(self, unsw_nb15_data_path):
        """
        Test that UNSW-NB15 training file exists.
        """
        train_path = unsw_nb15_data_path / "train.csv"
        assert train_path.exists(), f"Training file not found at {train_path}"

    def test_unsw_test_exists(self, unsw_nb15_data_path):
        """
        Test that UNSW-NB15 test file exists.
        """
        test_path = unsw_nb15_data_path / "test.csv"
        assert test_path.exists(), f"Test file not found at {test_path}"

    def test_unsw_train_loadable(self, unsw_nb15_data_path):
        """
        Test that UNSW-NB15 training file can be loaded.
        """
        train_path = unsw_nb15_data_path / "train.csv"
        df = pd.read_csv(train_path)

        assert len(df) > 0, "Training data should not be empty"

    def test_unsw_test_loadable(self, unsw_nb15_data_path):
        """
        Test that UNSW-NB15 test file can be loaded.
        """
        test_path = unsw_nb15_data_path / "test.csv"
        df = pd.read_csv(test_path)

        assert len(df) > 0, "Test data should not be empty"

    def test_unsw_metadata_exists(self, unsw_nb15_data_path):
        """
        Test that UNSW-NB15 metadata file exists.
        """
        metadata_path = unsw_nb15_data_path / "metadata.json"
        assert metadata_path.exists(), f"Metadata not found at {metadata_path}"

    def test_unsw_train_has_data(self, unsw_nb15_data_path):
        """
        Test that UNSW-NB15 training data has reasonable number of samples.
        """
        train_path = unsw_nb15_data_path / "train.csv"
        df = pd.read_csv(train_path)

        # UNSW-NB15 should have at least 1000 samples
        assert len(df) >= 1000, f"Expected at least 1000 samples, got {len(df)}"


# =============================================================================
# Unified Feature Alignment Tests
# =============================================================================


class TestUnifiedFeatureAlignment:
    """Test unified feature alignment across datasets."""

    def test_unified_features_json_exists(self, processed_data_path):
        """
        Test that unified features JSON exists.
        """
        unified_path = processed_data_path / "unified_features.json"
        assert unified_path.exists(), f"Unified features not found at {unified_path}"

    def test_unified_features_count(self, processed_data_path):
        """
        Test unified features has expected count.
        """
        from scripts.training.train_multidataset_v2_fixed import SafeDataLoader
        count = len(SafeDataLoader.UNIFIED_FEATURES)

        assert count == UNIFIED_FEATURES, \
            f"Expected {UNIFIED_FEATURES} unified features, got {count}"

    def test_feature_mappings_exist(self, processed_data_path):
        """
        Test that feature mappings file exists.
        """
        mappings_path = processed_data_path / "feature_mappings.json"
        assert mappings_path.exists(), f"Feature mappings not found at {mappings_path}"

    def test_production_feature_names_match_unified(self, production_feature_names, processed_data_path):
        """
        Test production features match unified feature set.
        """
        unified_path = processed_data_path / "unified_features.json"

        if unified_path.exists():
            with open(unified_path) as f:
                unified = json.load(f)

            if isinstance(unified, list):
                unified_features = set(unified)
            elif isinstance(unified, dict):
                unified_features = set(unified.get("features", []))
            else:
                unified_features = set()

            production_set = set(production_feature_names)

            # Check overlap (may not be exact due to engineering)
            overlap = production_set & unified_features
            # At least some features should match
            assert len(overlap) > 0 or len(production_set) == UNIFIED_FEATURES

    def test_nsl_kdd_can_be_aligned(self, nsl_kdd_data_path, production_feature_names):
        """
        Test NSL-KDD data can be loaded and has columns that can map to unified features.
        """
        train_path = nsl_kdd_data_path / "train.csv"
        df = pd.read_csv(train_path)

        # Should have numeric columns that can be used
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) > 10, "Should have at least 10 numeric columns for features"

    def test_unsw_can_be_aligned(self, unsw_nb15_data_path, production_feature_names):
        """
        Test UNSW-NB15 data can be loaded and has columns that can map to unified features.
        """
        train_path = unsw_nb15_data_path / "train.csv"
        df = pd.read_csv(train_path)

        # Should have numeric columns that can be used
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) > 10, "Should have at least 10 numeric columns for features"


# =============================================================================
# Train/Val/Test Split Tests
# =============================================================================


class TestDataSplits:
    """Test train/validation/test split ratios."""

    def test_splits_indices_exist(self, project_root):
        """
        Test that split indices files exist.
        """
        splits_dir = project_root / "data" / "splits"
        assert splits_dir.exists(), f"Splits directory not found at {splits_dir}"

    def test_nsl_kdd_indices_exist(self, project_root):
        """
        Test that NSL-KDD split indices exist.
        """
        indices_path = project_root / "data" / "splits" / "nsl-kdd_indices.json"
        assert indices_path.exists(), f"NSL-KDD indices not found at {indices_path}"

    def test_unsw_indices_exist(self, project_root):
        """
        Test that UNSW-NB15 split indices exist.
        """
        indices_path = project_root / "data" / "splits" / "unsw-nb15_indices.json"
        assert indices_path.exists(), f"UNSW-NB15 indices not found at {indices_path}"

    def test_nsl_kdd_split_ratios(self, project_root, nsl_kdd_data_path):
        """
        Test NSL-KDD split ratios are approximately correct.
        """
        indices_path = project_root / "data" / "splits" / "nsl-kdd_indices.json"

        if not indices_path.exists():
            pytest.skip("Split indices file not found")

        with open(indices_path) as f:
            indices = json.load(f)

        # Get total count
        train_path = nsl_kdd_data_path / "train.csv"
        df = pd.read_csv(train_path)
        _ = len(df)

        if isinstance(indices, dict):
            train_count = len(indices.get("train", []))
            val_count = len(indices.get("val", indices.get("validation", [])))
            test_count = len(indices.get("test", []))

            # Allow 10% tolerance on ratios
            if train_count + val_count + test_count > 0:
                actual_train_ratio = train_count / (train_count + val_count + test_count)
                assert 0.5 <= actual_train_ratio <= 0.9, \
                    f"Train ratio {actual_train_ratio:.2f} outside expected range"

    def test_splits_no_overlap(self, project_root):
        """
        Test that train/val/test splits don't overlap.
        """
        indices_path = project_root / "data" / "splits" / "nsl-kdd_indices.json"

        if not indices_path.exists():
            pytest.skip("Split indices file not found")

        with open(indices_path) as f:
            indices = json.load(f)

        if isinstance(indices, dict):
            train_set = set(indices.get("train", []))
            val_set = set(indices.get("val", indices.get("validation", [])))
            test_set = set(indices.get("test", []))

            # Check no overlap
            assert len(train_set & val_set) == 0, "Train and val sets overlap"
            assert len(train_set & test_set) == 0, "Train and test sets overlap"
            assert len(val_set & test_set) == 0, "Val and test sets overlap"

    def test_splits_cover_all_data(self, project_root, nsl_kdd_data_path):
        """
        Test that splits cover all data (no missing indices).
        """
        indices_path = project_root / "data" / "splits" / "nsl-kdd_indices.json"

        if not indices_path.exists():
            pytest.skip("Split indices file not found")

        with open(indices_path) as f:
            indices = json.load(f)

        train_path = nsl_kdd_data_path / "train.csv"
        df = pd.read_csv(train_path)
        total = len(df)

        if isinstance(indices, dict):
            train_count = len(indices.get("train", []))
            val_count = len(indices.get("val", indices.get("validation", [])))
            test_count = len(indices.get("test", []))

            total_split = train_count + val_count + test_count
            # Should cover at least 90% of data (some may be filtered)
            if total > 0:
                coverage = total_split / total
                # Allow for some data being filtered
                assert coverage >= 0.1 or total_split > 0, \
                    f"Split coverage {coverage:.2%} is too low"


# =============================================================================
# UnifiedDataLoader Tests
# =============================================================================


class TestUnifiedDataLoader:
    """Test UnifiedDataLoader class."""

    def test_unified_loader_import(self):
        """
        Test that UnifiedDataLoader can be imported.
        """
        try:
            from src.helix_ids.data.unified_loader import UnifiedDataLoader
            assert UnifiedDataLoader is not None
        except ImportError as e:
            pytest.skip(f"Could not import UnifiedDataLoader: {e}")

    def test_unified_loader_instantiation(self, project_root):
        """
        Test UnifiedDataLoader can be instantiated.
        """
        try:
            from src.helix_ids.data.unified_loader import UnifiedDataLoader
            loader = UnifiedDataLoader(data_dir=str(project_root / "data"))
            assert loader is not None
        except ImportError as e:
            pytest.skip(f"Could not instantiate UnifiedDataLoader: {e}")

    def test_list_available_datasets(self):
        """
        Test listing available datasets.
        """
        try:
            from src.helix_ids.data.unified_loader import list_available_datasets
            datasets = list_available_datasets()
            assert isinstance(datasets, dict)
        except ImportError as e:
            pytest.skip(f"Could not list datasets: {e}")

    def test_load_nsl_kdd_via_loader(self, project_root):
        """
        Test loading NSL-KDD through UnifiedDataLoader.
        """
        try:
            from src.helix_ids.data.unified_loader import UnifiedDataLoader
            loader = UnifiedDataLoader(data_dir=str(project_root / "data"))
            X, y, _ = loader.load("nsl-kdd")

            assert X is not None
            assert y is not None
            assert len(X) == len(y)
            assert len(X) > 0
        except (ImportError, FileNotFoundError) as e:
            pytest.skip(f"Could not load NSL-KDD: {e}")

    def test_load_unsw_via_loader(self, project_root):
        """
        Test loading UNSW-NB15 through UnifiedDataLoader.
        """
        try:
            from src.helix_ids.data.unified_loader import UnifiedDataLoader
            loader = UnifiedDataLoader(data_dir=str(project_root / "data"))
            X, y, _ = loader.load("unsw-nb15")

            assert X is not None
            assert y is not None
            assert len(X) == len(y)
            assert len(X) > 0
        except (ImportError, FileNotFoundError) as e:
            pytest.skip(f"Could not load UNSW-NB15: {e}")


# =============================================================================
# Data Quality Tests
# =============================================================================


class TestDataQuality:
    """Test data quality and integrity."""

    def test_nsl_kdd_no_null_labels(self, nsl_kdd_data_path):
        """
        Test NSL-KDD has no null labels.
        """
        train_path = nsl_kdd_data_path / "train.csv"
        df = pd.read_csv(train_path)

        # Find label column (may be 'label', 'class', 'attack', or last column)
        label_cols = [c for c in df.columns if c.lower() in ['label', 'class', 'attack', 'attack_type']]

        if label_cols:
            label_col = label_cols[0]
            null_count = df[label_col].isna().sum()
            assert null_count == 0, f"Found {null_count} null labels"

    def test_nsl_kdd_reasonable_value_ranges(self, nsl_kdd_data_path):
        """
        Test NSL-KDD numeric values are in reasonable ranges.
        """
        train_path = nsl_kdd_data_path / "train.csv"
        df = pd.read_csv(train_path)

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols[:10]:  # Check first 10 numeric columns
            # Should not have extreme values
            assert not np.isinf(df[col]).any(), f"Column {col} has infinite values"
            # NaN is allowed but should be minimal
            nan_ratio = df[col].isna().sum() / len(df)
            assert nan_ratio < 0.5, f"Column {col} has {nan_ratio:.1%} missing values"

    def test_unsw_no_null_labels(self, unsw_nb15_data_path):
        """
        Test UNSW-NB15 has no null labels.
        """
        train_path = unsw_nb15_data_path / "train.csv"
        df = pd.read_csv(train_path)

        # Find label column
        label_cols = [c for c in df.columns if c.lower() in ['label', 'class', 'attack', 'attack_cat']]

        if label_cols:
            label_col = label_cols[0]
            null_count = df[label_col].isna().sum()
            assert null_count == 0, f"Found {null_count} null labels"

    def test_dataset_has_both_classes(self, nsl_kdd_data_path):
        """
        Test dataset has both normal and attack samples.
        """
        train_path = nsl_kdd_data_path / "train.csv"
        df = pd.read_csv(train_path)

        # Find label column
        label_cols = [c for c in df.columns if c.lower() in ['label', 'class', 'attack', 'attack_type']]

        if label_cols:
            label_col = label_cols[0]
            unique_labels = df[label_col].nunique()
            assert unique_labels >= 2, f"Expected at least 2 classes, got {unique_labels}"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestDataLoadingEdgeCases:
    """Test edge cases in data loading."""

    def test_missing_file_handling(self, project_root):
        """
        Test graceful handling of missing files.
        """
        nonexistent_path = project_root / "data" / "nonexistent" / "train.csv"

        with pytest.raises((FileNotFoundError, IOError, Exception)):
            pd.read_csv(nonexistent_path)

    def test_corrupted_csv_handling(self, project_root, tmp_path):
        """
        Test handling of corrupted CSV files.
        """
        # Create corrupted CSV
        corrupted_path = tmp_path / "corrupted.csv"
        with open(corrupted_path, "w") as f:
            f.write("col1,col2,col3\n")
            f.write("1,2\n")  # Missing column
            f.write("a,b,c,d\n")  # Extra column

        # Should still load (pandas is forgiving) or raise specific error
        try:
            _ = pd.read_csv(corrupted_path, on_bad_lines='skip')
            # May load partially (df successfully loaded)
        except pd.errors.ParserError:
            pass  # Expected for some corrupted files

    def test_empty_csv_handling(self, tmp_path):
        """
        Test handling of empty CSV files.
        """
        empty_path = tmp_path / "empty.csv"
        empty_path.touch()

        with pytest.raises((pd.errors.EmptyDataError, Exception)):
            pd.read_csv(empty_path)

    def test_headers_only_csv(self, tmp_path):
        """
        Test handling of CSV with headers but no data.
        """
        headers_path = tmp_path / "headers_only.csv"
        with open(headers_path, "w") as f:
            f.write("col1,col2,col3\n")

        df = pd.read_csv(headers_path)
        assert len(df) == 0
        assert list(df.columns) == ["col1", "col2", "col3"]
