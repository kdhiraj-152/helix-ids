"""
Unit tests for data harmonization and multi-dataset loading (Phase 1).
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.helix_ids.data.feature_harmonization import (
    COMMON_FEATURES,
    INVARIANT_FEATURES,
    ATTACK_TAXONOMY_7CLASS,
    NSLKDD_TO_7CLASS,
    UNSW_TO_7CLASS,
    CICIDS_TO_7CLASS,
    harmonize_features,
    create_nslkdd_mapping,
    create_unsw_mapping,
    create_cicids_mapping,
    normalize_column_name,
)
from src.helix_ids.data.multi_dataset_loader import MultiDatasetLoader


class TestFeatureHarmonization:
    """Test feature harmonization module."""
    
    def test_common_features_count(self):
        """Verify invariant feature set size."""
        assert len(COMMON_FEATURES) == 15
        assert isinstance(COMMON_FEATURES, list)
    
    def test_attack_taxonomy_7class(self):
        """Verify 7-class attack taxonomy."""
        assert len(ATTACK_TAXONOMY_7CLASS) == 7
        assert 0 in ATTACK_TAXONOMY_7CLASS  # Normal
        assert ATTACK_TAXONOMY_7CLASS[0] == "Normal"
    
    def test_nslkdd_mapping(self):
        """Test NSL-KDD feature mapping."""
        mapping = create_nslkdd_mapping()
        assert mapping.dataset_name == "nsl_kdd"
        assert len(mapping.common_features) == len(COMMON_FEATURES)
        for key in ["duration", "src_bytes", "dst_bytes", "protocol", "state"]:
            assert key in mapping.feature_mapping
    
    def test_unsw_mapping(self):
        """Test UNSW-NB15 feature mapping."""
        mapping = create_unsw_mapping()
        assert mapping.dataset_name == "unsw_nb15"
        for key in ["duration", "src_bytes", "dst_bytes", "protocol", "state"]:
            assert key in mapping.feature_mapping
    
    def test_cicids_mapping(self):
        """Test CICIDS feature mapping."""
        mapping = create_cicids_mapping()
        assert mapping.dataset_name == "cicids"
        for key in ["duration", "src_bytes", "dst_bytes", "protocol", "syn_count", "rst_count"]:
            assert key in mapping.feature_mapping
    
    def test_harmonize_handles_cicids_messy_columns(self):
        """Harmonization should work when CICIDS columns have spaces/case issues."""
        df = pd.DataFrame(
            {
                "  Flow Duration ": [1000, 2000],
                "Protocol": [6, 17],
                " Dst Port": [80, 443],
                "TotLen Fwd Pkts": [100.0, np.inf],
                "TotLen Bwd Pkts": [200.0, np.nan],
                "SYN Flag Cnt": [10, 2],
                "RST Flag Cnt": [0, 1],
                "ACK Flag Cnt": [9, 1],
                "Flow IAT Mean": [100.0, 200.0],
                "Fwd IAT Max": [130.0, 260.0],
                "Fwd IAT Min": [70.0, 120.0],
                "Bwd IAT Max": [110.0, 210.0],
                "Bwd IAT Min": [50.0, 140.0],
                "Label ": [" Benign", "Bot  "],
            }
        )
        mapping = create_cicids_mapping()
        harmonized = harmonize_features(df, mapping, label_col="label")

        assert "label" in harmonized.columns
        assert len(harmonized.columns) == len(COMMON_FEATURES) + 1
        assert np.isfinite(harmonized[COMMON_FEATURES].replace({np.nan: 0.0}).values).all()
        assert harmonized["label"].iloc[0].strip().lower() == "benign"

    def test_normalize_column_name(self):
        """Column normalization should collapse spacing/case differences."""
        assert normalize_column_name(" Label ") == "label"
        assert normalize_column_name("Fwd_Pkt_Len_Mean") == "fwd pkt len mean"


class TestMultiDatasetLoader:
    """Test multi-dataset loader."""
    
    def test_loader_initialization(self):
        """Test loader initialization."""
        loader = MultiDatasetLoader()
        assert loader.project_root.exists()
        assert loader.data_dir.exists()
    
    def test_load_nslkdd(self):
        """Test loading NSL-KDD dataset."""
        loader = MultiDatasetLoader()
        try:
            df = loader.load_nslkdd()
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            print(f"✅ NSL-KDD loaded: {df.shape}")
        except FileNotFoundError as e:
            print(f"⚠️ NSL-KDD not found: {e}")
    
    def test_load_unsw(self):
        """Test loading UNSW-NB15 dataset."""
        loader = MultiDatasetLoader()
        try:
            df = loader.load_unsw()
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            print(f"✅ UNSW-NB15 loaded: {df.shape}")
        except FileNotFoundError as e:
            print(f"⚠️ UNSW-NB15 not found: {e}")
    
    def test_harmonize_nslkdd(self):
        """Test NSL-KDD harmonization."""
        loader = MultiDatasetLoader()
        try:
            df = loader.load_nslkdd()
            harmonized = loader.harmonize_nslkdd(df)
            assert "label" in harmonized.columns
            assert len(harmonized.columns) == len(COMMON_FEATURES) + 1
            print(f"✅ NSL-KDD harmonized: {harmonized.shape}")
        except FileNotFoundError:
            pytest.skip("NSL-KDD not found")
    
    def test_normalize_per_dataset(self):
        """Test per-dataset normalization."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            feat: rng.uniform(0, 100, 100)
            for feat in COMMON_FEATURES
        })
        df["label"] = rng.integers(0, 7, 100)
        
        loader = MultiDatasetLoader()
        df_norm = loader.normalize_per_dataset(df, dataset_code=0, fit=True)
        
        # Check normalization worked
        for feat in COMMON_FEATURES:
            assert df_norm[feat].min() >= -0.01  # Small tolerance for floating point
            assert df_norm[feat].max() <= 1.01
        
        print("✅ Normalization verified")
    
    def test_create_splits(self):
        """Test split creation."""
        # Create synthetic data
        rng = np.random.default_rng(123)
        df = pd.DataFrame({
            feat: rng.uniform(0, 100, 100)
            for feat in COMMON_FEATURES
        })
        df["label"] = rng.integers(0, 7, 100)
        
        loader = MultiDatasetLoader()
        splits = loader.create_splits([df])
        
        assert "X_train" in splits
        assert "y_train" in splits
        assert "X_val" in splits
        assert "y_val" in splits
        assert "X_test_nsl_kdd" in splits
        assert "y_test_nsl_kdd" in splits
        
        # Verify shapes
        assert splits["X_train"].shape[0] > 0
        assert splits["X_train"].shape[1] == len(INVARIANT_FEATURES)
        assert len(splits["y_train"]) == splits["X_train"].shape[0]
        assert "train_class_weights" in splits
        assert splits["train_class_weights"].ndim == 1
        assert "X_val_nsl_kdd" in splits
        assert "X_test_nsl_kdd" in splits
        
        print("✅ Splits created correctly")

    def test_clean_cicids_frame(self):
        """CICIDS cleaner should strip labels and preserve NaNs for split-time imputation."""
        loader = MultiDatasetLoader()
        dirty = pd.DataFrame(
            {
                " Flow Duration ": [1.0, np.inf],
                "TotLen Fwd Pkts": [np.nan, 2.0],
                " Label ": [" Benign", " DDoS  "],
            }
        )

        cleaned = loader._clean_cicids_frame(dirty)
        assert "attack_type" in cleaned.columns
        assert cleaned["attack_type"].tolist() == ["Benign", "DDoS"]
        numeric = cleaned.drop(columns=["attack_type"])
        assert not np.isinf(numeric.values).any()
        assert np.isnan(numeric.values).any()


if __name__ == "__main__":
    # Run quick validation
    pytest.main([__file__, "-v"])
