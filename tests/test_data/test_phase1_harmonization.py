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
        assert len(COMMON_FEATURES) == 19
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
                "FIN Flag Cnt": [1, 0],
                "Tot Fwd Pkts": [12, 3],
                "Tot Bwd Pkts": [9, 1],
                "Flow IAT Mean": [100.0, 200.0],
                "Fwd IAT Max": [130.0, 260.0],
                "Fwd IAT Min": [70.0, 120.0],
                "Bwd IAT Max": [110.0, 210.0],
                "Bwd IAT Min": [50.0, 140.0],
                "Active Mean": [20.0, 40.0],
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

    @staticmethod
    def _assert_invariant_feature_bounds(df: pd.DataFrame) -> None:
        bounded_01 = [
            "bytes_forward_ratio",
            "rst_fraction",
            "handshake_completion_rate",
            "fin_fraction",
            "connection_attempt_rate",
        ]
        bounded_m11 = [
            "bytes_asymmetry",
            "byte_direction_ratio",
            "packet_direction_ratio",
        ]
        binary_cols = [
            "proto_tcp",
            "proto_udp",
            "proto_icmp",
            "proto_other",
            "state_error_indicator",
            "state_reset_retrans_indicator",
        ]

        assert (df["duration_log"] >= 0.0).all()
        assert (df["total_bytes_log"] >= 0.0).all()

        for col in bounded_01:
            assert (df[col] >= 0.0).all(), f"{col} has values below 0"
            assert (df[col] <= 1.0).all(), f"{col} has values above 1"

        for col in bounded_m11:
            assert (df[col] >= -1.0).all(), f"{col} has values below -1"
            assert (df[col] <= 1.0).all(), f"{col} has values above 1"

        for col in binary_cols:
            assert set(np.unique(df[col].to_numpy())).issubset({0.0, 1.0})

    def test_nsl_harmonization_shape_order_and_bounds(self):
        """NSL harmonization must produce 19 invariant features in stable order."""
        loader = MultiDatasetLoader()
        df = pd.DataFrame(
            {
                "duration": [1.0, 4.0],
                "src_bytes": [10.0, 100.0],
                "dst_bytes": [5.0, 50.0],
                "protocol_type": ["tcp", "udp"],
                "service": ["http", "dns"],
                "flag": ["SF", "REJ"],
                "rerror_rate": [0.0, 0.2],
                "srv_rerror_rate": [0.0, 0.1],
                "dst_host_rerror_rate": [0.0, 0.1],
                "serror_rate": [0.0, 0.3],
                "srv_serror_rate": [0.0, 0.2],
                "dst_host_serror_rate": [0.0, 0.2],
                "count": [2.0, 10.0],
                "srv_count": [2.0, 3.0],
                "diff_srv_rate": [0.1, 0.6],
                "label": ["Normal", "DoS"],
            }
        )

        harmonized = loader.harmonize_nslkdd(df)
        assert harmonized.shape[1] == 20
        assert list(harmonized.columns[:-1]) == COMMON_FEATURES
        assert np.isfinite(harmonized[COMMON_FEATURES].to_numpy()).all()
        self._assert_invariant_feature_bounds(harmonized[COMMON_FEATURES])

    def test_unsw_harmonization_shape_order_and_bounds(self):
        """UNSW harmonization must produce 19 invariant features in stable order."""
        loader = MultiDatasetLoader()
        df = pd.DataFrame(
            {
                "dur": [0.1, 3.2],
                "sbytes": [12.0, 140.0],
                "dbytes": [6.0, 35.0],
                "proto": ["tcp", "icmp"],
                "service": ["http", "-"],
                "state": ["CON", "RST"],
                "ct_src_ltm": [3.0, 12.0],
                "ct_srv_src": [2.0, 6.0],
                "dsport": [80, 443],
                "Sintpkt": [0.02, 0.50],
                "Dintpkt": [0.01, 0.25],
                "Sjit": [0.01, 0.20],
                "Djit": [0.01, 0.20],
                "ct_src_dport_ltm": [1.0, 4.0],
                "Spkts": [10.0, 40.0],
                "Dpkts": [5.0, 30.0],
                "label": ["Normal", "Backdoors"],
            }
        )

        harmonized = loader.harmonize_unsw(df)
        assert harmonized.shape[1] == 20
        assert list(harmonized.columns[:-1]) == COMMON_FEATURES
        assert np.isfinite(harmonized[COMMON_FEATURES].to_numpy()).all()
        self._assert_invariant_feature_bounds(harmonized[COMMON_FEATURES])

    def test_cicids_harmonization_shape_order_and_bounds(self):
        """CICIDS harmonization must produce 19 invariant features in stable order."""
        loader = MultiDatasetLoader()
        df = pd.DataFrame(
            {
                "Flow Duration": [1000.0, 2000.0],
                "TotLen Fwd Pkts": [100.0, 350.0],
                "TotLen Bwd Pkts": [50.0, 20.0],
                "Protocol": [6, 17],
                "SYN Flag Cnt": [6.0, 1.0],
                "RST Flag Cnt": [0.0, 1.0],
                "ACK Flag Cnt": [5.0, 1.0],
                "Tot Fwd Pkts": [10.0, 15.0],
                "Dst Port": [80, 443],
                "Flow IAT Mean": [100.0, 300.0],
                "Fwd IAT Mean": [80.0, 220.0],
                "Bwd IAT Mean": [70.0, 190.0],
                "Fwd IAT Max": [150.0, 320.0],
                "Bwd IAT Max": [130.0, 300.0],
                "Fwd IAT Min": [40.0, 90.0],
                "Bwd IAT Min": [35.0, 80.0],
                "Tot Bwd Pkts": [8.0, 4.0],
                "Active Mean": [50.0, 30.0],
                "Label": ["BENIGN", "DDoS"],
            }
        )

        harmonized = loader.harmonize_cicids(df)
        assert harmonized.shape[1] == 20
        assert list(harmonized.columns[:-1]) == COMMON_FEATURES
        assert np.isfinite(harmonized[COMMON_FEATURES].to_numpy()).all()
        self._assert_invariant_feature_bounds(harmonized[COMMON_FEATURES])

    def test_loader_exposes_no_normalization_surface(self):
        """Loader must not expose dataset transformation APIs."""
        loader = MultiDatasetLoader()
        assert not hasattr(loader, "normalize_per_dataset")


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
    
    def test_create_splits_preserves_unscaled_feature_range(self):
        """Split creation should not normalize features in-loader."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            feat: rng.uniform(0, 100, 100)
            for feat in COMMON_FEATURES
        })
        df["label"] = rng.integers(0, 7, 100)

        loader = MultiDatasetLoader()
        splits = loader.create_splits([df])

        train_x = splits["X_train"]
        assert train_x.shape[1] == len(INVARIANT_FEATURES)
        assert np.isfinite(train_x).all()
        assert float(train_x.max()) > 1.0
    
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
