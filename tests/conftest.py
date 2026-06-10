"""
Pytest fixtures for HELIX-IDS testing.

Provides common test fixtures including:
- Sample data batches (NSL-KDD format)
- Labels (binary and multiclass)
- Pre-instantiated model variants
- Production model loading
- Scaler fixtures
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Skip all tests if torch is not available
pytest.importorskip("torch")

# =============================================================================
# Constants
# =============================================================================

PRODUCTION_MODEL_PATH = PROJECT_ROOT / "models" / "production" / "helix_ids_v1.pt"
PRODUCTION_SCALER_PATH = PROJECT_ROOT / "models" / "production" / "scaler.pkl"
PRODUCTION_FEATURE_NAMES_PATH = PROJECT_ROOT / "models" / "production" / "feature_names.json"
PRODUCTION_MODEL_CARD_PATH = PROJECT_ROOT / "models" / "production" / "model_card.json"

INPUT_DIM = 32
OUTPUT_CLASSES = 2

PLATFORM_PATHS = {
    "production": PROJECT_ROOT / "models" / "production",
    "rpi_4": PROJECT_ROOT / "models" / "rpi_4",
    "rpi_zero": PROJECT_ROOT / "models" / "rpi_zero",
    "esp32": PROJECT_ROOT / "models" / "esp32",
}

# Feature names for 32-feature production model
PRODUCTION_FEATURE_NAMES = [
    "bytes_per_sec", "log_src_bytes", "src_bytes", "bytes_ratio", "dst_bytes",
    "log_dst_bytes", "service", "bytes_imbalance", "conn_intensity", "srv_conn_ratio",
    "dst_host_srv_ratio", "weighted_conn_score", "dst_host_activity", "flag",
    "same_srv_rate", "diff_srv_rate", "srv_diversity_score", "dst_host_srv_count",
    "dst_host_same_srv_rate", "packets_per_sec", "dst_host_diff_srv_rate", "logged_in",
    "count", "log_count", "serror_rate", "dst_host_srv_diff_host_rate", "dst_host_count",
    "dst_host_same_src_port_rate", "srv_diff_host_rate", "srv_conn_rate", "rerror_rate",
    "log_srv_count"
]


# =============================================================================
# Data Fixtures (Production 32 features)
# =============================================================================


@pytest.fixture
def sample_batch_32():
    """
    Random tensor batch for production model testing.
    Shape: (32, 32) to match 32 feature production model.
    """
    torch.manual_seed(42)
    return torch.randn(32, INPUT_DIM)


@pytest.fixture
def sample_batch_single():
    """Single sample for inference testing. Shape: (1, 32)."""
    torch.manual_seed(42)
    return torch.randn(1, INPUT_DIM)


@pytest.fixture
def sample_batch_large():
    """Large batch for performance testing. Shape: (1000, 32)."""
    torch.manual_seed(42)
    return torch.randn(1000, INPUT_DIM)


@pytest.fixture
def sample_labels_binary_32():
    """Binary labels for batch of 32. Normal(0) vs Attack(1)."""
    torch.manual_seed(42)
    return torch.randint(0, OUTPUT_CLASSES, (32,))


@pytest.fixture
def sample_numpy_32():
    """NumPy array for preprocessing tests. Shape: (100, 32)."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((100, INPUT_DIM)).astype(np.float32)
    y = rng.integers(0, OUTPUT_CLASSES, 100)
    return x, y


@pytest.fixture
def sample_dataframe_32():
    """DataFrame with production feature names."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((100, INPUT_DIM)).astype(np.float32)
    return pd.DataFrame(data, columns=PRODUCTION_FEATURE_NAMES)


# =============================================================================
# Data Fixtures (Legacy 41 features - NSL-KDD raw)
# =============================================================================


@pytest.fixture
def sample_batch():
    """
    Random tensor batch for testing.
    Shape: (32, 41) to match NSL-KDD feature count.
    """
    torch.manual_seed(42)
    return torch.randn(32, 41)


@pytest.fixture
def sample_batch_small():
    """Smaller batch for faster tests. Shape: (8, 41)."""
    torch.manual_seed(42)
    return torch.randn(8, 41)


@pytest.fixture
def sample_labels_binary():
    """Binary labels for Normal(0) vs Attack(1)."""
    torch.manual_seed(42)
    return torch.randint(0, 2, (32,))


@pytest.fixture
def sample_labels_multiclass():
    """5-class labels: Normal(0), DoS(1), Probe(2), R2L(3), U2R(4)."""
    torch.manual_seed(42)
    return torch.randint(0, 5, (32,))


@pytest.fixture
def sample_labels_family():
    """4-class family labels: DoS(0), Probe(1), R2L(2), U2R(3)."""
    torch.manual_seed(42)
    return torch.randint(0, 4, (32,))


@pytest.fixture
def sample_attack_logits():
    """Attack logits for attention conditioning. Shape: (32, 5)."""
    torch.manual_seed(42)
    return torch.randn(32, 5)


@pytest.fixture
def sample_numpy_data():
    """NumPy arrays for sklearn-based tests."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((100, 41)).astype(np.float32)
    y = rng.integers(0, 5, 100)
    return x, y


@pytest.fixture
def imbalanced_numpy_data():
    """Imbalanced dataset for augmentation tests."""
    rng = np.random.default_rng(42)
    # Create imbalanced distribution: Normal(60), DoS(25), Probe(10), R2L(4), U2R(1)
    x = rng.standard_normal((100, 41)).astype(np.float32)
    y = np.array([0] * 60 + [1] * 25 + [2] * 10 + [3] * 4 + [4] * 1)
    # Shuffle
    indices = rng.permutation(len(y))
    return x[indices], y[indices]


# =============================================================================
# Feature Engineering Fixtures
# =============================================================================


@pytest.fixture
def raw_network_data():
    """
    Simulated raw network flow data for feature engineering tests.
    Includes fields needed for rate and ratio calculations.
    """
    rng = np.random.default_rng(42)
    n_samples = 100

    data = {
        "src_bytes": rng.integers(0, 100000, n_samples).astype(float),
        "dst_bytes": rng.integers(0, 100000, n_samples).astype(float),
        "duration": rng.uniform(0.001, 10.0, n_samples),
        "count": rng.integers(1, 100, n_samples),
        "srv_count": rng.integers(1, 50, n_samples),
        "same_srv_rate": rng.uniform(0, 1, n_samples),
        "diff_srv_rate": rng.uniform(0, 1, n_samples),
        "serror_rate": rng.uniform(0, 1, n_samples),
        "rerror_rate": rng.uniform(0, 1, n_samples),
        "dst_host_count": rng.integers(1, 256, n_samples),
        "dst_host_srv_count": rng.integers(1, 256, n_samples),
        "dst_host_same_srv_rate": rng.uniform(0, 1, n_samples),
        "dst_host_diff_srv_rate": rng.uniform(0, 1, n_samples),
        "dst_host_srv_diff_host_rate": rng.uniform(0, 1, n_samples),
        "dst_host_same_src_port_rate": rng.uniform(0, 1, n_samples),
        "flag": rng.integers(0, 11, n_samples),
        "service": rng.integers(0, 70, n_samples),
        "logged_in": rng.integers(0, 2, n_samples),
    }
    return pd.DataFrame(data)


@pytest.fixture
def edge_case_network_data():
    """
    Network data with edge cases: zeros, very large values, NaN.
    """
    data = {
        "src_bytes": [0, 0, 1e9, np.nan, 100],
        "dst_bytes": [0, 100, 0, 1e9, np.nan],
        "duration": [0, 0.001, 10.0, np.nan, 0],
        "count": [0, 1, 1000, 50, np.nan],
        "srv_count": [0, 1, 500, 25, 10],
    }
    return pd.DataFrame(data)


# =============================================================================
# Model Fixtures
# =============================================================================


@pytest.fixture
def helix_nano_model():
    """Instantiated HELIX-Nano model."""
    from src.helix_ids.models.helix_ids import create_helix_model
    return create_helix_model("nano", input_dim=41)


@pytest.fixture
def helix_lite_model():
    """Instantiated HELIX-Lite model."""
    from src.helix_ids.models.helix_ids import create_helix_model
    return create_helix_model("lite", input_dim=41)


@pytest.fixture
def helix_full_model():
    """Instantiated HELIX-Full model."""
    from src.helix_ids.models.helix_ids import create_helix_model
    return create_helix_model("full", input_dim=41)


@pytest.fixture
def production_model_32():
    """
    Production model with 32 input features.
    Architecture: 32→64→32→16→2
    """
    from src.helix_ids.models.helix_ids import create_helix_model
    return create_helix_model("nano", input_dim=INPUT_DIM, num_classes=OUTPUT_CLASSES)


@pytest.fixture
def loaded_production_model():
    """
    Load the actual production model from disk.
    Returns None if model file doesn't exist.
    """
    if not PRODUCTION_MODEL_PATH.exists():
        pytest.skip(f"Production model not found at {PRODUCTION_MODEL_PATH}")

    from src.helix_ids.models.helix_ids import create_helix_model
    model = create_helix_model("nano", input_dim=INPUT_DIM, num_classes=OUTPUT_CLASSES)

    # Load state dict if file has content
    if PRODUCTION_MODEL_PATH.stat().st_size > 0:
        state_dict = torch.load(PRODUCTION_MODEL_PATH, map_location="cpu", weights_only=True)

        # Accept common checkpoint formats and tolerate partial/legacy keys.
        if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]

        if not isinstance(state_dict, dict):
            pytest.skip("Production model checkpoint format is not a state dict")

        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model


# =============================================================================
# Scaler Fixtures
# =============================================================================


@pytest.fixture
def production_scaler():
    """
    Load the production MinMaxScaler.
    Returns a fitted scaler or None if not found.
    """
    if not PRODUCTION_SCALER_PATH.exists():
        pytest.skip(f"Production scaler not found at {PRODUCTION_SCALER_PATH}")

    if PRODUCTION_SCALER_PATH.stat().st_size == 0:
        pytest.skip("Production scaler file is empty")

    with open(PRODUCTION_SCALER_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture
def fitted_minmax_scaler():
    """
    Create a fitted MinMaxScaler for testing.
    """
    from sklearn.preprocessing import MinMaxScaler

    rng = np.random.default_rng(42)
    X = rng.standard_normal((1000, INPUT_DIM))

    scaler = MinMaxScaler()
    scaler.fit(X)
    return scaler


@pytest.fixture
def production_feature_names():
    """Load production feature names from JSON."""
    if not PRODUCTION_FEATURE_NAMES_PATH.exists():
        return PRODUCTION_FEATURE_NAMES

    with open(PRODUCTION_FEATURE_NAMES_PATH) as f:
        return json.load(f)


@pytest.fixture
def model_card():
    """Load model card metadata."""
    if not PRODUCTION_MODEL_CARD_PATH.exists():
        pytest.skip(f"Model card not found at {PRODUCTION_MODEL_CARD_PATH}")

    with open(PRODUCTION_MODEL_CARD_PATH) as f:
        return json.load(f)


# =============================================================================
# Attention Module Fixtures
# =============================================================================


@pytest.fixture
def tam_nano():
    """TAM-Nano attention module."""
    from src.helix_ids.models.attention import TAMNano
    return TAMNano(n_features=41)


@pytest.fixture
def tam_lite():
    """TAM-Lite attention module."""
    from src.helix_ids.models.attention import TAMLite
    return TAMLite(n_features=41)


@pytest.fixture
def tam_full():
    """TAM-Full attention module."""
    from src.helix_ids.models.attention import TAMFull
    return TAMFull(n_features=41)


# =============================================================================
# Loss Function Fixtures
# =============================================================================


@pytest.fixture
def focal_loss():
    """ThreatAwareFocalLoss instance."""
    from src.helix_ids.models.loss import ThreatAwareFocalLoss
    return ThreatAwareFocalLoss(gamma=2.0)


@pytest.fixture
def multitask_loss():
    """MultiTaskLoss instance."""
    from src.helix_ids.models.loss import MultiTaskLoss
    return MultiTaskLoss()


# =============================================================================
# Classifier Fixtures
# =============================================================================


@pytest.fixture
def hierarchical_classifier():
    """HierarchicalClassifier with default config."""
    from src.helix_ids.models.classifier import HierarchicalClassifier
    return HierarchicalClassifier(input_dim=48)


@pytest.fixture
def hierarchical_classifier_nano():
    """HierarchicalClassifierNano."""
    from src.helix_ids.models.classifier import HierarchicalClassifierNano
    return HierarchicalClassifierNano(input_dim=32)


# =============================================================================
# Preprocessing Fixtures
# =============================================================================


@pytest.fixture
def preprocessor():
    """DataPreprocessor instance."""
    from src.helix_ids.data.preprocessing import DataPreprocessor
    return DataPreprocessor()


@pytest.fixture
def augmentation_config():
    """AugmentationConfig instance."""
    from src.helix_ids.data.augmentation import AugmentationConfig
    return AugmentationConfig()


# =============================================================================
# Data Loading Fixtures
# =============================================================================


@pytest.fixture
def nsl_kdd_data_path():
    """Path to NSL-KDD dataset."""
    path = PROJECT_ROOT / "data" / "nsl_kdd"
    if not path.exists():
        pytest.skip(f"NSL-KDD data not found at {path}")
    return path


@pytest.fixture
def unsw_nb15_data_path():
    """Path to UNSW-NB15 dataset."""
    path = PROJECT_ROOT / "data" / "unsw_nb15"
    if not path.exists():
        pytest.skip(f"UNSW-NB15 data not found at {path}")
    return path


@pytest.fixture
def processed_data_path():
    """Path to processed data directory."""
    path = PROJECT_ROOT / "data" / "processed"
    if not path.exists():
        pytest.skip(f"Processed data not found at {path}")
    return path


# =============================================================================
# Callback Fixtures
# =============================================================================


@pytest.fixture
def early_stopping_callback():
    """EarlyStopping callback."""
    from src.helix_ids.utils.callbacks import EarlyStopping
    return EarlyStopping(monitor="loss", patience=3, mode="min", verbose=False)


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def device():
    """Get appropriate device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def class_names():
    """Standard 5-class names."""
    return ["Normal", "DoS", "Probe", "R2L", "U2R"]


@pytest.fixture
def binary_class_names():
    """Binary class names."""
    return ["Normal", "Attack"]


@pytest.fixture
def platform_model_paths():
    """Dictionary of platform model paths."""
    return PLATFORM_PATHS


@pytest.fixture
def project_root():
    """Project root path."""
    return PROJECT_ROOT
