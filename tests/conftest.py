"""pytest conftest — register Hypothesis 'dev' (fast), 'ci', and 'thorough' profiles + shared fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch


def pytest_configure(config):
    """Register Hypothesis profiles for different environments.

    Use: pytest --hypothesis-profile=ci
         pytest --hypothesis-profile=thorough (1000 examples)
         pytest --hypothesis-profile=dev (default, 50 examples)
    """
    import hypothesis

    hypothesis.settings.register_profile(
        "ci",
        max_examples=50,
        deadline=1000,
        suppress_health_check=list(hypothesis.HealthCheck),
    )
    hypothesis.settings.register_profile(
        "thorough",
        max_examples=1000,
        deadline=None,
        suppress_health_check=list(hypothesis.HealthCheck),
    )
    hypothesis.settings.register_profile(
        "dev",
        max_examples=50,
        deadline=None,
        suppress_health_check=list(hypothesis.HealthCheck),
    )


# ── Shared fixtures for feature-engineering tests ─────────────────────────────


@pytest.fixture
def sample_batch() -> torch.Tensor:
    """A small random input tensor for helix_ids model inference tests."""
    return torch.randn(32, 41)


@pytest.fixture
def sample_batch_small() -> torch.Tensor:
    """A tiny batch (4 samples) for ensemble escalation tests."""
    return torch.randn(4, 41)


@pytest.fixture
def raw_network_data() -> pd.DataFrame:
    """A small realistic DataFrame of raw network connection features."""
    return pd.DataFrame({
        "src_bytes": [500.0, 1200.0, 80.0, 9999.0, 0.0],
        "dst_bytes": [200.0, 800.0, 40.0, 500.0, 0.0],
        "duration": [10.0, 5.0, 1.0, 30.0, 0.0],
        "count": [10, 25, 3, 100, 0],
        "srv_count": [5, 12, 1, 50, 0],
    })


@pytest.fixture
def production_feature_names() -> list[str]:
    """The canonical set of 32 engineered feature names."""
    return [
        "duration",
        "src_bytes",
        "dst_bytes",
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
        "bytes_per_sec",
        "packets_per_sec",
        "bytes_ratio",
        "bytes_imbalance",
        "log_src_bytes",
        "log_dst_bytes",
        "log_count",
        "log_srv_count",
        "protocol_type",
        "service",
    ]


@pytest.fixture
def edge_case_network_data() -> pd.DataFrame:
    """A small DataFrame with NaN values in network columns."""
    return pd.DataFrame({
        "src_bytes": [float("nan"), 500.0, 1000.0],
        "dst_bytes": [200.0, float("nan"), 300.0],
        "duration": [5.0, float("nan"), 10.0],
        "count": [10, 20, float("nan")],
    })


# ── Shared fixtures for preprocessing tests ───────────────────────────────────


@pytest.fixture
def project_root() -> Path:
    """The repository root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def production_scaler() -> Any:
    """A scikit-learn MinMaxScaler (unfitted)."""
    from sklearn.preprocessing import MinMaxScaler

    return MinMaxScaler()


@pytest.fixture
def fitted_minmax_scaler() -> Any:
    """A fitted MinMaxScaler trained on random data with 32 features."""
    from sklearn.preprocessing import MinMaxScaler

    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 32)).astype(np.float64)
    scaler = MinMaxScaler()
    scaler.fit(X)
    return scaler


@pytest.fixture
def sample_numpy_32() -> tuple[np.ndarray, np.ndarray]:
    """A random (X, y) pair with 32 features and 50 samples."""
    rng = np.random.default_rng(99)
    X = rng.standard_normal((50, 32)).astype(np.float32)
    y = rng.integers(0, 2, size=(50,)).astype(np.int64)
    return X, y


@pytest.fixture
def sample_numpy_data() -> tuple[np.ndarray, np.ndarray]:
    """A random (X, y) pair with 32 features and 100 samples."""
    rng = np.random.default_rng(123)
    X = rng.standard_normal((100, 32)).astype(np.float32)
    y = rng.integers(0, 2, size=(100,)).astype(np.int64)
    return X, y


@pytest.fixture
def preprocessor() -> Any:
    """A DataPreprocessor instance."""
    from helix_ids.data.preprocessing import DataPreprocessor

    return DataPreprocessor()


# ── Shared fixtures for model inference tests ────────────────────────────────


@pytest.fixture
def sample_labels_multiclass() -> torch.Tensor:
    """A 1-D tensor of 32 integer class labels (0-4) for loss tests."""
    return torch.randint(0, 5, (32,))


@pytest.fixture
def sample_batch_single() -> torch.Tensor:
    """A single-sample input tensor (1, 32) for inference tests."""
    return torch.randn(1, 32)


@pytest.fixture
def sample_batch_32() -> torch.Tensor:
    """A small-batch input tensor (32, 32) for inference tests."""
    return torch.randn(32, 32)


@pytest.fixture
def sample_batch_large() -> torch.Tensor:
    """A large-batch input tensor (1000, 32) for throughput tests."""
    return torch.randn(1000, 32)


@pytest.fixture
def platform_model_paths() -> dict[str, Path]:
    """Platform model directory paths relative to the project root."""
    root = Path(__file__).parent.parent
    return {
        "production": root / "models" / "production",
        "rpi_4": root / "models" / "rpi_4",
        "rpi_zero": root / "models" / "rpi_zero",
        "esp32": root / "models" / "esp32",
    }


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Return the available compute device (CUDA if available, else CPU)."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
