from typing import cast

import numpy as np
import pandas as pd
import pytest

from src.helix_ids.data.dataset_config import CICIDS_2018_CONFIG
from src.helix_ids.data.label_mapping import _map_cicids_labels, encode_labels
from src.helix_ids.data.loader_core import UnifiedDataLoader


def test_map_cicids_2018_labels_variants():
    labels = np.array(
        [
            "Benign",
            "DDOS attack-HOIC",
            "DoS attacks-Hulk",
            "Brute Force -Web",
            "Brute Force -XSS",
            "SQL Injection",
            "Infilteration",
            "FTP-BruteForce",
            "SSH-Bruteforce",
        ]
    )

    mapped = _map_cicids_labels(labels)

    assert mapped.tolist() == [
        "BENIGN",
        "DDoS",
        "DoS Hulk",
        "Web Attack - Brute Force",
        "Web Attack - XSS",
        "Web Attack - Sql Injection",
        "Infiltration",
        "FTP-Patator",
        "SSH-Patator",
    ]


def test_extract_features_labels_skips_embedded_header_rows():
    loader = UnifiedDataLoader(verbose=False)

    df = pd.DataFrame(
        {
            "Flow ID": ["f1", "f2", "f3"],
            "f_a": [1.0, 2.0, 3.0],
            "f_b": [4.0, 5.0, 6.0],
            "Label": ["Benign", "Label", "Bot"],
        }
    )

    X, y = loader._extract_features_labels(df, CICIDS_2018_CONFIG)

    assert len(y) == 2
    assert y.tolist() == ["Benign", "Bot"]
    assert X.shape[0] == 2


def test_unified_5class_encoding_has_stable_order():
    y = np.array(["DoS", "Normal", "U2R", "R2L", "Probe"])
    encoded, class_names, _ = encode_labels(y, CICIDS_2018_CONFIG, label_mode="unified_5class")

    assert class_names == ["Normal", "DoS", "Probe", "R2L", "U2R"]
    assert encoded.tolist() == [1, 0, 4, 3, 2]


@pytest.fixture
def nsl_fixture_df() -> pd.DataFrame:
    labels = ["normal", "neptune", "satan", "warezclient", "buffer_overflow"] * 6
    rows = len(labels)
    return pd.DataFrame(
        {
            "duration": np.linspace(0.0, 1.0, rows),
            "src_bytes": np.linspace(1.0, 10.0, rows),
            "dst_bytes": np.linspace(2.0, 11.0, rows),
            "label": labels,
        }
    )


def test_load_can_return_domain_ids(monkeypatch: pytest.MonkeyPatch, nsl_fixture_df: pd.DataFrame):
    loader = UnifiedDataLoader(scale_features=False, handle_outliers=False, handle_missing=False)

    def _fake_load_dataframes(self, config, split):
        return nsl_fixture_df.copy()

    monkeypatch.setattr(UnifiedDataLoader, "_load_dataframes", _fake_load_dataframes)

    X, y, class_names, domain_ids = loader.load_with_domain_ids(
        "nsl-kdd",
        return_class_names=True,
    )

    assert X.shape[0] == y.shape[0] == domain_ids.shape[0]
    assert class_names == ["Normal", "DoS", "Probe", "R2L", "U2R"]
    assert domain_ids.dtype == np.int64
    assert np.unique(domain_ids).tolist() == [0]


def test_get_splits_can_include_domain_ids(
    monkeypatch: pytest.MonkeyPatch,
    nsl_fixture_df: pd.DataFrame,
):
    loader = UnifiedDataLoader(scale_features=False, handle_outliers=False, handle_missing=False)

    def _fake_load_dataframes(self, config, split):
        return nsl_fixture_df.copy()

    monkeypatch.setattr(UnifiedDataLoader, "_load_dataframes", _fake_load_dataframes)

    splits = loader.get_splits(
        "nsl-kdd",
        test_size=0.2,
        val_size=0.2,
        random_state=7,
        include_domain_id=True,
    )

    x_train, y_train, d_train = cast(tuple[np.ndarray, np.ndarray, np.ndarray], splits["train"])
    x_val, y_val, d_val = cast(tuple[np.ndarray, np.ndarray, np.ndarray], splits["val"])
    x_test, y_test, d_test = cast(tuple[np.ndarray, np.ndarray, np.ndarray], splits["test"])

    assert x_train.shape[0] == y_train.shape[0] == d_train.shape[0]
    assert x_val.shape[0] == y_val.shape[0] == d_val.shape[0]
    assert x_test.shape[0] == y_test.shape[0] == d_test.shape[0]
    assert np.unique(np.concatenate([d_train, d_val, d_test])).tolist() == [0]
