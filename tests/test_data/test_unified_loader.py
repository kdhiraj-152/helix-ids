import numpy as np
import pandas as pd

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
