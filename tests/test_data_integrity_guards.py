"""Unit tests for data-integrity guards in multi-dataset splitting."""

from pathlib import Path

import numpy as np
import pandas as pd

from helix_ids.data.multi_dataset_loader import MultiDatasetLoader
from helix_ids.data.feature_harmonization import COMMON_FEATURES


def _loader() -> MultiDatasetLoader:
    project_root = Path(__file__).resolve().parents[1]
    return MultiDatasetLoader(project_root=project_root, random_state=42)


def test_downsample_majority_class_caps_ratio() -> None:
    loader = _loader()
    rng = np.random.default_rng(42)
    x = rng.normal(size=(100, 4)).astype(np.float32)
    y = np.array([0] * 95 + [1] * 5, dtype=np.int64)

    x_balanced, y_balanced = loader._downsample_majority_class(
        x,
        y,
        max_majority_ratio=0.90,
    )

    assert x_balanced.shape[0] == y_balanced.shape[0]
    counts = np.bincount(y_balanced)
    assert counts.max() / y_balanced.size <= 0.90
    # Minority class must be retained after downsampling.
    assert np.any(y_balanced == 1)


def test_remove_cross_split_overlap_removes_duplicates() -> None:
    loader = _loader()

    x_train = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    y_train = np.array([0, 1], dtype=np.int64)

    x_val = np.array([[0.1, 0.2], [0.5, 0.6]], dtype=np.float32)
    y_val = np.array([0, 1], dtype=np.int64)

    x_test = np.array([[0.3, 0.4], [0.7, 0.8]], dtype=np.float32)
    y_test = np.array([1, 0], dtype=np.int64)

    _, _, x_val_clean, y_val_clean, x_test_clean, y_test_clean = loader._remove_cross_split_overlap(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        dataset_name="synthetic",
    )

    train_fp = loader._fingerprint_rows(x_train, y_train)
    val_fp = loader._fingerprint_rows(x_val_clean, y_val_clean)
    test_fp = loader._fingerprint_rows(x_test_clean, y_test_clean)

    assert not np.isin(val_fp, train_fp).any()
    assert not np.isin(test_fp, train_fp).any()
    assert not np.isin(test_fp, val_fp).any()


def test_build_group_keys_uses_coarse_fingerprint() -> None:
    loader = _loader()
    x = np.array(
        [
            [1.1111, 2.2222, 3.3333],
            [1.1112, 2.2221, 3.3334],
            [9.0, 9.0, 9.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 1], dtype=np.int64)

    keys = loader._build_group_keys(x, y, dataset_code=0)

    assert keys[0] == keys[1]
    assert keys[0] != keys[2]


def test_create_splits_has_no_train_val_test_overlap() -> None:
    loader = _loader()
    rng = np.random.default_rng(123)

    rows = 900
    df = pd.DataFrame(
        {feat: rng.normal(loc=0.0, scale=1.0, size=rows) for feat in COMMON_FEATURES}
    )
    y = np.array([0] * 300 + [1] * 300 + [2] * 300, dtype=np.int64)
    rng.shuffle(y)
    df["label"] = y

    splits = loader.create_splits([df], test_size=0.2, val_size=0.2)

    train_fp = loader._fingerprint_rows(splits["X_train_nsl_kdd"], splits["y_train_nsl_kdd"])
    val_fp = loader._fingerprint_rows(splits["X_val_nsl_kdd"], splits["y_val_nsl_kdd"])
    test_fp = loader._fingerprint_rows(splits["X_test_nsl_kdd"], splits["y_test_nsl_kdd"])

    assert not np.isin(train_fp, val_fp).any()
    assert not np.isin(train_fp, test_fp).any()
    assert not np.isin(val_fp, test_fp).any()


def test_create_splits_preserves_class_distribution() -> None:
    loader = _loader()
    rng = np.random.default_rng(99)

    rows = 1200
    df = pd.DataFrame(
        {feat: rng.uniform(0.0, 50.0, size=rows) for feat in COMMON_FEATURES}
    )
    y = np.array([0] * 720 + [1] * 300 + [2] * 180, dtype=np.int64)
    rng.shuffle(y)
    df["label"] = y

    splits = loader.create_splits([df], test_size=0.2, val_size=0.2)

    original = np.bincount(y) / y.size

    for key in ("y_train_nsl_kdd", "y_val_nsl_kdd", "y_test_nsl_kdd"):
        y_split = splits[key]
        split_dist = np.bincount(y_split, minlength=original.size) / y_split.size
        # Stratification tolerance for finite sample splits.
        assert np.all(np.abs(split_dist - original) < 0.08), f"distribution drift in {key}"
