"""Critical invariance-path tests for multi-dataset training and evaluation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from helix_ids.utils.entropy_diagnostics import calculate_entropy_stable
from helix_ids.utils.metrics import compute_macro_f1
from scripts.training import train_helix_ids_full as train_mod


class _NoopModel:
    def eval(self) -> None:
        return None


def _make_trainer() -> train_mod.HelixFullTrainer:
    trainer = train_mod.HelixFullTrainer.__new__(train_mod.HelixFullTrainer)
    trainer.model = _NoopModel()
    trainer.logger = logging.getLogger("critical_invariance_tests")
    return trainer


def test_validate_uses_per_dataset_worst_case_not_averaging() -> None:
    trainer = _make_trainer()

    # Deliberately imbalanced sample counts to catch accidental weighted averaging.
    metrics_by_loader = {
        "loader_nsl": {
            "num_samples": 10000.0,
            "val_loss": 0.20,
            "val_calibrated_loss": 0.21,
            "val_binary_acc": 0.95,
            "val_family_acc": 0.91,
            "val_binary_auroc": 0.97,
            "val_binary_auprc": 0.96,
            "val_family_macro_f1": 0.92,
            "val_family_minority_recall_min": 0.82,
            "val_family_entropy": 0.45,
            "val_family_zero_prediction_classes": 0.0,
        },
        "loader_unsw": {
            "num_samples": 50.0,
            "val_loss": 0.70,
            "val_calibrated_loss": 0.72,
            "val_binary_acc": 0.62,
            "val_family_acc": 0.59,
            "val_binary_auroc": 0.61,
            "val_binary_auprc": 0.58,
            "val_family_macro_f1": 0.57,
            "val_family_minority_recall_min": 0.33,
            "val_family_entropy": 0.22,
            "val_family_zero_prediction_classes": 1.0,
        },
        "loader_cicids": {
            "num_samples": 200.0,
            "val_loss": 0.35,
            "val_calibrated_loss": 0.40,
            "val_binary_acc": 0.77,
            "val_family_acc": 0.74,
            "val_binary_auroc": 0.79,
            "val_binary_auprc": 0.73,
            "val_family_macro_f1": 0.71,
            "val_family_minority_recall_min": 0.55,
            "val_family_entropy": 0.31,
            "val_family_zero_prediction_classes": 0.0,
        },
    }

    trainer.val_loaders = {
        "nsl_kdd": "loader_nsl",
        "unsw_nb15": "loader_unsw",
        "cicids": "loader_cicids",
    }
    trainer._evaluate_loader = lambda loader: metrics_by_loader[loader]

    aggregated = train_mod.HelixFullTrainer.validate(trainer)

    nsl_f1 = metrics_by_loader["loader_nsl"]["val_family_macro_f1"]
    unsw_f1 = metrics_by_loader["loader_unsw"]["val_family_macro_f1"]
    cicids_f1 = metrics_by_loader["loader_cicids"]["val_family_macro_f1"]
    worst_case_f1 = aggregated["val_family_macro_f1"]

    # Required explicit invariant for strict worst-case evaluation.
    assert worst_case_f1 == min(nsl_f1, unsw_f1, cicids_f1)

    # Ensure no sample-count averaging path is used.
    weighted_mean = (
        nsl_f1 * metrics_by_loader["loader_nsl"]["num_samples"]
        + unsw_f1 * metrics_by_loader["loader_unsw"]["num_samples"]
        + cicids_f1 * metrics_by_loader["loader_cicids"]["num_samples"]
    ) / (
        metrics_by_loader["loader_nsl"]["num_samples"]
        + metrics_by_loader["loader_unsw"]["num_samples"]
        + metrics_by_loader["loader_cicids"]["num_samples"]
    )
    assert not np.isclose(worst_case_f1, weighted_mean)


def test_evaluate_per_dataset_is_independent() -> None:
    trainer = _make_trainer()
    trainer.test_loaders = {
        "nsl_kdd": "test_nsl",
        "unsw_nb15": "test_unsw",
        "cicids": "test_cicids",
    }

    per_loader = {
        "test_nsl": {"family_macro_f1": 0.88},
        "test_unsw": {"family_macro_f1": 0.61},
        "test_cicids": {"family_macro_f1": 0.73},
    }
    trainer._evaluate_test_loader = lambda loader: per_loader[loader]

    results = train_mod.HelixFullTrainer.evaluate_per_dataset(trainer)

    assert set(results) == {"nsl_kdd", "unsw_nb15", "cicids"}
    assert results["nsl_kdd"]["family_macro_f1"] == 0.88
    assert results["unsw_nb15"]["family_macro_f1"] == 0.61
    assert results["cicids"]["family_macro_f1"] == 0.73


def test_macro_f1_matches_sklearn_fixed_input() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2, 2, 1, 0])
    y_pred = np.array([0, 1, 2, 1, 1, 0, 2, 0, 0])

    expected = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    actual = compute_macro_f1(y_true, y_pred)

    assert actual == expected


def test_entropy_uniform_higher_than_confident() -> None:
    uniform_probs = np.full((32, 4), 0.25, dtype=np.float64)
    confident_probs = np.tile(np.array([0.99, 0.003, 0.003, 0.004], dtype=np.float64), (32, 1))

    uniform_entropy = calculate_entropy_stable(uniform_probs)
    confident_entropy = calculate_entropy_stable(confident_probs)

    assert float(uniform_entropy.mean()) > float(confident_entropy.mean())


def test_minority_recall_correct_when_present() -> None:
    confusion = torch.tensor(
        [
            [40, 5, 0],
            [2, 8, 0],
            [1, 0, 4],
        ],
        dtype=torch.int64,
    )

    stats = train_mod.HelixFullTrainer._compute_f1_stats_from_confusion(confusion)
    assert stats["minority_recall_min"] == 0.8


def test_minority_recall_zero_when_minority_not_predicted() -> None:
    confusion = torch.tensor(
        [
            [50, 0, 0],
            [10, 0, 0],
            [5, 0, 0],
        ],
        dtype=torch.int64,
    )

    stats = train_mod.HelixFullTrainer._compute_f1_stats_from_confusion(confusion)
    assert stats["minority_recall_min"] == 0.0
    assert stats["zero_prediction_classes"] == [1, 2]


def test_precomputed_split_cache_invalidated_on_feature_dim_change(tmp_path: Path) -> None:
    required_arrays = {
        "X_train.npy": np.zeros((10, 18), dtype=np.float32),
        "y_train.npy": np.zeros((10,), dtype=np.int64),
        "X_val.npy": np.zeros((4, 18), dtype=np.float32),
        "y_val.npy": np.zeros((4,), dtype=np.int64),
        "X_test_nsl_kdd.npy": np.zeros((4, 18), dtype=np.float32),
        "y_test_nsl_kdd.npy": np.zeros((4,), dtype=np.int64),
        "X_test_unsw_nb15.npy": np.zeros((4, 18), dtype=np.float32),
        "y_test_unsw_nb15.npy": np.zeros((4,), dtype=np.int64),
        "X_test_cicids.npy": np.zeros((4, 18), dtype=np.float32),
        "y_test_cicids.npy": np.zeros((4,), dtype=np.int64),
    }

    for name, arr in required_arrays.items():
        np.save(tmp_path / name, arr)

    loaded = train_mod._load_precomputed_splits(
        splits_dir=tmp_path,
        logger=logging.getLogger("cache_dim_guard"),
        expected_feature_dim=19,
    )

    assert loaded is None


def test_eval_array_falls_back_when_cached_dim_mismatch(tmp_path: Path, monkeypatch) -> None:
    fallback = np.ones((3, 19), dtype=np.float32)
    splits = {"X_val_nsl_kdd": fallback}

    cache_dir = tmp_path / "data" / "processed" / "multi_dataset_v1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "X_val_nsl_kdd.npy", np.zeros((3, 18), dtype=np.float32))

    monkeypatch.setattr(train_mod, "PROJECT_ROOT", tmp_path)

    loaded = train_mod._load_eval_array(
        splits=splits,
        dataset_name="nsl_kdd",
        split_name="val",
        prefix="X",
        logger=logging.getLogger("eval_array_guard"),
        expected_feature_dim=19,
    )

    assert loaded.shape[1] == 19
    assert np.array_equal(np.asarray(loaded), fallback)
