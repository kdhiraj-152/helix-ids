"""Critical invariance-path tests for multi-dataset training and evaluation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import f1_score
from typing import Any, cast

from helix_ids.utils.entropy_diagnostics import calculate_entropy_stable
from helix_ids.utils.metrics import compute_macro_f1
from helix_ids.data.multi_dataset_loader import MultiDatasetLoader
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

    trainer.val_loaders = cast(dict[str, Any], {
        "nsl_kdd": "loader_nsl",
        "unsw_nb15": "loader_unsw",
        "cicids": "loader_cicids",
    })
    object.__setattr__(
        trainer,
        "_evaluate_loader",
        lambda loader, dataset_name="unknown": metrics_by_loader[loader],
    )

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
    trainer.test_loaders = cast(dict[str, Any], {
        "nsl_kdd": "test_nsl",
        "unsw_nb15": "test_unsw",
        "cicids": "test_cicids",
    })

    per_loader = {
        "test_nsl": {"family_macro_f1": 0.88},
        "test_unsw": {"family_macro_f1": 0.61},
        "test_cicids": {"family_macro_f1": 0.73},
    }
    object.__setattr__(trainer, "_evaluate_test_loader", lambda loader: per_loader[loader])

    results = train_mod.HelixFullTrainer.evaluate_per_dataset(trainer)

    assert set(results) == {"nsl_kdd", "unsw_nb15", "cicids"}
    assert results["nsl_kdd"]["family_macro_f1"] == pytest.approx(0.88)
    assert results["unsw_nb15"]["family_macro_f1"] == pytest.approx(0.61)
    assert results["cicids"]["family_macro_f1"] == pytest.approx(0.73)


def test_macro_f1_matches_sklearn_fixed_input() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2, 2, 1, 0])
    y_pred = np.array([0, 1, 2, 1, 1, 0, 2, 0, 0])

    expected = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    actual = compute_macro_f1(y_true, y_pred)

    assert actual == expected


def test_train_inference_parity() -> None:
    snapshot_path = Path(__file__).resolve().parent / "fixtures" / "cicids_snapshot.csv"
    snapshot = pd.read_csv(snapshot_path)

    loader = MultiDatasetLoader()
    train_sample = snapshot.iloc[:2].copy()
    infer_sample = snapshot.iloc[1:].copy()

    train_out = loader.harmonize_cicids(train_sample)
    infer_out = loader.harmonize_cicids(infer_sample)

    assert train_out.shape[1] == infer_out.shape[1]
    assert list(train_out.columns) == list(infer_out.columns)


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
    assert stats["minority_recall_min"] == pytest.approx(0.8)


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
    assert stats["minority_recall_min"] == pytest.approx(0.0, abs=1e-9)
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
        np.save(str(tmp_path / name), cast(Any, arr))

    loaded = train_mod._load_precomputed_splits(
        splits_dir=tmp_path,
        logger=logging.getLogger("cache_dim_guard"),
        expected_feature_dim=17,
    )

    assert loaded is None


def test_eval_array_falls_back_when_cached_dim_mismatch(tmp_path: Path, monkeypatch) -> None:
    fallback = np.ones((3, 17), dtype=np.float32)
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
        expected_feature_dim=17,
    )

    assert loaded.shape[1] == 17
    assert np.array_equal(np.asarray(loaded), fallback)


def test_high_accuracy_high_loss_guard_requires_two_consecutive_epochs() -> None:
    trainer = _make_trainer()
    trainer.val_gap_collapse_streak = 0
    trainer.entropy_collapse_streak = 0
    trainer.high_accuracy_high_loss_streak = 0
    trainer.epoch = 0

    train_metrics = {
        "train_calibrated_loss": 0.88,
        "train_binary_acc": 0.97,
        "train_family_acc": 0.90,
    }
    val_metrics = {
        "val_calibrated_loss": 0.41,
        "val_binary_acc": 0.96,
        "val_family_acc": 0.89,
        "val_family_macro_f1": 0.82,
        "val_family_minority_recall_min": 0.45,
        "val_family_entropy": 0.28,
        "val_entropy_missing_same_dataset": 0.0,
    }

    first = train_mod.HelixFullTrainer._hard_stop_reason(trainer, train_metrics, val_metrics)
    assert first is None
    assert trainer.high_accuracy_high_loss_streak == 1

    trainer.epoch = 1
    second = train_mod.HelixFullTrainer._hard_stop_reason(trainer, train_metrics, val_metrics)
    assert second == "high_accuracy_with_high_loss"
    assert trainer.high_accuracy_high_loss_streak == 2


def test_high_accuracy_high_loss_streak_resets_when_signal_clears() -> None:
    trainer = _make_trainer()
    trainer.val_gap_collapse_streak = 0
    trainer.entropy_collapse_streak = 0
    trainer.high_accuracy_high_loss_streak = 0
    trainer.epoch = 0

    train_metrics_bad = {
        "train_calibrated_loss": 0.86,
        "train_binary_acc": 0.96,
        "train_family_acc": 0.91,
    }
    val_metrics_common = {
        "val_calibrated_loss": 0.40,
        "val_binary_acc": 0.95,
        "val_family_acc": 0.88,
        "val_family_macro_f1": 0.80,
        "val_family_minority_recall_min": 0.44,
        "val_family_entropy": 0.27,
        "val_entropy_missing_same_dataset": 0.0,
    }
    train_mod.HelixFullTrainer._hard_stop_reason(trainer, train_metrics_bad, val_metrics_common)
    assert trainer.high_accuracy_high_loss_streak == 1

    train_metrics_good = {
        "train_calibrated_loss": 0.42,
        "train_binary_acc": 0.96,
        "train_family_acc": 0.91,
    }
    reason = train_mod.HelixFullTrainer._hard_stop_reason(
        trainer,
        train_metrics_good,
        val_metrics_common,
    )
    assert reason is None
    assert trainer.high_accuracy_high_loss_streak == 0


def test_entropy_missing_class_guard_requires_two_consecutive_epochs() -> None:
    trainer = _make_trainer()
    trainer.val_gap_collapse_streak = 0
    trainer.entropy_collapse_streak = 0
    trainer.entropy_missing_class_streak = 0
    trainer.high_accuracy_high_loss_streak = 0
    trainer.config = cast(Any, type("Cfg", (), {"epochs": 150})())
    trainer.epoch = 1

    train_metrics = {
        "train_calibrated_loss": 0.40,
        "train_binary_acc": 0.84,
        "train_family_acc": 0.80,
    }
    val_metrics = {
        "val_calibrated_loss": 0.35,
        "val_binary_acc": 0.83,
        "val_family_acc": 0.78,
        "val_family_macro_f1": 0.55,
        "val_family_minority_recall_min": 0.30,
        "val_family_entropy": 0.11,
        "val_entropy_missing_same_dataset": 1.0,
    }

    first = train_mod.HelixFullTrainer._hard_stop_reason(trainer, train_metrics, val_metrics)
    assert first is None
    assert trainer.entropy_missing_class_streak == 1

    trainer.epoch = 2
    second = train_mod.HelixFullTrainer._hard_stop_reason(trainer, train_metrics, val_metrics)
    assert second == "prediction_entropy_collapse_with_missing_classes"
    assert trainer.entropy_missing_class_streak == 2


def test_entropy_missing_class_streak_resets_when_signal_clears() -> None:
    trainer = _make_trainer()
    trainer.val_gap_collapse_streak = 0
    trainer.entropy_collapse_streak = 0
    trainer.entropy_missing_class_streak = 0
    trainer.high_accuracy_high_loss_streak = 0
    trainer.config = cast(Any, type("Cfg", (), {"epochs": 150})())
    trainer.epoch = 1

    train_metrics = {
        "train_calibrated_loss": 0.40,
        "train_binary_acc": 0.84,
        "train_family_acc": 0.80,
    }
    val_metrics_bad = {
        "val_calibrated_loss": 0.35,
        "val_binary_acc": 0.83,
        "val_family_acc": 0.78,
        "val_family_macro_f1": 0.55,
        "val_family_minority_recall_min": 0.30,
        "val_family_entropy": 0.11,
        "val_entropy_missing_same_dataset": 1.0,
    }
    train_mod.HelixFullTrainer._hard_stop_reason(trainer, train_metrics, val_metrics_bad)
    assert trainer.entropy_missing_class_streak == 1

    val_metrics_good = dict(val_metrics_bad)
    val_metrics_good["val_family_entropy"] = 0.20
    val_metrics_good["val_entropy_missing_same_dataset"] = 0.0

    reason = train_mod.HelixFullTrainer._hard_stop_reason(
        trainer,
        train_metrics,
        val_metrics_good,
    )
    assert reason is None
    assert trainer.entropy_missing_class_streak == 0


def test_entropy_missing_class_guard_smoke_mode_requires_stronger_signal() -> None:
    trainer = _make_trainer()
    trainer.val_gap_collapse_streak = 0
    trainer.entropy_collapse_streak = 0
    trainer.entropy_missing_class_streak = 0
    trainer.high_accuracy_high_loss_streak = 0
    trainer.config = cast(Any, type("Cfg", (), {"epochs": 10})())
    trainer.epoch = 2

    train_metrics = {
        "train_calibrated_loss": 0.40,
        "train_binary_acc": 0.84,
        "train_family_acc": 0.80,
    }
    val_metrics_smoke_tolerated = {
        "val_calibrated_loss": 0.35,
        "val_binary_acc": 0.83,
        "val_family_acc": 0.78,
        "val_family_macro_f1": 0.55,
        "val_family_minority_recall_min": 0.30,
        "val_family_entropy": 0.11,
        "val_entropy_missing_same_dataset": 1.0,
    }

    first = train_mod.HelixFullTrainer._hard_stop_reason(
        trainer,
        train_metrics,
        val_metrics_smoke_tolerated,
    )
    assert first is None
    assert trainer.entropy_missing_class_streak == 0

    val_metrics_critical = dict(val_metrics_smoke_tolerated)
    val_metrics_critical["val_family_entropy"] = 0.09

    trainer.epoch = 3
    second = train_mod.HelixFullTrainer._hard_stop_reason(
        trainer,
        train_metrics,
        val_metrics_critical,
    )
    assert second is None
    assert trainer.entropy_missing_class_streak == 1

    trainer.epoch = 4
    third = train_mod.HelixFullTrainer._hard_stop_reason(
        trainer,
        train_metrics,
        val_metrics_critical,
    )
    assert third is None
    assert trainer.entropy_missing_class_streak == 2

    trainer.epoch = 5
    fourth = train_mod.HelixFullTrainer._hard_stop_reason(
        trainer,
        train_metrics,
        val_metrics_critical,
    )
    assert fourth == "prediction_entropy_collapse_with_missing_classes"
    assert trainer.entropy_missing_class_streak == 3


def test_post_training_macro_floor_smoke_budget() -> None:
    trainer = _make_trainer()
    trainer.config = cast(Any, type("Cfg", (), {"epochs": 10})())

    floor = train_mod.HelixFullTrainer._post_training_macro_floor(trainer)
    assert floor == pytest.approx(0.15)


def test_post_training_macro_floor_full_budget() -> None:
    trainer = _make_trainer()
    trainer.config = cast(Any, type("Cfg", (), {"epochs": 150})())

    floor = train_mod.HelixFullTrainer._post_training_macro_floor(trainer)
    assert floor == pytest.approx(0.25)


def test_resolve_governance_policy_smoke_budget_relaxes_ci_lower_bound() -> None:
    cfg = cast(Any, type("Cfg", (), {"epochs": 10})())

    policy = train_mod._resolve_governance_policy(cfg)
    assert policy.bootstrap.min_ci95_lower_bound == pytest.approx(0.15)
    assert policy.drift.max_abs_z_score == pytest.approx(50.0)
    assert policy.promotion.min_seed_runs == 1


def test_resolve_governance_policy_full_budget_keeps_strict_ci_lower_bound() -> None:
    cfg = cast(Any, type("Cfg", (), {"epochs": 150})())

    policy = train_mod._resolve_governance_policy(cfg)
    assert policy.bootstrap.min_ci95_lower_bound == pytest.approx(0.50)
    assert policy.drift.max_abs_z_score == pytest.approx(2.5)
    assert policy.promotion.min_seed_runs == 3


def test_resolve_class_balance_strategy_none_maps_to_unweighted_ce() -> None:
    strategy, use_class_weights = train_mod._resolve_class_balance_strategy("none")
    assert strategy == "weighted_ce"
    assert use_class_weights is False


def test_resolve_class_balance_strategy_keeps_existing_modes_weighted() -> None:
    weighted_strategy, weighted_use_weights = train_mod._resolve_class_balance_strategy("weighted_ce")
    sqrt_strategy, sqrt_use_weights = train_mod._resolve_class_balance_strategy("sqrt_weighted_ce")
    focal_strategy, focal_use_weights = train_mod._resolve_class_balance_strategy("focal")

    assert weighted_strategy == "weighted_ce"
    assert weighted_use_weights is True
    assert sqrt_strategy == "weighted_ce"
    assert sqrt_use_weights is True
    assert focal_strategy == "focal"
    assert focal_use_weights is True


def test_sqrt_inverse_frequency_weights_soften_extreme_tail_weighting() -> None:
    y = np.array([0] * 1000 + [1] * 10, dtype=np.int64)

    inv = train_mod._inverse_frequency_weights(y, minlength=2)
    sqrt_inv = train_mod._sqrt_inverse_frequency_weights(y, minlength=2)

    assert float(inv[1]) > float(sqrt_inv[1])
    assert float(inv[0]) < float(sqrt_inv[0])


def test_apply_disable_early_stopping_extends_patience_beyond_epoch_budget() -> None:
    cfg = cast(Any, type("Cfg", (), {"epochs": 50, "early_stopping_patience": 15})())

    train_mod._apply_disable_early_stopping(cfg, disable_early_stopping=True)

    assert int(cfg.early_stopping_patience) >= 51


def test_resolve_class_balance_strategy_invalid_raises_value_error() -> None:
    with pytest.raises(ValueError, match="--class-balance-strategy"):
        train_mod._resolve_class_balance_strategy("unsupported")


def test_sampler_mode_literal_accepts_weighted_random_sampler() -> None:
    mode_weighted = str("weighted_random_sampler").strip().lower()
    mode_interleaved = str("interleaved_rr").strip().lower()
    assert mode_weighted in {"interleaved_rr", "weighted_random_sampler"}
    assert mode_interleaved in {"interleaved_rr", "weighted_random_sampler"}


def test_interleaved_round_robin_indices_enforce_batch_diversity() -> None:
    y = np.array([0] * 8 + [1] * 8 + [4] * 2, dtype=np.int64)
    batch_size = 6
    idx = train_mod._build_interleaved_round_robin_indices(
        y,
        batch_size=batch_size,
        seed=42,
        min_unique_classes_per_batch=2,
    )

    assert idx.shape[0] % batch_size == 0
    y_drawn = y[idx]
    for start in range(0, idx.shape[0], batch_size):
        batch_labels = y_drawn[start : start + batch_size]
        assert int(np.unique(batch_labels).shape[0]) >= 2


def test_interleaved_round_robin_indices_enforce_class4_quota() -> None:
    y = np.array([0] * 24 + [1] * 24 + [2] * 12 + [4] * 16, dtype=np.int64)
    batch_size = 8
    idx = train_mod._build_interleaved_round_robin_indices(
        y,
        batch_size=batch_size,
        seed=123,
        min_unique_classes_per_batch=2,
        class4_min_per_batch=4,
    )

    y_drawn = y[idx]
    assert idx.shape[0] % batch_size == 0
    for start in range(0, idx.shape[0], batch_size):
        batch_labels = y_drawn[start : start + batch_size]
        assert int(np.sum(batch_labels == 4)) >= 4


def test_min_class4_sample_enforcement_upsamples_to_target() -> None:
    x = np.arange(20, dtype=np.float32).reshape(10, 2)
    y_family = np.array([0, 0, 0, 1, 1, 2, 2, 3, 4, 4], dtype=np.int64)
    y_binary = (y_family != 0).astype(np.int64)

    train_class_index = train_mod.build_class_index(y_family)
    class4_indices = np.asarray(train_class_index.get(4, np.array([], dtype=np.int64)), dtype=np.int64)
    assert int(class4_indices.size) == 2

    target = 5
    deficit = int(target - int(class4_indices.size))
    oversampled = np.random.default_rng(42).choice(class4_indices, size=deficit, replace=True).astype(np.int64)
    up_idx = np.concatenate([np.arange(int(y_family.shape[0]), dtype=np.int64), oversampled], axis=0)

    x_up = x[up_idx]
    y_family_up = y_family[up_idx]
    y_binary_up = y_binary[up_idx]

    assert x_up.shape[0] == 13
    assert y_binary_up.shape[0] == 13
    assert int(np.sum(y_family_up == 4)) == target


def test_ab_gate_rejects_when_geometry_regresses_even_if_macro_f1_improves() -> None:
    baseline = {
        "dataset_id": "ds_v1",
        "split_snapshot_id": "snap-1",
        "batch_size": 512,
        "eval_label_path": "cpu",
        "k": 3,
        "seed": 42,
        "feature_signature": "feat-a",
        "cluster_objective": "kmeans",
        "cluster_spectral_affinity": "nearest_neighbors",
        "ratio": 0.33,
        "min_inter": 0.90,
        "macro_f1": 0.80,
        "zero_prediction_classes": 0.0,
        "cluster_sizes": [100, 110, 120],
    }
    current = dict(baseline)
    current.update(
        {
            "cluster_objective": "gmm",
            "ratio": 0.41,  # Worse than baseline => hard reject.
            "min_inter": 0.88,
            "macro_f1": 0.92,
        }
    )

    decision = train_mod.evaluate_ab_candidate(
        current=current,
        baseline=baseline,
        ab_track="objective",
        governance_z_score=0.2,
        governance_z_tolerance=2.5,
    )
    assert decision["accepted"] is False
    assert decision["reason"] == "tier1_geometry_regression"
    assert decision["tier_3_evaluated"] is False


def test_ab_gate_rejects_mixed_feature_and_objective_changes() -> None:
    baseline = {
        "dataset_id": "ds_v1",
        "split_snapshot_id": "snap-1",
        "batch_size": 512,
        "eval_label_path": "cpu",
        "k": 3,
        "seed": 42,
        "feature_signature": "feat-a",
        "cluster_objective": "kmeans",
        "cluster_spectral_affinity": "nearest_neighbors",
        "ratio": 0.33,
        "min_inter": 0.90,
        "macro_f1": 0.80,
        "zero_prediction_classes": 0.0,
        "cluster_sizes": [100, 110, 120],
    }
    current = dict(baseline)
    current.update(
        {
            "feature_signature": "feat-b",
            "cluster_objective": "gmm",
            "ratio": 0.31,
            "min_inter": 0.95,
        }
    )

    decision = train_mod.evaluate_ab_candidate(
        current=current,
        baseline=baseline,
        ab_track="objective",
        governance_z_score=0.1,
        governance_z_tolerance=2.5,
    )
    assert decision["accepted"] is False
    assert decision["reason"] == "ab_anti_pattern_mixed_feature_and_objective_change"


def test_ab_gate_rejects_cluster_mode_collapse() -> None:
    baseline = {
        "dataset_id": "ds_v1",
        "split_snapshot_id": "snap-1",
        "batch_size": 512,
        "eval_label_path": "cpu",
        "k": 3,
        "seed": 42,
        "feature_signature": "feat-a",
        "cluster_objective": "kmeans",
        "cluster_spectral_affinity": "nearest_neighbors",
        "ratio": 0.40,
        "min_inter": 0.70,
        "macro_f1": 0.75,
        "zero_prediction_classes": 0.0,
        "cluster_sizes": [100, 100, 100],
    }
    current = dict(baseline)
    current.update(
        {
            "cluster_objective": "gmm",
            "ratio": 0.30,
            "min_inter": 0.85,
            "cluster_sizes": [290, 5, 5],  # Dominant single cluster.
            "macro_f1": 0.80,
        }
    )

    decision = train_mod.evaluate_ab_candidate(
        current=current,
        baseline=baseline,
        ab_track="objective",
        governance_z_score=0.1,
        governance_z_tolerance=2.5,
    )
    assert decision["accepted"] is False
    assert decision["reason"] == "tier2_cluster_mode_collapse"


def test_ab_gate_accepts_valid_objective_promotion() -> None:
    baseline = {
        "dataset_id": "ds_v1",
        "split_snapshot_id": "snap-1",
        "batch_size": 512,
        "eval_label_path": "cpu",
        "k": 3,
        "seed": 42,
        "feature_signature": "feat-a",
        "cluster_objective": "kmeans",
        "cluster_spectral_affinity": "nearest_neighbors",
        "ratio": 0.40,
        "min_inter": 0.70,
        "macro_f1": 0.75,
        "zero_prediction_classes": 0.0,
        "cluster_sizes": [100, 100, 100],
    }
    current = dict(baseline)
    current.update(
        {
            "cluster_objective": "gmm",
            "ratio": 0.30,
            "min_inter": 0.90,
            "cluster_sizes": [90, 100, 110],
            "macro_f1": 0.78,
        }
    )

    decision = train_mod.evaluate_ab_candidate(
        current=current,
        baseline=baseline,
        ab_track="objective",
        governance_z_score=0.2,
        governance_z_tolerance=2.5,
    )
    assert decision["accepted"] is True
    assert decision["tier_1_geometry_pass"] is True
    assert decision["tier_2_cluster_quality_pass"] is True
    assert decision["tier_3_classifier_pass"] is True
    assert decision["tier_4_governance_pass"] is True


def test_trainer_eval_class4_logit_shift_subtracts_only_target_column() -> None:
    trainer = _make_trainer()
    trainer.class4_logit_shift = 1.25
    trainer.class4_logit_shift_class_id = 4

    logits = torch.tensor(
        [
            [0.10, 0.20, 0.30, 0.40, 1.20, 0.60, 0.70],
            [0.90, 0.10, 0.10, 0.10, 1.00, 0.20, 0.10],
        ],
        dtype=torch.float32,
    )

    shifted = train_mod.HelixFullTrainer._apply_eval_class4_logit_shift(trainer, logits)

    assert shifted.shape == logits.shape
    assert float(shifted[0, 4].item()) == pytest.approx(-0.05)
    assert float(shifted[1, 4].item()) == pytest.approx(-0.25)
    assert torch.allclose(shifted[:, :4], logits[:, :4])
    assert torch.allclose(shifted[:, 5:], logits[:, 5:])


def test_apply_class4_logit_shift_subtracts_only_target_column() -> None:
    logits = np.array(
        [
            [0.10, 0.20, 0.30, 0.40, 1.20, 0.60, 0.70],
            [0.90, 0.10, 0.10, 0.10, 1.00, 0.20, 0.10],
        ],
        dtype=np.float64,
    )

    shifted = train_mod._apply_class4_logit_shift(logits, class4_id=4, delta=0.10)

    assert shifted.shape == logits.shape
    assert shifted[0, 4] == pytest.approx(1.10)
    assert shifted[1, 4] == pytest.approx(0.90)
    assert np.allclose(shifted[:, :4], logits[:, :4])
    assert np.allclose(shifted[:, 5:], logits[:, 5:])


def test_calibrate_family_predictions_class4_logit_shift_changes_uncalibrated_ranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = np.array([4, 0, 0, 0], dtype=np.int64)
    logits = np.array(
        [
            [0.10, 0.20, 0.30, 0.40, 1.20, 0.60, 0.70],
            [0.95, 0.10, 0.10, 0.10, 1.00, 0.20, 0.10],
            [0.95, 0.10, 0.10, 0.10, 1.00, 0.20, 0.10],
            [0.95, 0.10, 0.10, 0.10, 1.00, 0.20, 0.10],
        ],
        dtype=np.float64,
    )

    def _fake_collect_eval_family_outputs(**kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _ = kwargs
        probs = np.zeros_like(logits)
        return labels.copy(), logits.copy(), probs

    monkeypatch.setattr(train_mod, "_collect_eval_family_outputs", _fake_collect_eval_family_outputs)

    payload_no_shift = train_mod._calibrate_family_predictions(
        model=cast(Any, object()),
        val_loader=cast(Any, object()),
        test_loader=cast(Any, object()),
        device="cpu",
        class4_id=4,
        threshold_grid=np.array([0.5], dtype=np.float64),
        min_class4_recall=0.0,
        class4_logit_shift=0.0,
    )
    payload_shift = train_mod._calibrate_family_predictions(
        model=cast(Any, object()),
        val_loader=cast(Any, object()),
        test_loader=cast(Any, object()),
        device="cpu",
        class4_id=4,
        threshold_grid=np.array([0.5], dtype=np.float64),
        min_class4_recall=0.0,
        class4_logit_shift=0.1,
    )

    base_precision = float(payload_no_shift["uncalibrated"]["test_argmax"]["class4_precision"])
    shifted_precision = float(payload_shift["uncalibrated"]["test_argmax"]["class4_precision"])

    assert shifted_precision > base_precision
    assert float(payload_shift["class4_logit_shift"]) == pytest.approx(0.1)


def test_normalize_metrics_payload_maps_legacy_keys_to_strict_contract() -> None:
    metrics = {
        "family_macro_f1": 0.77,
        "family_class4_precision": 0.31,
        "family_class4_recall": 0.84,
        "mean_entropy": 0.29,
        "zero_prediction_classes": 0,
    }

    normalized = train_mod._normalize_metrics_payload(metrics)

    assert normalized == {
        "macro_f1": pytest.approx(0.77),
        "class4_precision": pytest.approx(0.31),
        "class4_recall": pytest.approx(0.84),
        "entropy": pytest.approx(0.29),
        "zero_prediction_classes": 0,
    }
    assert "family_macro_f1" not in normalized
    assert "mean_entropy" not in normalized


def test_normalize_calibration_block_enforces_required_paths(tmp_path: Path) -> None:
    pr = tmp_path / "pr_curve.csv"
    cm = tmp_path / "confusion_matrices.json"
    abl = tmp_path / "ablation.json"
    for path in (pr, cm, abl):
        path.write_text("{}", encoding="utf-8")

    normalized = train_mod._normalize_calibration_block(
        calibration_payload={"temperature": 2.0, "tau_4": 0.6},
        calibration_artifacts={
            "pr_curve_csv": str(pr),
            "confusion_matrices_json": str(cm),
            "ablation_json": str(abl),
        },
    )

    assert normalized == {
        "temperature": pytest.approx(2.0),
        "tau_4": pytest.approx(0.6),
        "pr_curve_path": str(pr),
        "confusion_matrix_path": str(cm),
        "ablation_path": str(abl),
    }


def test_materialize_phase8_artifacts_creates_required_filenames(tmp_path: Path) -> None:
    source_before_after_csv = tmp_path / "dataset_before_after_seed42.csv"
    source_before_after_json = tmp_path / "dataset_before_after_seed42.json"
    source_pr = tmp_path / "dataset_pr_curve_seed42.csv"
    source_conf = tmp_path / "dataset_confusion_seed42.json"
    source_abl = tmp_path / "dataset_ablation_seed42.json"

    source_before_after_csv.write_text("phase,macro_f1\n", encoding="utf-8")
    source_before_after_json.write_text("{}", encoding="utf-8")
    source_pr.write_text("point_index,precision\n", encoding="utf-8")
    source_conf.write_text("{}", encoding="utf-8")
    source_abl.write_text("{}", encoding="utf-8")

    canonical = train_mod._materialize_phase8_artifacts(
        {
            "before_after_csv": str(source_before_after_csv),
            "before_after_json": str(source_before_after_json),
            "pr_curve_csv": str(source_pr),
            "confusion_matrices_json": str(source_conf),
            "ablation_json": str(source_abl),
        }
    )

    assert Path(canonical["before_after_csv"]).name == "before_after.csv"
    assert Path(canonical["before_after_json"]).name == "before_after.json"
    assert Path(canonical["pr_curve_csv"]).name == "pr_curve.csv"
    assert Path(canonical["confusion_matrices_json"]).name == "confusion_matrices.json"
    assert Path(canonical["ablation_json"]).name == "ablation.json"
    for path in canonical.values():
        assert Path(path).exists()


def test_multiseed_governance_output_strict_schema(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    eval_by_seed: dict[int, dict[str, Any]] = {}
    train_by_seed: dict[int, dict[str, Any]] = {}
    for seed in (42, 1337, 2026):
        eval_by_seed[seed] = {
            "results": {
                "nsl_kdd": {
                    "family_macro_f1": 0.70,
                }
            }
        }
        train_by_seed[seed] = {"results": {}}

    class _DummyProc:
        def __init__(self) -> None:
            self.returncode = 0
            self.stderr = ""
            self.stdout = ""

    class _DummyModel:
        config = cast(Any, type("Cfg", (), {"family_output_dim": 7})())

        def load_state_dict(self, _state: dict[str, Any]) -> None:
            return None

    monkeypatch.setattr(train_mod.subprocess, "run", lambda *args, **kwargs: _DummyProc())

    def _fake_load_json_dict(path: Path) -> dict[str, Any]:
        name = str(path.name)
        if name.startswith("eval_results_seed"):
            seed = int(name.split("seed", 1)[1].split(".", 1)[0])
            return eval_by_seed[seed]
        if name.startswith("training_results_seed"):
            seed = int(name.split("seed", 1)[1].split(".", 1)[0])
            return train_by_seed[seed]
        raise AssertionError(f"unexpected path requested: {path}")

    monkeypatch.setattr(train_mod, "_load_json_dict", _fake_load_json_dict)
    monkeypatch.setattr(train_mod.Path, "exists", lambda self: True)
    monkeypatch.setattr(train_mod.torch, "load", lambda *args, **kwargs: {"model_state_dict": {}})
    monkeypatch.setattr(train_mod, "create_helix_full", lambda *args, **kwargs: _DummyModel())
    monkeypatch.setattr(train_mod.np, "load", lambda *args, **kwargs: np.zeros((4, 17), dtype=np.float32))
    monkeypatch.setattr(train_mod, "MultiTaskNumpyDataset", lambda x, y: cast(Any, object()))
    monkeypatch.setattr(train_mod, "DataLoader", lambda *args, **kwargs: cast(Any, object()))

    seed_metrics = {
        42: {"macro_f1": 0.81, "class4_precision": 0.30, "class4_recall": 0.85, "mean_entropy": 0.31, "zero_prediction_classes": 0},
        1337: {"macro_f1": 0.82, "class4_precision": 0.29, "class4_recall": 0.84, "mean_entropy": 0.33, "zero_prediction_classes": 0},
        2026: {"macro_f1": 0.80, "class4_precision": 0.31, "class4_recall": 0.83, "mean_entropy": 0.32, "zero_prediction_classes": 0},
    }

    seed_order = iter([42, 1337, 2026])

    def _fake_calibrate(**kwargs: Any) -> dict[str, Any]:
        seed = next(seed_order)
        vals = seed_metrics[seed]
        return {
            "temperature": 2.2,
            "tau_4": 0.6,
            "test": vals,
        }

    def _fake_emit(**kwargs: Any) -> dict[str, str]:
        seed = int(kwargs["seed"])
        d = tmp_path / f"seed_{seed}"
        d.mkdir(parents=True, exist_ok=True)
        out = {
            "before_after_csv": d / "orig_before_after.csv",
            "before_after_json": d / "orig_before_after.json",
            "pr_curve_csv": d / "orig_pr_curve.csv",
            "confusion_matrices_json": d / "orig_confusion.json",
            "ablation_json": d / "orig_ablation.json",
        }
        for p in out.values():
            p.write_text("{}", encoding="utf-8")
        return {k: str(v) for k, v in out.items()}

    monkeypatch.setattr(train_mod, "_calibrate_family_predictions", _fake_calibrate)
    monkeypatch.setattr(train_mod, "_emit_calibration_artifacts", _fake_emit)

    payload = train_mod._run_multiseed_calibrated_governance(
        script_path=tmp_path / "dummy.py",
        argv=[],
        seeds=[42, 1337, 2026],
        max_temperature=5.0,
        class4_recall_floor=0.8,
    )

    assert set(payload.keys()) == {
        "macro_f1",
        "class4_precision",
        "class4_recall",
        "entropy",
        "zero_prediction_classes",
        "calibration",
        "governance",
    }
    assert "aggregate" not in payload
    assert "runs" not in payload

    calibration = cast(dict[str, Any], payload["calibration"])
    assert set(calibration.keys()) == {
        "temperature",
        "tau_4",
        "pr_curve_path",
        "confusion_matrix_path",
        "ablation_path",
    }
    for key in ("pr_curve_path", "confusion_matrix_path", "ablation_path"):
        assert Path(str(calibration[key])).exists()

    governance = cast(dict[str, Any], payload["governance"])
    assert set(governance.keys()) == {
        "mean_macro_f1",
        "std_macro_f1",
        "mean_class4_precision",
        "mean_class4_recall",
        "min_class4_recall",
        "mean_entropy",
        "max_zero_prediction_classes",
        "status",
        "failure_reasons",
        "actions",
    }
    assert governance["status"] == "PASS"
    assert governance["failure_reasons"] == []
    assert governance["actions"] == []
