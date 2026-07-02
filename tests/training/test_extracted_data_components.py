"""
Regression tests for Phase 12B-3 extracted data components.

Validates that extracted Dataset, Sampler, and validator classes/functions
behave identically to their original definitions in train_helix_ids_full.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.training.data import (
    ClassBalancedIndexSampler,
    FrozenIndexSampler,
    MultiTaskNumpyDataset,
    _apply_label_merges,
    _assert_feature_dimensions,
    _build_frozen_class_balanced_indices,
    _build_frozen_tempered_indices,
    _build_stratified_subset_indices,
    _build_stratified_val_subset,
    _chunk_finite_check,
    _coerce_finite_float,
    _compute_class4_metrics,
    _compute_multiclass_confusion,
    _default_tail_multiplier,
    _detect_cluster_mode_collapse,
    _inverse_frequency_weights,
    _normalize_engineered_feature_block,
    _normalize_metrics_payload,
    _normalized_entropy_from_counts,
    _sample_rows,
    _sqrt_inverse_frequency_weights,
    _summarize_prediction_coverage,
    build_class_index,
)

# ============================================================================
# MultiTaskNumpyDataset
# ============================================================================


class TestMultiTaskNumpyDataset:
    def test_len_and_getitem(self):
        n = 100
        features = np.random.randn(n, 32).astype(np.float32)
        labels = np.random.randint(0, 7, size=n).astype(np.int64)

        ds = MultiTaskNumpyDataset(features, labels)
        assert len(ds) == n
        x, y_bin, y_fam = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y_bin, torch.Tensor)
        assert isinstance(y_fam, torch.Tensor)
        assert y_bin.item() == (1 if y_fam.item() != 0 else 0)

    def test_length_mismatch_raises(self):
        features = np.random.randn(100, 32)
        labels = np.random.randint(0, 7, size=50)
        with pytest.raises(ValueError, match="length mismatch"):
            MultiTaskNumpyDataset(features, labels)

    def test_indexing_accuracy(self):
        features = np.array([[0.0], [1.0]], dtype=np.float32)
        labels = np.array([0, 3], dtype=np.int64)
        ds = MultiTaskNumpyDataset(features, labels)
        x0, yb0, yf0 = ds[0]
        assert yf0.item() == 0
        assert yb0.item() == 0
        x1, yb1, yf1 = ds[1]
        assert yf1.item() == 3
        assert yb1.item() == 1


# ============================================================================
# ClassBalancedIndexSampler
# ============================================================================


class TestClassBalancedIndexSampler:
    def test_basic_invariants(self):
        n = 500
        y = np.random.randint(0, 5, size=n)
        sampler = ClassBalancedIndexSampler(y, batch_size=32, seed=42)
        indices = list(sampler)
        assert len(indices) == sampler.steps_per_epoch * 32

    def test_every_class_appears_per_batch(self):
        y = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4])
        sampler = ClassBalancedIndexSampler(y, batch_size=10, min_per_class=1, seed=42)
        indices = list(sampler)
        for batch_start in range(0, len(indices), 10):
            batch = indices[batch_start : batch_start + 10]
            batch_classes = set(y[batch].tolist())
            for c in [0, 1, 2, 3, 4]:
                assert c in batch_classes, f"Class {c} missing in batch"

    def test_epoch_increment_reproduces(self):
        y = np.random.randint(0, 5, size=200)
        s1 = ClassBalancedIndexSampler(y, batch_size=16, seed=42)
        s2 = ClassBalancedIndexSampler(y, batch_size=16, seed=42)
        i1 = list(s1)
        i2 = list(s2)
        assert i1 == i2, "Same seed should give same first epoch"


# ============================================================================
# FrozenIndexSampler
# ============================================================================


class TestFrozenIndexSampler:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one index"):
            FrozenIndexSampler(np.array([], dtype=np.int64))

    def test_basic_iteration(self):
        indices = np.array([3, 1, 4, 1, 5, 9], dtype=np.int64)
        sampler = FrozenIndexSampler(indices)
        assert list(sampler) == [3, 1, 4, 1, 5, 9]
        assert len(sampler) == 6

    def test_deterministic(self):
        arr = np.array([0, 2, 4, 6, 8], dtype=np.int64)
        s1 = FrozenIndexSampler(arr)
        s2 = FrozenIndexSampler(arr)
        assert list(s1) == list(s2)


# ============================================================================
# build_class_index
# ============================================================================


class TestBuildClassIndex:
    def test_basic(self):
        y = np.array([0, 0, 1, 0, 2, 1, 2, 2, 3])
        idx = build_class_index(y)
        assert sorted(idx.keys()) == [0, 1, 2, 3]
        assert len(idx[0]) == 3
        assert len(idx[1]) == 2
        assert len(idx[2]) == 3
        assert len(idx[3]) == 1

    def test_all_same_class(self):
        y = np.array([5, 5, 5])
        idx = build_class_index(y)
        assert list(idx.keys()) == [5]


# ============================================================================
# _chunk_finite_check / _sample_rows
# ============================================================================


class TestChunkFiniteCheck:
    def test_finite_passes(self):
        x = np.random.randn(1000, 10)
        assert _chunk_finite_check(x) is True

    def test_nan_detected(self):
        x = np.random.randn(100, 10)
        x[50, 5] = np.nan
        assert _chunk_finite_check(x) is False

    def test_inf_detected(self):
        x = np.random.randn(100, 10)
        x[30, 3] = np.inf
        assert _chunk_finite_check(x) is False


class TestSampleRows:
    def test_small_returns_all(self):
        x = np.random.randn(100, 10)
        sampled = _sample_rows(x, seed=42)
        assert sampled.shape[0] == 100

    def test_large_samples_correctly(self):
        x = np.random.randn(100000, 10)
        sampled = _sample_rows(x, seed=42, max_rows=5000)
        assert sampled.shape[0] == 5000

    def test_deterministic(self):
        x = np.random.randn(100000, 10)
        s1 = _sample_rows(x, seed=42, max_rows=5000)
        s2 = _sample_rows(x, seed=42, max_rows=5000)
        np.testing.assert_array_equal(s1, s2)


# ============================================================================
# Stratified subset builders
# ============================================================================


class TestStratifiedSubset:
    def test_build_stratified_subset_indices(self):
        y = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 3, 4, 4])
        idx = _build_stratified_subset_indices(y, target_per_class=2, seed=42)
        unique, counts = np.unique(y[idx], return_counts=True)
        assert sorted(unique.tolist()) == [0, 1, 2, 3, 4]
        assert all(c >= 2 for c in counts)

    def test_build_stratified_val_subset(self):
        x = np.random.randn(100, 5)
        y = np.random.randint(0, 4, size=100)
        xs, ys = _build_stratified_val_subset(x, y, target_per_class=3, seed=42)
        assert xs.shape[0] == ys.shape[0]
        assert xs.shape[0] >= 12  # 4 classes * 3
        assert xs.shape[1] == 5


# ============================================================================
# Weight helpers
# ============================================================================


class TestWeightHelpers:
    def test_inverse_frequency_weights(self):
        y = np.array([0, 0, 0, 1, 1, 2])
        w = _inverse_frequency_weights(y, minlength=3)
        assert w.shape == (3,)
        assert w[0] < w[1] < w[2]  # less frequent = higher weight

    def test_sqrt_inverse_frequency_weights(self):
        y = np.array([0, 0, 0, 1, 1, 2])
        w = _sqrt_inverse_frequency_weights(y, minlength=3)
        assert w.shape == (3,)
        assert w[2] > w[0]  # tail classes get higher weight

    def test_default_tail_multiplier(self):
        assert _default_tail_multiplier(32) == 80.0
        assert _default_tail_multiplier(500) == 15.0
        assert _default_tail_multiplier(2000) == 1.0


# ============================================================================
# Label merges
# ============================================================================


class TestApplyLabelMerges:
    def test_no_merges(self):
        y = np.array([0, 1, 2, 3])
        result = _apply_label_merges(y, merges=[])
        np.testing.assert_array_equal(result, y)

    def test_simple_merge(self):
        y = np.array([1, 1, 2, 3, 2])
        result = _apply_label_merges(y, merges=[(2, 1)])
        np.testing.assert_array_equal(result, [1, 1, 1, 3, 1])

    def test_chain_merge(self):
        y = np.array([3, 2, 1])
        result = _apply_label_merges(y, merges=[(3, 2), (2, 1)])
        np.testing.assert_array_equal(result, [1, 1, 1])


# ============================================================================
# Assertion helpers
# ============================================================================


class TestFeatureAssertions:
    def test_assert_feature_dimensions_passes(self):
        x_train = np.random.randn(100, 10)
        x_val = np.random.randn(20, 10)
        _assert_feature_dimensions(
            dataset_name="test",
            x_train=x_train,
            x_val=x_val,
            feature_names=[f"f{i}" for i in range(10)],
            expected_feature_dim=10,
        )

    def test_assert_feature_dimensions_fails(self):
        x_train = np.random.randn(100, 10)
        x_val = np.random.randn(20, 9)
        with pytest.raises(RuntimeError, match="feature_dim_not_expected"):
            _assert_feature_dimensions(
                dataset_name="test",
                x_train=x_train,
                x_val=x_val,
                feature_names=[f"f{i}" for i in range(10)],
                expected_feature_dim=10,
            )


# ============================================================================
# Entropy / cluster diagnostics
# ============================================================================


class TestClusterDiagnostics:
    def test_normalized_entropy_homogeneous(self):
        e = _normalized_entropy_from_counts([100, 0, 0])
        assert e < 1e-6, f"Expected near-zero entropy, got {e}"

    def test_detect_cluster_mode_collapse_no_collapse(self):
        collapsed, _ = _detect_cluster_mode_collapse([30, 30, 30, 10])
        assert collapsed is False

    def test_detect_cluster_mode_collapse_dominant(self):
        collapsed, info = _detect_cluster_mode_collapse([90, 5, 5])
        assert collapsed is True
        assert info["dominant_cluster_fraction"] >= 0.85


# ============================================================================
# Classification metrics
# ============================================================================


class TestClassificationMetrics:
    def test_compute_multiclass_confusion(self):
        yt = np.array([0, 1, 2, 0, 1])
        yp = np.array([0, 1, 1, 0, 2])
        cm = _compute_multiclass_confusion(yt, yp, class_count=3)
        assert cm.shape == (3, 3)
        assert cm[0, 0] == 2
        assert cm[1, 1] == 1

    def test_compute_class4_metrics(self):
        yt = np.array([4, 0, 4, 1, 4, 2, 4, 4])
        yp = np.array([4, 0, 4, 1, 0, 2, 4, 3])
        m = _compute_class4_metrics(yt, yp, class4_id=4)
        assert 0 < m["class4_precision"] <= 1.0
        assert 0 < m["class4_recall"] <= 1.0

    def test_summarize_prediction_coverage(self):
        yt = np.array([0, 1, 2, 3, 4, 5])
        yp = np.array([0, 0, 0, 0, 0, 0])
        uncovered = _summarize_prediction_coverage(yt, yp)
        assert uncovered == 5  # classes 1-5 never predicted


# ============================================================================
# Metrics normalization
# ============================================================================


class TestMetricsNormalization:
    def test_coerce_finite_float_passes(self):
        assert _coerce_finite_float(3.14, field="test") == 3.14

    def test_coerce_finite_float_nan_raises(self):
        with pytest.raises(ValueError, match="must be finite"):
            _coerce_finite_float(float("nan"), field="test")

    def test_normalize_metrics_payload(self):
        metrics = {
            "macro_f1": 0.85,
            "family_class4_precision": 0.72,
            "family_minority_recall_min": 0.65,
            "family_entropy": 0.42,
            "family_zero_prediction_classes": 1,
        }
        normalized = _normalize_metrics_payload(metrics)
        assert normalized["macro_f1"] == 0.85
        assert normalized["class4_precision"] == 0.72
        assert normalized["class4_recall"] == 0.65
        assert normalized["entropy"] == 0.42


# ============================================================================
# Normalize engineered features
# ============================================================================


class TestNormalizeEngineeredFeatures:
    def test_no_engineered_features(self):
        x_train = np.random.randn(100, 5)
        x_val = np.random.randn(20, 5)
        x_test = np.random.randn(30, 5)
        xt, xv, xtest, stats = _normalize_engineered_feature_block(
            dataset_name="test",
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            feature_names=["a", "b", "c", "d", "e"],
            engineered_feature_names=set(),
        )
        assert stats == {}

    def test_engineered_features_normalized(self):
        x_train = np.random.randn(100, 5)
        x_val = np.random.randn(20, 5)
        x_test = np.random.randn(30, 5)
        xt, xv, xtest, stats = _normalize_engineered_feature_block(
            dataset_name="test",
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            feature_names=["log_src_bytes", "b", "c", "d", "e"],
            engineered_feature_names={"log_src_bytes"},
        )
        assert "log_src_bytes" in stats
        assert abs(stats["log_src_bytes"]["train_mean"]) < 1e-6  # z-scored
        assert abs(stats["log_src_bytes"]["train_std"] - 1.0) < 0.15


# ============================================================================
# Frozen index builders
# ============================================================================


class TestFrozenIndexBuilders:
    def test_build_frozen_class_balanced_indices(self):
        y = np.random.randint(0, 5, size=500)
        indices = _build_frozen_class_balanced_indices(y, batch_size=32, seed=42)
        assert indices.dtype == np.int64
        assert indices.shape[0] > 0
        # verify at least one of each class in first batch
        first_batch_classes = set(y[indices[:32]].tolist())
        assert first_batch_classes == set(range(5))

    def test_build_frozen_tempered_indices(self):
        y = np.random.randint(0, 5, size=500)
        indices, probs = _build_frozen_tempered_indices(y, batch_size=32, seed=42)
        assert indices.dtype == np.int64
        assert len(probs) == 5
        assert abs(sum(probs.values()) - 1.0) < 1e-6
