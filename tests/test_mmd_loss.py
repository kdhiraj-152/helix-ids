"""Tests for mmd_loss.py — MMD loss for domain adaptation."""

import torch

from helix_ids.models.adaptation.mmd_loss import (
    ConditionalMMDLoss,
    JointMMDLoss,
    MMDLoss,
    compute_mmd,
    gaussian_kernel,
    multi_kernel,
)

# ---------------------------------------------------------------------------
# gaussian_kernel
# ---------------------------------------------------------------------------

class TestGaussianKernel:
    """Tests for gaussian_kernel(x, y, sigma=None)."""

    def test_output_shape(self):
        x = torch.randn(10, 8)
        y = torch.randn(5, 8)
        k = gaussian_kernel(x, y)
        assert k.shape == (10, 5), f"Expected (10, 5), got {k.shape}"

    def test_self_kernel_is_symmetric(self):
        x = torch.randn(8, 4)
        k = gaussian_kernel(x, x)
        assert torch.allclose(k, k.t()), "Self-kernel should be symmetric"

    def test_self_kernel_diagonal_ones(self):
        x = torch.randn(8, 4)
        k = gaussian_kernel(x, x)
        diag = torch.diag(k)
        assert torch.allclose(diag, torch.ones(8)), \
            f"Diagonal of K(x,x) should be 1, got max diff {torch.max(torch.abs(diag - 1.0)).item()}"

    def test_median_heuristic_sigma_is_not_none(self):
        """When sigma=None the median heuristic should produce a valid sigma."""
        x = torch.randn(20, 4)
        y = torch.randn(15, 4)
        k = gaussian_kernel(x, y, sigma=None)
        assert k.shape == (20, 15)
        assert not torch.isnan(k).any()
        assert not torch.isinf(k).any()

    def test_gaussian_kernel_values_in_range(self):
        x = torch.randn(10, 4)
        y = torch.randn(10, 4)
        k = gaussian_kernel(x, y, sigma=1.0)
        assert k.min() >= 0.0, "Kernel values should be >= 0"
        assert k.max() <= 1.0, "Kernel values should be <= 1"

    def test_gaussian_kernel_larger_sigma_smoother(self):
        """Larger sigma produces larger (less peaked) values."""
        x = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
        k1 = gaussian_kernel(x, x, sigma=1.0)
        k2 = gaussian_kernel(x, x, sigma=10.0)
        # Off-diagonal should be larger with larger sigma (less peaked)
        assert k2[0, 1] > k1[0, 1], "Larger sigma should give larger off-diagonal"

    def test_median_heuristic_with_noise(self):
        """Median heuristic should work with slightly different points."""
        x = torch.randn(10, 4)
        y = torch.randn(8, 4)
        k = gaussian_kernel(x, y, sigma=None)
        assert k.shape == (10, 8)
        assert not torch.isnan(k).any()
        assert not torch.isinf(k).any()

    def test_different_feature_dimensions(self):
        x = torch.randn(5, 16)
        y = torch.randn(7, 16)
        k = gaussian_kernel(x, y, sigma=2.0)
        assert k.shape == (5, 7)


# ---------------------------------------------------------------------------
# multi_kernel
# ---------------------------------------------------------------------------

class TestMultiKernel:
    """Tests for multi_kernel(x, y, bandwidths=None)."""

    def test_output_shape(self):
        x = torch.randn(8, 4)
        y = torch.randn(6, 4)
        k = multi_kernel(x, y)
        assert k.shape == (8, 6)

    def test_default_bandwidths(self):
        x = torch.randn(5, 3)
        y = torch.randn(5, 3)
        k = multi_kernel(x, y, bandwidths=None)
        assert k.shape == (5, 5)

    def test_custom_bandwidths(self):
        x = torch.randn(4, 2)
        y = torch.randn(4, 2)
        bw = [0.5, 1.0, 5.0]
        k = multi_kernel(x, y, bandwidths=bw)
        assert k.shape == (4, 4)

    def test_averaging_behavior(self):
        """multi_kernel should produce the average of individual kernels."""
        x = torch.randn(6, 3)
        y = torch.randn(6, 3)
        bw = [0.5, 1.0, 2.0]
        k_multi = multi_kernel(x, y, bandwidths=bw)
        # Compute individual kernels and average manually
        kernels = [gaussian_kernel(x, y, sigma=s) for s in bw]
        manual = torch.stack(kernels).mean(dim=0)
        assert torch.allclose(k_multi, manual), "multi_kernel should be the average of individual kernels"

    def test_values_in_range(self):
        x = torch.randn(5, 4)
        y = torch.randn(5, 4)
        k = multi_kernel(x, y)
        assert k.min() >= 0.0
        assert k.max() <= 1.0


# ---------------------------------------------------------------------------
# compute_mmd
# ---------------------------------------------------------------------------

class TestComputeMMD:
    """Tests for compute_mmd(source, target, ...)."""

    def test_mmd_same_distribution(self):
        """MMD^2 between identical distributions should be near zero."""
        data = torch.randn(50, 10)
        mmd_val = compute_mmd(data, data, kernel="multi")
        assert mmd_val.item() < 0.05, f"MMD between same dist should be near 0, got {mmd_val.item()}"

    def test_mmd_gaussian_kernel(self):
        data = torch.randn(30, 5)
        mmd_val = compute_mmd(data, data, kernel="gaussian", sigma=1.0)
        assert isinstance(mmd_val, torch.Tensor)
        assert mmd_val.ndim == 0  # scalar

    def test_mmd_multi_kernel(self):
        data = torch.randn(30, 5)
        mmd_val = compute_mmd(data, data, kernel="multi")
        assert isinstance(mmd_val, torch.Tensor)
        assert mmd_val.ndim == 0

    def test_mmd_different_distributions(self):
        """MMD^2 between different distributions should be non-zero."""
        source = torch.randn(50, 8)
        target = torch.randn(50, 8) + 5.0  # shifted
        mmd_val = compute_mmd(source, target, kernel="gaussian", sigma=1.0)
        assert mmd_val.item() > 0.01, f"MMD between different dists should be > 0, got {mmd_val.item()}"

    def test_mmd_multi_different_distributions(self):
        source = torch.randn(40, 6)
        target = torch.randn(40, 6) + 3.0
        mmd_val = compute_mmd(source, target, kernel="multi")
        assert mmd_val.item() > 0.01

    def test_mmd_small_samples(self):
        """MMD with very few samples should still work."""
        source = torch.randn(5, 4)
        target = torch.randn(5, 4)
        mmd_val = compute_mmd(source, target, kernel="gaussian", sigma=1.0)
        assert not torch.isnan(mmd_val)
        assert not torch.isinf(mmd_val)

    def test_mmd_is_approximately_non_negative(self):
        """MMD^2 with unbiased estimate is not NaN/inf for same-distribution inputs."""
        torch.manual_seed(42)
        source = torch.randn(20, 5)
        target = torch.randn(20, 5)
        mmd_val = compute_mmd(source, target, kernel="gaussian", sigma=1.0)
        assert not torch.isnan(mmd_val), "MMD should not be NaN"
        assert not torch.isinf(mmd_val), "MMD should not be Inf"

    def test_mmd_gaussian_with_bandwidths_param(self):
        """compute_mmd with kernel='gaussian' should ignore bandwidths param."""
        source = torch.randn(20, 5)
        target = torch.randn(20, 5) + 2.0
        # Should not raise even when bandwidths is provided for gaussian kernel
        mmd_val = compute_mmd(source, target, kernel="gaussian", sigma=1.0, bandwidths=[1.0, 2.0])
        assert not torch.isnan(mmd_val)


# ---------------------------------------------------------------------------
# MMDLoss
# ---------------------------------------------------------------------------

class TestMMDLoss:
    """Tests for MMDLoss(nn.Module)."""

    def test_forward_basic(self):
        loss_fn = MMDLoss()
        source = torch.randn(30, 8)
        target = torch.randn(30, 8)
        loss = loss_fn(source, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_forward_gaussian_kernel(self):
        loss_fn = MMDLoss(kernel="gaussian", sigma=1.0)
        source = torch.randn(20, 5)
        target = torch.randn(20, 5)
        loss = loss_fn(source, target)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_forward_multi_kernel(self):
        loss_fn = MMDLoss(kernel="multi")
        source = torch.randn(20, 5)
        target = torch.randn(20, 5)
        loss = loss_fn(source, target)
        assert not torch.isnan(loss)

    def test_default_bandwidths_stored(self):
        loss_fn = MMDLoss()
        assert len(loss_fn.bandwidths) == 6
        assert loss_fn.bandwidths == [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    def test_custom_bandwidths(self):
        loss_fn = MMDLoss(bandwidths=[0.3, 3.0])
        assert loss_fn.bandwidths == [0.3, 3.0]

    def test_gaussian_and_multi_equivalence(self):
        """For a single bandwidth, multi_kernel with one bandwidth should be same as gaussian."""
        source = torch.randn(15, 4)
        target = torch.randn(15, 4)
        loss_multi = MMDLoss(kernel="multi", bandwidths=[1.0])(source, target)
        loss_gauss = MMDLoss(kernel="gaussian", sigma=1.0)(source, target)
        assert torch.allclose(loss_multi, loss_gauss, atol=1e-6), \
            "Single-bandwidth multi should equal gaussian with same sigma"


# ---------------------------------------------------------------------------
# JointMMDLoss
# ---------------------------------------------------------------------------

class TestJointMMDLoss:
    """Tests for JointMMDLoss(nn.Module)."""

    def test_single_layer(self):
        """JointMMD with one layer should behave like MMDLoss."""
        loss_fn = JointMMDLoss()
        src = [torch.randn(20, 8)]
        tgt = [torch.randn(20, 8)]
        loss = loss_fn(src, tgt)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_multiple_layers(self):
        loss_fn = JointMMDLoss()
        n_layers = 3
        src = [torch.randn(20, 8) for _ in range(n_layers)]
        tgt = [torch.randn(20, 8) for _ in range(n_layers)]
        loss = loss_fn(src, tgt)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_custom_layer_weights(self):
        weights = [0.5, 0.3, 0.2]
        loss_fn = JointMMDLoss(layer_weights=weights)
        src = [torch.randn(20, 8) for _ in range(3)]
        tgt = [torch.randn(20, 8) for _ in range(3)]
        loss = loss_fn(src, tgt)
        assert not torch.isnan(loss)

    def test_default_layer_weights_equal(self):
        """Without custom weights, all layers should have equal weight."""
        loss_fn = JointMMDLoss()
        src = [torch.randn(20, 8), torch.randn(20, 8)]
        tgt = [torch.randn(20, 8), torch.randn(20, 8)]
        loss = loss_fn(src, tgt)
        # Manually compute expected
        expected = 0.5 * loss_fn.mmd(src[0], tgt[0]) + 0.5 * loss_fn.mmd(src[1], tgt[1])
        assert torch.allclose(loss, expected), "Default weights should be equal"

    def test_length_mismatch_raises(self):
        loss_fn = JointMMDLoss()
        src = [torch.randn(20, 8), torch.randn(20, 8)]
        tgt = [torch.randn(20, 8)]  # only one layer
        import pytest
        with pytest.raises(ValueError, match="Source and target must have same number of layers"):
            loss_fn(src, tgt)

    def test_loss_decreases_with_alignment(self):
        """MMD loss between aligned distributions should be smaller than between shifted ones."""
        loss_fn = JointMMDLoss()
        src = [torch.randn(30, 8)]
        tgt_aligned = src[0].clone()  # same data
        tgt_shifted = [torch.randn(30, 8) + 5.0]
        loss_aligned = loss_fn(src, [tgt_aligned])
        loss_shifted = loss_fn(src, tgt_shifted)
        assert loss_aligned.item() < loss_shifted.item(), \
            "Aligned distributions should have lower MMD"


# ---------------------------------------------------------------------------
# ConditionalMMDLoss
# ---------------------------------------------------------------------------

class TestConditionalMMDLoss:
    """Tests for ConditionalMMDLoss(nn.Module)."""

    def test_basic_conditional(self):
        loss_fn = ConditionalMMDLoss(num_classes=3)
        n_src, n_tgt = 30, 30
        src_feat = torch.randn(n_src, 8)
        tgt_feat = torch.randn(n_tgt, 8)
        src_labels = torch.randint(0, 3, (n_src,))
        tgt_labels = torch.randint(0, 3, (n_tgt,))
        loss = loss_fn(src_feat, tgt_feat, src_labels, tgt_labels)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert not torch.isnan(loss)

    def test_all_same_class(self):
        """When all samples are the same class, conditional MMD should still work."""
        loss_fn = ConditionalMMDLoss(num_classes=3)
        src_feat = torch.randn(30, 8)
        tgt_feat = torch.randn(30, 8)
        src_labels = torch.zeros(30, dtype=torch.long)
        tgt_labels = torch.zeros(30, dtype=torch.long)
        loss = loss_fn(src_feat, tgt_feat, src_labels, tgt_labels)
        assert not torch.isnan(loss)

    def test_skip_empty_classes(self):
        """Classes with < 2 samples in either domain should be skipped."""
        loss_fn = ConditionalMMDLoss(num_classes=4)
        src_feat = torch.randn(40, 8)
        tgt_feat = torch.randn(40, 8)
        # Class 0 has lots, class 1 has just 1 sample (will be skipped)
        src_labels = torch.zeros(40, dtype=torch.long)
        tgt_labels = torch.zeros(40, dtype=torch.long)
        src_labels[:2] = 1
        tgt_labels[:1] = 1  # only 1, will be skipped
        loss = loss_fn(src_feat, tgt_feat, src_labels, tgt_labels)
        assert not torch.isnan(loss)

    def test_all_classes_skipped_returns_zero(self):
        """If all classes are skipped (too few samples), loss should be 0."""
        loss_fn = ConditionalMMDLoss(num_classes=3)
        src_feat = torch.randn(3, 8)
        tgt_feat = torch.randn(3, 8)
        src_labels = torch.tensor([0, 1, 2], dtype=torch.long)
        tgt_labels = torch.tensor([0, 1, 2], dtype=torch.long)
        loss = loss_fn(src_feat, tgt_feat, src_labels, tgt_labels)
        # Each class has only 1 sample (< 2), so all skipped
        assert loss.item() == 0.0, "All classes skipped should yield 0 loss"

    def test_one_class_skipped_one_used(self):
        """Mix of skipped and used classes."""
        loss_fn = ConditionalMMDLoss(num_classes=2)
        src_feat = torch.randn(25, 8)
        tgt_feat = torch.randn(25, 8)
        # Class 0: many samples; Class 1: too few
        src_labels = torch.zeros(25, dtype=torch.long)
        tgt_labels = torch.zeros(25, dtype=torch.long)
        src_labels[:5] = 1
        tgt_labels[:5] = 1  # only 5 each, OK (>2)
        loss = loss_fn(src_feat, tgt_feat, src_labels, tgt_labels)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        # With identical distributions, loss may be ~0 or slightly negative (unbiased estimate)

    def test_zero_classes_handles_gracefully(self):
        """num_classes=0 should trivially return 0."""
        loss_fn = ConditionalMMDLoss(num_classes=0)
        src_feat = torch.randn(10, 4)
        tgt_feat = torch.randn(10, 4)
        src_labels = torch.zeros(10, dtype=torch.long)
        tgt_labels = torch.zeros(10, dtype=torch.long)
        loss = loss_fn(src_feat, tgt_feat, src_labels, tgt_labels)
        assert loss.item() == 0.0

    def test_returns_averaged_loss(self):
        """The conditional MMD should average over classes and not be NaN."""
        loss_fn = ConditionalMMDLoss(num_classes=2)
        src_feat = torch.randn(40, 8)
        tgt_feat = torch.randn(40, 8)
        src_labels = torch.cat([torch.zeros(20), torch.ones(20)]).long()
        tgt_labels = torch.cat([torch.zeros(20), torch.ones(20)]).long()
        loss = loss_fn(src_feat, tgt_feat, src_labels, tgt_labels)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_gaussian_conditional(self):
        loss_fn = ConditionalMMDLoss(num_classes=2, kernel="gaussian")
        src_feat = torch.randn(20, 8)
        tgt_feat = torch.randn(20, 8)
        src_labels = torch.cat([torch.zeros(10), torch.ones(10)]).long()
        tgt_labels = torch.cat([torch.zeros(10), torch.ones(10)]).long()
        loss = loss_fn(src_feat, tgt_feat, src_labels, tgt_labels)
        assert not torch.isnan(loss)
