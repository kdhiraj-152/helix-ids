"""Maximum Mean Discrepancy (MMD) loss for domain adaptation.

MMD measures the distance between two distributions by comparing their
kernel mean embeddings. Minimizing MMD aligns source and target distributions.

Reference:
    Gretton et al., "A Kernel Two-Sample Test", JMLR 2012
"""

from __future__ import annotations

import torch
import torch.nn as nn


def gaussian_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: float | None = None,
) -> torch.Tensor:
    """Compute Gaussian RBF kernel matrix.

    K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))

    Args:
        x: Source samples [n, d]
        y: Target samples [m, d]
        sigma: Kernel bandwidth. If None, uses median heuristic.

    Returns:
        Kernel matrix [n, m]
    """
    # Compute pairwise distances
    x_sq = (x**2).sum(dim=1, keepdim=True)  # [n, 1]
    y_sq = (y**2).sum(dim=1, keepdim=True)  # [m, 1]

    dist = x_sq + y_sq.t() - 2 * torch.mm(x, y.t())  # [n, m]
    dist = torch.clamp(dist, min=0)

    # Median heuristic for bandwidth
    if sigma is None:
        median_dist = torch.median(dist[dist > 0])
        sigma = torch.sqrt(median_dist / 2).item()
        sigma = max(sigma, 1e-4)  # Avoid division by zero

    kernel = torch.exp(-dist / (2 * sigma**2))

    return kernel


def multi_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidths: list[float] | None = None,
) -> torch.Tensor:
    """Compute multi-kernel MMD using multiple bandwidths.

    Using multiple bandwidths makes the kernel more robust to different
    scales of the data.

    Args:
        x: Source samples [n, d]
        y: Target samples [m, d]
        bandwidths: List of kernel bandwidths. Defaults to [0.1, 1, 10].

    Returns:
        Combined kernel matrix [n, m]
    """
    if bandwidths is None:
        bandwidths = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    kernels = []
    for sigma in bandwidths:
        kernels.append(gaussian_kernel(x, y, sigma))

    # Average across bandwidths
    return torch.stack(kernels).mean(dim=0)


def compute_mmd(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel: str = "gaussian",
    sigma: float | None = None,
    bandwidths: list[float] | None = None,
) -> torch.Tensor:
    """Compute Maximum Mean Discrepancy between two distributions.

    MMD^2 = E[k(x_s, x_s')] + E[k(x_t, x_t')] - 2 * E[k(x_s, x_t)]

    Where x_s, x_s' are source samples and x_t is target sample.

    Args:
        source: Source domain samples [n_s, d]
        target: Target domain samples [n_t, d]
        kernel: Kernel type ("gaussian" or "multi")
        sigma: Bandwidth for gaussian kernel
        bandwidths: Bandwidths for multi-kernel

    Returns:
        MMD squared value (scalar)
    """
    n_s = source.size(0)
    n_t = target.size(0)

    if kernel == "multi":
        k_ss = multi_kernel(source, source, bandwidths)
        k_tt = multi_kernel(target, target, bandwidths)
        k_st = multi_kernel(source, target, bandwidths)
    else:
        k_ss = gaussian_kernel(source, source, sigma)
        k_tt = gaussian_kernel(target, target, sigma)
        k_st = gaussian_kernel(source, target, sigma)

    # Unbiased estimate of MMD^2
    # Remove diagonal elements for unbiased estimate
    diag_s = torch.diag(k_ss)
    diag_t = torch.diag(k_tt)

    sum_ss = (k_ss.sum() - diag_s.sum()) / (n_s * (n_s - 1))
    sum_tt = (k_tt.sum() - diag_t.sum()) / (n_t * (n_t - 1))
    sum_st = k_st.mean()

    mmd_sq = sum_ss + sum_tt - 2 * sum_st

    return mmd_sq


class MMDLoss(nn.Module):
    """MMD loss module for domain adaptation.

    Minimizing MMD loss encourages the model to produce similar
    feature distributions for source and target domains.
    """

    def __init__(
        self,
        kernel: str = "multi",
        sigma: float | None = None,
        bandwidths: list[float] | None = None,
    ):
        """Initialize MMD loss.

        Args:
            kernel: Kernel type ("gaussian" or "multi")
            sigma: Bandwidth for gaussian kernel
            bandwidths: Bandwidths for multi-kernel
        """
        super().__init__()
        self.kernel = kernel
        self.sigma = sigma
        self.bandwidths = bandwidths or [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MMD loss.

        Args:
            source: Source features [n_s, d]
            target: Target features [n_t, d]

        Returns:
            MMD loss value (scalar)
        """
        return compute_mmd(
            source,
            target,
            kernel=self.kernel,
            sigma=self.sigma,
            bandwidths=self.bandwidths,
        )


class JointMMDLoss(nn.Module):
    """Joint MMD loss across multiple layers.

    Computes MMD at multiple layers of the network and combines them.
    This provides stronger domain adaptation.
    """

    def __init__(
        self,
        kernel: str = "multi",
        bandwidths: list[float] | None = None,
        layer_weights: list[float] | None = None,
    ):
        """Initialize joint MMD loss.

        Args:
            kernel: Kernel type
            bandwidths: Kernel bandwidths
            layer_weights: Weights for each layer's MMD
        """
        super().__init__()
        self.mmd = MMDLoss(kernel=kernel, bandwidths=bandwidths)
        self.layer_weights = layer_weights

    def forward(
        self,
        source_features: list[torch.Tensor],
        target_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute joint MMD across layers.

        Args:
            source_features: List of source feature tensors from each layer
            target_features: List of target feature tensors from each layer

        Returns:
            Combined MMD loss
        """
        if len(source_features) != len(target_features):
            raise ValueError("Source and target must have same number of layers")

        n_layers = len(source_features)

        if self.layer_weights is None:
            weights = [1.0 / n_layers] * n_layers
        else:
            weights = self.layer_weights

        total_loss = 0.0
        for i, (src, tgt) in enumerate(zip(source_features, target_features)):
            total_loss = total_loss + weights[i] * self.mmd(src, tgt)

        return total_loss


class ConditionalMMDLoss(nn.Module):
    """Class-conditional MMD loss.

    Computes MMD separately for each class, which can be more effective
    when class distributions differ significantly between domains.
    """

    def __init__(
        self,
        num_classes: int,
        kernel: str = "multi",
        bandwidths: list[float] | None = None,
    ):
        """Initialize conditional MMD loss.

        Args:
            num_classes: Number of classes
            kernel: Kernel type
            bandwidths: Kernel bandwidths
        """
        super().__init__()
        self.num_classes = num_classes
        self.mmd = MMDLoss(kernel=kernel, bandwidths=bandwidths)

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_labels: torch.Tensor,
        target_pseudo_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute class-conditional MMD.

        Args:
            source_features: Source features [n_s, d]
            target_features: Target features [n_t, d]
            source_labels: Source class labels [n_s]
            target_pseudo_labels: Target pseudo-labels [n_t]

        Returns:
            Conditional MMD loss
        """
        total_loss = 0.0
        n_classes = 0

        for c in range(self.num_classes):
            src_mask = source_labels == c
            tgt_mask = target_pseudo_labels == c

            src_c = source_features[src_mask]
            tgt_c = target_features[tgt_mask]

            # Skip if either class is empty
            if len(src_c) < 2 or len(tgt_c) < 2:
                continue

            total_loss = total_loss + self.mmd(src_c, tgt_c)
            n_classes += 1

        if n_classes > 0:
            total_loss = total_loss / n_classes

        return total_loss
