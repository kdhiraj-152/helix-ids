"""CORAL (Correlation Alignment) loss for domain adaptation.

CORAL aligns the second-order statistics (covariance) of source and
target domain features, which can effectively reduce domain shift.

Reference:
    Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep Domain Adaptation",
    ECCV 2016 Workshops
"""

from __future__ import annotations

import torch
import torch.nn as nn


def compute_covariance(x: torch.Tensor) -> torch.Tensor:
    """Compute covariance matrix of features.

    Args:
        x: Feature tensor [n, d]

    Returns:
        Covariance matrix [d, d]
    """
    n = x.size(0)

    if n < 2:
        return torch.zeros(x.size(1), x.size(1), device=x.device)

    # Center the features
    x_centered = x - x.mean(dim=0, keepdim=True)

    # Compute covariance
    cov = torch.mm(x_centered.t(), x_centered) / (n - 1)

    return cov


def compute_coral(
    source: torch.Tensor,
    target: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute CORAL loss between source and target features.

    CORAL = ||C_s - C_t||_F^2 / (4 * d^2)

    Where C_s, C_t are covariance matrices and d is feature dimension.

    Args:
        source: Source domain features [n_s, d]
        target: Target domain features [n_t, d]
        normalize: Whether to normalize by dimension

    Returns:
        CORAL loss value (scalar)
    """
    d = source.size(1)

    # Compute covariance matrices
    cov_source = compute_covariance(source)
    cov_target = compute_covariance(target)

    # Frobenius norm of difference
    diff = cov_source - cov_target
    coral = torch.sum(diff * diff)

    if normalize:
        coral = coral / (4 * d * d)

    return coral


class CORALLoss(nn.Module):
    """CORAL loss module for domain adaptation.

    Minimizing CORAL loss aligns the covariance matrices of source
    and target domain features.
    """

    def __init__(self, normalize: bool = True):
        """Initialize CORAL loss.

        Args:
            normalize: Whether to normalize by feature dimension
        """
        super().__init__()
        self.normalize = normalize

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CORAL loss.

        Args:
            source: Source features [n_s, d]
            target: Target features [n_t, d]

        Returns:
            CORAL loss value
        """
        return compute_coral(source, target, self.normalize)


class DeepCORALLoss(nn.Module):
    """Deep CORAL loss with layer-wise alignment.

    Applies CORAL alignment at multiple layers of the network
    for stronger domain adaptation.
    """

    def __init__(
        self,
        normalize: bool = True,
        layer_weights: list[float] | None = None,
    ):
        """Initialize Deep CORAL loss.

        Args:
            normalize: Whether to normalize each layer's loss
            layer_weights: Weights for each layer's contribution
        """
        super().__init__()
        self.coral = CORALLoss(normalize=normalize)
        self.layer_weights = layer_weights

    def forward(
        self,
        source_features: list[torch.Tensor],
        target_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute Deep CORAL loss.

        Args:
            source_features: List of source feature tensors from each layer
            target_features: List of target feature tensors from each layer

        Returns:
            Combined CORAL loss
        """
        n_layers = len(source_features)

        if self.layer_weights is None:
            weights = [1.0 / n_layers] * n_layers
        else:
            weights = self.layer_weights

        total_loss = 0.0
        for i, (src, tgt) in enumerate(zip(source_features, target_features)):
            total_loss = total_loss + weights[i] * self.coral(src, tgt)

        return total_loss


class CombinedAlignmentLoss(nn.Module):
    """Combined MMD and CORAL alignment loss.

    Uses both first-order (MMD) and second-order (CORAL) statistics
    alignment for stronger domain adaptation.
    """

    def __init__(
        self,
        mmd_weight: float = 1.0,
        coral_weight: float = 1.0,
        kernel: str = "multi",
    ):
        """Initialize combined loss.

        Args:
            mmd_weight: Weight for MMD loss
            coral_weight: Weight for CORAL loss
            kernel: MMD kernel type
        """
        super().__init__()

        from .mmd_loss import MMDLoss

        self.mmd = MMDLoss(kernel=kernel)
        self.coral = CORALLoss(normalize=True)
        self.mmd_weight = mmd_weight
        self.coral_weight = coral_weight

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined alignment loss.

        Args:
            source: Source features [n_s, d]
            target: Target features [n_t, d]

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        mmd_loss = self.mmd(source, target)
        coral_loss = self.coral(source, target)

        total = self.mmd_weight * mmd_loss + self.coral_weight * coral_loss

        loss_dict = {
            "mmd": mmd_loss.item(),
            "coral": coral_loss.item(),
            "total": total.item(),
        }

        return total, loss_dict
