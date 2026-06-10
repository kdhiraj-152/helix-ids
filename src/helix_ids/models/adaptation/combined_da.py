"""Combined Domain Adaptation Loss.

Ensemble of DANN, MMD, and CORAL for robust domain adaptation.
This module combines three complementary approaches:
- DANN: Adversarial domain confusion via gradient reversal
- MMD: Distribution alignment via kernel mean matching
- CORAL: Second-order statistics (covariance) alignment

Reference:
    Sun et al., "Revisiting Deep Domain Adaptation", 2019
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .coral_loss import CORALLoss
from .mmd_loss import MMDLoss


@dataclass
class CombinedDAConfig:
    """Configuration for combined domain adaptation."""

    # Loss weights
    dann_weight: float = 0.5
    mmd_weight: float = 0.25
    coral_weight: float = 0.25

    # Overall DA strength
    lambda_da: float = 1.0

    # MMD kernel settings
    mmd_kernel: str = "multi"
    mmd_bandwidths: list[float] | None = None

    # CORAL settings
    coral_normalize: bool = True

    # GRL scheduling
    lambda_max: float = 1.0
    lambda_gamma: float = 10.0

    def __post_init__(self):
        # Validate weights sum to 1.0 (approximately)
        total = self.dann_weight + self.mmd_weight + self.coral_weight
        if abs(total - 1.0) > 1e-6:
            # Normalize weights
            self.dann_weight /= total
            self.mmd_weight /= total
            self.coral_weight /= total


class CombinedDomainAdaptation(nn.Module):
    """Combined domain adaptation using DANN + MMD + CORAL ensemble.

    This module provides a unified interface for multi-loss domain adaptation.
    Each loss component targets different aspects of distribution alignment:
    - DANN: Learns domain-invariant features through adversarial training
    - MMD: Minimizes distribution distance in kernel space
    - CORAL: Aligns feature covariance matrices

    Args:
        config: Configuration object, or uses defaults if None
        dann_weight: Weight for DANN loss (default: 0.5)
        mmd_weight: Weight for MMD loss (default: 0.25)
        coral_weight: Weight for CORAL loss (default: 0.25)
        lambda_da: Overall domain adaptation strength (default: 1.0)
        mmd_kernel: MMD kernel type ("gaussian" or "multi")
    """

    def __init__(
        self,
        config: CombinedDAConfig | None = None,
        dann_weight: float = 0.5,
        mmd_weight: float = 0.25,
        coral_weight: float = 0.25,
        lambda_da: float = 1.0,
        mmd_kernel: str = "multi",
    ):
        super().__init__()

        # Use config if provided, otherwise use individual parameters
        if config is not None:
            self.config = config
        else:
            self.config = CombinedDAConfig(
                dann_weight=dann_weight,
                mmd_weight=mmd_weight,
                coral_weight=coral_weight,
                lambda_da=lambda_da,
                mmd_kernel=mmd_kernel,
            )

        # Initialize sub-losses
        self.mmd_loss = MMDLoss(kernel=self.config.mmd_kernel)
        self.coral_loss = CORALLoss(normalize=self.config.coral_normalize)

        # Domain classifier loss for DANN
        self.domain_criterion = nn.BCEWithLogitsLoss()

        # Gradient reversal lambda (for DANN scheduling)
        self.register_buffer("grl_lambda", torch.tensor(0.0))

        # Track training progress
        self._progress = 0.0

    def update_lambda(self, progress: float) -> float:
        """Update gradient reversal lambda based on training progress.

        Uses the schedule from the original DANN paper:
            λ = λ_max * (2 / (1 + exp(-γ * p)) - 1)

        where p is training progress in [0, 1] and γ controls the shape.

        Args:
            progress: Training progress from 0 to 1

        Returns:
            Updated lambda value
        """
        self._progress = progress

        # Standard GRL schedule from DANN paper
        new_lambda = self.config.lambda_max * (
            2.0 / (1.0 + math.exp(-self.config.lambda_gamma * progress)) - 1.0
        )

        self.grl_lambda.fill_(new_lambda * self.config.lambda_da)
        return new_lambda

    def compute_dann_loss(
        self,
        source_domain_logits: torch.Tensor,
        target_domain_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DANN domain classifier loss.

        Args:
            source_domain_logits: Domain logits for source samples [n_s, 1]
            target_domain_logits: Domain logits for target samples [n_t, 1]

        Returns:
            Binary cross-entropy loss for domain classification
        """
        # Source = 0, Target = 1
        source_labels = torch.zeros_like(source_domain_logits)
        target_labels = torch.ones_like(target_domain_logits)

        domain_logits = torch.cat([source_domain_logits, target_domain_logits], dim=0)
        domain_labels = torch.cat([source_labels, target_labels], dim=0)

        return self.domain_criterion(domain_logits, domain_labels)

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_domain_logits: torch.Tensor | None = None,
        target_domain_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined domain adaptation loss.

        Args:
            source_features: Feature representations from source domain [n_s, d]
            target_features: Feature representations from target domain [n_t, d]
            source_domain_logits: Domain logits for source (required for DANN)
            target_domain_logits: Domain logits for target (required for DANN)

        Returns:
            Dictionary with individual and combined losses:
            - mmd_loss: MMD component (if weight > 0)
            - coral_loss: CORAL component (if weight > 0)
            - dann_loss: DANN component (if weight > 0 and logits provided)
            - combined_da_loss: Weighted sum of all components
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=source_features.device)

        # MMD Loss - distribution alignment via kernel mean matching
        if self.config.mmd_weight > 0:
            mmd = self.mmd_loss(source_features, target_features)
            losses["mmd_loss"] = mmd
            total_loss = total_loss + self.config.mmd_weight * mmd

        # CORAL Loss - covariance alignment
        if self.config.coral_weight > 0:
            coral = self.coral_loss(source_features, target_features)
            losses["coral_loss"] = coral
            total_loss = total_loss + self.config.coral_weight * coral

        # DANN Loss - adversarial domain confusion (requires domain predictions)
        if self.config.dann_weight > 0 and source_domain_logits is not None:
            dann = self.compute_dann_loss(source_domain_logits, target_domain_logits)
            losses["dann_loss"] = dann
            # Apply GRL lambda scaling for DANN
            total_loss = total_loss + self.config.dann_weight * self.grl_lambda * dann

        losses["combined_da_loss"] = total_loss * self.config.lambda_da

        return losses


class CombinedDANNLoss(nn.Module):
    """Combined loss for DANN training with MMD and CORAL.

    Extends DANNLoss to include MMD and CORAL distribution alignment.

    L = L_task + λ * (w_dann * L_dann + w_mmd * L_mmd + w_coral * L_coral)
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        adversarial_weight: float = 1.0,
        dann_weight: float = 0.5,
        mmd_weight: float = 0.25,
        coral_weight: float = 0.25,
    ):
        """Initialize combined DANN loss.

        Args:
            class_weights: Weights for task classification loss
            adversarial_weight: Overall weight for domain adaptation losses
            dann_weight: Relative weight for DANN component
            mmd_weight: Relative weight for MMD component
            coral_weight: Relative weight for CORAL component
        """
        super().__init__()
        self.class_weights = class_weights
        self.adversarial_weight = adversarial_weight

        # Task loss (cross-entropy)
        self.task_criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Combined DA loss
        self.da_loss = CombinedDomainAdaptation(
            dann_weight=dann_weight,
            mmd_weight=mmd_weight,
            coral_weight=coral_weight,
        )

    def forward(
        self,
        class_logits: torch.Tensor,
        class_labels: torch.Tensor,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_domain_logits: torch.Tensor,
        target_domain_logits: torch.Tensor,
        lambda_: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss for DANN training.

        Args:
            class_logits: Task classification logits [batch, num_classes]
            class_labels: Task classification labels [batch]
            source_features: Source domain features [n_s, d]
            target_features: Target domain features [n_t, d]
            source_domain_logits: Source domain logits [n_s, 1]
            target_domain_logits: Target domain logits [n_t, 1]
            lambda_: Current gradient reversal coefficient

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Task loss
        task_loss = self.task_criterion(class_logits, class_labels)

        # Update lambda in DA loss
        self.da_loss.grl_lambda.fill_(lambda_)

        # Combined domain adaptation loss
        da_losses = self.da_loss(
            source_features,
            target_features,
            source_domain_logits,
            target_domain_logits,
        )

        # Combined loss
        da_total = da_losses["combined_da_loss"]
        total_loss = task_loss + self.adversarial_weight * da_total

        # Build loss dict
        loss_dict = {
            "task_loss": task_loss.item(),
            "combined_da_loss": da_total.item(),
            "lambda": lambda_,
            "total_loss": total_loss.item(),
        }

        # Add individual DA components
        if "dann_loss" in da_losses:
            loss_dict["dann_loss"] = da_losses["dann_loss"].item()
        if "mmd_loss" in da_losses:
            loss_dict["mmd_loss"] = da_losses["mmd_loss"].item()
        if "coral_loss" in da_losses:
            loss_dict["coral_loss"] = da_losses["coral_loss"].item()

        return total_loss, loss_dict


def create_combined_da_loss(
    dann_weight: float = 0.5,
    mmd_weight: float = 0.25,
    coral_weight: float = 0.25,
    lambda_da: float = 1.0,
    mmd_kernel: str = "multi",
) -> CombinedDomainAdaptation:
    """Factory function to create combined domain adaptation loss.

    Args:
        dann_weight: Weight for DANN loss
        mmd_weight: Weight for MMD loss
        coral_weight: Weight for CORAL loss
        lambda_da: Overall domain adaptation strength
        mmd_kernel: MMD kernel type

    Returns:
        Configured CombinedDomainAdaptation module
    """
    config = CombinedDAConfig(
        dann_weight=dann_weight,
        mmd_weight=mmd_weight,
        coral_weight=coral_weight,
        lambda_da=lambda_da,
        mmd_kernel=mmd_kernel,
    )
    return CombinedDomainAdaptation(config)
