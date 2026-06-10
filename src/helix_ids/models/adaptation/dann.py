"""Domain Adversarial Neural Network (DANN) for distribution alignment.

DANN learns domain-invariant features by using a gradient reversal layer
that makes the feature extractor produce representations that are
indistinguishable between source (train) and target (test) domains.

Reference:
    Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
from torch.autograd import Function

logger = logging.getLogger(__name__)


class GradientReversalFunction(Function):
    """Gradient reversal function for adversarial training.

    During forward pass: identity function
    During backward pass: multiply gradient by -lambda
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """Gradient reversal layer for domain adversarial training.

    Implements the gradient reversal trick: during forward pass, acts as
    identity. During backward pass, reverses gradients by multiplying by -λ.
    """

    def __init__(self, lambda_: float = 1.0):
        """Initialize gradient reversal layer.

        Args:
            lambda_: Gradient reversal coefficient
        """
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reversed_x = GradientReversalFunction.apply(x, self.lambda_)
        return cast(torch.Tensor, reversed_x)

    def set_lambda(self, lambda_: float) -> None:
        """Update the gradient reversal coefficient."""
        self.lambda_ = lambda_


class DomainDiscriminator(nn.Module):
    """Domain discriminator network.

    Binary classifier that tries to distinguish source from target domain.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.5,
    ):
        """Initialize domain discriminator.

        Args:
            input_dim: Feature dimension from encoder
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
        """
        if hidden_dims is None:
            hidden_dims = [256, 128]
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.discriminator = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning domain logits.

        Args:
            x: Feature tensor [batch, input_dim]

        Returns:
            Domain logits [batch, 1]
        """
        logits = self.discriminator(x)
        return cast(torch.Tensor, logits)


@dataclass
class DANNConfig:
    """Configuration for DANN model."""

    # Feature extractor
    input_dim: int = 41  # NSL-KDD features
    encoder_dims: list[int] | None = None
    encoder_dropout: float = 0.3

    # Classifier
    num_classes: int = 5
    classifier_dims: list[int] | None = None
    classifier_dropout: float = 0.3

    # Domain discriminator
    discriminator_dims: list[int] | None = None
    discriminator_dropout: float = 0.5

    # Training
    lambda_init: float = 0.0
    lambda_max: float = 1.0
    lambda_gamma: float = 10.0  # For scheduling
    adversarial_weight: float = 1.0

    def __post_init__(self):
        if self.encoder_dims is None:
            self.encoder_dims = [256, 128, 64]
        if self.classifier_dims is None:
            self.classifier_dims = [64, 32]
        if self.discriminator_dims is None:
            self.discriminator_dims = [64, 32]


class DANN(nn.Module):
    """Domain Adversarial Neural Network.

    Architecture:
        Input → FeatureExtractor → [GRL → DomainDiscriminator]
                                 → [Classifier]

    The gradient reversal layer (GRL) makes the feature extractor learn
    domain-invariant representations that work well on both source and
    target domains.
    """

    def __init__(self, config: DANNConfig | None = None):
        """Initialize DANN model.

        Args:
            config: DANN configuration. Uses defaults if None.
        """
        super().__init__()
        self.config = config or DANNConfig()
        encoder_dims = cast(list[int], self.config.encoder_dims)
        discriminator_dims = cast(list[int], self.config.discriminator_dims)

        # Build feature extractor
        self.feature_extractor = self._build_encoder()

        # Get feature dimension
        self.feature_dim = encoder_dims[-1]

        # Build task classifier
        self.classifier = self._build_classifier()

        # Build domain discriminator
        self.domain_discriminator = DomainDiscriminator(
            input_dim=self.feature_dim,
            hidden_dims=discriminator_dims,
            dropout=self.config.discriminator_dropout,
        )

        # Gradient reversal layer
        self.grl = GradientReversalLayer(self.config.lambda_init)

        # Current training progress (for lambda scheduling)
        self._progress = 0.0

    def _build_encoder(self) -> nn.Module:
        """Build feature extractor network."""
        layers = []
        prev_dim = self.config.input_dim
        encoder_dims = cast(list[int], self.config.encoder_dims)

        for hidden_dim in encoder_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.config.encoder_dropout),
                ]
            )
            prev_dim = hidden_dim

        return nn.Sequential(*layers)

    def _build_classifier(self) -> nn.Module:
        """Build task classifier network."""
        layers = []
        prev_dim = self.feature_dim
        classifier_dims = cast(list[int], self.config.classifier_dims)

        for hidden_dim in classifier_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.config.classifier_dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.config.num_classes))

        return nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for classification.

        Args:
            x: Input tensor [batch, input_dim]
            return_features: Whether to also return features

        Returns:
            Class logits [batch, num_classes], or tuple with features
        """
        features = cast(torch.Tensor, self.feature_extractor(x))
        logits = cast(torch.Tensor, self.classifier(features))

        if return_features:
            return logits, features
        return logits

    def forward_dann(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for DANN training.

        Args:
            x_source: Source domain inputs [batch, input_dim]
            x_target: Target domain inputs [batch, input_dim]

        Returns:
            Tuple of (source_class_logits, source_domain_logits, target_domain_logits)
        """
        # Extract features
        features_source = self.feature_extractor(x_source)
        features_target = self.feature_extractor(x_target)

        # Task classification (source only, as target has no labels)
        class_logits = self.classifier(features_source)

        # Domain classification (both domains)
        features_source_grl = self.grl(features_source)
        features_target_grl = self.grl(features_target)

        domain_logits_source = self.domain_discriminator(features_source_grl)
        domain_logits_target = self.domain_discriminator(features_target_grl)

        return class_logits, domain_logits_source, domain_logits_target

    def update_lambda(self, progress: float) -> float:
        """Update gradient reversal coefficient based on training progress.

        Uses the schedule from the original DANN paper:
            λ = 2 / (1 + exp(-γ * p)) - 1

        where p is training progress in [0, 1] and γ controls the shape.

        Args:
            progress: Training progress in [0, 1]

        Returns:
            Updated lambda value
        """
        import math

        self._progress = progress

        # Schedule from DANN paper
        lambda_ = self.config.lambda_max * (
            2.0 / (1.0 + math.exp(-self.config.lambda_gamma * progress)) - 1.0
        )

        self.grl.set_lambda(lambda_)

        return lambda_

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features for visualization or analysis.

        Args:
            x: Input tensor [batch, input_dim]

        Returns:
            Feature tensor [batch, feature_dim]
        """
        features = self.feature_extractor(x)
        return cast(torch.Tensor, features)


class DANNLoss(nn.Module):
    """Combined loss for DANN training.

    L = L_task + λ * L_domain

    Where L_task is cross-entropy for classification and L_domain is
    binary cross-entropy for domain discrimination.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        adversarial_weight: float = 1.0,
    ):
        """Initialize DANN loss.

        Args:
            class_weights: Weights for task classification loss
            adversarial_weight: Weight for domain adversarial loss
        """
        super().__init__()
        self.class_weights = class_weights
        self.adversarial_weight = adversarial_weight

        # Task loss (cross-entropy)
        self.task_criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Domain loss (binary cross-entropy)
        self.domain_criterion = nn.BCEWithLogitsLoss()

    def forward(
        self,
        class_logits: torch.Tensor,
        class_labels: torch.Tensor,
        domain_logits_source: torch.Tensor,
        domain_logits_target: torch.Tensor,
        lambda_: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined DANN loss.

        Args:
            class_logits: Task classification logits [batch, num_classes]
            class_labels: Task classification labels [batch]
            domain_logits_source: Source domain logits [batch, 1]
            domain_logits_target: Target domain logits [batch, 1]
            lambda_: Current gradient reversal coefficient

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Task loss
        task_loss = self.task_criterion(class_logits, class_labels)

        # Domain labels: 0 for source, 1 for target
        source_labels = torch.zeros_like(domain_logits_source)
        target_labels = torch.ones_like(domain_logits_target)

        # Domain loss
        domain_loss_source = self.domain_criterion(domain_logits_source, source_labels)
        domain_loss_target = self.domain_criterion(domain_logits_target, target_labels)
        domain_loss = (domain_loss_source + domain_loss_target) / 2

        # Combined loss
        # Note: gradient reversal is in the model, so we ADD domain loss
        total_loss = task_loss + self.adversarial_weight * domain_loss

        loss_dict = {
            "task_loss": task_loss.item(),
            "domain_loss": domain_loss.item(),
            "domain_source": domain_loss_source.item(),
            "domain_target": domain_loss_target.item(),
            "lambda": lambda_,
            "total_loss": total_loss.item(),
        }

        return total_loss, loss_dict


def create_dann_model(
    input_dim: int = 41,
    num_classes: int = 5,
    encoder_dims: list[int] | None = None,
    lambda_max: float = 1.0,
) -> DANN:
    """Factory function to create DANN model.

    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        encoder_dims: Feature extractor hidden dimensions
        lambda_max: Maximum gradient reversal coefficient

    Returns:
        Configured DANN model
    """
    config = DANNConfig(
        input_dim=input_dim,
        num_classes=num_classes,
        encoder_dims=encoder_dims,
        lambda_max=lambda_max,
    )
    return DANN(config)
