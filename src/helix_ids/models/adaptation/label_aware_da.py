"""Label-Aware Domain Adaptation for handling label shift in cross-dataset transfer.

This module extends standard DA methods (DANN, MMD, CORAL) to handle label shift,
which occurs when attack class distributions vary significantly across datasets
(e.g., R2L/U2R present in one dataset but absent or rare in another).

Key techniques:
1. Conditional Adversarial DANN: Separate domain discriminators per class
2. Partial Transfer: Down-weight source classes absent in target
3. Class-Conditional MMD: Align distributions separately per class
4. Sample Reweighting: Weight source samples by class importance

Reference:
    Chen et al., "Partial Transfer Learning with Selective Adversarial Networks",
    CVPR 2018
    Cao et al., "Partial Transfer Learning with Instance Adaptation Regression",
    IJCAI 2019
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn

from .coral_loss import CORALLoss
from .dann import DomainDiscriminator, GradientReversalLayer
from .mmd_loss import MMDLoss, gaussian_kernel, multi_kernel

logger = logging.getLogger(__name__)


@dataclass
class LabelAwareDAConfig:
    """Configuration for label-aware domain adaptation."""

    # Feature extractor
    input_dim: int = 41
    encoder_dims: list[int] | None = None
    encoder_dropout: float = 0.3

    # Classifier
    num_classes: int = 5
    classifier_dims: list[int] | None = None
    classifier_dropout: float = 0.3

    # Domain discriminator (per-class)
    discriminator_dims: list[int] | None = None
    discriminator_dropout: float = 0.5

    # Label-aware DA settings
    use_conditional_dann: bool = True  # Conditional adversarial DA
    use_partial_transfer: bool = True  # Handle label shift
    use_class_conditional_mmd: bool = True  # Class-conditional alignment

    # Training
    lambda_init: float = 0.0
    lambda_max: float = 1.0
    lambda_gamma: float = 10.0
    adversarial_weight: float = 1.0
    mmd_weight: float = 0.5
    coral_weight: float = 0.25
    reweight_coeff: float = 0.5  # For partial transfer reweighting

    # MMD settings
    mmd_kernel: str = "multi"
    mmd_bandwidths: list[float] | None = None

    def __post_init__(self):
        if self.encoder_dims is None:
            self.encoder_dims = [256, 128, 64]
        if self.classifier_dims is None:
            self.classifier_dims = [64, 32]
        if self.discriminator_dims is None:
            self.discriminator_dims = [64, 32]
        if self.mmd_bandwidths is None:
            self.mmd_bandwidths = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


class ConditionalDomainDiscriminator(nn.Module):
    """Per-class domain discriminators for conditional adversarial DA.

    Instead of a single global domain discriminator, maintain separate
    discriminators for each class to enable class-conditional alignment.
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.5,
    ):
        """Initialize conditional domain discriminators.

        Args:
            num_classes: Number of classes
            input_dim: Feature dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
        """
        if hidden_dims is None:
            hidden_dims = [64, 32]
        super().__init__()

        self.num_classes = num_classes

        # Create a domain discriminator for each class
        self.discriminators = nn.ModuleList(
            [DomainDiscriminator(input_dim, hidden_dims, dropout) for _ in range(num_classes)]
        )

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for class-conditional domain classification.

        Args:
            features: Feature tensor [batch, input_dim]
            labels: Class labels [batch]

        Returns:
            Domain logits [batch, 1]
        """
        batch_size = features.size(0)
        device = features.device
        domain_logits = torch.zeros(batch_size, 1, device=device)

        # For each class, use the corresponding discriminator
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                class_features = features[mask]
                # Skip if batch size is too small for BatchNorm
                if len(class_features) > 1:
                    class_logits = self.discriminators[c](class_features)
                    domain_logits[mask] = class_logits
                else:
                    # For single samples, use discriminator in eval mode temporarily
                    self.discriminators[c].eval()
                    with torch.no_grad():
                        class_logits = self.discriminators[c](class_features)
                    self.discriminators[c].train()
                    domain_logits[mask] = class_logits

        return domain_logits

    def forward_per_class(
        self,
        features: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Get domain logits for each class separately (no label filtering).

        Useful for inference or analysis.

        Args:
            features: Feature tensor [batch, input_dim]

        Returns:
            List of domain logits for each class [batch, 1] x num_classes
        """
        logits_per_class = []
        for c in range(self.num_classes):
            logits = self.discriminators[c](features)
            logits_per_class.append(logits)
        return logits_per_class


class PartialTransferReweighter(nn.Module):
    """Handle label shift via partial transfer learning.

    Reweight source samples based on target class distribution to mitigate
    negative transfer from source classes absent in target (e.g., U2R in
    target domain when source has many U2R samples).

    Strategy:
    1. Estimate target class distribution from pseudo-labels
    2. Identify "shared" classes (present in both domains)
    3. Down-weight "source-only" classes
    4. Up-weight classes more common in target
    """

    def __init__(
        self,
        num_classes: int,
        reweight_coeff: float = 0.5,
    ):
        """Initialize partial transfer reweighter.

        Args:
            num_classes: Number of classes
            reweight_coeff: Coefficient for reweighting (0-1)
                - 0: No reweighting (standard DA)
                - 1: Full reweighting to match target distribution
        """
        super().__init__()
        self.num_classes = num_classes
        self.reweight_coeff = reweight_coeff

        # Initialize uniform target distribution estimate
        self.register_buffer(
            "target_class_dist",
            torch.ones(num_classes) / num_classes,
        )

    def update_target_distribution(
        self,
        target_pseudo_labels: torch.Tensor,
    ) -> None:
        """Update estimate of target class distribution.

        Args:
            target_pseudo_labels: Pseudo-labels for target domain [n_target]
        """
        # Compute class distribution in target domain
        dist = torch.bincount(
            target_pseudo_labels,
            minlength=self.num_classes,
        ).float()
        dist = dist / dist.sum()

        # Smooth estimate to avoid extreme reweighting
        self.target_class_dist = dist

    def compute_sample_weights(
        self,
        source_labels: torch.Tensor,
        source_class_dist: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-sample importance weights for partial transfer.

        Samples from classes abundant in source but rare in target are
        down-weighted to reduce negative transfer.

        Args:
            source_labels: Source class labels [n_source]
            source_class_dist: Source class distribution. If None, computed
                from source_labels.

        Returns:
            Sample weights [n_source]
        """
        source_labels.size(0)

        # Compute source class distribution if not provided
        if source_class_dist is None:
            dist = torch.bincount(
                source_labels,
                minlength=self.num_classes,
            ).float()
            source_class_dist = dist / dist.sum()

        # Identify shared classes (present in both domains)
        target_has_class = self.target_class_dist > 1e-6
        source_has_class = source_class_dist > 1e-6
        shared_classes = target_has_class & source_has_class

        # Compute importance weights
        # w_c = min(p_t(c), p_s(c)) / p_s(c)
        # This down-weights source classes not in target
        weights = torch.zeros(self.num_classes, device=source_labels.device)

        for c in range(self.num_classes):
            if source_class_dist[c] > 1e-6:
                if shared_classes[c]:
                    # For shared classes: ratio of target to source proportion
                    ratio = self.target_class_dist[c] / source_class_dist[c]
                    weights[c] = ratio
                else:
                    # For source-only classes: down-weight heavily
                    weights[c] = 0.1 * self.reweight_coeff

        # Normalize weights
        max_weight = weights[weights > 0].max() if (weights > 0).any() else 1.0
        weights = weights / max_weight

        # Get per-sample weights
        sample_weights = weights[source_labels]

        # Interpolate: (1 - coeff) * 1.0 + coeff * computed_weight
        # This allows tuning how much to apply reweighting
        sample_weights = (1.0 - self.reweight_coeff) + self.reweight_coeff * sample_weights

        return sample_weights


class ClassConditionalMMDLoss(nn.Module):
    """Multi-kernel MMD with class conditioning.

    Aligns distributions separately for each class using multiple kernels,
    which is more effective than global alignment when class distributions
    differ significantly.
    """

    def __init__(
        self,
        num_classes: int,
        kernel: str = "multi",
        bandwidths: list[float] | None = None,
    ):
        """Initialize class-conditional MMD loss.

        Args:
            num_classes: Number of classes
            kernel: Kernel type ("gaussian" or "multi")
            bandwidths: Kernel bandwidths for multi-kernel
        """
        super().__init__()
        self.num_classes = num_classes
        self.kernel = kernel
        self.bandwidths = bandwidths or [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_labels: torch.Tensor,
        target_pseudo_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute class-conditional MMD loss.

        Args:
            source_features: Source features [n_s, d]
            target_features: Target features [n_t, d]
            source_labels: Source labels [n_s]
            target_pseudo_labels: Target pseudo-labels [n_t]

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        observed_classes = torch.unique(torch.cat((source_labels, target_pseudo_labels))).numel()
        if int(observed_classes) < 2:
            loss = torch.tensor(0.0, device=source_features.device)
            loss_dict: dict[str, float] = {
                "mmd_loss": 0.0,
                "total": 0.0,
                "classes_evaluated": 0.0,
            }
            assert loss is not None
            assert isinstance(loss_dict, dict)
            assert bool(torch.isfinite(loss).item())
            return loss, loss_dict

        total_loss = torch.tensor(0.0, dtype=source_features.dtype, device=source_features.device)
        n_shared_classes = 0
        loss_dict = {}

        for c in range(self.num_classes):
            src_mask = source_labels == c
            tgt_mask = target_pseudo_labels == c

            src_c = source_features[src_mask]
            tgt_c = target_features[tgt_mask]

            # Skip if either domain has too few samples of this class
            if len(src_c) < 2 or len(tgt_c) < 2:
                continue

            # Compute class-specific MMD using multi-kernel
            if self.kernel == "multi":
                k_ss = multi_kernel(src_c, src_c, self.bandwidths)
                k_tt = multi_kernel(tgt_c, tgt_c, self.bandwidths)
                k_st = multi_kernel(src_c, tgt_c, self.bandwidths)
            else:
                k_ss = gaussian_kernel(src_c, src_c)
                k_tt = gaussian_kernel(tgt_c, tgt_c)
                k_st = gaussian_kernel(src_c, tgt_c)

            # Unbiased MMD^2 estimate
            n_s = src_c.size(0)
            n_t = tgt_c.size(0)

            diag_s = torch.diag(k_ss)
            diag_t = torch.diag(k_tt)

            sum_ss = (k_ss.sum() - diag_s.sum()) / (n_s * (n_s - 1))
            sum_tt = (k_tt.sum() - diag_t.sum()) / (n_t * (n_t - 1))
            sum_st = k_st.mean()

            mmd_sq = sum_ss + sum_tt - 2 * sum_st
            mmd_sq = torch.clamp(mmd_sq, min=0)

            total_loss = total_loss + mmd_sq
            loss_dict[f"mmd_class_{c}"] = float(mmd_sq.item())
            n_shared_classes += 1

        if n_shared_classes > 0:
            total_loss = total_loss / n_shared_classes

        loss_dict["mmd_loss"] = float(total_loss.item())
        loss_dict["total"] = float(total_loss.item())
        loss_dict["classes_evaluated"] = float(n_shared_classes)

        assert total_loss is not None
        assert isinstance(loss_dict, dict)
        assert bool(torch.isfinite(total_loss).item())

        return total_loss, loss_dict


class LabelAwareDANN(nn.Module):
    """Label-Aware Domain Adversarial Neural Network.

    Extends standard DANN with:
    1. Conditional domain discriminators (per-class)
    2. Partial transfer reweighting
    3. Class-conditional MMD loss
    4. Handles label shift across datasets

    Architecture:
        Input → FeatureExtractor → Classifier
                                → [GRL → ConditionalDomainDiscriminator]
                                → [ClassConditionalMMD]
    """

    def __init__(self, config: LabelAwareDAConfig | None = None):
        """Initialize label-aware DANN.

        Args:
            config: Configuration object. Uses defaults if None.
        """
        super().__init__()
        self.config = config or LabelAwareDAConfig()
        encoder_dims = self.config.encoder_dims or [256, 128, 64]
        discriminator_dims = self.config.discriminator_dims or [64, 32]

        # Build feature extractor
        self.feature_extractor = self._build_encoder()
        self.feature_dim = encoder_dims[-1]

        # Build task classifier
        self.classifier = self._build_classifier()

        # Conditional domain discriminators
        self.domain_discriminator: ConditionalDomainDiscriminator | DomainDiscriminator
        if self.config.use_conditional_dann:
            self.domain_discriminator = ConditionalDomainDiscriminator(
                num_classes=self.config.num_classes,
                input_dim=self.feature_dim,
                hidden_dims=discriminator_dims,
                dropout=self.config.discriminator_dropout,
            )
        else:
            # Fall back to standard single discriminator
            self.domain_discriminator = DomainDiscriminator(
                input_dim=self.feature_dim,
                hidden_dims=discriminator_dims,
                dropout=self.config.discriminator_dropout,
            )

        # Gradient reversal layer
        self.grl = GradientReversalLayer(self.config.lambda_init)

        # Class-conditional MMD
        self.mmd_loss: ClassConditionalMMDLoss | MMDLoss
        if self.config.use_class_conditional_mmd:
            self.mmd_loss = ClassConditionalMMDLoss(
                num_classes=self.config.num_classes,
                kernel=self.config.mmd_kernel,
                bandwidths=self.config.mmd_bandwidths,
            )
        else:
            self.mmd_loss = MMDLoss(kernel=self.config.mmd_kernel)

        # CORAL loss for covariance alignment
        self.coral_loss = CORALLoss(normalize=True)

        # Partial transfer reweighter
        self.reweighter: PartialTransferReweighter | None
        if self.config.use_partial_transfer:
            self.reweighter = PartialTransferReweighter(
                num_classes=self.config.num_classes,
                reweight_coeff=self.config.reweight_coeff,
            )
        else:
            self.reweighter = None

        self._progress = 0.0

    def _build_encoder(self) -> nn.Module:
        """Build feature extractor network."""
        layers = []
        prev_dim = self.config.input_dim
        encoder_dims = self.config.encoder_dims or [256, 128, 64]

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
        classifier_dims = self.config.classifier_dims or [64, 32]

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
            return_features: Whether to return features

        Returns:
            Class logits or (logits, features)
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        logits_t = cast(torch.Tensor, logits)
        features_t = cast(torch.Tensor, features)

        if return_features:
            return logits_t, features_t
        return logits_t

    def forward_label_aware_da(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
        y_source: torch.Tensor,
        y_target_pseudo: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, float] | None]:
        """Forward pass for label-aware DA training.

        Args:
            x_source: Source inputs [n_s, input_dim]
            x_target: Target inputs [n_t, input_dim]
            y_source: Source labels [n_s]
            y_target_pseudo: Target pseudo-labels [n_t]

        Returns:
            Dictionary with logits and intermediate values
        """
        # Extract features
        features_source = self.feature_extractor(x_source)
        features_target = self.feature_extractor(x_target)

        # Task classification (source only)
        class_logits = self.classifier(features_source)

        # Update partial transfer reweighting
        if self.reweighter is not None:
            self.reweighter.update_target_distribution(y_target_pseudo)

        # Domain classification (conditional, per-class)
        features_source_grl = self.grl(features_source)
        features_target_grl = self.grl(features_target)

        if isinstance(self.domain_discriminator, ConditionalDomainDiscriminator):
            domain_logits_source = self.domain_discriminator(
                features_source_grl,
                y_source,
            )
            domain_logits_target = self.domain_discriminator(
                features_target_grl,
                y_target_pseudo,
            )
        else:
            # Standard domain discriminator
            domain_logits_source = self.domain_discriminator(features_source_grl)
            domain_logits_target = self.domain_discriminator(features_target_grl)

        # Class-conditional MMD
        if self.config.use_class_conditional_mmd:
            mmd_loss, mmd_dict = self.mmd_loss(
                features_source,
                features_target,
                y_source,
                y_target_pseudo,
            )
        else:
            mmd_loss = self.mmd_loss(features_source, features_target)
            mmd_dict = {"mmd": mmd_loss.item()}

        # Compute CORAL loss
        coral_loss = self.coral_loss(features_source, features_target)

        return {
            "class_logits": class_logits,
            "domain_logits_source": domain_logits_source,
            "domain_logits_target": domain_logits_target,
            "features_source": features_source,
            "features_target": features_target,
            "mmd_loss": mmd_loss,
            "coral_loss": coral_loss,
            "sample_weights": (
                self.reweighter.compute_sample_weights(y_source)
                if self.reweighter is not None
                else None
            ),
            "mmd_dict": mmd_dict,
        }

    def update_lambda(self, progress: float) -> float:
        """Update gradient reversal coefficient.

        Args:
            progress: Training progress in [0, 1]

        Returns:
            Updated lambda value
        """
        import math

        self._progress = progress
        lambda_ = self.config.lambda_max * (
            2.0 / (1.0 + math.exp(-self.config.lambda_gamma * progress)) - 1.0
        )
        self.grl.set_lambda(lambda_)
        return lambda_

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features for analysis.

        Args:
            x: Input tensor [batch, input_dim]

        Returns:
            Features [batch, feature_dim]
        """
        return cast(torch.Tensor, self.feature_extractor(x))


class LabelAwareDALoss(nn.Module):
    """Combined loss for label-aware DANN training.

    L_total = L_task + λ * (L_dann + w_mmd * L_mmd + w_coral * L_coral)

    With reweighting from partial transfer for handling label shift.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        adversarial_weight: float = 1.0,
        mmd_weight: float = 0.5,
        coral_weight: float = 0.25,
    ):
        """Initialize label-aware DA loss.

        Args:
            class_weights: Class weights for task classification
            adversarial_weight: Weight for domain adversarial loss
            mmd_weight: Weight for MMD loss
            coral_weight: Weight for CORAL loss
        """
        super().__init__()
        self.class_weights = class_weights
        self.adversarial_weight = adversarial_weight
        self.mmd_weight = mmd_weight
        self.coral_weight = coral_weight

        # Task loss
        self.task_criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Domain loss
        self.domain_criterion = nn.BCEWithLogitsLoss()

    def forward(
        self,
        forward_outputs: dict[str, torch.Tensor],
        lambda_: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined label-aware DA loss.

        Args:
            forward_outputs: Dictionary from forward_label_aware_da
            lambda_: Current gradient reversal coefficient

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        forward_outputs["class_logits"]
        domain_logits_source = forward_outputs["domain_logits_source"]
        domain_logits_target = forward_outputs["domain_logits_target"]
        mmd_loss = forward_outputs["mmd_loss"]
        coral_loss = forward_outputs["coral_loss"]
        forward_outputs["sample_weights"]

        # Get labels (should be in forward_outputs)
        # Note: These should be passed separately to forward() or set externally
        # For now, we'll compute them from the forward pass context
        # This is a placeholder - actual implementation needs refactoring

        # Task loss (with per-sample reweighting for partial transfer)
        # This will be computed by the training loop
        # Here we just compute domain and adaptation losses

        # Domain labels
        domain_logits_source.size(0)
        domain_logits_target.size(0)

        source_labels = torch.zeros_like(domain_logits_source)
        target_labels = torch.ones_like(domain_logits_target)

        # Domain loss
        domain_loss_source = self.domain_criterion(domain_logits_source, source_labels)
        domain_loss_target = self.domain_criterion(domain_logits_target, target_labels)
        domain_loss = (domain_loss_source + domain_loss_target) / 2

        # Combined adaptation loss
        adaptation_loss = (
            self.adversarial_weight * domain_loss
            + self.mmd_weight * mmd_loss
            + self.coral_weight * coral_loss
        )

        loss_dict = {
            "domain_loss": domain_loss.item(),
            "domain_source": domain_loss_source.item(),
            "domain_target": domain_loss_target.item(),
            "mmd_loss": mmd_loss.item(),
            "coral_loss": coral_loss.item(),
            "adaptation_loss": adaptation_loss.item(),
            "lambda": lambda_,
        }

        # Add per-class MMD losses if available
        if "mmd_dict" in forward_outputs:
            loss_dict.update(forward_outputs["mmd_dict"])

        return adaptation_loss, loss_dict


def create_label_aware_dann(
    input_dim: int = 41,
    num_classes: int = 5,
    encoder_dims: list[int] | None = None,
    lambda_max: float = 1.0,
    use_conditional: bool = True,
    use_partial_transfer: bool = True,
    use_class_conditional_mmd: bool = True,
) -> LabelAwareDANN:
    """Factory function to create label-aware DANN.

    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        encoder_dims: Feature extractor dimensions
        lambda_max: Maximum gradient reversal coefficient
        use_conditional: Use conditional domain discriminators
        use_partial_transfer: Use partial transfer reweighting
        use_class_conditional_mmd: Use class-conditional MMD

    Returns:
        Configured label-aware DANN model
    """
    config = LabelAwareDAConfig(
        input_dim=input_dim,
        num_classes=num_classes,
        encoder_dims=encoder_dims,
        lambda_max=lambda_max,
        use_conditional_dann=use_conditional,
        use_partial_transfer=use_partial_transfer,
        use_class_conditional_mmd=use_class_conditional_mmd,
    )
    return LabelAwareDANN(config)
