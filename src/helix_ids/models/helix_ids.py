"""
HELIX-IDS: Hierarchical Edge-optimized Lightweight Intrusion eXpert

Main model combining:
- Temporal Attention Module (TAM)
- Hierarchical Classification Head
- Threat-Aware Focal Loss

Three variants for different edge deployment targets:
- HELIX-Nano: <30KB for ESP32
- HELIX-Lite: <200KB for RPi Zero
- HELIX-Full: <2MB for RPi 4
"""

from dataclasses import dataclass
from typing import Optional, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import create_tam
from .classifier import (
    HierarchicalClassifierFull,
    HierarchicalClassifierLite,
    HierarchicalClassifierNano,
)
from .loss import MultiTaskLoss, create_loss_function


@dataclass
class HELIXConfig:
    """Configuration for HELIX-IDS model."""

    # Model variant
    variant: str = "lite"  # 'nano', 'lite', 'full'

    # Input dimensions
    input_dim: int = 41  # NSL-KDD features

    # Architecture
    hidden_dim: int = 48
    attention_heads: int = 4
    num_classes: int = 5  # Normal, DoS, Probe, R2L, U2R
    num_fine_classes: int = 23  # Fine-grained attack types

    # Backbone layers
    backbone_layers: tuple[int, ...] = (96, 64, 48)

    # Regularization
    dropout: float = 0.2

    # Curriculum learning
    use_curriculum: bool = True
    curriculum_epochs: tuple[int, int, int, int] = (1, 25, 55, 85)

    # Edge optimization
    quantization_aware: bool = False
    target_size_kb: float = 200.0


# Predefined variant configurations
HELIX_VARIANTS = {
    "nano": HELIXConfig(
        variant="nano",
        hidden_dim=32,
        attention_heads=2,
        backbone_layers=(64, 32),
        dropout=0.15,
        target_size_kb=30.0,
    ),
    "lite": HELIXConfig(
        variant="lite",
        hidden_dim=48,
        attention_heads=4,
        backbone_layers=(96, 64, 48),
        dropout=0.2,
        target_size_kb=200.0,
    ),
    "full": HELIXConfig(
        variant="full",
        hidden_dim=64,
        attention_heads=4,
        backbone_layers=(128, 96, 64),
        dropout=0.25,
        target_size_kb=2000.0,
    ),
}


class FeatureBackbone(nn.Module):
    """
    Feature extraction backbone with configurable layers.

    Converts raw input features to a latent representation
    suitable for attention and classification.
    """

    def __init__(
        self, input_dim: int, hidden_dims: tuple[int, ...], output_dim: int, dropout: float = 0.2
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = h_dim

        # Final projection to output_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.network(x))


class HELIXIDS(nn.Module):
    """
    HELIX-IDS: Hierarchical Edge-optimized Lightweight Intrusion eXpert

    A production-ready intrusion detection system designed to solve
    minority class suppression while maintaining edge deployability.

    Key features:
    - Hierarchical classification (Binary → Family → Fine-grained)
    - Temporal attention for feature importance learning
    - Threat-aware focal loss with curriculum learning
    - Confidence calibration for uncertainty estimation

    Args:
        config: HELIXConfig or variant name ('nano', 'lite', 'full')
    """

    def __init__(self, config: Union[HELIXConfig, str] = "lite"):
        super().__init__()

        # Handle string variant names
        if isinstance(config, str):
            if config not in HELIX_VARIANTS:
                raise ValueError(
                    f"Unknown variant: {config}. Choose from {list(HELIX_VARIANTS.keys())}"
                )
            config = HELIX_VARIANTS[config]

        self.config = config

        # Feature extraction backbone
        self.backbone = FeatureBackbone(
            input_dim=config.input_dim,
            hidden_dims=config.backbone_layers,
            output_dim=config.hidden_dim,
            dropout=config.dropout,
        )

        # Temporal Attention Module
        self.attention = create_tam(
            variant=config.variant,
            n_features=config.hidden_dim,
            n_attack_classes=config.num_classes,
        )

        # Hierarchical Classification Head
        classifier_factories = {
            "nano": HierarchicalClassifierNano,
            "lite": HierarchicalClassifierLite,
            "full": HierarchicalClassifierFull,
        }
        self.classifier = classifier_factories[config.variant](input_dim=config.hidden_dim)

        # Loss function (created separately, not part of model params)
        self.loss_fn: Optional[MultiTaskLoss] = None

        # Track curriculum phase
        self.current_epoch = 0

    def forward(
        self, x: torch.Tensor, attack_logits: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through HELIX-IDS.

        Args:
            x: Input features [batch_size, input_dim]
            attack_logits: Optional attack logits for attention conditioning
                          [batch_size, num_classes]

        Returns:
            Dictionary with keys:
            - 'binary': Binary classification logits [batch, 2]
            - 'family': Attack family logits [batch, 4]
            - 'fine': Fine-grained attack logits [batch, num_fine_classes] (if enabled)
            - 'confidence': Confidence score [batch, 1] (if enabled)
            - 'features': Attended features [batch, hidden_dim]
        """
        # Feature extraction
        features = self.backbone(x)

        # Temporal attention with optional attack conditioning
        attended_features = self.attention(features, attack_logits)

        # Hierarchical classification
        classifier_output = self.classifier(attended_features)

        # Combine outputs
        output = {
            "binary": classifier_output["binary"],
            "family": classifier_output["family"],
            "features": attended_features,
        }

        # Backward-compatible aliases for callers expecting flat logits keys.
        output["binary_logits"] = classifier_output["binary"]
        output["logits"] = classifier_output["binary"]

        # Add optional outputs
        if "fine" in classifier_output:
            output["fine"] = classifier_output["fine"]
        if "confidence" in classifier_output:
            output["confidence"] = classifier_output["confidence"]

        return output

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict 5-class labels (Normal, DoS, Probe, R2L, U2R).

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Predicted class indices [batch_size]
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)

            binary_pred = torch.argmax(output["binary"], dim=1)

            # In binary mode, return binary labels directly.
            if self.config.num_classes <= 2:
                return binary_pred

            family_pred = torch.argmax(output["family"], dim=1)

            # Combine: if binary=0 (Normal), class=0; else use family+1
            predictions = torch.where(
                binary_pred == 0, torch.zeros_like(family_pred), family_pred + 1
            )

            return predictions

    def predict_with_confidence(
        self, x: torch.Tensor, confidence_threshold: float = 0.8
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with confidence scores.

        Args:
            x: Input features
            confidence_threshold: Threshold for high-confidence predictions

        Returns:
            Tuple of (predictions, confidence_scores, high_confidence_mask)
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            predictions = self.predict(x)

            # Use confidence head if available, else use max probability
            if "confidence" in output:
                confidence = output["confidence"].squeeze(-1)
            else:
                proba = self.predict_proba(x)
                confidence = proba.max(dim=1)[0]

            high_conf_mask = confidence >= confidence_threshold

            return predictions, confidence, high_conf_mask

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Class probabilities [batch_size, 5]
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)

            # Get binary and family probabilities
            binary_proba = F.softmax(output["binary"], dim=1)  # [batch, 2]

            if self.config.num_classes <= 2:
                return binary_proba

            family_proba = F.softmax(output["family"], dim=1)  # [batch, 4]

            # Combine: P(class) = P(binary) * P(family|attack)
            # Class 0 (Normal): P(binary=0)
            # Class 1-4: P(binary=1) * P(family=k)

            proba = torch.zeros(x.size(0), 5, device=x.device)
            proba[:, 0] = binary_proba[:, 0]  # Normal
            proba[:, 1:] = binary_proba[:, 1:2] * family_proba  # Attack families

            return proba

    def get_feature_importance(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get learned feature importance scores.

        Args:
            x: Input features

        Returns:
            Feature importance scores [batch_size, hidden_dim] or None
        """
        self.eval()
        with torch.no_grad():
            features = self.backbone(x)
            _ = self.attention(features)  # Run forward to populate attention weights
            return self.attention.get_feature_importance()

    def set_epoch(self, epoch: int):
        """Update curriculum learning phase based on epoch."""
        self.current_epoch = epoch
        if self.loss_fn is not None:
            self.loss_fn.set_epoch(epoch)

    def compute_loss(
        self, output: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute loss using threat-aware focal loss with curriculum.

        Args:
            output: Model output dictionary
            targets: Dictionary with 'binary', 'family', 'fine' labels

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        if self.loss_fn is None:
            self.loss_fn = cast(
                MultiTaskLoss,
                create_loss_function(
                    "multi_task",
                    num_classes=self.config.num_classes,
                    num_fine_classes=self.config.num_fine_classes,
                    curriculum_epochs=self.config.curriculum_epochs,
                ),
            )
            self.loss_fn.set_epoch(self.current_epoch)

        return cast(
            tuple[torch.Tensor, dict[str, torch.Tensor]],
            self.loss_fn(output, targets),
        )

    def count_parameters(self) -> dict[str, int]:
        """Count parameters by component."""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        attention_params = sum(p.numel() for p in self.attention.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        total_params = sum(p.numel() for p in self.parameters())

        return {
            "backbone": backbone_params,
            "attention": attention_params,
            "classifier": classifier_params,
            "total": total_params,
        }

    def estimate_size_kb(self) -> float:
        """Estimate model size in KB (float32)."""
        total_params = sum(p.numel() for p in self.parameters())
        return float(total_params * 4 / 1024)  # 4 bytes per float32

    def check_size_constraint(self) -> tuple[bool, str]:
        """Check if model meets size constraint for target variant."""
        actual_kb = self.estimate_size_kb()
        target_kb = self.config.target_size_kb
        meets_constraint = actual_kb <= target_kb

        message = (
            f"HELIX-{self.config.variant.capitalize()}: "
            f"{actual_kb:.2f} KB / {target_kb:.2f} KB target"
        )

        if meets_constraint:
            message += " ✓"
        else:
            message += f" ✗ (exceeds by {actual_kb - target_kb:.2f} KB)"

        return meets_constraint, message


def create_helix_model(
    variant: str = "lite",
    input_dim: int = 41,
    num_classes: int = 5,
    num_fine_classes: int = 23,
    **kwargs,
) -> HELIXIDS:
    """
    Factory function to create HELIX-IDS model.

    Args:
        variant: 'nano', 'lite', or 'full'
        input_dim: Number of input features
        num_classes: Number of main classes (default: 5)
        num_fine_classes: Number of fine-grained attack types
        **kwargs: Additional config overrides

    Returns:
        HELIXIDS model instance
    """
    if variant not in HELIX_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")

    config = HELIX_VARIANTS[variant]
    config.input_dim = input_dim
    config.num_classes = num_classes
    config.num_fine_classes = num_fine_classes

    # Apply any overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return HELIXIDS(config)


class HELIXEnsemble(nn.Module):
    """
    Multi-tier HELIX ensemble for adaptive edge deployment.

    Implements escalation logic:
    1. Nano predicts first (fastest, lowest power)
    2. If confidence < threshold, escalate to Lite
    3. If still uncertain, escalate to Full

    This achieves power savings on easy samples while maintaining
    accuracy on difficult samples.
    """

    def __init__(
        self, input_dim: int = 41, nano_threshold: float = 0.85, lite_threshold: float = 0.90
    ):
        super().__init__()

        self.nano = create_helix_model("nano", input_dim=input_dim)
        self.lite = create_helix_model("lite", input_dim=input_dim)
        self.full = create_helix_model("full", input_dim=input_dim)

        self.nano_threshold = nano_threshold
        self.lite_threshold = lite_threshold

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through appropriate tier."""
        # Start with Nano
        predictions, confidence, _ = self.nano.predict_with_confidence(x, self.nano_threshold)

        # Track which samples need escalation
        needs_lite = confidence < self.nano_threshold
        needs_full = torch.zeros_like(needs_lite)

        # Escalate uncertain samples to Lite
        if needs_lite.any():
            lite_preds, lite_conf, _ = self.lite.predict_with_confidence(
                x[needs_lite], self.lite_threshold
            )
            predictions[needs_lite] = lite_preds
            confidence[needs_lite] = lite_conf

            # Check if any need Full
            needs_full_subset = lite_conf < self.lite_threshold
            needs_full[needs_lite] = needs_full_subset

        # Escalate to Full for remaining uncertain samples
        if needs_full.any():
            full_preds, full_conf, _ = self.full.predict_with_confidence(x[needs_full])
            predictions[needs_full] = full_preds
            confidence[needs_full] = full_conf

        return {
            "predictions": predictions,
            "confidence": confidence,
            "tier_used": self._get_tier_used(needs_lite, needs_full),
        }

    def _get_tier_used(self, needs_lite: torch.Tensor, needs_full: torch.Tensor) -> torch.Tensor:
        """Return which tier was used for each sample (0=nano, 1=lite, 2=full)."""
        tier = torch.zeros_like(needs_lite, dtype=torch.long)
        tier[needs_lite] = 1
        tier[needs_full] = 2
        return tier

    def estimate_power_savings(self, tier_distribution: dict[int, float]) -> float:
        """
        Estimate power savings compared to always using Full.

        Args:
            tier_distribution: Dict of tier -> proportion of samples

        Returns:
            Estimated power savings percentage
        """
        # Relative power consumption (Full = 1.0)
        power_ratios = {0: 0.1, 1: 0.3, 2: 1.0}  # Nano=10%, Lite=30%, Full=100%

        weighted_power = sum(
            tier_distribution.get(tier, 0) * power_ratios[tier] for tier in power_ratios
        )

        savings = (1.0 - weighted_power) * 100
        return savings


# Export convenience aliases
def helix_nano(**kwargs):
    return create_helix_model("nano", **kwargs)


def helix_lite(**kwargs):
    return create_helix_model("lite", **kwargs)


def helix_full(**kwargs):
    return create_helix_model("full", **kwargs)


# Export convenience aliases (capitalized for consistency with codebase)
HELIXNano = helix_nano
HELIXLite = helix_lite
HELIXFull = helix_full
