"""
Hierarchical Classification Head for HELIX-IDS.

Key Innovation: Instead of direct 5-class classification (which causes minority class collapse),
use hierarchical structure that forces explicit learning of R2L/U2R vs Normal boundary.

Hierarchy:
- Level 1: Binary (Normal=0 vs Attack=1)
- Level 2: Attack Family (DoS=0, Probe=1, R2L=2, U2R=3) - only for attack samples
- Level 3: Fine-grained attack types (23 specific attacks in NSL-KDD)

This approach addresses the class imbalance problem by:
1. First learning to distinguish attacks from normal traffic
2. Then learning attack family classification conditioned on attack detection
3. Optionally learning fine-grained attack types for detailed forensics
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# NSL-KDD attack type mappings
ATTACK_FAMILIES = {
    "Normal": 0,
    "DoS": 1,
    "Probe": 2,
    "R2L": 3,
    "U2R": 4,
}

FAMILY_TO_FINE_GRAINED = {
    "DoS": [
        "back",
        "land",
        "neptune",
        "pod",
        "smurf",
        "teardrop",
        "apache2",
        "mailbomb",
        "processtable",
        "udpstorm",
    ],
    "Probe": ["ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"],
    "R2L": [
        "ftp_write",
        "guess_passwd",
        "imap",
        "multihop",
        "phf",
        "spy",
        "warezclient",
        "warezmaster",
        "sendmail",
        "named",
        "snmpgetattack",
        "snmpguess",
        "xlock",
        "xsnoop",
        "worm",
    ],
    "U2R": [
        "buffer_overflow",
        "loadmodule",
        "perl",
        "rootkit",
        "httptunnel",
        "ps",
        "sqlattack",
        "xterm",
    ],
}

# Total 23 fine-grained attack types (common subset)
FINE_GRAINED_ATTACKS = (
    FAMILY_TO_FINE_GRAINED["DoS"][:6]  # 6 DoS
    + FAMILY_TO_FINE_GRAINED["Probe"][:4]  # 4 Probe
    + FAMILY_TO_FINE_GRAINED["R2L"][:8]  # 8 R2L
    + FAMILY_TO_FINE_GRAINED["U2R"][:5]  # 5 U2R
)


@dataclass
class ClassifierConfig:
    """Configuration for HierarchicalClassifier."""

    hidden_dim: int = 128
    num_binary_classes: int = 2  # Normal, Attack
    num_family_classes: int = 4  # DoS, Probe, R2L, U2R
    num_fine_classes: int = 23  # Fine-grained attack types
    dropout: float = 0.3
    enable_fine_grained: bool = True
    enable_confidence: bool = True
    use_layer_norm: bool = True

    # Variant presets
    @classmethod
    def nano(cls) -> "ClassifierConfig":
        """Nano variant for extreme edge (Raspberry Pi Zero)."""
        return cls(
            hidden_dim=32,
            dropout=0.1,
            enable_fine_grained=False,
            enable_confidence=False,
            use_layer_norm=False,
        )

    @classmethod
    def lite(cls) -> "ClassifierConfig":
        """Lite variant for edge devices (Raspberry Pi 4)."""
        return cls(
            hidden_dim=64,
            dropout=0.2,
            enable_fine_grained=False,
            enable_confidence=True,
            use_layer_norm=True,
        )

    @classmethod
    def full(cls) -> "ClassifierConfig":
        """Full variant for server deployment."""
        return cls(
            hidden_dim=128,
            dropout=0.3,
            enable_fine_grained=True,
            enable_confidence=True,
            use_layer_norm=True,
        )


class ClassificationHead(nn.Module):
    """Single classification head with optional conditioning."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        condition_dim: int = 0,
        dropout: float = 0.3,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        total_input = input_dim + condition_dim

        self.layers = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        features: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Input features [batch_size, input_dim]
            condition: Optional conditioning tensor [batch_size, condition_dim]

        Returns:
            Logits [batch_size, output_dim]
        """
        if condition is not None:
            features = torch.cat([features, condition], dim=-1)
        return self.layers(features)  # type: ignore[no-any-return]


class ConfidenceHead(nn.Module):
    """
    Confidence calibration head.

    Predicts the probability that the model's prediction is correct.
    Useful for drift detection and uncertainty quantification.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        features: torch.Tensor,
        binary_logits: torch.Tensor,
        family_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Hidden features [batch_size, input_dim]
            binary_logits: Binary classification logits [batch_size, 2]
            family_logits: Family classification logits [batch_size, 4]

        Returns:
            Confidence scores [batch_size, 1]
        """
        # Concatenate features with prediction entropy
        binary_probs = F.softmax(binary_logits, dim=-1)
        family_probs = F.softmax(family_logits, dim=-1)

        # Compute entropy as uncertainty measure
        binary_entropy = -torch.sum(
            binary_probs * torch.log(binary_probs + 1e-8), dim=-1, keepdim=True
        )
        family_entropy = -torch.sum(
            family_probs * torch.log(family_probs + 1e-8), dim=-1, keepdim=True
        )

        # Max probability as confidence proxy
        binary_max_prob = torch.max(binary_probs, dim=-1, keepdim=True)[0]
        family_max_prob = torch.max(family_probs, dim=-1, keepdim=True)[0]

        # Combine all signals
        combined = torch.cat(
            [
                features,
                binary_entropy,
                family_entropy,
                binary_max_prob,
                family_max_prob,
            ],
            dim=-1,
        )

        return self.layers(combined)  # type: ignore[no-any-return]


class HierarchicalClassifier(nn.Module):
    """
    Hierarchical Classification Head for HELIX-IDS.

    Implements a three-level hierarchy:
    - Level 1: Binary (Normal vs Attack)
    - Level 2: Attack Family (DoS, Probe, R2L, U2R)
    - Level 3: Fine-grained attack types (23 specific attacks)

    This hierarchical approach forces the model to:
    1. First learn robust attack detection
    2. Then learn attack family boundaries (especially R2L/U2R)
    3. Optionally learn fine-grained classification

    Args:
        input_dim: Dimension of input features
        config: ClassifierConfig instance
    """

    def __init__(
        self,
        input_dim: int,
        config: Optional[ClassifierConfig] = None,
    ):
        super().__init__()

        self.config = config or ClassifierConfig()
        self.input_dim = input_dim

        # Level 1: Binary classification (Normal vs Attack)
        self.binary_head = ClassificationHead(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.num_binary_classes,
            condition_dim=0,
            dropout=self.config.dropout,
            use_layer_norm=self.config.use_layer_norm,
        )

        # Level 2: Attack family classification (conditioned on binary)
        self.family_head = ClassificationHead(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.num_family_classes,
            condition_dim=self.config.num_binary_classes,  # Conditioned on binary logits
            dropout=self.config.dropout,
            use_layer_norm=self.config.use_layer_norm,
        )

        # Level 3: Fine-grained classification (optional)
        self.fine_head: Optional[ClassificationHead] = None
        if self.config.enable_fine_grained:
            self.fine_head = ClassificationHead(
                input_dim=input_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=self.config.num_fine_classes,
                condition_dim=self.config.num_family_classes,  # Conditioned on family logits
                dropout=self.config.dropout,
                use_layer_norm=self.config.use_layer_norm,
            )

        # Confidence calibration head (optional)
        self.confidence_head: Optional[ConfidenceHead] = None
        if self.config.enable_confidence:
            # Input: features + entropy/prob signals (4 extra dimensions)
            self.confidence_head = ConfidenceHead(
                input_dim=input_dim + 4,
                hidden_dim=self.config.hidden_dim // 2,
                dropout=self.config.dropout,
            )

        # Family to 5-class mapping for predict_5class
        # Maps family indices (0-3) to 5-class indices (1-4), with Normal being 0
        self.register_buffer(
            "family_to_5class",
            torch.tensor([1, 2, 3, 4], dtype=torch.long),
        )

    def forward(
        self,
        features: torch.Tensor,
        return_intermediates: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical classifier.

        Args:
            features: Input features [batch_size, input_dim]
            return_intermediates: If True, include intermediate representations

        Returns:
            Dictionary with keys:
            - 'binary': Binary logits [batch_size, 2]
            - 'family': Family logits [batch_size, 4]
            - 'fine': Fine-grained logits [batch_size, 23] (if enabled)
            - 'confidence': Confidence scores [batch_size, 1] (if enabled)
        """
        outputs = {}

        # Level 1: Binary classification
        binary_logits = self.binary_head(features)
        outputs["binary"] = binary_logits

        # Level 2: Family classification (conditioned on binary)
        # Use soft conditioning with logits/probabilities
        binary_probs = F.softmax(binary_logits, dim=-1)
        family_logits = self.family_head(features, condition=binary_probs)
        outputs["family"] = family_logits

        # Level 3: Fine-grained classification (optional, conditioned on family)
        if self.fine_head is not None:
            family_probs = F.softmax(family_logits, dim=-1)
            fine_logits = self.fine_head(features, condition=family_probs)
            outputs["fine"] = fine_logits

        # Confidence estimation (optional)
        if self.confidence_head is not None:
            confidence = self.confidence_head(features, binary_logits, family_logits)
            outputs["confidence"] = confidence

        if return_intermediates:
            outputs["binary_probs"] = binary_probs
            outputs["family_probs"] = F.softmax(family_logits, dim=-1)
            if self.fine_head is not None:
                outputs["fine_probs"] = F.softmax(outputs["fine"], dim=-1)

        return outputs

    def predict_5class(
        self,
        features: torch.Tensor,
        threshold: float = 0.5,
        return_probs: bool = False,
    ) -> torch.Tensor:
        """
        Convert hierarchical outputs to standard 5-class predictions.

        Maps hierarchical predictions to:
        - 0: Normal
        - 1: DoS
        - 2: Probe
        - 3: R2L
        - 4: U2R

        Args:
            features: Input features [batch_size, input_dim]
            threshold: Binary classification threshold
            return_probs: If True, return probabilities instead of class indices

        Returns:
            If return_probs=False: Class predictions [batch_size]
            If return_probs=True: Class probabilities [batch_size, 5]
        """
        outputs = self.forward(features)

        binary_probs = F.softmax(outputs["binary"], dim=-1)
        family_probs = F.softmax(outputs["family"], dim=-1)

        attack_prob = binary_probs[:, 1:2]  # [batch_size, 1]
        normal_prob = binary_probs[:, 0:1]  # [batch_size, 1]

        # P(family|attack) * P(attack) for each attack family
        # Family indices: DoS=0, Probe=1, R2L=2, U2R=3
        attack_family_probs = family_probs * attack_prob  # [batch_size, 4]

        # Combine into 5-class probabilities
        # Order: Normal, DoS, Probe, R2L, U2R
        five_class_probs = torch.cat([normal_prob, attack_family_probs], dim=-1)

        if return_probs:
            return five_class_probs

        # Return argmax predictions
        return torch.argmax(five_class_probs, dim=-1)

    def predict_with_confidence(
        self,
        features: torch.Tensor,
        confidence_threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with confidence scores.

        Args:
            features: Input features [batch_size, input_dim]
            confidence_threshold: Threshold for high-confidence predictions

        Returns:
            Tuple of:
            - predictions: 5-class predictions [batch_size]
            - confidence: Confidence scores [batch_size]
            - high_confidence_mask: Boolean mask for high-confidence predictions [batch_size]
        """
        outputs = self.forward(features)
        predictions = self.predict_5class(features)

        if self.confidence_head is not None:
            confidence = outputs["confidence"].squeeze(-1)
        else:
            # Use max probability as fallback confidence
            five_class_probs = self.predict_5class(features, return_probs=True)
            confidence = torch.max(five_class_probs, dim=-1)[0]

        high_confidence_mask = confidence >= confidence_threshold

        return predictions, confidence, high_confidence_mask

    def get_hierarchical_loss_weights(
        self,
        binary_labels: torch.Tensor,
        family_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Get sample weights for hierarchical training.

        Family loss should only be computed for attack samples.
        Fine-grained loss should only be computed for respective family samples.

        Args:
            binary_labels: Binary labels (0=Normal, 1=Attack) [batch_size]
            family_labels: Family labels (0-3 for attacks) [batch_size]

        Returns:
            Dictionary with loss weights for each level
        """
        # Binary loss: all samples
        binary_weights = torch.ones_like(binary_labels, dtype=torch.float)

        # Family loss: only attack samples
        family_weights = (binary_labels == 1).float()

        return {
            "binary": binary_weights,
            "family": family_weights,
        }

    def freeze_binary_head(self):
        """Freeze binary head for curriculum learning."""
        for param in self.binary_head.parameters():
            param.requires_grad = False

    def unfreeze_binary_head(self):
        """Unfreeze binary head."""
        for param in self.binary_head.parameters():
            param.requires_grad = True

    def get_num_parameters(self) -> dict[str, int]:
        """Get parameter counts for each head."""
        counts = {
            "binary_head": sum(p.numel() for p in self.binary_head.parameters()),
            "family_head": sum(p.numel() for p in self.family_head.parameters()),
        }

        if self.fine_head is not None:
            counts["fine_head"] = sum(p.numel() for p in self.fine_head.parameters())

        if self.confidence_head is not None:
            counts["confidence_head"] = sum(p.numel() for p in self.confidence_head.parameters())

        counts["total"] = sum(counts.values())

        return counts


# Convenience factory functions for variants


def hierarchical_classifier_nano(input_dim: int) -> HierarchicalClassifier:
    """
    Nano variant for extreme edge deployment (e.g., Raspberry Pi Zero).

    - Minimal hidden dimensions (32)
    - No fine-grained classification
    - No confidence head
    - ~2K parameters
    """
    return HierarchicalClassifier(input_dim, ClassifierConfig.nano())


def hierarchical_classifier_lite(input_dim: int) -> HierarchicalClassifier:
    """
    Lite variant for edge devices (e.g., Raspberry Pi 4, Jetson Nano).

    - Moderate hidden dimensions (64)
    - No fine-grained classification
    - Includes confidence head
    - ~8K parameters
    """
    return HierarchicalClassifier(input_dim, ClassifierConfig.lite())


def hierarchical_classifier_full(input_dim: int) -> HierarchicalClassifier:
    """
    Full variant for server/cloud deployment.

    - Full hidden dimensions (128)
    - Includes fine-grained classification
    - Includes confidence head
    - ~25K parameters
    """
    return HierarchicalClassifier(input_dim, ClassifierConfig.full())


HierarchicalClassifierNano = hierarchical_classifier_nano
HierarchicalClassifierLite = hierarchical_classifier_lite
HierarchicalClassifierFull = hierarchical_classifier_full


# Utility functions


def convert_labels_to_hierarchical(
    five_class_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert 5-class labels to hierarchical format.

    Args:
        five_class_labels: Labels in [0-4] format
                          (0=Normal, 1=DoS, 2=Probe, 3=R2L, 4=U2R)

    Returns:
        Tuple of (binary_labels, family_labels)
        - binary_labels: 0 for Normal, 1 for Attack
        - family_labels: 0-3 for attack families (0=DoS, 1=Probe, 2=R2L, 3=U2R)
                        -1 for Normal samples (to be masked in loss)
    """
    # Binary: Normal(0) vs Attack(1-4)
    binary_labels = (five_class_labels > 0).long()

    # Family: Map 1-4 to 0-3, set Normal to -1 (will be masked)
    family_labels = five_class_labels.clone()
    family_labels[family_labels > 0] -= 1  # Shift attack labels: 1-4 -> 0-3
    family_labels[five_class_labels == 0] = -1  # Mark Normal as -1 for masking

    return binary_labels, family_labels


def hierarchical_to_5class(
    binary_pred: torch.Tensor,
    family_pred: torch.Tensor,
) -> torch.Tensor:
    """
    Convert hierarchical predictions back to 5-class format.

    Args:
        binary_pred: Binary predictions (0=Normal, 1=Attack)
        family_pred: Family predictions (0-3 for attacks)

    Returns:
        5-class predictions (0=Normal, 1-4=Attack families)
    """
    result = torch.zeros_like(binary_pred)

    # Where binary predicts attack, use family + 1
    attack_mask = binary_pred == 1
    result[attack_mask] = family_pred[attack_mask] + 1

    return result
