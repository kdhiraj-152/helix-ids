"""
HelixIDS-Full: Production model for multi-dataset IDS.

Architecture:
- 4-layer MLP backbone (~500K params)
- Binary head (Normal vs Attack): 2 classes
- Family head (7-class attack families): 7 classes
- Multi-task loss: binary + family classification

No QAT (use PTQ in Phase 4 if needed).
Input: 17 audited invariant flow features
Output: 2-dim (binary) + 7-dim (family)
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class HelixFullConfig:
    """Configuration for HelixIDS-Full."""

    input_dim: int = 17  # audited invariant feature set (no dataset-origin columns)
    hidden_dims: tuple[int, ...] = (512, 384, 256, 256)  # ~500K params, matches existing checkpoints
    dropout_rates: tuple[float, ...] = (0.3, 0.3, 0.25, 0.2)
    activation: str = "relu"
    binary_output_dim: int = 2  # Normal vs Attack
    family_output_dim: int = 7  # 7-class attack families
    use_batch_norm: bool = True
    use_layer_norm: bool = False


# ============================================================================
# HelixIDS-Full Model
# ============================================================================


class HelixIDSFull(nn.Module):
    """
    HelixIDS-Full: Multi-task learning model for cross-dataset IDS.

    Architecture:
    - Shared backbone: 4-layer MLP
    - Binary head: Normal vs Attack (2-class)
    - Family head: DoS, Probe, R2L, U2R, Generic, Backdoor, + Normal (7-class)

    Forward pass:
    - Input: (batch_size, input_dim)
    - Output: (batch_size, 2) for binary, (batch_size, 7) for family
    """

    def __init__(self, config: Optional[HelixFullConfig] = None):
        super().__init__()

        if config is None:
            config = HelixFullConfig()

        self.config = config
        self.input_dim = config.input_dim

        # ===== Shared Backbone =====
        layers: list[nn.Module] = []
        prev_dim = config.input_dim

        for i, hidden_dim in enumerate(config.hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch norm (optional)
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            if config.activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif config.activation.lower() == "elu":
                layers.append(nn.ELU())
            else:
                raise ValueError(f"Unknown activation: {config.activation}")

            # Dropout
            layers.append(nn.Dropout(config.dropout_rates[i]))

            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # ===== Binary Head =====
        self.binary_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, config.binary_output_dim),
        )

        # ===== Family Head =====
        # Decouple family classification from raw backbone embedding.
        projection_hidden = max(128, int(prev_dim * 2))
        projection_bottleneck = max(64, int(prev_dim // 2))
        self.family_projection = nn.Sequential(
            nn.Linear(prev_dim, projection_hidden),
            nn.GELU(),
            nn.LayerNorm(projection_hidden),
            nn.Dropout(0.1),
            nn.Linear(projection_hidden, projection_bottleneck),
            nn.GELU(),
            nn.LayerNorm(projection_bottleneck),
        )
        self.family_whiten_eps = 1e-5

        # Can condition on binary output if needed, but start simple
        self.family_head = nn.Sequential(
            nn.Linear(projection_bottleneck, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, config.family_output_dim),
        )

        # Parameter count
        self.param_count = sum(p.numel() for p in self.parameters())

    def _whiten_family_features(self, family_features: torch.Tensor) -> torch.Tensor:
        """Whiten family features using per-batch stats (no running estimates)."""
        if family_features.ndim != 2:
            return family_features
        if int(family_features.shape[0]) <= 1:
            return family_features

        mean = family_features.mean(dim=0)
        var = family_features.var(dim=0, unbiased=False)
        return (family_features - mean) / torch.sqrt(var + float(self.family_whiten_eps))

    def forward(self, x: torch.Tensor, return_features: bool = False) -> tuple[torch.Tensor, ...]:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, input_dim)
            return_features: If True, also return backbone features (batch_size, hidden_dims[-1])

        Returns:
            (binary_logits, family_logits) each (batch_size, output_dim)
            If return_features=True: (binary_logits, family_logits, features)
        """
        # Handle batch size 1 by temporarily disabling batch norm training
        if x.shape[0] == 1 and self.training and self.config.use_batch_norm:
            was_training = self.training
            self.eval()
            with torch.no_grad():
                features = self.backbone(x)
            self.train(was_training)
        else:
            # Shared backbone
            features = self.backbone(x)

        # Binary head
        binary_logits = self.binary_head(features)

        # Family head
        family_features = self.family_projection(features)
        family_features = self._whiten_family_features(family_features)
        family_logits = self.family_head(family_features)

        if return_features:
            return binary_logits, family_logits, features
        else:
            return binary_logits, family_logits

    def get_param_count(self) -> int:
        """Get total parameter count."""
        return self.param_count


# ============================================================================
# Multi-Task Loss
# ============================================================================


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining binary classification and family classification.

    Loss = λ_binary * CE(binary_head, binary_labels)
         + λ_family * CE(family_head, family_labels)
         + λ_balance * class_weight_loss

    With optional class weighting for imbalanced datasets.
    """

    def __init__(
        self,
        lambda_binary: float = 1.0,
        lambda_family: float = 0.8,
        lambda_balance: float = 0.5,
        use_class_weights: bool = True,
        balance_strategy: str = "weighted_ce",
        focal_gamma: float = 1.25,
        focal_use_class_weights: bool = True,
        label_smoothing: float = 0.1,
        entropy_regularization: float = 0.0,
        family_logit_margin: float = 1.0,
        family_margin_loss_weight: float = 0.0,
        family_class4_logit_penalty_weight: float = 0.0,
        family_class4_logit_penalty_class: int = 4,
        family_feature_separation_weight: float = 0.0,
        family_feature_separation_class: int = 4,
        family_class4_target_scale: float = 1.0,
    ):
        super().__init__()

        self.lambda_binary = lambda_binary
        self.lambda_family = lambda_family
        self.lambda_balance = lambda_balance
        self.use_class_weights = use_class_weights
        self.balance_strategy = str(balance_strategy).strip().lower()
        self.focal_gamma = float(focal_gamma)
        self.focal_use_class_weights = bool(focal_use_class_weights)
        self.label_smoothing = float(label_smoothing)
        self.entropy_regularization = float(entropy_regularization)

        if self.balance_strategy not in {"weighted_ce", "focal"}:
            raise ValueError(
                "balance_strategy must be one of {'weighted_ce', 'focal'}, "
                f"got: {balance_strategy!r}"
            )
        if self.focal_gamma < 0.0:
            raise ValueError(f"focal_gamma must be >= 0.0, got: {self.focal_gamma}")
        if not (0.0 <= self.label_smoothing < 1.0):
            raise ValueError(
                "label_smoothing must be in [0.0, 1.0), "
                f"got: {self.label_smoothing}"
            )
        if self.entropy_regularization < 0.0:
            raise ValueError(
                "entropy_regularization must be >= 0.0, "
                f"got: {self.entropy_regularization}"
            )

        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")
        self.family_logit_margin = float(max(0.0, family_logit_margin))
        self.family_margin_loss_weight = float(max(0.0, family_margin_loss_weight))
        self.family_class4_logit_penalty_weight = float(
            max(0.0, family_class4_logit_penalty_weight)
        )
        self.family_class4_logit_penalty_class = int(family_class4_logit_penalty_class)
        self.family_feature_separation_weight = float(
            max(0.0, family_feature_separation_weight)
        )
        self.family_feature_separation_class = int(family_feature_separation_class)
        self.family_class4_target_scale = float(max(0.0, min(1.0, family_class4_target_scale)))

    def _family_target_scale_weights(self, labels: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Per-sample class-conditional scaling: class-4 labels get reduced target pressure."""
        if self.family_class4_target_scale >= 1.0:
            return torch.ones_like(labels, dtype=dtype, device=device)
        safe_labels = labels.to(device=device, dtype=torch.long)
        class4_mask = safe_labels == int(self.family_class4_logit_penalty_class)
        scale = torch.ones_like(safe_labels, dtype=dtype, device=device)
        scale = torch.where(
            class4_mask,
            torch.full_like(scale, fill_value=float(self.family_class4_target_scale)),
            scale,
        )
        return scale

    def _classification_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute per-task loss using the configured class-balance strategy."""
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        weights = class_weights if (self.use_class_weights and class_weights is not None) else None

        if self.balance_strategy == "weighted_ce":
            per_sample = F.cross_entropy(
                logits,
                labels,
                weight=weights,
                reduction="none",
                label_smoothing=self.label_smoothing,
            )
            if int(logits.shape[1]) >= 7:
                sample_scale = self._family_target_scale_weights(
                    labels,
                    dtype=per_sample.dtype,
                    device=per_sample.device,
                )
                per_sample = per_sample * sample_scale
            return torch.mean(per_sample)

        focal_weights = (
            weights if (self.focal_use_class_weights and weights is not None) else None
        )
        ce_loss = F.cross_entropy(
            logits,
            labels,
            weight=focal_weights,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        if int(logits.shape[1]) >= 7:
            sample_scale = self._family_target_scale_weights(
                labels,
                dtype=ce_loss.dtype,
                device=ce_loss.device,
            )
            ce_loss = ce_loss * sample_scale
        pt = torch.exp(-ce_loss)
        focal_modulation = (1.0 - pt).pow(self.focal_gamma)
        return torch.mean(focal_modulation * ce_loss)

    def _family_margin_penalty(
        self,
        family_logits: torch.Tensor,
        family_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Hinge-style class-separation penalty on family logits.

        penalty = max(0, margin - (logit_true - max_other))
        """
        if self.family_margin_loss_weight <= 0.0:
            return torch.zeros((), dtype=family_logits.dtype, device=family_logits.device)
        if int(family_logits.ndim) != 2 or int(family_logits.shape[0]) <= 0:
            return torch.zeros((), dtype=family_logits.dtype, device=family_logits.device)

        batch_idx = torch.arange(
            int(family_logits.shape[0]),
            device=family_logits.device,
            dtype=torch.long,
        )
        labels = family_labels.long().clamp(min=0, max=int(family_logits.shape[1]) - 1)
        true_logits = family_logits[batch_idx, labels]

        masked_logits = family_logits.clone()
        masked_logits[batch_idx, labels] = float("-inf")
        max_other = torch.max(masked_logits, dim=1).values
        max_other = torch.where(torch.isfinite(max_other), max_other, torch.zeros_like(true_logits))

        margins = true_logits - max_other
        hinge = F.relu(float(self.family_logit_margin) - margins)
        return torch.mean(hinge)

    def _family_class4_logit_penalty(
        self,
        family_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Relative ranking penalty on class-4 dominance over competing logits.

        penalty = mean(relu(logit_class4 - max_other_logits))
        """
        if self.family_class4_logit_penalty_weight <= 0.0:
            return torch.zeros((), dtype=family_logits.dtype, device=family_logits.device)
        if int(family_logits.ndim) != 2 or int(family_logits.shape[0]) <= 0:
            return torch.zeros((), dtype=family_logits.dtype, device=family_logits.device)

        class_idx = int(self.family_class4_logit_penalty_class)
        if class_idx < 0 or class_idx >= int(family_logits.shape[1]):
            return torch.zeros((), dtype=family_logits.dtype, device=family_logits.device)

        class_logits = family_logits[:, class_idx]
        competing_logits = family_logits.clone()
        competing_logits[:, class_idx] = float("-inf")
        max_other_logits = torch.max(competing_logits, dim=1).values
        max_other_logits = torch.where(
            torch.isfinite(max_other_logits),
            max_other_logits,
            torch.zeros_like(class_logits),
        )
        return torch.mean(F.relu(class_logits - max_other_logits))

    def _family_feature_separation_term(
        self,
        feature_embeddings: Optional[torch.Tensor],
        family_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Bounded negative centroid-separation term for class-4 vs non-class-4 features.

        L_sep = -tanh(||mean(z4) - mean(z_not4)||^2)
        """
        if feature_embeddings is None:
            return torch.zeros((), dtype=torch.float32, device=family_labels.device)
        if self.family_feature_separation_weight <= 0.0:
            return torch.zeros((), dtype=feature_embeddings.dtype, device=feature_embeddings.device)
        if int(feature_embeddings.ndim) != 2 or int(feature_embeddings.shape[0]) <= 0:
            return torch.zeros((), dtype=feature_embeddings.dtype, device=feature_embeddings.device)

        class_idx = int(self.family_feature_separation_class)
        labels = family_labels.to(device=feature_embeddings.device, dtype=torch.long)
        mask_class = labels == class_idx
        mask_not_class = labels != class_idx
        if int(mask_class.sum().item()) == 0 or int(mask_not_class.sum().item()) == 0:
            return torch.zeros((), dtype=feature_embeddings.dtype, device=feature_embeddings.device)

        z_class = feature_embeddings[mask_class]
        z_not_class = feature_embeddings[mask_not_class]
        mean_class = torch.mean(z_class, dim=0)
        mean_not_class = torch.mean(z_not_class, dim=0)
        sq_dist = torch.sum((mean_class - mean_not_class) ** 2)
        return -torch.tanh(sq_dist)

    def forward(
        self,
        binary_logits: torch.Tensor,
        binary_labels: torch.Tensor,
        family_logits: torch.Tensor,
        family_labels: torch.Tensor,
        binary_class_weights: Optional[torch.Tensor] = None,
        family_class_weights: Optional[torch.Tensor] = None,
        feature_embeddings: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute multi-task loss.

        Args:
            binary_logits: (batch_size, 2)
            binary_labels: (batch_size,)
            family_logits: (batch_size, 7)
            family_labels: (batch_size,)
            binary_class_weights: Optional (2,)
            family_class_weights: Optional (7,)

        Returns:
            (total_loss, dict of component losses)
        """
        # Binary classification loss
        ce_binary = self._classification_loss(
            binary_logits,
            binary_labels,
            binary_class_weights,
        )

        # Family classification loss
        ce_family = self._classification_loss(
            family_logits,
            family_labels,
            family_class_weights,
        )
        family_margin_penalty = self._family_margin_penalty(family_logits, family_labels)
        family_class4_logit_penalty = self._family_class4_logit_penalty(family_logits)
        family_feature_separation = self._family_feature_separation_term(
            feature_embeddings,
            family_labels,
        )

        entropy_bonus = torch.zeros((), dtype=ce_family.dtype, device=ce_family.device)
        if self.entropy_regularization > 0.0:
            family_prob = torch.softmax(family_logits, dim=1)
            safe_family_prob = torch.clamp(family_prob, min=1e-12, max=1.0)
            class_count = max(2.0, float(family_logits.shape[1]))
            norm = torch.log(torch.tensor(class_count, dtype=family_logits.dtype, device=family_logits.device))
            family_entropy = -torch.sum(family_prob * torch.log(safe_family_prob), dim=1) / norm
            entropy_bonus = self.entropy_regularization * torch.mean(family_entropy)

        # Total loss
        total_loss = (
            self.lambda_binary * ce_binary
            + self.lambda_family * ce_family
            + self.family_margin_loss_weight * family_margin_penalty
            + self.family_class4_logit_penalty_weight * family_class4_logit_penalty
            + self.family_feature_separation_weight * family_feature_separation
            - entropy_bonus
        )

        # Return loss and component breakdown
        loss_dict = {
            "total": total_loss.item(),
            "binary": ce_binary.item(),
            "family": ce_family.item(),
            "family_margin": family_margin_penalty.item(),
            "family_logit_margin": self.family_logit_margin,
            "family_margin_loss_weight": self.family_margin_loss_weight,
            "family_class4_logit_penalty": family_class4_logit_penalty.item(),
            "family_class4_logit_penalty_weight": self.family_class4_logit_penalty_weight,
            "family_class4_logit_penalty_class": self.family_class4_logit_penalty_class,
            "family_feature_separation": family_feature_separation.item(),
            "family_feature_separation_weight": self.family_feature_separation_weight,
            "family_feature_separation_class": self.family_feature_separation_class,
            "entropy_bonus": entropy_bonus.item(),
            "entropy_regularization": self.entropy_regularization,
            "balance_strategy": self.balance_strategy,
        }

        return total_loss, loss_dict


# ============================================================================
# Utility Functions
# ============================================================================


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_helix_full(config: Optional[HelixFullConfig] = None) -> HelixIDSFull:
    """Factory function to create HelixIDS-Full model."""
    if config is None:
        config = HelixFullConfig()
    return HelixIDSFull(config)


# ============================================================================
# Testing & Validation
# ============================================================================

if __name__ == "__main__":
    # Create model
    config = HelixFullConfig()
    model = create_helix_full(config)

    print("HelixIDS-Full Model")
    print("=" * 50)
    print(f"Input dimension: {config.input_dim}")
    print(f"Hidden dimensions: {config.hidden_dims}")
    print(f"Binary output: {config.binary_output_dim}")
    print(f"Family output: {config.family_output_dim}")
    print(f"Total parameters: {count_parameters(model):,}")
    print()

    # Test forward pass
    print("Testing forward pass...")
    batch_size = 32
    x = torch.randn(batch_size, config.input_dim)
    binary_logits, family_logits = model(x)

    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Binary output shape: {binary_logits.shape}")
    print(f"✅ Family output shape: {family_logits.shape}")

    # Test loss
    print("\nTesting multi-task loss...")
    loss_fn = MultiTaskLoss()
    binary_labels = torch.randint(0, 2, (batch_size,))
    family_labels = torch.randint(0, 6, (batch_size,))

    total_loss, loss_dict = loss_fn(binary_logits, binary_labels, family_logits, family_labels)

    print(f"✅ Total loss: {total_loss.item():.4f}")
    print(f"   Binary loss: {loss_dict['binary']:.4f}")
    print(f"   Family loss: {loss_dict['family']:.4f}")

    # Test with features
    print("\nTesting with feature extraction...")
    binary_logits, family_logits, features = model(x, return_features=True)
    print(f"✅ Features shape: {features.shape}")
