"""
Temporal Attention Module (TAM) for HELIX-IDS.

Key Innovation: Feature attention is CONDITIONED on preliminary attack family prediction,
allowing attack-specific feature weighting. This helps the model focus on different features
for different attack types (e.g., num_root for U2R, count for DoS).

Variants:
- TAM-Nano: 2 heads, 32 hidden dim (~2.6K params)
- TAM-Lite: 4 heads, 48 hidden dim (~7.4K params)
- TAM-Full: 4 heads, 64 hidden dim (~13K params)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionModule(nn.Module):
    """
    Multi-head self-attention over features with attack-conditioned modulation.

    The attention mechanism allows the model to learn which features are most
    relevant for each prediction. When attack_logits are provided, the attention
    weights are modulated based on the predicted attack type.

    Args:
        n_features: Number of input features (default: 41 for NSL-KDD)
        hidden_dim: Hidden dimension for attention (32, 48, or 64)
        n_heads: Number of attention heads (2 or 4)
        n_attack_classes: Number of attack family classes for conditioning (default: 5)
        dropout: Dropout rate for attention weights
    """

    def __init__(
        self,
        n_features: int = 41,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_attack_classes: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_attack_classes = n_attack_classes
        self.head_dim = hidden_dim // n_heads

        assert hidden_dim % n_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
        )

        # Feature embedding: project each feature to hidden_dim
        # We treat each feature as a "token" for self-attention
        self.feature_embedding = nn.Linear(1, hidden_dim)

        # Positional encoding for feature positions
        self.position_encoding = nn.Parameter(torch.randn(1, n_features, hidden_dim) * 0.02)

        # Multi-head attention projections
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        # Attack-conditioned modulation
        # Maps attack logits to attention bias for each head
        self.attack_modulation = nn.Sequential(
            nn.Linear(n_attack_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_heads),
        )

        # Attack-to-feature attention bias
        # Learns which features are important for each attack type
        self.attack_feature_bias = nn.Parameter(torch.zeros(n_attack_classes, n_features))

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # Feed-forward network after attention
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

        # Output projection back to feature space
        self.output_layer = nn.Linear(hidden_dim, 1)

        # Dropout
        self.attention_dropout = nn.Dropout(dropout)

        # Store attention weights for interpretability
        self._attention_weights: Optional[torch.Tensor] = None
        self._modulated_weights: Optional[torch.Tensor] = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for module in [self.query, self.key, self.value, self.output_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.feature_embedding.weight)
        nn.init.zeros_(self.feature_embedding.bias)

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def _scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attack_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention with optional attack-based modulation.

        Args:
            query: (batch, n_heads, n_features, head_dim)
            key: (batch, n_heads, n_features, head_dim)
            value: (batch, n_heads, n_features, head_dim)
            attack_bias: Optional (batch, n_heads, 1, 1) modulation factor

        Returns:
            output: (batch, n_heads, n_features, head_dim)
            attention_weights: (batch, n_heads, n_features, n_features)
        """
        scale = math.sqrt(self.head_dim)

        # Compute attention scores: (batch, n_heads, n_features, n_features)
        scores = torch.matmul(query, key.transpose(-2, -1)) / scale

        # Apply attack-conditioned modulation if provided
        if attack_bias is not None:
            # Modulate the attention scores per head
            scores = scores * (1.0 + attack_bias)

        # Softmax normalization
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, value)

        return output, attention_weights

    def forward(
        self,
        x: torch.Tensor,
        attack_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional attack-conditioned attention.

        Args:
            x: Input features of shape (batch_size, n_features)
            attack_logits: Optional attack family logits (batch_size, n_attack_classes)
                          Used to condition attention on predicted attack type

        Returns:
            Attended features of shape (batch_size, n_features)
        """
        batch_size = x.shape[0]

        # Reshape input: (batch, n_features) -> (batch, n_features, 1)
        x_expanded = x.unsqueeze(-1)

        # Embed each feature: (batch, n_features, hidden_dim)
        embedded = self.feature_embedding(x_expanded)

        # Add positional encoding
        embedded = embedded + self.position_encoding

        # Store for residual connection
        residual = embedded

        # Layer norm before attention
        embedded = self.layer_norm1(embedded)

        # Compute Q, K, V projections
        q = self.query(embedded)
        k = self.key(embedded)
        v = self.value(embedded)

        # Reshape for multi-head attention
        # (batch, n_features, hidden_dim) -> (batch, n_heads, n_features, head_dim)
        q = q.view(batch_size, self.n_features, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, self.n_features, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, self.n_features, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attack modulation if attack_logits provided
        attack_bias = None
        feature_bias = None
        if attack_logits is not None:
            # Normalize attack logits to probabilities
            attack_probs = F.softmax(attack_logits, dim=-1)

            # Compute per-head modulation: (batch, n_heads)
            head_modulation = self.attack_modulation(attack_probs)
            # Reshape for broadcasting: (batch, n_heads, 1, 1)
            attack_bias = head_modulation.unsqueeze(-1).unsqueeze(-1)

            # Compute feature-level bias based on attack type
            # (batch, n_attack_classes) @ (n_attack_classes, n_features) -> (batch, n_features)
            feature_bias = torch.matmul(attack_probs, self.attack_feature_bias)

        # Scaled dot-product attention
        attended, attention_weights = self._scaled_dot_product_attention(q, k, v, attack_bias)

        # Store attention weights for interpretability
        self._attention_weights = attention_weights.detach()

        # Reshape back: (batch, n_heads, n_features, head_dim) -> (batch, n_features, hidden_dim)
        attended = (
            attended.transpose(1, 2).contiguous().view(batch_size, self.n_features, self.hidden_dim)
        )

        # Output projection
        attended = self.output_proj(attended)

        # Residual connection
        attended = attended + residual

        # Second residual block with FFN
        residual2 = attended
        attended = self.layer_norm2(attended)
        attended = self.ffn(attended)
        attended = attended + residual2

        # Project back to feature space: (batch, n_features, hidden_dim) -> (batch, n_features, 1)
        output = self.output_layer(attended).squeeze(-1)

        # Apply feature-level bias from attack conditioning
        if feature_bias is not None:
            output = output + feature_bias
            self._modulated_weights = feature_bias.detach()
        else:
            self._modulated_weights = None

        # Residual connection from input
        output = output + x

        return output  # type: ignore[no-any-return]

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Get the attention weights from the last forward pass.

        Returns:
            Attention weights of shape (batch, n_heads, n_features, n_features)
            or None if forward() hasn't been called yet.
        """
        return self._attention_weights

    def get_feature_importance(self) -> Optional[torch.Tensor]:
        """
        Get aggregated feature importance from attention weights.

        Returns:
            Feature importance scores of shape (batch, n_features)
            or None if forward() hasn't been called yet.
        """
        if self._attention_weights is None:
            return None

        # Average attention received by each feature across heads
        # (batch, n_heads, n_features, n_features) -> (batch, n_features)
        importance = self._attention_weights.mean(dim=1).sum(dim=-2)
        return importance

    def get_attack_feature_bias(self) -> torch.Tensor:
        """
        Get the learned attack-to-feature bias matrix.

        Returns:
            Bias matrix of shape (n_attack_classes, n_features)
        """
        return self.attack_feature_bias.detach()

    def count_parameters(self) -> int:
        """
        Count the total number of trainable parameters.

        Returns:
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> dict:
        """
        Get a breakdown of parameters by component.

        Returns:
            Dictionary mapping component names to parameter counts.
        """
        breakdown = {
            "feature_embedding": sum(p.numel() for p in self.feature_embedding.parameters()),
            "position_encoding": self.position_encoding.numel(),
            "query_projection": sum(p.numel() for p in self.query.parameters()),
            "key_projection": sum(p.numel() for p in self.key.parameters()),
            "value_projection": sum(p.numel() for p in self.value.parameters()),
            "output_projection": sum(p.numel() for p in self.output_proj.parameters()),
            "attack_modulation": sum(p.numel() for p in self.attack_modulation.parameters()),
            "attack_feature_bias": self.attack_feature_bias.numel(),
            "layer_norms": sum(
                p.numel() for ln in [self.layer_norm1, self.layer_norm2] for p in ln.parameters()
            ),
            "ffn": sum(p.numel() for p in self.ffn.parameters()),
            "output_layer": sum(p.numel() for p in self.output_layer.parameters()),
        }
        breakdown["total"] = sum(breakdown.values())
        return breakdown


class TAMNano(TemporalAttentionModule):
    """
    Nano variant of TAM for extremely resource-constrained environments.

    Configuration:
    - 2 attention heads
    - 32 hidden dimension
    - ~2.6K parameters
    """

    def __init__(
        self,
        n_features: int = 41,
        n_attack_classes: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__(
            n_features=n_features,
            hidden_dim=32,
            n_heads=2,
            n_attack_classes=n_attack_classes,
            dropout=dropout,
        )


class TAMLite(TemporalAttentionModule):
    """
    Lite variant of TAM for resource-constrained edge devices.

    Configuration:
    - 4 attention heads
    - 48 hidden dimension
    - ~7.4K parameters
    """

    def __init__(
        self,
        n_features: int = 41,
        n_attack_classes: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__(
            n_features=n_features,
            hidden_dim=48,
            n_heads=4,
            n_attack_classes=n_attack_classes,
            dropout=dropout,
        )


class TAMFull(TemporalAttentionModule):
    """
    Full variant of TAM for maximum performance.

    Configuration:
    - 4 attention heads
    - 64 hidden dimension
    - ~13K parameters
    """

    def __init__(
        self,
        n_features: int = 41,
        n_attack_classes: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__(
            n_features=n_features,
            hidden_dim=64,
            n_heads=4,
            n_attack_classes=n_attack_classes,
            dropout=dropout,
        )


def create_tam(
    variant: str = "full",
    n_features: int = 41,
    n_attack_classes: int = 5,
    dropout: float = 0.1,
) -> TemporalAttentionModule:
    """
    Factory function to create TAM variants.

    Args:
        variant: One of "nano", "lite", or "full"
        n_features: Number of input features
        n_attack_classes: Number of attack classes for conditioning
        dropout: Dropout rate

    Returns:
        Configured TemporalAttentionModule instance
    """
    variants = {
        "nano": TAMNano,
        "lite": TAMLite,
        "full": TAMFull,
    }

    if variant.lower() not in variants:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(variants.keys())}")

    return variants[variant.lower()](
        n_features=n_features,
        n_attack_classes=n_attack_classes,
        dropout=dropout,
    )


if __name__ == "__main__":
    # Quick test and parameter counting
    print("TAM Variant Parameter Counts:")
    print("-" * 40)

    for name, cls in [("Nano", TAMNano), ("Lite", TAMLite), ("Full", TAMFull)]:
        model = cls(n_features=41)
        params = model.count_parameters()
        print(f"TAM-{name}: {params:,} parameters")

    print("\n" + "-" * 40)
    print("Testing forward pass...")

    # Test with batch of data
    batch_size = 32
    n_features = 41
    n_attack_classes = 5

    model = TAMFull(n_features=n_features)
    x = torch.randn(batch_size, n_features)
    attack_logits = torch.randn(batch_size, n_attack_classes)

    # Forward without attack conditioning
    out1 = model(x)
    print(f"Output shape (no conditioning): {out1.shape}")

    # Forward with attack conditioning
    out2 = model(x, attack_logits=attack_logits)
    print(f"Output shape (with conditioning): {out2.shape}")

    # Check attention weights
    weights = model.get_attention_weights()
    if weights is not None:
        print(f"Attention weights shape: {weights.shape}")

    # Feature importance
    importance = model.get_feature_importance()
    if importance is not None:
        print(f"Feature importance shape: {importance.shape}")

    # Parameter breakdown
    print("\nParameter breakdown (TAM-Full):")
    breakdown = model.parameter_breakdown()
    for component, count in breakdown.items():
        print(f"  {component}: {count:,}")
