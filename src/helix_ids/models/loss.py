"""
Threat-Aware Focal Loss and Multi-Task Loss for HELIX-IDS.

This module addresses the critical class imbalance problem in network intrusion
detection where rare but dangerous attacks (U2R, R2L) achieve F1=0.000 with
standard cross-entropy loss.

Key innovations (v2):
- Focal Loss: Down-weights easy examples, focuses on hard ones
- Conservative Threat Weighting: Prevents gradient collapse (Normal:1.0, DoS:1.2, Probe:1.5, R2L:3.0, U2R:4.0)
- Reduced Gamma: 1.5 instead of 2.0 for numerical stability
- Warmup Curriculum: Uses CrossEntropyLoss for first 10 epochs before switching to focal
- Label Smoothing: 0.1 default improves generalization
- Numerical Stability: p_t clamping prevents NaN/Inf
- Multi-Task Learning: Curriculum-based hierarchical classification
- Calibration Loss: Improves confidence estimation for uncertain predictions
"""

from typing import Optional, Union, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from helix_ids.contracts.attack_taxonomy import (
    threat_weight_tensor,
)

# Default threat weight tensor (indexed by class)
# Order: Normal=0, DoS=1, Probe=2, R2L=3, U2R=4
DEFAULT_THREAT_WEIGHTS = threat_weight_tensor()


def get_class_weights(
    y: Union[Tensor, np.ndarray],
    num_classes: Optional[int] = None,
    smoothing: float = 0.1,
    min_weight: float = 0.1,
    max_weight: float = 100.0,
) -> Tensor:
    """
    Compute inverse frequency class weights for balanced learning.

    Args:
        y: Target labels (1D tensor or array)
        num_classes: Number of classes (inferred from y if None)
        smoothing: Laplace smoothing to prevent division by zero
        min_weight: Minimum weight to prevent underflow
        max_weight: Maximum weight to prevent gradient explosion

    Returns:
        Tensor of shape (num_classes,) with inverse frequency weights

    Example:
        >>> y = torch.tensor([0, 0, 0, 1, 2])  # 3 class 0, 1 class 1, 1 class 2
        >>> weights = get_class_weights(y, num_classes=3)
        >>> # Class 0 gets lower weight, classes 1,2 get higher weights
    """
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)

    y = y.long().flatten()

    if num_classes is None:
        num_classes = int(y.max().item()) + 1

    # Count samples per class
    counts = torch.bincount(y, minlength=num_classes).float()

    # Apply Laplace smoothing
    counts = counts + smoothing

    # Compute inverse frequency weights
    total = counts.sum()
    weights = total / (num_classes * counts)

    # Clamp to prevent extreme values
    weights = weights.clamp(min=min_weight, max=max_weight)

    # Normalize to mean of 1.0 for stable gradients
    weights = weights / weights.mean()

    return weights


class ThreatAwareFocalLoss(nn.Module):
    """
    Threat-Aware Focal Loss for imbalanced intrusion detection.

    Combines three key mechanisms:
    1. Focal Loss: (1 - p_t)^γ down-weights easy examples
    2. Class Balancing: α_t from inverse class frequency
    3. Threat Weighting: w_t prioritizes dangerous attack types

    Loss formula:
        L = -α_t * (1 - p_t)^γ * log(p_t) * w_t

    FIXES (v2):
    - Conservative weights prevent gradient vanishing/explosion
    - Lower gamma (1.5) reduces numerical instability
    - Label smoothing (0.1) improves generalization
    - Optional gradient clipping for stability
    - Warmup epoch tracking for curriculum learning

    Args:
        gamma: Focal parameter (default=1.5). Lower than original 2.0
               to prevent vanishing gradients with threat weights.
        alpha: Class balance weights. Can be:
               - None: No class balancing
               - float: Applied to positive class (binary)
               - Tensor: Per-class weights
        threat_weights: Attack severity weights. Can be:
                       - None: Use DEFAULT_THREAT_WEIGHTS (conservative)
                       - Tensor: Custom weights per class
        reduction: 'mean', 'sum', or 'none'
        label_smoothing: Smoothing factor for soft labels (default=0.1)
        use_warmup: If True, use warmup epochs for curriculum (default=True)
        warmup_epochs: Number of warmup epochs using CrossEntropyLoss (default=10)

    Example:
        >>> loss_fn = ThreatAwareFocalLoss(gamma=1.5, label_smoothing=0.1)
        >>> logits = model(x)  # (batch, num_classes)
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        gamma: float = 1.5,
        alpha: Optional[Union[float, Tensor]] = None,
        threat_weights: Optional[Tensor] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.1,
        use_warmup: bool = True,
        warmup_epochs: int = 10,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.use_warmup = use_warmup
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 1

        # Register alpha as buffer (moves with model to device)
        if alpha is not None:
            if isinstance(alpha, (int, float)):
                alpha = torch.tensor([1 - alpha, alpha])
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

        # Register threat weights (conservative defaults prevent collapse)
        if threat_weights is None:
            threat_weights = DEFAULT_THREAT_WEIGHTS.clone()
        self.register_buffer("threat_weights", threat_weights)

        # Warmup CE loss for early epochs
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing)

    def set_epoch(self, epoch: int) -> None:
        """
        Set current epoch for warmup scheduling.

        Args:
            epoch: Current training epoch (1-indexed)
        """
        self.current_epoch = epoch

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Compute threat-aware focal loss.

        Args:
            logits: Model output (batch_size, num_classes) - raw scores
            targets: Ground truth labels (batch_size,) - class indices

        Returns:
            Loss value (scalar if reduction='mean'/'sum', else (batch_size,))
        """
        # Warmup phase: use cross-entropy loss for first N epochs
        if self.use_warmup and self.current_epoch <= self.warmup_epochs:
            return cast(Tensor, self.ce_loss(logits, targets))

        num_classes = logits.size(-1)

        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            targets_one_hot = F.one_hot(targets, num_classes).float()
            targets_smooth = (
                targets_one_hot * (1 - self.label_smoothing) + self.label_smoothing / num_classes
            )
        else:
            targets_smooth = None

        # Compute softmax probabilities
        probs = F.softmax(logits, dim=-1)

        # Get probability of true class
        targets_expanded = targets.unsqueeze(-1)
        p_t = probs.gather(-1, targets_expanded).squeeze(-1)

        # Compute focal weight: (1 - p_t)^gamma
        # Clamp p_t to avoid numerical issues
        p_t = p_t.clamp(min=1e-7, max=1 - 1e-7)
        focal_weight = (1 - p_t).pow(self.gamma)

        # Compute cross-entropy
        if targets_smooth is not None:
            # Soft cross-entropy for label smoothing
            log_probs = F.log_softmax(logits, dim=-1)
            ce_loss = -(targets_smooth * log_probs).sum(dim=-1)
        else:
            ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # Apply focal weighting
        focal_loss = focal_weight * ce_loss

        # Apply class balance weights (alpha)
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            if alpha.size(0) < num_classes:
                # Expand alpha if needed
                alpha = alpha.expand(num_classes)
            alpha_t = alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        # Apply threat severity weights
        threat_weights_tensor = cast(Tensor, self.threat_weights)
        threat_w = threat_weights_tensor.to(logits.device)
        if threat_w.size(0) >= num_classes:
            threat_w_t = threat_w[:num_classes].gather(0, targets)
            focal_loss = threat_w_t * focal_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"gamma={self.gamma}, "
            f"reduction='{self.reduction}', "
            f"label_smoothing={self.label_smoothing}, "
            f"warmup_epochs={self.warmup_epochs})"
        )


class CalibrationLoss(nn.Module):
    """
    Calibration loss to improve confidence estimation.

    Trains the model to output well-calibrated confidence scores by
    minimizing the binary cross-entropy between predicted confidence
    and actual correctness.

    L_cal = BCE(confidence, correctness)

    Where:
    - confidence: max(softmax(logits)) - model's confidence
    - correctness: 1 if prediction correct, 0 otherwise

    This helps the model learn when it's likely to be wrong, which is
    critical for security applications where uncertain predictions
    should trigger alerts.

    Args:
        temperature: Temperature for confidence scaling (default=1.0)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        temperature: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Compute calibration loss.

        Args:
            logits: Model output (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)

        Returns:
            Calibration loss value
        """
        # Apply temperature scaling
        scaled_logits = logits / self.temperature

        # Compute confidence (max probability)
        probs = F.softmax(scaled_logits, dim=-1)
        confidence = probs.max(dim=-1).values

        # Compute correctness (1 if prediction matches target)
        predictions = logits.argmax(dim=-1)
        correctness = (predictions == targets).float()

        # Binary cross-entropy between confidence and correctness
        # Clamp confidence to avoid log(0)
        confidence = confidence.clamp(min=1e-7, max=1 - 1e-7)

        bce_loss = -(
            correctness * torch.log(confidence) + (1 - correctness) * torch.log(1 - confidence)
        )

        if self.reduction == "mean":
            return bce_loss.mean()
        elif self.reduction == "sum":
            return bce_loss.sum()
        else:
            return bce_loss


class MultiTaskLoss(nn.Module):
    """
    Multi-Task Loss with curriculum learning for hierarchical classification.

    Combines four loss components with evolving weights:
    L = α·L_binary + β·L_family + γ·L_fine + δ·L_calibration

    Curriculum schedule (weights evolve during training):
    - Epochs 1-10:   α=1.0, β=0.0, γ=0.0, δ=0.0 (binary only)
    - Epochs 11-30:  α=0.5, β=0.5, γ=0.0, δ=0.0 (add family)
    - Epochs 31-50:  α=0.3, β=0.4, γ=0.3, δ=0.0 (add fine-grained)
    - Epochs 51+:    α=0.2, β=0.3, γ=0.3, δ=0.2 (add calibration)

    This curriculum helps the model first learn easy distinctions
    (normal vs attack) before tackling harder ones (specific attack types).

    Args:
        num_binary_classes: Classes for binary task (default=2: Normal/Attack)
        num_family_classes: Classes for family task (default=5: Normal/DoS/Probe/R2L/U2R)
        num_fine_classes: Classes for fine-grained task (specific attack types)
        focal_gamma: Gamma parameter for focal loss
        threat_weights: Threat severity weights
        label_smoothing: Label smoothing factor

    Example:
        >>> loss_fn = MultiTaskLoss(num_fine_classes=23)
        >>> loss_fn.set_epoch(25)  # Update curriculum weights
        >>> # outputs: dict with 'binary', 'family', 'fine' logits
        >>> # targets: dict with corresponding labels
        >>> loss = loss_fn(outputs, targets)
    """

    # Curriculum learning schedule: (epoch_start, alpha, beta, gamma, delta)
    CURRICULUM_SCHEDULE = [
        (1, 1.0, 0.0, 0.0, 0.0),  # Binary only
        (11, 0.5, 0.5, 0.0, 0.0),  # Add family
        (31, 0.3, 0.4, 0.3, 0.0),  # Add fine-grained
        (51, 0.2, 0.3, 0.3, 0.2),  # Add calibration
    ]

    def __init__(
        self,
        num_binary_classes: int = 2,
        num_family_classes: int = 5,
        num_fine_classes: int = 23,
        focal_gamma: float = 1.5,
        threat_weights: Optional[Tensor] = None,
        label_smoothing: float = 0.1,
    ):
        super().__init__()

        self.num_binary_classes = num_binary_classes
        self.num_family_classes = num_family_classes
        self.num_fine_classes = num_fine_classes

        # Initialize loss functions for each task
        # Binary: Normal vs Attack (simple threat weights)
        binary_threat = torch.tensor([1.0, 5.0])  # Attacks more important
        self.binary_loss = ThreatAwareFocalLoss(
            gamma=focal_gamma,
            threat_weights=binary_threat,
            label_smoothing=label_smoothing,
            use_warmup=True,
            warmup_epochs=10,
        )

        # Family: 5-class attack family classification
        self.family_loss = ThreatAwareFocalLoss(
            gamma=focal_gamma,
            threat_weights=threat_weights,
            label_smoothing=label_smoothing,
            use_warmup=True,
            warmup_epochs=10,
        )

        # Fine-grained: Specific attack type classification
        # Create expanded threat weights for fine-grained classes
        if threat_weights is None:
            threat_weights = DEFAULT_THREAT_WEIGHTS.clone()
        fine_threat = self._expand_threat_weights(threat_weights, num_fine_classes)
        self.fine_loss = ThreatAwareFocalLoss(
            gamma=focal_gamma,
            threat_weights=fine_threat,
            label_smoothing=label_smoothing,
            use_warmup=True,
            warmup_epochs=10,
        )

        # Calibration loss for confidence estimation
        self.calibration_loss = CalibrationLoss()

        # Current curriculum weights
        self.current_epoch = 1
        self._update_weights()

    def _expand_threat_weights(
        self,
        family_weights: Tensor,
        num_fine_classes: int,
    ) -> Tensor:
        """
        Expand family-level threat weights to fine-grained classes.

        Maps fine-grained attack types to their family threat weights.
        This is a simplified mapping; in practice, you'd use a proper
        mapping from fine-grained class index to family.
        """
        # Default: inherit from family, with slight variation
        # In practice, this would be a proper mapping
        fine_weights = torch.ones(num_fine_classes)

        # Assign weights based on expected class distribution
        # Class 0: Normal
        fine_weights[0] = family_weights[0]  # Normal

        # Classes 1-10: DoS variants (approximate)
        if num_fine_classes > 1:
            fine_weights[1 : min(11, num_fine_classes)] = family_weights[1]

        # Classes 11-16: Probe variants
        if num_fine_classes > 11:
            fine_weights[11 : min(17, num_fine_classes)] = family_weights[2]

        # Classes 17-20: R2L variants
        if num_fine_classes > 17:
            fine_weights[17 : min(21, num_fine_classes)] = family_weights[3]

        # Classes 21+: U2R variants
        if num_fine_classes > 21:
            fine_weights[21:] = family_weights[4]

        return fine_weights

    def _update_weights(self) -> None:
        """Update curriculum weights based on current epoch."""
        # Find the appropriate schedule entry
        weights = self.CURRICULUM_SCHEDULE[0][1:]  # Default to first

        for epoch_start, a, b, g, d in self.CURRICULUM_SCHEDULE:
            if self.current_epoch >= epoch_start:
                weights = (a, b, g, d)

        self.alpha, self.beta, self.gamma_weight, self.delta = weights

    def set_epoch(self, epoch: int) -> None:
        """
        Set current epoch and update curriculum weights.

        Args:
            epoch: Current training epoch (1-indexed)
        """
        self.current_epoch = epoch
        self._update_weights()
        # Propagate epoch to loss components for warmup tracking
        self.binary_loss.set_epoch(epoch)
        self.family_loss.set_epoch(epoch)
        self.fine_loss.set_epoch(epoch)

    def get_curriculum_weights(self) -> dict[str, float]:
        """Get current curriculum weights."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma_weight,
            "delta": self.delta,
            "epoch": self.current_epoch,
        }

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Compute multi-task loss with curriculum weighting.

        Args:
            outputs: Dict with keys 'binary', 'family', 'fine' containing logits
            targets: Dict with keys 'binary', 'family', 'fine' containing labels

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual
            losses for logging
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=next(iter(outputs.values())).device)

        # Binary loss (Normal vs Attack)
        if "binary" in outputs and self.alpha > 0:
            loss_binary = self.binary_loss(outputs["binary"], targets["binary"])
            loss_dict["loss_binary"] = loss_binary
            total_loss = total_loss + self.alpha * loss_binary

        # Family loss (5-class: Normal, DoS, Probe, R2L, U2R)
        if "family" in outputs and self.beta > 0:
            loss_family = self.family_loss(outputs["family"], targets["family"])
            loss_dict["loss_family"] = loss_family
            total_loss = total_loss + self.beta * loss_family

        # Fine-grained loss (specific attack types)
        if "fine" in outputs and self.gamma_weight > 0:
            loss_fine = self.fine_loss(outputs["fine"], targets["fine"])
            loss_dict["loss_fine"] = loss_fine
            total_loss = total_loss + self.gamma_weight * loss_fine

        # Calibration loss (confidence estimation)
        if "fine" in outputs and self.delta > 0:
            # Use fine-grained logits for calibration
            loss_cal = self.calibration_loss(outputs["fine"], targets["fine"])
            loss_dict["loss_calibration"] = loss_cal
            total_loss = total_loss + self.delta * loss_cal

        loss_dict["loss_total"] = total_loss

        return total_loss, loss_dict

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"binary={self.num_binary_classes}, "
            f"family={self.num_family_classes}, "
            f"fine={self.num_fine_classes}, "
            f"epoch={self.current_epoch}, "
            f"weights=(α={self.alpha}, β={self.beta}, "
            f"γ={self.gamma_weight}, δ={self.delta}))"
        )


class FocalLoss(nn.Module):
    """
    Standard Focal Loss without threat weighting.

    Simpler version for general use cases.
    L = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        gamma: Focusing parameter (default=2.0)
        alpha: Class balancing weights (optional)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute focal loss."""
        probs = F.softmax(logits, dim=-1)
        p_t = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        focal_weight = (1 - p_t).pow(self.gamma)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        focal_loss = focal_weight * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def create_loss_function(
    loss_type: str = "threat_focal",
    num_classes: int = 5,
    class_weights: Optional[Tensor] = None,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create appropriate loss function.

    Args:
        loss_type: One of 'ce', 'focal', 'threat_focal', 'multitask'
        num_classes: Number of classes
        class_weights: Pre-computed class weights (optional)
        **kwargs: Additional arguments passed to loss constructor

    Returns:
        Loss function module
    """
    if loss_type == "ce":
        # Standard cross-entropy
        return nn.CrossEntropyLoss(weight=class_weights)

    elif loss_type == "focal":
        # Standard focal loss
        return FocalLoss(alpha=class_weights, **kwargs)

    elif loss_type == "threat_focal":
        # Threat-aware focal loss
        return ThreatAwareFocalLoss(alpha=class_weights, **kwargs)

    elif loss_type == "multitask":
        # Multi-task loss with curriculum
        return MultiTaskLoss(num_fine_classes=num_classes, **kwargs)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
