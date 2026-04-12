"""
Adversarial Robustness Testing for HELIX-IDS.

Implements attack methods and robustness metrics:
- FGSM (Fast Gradient Sign Method)
- PGD (Projected Gradient Descent)
- Feature perturbation attacks
- Robustness metrics: accuracy under attack, attack success rate, minimum perturbation
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AdversarialMetrics:
    """Container for adversarial robustness evaluation metrics."""

    # Attack performance
    clean_accuracy: float = 0.0
    fgsm_accuracy: float = 0.0
    pgd_accuracy: float = 0.0
    feature_noise_accuracy: float = 0.0

    # Attack success rates
    fgsm_attack_success_rate: float = 0.0
    pgd_attack_success_rate: float = 0.0

    # Perturbation metrics
    fgsm_avg_perturbation: float = 0.0
    pgd_avg_perturbation: float = 0.0
    min_perturbation_fgsm: float = 0.0
    min_perturbation_pgd: float = 0.0

    # Robustness scores
    accuracy_drop_fgsm: float = 0.0
    accuracy_drop_pgd: float = 0.0
    robustness_score: float = 0.0

    # Per-class metrics
    per_class_robustness: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "clean_accuracy": self.clean_accuracy,
            "fgsm_accuracy": self.fgsm_accuracy,
            "pgd_accuracy": self.pgd_accuracy,
            "feature_noise_accuracy": self.feature_noise_accuracy,
            "fgsm_attack_success_rate": self.fgsm_attack_success_rate,
            "pgd_attack_success_rate": self.pgd_attack_success_rate,
            "fgsm_avg_perturbation": self.fgsm_avg_perturbation,
            "pgd_avg_perturbation": self.pgd_avg_perturbation,
            "min_perturbation_fgsm": self.min_perturbation_fgsm,
            "min_perturbation_pgd": self.min_perturbation_pgd,
            "accuracy_drop_fgsm": self.accuracy_drop_fgsm,
            "accuracy_drop_pgd": self.accuracy_drop_pgd,
            "robustness_score": self.robustness_score,
            "per_class_robustness": self.per_class_robustness,
        }


class AdversarialTester:
    """Adversarial robustness testing for neural network models.

    Implements multiple attack methods and metrics to evaluate model robustness:
    - FGSM: Single-step gradient attack
    - PGD: Multi-step iterative attack
    - Feature perturbation: Gaussian noise injection
    """

    def __init__(self, device: Optional[str] = None):
        """Initialize the adversarial tester.

        Args:
            device: Device to use for computation ('cuda' or 'cpu').
                   Auto-detects if None.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def fgsm_attack(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        epsilon: float = 0.1,
        _target_model_output: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate FGSM adversarial examples.

        Fast Gradient Sign Method (FGSM) performs a single gradient step
        in the direction of loss maximization, bounded by epsilon.

        Args:
            model: Target model to attack.
            x: Input features of shape (batch_size, num_features).
            y: Target labels.
            epsilon: Maximum perturbation magnitude (L-infinity norm).
            target_model_output: If set, uses 'logits' or 'probs' mode.

        Returns:
            Tuple of (adversarial_examples, perturbations)
        """
        model.eval()
        x_adv = x.clone().detach().requires_grad_(True)

        # Forward pass
        output = model(x_adv)

        # Handle different output types
        if isinstance(output, dict):
            logits = output.get("logits", output.get("output", output))
        else:
            logits = output

        # Ensure logits are 2D
        if logits.dim() == 1:
            logits = logits.unsqueeze(1)

        # Compute loss
        loss = F.cross_entropy(logits, y)

        # Compute gradient
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        loss.backward()

        # Apply FGSM perturbation
        with torch.no_grad():
            grad = x_adv.grad if x_adv.grad is not None else torch.zeros_like(x_adv)
            perturbation = epsilon * torch.sign(grad)
            x_adv_result = x + perturbation

            # Clamp to valid range [0, 1] if normalized
            x_adv_result = torch.clamp(x_adv_result, 0, 1)

        return x_adv_result, perturbation

    def pgd_attack(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        epsilon: float = 0.1,
        steps: int = 10,
        step_size: Optional[float] = None,
        random_start: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate PGD adversarial examples.

        Projected Gradient Descent (PGD) performs multiple iterative gradient steps
        with projection to maintain the epsilon-ball constraint.

        Args:
            model: Target model to attack.
            x: Input features of shape (batch_size, num_features).
            y: Target labels.
            epsilon: Maximum perturbation magnitude (L-infinity norm).
            steps: Number of attack iterations.
            step_size: Step size per iteration. Defaults to epsilon/steps.
            random_start: Initialize with random perturbation if True.

        Returns:
            Tuple of (adversarial_examples, perturbations)
        """
        if step_size is None:
            step_size = epsilon / steps

        model.eval()
        x_adv = x.clone().detach()

        # Random initialization
        if random_start:
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)

        for _ in range(steps):
            x_adv.requires_grad_(True)

            # Forward pass
            output = model(x_adv)

            # Handle different output types
            if isinstance(output, dict):
                logits = output.get("logits", output.get("output", output))
            else:
                logits = output

            # Ensure logits are 2D
            if logits.dim() == 1:
                logits = logits.unsqueeze(1)

            # Compute loss
            loss = F.cross_entropy(logits, y)

            # Compute gradient
            if x_adv.grad is not None:
                x_adv.grad.zero_()
            loss.backward()

            # Apply PGD step
            with torch.no_grad():
                grad = x_adv.grad if x_adv.grad is not None else torch.zeros_like(x_adv)
                perturbation = step_size * torch.sign(grad)
                x_adv = x_adv + perturbation

                # Project onto epsilon-ball
                perturbation_clipped = torch.clamp(x_adv - x, -epsilon, epsilon)
                x_adv = x + perturbation_clipped

                # Clamp to valid range
                x_adv = torch.clamp(x_adv, 0, 1)

            x_adv.detach_()

        perturbation = x_adv - x
        return x_adv, perturbation

    def feature_noise(
        self,
        x: torch.Tensor,
        noise_std: float = 0.1,
        seed: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add Gaussian noise to features.

        Args:
            x: Input features.
            noise_std: Standard deviation of Gaussian noise.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (noisy_features, noise)
        """
        if seed is not None:
            torch.manual_seed(seed)

        noise = torch.randn_like(x) * noise_std
        x_noisy = x + noise
        x_noisy = torch.clamp(x_noisy, 0, 1)

        return x_noisy, noise

    def evaluate_robustness(  # NOSONAR
        self,
        model: nn.Module,
        x_clean: torch.Tensor,
        y_true: torch.Tensor,
        epsilon: float = 0.1,
        pgd_steps: int = 10,
        noise_std: float = 0.1,
        class_names: Optional[list[str]] = None,
    ) -> AdversarialMetrics:
        """Evaluate model robustness against multiple attacks.

        Args:
            model: Model to evaluate.
            x_clean: Clean input features.
            y_true: True labels.
            epsilon: Maximum perturbation magnitude.
            pgd_steps: Number of PGD iterations.
            noise_std: Standard deviation for feature noise.
            class_names: Names of classes for per-class metrics.

        Returns:
            AdversarialMetrics object with comprehensive results.
        """
        model.eval()
        x_clean = x_clean.to(self.device)
        y_true = y_true.to(self.device)
        model = model.to(self.device)

        metrics = AdversarialMetrics()

        # Evaluate clean accuracy
        with torch.no_grad():
            output = model(x_clean)
            if isinstance(output, dict):
                logits = output.get("logits", output.get("output", output))
            else:
                logits = output

            if logits.dim() == 1:
                logits = logits.unsqueeze(1)

            pred_clean = torch.argmax(logits, dim=1)
            clean_correct = (pred_clean == y_true).float()
            metrics.clean_accuracy = clean_correct.mean().item()

        # FGSM attack
        x_fgsm, pert_fgsm = self.fgsm_attack(model, x_clean, y_true, epsilon)
        with torch.no_grad():
            output_fgsm = model(x_fgsm)
            if isinstance(output_fgsm, dict):
                logits_fgsm = output_fgsm.get("logits", output_fgsm.get("output", output_fgsm))
            else:
                logits_fgsm = output_fgsm

            if logits_fgsm.dim() == 1:
                logits_fgsm = logits_fgsm.unsqueeze(1)

            pred_fgsm = torch.argmax(logits_fgsm, dim=1)
            fgsm_correct = (pred_fgsm == y_true).float()
            metrics.fgsm_accuracy = fgsm_correct.mean().item()
            metrics.fgsm_attack_success_rate = (1 - fgsm_correct).mean().item()

        # PGD attack
        x_pgd, pert_pgd = self.pgd_attack(model, x_clean, y_true, epsilon=epsilon, steps=pgd_steps)
        with torch.no_grad():
            output_pgd = model(x_pgd)
            if isinstance(output_pgd, dict):
                logits_pgd = output_pgd.get("logits", output_pgd.get("output", output_pgd))
            else:
                logits_pgd = output_pgd

            if logits_pgd.dim() == 1:
                logits_pgd = logits_pgd.unsqueeze(1)

            pred_pgd = torch.argmax(logits_pgd, dim=1)
            pgd_correct = (pred_pgd == y_true).float()
            metrics.pgd_accuracy = pgd_correct.mean().item()
            metrics.pgd_attack_success_rate = (1 - pgd_correct).mean().item()

        # Feature noise attack
        x_noise, _ = self.feature_noise(x_clean, noise_std=noise_std)
        with torch.no_grad():
            output_noise = model(x_noise)
            if isinstance(output_noise, dict):
                logits_noise = output_noise.get("logits", output_noise.get("output", output_noise))
            else:
                logits_noise = output_noise

            if logits_noise.dim() == 1:
                logits_noise = logits_noise.unsqueeze(1)

            pred_noise = torch.argmax(logits_noise, dim=1)
            noise_correct = (pred_noise == y_true).float()
            metrics.feature_noise_accuracy = noise_correct.mean().item()

        # Perturbation statistics
        metrics.fgsm_avg_perturbation = (pert_fgsm.abs().sum(dim=1).mean()).item()
        metrics.pgd_avg_perturbation = (pert_pgd.abs().sum(dim=1).mean()).item()
        metrics.min_perturbation_fgsm = pert_fgsm.abs().sum(dim=1).min().item()
        metrics.min_perturbation_pgd = pert_pgd.abs().sum(dim=1).min().item()

        # Accuracy drops
        metrics.accuracy_drop_fgsm = (metrics.clean_accuracy - metrics.fgsm_accuracy) * 100
        metrics.accuracy_drop_pgd = (metrics.clean_accuracy - metrics.pgd_accuracy) * 100

        # Overall robustness score (normalized inverse of accuracy drop)
        avg_accuracy_drop = (metrics.accuracy_drop_fgsm + metrics.accuracy_drop_pgd) / 2
        metrics.robustness_score = max(0, 100 - avg_accuracy_drop) / 100

        # Per-class robustness
        if class_names:
            for class_idx, class_name in enumerate(class_names):
                class_mask = y_true == class_idx
                if class_mask.sum() > 0:
                    class_clean_acc = clean_correct[class_mask].mean().item()
                    class_pgd_acc = pgd_correct[class_mask].mean().item()
                    class_robustness = max(0, class_clean_acc - class_pgd_acc)
                    metrics.per_class_robustness[class_name] = class_robustness

        return metrics

    def find_minimum_perturbation(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        attack_type: str = "fgsm",
        epsilon_range: tuple[float, float] = (0.0, 1.0),
        num_steps: int = 20,
    ) -> tuple[float, float]:
        """Find minimum perturbation required for misclassification.

        Performs binary search over epsilon to find the minimum perturbation
        needed to achieve a target misclassification rate.

        Args:
            model: Target model.
            x: Input features.
            y: True labels.
            attack_type: Type of attack ('fgsm' or 'pgd').
            epsilon_range: Range of epsilon values to search.
            num_steps: Number of binary search steps.

        Returns:
            Tuple of (minimum_epsilon, misclassification_rate)
        """
        model.eval()
        x = x.to(self.device)
        y = y.to(self.device)
        model = model.to(self.device)

        low, high = epsilon_range
        best_epsilon = high
        best_misclass_rate = 0.0

        for _ in range(num_steps):
            epsilon = (low + high) / 2

            if attack_type == "fgsm":
                x_adv, _ = self.fgsm_attack(model, x, y, epsilon)
            else:  # pgd
                x_adv, _ = self.pgd_attack(model, x, y, epsilon)

            with torch.no_grad():
                output = model(x_adv)
                if isinstance(output, dict):
                    logits = output.get("logits", output.get("output", output))
                else:
                    logits = output

                if logits.dim() == 1:
                    logits = logits.unsqueeze(1)

                pred = torch.argmax(logits, dim=1)
                misclass_rate = (pred != y).float().mean().item()

            if misclass_rate >= 0.5:  # Target: at least 50% misclassification
                best_epsilon = epsilon
                best_misclass_rate = misclass_rate
                high = epsilon
            else:
                low = epsilon

        return best_epsilon, best_misclass_rate


def run_adversarial_evaluation(
    model: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    epsilon: float = 0.1,
    pgd_steps: int = 10,
    class_names: Optional[list[str]] = None,
) -> dict:
    """Run comprehensive adversarial robustness evaluation.

    Args:
        model: Model to evaluate.
        x_test: Test features.
        y_test: Test labels.
        epsilon: FGSM epsilon value.
        pgd_steps: Number of PGD steps.
        class_names: Class name mapping.

    Returns:
        Dictionary with evaluation results.
    """
    tester = AdversarialTester()

    metrics = tester.evaluate_robustness(
        model,
        x_test,
        y_test,
        epsilon=epsilon,
        pgd_steps=pgd_steps,
        class_names=class_names,
    )

    # Find minimum perturbations
    min_eps_fgsm, misclass_fgsm = tester.find_minimum_perturbation(
        model, x_test, y_test, attack_type="fgsm"
    )
    min_eps_pgd, misclass_pgd = tester.find_minimum_perturbation(
        model, x_test, y_test, attack_type="pgd"
    )

    results = {
        "metrics": metrics.to_dict(),
        "minimum_perturbations": {
            "fgsm": {
                "epsilon": min_eps_fgsm,
                "misclassification_rate": misclass_fgsm,
            },
            "pgd": {
                "epsilon": min_eps_pgd,
                "misclassification_rate": misclass_pgd,
            },
        },
        "evaluation_config": {
            "epsilon": epsilon,
            "pgd_steps": pgd_steps,
        },
    }

    return results
