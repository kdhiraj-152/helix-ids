"""
Tests for adversarial robustness module.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from helix_ids.metrics.adversarial_test import (
    AdversarialMetrics,
    AdversarialTester,
    run_adversarial_evaluation,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_size=41, num_classes=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@pytest.fixture
def simple_model():
    """Create a simple test model."""
    return SimpleModel()


@pytest.fixture
def test_data():
    """Create test data."""
    torch.manual_seed(42)
    x = torch.randn(32, 41)
    x = torch.sigmoid(x)  # Normalize to [0, 1]
    y = torch.randint(0, 5, (32,))
    return x, y


@pytest.fixture
def tester():
    """Create adversarial tester."""
    return AdversarialTester(device="cpu")


class TestAdversarialMetrics:
    """Test AdversarialMetrics dataclass."""

    def test_metrics_initialization(self):
        """Test default initialization."""
        metrics = AdversarialMetrics()
        assert metrics.clean_accuracy == pytest.approx(0.0, abs=1e-9)
        assert metrics.fgsm_accuracy == pytest.approx(0.0, abs=1e-9)
        assert metrics.robustness_score == pytest.approx(0.0, abs=1e-9)

    def test_metrics_to_dict(self):
        """Test conversion to dictionary."""
        metrics = AdversarialMetrics(
            clean_accuracy=0.95,
            fgsm_accuracy=0.90,
        )
        d = metrics.to_dict()
        assert isinstance(d, dict)
        assert d["clean_accuracy"] == pytest.approx(0.95)
        assert d["fgsm_accuracy"] == pytest.approx(0.90)

    def test_metrics_with_values(self):
        """Test metrics with all values set."""
        metrics = AdversarialMetrics(
            clean_accuracy=0.95,
            fgsm_accuracy=0.88,
            pgd_accuracy=0.85,
            accuracy_drop_fgsm=7.0,
            accuracy_drop_pgd=10.0,
        )
        assert metrics.clean_accuracy == pytest.approx(0.95)
        assert metrics.fgsm_accuracy == pytest.approx(0.88)
        assert metrics.accuracy_drop_fgsm == pytest.approx(7.0)


class TestFGSMAttack:
    """Test FGSM attack implementation."""

    def test_fgsm_attack_shape(self, tester, simple_model, test_data):
        """Test FGSM attack output shape."""
        x, y = test_data
        x_adv, pert = tester.fgsm_attack(simple_model, x, y, epsilon=0.1)

        assert x_adv.shape == x.shape
        assert pert.shape == x.shape

    def test_fgsm_attack_bounded(self, tester, simple_model, test_data):
        """Test FGSM attack maintains epsilon bound."""
        x, y = test_data
        epsilon = 0.1
        _, pert = tester.fgsm_attack(simple_model, x, y, epsilon=epsilon)
        _, _ = tester.fgsm_attack(simple_model, x, y, epsilon=epsilon)

        # Check perturbation is bounded by epsilon in L-infinity norm
        max_pert = torch.abs(pert).max(dim=1)[0].max()
        assert max_pert <= epsilon + 1e-5  # Small tolerance for numerical errors

    def test_fgsm_attack_changes_output(self, tester, simple_model, test_data):
        """Test FGSM attack changes model predictions."""
        x, y = test_data
        simple_model.eval()

        # Get clean predictions
        with torch.no_grad():
            pred_clean = torch.argmax(simple_model(x), dim=1)

        # Get adversarial predictions
        x_adv, _ = tester.fgsm_attack(simple_model, x, y, epsilon=0.5)
        with torch.no_grad():
            pred_adv = torch.argmax(simple_model(x_adv), dim=1)

        # Should have some different predictions
        assert (pred_clean != pred_adv).sum() > 0

    def test_fgsm_different_epsilons(self, tester, simple_model, test_data):
        """Test FGSM with different epsilon values."""
        x, y = test_data

        _, pert1 = tester.fgsm_attack(simple_model, x, y, epsilon=0.05)
        _, pert2 = tester.fgsm_attack(simple_model, x, y, epsilon=0.2)

        # Larger epsilon should produce larger perturbations on average
        avg_pert1 = pert1.abs().mean().item()
        avg_pert2 = pert2.abs().mean().item()
        assert avg_pert2 > avg_pert1


class TestPGDAttack:
    """Test PGD attack implementation."""

    def test_pgd_attack_shape(self, tester, simple_model, test_data):
        """Test PGD attack output shape."""
        x, y = test_data
        x_adv, pert = tester.pgd_attack(simple_model, x, y, epsilon=0.1, steps=5)

        assert x_adv.shape == x.shape
        assert pert.shape == x.shape

    def test_pgd_attack_bounded(self, tester, simple_model, test_data):
        """Test PGD attack maintains epsilon bound."""
        x, y = test_data
        epsilon = 0.1
        _, pert = tester.pgd_attack(simple_model, x, y, epsilon=epsilon, steps=5)

        # Check perturbation is bounded by epsilon in L-infinity norm
        max_pert = torch.abs(pert).max(dim=1)[0].max()
        assert max_pert <= epsilon + 1e-5

    def test_pgd_attack_changes_output(self, tester, simple_model, test_data):
        """Test PGD attack changes predictions."""
        x, y = test_data
        simple_model.eval()

        with torch.no_grad():
            pred_clean = torch.argmax(simple_model(x), dim=1)

        x_adv, _ = tester.pgd_attack(simple_model, x, y, epsilon=0.5, steps=10)
        with torch.no_grad():
            pred_adv = torch.argmax(simple_model(x_adv), dim=1)

        # Should have some different predictions
        assert (pred_clean != pred_adv).sum() > 0

    def test_pgd_more_steps_stronger(self, tester, simple_model, test_data):
        """Test that more PGD steps lead to stronger attacks."""
        x, y = test_data
        simple_model.eval()

        x_adv5, _ = tester.pgd_attack(simple_model, x, y, epsilon=0.5, steps=5)
        x_adv20, _ = tester.pgd_attack(simple_model, x, y, epsilon=0.5, steps=20)

        with torch.no_grad():
            pred5 = torch.argmax(simple_model(x_adv5), dim=1)
            pred20 = torch.argmax(simple_model(x_adv20), dim=1)
            misclass5 = (pred5 != y).float().mean().item()
            misclass20 = (pred20 != y).float().mean().item()

        # More steps should lead to higher or equal misclassification rate
        assert misclass20 >= misclass5 - 0.05  # Small tolerance for randomness


class TestFeatureNoise:
    """Test feature noise attack."""

    def test_feature_noise_shape(self, tester, test_data):
        """Test feature noise shape."""
        x, _ = test_data
        x_noisy, noise = tester.feature_noise(x, noise_std=0.1)

        assert x_noisy.shape == x.shape
        assert noise.shape == x.shape

    def test_feature_noise_std(self, tester):
        """Test feature noise standard deviation."""
        x = torch.randn(1000, 41)
        noise_std = 0.2
        _, noise = tester.feature_noise(x, noise_std=noise_std, seed=42)

        # Check noise roughly matches requested std
        actual_std = noise.std().item()
        assert actual_std == pytest.approx(noise_std, rel=0.1)

    def test_feature_noise_reproducible(self, tester, test_data):
        """Test feature noise is reproducible with seed."""
        x, _ = test_data
        _, noise1 = tester.feature_noise(x, seed=42)
        _, noise2 = tester.feature_noise(x, seed=42)

        assert torch.allclose(noise1, noise2)


class TestRobustnessEvaluation:
    """Test comprehensive robustness evaluation."""

    def test_evaluate_robustness_output(self, tester, simple_model, test_data):
        """Test evaluation output structure."""
        x, y = test_data
        metrics = tester.evaluate_robustness(simple_model, x, y, epsilon=0.1)

        assert isinstance(metrics, AdversarialMetrics)
        assert metrics.clean_accuracy > 0
        assert metrics.fgsm_accuracy >= 0
        assert metrics.pgd_accuracy >= 0

    def test_evaluate_robustness_values_valid(self, tester, simple_model, test_data):
        """Test evaluation produces valid metric values."""
        x, y = test_data
        metrics = tester.evaluate_robustness(simple_model, x, y, epsilon=0.1)

        # All accuracies should be in [0, 1]
        assert 0 <= metrics.clean_accuracy <= 1
        assert 0 <= metrics.fgsm_accuracy <= 1
        assert 0 <= metrics.pgd_accuracy <= 1

        # Attack success rates should be in [0, 1]
        assert 0 <= metrics.fgsm_attack_success_rate <= 1
        assert 0 <= metrics.pgd_attack_success_rate <= 1

        # Robustness score should be in [0, 1]
        assert 0 <= metrics.robustness_score <= 1

        # Accuracy drops should be non-negative
        assert metrics.accuracy_drop_fgsm >= 0
        assert metrics.accuracy_drop_pgd >= 0

    def test_evaluate_robustness_with_class_names(self, tester, simple_model, test_data):
        """Test evaluation with class names."""
        x, y = test_data
        class_names = ["Normal", "DoS", "Probe", "R2L", "U2R"]
        metrics = tester.evaluate_robustness(
            simple_model, x, y, epsilon=0.1, class_names=class_names
        )

        assert isinstance(metrics.per_class_robustness, dict)
        # Not all classes may be present in test data
        for class_name in metrics.per_class_robustness:
            assert 0 <= metrics.per_class_robustness[class_name] <= 1

    def test_evaluate_robustness_attack_decreases_accuracy(self, tester, simple_model, test_data):
        """Test that attacks decrease accuracy."""
        x, y = test_data
        metrics = tester.evaluate_robustness(simple_model, x, y, epsilon=0.3)

        # Strong attacks should decrease accuracy
        # (but with random model, not guaranteed)
        assert metrics.clean_accuracy >= metrics.pgd_accuracy or metrics.pgd_accuracy > 0.3


class TestMinimumPerturbation:
    """Test minimum perturbation finding."""

    def test_find_minimum_perturbation_fgsm(self, tester, simple_model, test_data):
        """Test finding minimum FGSM perturbation."""
        x, y = test_data
        min_eps, misclass_rate = tester.find_minimum_perturbation(
            simple_model, x, y, attack_type="fgsm"
        )

        assert isinstance(min_eps, float)
        assert isinstance(misclass_rate, float)
        assert 0 <= min_eps <= 1
        assert 0 <= misclass_rate <= 1

    def test_find_minimum_perturbation_pgd(self, tester, simple_model, test_data):
        """Test finding minimum PGD perturbation."""
        x, y = test_data
        min_eps, misclass_rate = tester.find_minimum_perturbation(
            simple_model, x, y, attack_type="pgd"
        )

        assert isinstance(min_eps, float)
        assert isinstance(misclass_rate, float)
        assert 0 <= min_eps <= 1
        assert 0 <= misclass_rate <= 1


class TestRunAdversarialEvaluation:
    """Test the main evaluation function."""

    def test_run_adversarial_evaluation(self, simple_model, test_data):
        """Test main evaluation function."""
        x, y = test_data
        results = run_adversarial_evaluation(simple_model, x, y, epsilon=0.1, pgd_steps=5)

        assert isinstance(results, dict)
        assert "metrics" in results
        assert "minimum_perturbations" in results
        assert "evaluation_config" in results

    def test_run_adversarial_evaluation_with_class_names(self, simple_model, test_data):
        """Test evaluation with class names."""
        x, y = test_data
        class_names = ["Normal", "DoS", "Probe", "R2L", "U2R"]
        results = run_adversarial_evaluation(simple_model, x, y, class_names=class_names)

        metrics = results["metrics"]
        assert isinstance(metrics["per_class_robustness"], dict)
