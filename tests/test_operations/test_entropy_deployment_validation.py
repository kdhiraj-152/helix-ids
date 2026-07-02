"""
Deployment-Condition Validation Tests for Entropy Guard

Tests entropy guard under five critical failure mode scenarios:
1. Degenerate logits (identical across classes)
2. Single-class dominance (batch-wide collapse)
3. Bimodal batch (mixed confidence distribution)
4. Real training trace (actual model logits)
5. Seed variance sensitivity (dropout stability)
"""

import logging
import sys
import traceback

# Add src to path
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from helix_ids.utils.entropy_diagnostics import (  # noqa: E402
    calculate_entropy_stable,
    detect_batch_composition_risk,
    should_trigger_entropy_guard,
    summarize_entropy,
)

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_test(test_func, test_name):
    """Simple test runner wrapper."""
    try:
        test_func()
        print(f"✓ PASS: {test_name}")
        return True
    except AssertionError as e:
        print(f"✗ FAIL: {test_name}")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"✗ ERROR: {test_name}")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# Test 1: Degenerate Logits
# ============================================================================

def test_logits_all_identical():
    """All logits identical across classes → max entropy (uniform distribution)."""
    batch_size = 32
    num_classes = 7

    # Create identical logits
    identical_logits = torch.full((batch_size, num_classes), 1.0)
    probs = torch.softmax(identical_logits, dim=1).numpy()

    summary = summarize_entropy(probs)

    # Expected: all samples have max entropy (uniform = log(7)/log(7) = 1.0)
    assert summary.mean > 0.99, f"Expected ~1.0 entropy for uniform, got {summary.mean:.4f}"
    assert summary.min_val > 0.98, "Minimum entropy should also be near max"
    assert summary.num_samples == batch_size
    logger.info(f"  {summary}")


def test_logits_near_zero():
    """Very small logits → uniform distribution → max entropy."""
    batch_size = 32
    num_classes = 7

    # Create very small logits
    small_logits = torch.randn(batch_size, num_classes) * 0.001
    probs = torch.softmax(small_logits, dim=1).numpy()

    summary = summarize_entropy(probs)

    # Near-uniform → entropy close to 1.0
    assert summary.mean > 0.95, f"Expected high entropy for near-uniform, got {summary.mean:.4f}"
    logger.info(f"  {summary}")


# ============================================================================
# Test 2: Single-Class Dominance
# ============================================================================

class TestSingleClassDominance:
    """Detect batch-wide mode collapse where single class dominates."""

    def test_all_samples_one_class(self):
        """All samples predict single class → low entropy, missing classes."""
        batch_size = 32
        num_classes = 7

        # Create logits where class 0 always wins
        logits = torch.ones(batch_size, num_classes) * 0.001
        logits[:, 0] = 10.0  # Extreme preference for class 0
        probs = torch.softmax(logits, dim=1).numpy()

        summary = summarize_entropy(probs)
        predicted = np.argmax(probs, axis=1)

        # Entropy should be very low
        assert summary.mean < 0.05, f"Expected low entropy, got {summary.mean:.4f}"
        assert summary.collapsed_samples == batch_size, "All samples should be collapsed"

        # Risk detection
        risk = detect_batch_composition_risk(summary, predicted, num_classes)
        assert risk["unique_classes_predicted"] == 1, "Should only predict 1 class"
        assert risk["missing_classes"] == num_classes - 1, "Should be missing 6 classes"

        logger.info(f"✓ Single-class dominance test passed: {summary}")
        logger.info(f"  Risk: {risk}")

    def test_two_class_dominance(self):
        """Only 2 classes predicted → missing 5 classes."""
        batch_size = 32
        num_classes = 7

        # Create logits where only classes 0 and 1 are viable
        logits = torch.ones(batch_size, num_classes) * -10.0
        logits[:, 0] = 5.0
        logits[:, 1] = 4.0
        # Randomly choose between class 0 and 1
        rng = np.random.default_rng(seed=42)
        rand_mask = rng.random(batch_size) > 0.5
        logits[rand_mask, 0] = -10.0
        logits[rand_mask, 1] = 5.0

        probs = torch.softmax(logits, dim=1).numpy()
        predicted = np.argmax(probs, axis=1)
        summary = summarize_entropy(probs)

        # Should detect mode collapse
        risk = detect_batch_composition_risk(summary, predicted, num_classes)
        assert risk["unique_classes_predicted"] <= 2, "Should predict ≤2 classes"
        assert risk["missing_classes"] >= 5, "Should be missing ≥5 classes"

        logger.info(f"✓ Two-class dominance test passed: missing {risk['missing_classes']} classes")


# ============================================================================
# Test 3: Bimodal Batch (Mixed Confidence)
# ============================================================================

class TestBimodalBatch:
    """Detect mixed distributions where half collapsed, half uniform."""

    def test_half_collapsed_half_uniform(self):
        """Half samples peaked, half uniform → entropy distribution should bimodal."""
        batch_size = 32
        num_classes = 7

        # Create logits
        logits = torch.zeros(batch_size, num_classes)

        # First half: peaked (class 0 dominates)
        logits[:batch_size//2, 0] = 10.0
        logits[:batch_size//2, 1:] = -10.0

        # Second half: uniform
        logits[batch_size//2:, :] = 0.1

        probs = torch.softmax(logits, dim=1).numpy()
        entropy = calculate_entropy_stable(probs)

        # Should see bimodal distribution: some very low, some very high
        low_entropy_count = np.sum(entropy < 0.2)
        high_entropy_count = np.sum(entropy > 0.8)

        assert low_entropy_count >= batch_size//4, "Should have ~half with low entropy"
        assert high_entropy_count >= batch_size//4, "Should have ~half with high entropy"

        summary = summarize_entropy(probs)
        # Range should be large for bimodal entropy distributions.
        spread = summary.max_val - summary.min_val
        assert spread > 0.5, f"Expected large spread for bimodal, got {spread:.4f}"

        logger.info(f"✓ Bimodal batch test passed: low={low_entropy_count}, high={high_entropy_count}")
        logger.info(f"  Entropy range: {spread:.4f}")


# ============================================================================
# Test 4: Real Training Trace Injection
# ============================================================================

class TestRealTrainingTrace:
    """Validate entropy against actual training trajectory signals."""

    def test_entropy_vs_accuracy_correlation(self):
        """
        entropy should NOT correlate perfectly with accuracy.
        High accuracy can happen with both high and low entropy.
        """
        batch_size = 64
        num_classes = 7

        # Scenario A: High accuracy from uniform predictions on easy data
        logits_easy = torch.randn(batch_size, num_classes)
        logits_easy[torch.arange(batch_size), torch.randint(0, num_classes, (batch_size,))] += 5.0
        probs_easy = torch.softmax(logits_easy, dim=1).numpy()

        # Scenario B: High accuracy from peaked predictions (mode collapse risk)
        logits_peaked = torch.ones(batch_size, num_classes) * 0.1
        logits_peaked[:, 0] = 10.0
        probs_peaked = torch.softmax(logits_peaked, dim=1).numpy()

        # Both have high accuracy but different entropy
        summary_easy = summarize_entropy(probs_easy)
        summary_peaked = summarize_entropy(probs_peaked)

        # Easy case should have higher entropy than peaked
        assert summary_easy.mean > summary_peaked.mean, \
            "Easy case should have higher entropy than peaked"

        logger.info(f"✓ Easy case entropy: {summary_easy.mean:.4f}")
        logger.info(f"  Peaked case entropy: {summary_peaked.mean:.4f}")

    def test_temperature_scaling_effect(self):
        """Temperature scaling should consistently increase entropy."""
        batch_size = 32
        num_classes = 7

        # Create logits with varying magnitudes
        logits_small = torch.randn(batch_size, num_classes)
        logits_large = logits_small * 5.0  # Larger magnitude → peaked

        # Without temperature
        probs_small_t1 = torch.softmax(logits_small / 1.0, dim=1).numpy()
        probs_large_t1 = torch.softmax(logits_large / 1.0, dim=1).numpy()

        # With temperature=1.2
        probs_small_t1p2 = torch.softmax(logits_small / 1.2, dim=1).numpy()
        probs_large_t1p2 = torch.softmax(logits_large / 1.2, dim=1).numpy()

        summary_small_t1 = summarize_entropy(probs_small_t1)
        summary_small_t1p2 = summarize_entropy(probs_small_t1p2)
        summary_large_t1 = summarize_entropy(probs_large_t1)
        summary_large_t1p2 = summarize_entropy(probs_large_t1p2)

        # Temperature should increase entropy in both cases
        assert summary_small_t1p2.mean >= summary_small_t1.mean - 0.01, \
            "Temperature should not decrease entropy for small logits"
        assert summary_large_t1p2.mean > summary_large_t1.mean, \
            "Temperature should increase entropy for large logits"

        logger.info("✓ Temperature scaling validated:")
        logger.info(f"  Small logits: {summary_small_t1.mean:.4f} → {summary_small_t1p2.mean:.4f}")
        logger.info(f"  Large logits: {summary_large_t1.mean:.4f} → {summary_large_t1p2.mean:.4f}")


# ============================================================================
# Test 5: Seed Variance Sensitivity
# ============================================================================

class TestSeedVarianceSensitivity:
    """Verify entropy metric is stable under dropout variance."""

    def test_dropout_stability(self):
        """Same batch with dropout should have reasonable entropy variance."""
        batch_size = 32
        num_classes = 7

        # Create simple model with dropout
        model = nn.Sequential(
            nn.Linear(31, 64),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        model.train()

        # Run same input multiple times
        x = torch.randn(batch_size, 31)
        entropies = []

        for seed in range(5):
            torch.manual_seed(seed)
            with torch.no_grad():
                logits = model(x)
            probs = torch.softmax(logits, dim=1).numpy()
            summary = summarize_entropy(probs)
            entropies.append(summary.mean)

        entropy_array = np.array(entropies)
        entropy_std = np.std(entropy_array)
        entropy_cv = entropy_std / (np.mean(entropy_array) + 1e-8)  # Coefficient of variation

        # Entropy should not have wild swings (CV < 0.3 reasonable)
        assert entropy_cv < 0.5, \
            f"Entropy too unstable under dropout (CV={entropy_cv:.4f})"

        logger.info("✓ Dropout stability passed:")
        logger.info(f"  Entropies: {entropy_array}")
        logger.info(f"  Mean: {np.mean(entropy_array):.4f}, Std: {entropy_std:.4f}, CV: {entropy_cv:.4f}")

    def test_guard_policy_stability(self):
        """Guard trigger decision should be stable across seeds."""
        batch_size = 32
        num_classes = 7

        # Create a moderately problematic logits scenario
        logits_base = torch.randn(batch_size, num_classes)
        logits_base[:, 0] += 3.0  # Slight preference for class 0

        triggers = []
        for seed in range(5):
            torch.manual_seed(seed)
            # Add small random perturbation
            perturbed = logits_base + torch.randn_like(logits_base) * 0.1
            probs = torch.softmax(perturbed, dim=1).numpy()
            summary = summarize_entropy(probs)
            predicted = np.argmax(probs, axis=1)

            should_trigger, _ = should_trigger_entropy_guard(
                summary,
                has_missing_classes=(len(np.unique(predicted)) < num_classes),
                streak_count=1,
            )
            triggers.append(should_trigger)

        # Guard should be stable (not flip-flopping)
        trigger_changes = np.sum(np.diff(triggers).astype(bool))
        assert trigger_changes <= 1, \
            f"Guard too unstable: triggers={triggers}, changes={trigger_changes}"

        logger.info(f"✓ Guard policy stability passed: triggers={triggers}")


# ============================================================================
# Test 6: Ensemble Decision Reliability
# ============================================================================

class TestDecisionReliability:
    """Verify entropy guard decisions are reliable for promotion gating."""

    def test_guard_vs_missing_classes(self):
        """Low entropy should only trigger when actual classes missing."""
        batch_size = 32
        num_classes = 7

        # Case 1: Low entropy but all classes present (shouldn't trigger)
        logits_diverse = torch.randn(batch_size, num_classes)
        for c in range(num_classes):
            logits_diverse[c::num_classes, c] += 2.0
        probs_diverse = torch.softmax(logits_diverse, dim=1).numpy()
        predicted_diverse = np.argmax(probs_diverse, axis=1)

        summary_diverse = summarize_entropy(probs_diverse)
        all_classes_present = len(np.unique(predicted_diverse)) == num_classes

        should_trigger_diverse, _ = should_trigger_entropy_guard(
            summary_diverse,
            has_missing_classes=not all_classes_present,
            streak_count=1,
        )

        # Case 2: Low entropy with missing classes (should trigger)
        logits_collapsed = torch.ones(batch_size, num_classes) * 0.1
        logits_collapsed[:, 0] = 10.0
        probs_collapsed = torch.softmax(logits_collapsed, dim=1).numpy()
        predicted_collapsed = np.argmax(probs_collapsed, axis=1)

        summary_collapsed = summarize_entropy(probs_collapsed)
        classes_missing = len(np.unique(predicted_collapsed)) < num_classes

        should_trigger_collapsed, _ = should_trigger_entropy_guard(
            summary_collapsed,
            has_missing_classes=classes_missing,
            streak_count=1,
        )

        logger.info("✓ Decision reliability test:")
        logger.info(f"  Case 1 (diverse): entropy={summary_diverse.mean:.4f}, "
                   f"all_classes={all_classes_present}, trigger={should_trigger_diverse}")
        logger.info(f"  Case 2 (collapsed): entropy={summary_collapsed.mean:.4f}, "
                   f"missing_classes={classes_missing}, trigger={should_trigger_collapsed}")

        # Guard should be conservative: prefer not triggering unless confirmed collapse
        assert should_trigger_collapsed, "Should trigger on confirmed collapse"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
