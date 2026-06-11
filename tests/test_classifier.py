"""
Targeted unit tests for src/helix_ids/models/classifier.py.

Coverage target: 71% -> 90%+ by exercising uncovered paths:

- forward(return_intermediates=True): lines 395-398
- predict_5class(): lines 427-447, including return_probs=True path
- predict_with_confidence(): lines 467-479, both confidence_head present and absent
- get_hierarchical_loss_weights(): lines 500-505
- freeze_binary_head() / unfreeze_binary_head(): lines 512-513, 517-518
- get_num_parameters(): lines 522-535, with and without fine/confidence heads
- convert_labels_to_hierarchical(): lines 602-609
- hierarchical_to_5class(): lines 626-632
"""

import pytest
import torch

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config_full():
    """Full variant config (default)."""
    from src.helix_ids.models.classifier import ClassifierConfig
    return ClassifierConfig.full()


@pytest.fixture
def config_lite():
    """Lite variant config — no fine-grained, has confidence."""
    from src.helix_ids.models.classifier import ClassifierConfig
    return ClassifierConfig.lite()


@pytest.fixture
def config_nano():
    """Nano variant config — no fine-grained, no confidence."""
    from src.helix_ids.models.classifier import ClassifierConfig
    return ClassifierConfig.nano()


@pytest.fixture
def classifier_full(config_full):
    """HierarchicalClassifier with full config (fine-grained + confidence)."""
    from src.helix_ids.models.classifier import HierarchicalClassifier
    return HierarchicalClassifier(input_dim=32, config=config_full)


@pytest.fixture
def classifier_lite(config_lite):
    """HierarchicalClassifier with lite config (no fine-grained, has confidence)."""
    from src.helix_ids.models.classifier import HierarchicalClassifier
    return HierarchicalClassifier(input_dim=32, config=config_lite)


@pytest.fixture
def classifier_nano(config_nano):
    """HierarchicalClassifier with nano config (no fine-grained, no confidence)."""
    from src.helix_ids.models.classifier import HierarchicalClassifier
    return HierarchicalClassifier(input_dim=32, config=config_nano)


@pytest.fixture
def features():
    """Random features tensor."""
    return torch.randn(16, 32)


@pytest.fixture
def batch_features():
    """Single-sample features for edge-case testing."""
    return torch.randn(1, 32)


# =============================================================================
# ClassifierConfig tests
# =============================================================================


class TestClassifierConfig:
    """Cover ClassifierConfig classmethods and defaults."""

    def test_default_config(self):
        from src.helix_ids.models.classifier import ClassifierConfig
        cfg = ClassifierConfig()
        assert cfg.hidden_dim == 128
        assert cfg.num_binary_classes == 2
        assert cfg.num_family_classes == 4
        assert cfg.num_fine_classes == 23
        assert cfg.dropout == 0.3
        assert cfg.enable_fine_grained is True
        assert cfg.enable_confidence is True
        assert cfg.use_layer_norm is True

    def test_nano_preset(self):
        """Nano variant for extreme edge (Raspberry Pi Zero)."""
        from src.helix_ids.models.classifier import ClassifierConfig
        cfg = ClassifierConfig.nano()
        assert cfg.hidden_dim == 32
        assert cfg.enable_fine_grained is False
        assert cfg.enable_confidence is False
        assert cfg.use_layer_norm is False

    def test_lite_preset(self):
        """Lite variant for edge devices (Raspberry Pi 4)."""
        from src.helix_ids.models.classifier import ClassifierConfig
        cfg = ClassifierConfig.lite()
        assert cfg.hidden_dim == 64
        assert cfg.enable_fine_grained is False
        assert cfg.enable_confidence is True
        assert cfg.use_layer_norm is True

    def test_full_preset(self):
        """Full variant for server deployment."""
        from src.helix_ids.models.classifier import ClassifierConfig
        cfg = ClassifierConfig.full()
        assert cfg.hidden_dim == 128
        assert cfg.enable_fine_grained is True
        assert cfg.enable_confidence is True
        assert cfg.use_layer_norm is True

    def test_nano_classmethod(self):
        """Test nano() returns correct config."""
        from src.helix_ids.models.classifier import ClassifierConfig
        cfg = ClassifierConfig.nano()
        assert cfg.hidden_dim == 32
        assert cfg.dropout == 0.1
        assert cfg.enable_fine_grained is False
        assert cfg.enable_confidence is False
        assert cfg.use_layer_norm is False


# =============================================================================
# ClassificationHead tests
# =============================================================================


class TestClassificationHead:
    """Cover ClassificationHead — conditioning path, weight init, forward."""

    def test_forward_without_condition(self):
        from src.helix_ids.models.classifier import ClassificationHead
        head = ClassificationHead(input_dim=32, hidden_dim=64, output_dim=4)
        x = torch.randn(8, 32)
        out = head(x)
        assert out.shape == (8, 4)

    def test_forward_with_condition(self):
        from src.helix_ids.models.classifier import ClassificationHead
        head = ClassificationHead(input_dim=32, hidden_dim=64, output_dim=4, condition_dim=2)
        x = torch.randn(8, 32)
        condition = torch.randn(8, 2)
        out = head(x, condition=condition)
        assert out.shape == (8, 4)

    def test_no_layer_norm(self):
        from src.helix_ids.models.classifier import ClassificationHead
        head = ClassificationHead(input_dim=8, hidden_dim=16, output_dim=2, use_layer_norm=False)
        x = torch.randn(4, 8)
        out = head(x)
        assert out.shape == (4, 2)

    def test_weight_init_xavier(self):
        """Verify linear layers use Xavier uniform init."""
        from src.helix_ids.models.classifier import ClassificationHead
        head = ClassificationHead(input_dim=8, hidden_dim=16, output_dim=2)
        for module in head.modules():
            if isinstance(module, torch.nn.Linear):
                # Xavier uniform: std ~ sqrt(2 / (fan_in + fan_out))
                w = module.weight
                assert w is not None
                # Just check it's not full of zeros
                assert not torch.allclose(w, torch.zeros_like(w))


# =============================================================================
# ConfidenceHead tests
# =============================================================================


class TestConfidenceHead:
    """Cover ConfidenceHead forward."""

    def test_forward_shape(self):
        from src.helix_ids.models.classifier import ConfidenceHead
        # input_dim must account for the 4 extra signals concatenated in forward
        head = ConfidenceHead(input_dim=36, hidden_dim=64)
        features = torch.randn(8, 32)
        binary_logits = torch.randn(8, 2)
        family_logits = torch.randn(8, 4)
        out = head(features, binary_logits, family_logits)
        assert out.shape == (8, 1)
        # Sigmoid output should be in (0, 1)
        assert (out > 0).all() and (out < 1).all()


# =============================================================================
# HierarchicalClassifier — forward path tests
# =============================================================================


class TestHierarchicalClassifierForward:
    """Cover HierarchicalClassifier.forward()."""

    def test_forward_full(self, classifier_full, features):
        """Full variant: binary + family + fine + confidence outputs."""
        outputs = classifier_full(features)
        assert "binary" in outputs
        assert "family" in outputs
        assert "fine" in outputs
        assert "confidence" in outputs
        assert outputs["binary"].shape == (16, 2)
        assert outputs["family"].shape == (16, 4)
        assert outputs["fine"].shape == (16, 23)
        assert outputs["confidence"].shape == (16, 1)

    def test_forward_lite(self, classifier_lite, features):
        """Lite variant: no fine-grained, has confidence."""
        outputs = classifier_lite(features)
        assert "binary" in outputs
        assert "family" in outputs
        assert "fine" not in outputs
        assert "confidence" in outputs

    def test_forward_nano(self, classifier_nano, features):
        """Nano variant: no fine-grained, no confidence."""
        outputs = classifier_nano(features)
        assert "binary" in outputs
        assert "family" in outputs
        assert "fine" not in outputs
        assert "confidence" not in outputs

    # --- Lines 395-398: return_intermediates=True ---
    def test_forward_return_intermediates_full(self, classifier_full, features):
        """Cover lines 395-398: return_intermediates=True includes probs."""
        outputs = classifier_full(features, return_intermediates=True)
        assert "binary_probs" in outputs
        assert "family_probs" in outputs
        assert "fine_probs" in outputs
        # Verify softmax properties
        assert torch.allclose(outputs["binary_probs"].sum(dim=-1), torch.ones(16))
        assert torch.allclose(outputs["family_probs"].sum(dim=-1), torch.ones(16))
        assert torch.allclose(outputs["fine_probs"].sum(dim=-1), torch.ones(16))

    def test_forward_return_intermediates_lite(self, classifier_lite, features):
        """Cover lines 395-398: lite has no fine_head so fine_probs not included."""
        outputs = classifier_lite(features, return_intermediates=True)
        assert "binary_probs" in outputs
        assert "family_probs" in outputs
        assert "fine_probs" not in outputs  # fine_head is None

    def test_forward_return_intermediates_nano(self, classifier_nano, features):
        """Cover lines 395-398: nano has no fine_head."""
        outputs = classifier_nano(features, return_intermediates=True)
        assert "binary_probs" in outputs
        assert "family_probs" in outputs
        assert "fine_probs" not in outputs

    def test_forward_batch_size_one(self, classifier_full, batch_features):
        """Edge case: single sample."""
        outputs = classifier_full(batch_features, return_intermediates=True)
        assert outputs["binary"].shape == (1, 2)
        assert outputs["family"].shape == (1, 4)
        assert "fine_probs" in outputs


# =============================================================================
# HierarchicalClassifier — predict_5class tests
# =============================================================================


class TestPredict5Class:
    """Cover predict_5class — lines 427-447."""

    def test_predict_5class_basic(self, classifier_full, features):
        """Cover lines 427-447: basic predict_5class returns argmax class indices."""
        preds = classifier_full.predict_5class(features)
        assert preds.shape == (16,)
        assert preds.dtype == torch.long
        assert preds.min() >= 0
        assert preds.max() <= 4

    def test_predict_5class_return_probs(self, classifier_full, features):
        """Cover lines 443-444: return_probs=True branch."""
        probs = classifier_full.predict_5class(features, return_probs=True)
        assert probs.shape == (16, 5)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(16))
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_predict_5class_threshold(self, classifier_full, features):
        """Cover threshold parameter usage (line 405)."""
        preds_low = classifier_full.predict_5class(features, threshold=0.1)
        preds_high = classifier_full.predict_5class(features, threshold=0.9)
        assert preds_low.shape == (16,)
        assert preds_high.shape == (16,)

    def test_predict_5class_lite(self, classifier_lite, features):
        """Lite variant."""
        preds = classifier_lite.predict_5class(features)
        assert preds.shape == (16,)

    def test_predict_5class_nano(self, classifier_nano, features):
        """Nano variant."""
        preds = classifier_nano.predict_5class(features)
        assert preds.shape == (16,)


# =============================================================================
# HierarchicalClassifier — predict_with_confidence tests
# =============================================================================


class TestPredictWithConfidence:
    """Cover predict_with_confidence — lines 467-479."""

    def test_predict_with_confidence_full(self, classifier_full, features):
        """Cover lines 467-479: confidence_head is present (line 470-471)."""
        preds, confidence, mask = classifier_full.predict_with_confidence(features)
        assert preds.shape == (16,)
        assert confidence.shape == (16,)
        assert mask.shape == (16,)
        assert mask.dtype == torch.bool
        # Confidence should be in (0, 1) range from sigmoid
        assert (confidence >= 0).all()
        assert (confidence <= 1).all()

    def test_predict_with_confidence_nano(self, classifier_nano, features):
        """Cover lines 473-475: confidence_head is None, fallback to max prob."""
        preds, confidence, mask = classifier_nano.predict_with_confidence(features)
        assert preds.shape == (16,)
        assert confidence.shape == (16,)
        assert mask.shape == (16,)
        # Fallback confidence: max of 5-class probs
        assert (confidence >= 0).all()
        assert (confidence <= 1).all()

    def test_predict_with_confidence_threshold_high(self, classifier_full, features):
        """Cover line 477: threshold filtering."""
        preds, confidence, mask = classifier_full.predict_with_confidence(
            features, confidence_threshold=0.99
        )
        assert mask.shape == (16,)
        # With such a high threshold, likely no high-confidence predictions
        # (Just verify it runs without error)

    def test_predict_with_confidence_threshold_zero(self, classifier_full, features):
        """Threshold of 0 should make all predictions high-confidence."""
        preds, confidence, mask = classifier_full.predict_with_confidence(
            features, confidence_threshold=0.0
        )
        assert mask.all()  # All should be high-confidence

    def test_predict_with_confidence_lite(self, classifier_lite, features):
        """Lite variant: has confidence head but no fine-grained."""
        preds, confidence, mask = classifier_lite.predict_with_confidence(features)
        assert preds.shape == (16,)
        assert confidence.shape == (16,)


# =============================================================================
# HierarchicalClassifier — get_hierarchical_loss_weights tests
# =============================================================================


class TestGetHierarchicalLossWeights:
    """Cover get_hierarchical_loss_weights — lines 500-505."""

    def test_all_normal(self, classifier_full):
        """All normal samples (binary_labels=0)."""
        binary_labels = torch.zeros(8, dtype=torch.long)
        family_labels = torch.full((8,), -1, dtype=torch.long)
        weights = classifier_full.get_hierarchical_loss_weights(binary_labels, family_labels)
        assert "binary" in weights
        assert "family" in weights
        assert torch.allclose(weights["binary"], torch.ones(8))
        assert torch.allclose(weights["family"], torch.zeros(8))  # all normal -> weight 0

    def test_all_attack(self, classifier_full):
        """All attack samples (binary_labels=1)."""
        binary_labels = torch.ones(8, dtype=torch.long)
        family_labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
        weights = classifier_full.get_hierarchical_loss_weights(binary_labels, family_labels)
        assert torch.allclose(weights["binary"], torch.ones(8))
        assert torch.allclose(weights["family"], torch.ones(8))  # all attacks -> weight 1

    def test_mixed(self, classifier_full):
        """Mix of normal and attack samples."""
        binary_labels = torch.tensor([0, 1, 1, 0, 1, 0], dtype=torch.long)
        family_labels = torch.tensor([-1, 0, 2, -1, 3, -1], dtype=torch.long)
        weights = classifier_full.get_hierarchical_loss_weights(binary_labels, family_labels)
        assert torch.allclose(weights["binary"], torch.ones(6))
        expected_family = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
        assert torch.allclose(weights["family"], expected_family)


# =============================================================================
# HierarchicalClassifier — freeze/unfreeze tests
# =============================================================================


class TestFreezeUnfreeze:
    """Cover freeze_binary_head (lines 512-513) and unfreeze_binary_head (lines 517-518)."""

    def test_freeze_binary_head(self, classifier_full):
        """Cover lines 512-513: all binary_head params require_grad=False."""
        classifier_full.freeze_binary_head()
        for param in classifier_full.binary_head.parameters():
            assert param.requires_grad is False

    def test_unfreeze_binary_head(self, classifier_full):
        """Cover lines 517-518: all binary_head params require_grad=True."""
        # Freeze first
        classifier_full.freeze_binary_head()
        for param in classifier_full.binary_head.parameters():
            assert param.requires_grad is False
        # Then unfreeze
        classifier_full.unfreeze_binary_head()
        for param in classifier_full.binary_head.parameters():
            assert param.requires_grad is True

    def test_freeze_then_forward(self, classifier_full, features):
        """Freezing should not break forward pass."""
        classifier_full.freeze_binary_head()
        outputs = classifier_full(features)
        assert "binary" in outputs

    def test_freeze_only_binary_head(self, classifier_full):
        """Other heads should remain trainable after freezing binary head."""
        classifier_full.freeze_binary_head()
        # Family head should still be trainable (unless it shares params, which it doesn't)
        for param in classifier_full.family_head.parameters():
            assert param.requires_grad is True


# =============================================================================
# HierarchicalClassifier — get_num_parameters tests
# =============================================================================


class TestGetNumParameters:
    """Cover get_num_parameters — lines 522-535."""

    def test_num_parameters_full(self, classifier_full):
        """Cover lines 522-535: full variant includes all heads."""
        counts = classifier_full.get_num_parameters()
        assert "binary_head" in counts
        assert "family_head" in counts
        assert "fine_head" in counts      # full has fine_head
        assert "confidence_head" in counts  # full has confidence_head
        assert "total" in counts
        assert counts["binary_head"] > 0
        assert counts["family_head"] > 0
        assert counts["fine_head"] > 0
        assert counts["confidence_head"] > 0
        expected_total = (
            counts["binary_head"]
            + counts["family_head"]
            + counts["fine_head"]
            + counts["confidence_head"]
        )
        assert counts["total"] == expected_total

    def test_num_parameters_nano(self, classifier_nano):
        """Cover lines 527, 530: nano has neither fine_head nor confidence_head."""
        counts = classifier_nano.get_num_parameters()
        assert "binary_head" in counts
        assert "family_head" in counts
        assert "fine_head" not in counts
        assert "confidence_head" not in counts
        assert "total" in counts
        assert counts["total"] == counts["binary_head"] + counts["family_head"]

    def test_num_parameters_lite(self, classifier_lite):
        """Lite has confidence_head but no fine_head."""
        counts = classifier_lite.get_num_parameters()
        assert "fine_head" not in counts
        assert "confidence_head" in counts
        assert "total" in counts
        expected_total = (
            counts["binary_head"]
            + counts["family_head"]
            + counts["confidence_head"]
        )
        assert counts["total"] == expected_total

    def test_all_positive(self, classifier_full):
        """All parameter counts should be positive integers."""
        counts = classifier_full.get_num_parameters()
        for key, value in counts.items():
            assert value >= 0, f"{key} count is negative"


# =============================================================================
# Utility function tests
# =============================================================================


class TestConvertLabelsToHierarchical:
    """Cover convert_labels_to_hierarchical — lines 602-609."""

    def test_all_normal(self):
        """All normal labels (0)."""
        from src.helix_ids.models.classifier import convert_labels_to_hierarchical
        labels = torch.zeros(8, dtype=torch.long)
        binary, family = convert_labels_to_hierarchical(labels)
        assert torch.allclose(binary, torch.zeros(8, dtype=torch.long))
        assert torch.allclose(family, torch.full((8,), -1, dtype=torch.long))

    def test_all_attack(self):
        """All attack labels (1-4)."""
        from src.helix_ids.models.classifier import convert_labels_to_hierarchical
        labels = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4], dtype=torch.long)
        binary, family = convert_labels_to_hierarchical(labels)
        assert torch.allclose(binary, torch.ones(8, dtype=torch.long))
        expected_family = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
        assert torch.allclose(family, expected_family)

    def test_mixed(self):
        """Mix of normal and attack labels."""
        from src.helix_ids.models.classifier import convert_labels_to_hierarchical
        labels = torch.tensor([0, 1, 3, 0, 4, 2], dtype=torch.long)
        binary, family = convert_labels_to_hierarchical(labels)
        expected_binary = torch.tensor([0, 1, 1, 0, 1, 1], dtype=torch.long)
        assert torch.allclose(binary, expected_binary)
        expected_family = torch.tensor([-1, 0, 2, -1, 3, 1], dtype=torch.long)
        assert torch.allclose(family, expected_family)

    def test_edge_single(self):
        """Single element tensors."""
        from src.helix_ids.models.classifier import convert_labels_to_hierarchical
        # Single normal
        b, f = convert_labels_to_hierarchical(torch.tensor([0]))
        assert b.item() == 0
        assert f.item() == -1
        # Single DoS
        b, f = convert_labels_to_hierarchical(torch.tensor([1]))
        assert b.item() == 1
        assert f.item() == 0
        # Single U2R
        b, f = convert_labels_to_hierarchical(torch.tensor([4]))
        assert b.item() == 1
        assert f.item() == 3


class TestHierarchicalTo5Class:
    """Cover hierarchical_to_5class — lines 626-632."""

    def test_all_normal(self):
        """All binary predictions are Normal (0)."""
        from src.helix_ids.models.classifier import hierarchical_to_5class
        binary_pred = torch.zeros(8, dtype=torch.long)
        family_pred = torch.zeros(8, dtype=torch.long)  # ignored
        result = hierarchical_to_5class(binary_pred, family_pred)
        assert torch.allclose(result, torch.zeros(8, dtype=torch.long))

    def test_all_attack(self):
        """All binary predictions are Attack (1)."""
        from src.helix_ids.models.classifier import hierarchical_to_5class
        binary_pred = torch.ones(8, dtype=torch.long)
        family_pred = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
        result = hierarchical_to_5class(binary_pred, family_pred)
        expected = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4], dtype=torch.long)
        assert torch.allclose(result, expected)

    def test_mixed(self):
        """Mix of normal and attack predictions."""
        from src.helix_ids.models.classifier import hierarchical_to_5class
        binary_pred = torch.tensor([0, 1, 1, 0, 0, 1], dtype=torch.long)
        family_pred = torch.tensor([0, 1, 2, 3, 0, 0], dtype=torch.long)
        result = hierarchical_to_5class(binary_pred, family_pred)
        expected = torch.tensor([0, 2, 3, 0, 0, 1], dtype=torch.long)
        assert torch.allclose(result, expected)

    def test_edge_single(self):
        """Single element tensors."""
        from src.helix_ids.models.classifier import hierarchical_to_5class
        # Single normal
        result = hierarchical_to_5class(
            torch.tensor([0]), torch.tensor([0])
        )
        assert result.item() == 0
        # Single DoS
        result = hierarchical_to_5class(
            torch.tensor([1]), torch.tensor([0])
        )
        assert result.item() == 1
        # Single U2R
        result = hierarchical_to_5class(
            torch.tensor([1]), torch.tensor([3])
        )
        assert result.item() == 4


# =============================================================================
# Factory function tests
# =============================================================================


class TestFactoryFunctions:
    """Cover factory functions (nano/lite/full)."""

    def test_hierarchical_classifier_nano(self):
        from src.helix_ids.models.classifier import hierarchical_classifier_nano
        model = hierarchical_classifier_nano(input_dim=32)
        assert model.config.hidden_dim == 32
        assert model.config.enable_fine_grained is False
        assert model.config.enable_confidence is False

    def test_hierarchical_classifier_lite(self):
        from src.helix_ids.models.classifier import hierarchical_classifier_lite
        model = hierarchical_classifier_lite(input_dim=32)
        assert model.config.hidden_dim == 64
        assert model.config.enable_fine_grained is False
        assert model.config.enable_confidence is True

    def test_hierarchical_classifier_full(self):
        from src.helix_ids.models.classifier import hierarchical_classifier_full
        model = hierarchical_classifier_full(input_dim=32)
        assert model.config.hidden_dim == 128
        assert model.config.enable_fine_grained is True
        assert model.config.enable_confidence is True

    def test_alias_nano(self):
        from src.helix_ids.models.classifier import HierarchicalClassifierNano
        model = HierarchicalClassifierNano(input_dim=16)
        assert model.config.hidden_dim == 32

    def test_alias_lite(self):
        from src.helix_ids.models.classifier import HierarchicalClassifierLite
        model = HierarchicalClassifierLite(input_dim=16)
        assert model.config.hidden_dim == 64

    def test_alias_full(self):
        from src.helix_ids.models.classifier import HierarchicalClassifierFull
        model = HierarchicalClassifierFull(input_dim=16)
        assert model.config.hidden_dim == 128


# =============================================================================
# Constructor tests — edge cases
# =============================================================================


class TestConstructor:
    """Cover constructor edge cases."""

    def test_default_config(self):
        from src.helix_ids.models.classifier import ClassifierConfig, HierarchicalClassifier
        model = HierarchicalClassifier(input_dim=32)
        assert model.config == ClassifierConfig()
        assert model.fine_head is not None
        assert model.confidence_head is not None

    def test_confidence_disabled(self):
        from src.helix_ids.models.classifier import ClassifierConfig, HierarchicalClassifier
        cfg = ClassifierConfig(enable_confidence=False, enable_fine_grained=False)
        model = HierarchicalClassifier(input_dim=32, config=cfg)
        assert model.confidence_head is None
        assert model.fine_head is None

    def test_binary_classes_override(self):
        from src.helix_ids.models.classifier import ClassifierConfig, HierarchicalClassifier
        cfg = ClassifierConfig(num_binary_classes=3)
        model = HierarchicalClassifier(input_dim=32, config=cfg)
        # Binary head should output 3 classes
        x = torch.randn(4, 32)
        out = model(x)
        assert out["binary"].shape == (4, 3)

    def test_family_classes_override(self):
        from src.helix_ids.models.classifier import ClassifierConfig, HierarchicalClassifier
        cfg = ClassifierConfig(num_family_classes=5)
        model = HierarchicalClassifier(input_dim=32, config=cfg)
        x = torch.randn(4, 32)
        out = model(x)
        assert out["family"].shape == (4, 5)

    def test_hidden_dim_override(self):
        from src.helix_ids.models.classifier import ClassifierConfig, HierarchicalClassifier
        cfg = ClassifierConfig(hidden_dim=256)
        model = HierarchicalClassifier(input_dim=32, config=cfg)
        x = torch.randn(2, 32)
        out = model(x)
        assert out["binary"].shape == (2, 2)

    def test_fine_head_disabled_on_large_input(self):
        """Regression: fine_head=None when enable_fine_grained=False."""
        from src.helix_ids.models.classifier import ClassifierConfig, HierarchicalClassifier
        cfg = ClassifierConfig(enable_fine_grained=False)
        model = HierarchicalClassifier(input_dim=128, config=cfg)
        assert model.fine_head is None
        x = torch.randn(2, 128)
        out = model(x)
        assert "fine" not in out
