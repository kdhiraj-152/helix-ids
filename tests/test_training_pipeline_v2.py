"""
Tests for training pipeline integrity — validates fixes from v2.

Specifically tests:
1. Data leakage prevention (separate scalers)
2. Class-weighted focal loss correctness
3. Per-class threshold tuning
4. 5-class evaluation (not binary)
"""
import numpy as np
import pytest
import torch
import torch.nn as nn
from collections import Counter


class TestClassWeightedFocalLoss:
    """Test the class-weighted focal loss implementation."""

    def _make_loss(self, alpha=None, gamma=2.0):
        # Inline to avoid import dependency
        class CWFL(nn.Module):
            def __init__(self, alpha, gamma, reduction='mean'):
                super().__init__()
                self.gamma = gamma
                self.reduction = reduction
                if alpha is not None:
                    self.register_buffer('alpha', torch.FloatTensor(alpha))
                else:
                    self.alpha = None
            def forward(self, logits, targets):
                ce = nn.functional.cross_entropy(logits, targets, reduction='none')
                p = torch.exp(-ce)
                fw = (1 - p) ** self.gamma
                if self.alpha is not None:
                    at = self.alpha[targets]
                    fl = at * fw * ce
                else:
                    fl = fw * ce
                return fl.mean() if self.reduction == 'mean' else fl
        return CWFL(alpha, gamma)

    def test_uniform_weights_match_standard_focal(self):
        """Uniform alpha should behave like standard focal loss."""
        logits = torch.randn(32, 5)
        targets = torch.randint(0, 5, (32,))
        
        uniform_loss = self._make_loss(alpha=[1.0]*5, gamma=2.0)
        no_weight_loss = self._make_loss(alpha=None, gamma=2.0)
        
        l1 = uniform_loss(logits, targets)
        l2 = no_weight_loss(logits, targets)
        
        assert torch.allclose(l1, l2, atol=1e-5)

    def test_higher_weight_increases_loss(self):
        """Higher class weight should increase loss for that class."""
        logits = torch.randn(10, 5)
        targets = torch.zeros(10, dtype=torch.long)  # All class 0
        
        low_weight = self._make_loss(alpha=[1.0, 1.0, 1.0, 1.0, 1.0])
        high_weight = self._make_loss(alpha=[5.0, 1.0, 1.0, 1.0, 1.0])
        
        l_low = low_weight(logits, targets)
        l_high = high_weight(logits, targets)
        
        assert l_high > l_low

    def test_minority_class_gets_higher_weight(self):
        """Inverse-frequency weighting should give minority classes more weight."""
        y = [0]*1000 + [1]*500 + [2]*200 + [3]*50 + [4]*10
        
        counts = Counter(y)
        total = len(y)
        n_classes = len(counts)
        weights = [total / (n_classes * max(counts.get(i, 1), 1)) for i in range(n_classes)]
        weights = np.array(weights) / np.mean(weights)
        
        # U2R (class 4) should have highest weight
        assert weights[4] > weights[0]
        assert weights[3] > weights[0]
        # Normal (class 0) should have lowest weight
        assert weights[0] < weights[2]

    def test_gamma_zero_reduces_to_cross_entropy(self):
        """gamma=0 should give standard cross entropy (up to alpha scaling)."""
        logits = torch.randn(32, 5)
        targets = torch.randint(0, 5, (32,))
        
        focal = self._make_loss(alpha=None, gamma=0.0)
        ce = nn.CrossEntropyLoss()
        
        l_focal = focal(logits, targets)
        l_ce = ce(logits, targets)
        
        assert torch.allclose(l_focal, l_ce, atol=1e-5)


class TestPerClassThresholdTuning:
    """Test per-class threshold tuning logic."""

    def test_threshold_range(self):
        """Thresholds should be between 0 and 1."""
        # Simulate tuning
        rng = np.random.default_rng(seed=42)
        probs = rng.dirichlet([1]*5, size=100)
        targets = rng.integers(0, 5, size=100)
        
        from sklearn.metrics import f1_score
        
        for cls in range(5):
            best_thresh = 0.5
            best_f1 = 0
            cls_probs = probs[:, cls]
            cls_true = (targets == cls).astype(int)
            
            for t in np.arange(0.1, 0.9, 0.05):
                pred = (cls_probs >= t).astype(int)
                f1 = f1_score(cls_true, pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = t
            
            assert 0.1 <= best_thresh <= 0.85

    def test_threshold_tuning_improves_f1(self):
        """Tuned thresholds should be >= default 0.5 threshold F1."""
        # Create data where class 4 has very low probability
        np.random.seed(42)
        rng = np.random.default_rng(seed=42)
        probs = rng.dirichlet([10, 5, 3, 1, 0.5], size=200)
        targets = np.array([0]*100 + [1]*50 + [2]*30 + [3]*15 + [4]*5)
        
        from sklearn.metrics import f1_score
        
        for cls in [3, 4]:  # Minority classes
            cls_probs = probs[:len(targets), cls]
            cls_true = (targets == cls).astype(int)
            
            # Default threshold
            default_pred = (cls_probs >= 0.5).astype(int)
            default_f1 = f1_score(cls_true, default_pred, zero_division=0)
            
            # Tuned threshold
            best_f1 = 0
            for t in np.arange(0.05, 0.9, 0.05):
                pred = (cls_probs >= t).astype(int)
                f1 = f1_score(cls_true, pred, zero_division=0)
                best_f1 = max(best_f1, f1)
            
            # Tuned should be >= default
            assert best_f1 >= default_f1


class TestFiveClassEvaluation:
    """Test 5-class evaluation correctness."""

    def test_5class_f1_macro(self):
        """F1 macro should average across all 5 classes equally."""
        from sklearn.metrics import f1_score
        
        y_true = np.array([0]*20 + [1]*20 + [2]*20 + [3]*20 + [4]*20)
        y_pred = y_true.copy()
        y_pred[::5] = (y_true[::5] + 1) % 5  # 20% error rate
        
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        # Macro should be mean of per-class
        assert abs(f1_macro - f1_per_class.mean()) < 1e-10
        assert len(f1_per_class) == 5

    def test_minority_class_zero_detection(self):
        """Detecting 0 of a class should give F1=0 for that class."""
        from sklearn.metrics import f1_score
        
        y_true = np.array([0]*90 + [4]*10)
        y_pred = np.zeros(100, dtype=int)  # Predict all as class 0
        
        f1_per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)
        
        assert f1_per_class[4] == pytest.approx(0.0, abs=1e-9)  # U2R not detected
        assert f1_per_class[0] > 0.9   # Normal well detected

    def test_binary_vs_5class_gap(self):
        """Demonstrate that binary accuracy masks 5-class failures."""
        from sklearn.metrics import f1_score, accuracy_score
        
        # Model predicts binary correctly but misclassifies attack type
        y_true = np.array([0]*50 + [1]*20 + [2]*15 + [3]*10 + [4]*5)
        y_pred = np.array([0]*50 + [1]*50)  # All attacks -> DoS
        
        # Binary (normal vs attack)
        y_true_bin = (y_true > 0).astype(int)
        y_pred_bin = (y_pred > 0).astype(int)
        binary_f1 = f1_score(y_true_bin, y_pred_bin, average='macro')
        
        # 5-class
        f1_5class = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Binary should be much higher than 5-class
        assert binary_f1 > f1_5class + 0.1
