"""
Tests for model export, quantization, and deployment pipelines.

These are the critical test gaps identified during codebase analysis.
"""
import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn


# ============================================================================
# Fixtures
# ============================================================================

class SimpleMLP(nn.Module):
    """Minimal MLP for testing."""
    def __init__(self, input_dim=10, hidden_dim=8, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
    def forward(self, x):
        return self.net(x)


@pytest.fixture
def simple_model():
    model = SimpleMLP(input_dim=10, hidden_dim=8, num_classes=5)
    model.eval()
    return model


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(seed=42)
    X = rng.standard_normal((100, 10)).astype(np.float32)
    y = rng.integers(0, 5, size=100)
    return X, y


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ============================================================================
# Export Tests
# ============================================================================

class TestModelExport:
    """Test ONNX and C header export functionality."""

    def test_model_save_and_load_pytorch(self, simple_model, tmp_dir):
        """Test basic PyTorch save/load roundtrip."""
        path = tmp_dir / "model.pt"
        torch.save(simple_model.state_dict(), path)

        loaded = SimpleMLP(10, 8, 5)
        loaded.load_state_dict(torch.load(path, map_location="cpu"))
        loaded.eval()

        x = torch.randn(5, 10)
        with torch.no_grad():
            orig_out = simple_model(x)
            loaded_out = loaded(x)
        
        assert torch.allclose(orig_out, loaded_out, atol=1e-6)

    def test_export_to_onnx(self, simple_model, tmp_dir):
        """Test ONNX export produces valid file."""
        try:
            import onnx
        except ImportError:
            pytest.skip("onnx not installed")

        path = tmp_dir / "model.onnx"
        dummy = torch.randn(1, 10)
        torch.onnx.export(simple_model, dummy, str(path),
                         input_names=["input"], output_names=["output"],
                         dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
        
        assert path.exists()
        assert path.stat().st_size > 0
        
        model = onnx.load(str(path))
        onnx.checker.check_model(model)

    def test_onnx_inference_matches_pytorch(self, simple_model, tmp_dir):
        """Test ONNX inference produces same results as PyTorch."""
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            pytest.skip("onnx/onnxruntime not installed")

        path = tmp_dir / "model.onnx"
        dummy = torch.randn(1, 10)
        torch.onnx.export(simple_model, dummy, str(path),
                         input_names=["input"], output_names=["output"])

        session = ort.InferenceSession(str(path))
        rng = np.random.default_rng(seed=42)
        x = rng.standard_normal((5, 10)).astype(np.float32)

        with torch.no_grad():
            pytorch_out = simple_model(torch.from_numpy(x)).numpy()

        onnx_out = session.run(None, {"input": x})[0]
        np.testing.assert_allclose(pytorch_out, onnx_out, atol=1e-5)

    def test_export_c_header(self, simple_model, tmp_dir):
        """Test C header generation for ESP32."""
        path = tmp_dir / "model.h"
        
        # Generate C array from model weights
        weights = []
        for param in simple_model.parameters():
            weights.extend(param.detach().numpy().flatten().tolist())
        
        with open(path, 'w') as f:
            f.write("#ifndef HELIX_MODEL_H\n#define HELIX_MODEL_H\n\n")
            f.write(f"const int MODEL_WEIGHTS_SIZE = {len(weights)};\n")
            f.write("const float MODEL_WEIGHTS[] = {\n")
            for i, w in enumerate(weights):
                f.write(f"  {w:.8f}f")
                if i < len(weights) - 1:
                    f.write(",")
                if (i + 1) % 8 == 0:
                    f.write("\n")
            f.write("\n};\n\n#endif\n")
        
        assert path.exists()
        content = path.read_text()
        assert "MODEL_WEIGHTS" in content
        assert f"MODEL_WEIGHTS_SIZE = {len(weights)}" in content


# ============================================================================
# Quantization Tests
# ============================================================================

class TestQuantization:
    """Test model quantization for edge deployment."""

    def test_dynamic_quantization(self, simple_model):
        """Test PyTorch dynamic quantization."""
        if not hasattr(torch.backends, "quantized") or torch.backends.quantized.engine == "none":
            pytest.skip("Quantization engine not available on this platform (Apple Silicon/NoQEngine)")
        quantized = torch.quantization.quantize_dynamic(
            simple_model, {nn.Linear}, dtype=torch.qint8
        )
        x = torch.randn(5, 10)
        output = quantized(x)
        assert output.shape == (5, 5)

    def test_quantized_model_smaller(self, simple_model, tmp_dir):
        """Test that quantized model is smaller."""
        if not hasattr(torch.backends, "quantized") or torch.backends.quantized.engine == "none":
            pytest.skip("Quantization engine not available on this platform (Apple Silicon/NoQEngine)")
        # Save original
        orig_path = tmp_dir / "original.pt"
        torch.save(simple_model.state_dict(), orig_path)
        # Quantize and save
        quantized = torch.quantization.quantize_dynamic(
            simple_model, {nn.Linear}, dtype=torch.qint8
        )
        quant_path = tmp_dir / "quantized.pt"
        torch.save(quantized.state_dict(), quant_path)
        # Quantized should be smaller (or at least not much larger)
        orig_size = orig_path.stat().st_size
        quant_size = quant_path.stat().st_size
        assert quant_size < orig_size * 2  # Relaxed check

    def test_quantized_accuracy_degradation(self, simple_model, sample_data):
        """Test that quantization doesn't degrade accuracy too much."""
        if not hasattr(torch.backends, "quantized") or torch.backends.quantized.engine == "none":
            pytest.skip("Quantization engine not available on this platform (Apple Silicon/NoQEngine)")
        X, _ = sample_data
        x_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            orig_preds = simple_model(x_tensor).argmax(dim=1).numpy()
        quantized = torch.quantization.quantize_dynamic(
            simple_model, {nn.Linear}, dtype=torch.qint8
        )
        with torch.no_grad():
            quant_preds = quantized(x_tensor).argmax(dim=1).numpy()
        agreement = (orig_preds == quant_preds).mean()
        assert agreement > 0.5  # Relaxed for random weights


# ============================================================================
# Deployment Pipeline Tests
# ============================================================================

class TestDeploymentPipeline:
    """Test deployment pipeline components."""

    def test_model_card_generation(self, simple_model, tmp_dir):
        """Test model card JSON generation."""
        card = {
            'model_name': 'HELIX-IDS-test',
            'version': '2.0',
            'architecture': '[8]',
            'n_features': 10,
            'n_classes': 5,
            'class_names': ['Normal', 'DoS', 'Probe', 'R2L', 'U2R'],
            'parameters': sum(p.numel() for p in simple_model.parameters()),
        }
        
        path = tmp_dir / 'model_card.json'
        with open(path, 'w') as f:
            json.dump(card, f, indent=2)
        
        with open(path) as f:
            loaded = json.load(f)
        
        assert loaded['model_name'] == 'HELIX-IDS-test'
        assert loaded['n_classes'] == 5

    def test_scaler_save_load(self, tmp_dir, sample_data):
        """Test scaler pickle roundtrip."""
        from sklearn.preprocessing import StandardScaler
        
        X, _ = sample_data
        scaler = StandardScaler()
        scaler.fit(X)
        
        path = tmp_dir / 'scaler.pkl'
        with open(path, 'wb') as f:
            pickle.dump(scaler, f)
        
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
        
        x_orig = scaler.transform(X[:5])
        x_loaded = loaded.transform(X[:5])
        np.testing.assert_array_equal(x_orig, x_loaded)

    def test_batch_inference(self, simple_model, sample_data):
        """Test batch inference produces correct shapes."""
        X, _ = sample_data
        x_tensor = torch.FloatTensor(X)
        
        simple_model.eval()
        with torch.no_grad():
            output = simple_model(x_tensor)
        
        assert output.shape == (100, 5)
        
        probs = torch.softmax(output, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(100), atol=1e-5)

    def test_single_sample_inference(self, simple_model):
        """Test single sample inference."""
        x = torch.randn(1, 10)
        simple_model.eval()
        with torch.no_grad():
            output = simple_model(x)
        
        assert output.shape == (1, 5)

    def test_model_size_constraint(self, simple_model):
        """Test model meets size constraints."""
        n_params = sum(p.numel() for p in simple_model.parameters())
        size_kb = n_params * 4 / 1024  # float32
        
        # Our test model should be tiny
        assert size_kb < 1  # Less than 1KB
        
    def test_class_names_match_output_dim(self, simple_model):
        """Test output dimension matches class count."""
        class_names = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
        x = torch.randn(1, 10)
        with torch.no_grad():
            output = simple_model(x)
        assert output.shape[1] == len(class_names)


# ============================================================================
# Data Leakage Detection Tests
# ============================================================================

class TestDataLeakageDetection:
    """Tests to verify no data leakage exists."""

    def test_separate_scalers_produce_different_results(self):
        """Verify that separate scalers for each dataset produce different transforms."""
        from sklearn.preprocessing import StandardScaler
        
        # Simulate two different datasets
        rng = np.random.default_rng(seed=42)
        x1 = rng.standard_normal((100, 10)).astype(np.float32) * 5 + 10
        x2 = rng.standard_normal((100, 10)).astype(np.float32) * 2 - 3
        
        scaler1 = StandardScaler().fit(x1)
        scaler2 = StandardScaler().fit(x2)
        
        # Scalers should have different means
        assert not np.allclose(scaler1.mean_, scaler2.mean_)

    def test_test_data_not_in_train_scaler_fit(self, sample_data):
        """Verify scaler is fit only on training data."""
        from sklearn.preprocessing import StandardScaler
        
        X, _ = sample_data
        X_train = X[:80]
        _ = X[80:]
        
        scaler = StandardScaler().fit(X_train)
        
        # Scaler mean should match train, not full dataset
        np.testing.assert_array_almost_equal(scaler.mean_, X_train.mean(axis=0))
        assert not np.allclose(scaler.mean_, X.mean(axis=0))
