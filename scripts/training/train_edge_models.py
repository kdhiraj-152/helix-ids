"""
HELIX-IDS: Train Edge Models with Engineered Features

Retrains RPi Zero, RPi 4, and ESP32 models using the new 32-feature
engineering pipeline that achieved F1=0.9869 on the production model.

Usage:
    python scripts/train_edge_models.py --platform rpi_zero
    python scripts/train_edge_models.py --platform rpi_4
    python scripts/train_edge_models.py --platform all
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset
from helix_ids.contracts.schema_contract import runtime_contract_payload

# Platform configurations
PLATFORM_CONFIGS = {
    "rpi_zero": {
        "hidden_dims": [32, 16],
        "dropout": 0.3,
        "max_params": 10000,
        "max_size_kb": 100,
        "description": "Raspberry Pi Zero W (512MB RAM)",
    },
    "rpi_4": {
        "hidden_dims": [64, 32, 16],
        "dropout": 0.3,
        "max_params": 50000,
        "max_size_kb": 500,
        "description": "Raspberry Pi 4 (4GB RAM)",
    },
    "esp32": {
        "hidden_dims": [16, 8],
        "dropout": 0.2,
        "max_params": 5000,
        "max_size_kb": 20,
        "description": "ESP32 Microcontroller (requires quantization)",
    },
}

PLATFORM_CONFIGS = dict(PLATFORM_CONFIGS)


class EdgeMLP(nn.Module):
    """Lightweight MLP for edge deployment"""

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dims: list | None = None,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [32, 16]

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_data():
    """Load preprocessed data from Phase 2a"""
    processed_dir = Path("results/processed_data")
    pipeline_dir = Path("results/preprocessing_pipeline")

    x_train = np.load(processed_dir / "X_train.npy")
    x_val = np.load(processed_dir / "X_val.npy")
    x_test = np.load(processed_dir / "X_test.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    y_val = np.load(processed_dir / "y_val.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    with open(pipeline_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open(pipeline_dir / "feature_names.json") as f:
        feature_names = json.load(f)

    return x_train, x_val, x_test, y_train, y_val, y_test, scaler, feature_names


def train_edge_model(platform: str, x_train, x_val, x_test, y_train, y_val, y_test, scaler):
    """Train model for specific edge platform"""

    config = PLATFORM_CONFIGS[platform]
    print(f"\n{'=' * 80}")
    print(f"Training for: {platform.upper()}")
    print(f"Description: {config['description']}")
    print(f"{'=' * 80}")

    # Scale data
    x_train_scaled = scaler.transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    # Create model
    hidden_dims = config["hidden_dims"]
    assert isinstance(hidden_dims, list)
    dropout = config["dropout"]
    assert isinstance(dropout, float)
    max_params = config["max_params"]
    assert isinstance(max_params, int)

    model = EdgeMLP(input_dim=x_train.shape[1], hidden_dims=hidden_dims, dropout=dropout)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {config['hidden_dims']}")
    print(f"Parameters: {num_params:,} (max: {config['max_params']:,})")

    if num_params > max_params:
        print("⚠️  Warning: Model exceeds parameter limit!")

    # Create dataloaders
    train_dataset = TensorDataset(torch.FloatTensor(x_train_scaled), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(x_val_scaled), torch.LongTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=0)

    # Training
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5)

    best_val_f1 = 0.0
    patience_counter = 0
    best_state = None

    print("\nTraining...")
    for epoch in range(50):
        # Train
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        all_preds = []
        with torch.no_grad():
            for x_batch, _ in val_loader:
                preds = model(x_batch).argmax(dim=1)
                all_preds.extend(preds.numpy())

        val_f1 = f1_score(y_val, all_preds, average="macro")
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1:2d}/50 | Val F1: {val_f1:.4f}")

        if patience_counter >= 10:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    x_test_tensor = torch.FloatTensor(x_test_scaled)
    with torch.no_grad():
        test_preds = model(x_test_tensor).argmax(dim=1).numpy()

    test_f1 = f1_score(y_test, test_preds, average="macro")

    print("\n✓ Results:")
    print(f"  • Best Val F1: {best_val_f1:.4f}")
    print(f"  • Test F1:     {test_f1:.4f}")
    print(f"\n{classification_report(y_test, test_preds, target_names=['Normal', 'Attack'])}")

    return model, test_f1, num_params


def export_model(model, platform: str, scaler, feature_names, test_f1: float, num_params: int):
    """Export trained model"""

    export_dir = Path(f"models/{platform}")
    export_dir.mkdir(parents=True, exist_ok=True)

    # Save PyTorch model payload (state dict + immutable runtime contract)
    path = export_dir / f"helix_{platform}.pt"
    payload = {
        "model_state_dict": model.state_dict(),
        "metadata": {
            "platform": platform,
            "description": PLATFORM_CONFIGS[platform]["description"],
            "architecture": f"MLP {PLATFORM_CONFIGS[platform]['hidden_dims']}",
            "input_features": 32,
            "parameters": num_params,
            "test_f1": float(test_f1),
            "feature_engineering": "Phase 1b (32 engineered features)",
        },
        "feature_names": feature_names,
    }
    # Embed immutable runtime contract metadata and write canonical sidecars
    payload.update(runtime_contract_payload())
    torch.save(payload, path)
    # Write sidecars next to checkpoint
    try:
        contract_path = path.with_suffix(path.suffix + ".contract.json")
        feature_order_path = path.with_suffix(path.suffix + ".feature_order.json")
        schema_hash_path = path.with_suffix(path.suffix + ".schema_hash.txt")
        contract_path.write_text(json.dumps(runtime_contract_payload(), indent=2), encoding="utf-8")
        feature_order_path.write_text(json.dumps(runtime_contract_payload()["feature_order"], indent=2), encoding="utf-8")
        schema_hash_path.write_text(str(runtime_contract_payload()["schema_hash"]) + "\n", encoding="utf-8")
    except Exception:
        # If sidecar emission fails, raise to avoid producing non-canonical artifact
        raise

    # Save scaler
    with open(export_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save feature names
    with open(export_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    # Save metadata
    config = PLATFORM_CONFIGS[platform]
    metadata = {
        "platform": platform,
        "description": config["description"],
        "architecture": f"MLP {config['hidden_dims']}",
        "input_features": 32,
        "parameters": num_params,
        "test_f1": float(test_f1),
        "feature_engineering": "Phase 1b (32 engineered features)",
        "constraints": {"max_params": config["max_params"], "max_size_kb": config["max_size_kb"]},
    }

    with open(export_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Calculate model size
    model_size_kb = sum(p.numel() * 4 for p in model.parameters()) / 1024

    print(f"\n✓ Exported to {export_dir}/")
    print(f"  • Model size: {model_size_kb:.1f} KB (max: {config['max_size_kb']} KB)")

    return export_dir


def main():
    parser = argparse.ArgumentParser(description="Train HELIX-IDS edge models")
    parser.add_argument(
        "--platform",
        type=str,
        default="all",
        choices=["rpi_zero", "rpi_4", "esp32", "all"],
        help="Target platform",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("HELIX-IDS: Train Edge Models with Engineered Features")
    print("=" * 80)

    # Load data
    X_train, x_val, X_test, y_train, y_val, y_test, scaler, feature_names = load_data()
    print(
        f"\n✓ Data loaded: {len(y_train) + len(y_val) + len(y_test):,} samples, {X_train.shape[1]} features"
    )

    # Determine platforms to train
    platforms = list(PLATFORM_CONFIGS.keys()) if args.platform == "all" else [args.platform]

    results = {}

    for platform in platforms:
        model, test_f1, num_params = train_edge_model(
            platform, X_train, x_val, X_test, y_train, y_val, y_test, scaler
        )
        export_dir = export_model(model, platform, scaler, feature_names, test_f1, num_params)
        results[platform] = {"f1": test_f1, "params": num_params, "path": str(export_dir)}

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\n{'Platform':<15} {'F1 Score':>10} {'Parameters':>12} {'Path':<30}")
    print("-" * 70)
    for platform, res in results.items():
        print(f"{platform:<15} {res['f1']:>10.4f} {res['params']:>12,} {res['path']:<30}")

    print("\n✅ All edge models trained with new feature engineering pipeline")


if __name__ == "__main__":
    main()
