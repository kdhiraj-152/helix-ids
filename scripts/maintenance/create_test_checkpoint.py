#!/usr/bin/env python3
"""Create a test checkpoint for Phase 4 quantization testing."""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from helix_ids.models.full import HelixFullConfig, HelixIDSFull

if __name__ == "__main__":
    # Create model
    config = HelixFullConfig()
    model = HelixIDSFull(config)
    model.eval()

    # Save checkpoint
    checkpoint_dir = Path("models/helix_full")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "helix_full_best.pt"

    torch.save(model.state_dict(), checkpoint_path)
    print(f"✅ Test checkpoint created: {checkpoint_path}")
    print(f"   Model parameters: {model.param_count:,}")
