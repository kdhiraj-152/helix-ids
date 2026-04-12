"""Determinism and global seed enforcement helpers."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class DeterminismState:
    """Captured deterministic runtime settings for lineage metadata."""

    seed: int
    torch_deterministic_algorithms: bool
    torch_cudnn_deterministic: bool
    torch_cudnn_benchmark: bool
    python_hash_seed: str

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "torch_deterministic_algorithms": self.torch_deterministic_algorithms,
            "torch_cudnn_deterministic": self.torch_cudnn_deterministic,
            "torch_cudnn_benchmark": self.torch_cudnn_benchmark,
            "python_hash_seed": self.python_hash_seed,
        }


def set_global_determinism(seed: int) -> DeterminismState:
    """Set deterministic execution across Python, NumPy, and PyTorch backends."""
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Safe even on non-CUDA backends.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return DeterminismState(
        seed=seed,
        torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        torch_cudnn_deterministic=torch.backends.cudnn.deterministic,
        torch_cudnn_benchmark=torch.backends.cudnn.benchmark,
        python_hash_seed=os.environ.get("PYTHONHASHSEED", ""),
    )


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader worker processes deterministically."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
