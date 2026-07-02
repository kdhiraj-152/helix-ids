"""Tests for strict evaluation schema in direct adaptation runner."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Archived under archive/phase24a/scripts/training/ (Phase 24A)
ARCHIVE_SCRIPTS = Path(__file__).resolve().parent.parent / "archive" / "phase24a"
sys.path.insert(0, str(ARCHIVE_SCRIPTS))

from scripts.training import train_unified_rebalanced as runner  # noqa: E402


class _DummyLoader:
    def load(self, _dataset_name: str, return_class_names: bool = True):
        x = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.5, 0.5],
                [0.2, 0.8],
            ],
            dtype=np.float32,
        )
        y = np.array([0, 1, 2, 1], dtype=np.int64)
        class_names = ["normal", "dos", "probe"]
        return x, y, class_names


class _DummyPretrainer:
    def predict(self, x: np.ndarray, dataset_name: str | None = None) -> np.ndarray:
        del x, dataset_name
        return np.array([0, 1, 1, 1], dtype=np.int64)


def test_dataset_eval_contains_per_class_and_confusion(monkeypatch):
    monkeypatch.setattr(runner, "UnifiedDataLoader", lambda **kwargs: _DummyLoader())

    result = runner._dataset_eval(_DummyPretrainer(), "nsl-kdd")

    assert result["samples"] == 4
    assert "macro_f1" in result
    assert "weighted_f1" in result
    assert "per_class_f1" in result
    assert "per_class_precision" in result
    assert "per_class_recall" in result
    assert "per_class_support" in result
    assert "confusion_matrix" in result
    assert result["predicted_class_count"] == 2

    confusion = np.asarray(result["confusion_matrix"], dtype=np.int64)
    assert confusion.shape == (3, 3)
