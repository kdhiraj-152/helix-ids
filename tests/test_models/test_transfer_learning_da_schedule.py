"""Tests for staged DA schedule and class-conditional MMD wiring."""

import numpy as np
import pytest

from src.helix_ids.models.adaptation.transfer_learning import (
    MultiDatasetPretrainer,
    TransferLearningConfig,
)


def test_staged_da_weight_schedule_matches_policy():
    config = TransferLearningConfig(
        use_staged_schedule=True,
        cls_only_last_epoch=3,
        full_da_start_epoch=10,
    )
    trainer = MultiDatasetPretrainer(config)

    assert trainer._get_da_weight(0) == pytest.approx(0.0)
    assert trainer._get_da_weight(3) == pytest.approx(0.0)

    ramp_weight = trainer._get_da_weight(6)
    assert 0.0 < ramp_weight < 1.0

    assert trainer._get_da_weight(10) == pytest.approx(1.0)
    assert trainer._get_da_weight(25) == pytest.approx(1.0)


def test_staged_schedule_can_be_disabled():
    config = TransferLearningConfig(use_staged_schedule=False)
    trainer = MultiDatasetPretrainer(config)

    assert trainer._get_da_weight(0) == pytest.approx(1.0)
    assert trainer._get_da_weight(5) == pytest.approx(1.0)
    assert trainer._get_da_weight(100) == pytest.approx(1.0)


def test_class_conditional_mmd_initialization_toggle():
    enabled_config = TransferLearningConfig(use_class_conditional_mmd=True)
    enabled = MultiDatasetPretrainer(enabled_config)
    enabled._initialize_models({"nsl-kdd": 41, "unsw-nb15": 47})
    assert enabled._class_mmd_loss is not None

    disabled_config = TransferLearningConfig(use_class_conditional_mmd=False)
    disabled = MultiDatasetPretrainer(disabled_config)
    disabled._initialize_models({"nsl-kdd": 41, "unsw-nb15": 47})
    assert disabled._class_mmd_loss is None


def test_class_weight_computation_handles_imbalance():
    y = np.array([0, 0, 0, 0, 1, 2], dtype=np.int64)
    weights = MultiDatasetPretrainer._compute_class_weights(y, num_classes=5)

    assert weights.shape[0] == 5
    assert weights[0] < weights[1]
    assert weights[0] < weights[2]
    assert weights[3] == pytest.approx(0.0)
    assert weights[4] == pytest.approx(0.0)


def test_expected_domain_chance_matches_three_domain_target():
    assert MultiDatasetPretrainer._expected_domain_chance_acc(3) == pytest.approx(1.0 / 3.0)
    assert MultiDatasetPretrainer._expected_domain_chance_acc(1) == pytest.approx(1.0)
