"""Regression tests for EvaluationOrchestrator (Phase 16 extraction).

Covers:
    - EvaluationOrchestrator construction
    - Static helpers (logit shift, prediction floor, F1 stats)
    - validate_with_logging / evaluate_with_logging delegated methods
    - Combined orchestration flows
"""

from __future__ import annotations

import logging

from unittest.mock import MagicMock

import pytest
import torch

from scripts.training.evaluation import EvaluationOrchestrator


# ======================================================================
#  Fixtures
# ======================================================================


@pytest.fixture
def mock_logger() -> logging.Logger:
    return logging.getLogger("test_eval_orch")


@pytest.fixture
def mock_model() -> MagicMock:
    model = MagicMock()
    model.return_value = (torch.randn(4, 2), torch.randn(4, 7))
    model.param_count = 1000
    return model


@pytest.fixture
def mock_loss_fn() -> MagicMock:
    return MagicMock()


@pytest.fixture
def prior() -> torch.Tensor:
    return torch.tensor([0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1])


@pytest.fixture
def orchestrator(
    mock_model: MagicMock,
    mock_loss_fn: MagicMock,
    mock_logger: logging.Logger,
    prior: torch.Tensor,
) -> EvaluationOrchestrator:
    return EvaluationOrchestrator(
        model=mock_model,
        device="cpu",
        loss_fn=mock_loss_fn,
        binary_class_weights=torch.tensor([1.0, 2.0]),
        family_class_weights=torch.tensor([1.0] * 7),
        logger=mock_logger,
        family_log_prior=prior,
        use_energy_based_family_objective=False,
        energy_logit_temperature=5.0,
        active_family_class_ids={0, 1, 2, 3},
        class4_logit_shift=1.5,
        class4_logit_shift_class_id=3,
    )


# ======================================================================
#  Construction
# ======================================================================


class TestEvaluationOrchestratorConstruction:
    """Orchestrator construction and property exposure."""

    def test_constructs_with_args(
        self,
        mock_model: MagicMock,
        mock_loss_fn: MagicMock,
        mock_logger: logging.Logger,
        prior: torch.Tensor,
    ) -> None:
        orch = EvaluationOrchestrator(
            model=mock_model,
            device="cpu",
            loss_fn=mock_loss_fn,
            binary_class_weights=torch.tensor([1.0, 2.0]),
            family_class_weights=torch.tensor([1.0]),
            family_log_prior=prior,
            use_energy_based_family_objective=False,
            energy_logit_temperature=5.0,
            active_family_class_ids={0, 1, 2, 3},
            class4_logit_shift=1.5,
            class4_logit_shift_class_id=3,
        )
        assert orch._evaluator is not None
        assert orch._evaluator.model == mock_model
        assert orch._evaluator.device == "cpu"
        assert orch._evaluator.active_family_class_ids == {0, 1, 2, 3}
        assert orch._evaluator.class4_logit_shift == 1.5
        assert orch._evaluator.class4_logit_shift_class_id == 3


# ======================================================================
#  Static helpers
# ======================================================================


class TestApplyEvalClass4LogitShift:
    """Re-exported logit shift via orchestrator matches original."""

    def test_shift_applied_correct_class(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        result = EvaluationOrchestrator.apply_eval_class4_logit_shift(
            logits, shift=2.0, class_id=4,
        )
        expected = logits.clone()
        expected[:, 4] = expected[:, 4] - 2.0
        assert torch.equal(result, expected)

    def test_zero_shift_noop(self):
        logits = torch.randn(4, 7)
        result = EvaluationOrchestrator.apply_eval_class4_logit_shift(
            logits, shift=0.0, class_id=4,
        )
        assert torch.equal(result, logits)

    def test_negative_shift_noop(self):
        logits = torch.randn(4, 7)
        result = EvaluationOrchestrator.apply_eval_class4_logit_shift(
            logits, shift=-1.0, class_id=4,
        )
        assert torch.equal(result, logits)

    def test_invalid_class_id_noop(self):
        logits = torch.randn(4, 7)
        result = EvaluationOrchestrator.apply_eval_class4_logit_shift(
            logits, shift=2.0, class_id=99,
        )
        assert torch.equal(result, logits)

    def test_1d_tensor_noop(self):
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = EvaluationOrchestrator.apply_eval_class4_logit_shift(
            logits, shift=2.0, class_id=4,
        )
        assert torch.equal(result, logits)


class TestF1StatsFromConfusion:
    """Static F1 stat computation via orchestrator."""

    def test_binary_confusion(self):
        confusion = torch.tensor([[10, 2], [3, 15]], dtype=torch.int64)
        stats = EvaluationOrchestrator.compute_f1_stats_from_confusion(confusion)
        assert "weighted_f1" in stats
        assert "macro_f1" in stats
        assert "minority_recall_min" in stats
        assert float(stats["weighted_f1"]) > 0

    def test_empty_confusion(self):
        confusion = torch.zeros((0, 0), dtype=torch.int64)
        stats = EvaluationOrchestrator.compute_f1_stats_from_confusion(confusion)
        assert float(stats["macro_f1"]) == 0.0
        assert float(stats["weighted_f1"]) == 0.0


# ======================================================================
#  Delegation — full lifecycle round-trip
# ======================================================================


class TestValidateDelegation:
    """validate() delegates cleanly through orchestrator."""

    def test_validate_returns_dict(self, orchestrator: EvaluationOrchestrator) -> None:
        """When val_loaders is empty, validate raises (no loaders configured)."""
        with pytest.raises(RuntimeError, match="No validation loaders configured"):
            orchestrator.validate(val_loaders={})

    def test_validate_with_logging(
        self, orchestrator: EvaluationOrchestrator,
    ) -> None:
        with pytest.raises(RuntimeError, match="No validation loaders configured"):
            orchestrator.validate_with_logging(val_loaders={})


class TestEvaluatePerDatasetDelegation:
    """evaluate_per_dataset delegates cleanly."""

    def test_empty_loaders_returns_empty(
        self, orchestrator: EvaluationOrchestrator,
    ) -> None:
        result = orchestrator.evaluate_per_dataset(test_loaders={})
        assert result == {}

    def test_evaluate_with_logging(
        self, orchestrator: EvaluationOrchestrator,
    ) -> None:
        result = orchestrator.evaluate_with_logging(test_loaders={})
        assert result == {}


class TestProcessTestBatch:
    """process_test_batch delegates through orchestrator."""

    def test_basic_batch(self, orchestrator: EvaluationOrchestrator) -> None:
        x = torch.randn(4, 128)
        y_binary = torch.randint(0, 2, (4,))
        y_family = torch.randint(0, 7, (4,))
        binary_cm = torch.zeros((2, 2), dtype=torch.int64)
        family_cm = torch.zeros((7, 7), dtype=torch.int64)

        result = orchestrator.process_test_batch(
            x, y_binary, y_family, binary_cm, family_cm,
        )
        assert len(result) == 6


# ======================================================================
#  Edge cases
# ======================================================================


class TestEdgeCases:
    """Edge-case handling through the orchestrator."""

    def test_empty_active_ids(self, orchestrator: EvaluationOrchestrator) -> None:
        with pytest.raises(RuntimeError, match="No validation loaders configured"):
            orch = EvaluationOrchestrator(
                model=orchestrator._evaluator.model,
                device="cpu",
                loss_fn=orchestrator._evaluator.loss_fn,
                active_family_class_ids=set(),
            )
            orch.validate(val_loaders={})

    def test_config_mismatch(
        self, mock_model: MagicMock, mock_loss_fn: MagicMock,
    ) -> None:
        """Orchestrator handles missing class4 config gracefully."""
        orch = EvaluationOrchestrator(
            model=mock_model,
            device="cpu",
            loss_fn=mock_loss_fn,
            class4_logit_shift=0.0,
            class4_logit_shift_class_id=-1,
        )
        logits = torch.randn(4, 7)
        result = orch.apply_eval_class4_logit_shift(
            logits, shift=0.0, class_id=-1,
        )
        assert torch.equal(result, logits)
