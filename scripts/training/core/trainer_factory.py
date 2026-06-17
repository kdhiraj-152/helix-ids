"""
trainer_factory: Construction helpers for HelixFullTrainer and its core subsystem.

Phase 18: Provides factory methods for creating the trainer with its
state, facade, and recovery manager from configuration objects.

Usage::

    from scripts.training.core.trainer_factory import TrainerFactory
    factory = TrainerFactory(model, train_loader, ..., config=config)
    trainer = factory.build_trainer()
"""

from __future__ import annotations

import logging

import torch
from torch import optim
from torch.utils.data import DataLoader

from helix_ids.config.helix_full_config import TrainingConfig
from helix_ids.models.full import HelixIDSFull, MultiTaskLoss
from scripts.training.core.trainer_facade import TrainerFacade
from scripts.training.core.trainer_state import TrainerState


class TrainerFactory:
    """Factory for constructing HelixFullTrainer and its core subsystems."""

    def __init__(
        self,
        model: HelixIDSFull,
        train_loader: DataLoader,
        val_loaders: dict[str, DataLoader],
        test_loaders: dict[str, DataLoader],
        optimizer: optim.Optimizer,
        loss_fn: MultiTaskLoss,
        config: TrainingConfig,
        binary_class_weights: torch.Tensor | None = None,
        family_class_weights: torch.Tensor | None = None,
        train_family_class_count: int | None = None,
        run_seed: int = 42,
        device: str = "mps",
        logger: logging.Logger | None = None,
    ) -> None:
        # Store each parameter as a typed attribute for type-safe forwarding
        self._model: HelixIDSFull = model
        self._train_loader: DataLoader = train_loader
        self._val_loaders: dict[str, DataLoader] = val_loaders
        self._test_loaders: dict[str, DataLoader] = test_loaders
        self._optimizer: optim.Optimizer = optimizer
        self._loss_fn: MultiTaskLoss = loss_fn
        self._config: TrainingConfig = config
        self._binary_class_weights: torch.Tensor | None = binary_class_weights
        self._family_class_weights: torch.Tensor | None = family_class_weights
        self._train_family_class_count: int | None = train_family_class_count
        self._run_seed: int = run_seed
        self._device: str = device
        self._logger: logging.Logger | None = logger

    def build_state(self) -> TrainerState:
        """Build TrainerState from constructor parameters."""
        return TrainerState(
            model=self._model,
            train_loader=self._train_loader,
            val_loaders=self._val_loaders,
            test_loaders=self._test_loaders,
            optimizer=self._optimizer,
            loss_fn=self._loss_fn,
            config=self._config,
            binary_class_weights=self._binary_class_weights,
            family_class_weights=self._family_class_weights,
            train_family_class_count=self._train_family_class_count,
            run_seed=self._run_seed,
            device=self._device,
            logger=self._logger,
        )

    def build_facade(self, state: TrainerState) -> TrainerFacade:
        """Build and wire TrainerFacade from state."""
        return TrainerFacade(state).build()
