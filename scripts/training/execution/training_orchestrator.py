"""Training orchestrator — top-level fit() lifecycle.

Calls the trainer's train_epoch(), validate(), and other callbacks.
This is a thin coordinator that orchestrates the training loop
without owning any execution logic.
"""

from typing import Any, Callable, Optional

import torch

from scripts.training.execution.epoch_runner import EpochRunner


class TrainingOrchestrator:
    """Coordinates the full training lifecycle via injected callbacks.

    Parameters are received at ``fit()`` time, not at construction,
    so the orchestrator stays decoupled from ``HelixFullTrainer``.
    """

    def __init__(
        self,
        *,
        epoch_runner: EpochRunner,
        logger: Any = None,
        config: Any = None,
    ) -> None:
        self._epoch_runner = epoch_runner
        self._logger = logger
        self._config = config

    def set_logger(self, logger: Any) -> None:
        """Override the stored logger (useful when ``fit()`` re-binds)."""
        self._logger = logger

    def fit(
        self,
        # --- config ---
        epochs: int,
        val_interval: int,
        freeze_backbone_epochs: int,
        model: torch.nn.Module,
        device: torch.device,
        # --- trainer-side callbacks ---
        reseed_generators: Callable[[], None],
        set_backbone_freeze_state: Callable[[bool], None],
        set_learning_rate: Callable[[], None],
        train_epoch: Callable[[], dict[str, float]],
        validate: Callable[[], dict[str, float]],
        log_per_dataset_results: Callable[[dict], None],
        post_training_macro_floor: Callable[[], float],
        evaluate_per_dataset: Callable[[], dict],
        hard_stop_reason: Callable[[dict, dict], Optional[str]],
        update_early_stopping: Callable[[dict, dict], bool],
        save_checkpoint: Callable[[], None],
        # --- validation callbacks (orchestrator-level) ---
        detect_coverage_collapse: Callable[[dict, int], bool],
        check_zero_prediction_classes: Callable[[dict], None],
        check_per_dataset_macro_floor: Callable[[dict, float], None],
        # --- trainer state ---
        epoch: int,
        training_history: dict[str, list[float]],
        best_model_state: Optional[dict],
        best_val_loss: float,
        representation_diagnostics: dict[str, Any],
        train_family_class_count: int,
        use_energy_based_family_objective: bool,
        disable_integrity_hard_stops: bool,
        logger: Any,
    ) -> dict[str, Any]:
        """Run the full training loop and return results.

        Returns the same dict shape as the original ``HelixFullTrainer.fit()``.
        """
        self.set_logger(logger)
        self._logger.info("=" * 80)
        self._logger.info("Starting HelixIDS-Full Training")
        self._logger.info(f"Device: {device}")
        self._logger.info(f"Model parameters: {model.param_count:,}")
        # Use self._config for batch size if available, otherwise default
        batch_size = getattr(self._config, "batch_size", "?")
        self._logger.info(f"Epochs: {epochs}")
        self._logger.info(f"Batch size: {batch_size}")
        self._logger.info("=" * 80)

        local_epoch = epoch
        local_training_history: dict[str, list[float]] = (
            dict(training_history) if training_history else {}
        )
        local_best_val_loss = best_val_loss
        local_representation_diagnostics: dict[str, Any] = (
            dict(representation_diagnostics) if representation_diagnostics else {}
        )

        for epoch_idx in range(epochs):
            local_epoch = epoch_idx
            reseed_generators()
            if not getattr(model, "representation_diagnostic_mode", False):
                set_backbone_freeze_state(epoch_idx < freeze_backbone_epochs)
            set_learning_rate()

            # --- Train epoch (delegates through trainer which handles warmup) ---
            train_metrics = train_epoch()

            # --- Validate every N epochs ---
            if epoch_idx % val_interval == 0:
                val_metrics = validate()

                for key, val in train_metrics.items():
                    local_training_history.setdefault(key, []).append(val)
                for key, val in val_metrics.items():
                    local_training_history.setdefault(key, []).append(val)

                self._logger.info(
                    f"Epoch {epoch_idx:3d} | "
                    f"Train Loss: {train_metrics['train_loss']:.4f} | "
                    f"Train Cal Loss: "
                    f"{train_metrics['train_calibrated_loss']:.4f} | "
                    f"Train Logit Range: "
                    f"[{train_metrics.get('train_family_logit_min', 0.0):.4f}, "
                    f"{train_metrics.get('train_family_logit_max', 0.0):.4f}] | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Val Cal Loss: "
                    f"{val_metrics['val_calibrated_loss']:.4f} | "
                    f"Val Binary Acc: "
                    f"{val_metrics['val_binary_acc']:.4f} | "
                    f"Val Family Acc: "
                    f"{val_metrics['val_family_acc']:.4f} | "
                    f"Val Entropy: "
                    f"{val_metrics.get('val_family_entropy', 0.0):.4f}"
                )

                detect_coverage_collapse(
                    val_metrics, train_family_class_count,
                )
                check_zero_prediction_classes(val_metrics)

                hsr = hard_stop_reason(train_metrics, val_metrics)
                if hsr is not None:
                    raise RuntimeError(
                        f"Hard-stop integrity guard triggered: {hsr}",
                    )

                should_stop = update_early_stopping(
                    train_metrics, val_metrics,
                )
                save_checkpoint()
                if should_stop:
                    break

        # --- Post-training ---
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            self._logger.info("Loaded best model state")

        per_dataset_results = evaluate_per_dataset()
        log_per_dataset_results(per_dataset_results)
        macro_floor = post_training_macro_floor()
        check_per_dataset_macro_floor(per_dataset_results, macro_floor)

        return {
            "training_history": local_training_history,
            "per_dataset_results": per_dataset_results,
            "representation_diagnostics": local_representation_diagnostics,
            "best_val_loss": local_best_val_loss,
            "epochs_trained": local_epoch + 1,
        }
