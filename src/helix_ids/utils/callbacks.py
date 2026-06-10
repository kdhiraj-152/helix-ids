"""
HELIX-IDS Training Callbacks Module

Provides training infrastructure callbacks for PyTorch-based training:
- EarlyStopping: Stop training when metrics plateau
- ModelCheckpoint: Save best models during training
- LearningRateScheduler: Dynamic learning rate adjustment
- TrainingLogger: Console, file, and TensorBoard logging
- CallbackList: Container for managing multiple callbacks
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

import torch
import torch.optim as optim

from ..contracts.schema_contract import runtime_contract_payload
from ..governance.provenance import (
    ARTIFACT_MANIFEST_KEY,
    checkpoint_manifest_payload,
    write_contract_sidecars,
)
from .export import build_export_manifest, finalize_export_artifact, verify_export_artifact

# Setup module logger
logger = logging.getLogger(__name__)


class Callback:
    """
    Base class for training callbacks.

    Callbacks receive notifications at various points during training
    and can perform actions like saving models, adjusting learning rates,
    or stopping training early.
    """

    def __init__(self):
        self.model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None

    def set_model(self, model: torch.nn.Module) -> None:
        """Set the model reference."""
        self.model = model

    def set_optimizer(self, optimizer: optim.Optimizer) -> None:
        """Set the optimizer reference."""
        self.optimizer = optimizer

    def on_train_begin(self, _logs: Optional[dict[str, Any]] = None) -> None:
        """Called at the start of training."""
        pass

    def on_train_end(self, _logs: Optional[dict[str, Any]] = None) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, _logs: Optional[dict[str, Any]] = None) -> None:
        """Called at the start of an epoch."""
        pass

    def on_epoch_end(self, epoch: int, _logs: Optional[dict[str, Any]] = None) -> bool:
        """
        Called at the end of an epoch.

        Args:
            epoch: Current epoch number (0-indexed)
            _logs: Dictionary containing metrics from training/validation
        """
        return True

    def on_batch_begin(self, batch: int, _logs: Optional[dict[str, Any]] = None) -> None:
        """Called at the start of a batch."""
        pass

    def on_batch_end(self, batch: int, _logs: Optional[dict[str, Any]] = None) -> None:
        """Called at the end of a batch."""
        pass


class EarlyStopping(Callback):
    """
    Stop training when a monitored metric stops improving.

    Monitors a specified metric and stops training if it doesn't improve
    for a given number of epochs (patience).

    Args:
        monitor: Metric name to monitor (default: 'threat_weighted_f1')
        patience: Number of epochs with no improvement before stopping
        min_delta: Minimum change to qualify as an improvement
        mode: 'max' for metrics to maximize (e.g., F1), 'min' for metrics to minimize (e.g., loss)
        baseline: Baseline value; training stops if metric doesn't improve over baseline
        restore_best_weights: Whether to restore model weights from best epoch
        verbose: Whether to print messages

    Example:
        >>> early_stop = EarlyStopping(
        ...     monitor='threat_weighted_f1',
        ...     patience=15,
        ...     mode='max',
        ...     restore_best_weights=True
        ... )
        >>> # In training loop:
        >>> should_continue = early_stop.on_epoch_end(epoch, {'threat_weighted_f1': 0.85})
    """

    def __init__(
        self,
        monitor: str = "threat_weighted_f1",
        patience: int = 15,
        min_delta: float = 0.0,
        mode: Literal["max", "min"] = "max",
        baseline: Optional[float] = None,
        restore_best_weights: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        # State
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights: Optional[dict[str, torch.Tensor]] = None
        self.best_epoch = 0

        # Setup comparison based on mode
        if mode == "max":
            self.monitor_op = lambda a, b: a > b + min_delta
            self.best = float("-inf") if baseline is None else baseline
        else:
            self.monitor_op = lambda a, b: a < b - min_delta
            self.best = float("inf") if baseline is None else baseline

    def on_train_begin(self, logs: Optional[dict[str, Any]] = None) -> None:
        """Reset state at training start."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_epoch = 0

        if self.mode == "max":
            self.best = float("-inf") if self.baseline is None else self.baseline
        else:
            self.best = float("inf") if self.baseline is None else self.baseline

    def on_epoch_end(self, epoch: int, logs: Optional[dict[str, Any]] = None) -> bool:  # NOSONAR
        """
        Check if training should stop.

        Returns:
            False to stop training, True to continue
        """
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            logger.warning(
                f"EarlyStopping: metric '{self.monitor}' not found in logs. "
                f"Available: {list(logs.keys())}"
            )
            return True

        # Check if metric improved
        improved = self.monitor_op(current, self.best)
        if improved:
            self._handle_improvement(epoch, current)
        else:
            self._handle_no_improvement(epoch)

        return self.wait < self.patience  # Continue if patience not exceeded

    def _handle_improvement(self, epoch: int, current: float) -> None:
        """Handle metric improvement."""
        self.best = current
        self.best_epoch = epoch
        self.wait = 0

        # Save best weights
        if self.restore_best_weights and self.model is not None:
            self.best_weights = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}

    def _handle_no_improvement(self, epoch: int) -> None:
        """Handle metric not improving."""
        self.wait += 1

        if self.wait >= self.patience:
            self.stopped_epoch = epoch

            if self.verbose:
                print(
                    f"\nEarlyStopping: Training stopped at epoch {epoch + 1}. "
                    f"Best {self.monitor}: {self.best:.4f} at epoch {self.best_epoch + 1}"
                )

            # Restore best weights
            if (
                self.restore_best_weights
                and self.best_weights is not None
                and self.model is not None
            ):
                if self.verbose:
                    print(f"Restoring model weights from epoch {self.best_epoch + 1}")
                self.model.load_state_dict(self.best_weights)

    def get_best_metric(self) -> float:
        """Return the best metric value observed."""
        return self.best

    def get_best_epoch(self) -> int:
        """Return the epoch with the best metric (0-indexed)."""
        return self.best_epoch


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.

    Monitors a metric and saves the model when the metric improves.
    Supports formatting the filepath with epoch and metric values.

    Args:
        filepath: Path to save the model. Can include formatting placeholders:
            - {epoch}: Current epoch number
            - {metric}: Current metric value
            - {monitor}: Name of monitored metric
        monitor: Metric name to monitor
        mode: 'max' or 'min' for the monitored metric
        save_best_only: If True, only save when metric improves
        save_weights_only: If True, only save model state_dict (not full checkpoint)
        verbose: Whether to print messages when saving

    Example:
        >>> checkpoint = ModelCheckpoint(
        ...     filepath='models/helix_{epoch:03d}_{threat_weighted_f1:.4f}.pt',
        ...     monitor='threat_weighted_f1',
        ...     save_best_only=True,
        ...     mode='max'
        ... )
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        monitor: str = "threat_weighted_f1",
        mode: Literal["max", "min"] = "max",
        save_best_only: bool = True,
        save_weights_only: bool = False,
        verbose: bool = True,
    ):
        super().__init__()
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose

        # State
        if mode == "max":
            self.best = float("-inf")
            self.monitor_op = lambda a, b: a > b
        else:
            self.best = float("inf")
            self.monitor_op = lambda a, b: a < b

        self.best_filepath: Optional[Path] = None

    def on_train_begin(self, logs: Optional[dict[str, Any]] = None) -> None:
        """Ensure save directory exists."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        if self.mode == "max":
            self.best = float("-inf")
        else:
            self.best = float("inf")

    def on_epoch_end(self, epoch: int, logs: Optional[dict[str, Any]] = None) -> bool:  # NOSONAR
        """
        Check if model should be saved.

        Returns:
            True (always continues training)
        """
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            logger.warning(f"ModelCheckpoint: metric '{self.monitor}' not found in logs")
            return True

        # Determine if we should save
        should_save = (not self.save_best_only) or self.monitor_op(current, self.best)

        if should_save:
            if self.monitor_op(current, self.best):
                self.best = current
            self._save_model(epoch, current, logs)

        return True  # Always continue training

    def _save_model(self, epoch: int, metric_value: float, logs: dict[str, Any]) -> None:
        """Save the model checkpoint."""
        if self.model is None:
            logger.warning("ModelCheckpoint: No model set, cannot save")
            return

        # Format filepath
        filepath_str = str(self.filepath)
        filepath_str = filepath_str.replace("{epoch}", f"{epoch + 1:03d}")
        filepath_str = filepath_str.replace("{metric}", f"{metric_value:.4f}")
        filepath_str = filepath_str.replace("{monitor}", self.monitor)

        # Also support f-string style formatting for metric names
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                filepath_str = filepath_str.replace(f"{{{key}}}", f"{value:.4f}")
                filepath_str = filepath_str.replace(f"{{{key}:.4f}}", f"{value:.4f}")

        save_path = Path(filepath_str)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if self.save_weights_only:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "metrics": logs,
                "best_metric": self.best,
                "monitor": self.monitor,
            }
        else:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "metrics": logs,
                "best_metric": self.best,
                "monitor": self.monitor,
            }

            if self.optimizer is not None:
                checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()

        # Attach canonical runtime contract to the checkpoint payload
        contract_payload = runtime_contract_payload()
        checkpoint.update(contract_payload)

        # Build a provenance manifest (without artifact sha256 yet) and embed it into the
        # checkpoint payload so that the artifact carries its provenance.
        manifest_base = build_export_manifest(
            contract=contract_payload,
            model_architecture=self.model.__class__.__name__,
            export_config={"format": "checkpoint"},
        )

        # Embed the manifest (without artifact_sha256) in the saved payload
        checkpoint[ARTIFACT_MANIFEST_KEY] = checkpoint_manifest_payload(manifest_base)

        # Persist the checkpoint file
        torch.save(checkpoint, save_path)

        # Write traditional contract sidecars for compatibility and human inspection
        sidecars = write_contract_sidecars(save_path, contract_payload)

        # Finalize the manifest (compute and write artifact sha256 sidecar)
        finalize_export_artifact(save_path, manifest_base, sidecars=sidecars)

        # Verify provenance immediately to catch save-time inconsistencies
        try:
            verify_export_artifact(
                save_path,
                kind="checkpoint",
                contract=contract_payload,
                embedded_manifest=checkpoint_manifest_payload(manifest_base),
            )
        except Exception:  # pragma: no cover - defensive check
            logger.exception("ModelCheckpoint: saved artifact failed provenance verification")
            raise

        self.best_filepath = save_path

        if self.verbose:
            print(
                f"ModelCheckpoint: Saved model to {save_path} ({self.monitor}={metric_value:.4f})"
            )

    def get_best_filepath(self) -> Optional[Path]:
        """Return the path to the best saved model."""
        return self.best_filepath


class LearningRateScheduler(Callback):
    """
    Adjust learning rate during training.

    Supports multiple scheduling strategies:
    - warmup: Gradual increase from initial LR
    - cosine: Cosine annealing decay
    - step: Step decay at specific epochs
    - custom: User-provided schedule function

    Args:
        schedule_type: Type of schedule ('warmup_cosine', 'step', 'custom')
        initial_lr: Starting learning rate (required for warmup_cosine and step)
        warmup_epochs: Number of warmup epochs (for warmup_cosine)
        total_epochs: Total training epochs (for warmup_cosine)
        min_lr: Minimum learning rate (for warmup_cosine)
        step_size: Epochs between LR reductions (for step)
        gamma: LR multiplication factor (for step)
        milestones: Epochs at which to reduce LR (for step with milestones)
        schedule_fn: Custom function(epoch, initial_lr) -> lr (for custom)
        verbose: Whether to print LR changes

    Example:
        >>> # Warmup + cosine decay
        >>> scheduler = LearningRateScheduler(
        ...     schedule_type='warmup_cosine',
        ...     initial_lr=1e-3,
        ...     warmup_epochs=5,
        ...     total_epochs=100,
        ...     min_lr=1e-6
        ... )
        >>>
        >>> # Step decay
        >>> scheduler = LearningRateScheduler(
        ...     schedule_type='step',
        ...     initial_lr=1e-3,
        ...     milestones=[30, 60, 90],
        ...     gamma=0.1
        ... )
    """

    def __init__(
        self,
        schedule_type: Literal["warmup_cosine", "step", "custom"] = "warmup_cosine",
        initial_lr: float = 1e-3,
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        min_lr: float = 1e-6,
        step_size: int = 30,
        gamma: float = 0.1,
        milestones: Optional[list[int]] = None,
        schedule_fn: Optional[Callable[[int, float], float]] = None,
        verbose: bool = True,
    ):
        super().__init__()
        self.schedule_type = schedule_type
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.step_size = step_size
        self.gamma = gamma
        self.milestones = milestones or []
        self.schedule_fn = schedule_fn
        self.verbose = verbose

        # State
        self.current_lr = initial_lr
        self.lr_history: list[float] = []

    def _compute_lr(self, epoch: int) -> float:
        """Compute learning rate for the given epoch."""
        if self.schedule_type == "warmup_cosine":
            return self._warmup_cosine_lr(epoch)
        elif self.schedule_type == "step":
            return self._step_lr(epoch)
        elif self.schedule_type == "custom" and self.schedule_fn is not None:
            return self.schedule_fn(epoch, self.initial_lr)
        else:
            return self.initial_lr

    def _warmup_cosine_lr(self, epoch: int) -> float:
        """Compute warmup + cosine annealing learning rate."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            return self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay

    def _step_lr(self, epoch: int) -> float:
        """Compute step decay learning rate."""
        lr = self.initial_lr

        if self.milestones:
            # Use milestones
            for milestone in self.milestones:
                if epoch >= milestone:
                    lr *= self.gamma
        else:
            # Use step_size
            lr *= self.gamma ** (epoch // self.step_size)

        return max(lr, self.min_lr)

    def on_epoch_begin(self, epoch: int, logs: Optional[dict[str, Any]] = None) -> None:
        """Update learning rate at the start of each epoch."""
        new_lr = self._compute_lr(epoch)

        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr

        if self.verbose and abs(new_lr - self.current_lr) > 1e-8:
            print(
                f"LearningRateScheduler: Epoch {epoch + 1}, LR: {self.current_lr:.6f} -> {new_lr:.6f}"
            )

        self.current_lr = new_lr
        self.lr_history.append(new_lr)

    def get_lr_history(self) -> list[float]:
        """Return the learning rate history."""
        return self.lr_history.copy()


class TrainingLogger(Callback):
    """
    Log training metrics to console, file, and optionally TensorBoard.

    Args:
        log_dir: Directory for log files
        log_filename: Name of JSON log file
        console_format: Format string for console output
        log_every_n_epochs: Log to console every N epochs
        tensorboard: Whether to use TensorBoard (requires tensorboard package)
        verbose: Whether to print to console

    Example:
        >>> logger = TrainingLogger(
        ...     log_dir='logs/helix_training',
        ...     tensorboard=True,
        ...     log_every_n_epochs=1
        ... )
    """

    def __init__(
        self,
        log_dir: Union[str, Path] = "logs",
        log_filename: str = "training_log.json",
        console_format: str = "default",
        log_every_n_epochs: int = 1,
        tensorboard: bool = False,
        verbose: bool = True,
    ):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_filename = log_filename
        self.console_format = console_format
        self.log_every_n_epochs = log_every_n_epochs
        self.use_tensorboard = tensorboard
        self.verbose = verbose

        # State
        self.history: list[dict[str, Any]] = []
        self.start_time: Optional[datetime] = None
        self.tb_writer: Any | None = None

        # TensorBoard setup (lazy import)
        self._tb_available = False

    def _setup_tensorboard(self) -> bool:
        """Attempt to set up TensorBoard writer."""
        if not self.use_tensorboard:
            return False

        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
            self._tb_available = True
            return True
        except ImportError:
            logger.warning(
                "TensorBoard requested but not installed. Install with: pip install tensorboard"
            )
            return False

    def on_train_begin(self, logs: Optional[dict[str, Any]] = None) -> None:
        """Setup logging infrastructure."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = []
        self.start_time = datetime.now()

        if self.use_tensorboard:
            self._setup_tensorboard()

        if self.verbose:
            print("\n" + "=" * 70)
            print("HELIX-IDS TRAINING")
            print("=" * 70)
            print(f"Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Log directory: {self.log_dir}")
            if self._tb_available:
                print(f"TensorBoard: {self.log_dir / 'tensorboard'}")
            print("-" * 70)

    def on_epoch_end(self, epoch: int, logs: Optional[dict[str, Any]] = None) -> bool:
        """Log metrics at the end of each epoch."""
        logs = logs or {}

        # Add epoch and timestamp
        log_entry = {"epoch": epoch + 1, "timestamp": datetime.now().isoformat(), **logs}
        self.history.append(log_entry)

        # Console output
        if self.verbose and (epoch + 1) % self.log_every_n_epochs == 0:
            self._print_epoch(epoch, logs)

        # TensorBoard logging
        if self._tb_available and self.tb_writer is not None:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, epoch + 1)

        # JSON file logging (update after each epoch for crash recovery)
        self._save_json_log()

        return True  # Continue training

    def _print_epoch(self, epoch: int, logs: dict[str, Any]) -> None:
        """Print formatted epoch summary."""
        if self.console_format == "default":
            # Default HELIX-IDS format
            loss = logs.get("train_loss", logs.get("loss", 0))
            acc = logs.get("val_accuracy", logs.get("accuracy", 0))
            macro_f1 = logs.get("val_macro_f1", logs.get("macro_f1", 0))
            threat_f1 = logs.get("val_threat_f1", logs.get("threat_weighted_f1", 0))
            r2l_f1 = logs.get("val_r2l_f1", logs.get("r2l_f1", 0))
            u2r_f1 = logs.get("val_u2r_f1", logs.get("u2r_f1", 0))
            _ = logs.get("lr", logs.get("learning_rate", 0)) or 0

            print(
                f"Epoch {epoch + 1:>4} | "
                f"Loss: {loss:.4f} | "
                f"Acc: {acc:.4f} | "
                f"F1: {macro_f1:.4f} | "
                f"Threat-F1: {threat_f1:.4f} | "
                f"R2L: {r2l_f1:.4f} | "
                f"U2R: {u2r_f1:.4f}"
            )
        else:
            # Simple format
            metrics_str = " | ".join(
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in logs.items()
            )
            print(f"Epoch {epoch + 1}: {metrics_str}")

    def _save_json_log(self) -> None:
        """Save history to JSON file."""
        log_path = self.log_dir / self.log_filename
        with open(log_path, "w") as f:
            json.dump(
                {
                    "start_time": self.start_time.isoformat() if self.start_time else None,
                    "history": self.history,
                },
                f,
                indent=2,
            )

    def on_train_end(self, logs: Optional[dict[str, Any]] = None) -> None:
        """Finalize logging."""
        end_time = datetime.now()
        duration = end_time - self.start_time if self.start_time else None

        # Close TensorBoard writer
        if self.tb_writer is not None:
            self.tb_writer.close()

        # Final JSON save with summary
        log_path = self.log_dir / self.log_filename
        with open(log_path, "w") as f:
            json.dump(
                {
                    "start_time": self.start_time.isoformat() if self.start_time else None,
                    "end_time": end_time.isoformat(),
                    "duration_seconds": duration.total_seconds() if duration else None,
                    "total_epochs": len(self.history),
                    "history": self.history,
                },
                f,
                indent=2,
            )

        if self.verbose:
            print("-" * 70)
            print(f"Training completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            if duration:
                print(f"Total duration: {duration}")
            print(f"Log saved to: {log_path}")
            print("=" * 70)

    def get_history(self) -> list[dict[str, Any]]:
        """Return the training history."""
        return self.history.copy()


class CallbackList:
    """
    Container for managing multiple callbacks.

    Delegates callback method calls to all registered callbacks
    and handles stopping conditions.

    Args:
        callbacks: List of Callback objects

    Example:
        >>> callbacks = CallbackList([
        ...     EarlyStopping(patience=10),
        ...     ModelCheckpoint('models/best.pt'),
        ...     TrainingLogger('logs/')
        ... ])
        >>>
        >>> callbacks.set_model(model)
        >>> callbacks.set_optimizer(optimizer)
        >>>
        >>> callbacks.on_train_begin()
        >>> for epoch in range(num_epochs):
        ...     # ... training code ...
        ...     should_continue = callbacks.on_epoch_end(epoch, logs)
        ...     if not should_continue:
        ...         break
        >>> callbacks.on_train_end()
    """

    def __init__(self, callbacks: Optional[list[Callback]] = None):
        self.callbacks = callbacks or []

    def append(self, callback: Callback) -> None:
        """Add a callback to the list."""
        self.callbacks.append(callback)

    def extend(self, callbacks: list[Callback]) -> None:
        """Add multiple callbacks to the list."""
        self.callbacks.extend(callbacks)

    def set_model(self, model: torch.nn.Module) -> None:
        """Set model reference for all callbacks."""
        for callback in self.callbacks:
            callback.set_model(model)

    def set_optimizer(self, optimizer: optim.Optimizer) -> None:
        """Set optimizer reference for all callbacks."""
        for callback in self.callbacks:
            callback.set_optimizer(optimizer)

    def on_train_begin(self, logs: Optional[dict[str, Any]] = None) -> None:
        """Call on_train_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs: Optional[dict[str, Any]] = None) -> None:
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[dict[str, Any]] = None) -> None:
        """Call on_epoch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[dict[str, Any]] = None) -> bool:
        """
        Call on_epoch_end for all callbacks.

        Returns:
            False if any callback returns False (stop training), True otherwise
        """
        should_continue = True
        for callback in self.callbacks:
            result = callback.on_epoch_end(epoch, logs)
            if result is False:
                should_continue = False
        return should_continue

    def on_batch_begin(self, batch: int, logs: Optional[dict[str, Any]] = None) -> None:
        """Call on_batch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch: int, logs: Optional[dict[str, Any]] = None) -> None:
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def __iter__(self):
        """Iterate over callbacks."""
        return iter(self.callbacks)

    def __len__(self) -> int:
        """Return number of callbacks."""
        return len(self.callbacks)


# Convenience function to create common callback configurations
def create_helix_callbacks(
    output_dir: Union[str, Path],
    monitor: str = "threat_weighted_f1",
    patience: int = 15,
    initial_lr: float = 1e-3,
    total_epochs: int = 100,
    warmup_epochs: int = 5,
    use_tensorboard: bool = False,
    verbose: bool = True,
) -> CallbackList:
    """
    Create a standard set of callbacks for HELIX-IDS training.

    Args:
        output_dir: Directory for saving models and logs
        monitor: Metric to monitor for early stopping and checkpointing
        patience: Early stopping patience
        initial_lr: Initial learning rate for scheduler
        total_epochs: Total training epochs for LR scheduler
        warmup_epochs: Warmup epochs for LR scheduler
        use_tensorboard: Whether to enable TensorBoard logging
        verbose: Whether to print progress

    Returns:
        CallbackList with standard callbacks configured

    Example:
        >>> callbacks = create_helix_callbacks(
        ...     output_dir='results/experiment_1',
        ...     patience=15,
        ...     total_epochs=100,
        ...     use_tensorboard=True
        ... )
    """
    output_dir = Path(output_dir)

    callbacks = CallbackList(
        [
            EarlyStopping(
                monitor=monitor,
                patience=patience,
                mode="max",
                restore_best_weights=True,
                verbose=verbose,
            ),
            ModelCheckpoint(
                filepath=output_dir / "checkpoints" / "helix_epoch_{epoch}_{metric}.pt",
                monitor=monitor,
                mode="max",
                save_best_only=True,
                verbose=verbose,
            ),
            LearningRateScheduler(
                schedule_type="warmup_cosine",
                initial_lr=initial_lr,
                warmup_epochs=warmup_epochs,
                total_epochs=total_epochs,
                verbose=verbose,
            ),
            TrainingLogger(
                log_dir=output_dir / "logs", tensorboard=use_tensorboard, verbose=verbose
            ),
        ]
    )

    return callbacks
