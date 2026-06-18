"""Thread-safe log context for structured logging.

Provides a context manager that tracks run_id, experiment_id,
checkpoint_id, phase, epoch, and step across the logging lifecycle.
Uses Python's contextvars for thread-/async-safe context propagation.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Any

_log_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "_structured_log_context"
)


@dataclass
class LogContextManager:
    """Context manager that enriches all structured log records within its scope.

    Example
    -------
    with LogContext(run_id="run_abc", phase="training"):
        logger.info("Training started")  # auto-includes run_id and phase
    """

    run_id: str | None = None
    experiment_id: str | None = None
    checkpoint_id: str | None = None
    phase: str | None = None
    epoch: int | None = None
    step: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __enter__(self) -> LogContextManager:
        existing = _log_context.get({}).copy()
        if self.run_id is not None:
            existing["run_id"] = self.run_id
        if self.experiment_id is not None:
            existing["experiment_id"] = self.experiment_id
        if self.checkpoint_id is not None:
            existing["checkpoint_id"] = self.checkpoint_id
        if self.phase is not None:
            existing["phase"] = self.phase
        if self.epoch is not None:
            existing["epoch"] = self.epoch
        if self.step is not None:
            existing["step"] = self.step
        existing.update(self.extra)

        self._token = _log_context.set(existing)
        return self

    def __exit__(self, *args: Any) -> None:
        _log_context.reset(self._token)


def current_log_context() -> dict[str, Any]:
    return _log_context.get({}).copy()


# Convenience alias
LogContext = LogContextManager
