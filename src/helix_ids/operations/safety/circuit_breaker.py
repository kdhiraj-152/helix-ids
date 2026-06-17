"""Runtime circuit breakers for production training safety.

Protects training pipelines from NaN explosions, loss explosions, memory
exhaustion, invalid gradients, empty batches, and label corruption.

Each breaker is a state machine: GREEN -> TRIPPED. Once tripped it stays
tripped until manually reset, preventing silent degradation.

Usage
-----
    breaker = CircuitBreaker(name="loss_check", max_threshold=1e6, mode="max")
    if not breaker.check(tensor):
        raise TrainingHaltError("Loss explosion detected")

    # Or use decorators:
    @trip_on_nan
    def compute_loss(...):
        ...

    # Context manager:
    with circuit_breaker(name="grad_norm", max_threshold=500.0):
        # compute gradients
        ...
"""

from __future__ import annotations

import enum
import functools
import logging
import time
from dataclasses import dataclass
from math import isinf, isnan
from typing import Any, Callable, TypeVar

import torch

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ── Enum ─────────────────────────────────────────────────────────────────────


class BreakerState(enum.Enum):
    """State of a circuit breaker."""

    GREEN = "green"
    TRIPPED = "tripped"


class CheckMode(enum.Enum):
    """Check strategy for threshold-based breakers."""

    MAX = "max"          # trip if any value > threshold
    MIN = "min"          # trip if any value < threshold
    BOTH = "both"        # trip if value outside [low, high]


# ── Error ────────────────────────────────────────────────────────────────────


class CircuitBreakerError(RuntimeError):
    """Raised when a circuit breaker trips."""

    def __init__(
        self,
        message: str,
        *,
        breaker_name: str,
        metric_value: float | None = None,
        threshold: float | None = None,
    ) -> None:
        super().__init__(message)
        self.breaker_name = breaker_name
        self.metric_value = metric_value
        self.threshold = threshold


# ── CircuitBreaker ────────────────────────────────────────────────────────────


@dataclass
class CircuitBreaker:
    """A stateful guard that monitors a scalar metric and trips on violation.

    Parameters
    ----------
    name : str
        Human-readable breaker name for logging.
    max_threshold : float or None
        Upper bound. Trip if value > max_threshold.  ``inf`` disables.
    min_threshold : float or None
        Lower bound. Trip if value < min_threshold. ``-inf`` disables.
    patience : int
        Number of consecutive violations allowed before tripping (default 1).
    cooldown_seconds : float
        Seconds to wait after trip before allowing reset (default 60).
    auto_reset : bool
        If True, reset automatically after cooldown (default False).
    mode : CheckMode
        Check strategy (max / min / both). Inferred when thresholds provided.
    """

    name: str
    max_threshold: float | None = None
    min_threshold: float | None = None
    patience: int = 1
    cooldown_seconds: float = 60.0
    auto_reset: bool = False

    # Internal state
    state: BreakerState = BreakerState.GREEN
    violation_count: int = 0
    last_trip_time: float = 0.0
    trip_count: int = 0
    last_checked_value: float = 0.0
    last_trip_reason: str = ""

    def __post_init__(self) -> None:
        if self.max_threshold is None and self.min_threshold is None:
            self.max_threshold = float("inf")
        if self.max_threshold is not None and self.min_threshold is not None:
            self.mode = CheckMode.BOTH
        elif self.max_threshold is not None:
            self.mode = CheckMode.MAX
        elif self.min_threshold is not None:
            self.mode = CheckMode.MIN
        else:
            self.mode = CheckMode.MAX

    # ── Public API ──────────────────────────────────────────────────────────

    def check(self, value: float | torch.Tensor, context: str = "") -> bool:
        """Check *value* against thresholds. Returns True if safe, False if tripped.

        If already tripped and ``auto_reset`` is True, attempt auto-reset
        after cooldown.
        """
        if self.state == BreakerState.TRIPPED:
            if self.auto_reset and self._cooldown_expired():
                self.reset()
            else:
                return False

        # Convert tensor to scalar
        if isinstance(value, torch.Tensor):
            v = value.detach().cpu().item()
        else:
            v = float(value)

        self.last_checked_value = v

        # Check
        violation = self._is_violation(v)
        if violation:
            self.violation_count += 1
            if self.violation_count >= self.patience:
                self._trip(v, context)
                return False
        else:
            self.violation_count = 0

        return True

    def check_batch(
        self, values: float | torch.Tensor, context: str = "", batch_size: int = 1
    ) -> bool:
        """Check a batch-level metric. Same as check()."""
        return self.check(values, context)

    def reset(self) -> None:
        """Manually reset the breaker to GREEN state."""
        self.state = BreakerState.GREEN
        self.violation_count = 0
        self.last_trip_reason = ""

    @property
    def is_tripped(self) -> bool:
        return self.state == BreakerState.TRIPPED

    @property
    def status_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable status snapshot."""
        return {
            "name": self.name,
            "state": self.state.value,
            "violation_count": self.violation_count,
            "trip_count": self.trip_count,
            "last_checked_value": self.last_checked_value,
            "last_trip_reason": self.last_trip_reason,
            "max_threshold": self.max_threshold,
            "min_threshold": self.min_threshold,
            "patience": self.patience,
            "cooldown_seconds": self.cooldown_seconds,
            "auto_reset": self.auto_reset,
        }

    # ── Internal ────────────────────────────────────────────────────────────

    def _is_violation(self, value: float) -> bool:
        if isnan(value) or isinf(value):
            return True
        if self.mode in (CheckMode.MAX, CheckMode.BOTH) and self.max_threshold is not None:
            if value > self.max_threshold:
                return True
        if self.mode in (CheckMode.MIN, CheckMode.BOTH) and self.min_threshold is not None:
            if value < self.min_threshold:
                return True
        return False

    def _trip(self, value: float, context: str) -> None:
        self.state = BreakerState.TRIPPED
        self.trip_count += 1
        self.last_trip_time = time.time()
        self.last_trip_reason = (
            f"[{self.name}] tripped: value={value:.4f}, "
            f"max={self.max_threshold}, min={self.min_threshold}, ctx={context}"
        )
        logger.warning(self.last_trip_reason)

    def _cooldown_expired(self) -> bool:
        return (time.time() - self.last_trip_time) >= self.cooldown_seconds


# ── Context manager ──────────────────────────────────────────────────────────


class _CircuitBreakerContext:
    """Context manager wrapping a CircuitBreaker."""

    def __init__(self, breaker: CircuitBreaker) -> None:
        self._breaker = breaker

    def __enter__(self) -> CircuitBreaker:
        return self._breaker

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        if exc_type is CircuitBreakerError:
            # Already handled — propagate
            return


def circuit_breaker(
    name: str,
    max_threshold: float | None = None,
    min_threshold: float | None = None,
    patience: int = 1,
    **kwargs: Any,
) -> _CircuitBreakerContext:
    """Context manager shorthand for creating and checking a CircuitBreaker.

    Example
    -------
        with circuit_breaker(name="loss", max_threshold=1e6):
            loss = compute_loss(...)
    """
    breaker = CircuitBreaker(
        name=name,
        max_threshold=max_threshold,
        min_threshold=min_threshold,
        patience=patience,
        **kwargs,
    )
    return _CircuitBreakerContext(breaker)


# ── Decorators ───────────────────────────────────────────────────────────────


def trip_on_nan(func: F) -> F:
    """Decorator: trip breaker if any output value is NaN."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        if isinstance(result, torch.Tensor):
            if torch.isnan(result).any():
                raise CircuitBreakerError(
                    f"NaN detected in output of {func.__name__}",
                    breaker_name=func.__name__,
                )
        return result

    return wrapper  # type: ignore[return-value]


def trip_on_inf(func: F) -> F:
    """Decorator: trip breaker if any output value is Inf."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        if isinstance(result, torch.Tensor):
            if torch.isinf(result).any():
                raise CircuitBreakerError(
                    f"Inf detected in output of {func.__name__}",
                    breaker_name=func.__name__,
                )
        return result

    return wrapper  # type: ignore[return-value]


def trip_on_exceed(max_threshold: float) -> Callable[[F], F]:
    """Decorator: trip breaker if output exceeds *max_threshold*."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)
            if isinstance(result, torch.Tensor):
                v = result.detach().cpu().item()
                if isnan(v) or isinf(v) or v > max_threshold:
                    raise CircuitBreakerError(
                        f"Output {v:.4f} exceeds threshold {max_threshold} "
                        f"in {func.__name__}",
                        breaker_name=func.__name__,
                        metric_value=v,
                        threshold=max_threshold,
                    )
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


# ── Specialised Guard Functions ──────────────────────────────────────────────


def trip_on_memory(threshold_gb: float = 0.90) -> CircuitBreakerError | None:
    """Check GPU memory usage and return a CircuitBreakerError if exhausted.

    Parameters
    ----------
    threshold_gb : float
        Fraction of total memory considered exhausted (default 0.90 = 90%).

    Returns
    -------
    CircuitBreakerError or None
        None if memory is safe.
    """
    if not torch.cuda.is_available():
        return None

    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    total = torch.cuda.get_device_properties(0).total_memory

    usage_ratio = reserved / max(total, 1)
    if usage_ratio >= threshold_gb:
        return CircuitBreakerError(
            f"GPU memory threshold breached: {usage_ratio:.1%} used "
            f"({allocated / 1024**3:.1f}GB / {total / 1024**3:.1f}GB)",
            breaker_name="gpu_memory",
            metric_value=usage_ratio,
            threshold=threshold_gb,
        )
    return None


def trip_on_empty(data: Any, name: str = "batch") -> CircuitBreakerError | None:
    """Check if data container is empty and return a CircuitBreakerError if so.

    Parameters
    ----------
    data : Any
        The data to check (tensor, list, dict, DataLoader batch, etc.)
    name : str
        A descriptive name for error messages.

    Returns
    -------
    CircuitBreakerError or None
        None if data is not empty.
    """
    if data is None:
        return CircuitBreakerError(
            f"Empty data: {name} is None",
            breaker_name=name,
        )

    if isinstance(data, torch.Tensor):
        if data.numel() == 0:
            return CircuitBreakerError(
                f"Empty batch: {name} tensor has 0 elements",
                breaker_name=name,
            )
    elif isinstance(data, (list, tuple)):
        if len(data) == 0:
            return CircuitBreakerError(
                f"Empty data: {name} list/tuple is empty",
                breaker_name=name,
            )
    elif isinstance(data, dict):
        if not data:
            return CircuitBreakerError(
                f"Empty data: {name} dict is empty",
                breaker_name=name,
            )

    return None


def check_gradients(
    model: torch.nn.Module,
    *,
    max_grad_norm: float = 100.0,
    check_nan: bool = True,
    check_inf: bool = True,
) -> CircuitBreakerError | None:
    """Check model gradients for validity.

    Parameters
    ----------
    model : torch.nn.Module
        The model whose gradients to check.
    max_grad_norm : float
        Maximum allowed total gradient norm.
    check_nan : bool
        Check for NaN gradients (default True).
    check_inf : bool
        Check for Inf gradients (default True).

    Returns
    -------
    CircuitBreakerError or None
        None if gradients are valid.
    """
    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad = param.grad

        if check_nan and torch.isnan(grad).any():
            return CircuitBreakerError(
                f"NaN gradient in parameter {name}",
                breaker_name="gradients",
            )
        if check_inf and torch.isinf(grad).any():
            return CircuitBreakerError(
                f"Inf gradient in parameter {name}",
                breaker_name="gradients",
            )

    # Check total gradient norm
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    if total_norm > max_grad_norm:
        return CircuitBreakerError(
            f"Gradient norm {total_norm:.2f} exceeds limit {max_grad_norm}",
            breaker_name="gradients",
            metric_value=total_norm,
            threshold=max_grad_norm,
        )

    return None


# ── Breaker Registry ─────────────────────────────────────────────────────────


class BreakerRegistry:
    """A registry of circuit breakers for a training run.

    Provides a central point for creating, checking, and reporting on all
    active breakers.
    """

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}

    def register(self, breaker: CircuitBreaker) -> CircuitBreaker:
        """Register a breaker. Replaces any existing breaker with the same name."""
        self._breakers[breaker.name] = breaker
        return breaker

    def get(self, name: str) -> CircuitBreaker | None:
        """Get a registered breaker by name."""
        return self._breakers.get(name)

    def __getitem__(self, name: str) -> CircuitBreaker:
        """Get a registered breaker by name. Raises KeyError if not found."""
        return self._breakers[name]

    def check_all(self, values: dict[str, float | torch.Tensor]) -> dict[str, bool]:
        """Check all registered breakers against named values.

        Returns {breaker_name: safe} for each registered breaker.
        """
        results: dict[str, bool] = {}
        for name, value in values.items():
            breaker = self._breakers.get(name)
            if breaker is not None:
                results[name] = breaker.check(value)
        return results

    def all_safe(self, values: dict[str, float | torch.Tensor]) -> bool:
        """Check all registered breakers; return True only if all pass."""
        return all(self.check_all(values).values())

    def status_report(self) -> dict[str, dict[str, Any]]:
        """Return a {name: status_dict} snapshot of all breakers."""
        return {
            name: breaker.status_dict
            for name, breaker in self._breakers.items()
        }

    def reset_all(self) -> None:
        """Reset all breakers to GREEN state."""
        for breaker in self._breakers.values():
            breaker.reset()
