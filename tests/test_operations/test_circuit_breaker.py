"""
Tests for runtime circuit breakers.

Covers:
  - NaN / Inf detection
  - threshold-based tripping (max, min, both)
  - patience and auto-reset
  - memory exhaustion guard
  - empty batch guard
  - gradient validation
  - decorators (trip_on_nan, trip_on_inf, trip_on_exceed)
  - context manager
  - BreakerRegistry
"""

from __future__ import annotations

import math
import time

import pytest
import torch

from helix_ids.operations.safety.circuit_breaker import (
    BreakerRegistry,
    BreakerState,
    CheckMode,
    CircuitBreaker,
    CircuitBreakerError,
    _CircuitBreakerContext,
    check_gradients,
    circuit_breaker,
    trip_on_empty,
    trip_on_exceed,
    trip_on_inf,
    trip_on_memory,
    trip_on_nan,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CircuitBreaker — Core
# ═══════════════════════════════════════════════════════════════════════════════


class TestCircuitBreaker:
    def test_green_by_default(self) -> None:
        cb = CircuitBreaker(name="test", max_threshold=100.0)
        assert cb.state == BreakerState.GREEN
        assert not cb.is_tripped

    def test_check_within_threshold(self) -> None:
        cb = CircuitBreaker(name="test", max_threshold=100.0)
        assert cb.check(50.0) is True
        assert cb.state == BreakerState.GREEN

    def test_check_exceeds_threshold_trips(self) -> None:
        cb = CircuitBreaker(name="test", max_threshold=100.0)
        assert cb.check(50.0) is True
        assert cb.check(150.0) is False  # first violation → trip
        assert cb.state == BreakerState.TRIPPED

    def test_check_returns_false_when_already_tripped(self) -> None:
        """Cover the ``return False`` at line 145 when tripped with no auto-reset."""
        cb = CircuitBreaker(name="test", max_threshold=100.0)
        cb.check(200.0)  # trips
        assert cb.state == BreakerState.TRIPPED
        # Calling check() again while tripped must return False
        assert cb.check(10.0) is False  # not auto_reset → return False
        assert cb.is_tripped

    def test_check_nan_trips(self) -> None:
        cb = CircuitBreaker(name="test", max_threshold=100.0)
        assert cb.check(float("nan")) is False
        assert cb.state == BreakerState.TRIPPED

    def test_check_inf_trips(self) -> None:
        cb = CircuitBreaker(name="test", max_threshold=100.0)
        assert cb.check(float("inf")) is False
        assert cb.state == BreakerState.TRIPPED

    def test_check_negative_inf_trips(self) -> None:
        cb = CircuitBreaker(name="test", min_threshold=0.0)
        assert cb.check(float("-inf")) is False
        assert cb.state == BreakerState.TRIPPED

    def test_min_threshold(self) -> None:
        cb = CircuitBreaker(name="test", min_threshold=0.0)
        assert cb.check(10.0) is True
        assert cb.check(-5.0) is False  # below min

    def test_both_thresholds(self) -> None:
        cb = CircuitBreaker(name="test", max_threshold=100.0, min_threshold=0.0)
        assert cb.check(50.0) is True
        assert cb.check(-1.0) is False  # below min
        cb.reset()
        assert cb.check(200.0) is False  # above max

    def test_both_thresholds_no_violation_between(self) -> None:
        """BOTH mode: value between thresholds is safe."""
        cb = CircuitBreaker(name="test", max_threshold=100.0, min_threshold=0.0)
        assert cb.check(50.0) is True
        assert cb.state == BreakerState.GREEN

    def test_reset(self) -> None:
        cb = CircuitBreaker(name="test", max_threshold=100.0)
        cb.check(200.0)
        assert cb.is_tripped
        cb.reset()
        assert not cb.is_tripped
        assert cb.state == BreakerState.GREEN

    def test_reset_clears_all_state(self) -> None:
        """Verify reset clears violation_count and last_trip_reason."""
        cb = CircuitBreaker(name="test", max_threshold=100.0)
        cb.check(200.0)  # trips
        assert cb.is_tripped
        assert cb.violation_count > 0
        assert cb.last_trip_reason != ""
        cb.reset()
        assert cb.violation_count == 0
        assert cb.last_trip_reason == ""

    def test_patience_requires_multiple_violations(self) -> None:
        cb = CircuitBreaker(name="test", max_threshold=100.0, patience=3)
        assert cb.check(150.0) is True  # 1/3 violations
        assert cb.check(150.0) is True  # 2/3
        assert cb.check(150.0) is False  # 3/3 — tripped
        assert cb.is_tripped

    def test_patience_resets_on_good_value(self) -> None:
        cb = CircuitBreaker(name="test", max_threshold=100.0, patience=3)
        assert cb.check(150.0) is True  # 1/3
        assert cb.check(150.0) is True  # 2/3
        assert cb.check(50.0) is True  # reset counter
        assert cb.check(150.0) is True  # 1/3 again
        assert not cb.is_tripped

    def test_patience_one_trips_immediately(self) -> None:
        """patience=1 (default) trips on first violation."""
        cb = CircuitBreaker(name="test", max_threshold=100.0, patience=1)
        assert cb.check(200.0) is False
        assert cb.is_tripped

    def test_tensor_input(self) -> None:
        cb = CircuitBreaker(name="test", max_threshold=100.0)
        assert cb.check(torch.tensor(50.0)) is True
        assert cb.check(torch.tensor(200.0)) is False

    def test_status_dict(self) -> None:
        cb = CircuitBreaker(name="test", max_threshold=100.0)
        status = cb.status_dict
        assert status["name"] == "test"
        assert status["state"] == "green"
        assert status["max_threshold"] == 100.0
        assert status["patience"] == 1
        assert status["min_threshold"] is None
        assert status["violation_count"] == 0
        assert status["trip_count"] == 0
        assert status["last_checked_value"] == 0.0
        assert status["last_trip_reason"] == ""
        assert status["cooldown_seconds"] == 60.0
        assert status["auto_reset"] is False

    def test_status_dict_after_trip(self) -> None:
        """Verify status dict reflects trip state."""
        cb = CircuitBreaker(name="test", max_threshold=100.0)
        cb.check(200.0, context="spike")
        status = cb.status_dict
        assert status["state"] == "tripped"
        assert status["trip_count"] == 1
        assert status["last_checked_value"] == 200.0
        assert "spike" in status["last_trip_reason"]

    def test_trip_count_increments(self) -> None:
        cb = CircuitBreaker(name="test", max_threshold=100.0)
        cb.check(200.0)
        assert cb.trip_count == 1
        cb.reset()
        cb.check(200.0)
        assert cb.trip_count == 2

    def test_trip_with_context(self) -> None:
        """Context message should appear in trip reason."""
        cb = CircuitBreaker(name="test", max_threshold=100.0)
        cb.check(200.0, context="loss_spike_detected")
        assert "loss_spike_detected" in cb.last_trip_reason
        assert "test" in cb.last_trip_reason
        assert "200.0000" in cb.last_trip_reason

    def test_last_checked_value_is_stored(self) -> None:
        cb = CircuitBreaker(name="test", max_threshold=100.0)
        cb.check(42.5)
        assert cb.last_checked_value == 42.5

    def test_is_tripped_property(self) -> None:
        cb = CircuitBreaker(name="test", max_threshold=100.0)
        assert cb.is_tripped is False
        cb.check(200.0)
        assert cb.is_tripped is True
        cb.reset()
        assert cb.is_tripped is False

    def test_check_batch_delegates_to_check(self) -> None:
        """Cover check_batch which delegates to check (line 171)."""
        cb = CircuitBreaker(name="test", max_threshold=100.0)
        assert cb.check_batch(50.0) is True
        assert cb.check_batch(50.0, context="batch_check") is True
        assert cb.check_batch(torch.tensor(50.0)) is True
        assert cb.check_batch(200.0) is False
        assert cb.is_tripped

    def test_cooldown_expired_true(self) -> None:
        cb = CircuitBreaker(name="test", max_threshold=100.0)
        cb.check(200.0)  # trips, sets last_trip_time
        cb.last_trip_time = 0.0  # long ago
        assert cb.is_tripped
        assert cb._cooldown_expired() is True

    def test_cooldown_expired_false(self) -> None:
        cb = CircuitBreaker(name="test", max_threshold=100.0, cooldown_seconds=3600.0)
        cb.check(200.0)  # trips
        assert cb._cooldown_expired() is False  # cooldown not expired

    def test_exact_threshold_not_violation(self) -> None:
        """Value equal to threshold is not a violation."""
        cb = CircuitBreaker(name="test", max_threshold=100.0)
        assert cb.check(100.0) is True  # not > threshold
        assert not cb.is_tripped

    def test_exact_min_threshold_not_violation(self) -> None:
        cb = CircuitBreaker(name="test", min_threshold=0.0)
        assert cb.check(0.0) is True  # not < threshold
        assert not cb.is_tripped


class TestAutoReset:
    def test_auto_reset_after_cooldown(self) -> None:
        cb = CircuitBreaker(
            name="test",
            max_threshold=100.0,
            patience=1,
            cooldown_seconds=0.01,
            auto_reset=True,
        )
        cb.check(200.0)
        assert cb.is_tripped

        # After cooldown, check should auto-reset
        time.sleep(0.02)
        assert cb.check(50.0) is True  # auto-reset happened
        assert not cb.is_tripped

    def test_auto_reset_violates_again(self) -> None:
        cb = CircuitBreaker(
            name="test",
            max_threshold=100.0,
            patience=1,
            cooldown_seconds=0.01,
            auto_reset=True,
        )
        cb.check(200.0)
        time.sleep(0.02)

        # Auto-resets but value is still bad → trips again
        assert cb.check(200.0) is False  # trips on new violation

    def test_auto_reset_cooldown_not_expired_returns_false(self) -> None:
        """When auto_reset is True but cooldown hasn't expired, returns False."""
        cb = CircuitBreaker(
            name="test",
            max_threshold=100.0,
            patience=1,
            cooldown_seconds=3600.0,  # long cooldown
            auto_reset=True,
        )
        cb.check(200.0)
        assert cb.is_tripped
        # Cooldown hasn't expired, so check returns False
        assert cb.check(50.0) is False
        assert cb.is_tripped

    @pytest.mark.parametrize("mode", ["max", "min", "both"])
    def test_check_mode_inference(self, mode: str) -> None:
        if mode == "max":
            cb = CircuitBreaker(name="test", max_threshold=100.0)
            assert cb.mode == CheckMode.MAX
        elif mode == "min":
            cb = CircuitBreaker(name="test", min_threshold=0.0)
            assert cb.mode == CheckMode.MIN
        elif mode == "both":
            cb = CircuitBreaker(name="test", max_threshold=100.0, min_threshold=0.0)
            assert cb.mode == CheckMode.BOTH

    def test_no_threshold_defaults_to_inf_max(self) -> None:
        cb = CircuitBreaker(name="test")
        assert cb.max_threshold == float("inf")
        assert cb.mode == CheckMode.MAX

    def test_infinite_thresholds_never_violates(self) -> None:
        """inf max and -inf min should never cause a violation."""
        cb = CircuitBreaker(
            name="test",
            max_threshold=float("inf"),
            min_threshold=float("-inf"),
        )
        assert cb.mode == CheckMode.BOTH
        assert cb.check(1e10) is True
        assert cb.check(-1e10) is True
        assert not cb.is_tripped

    def test_post_init_else_branch(self) -> None:
        """Cover the unreachable else branch (line 131) by bypassing __init__."""
        # Bypass the dataclass __init__ to leave thresholds as None after __post_init__
        cb = object.__new__(CircuitBreaker)
        cb.name = "test"
        cb.max_threshold = None
        cb.min_threshold = None
        cb.patience = 1
        cb.cooldown_seconds = 60.0
        cb.auto_reset = False
        cb.state = BreakerState.GREEN
        cb.violation_count = 0
        cb.last_trip_time = 0.0
        cb.trip_count = 0
        cb.last_checked_value = 0.0
        cb.last_trip_reason = ""
        cb.__post_init__()
        assert cb.mode == CheckMode.MAX
        assert cb.max_threshold == float("inf")

    def test_patience_zero_still_first_violation(self) -> None:
        """patience=0 should trip on first violation since 0 >= 0."""
        cb = CircuitBreaker(
            name="test",
            max_threshold=100.0,
            patience=0,
        )
        assert cb.check(200.0) is False

    def test_auto_reset_while_tripped_multi_check(self) -> None:
        """Multiple check calls while auto-reset tripped but cooldown not expired."""
        cb = CircuitBreaker(
            name="test",
            max_threshold=100.0,
            patience=1,
            cooldown_seconds=3600.0,
            auto_reset=True,
        )
        cb.check(200.0)
        assert cb.check(50.0) is False  # still tripped, cooldown not expired
        assert cb.check(50.0) is False  # same
        assert cb.is_tripped


# ═══════════════════════════════════════════════════════════════════════════════
# Decorators
# ═══════════════════════════════════════════════════════════════════════════════


class TestDecorators:
    def test_trip_on_nan(self) -> None:
        @trip_on_nan
        def bad_func() -> torch.Tensor:
            return torch.tensor(float("nan"))

        with pytest.raises(CircuitBreakerError, match="NaN detected"):
            bad_func()

    def test_trip_on_nan_no_error(self) -> None:
        @trip_on_nan
        def good_func() -> torch.Tensor:
            return torch.tensor(42.0)

        result = good_func()
        assert result == 42.0

    def test_trip_on_nan_non_tensor(self) -> None:
        """Cover non-tensor return path (branch 283→289)."""

        @trip_on_nan
        def non_tensor_func() -> int:
            return 42

        assert non_tensor_func() == 42

    def test_trip_on_inf(self) -> None:
        @trip_on_inf
        def bad_func() -> torch.Tensor:
            return torch.tensor(float("inf"))

        with pytest.raises(CircuitBreakerError, match="Inf detected"):
            bad_func()

    def test_trip_on_inf_non_tensor(self) -> None:
        """Cover non-tensor return path (line 306)."""

        @trip_on_inf
        def non_tensor_func() -> int:
            return 99

        assert non_tensor_func() == 99

    def test_trip_on_inf_tensor_no_inf(self) -> None:
        """Cover the tensor-but-no-inf branch (301→306)."""

        @trip_on_inf
        def good_tensor_func() -> torch.Tensor:
            return torch.tensor(42.0)

        result = good_tensor_func()
        assert result == 42.0

    def test_trip_on_exceed(self) -> None:
        @trip_on_exceed(max_threshold=10.0)
        def bad_func() -> torch.Tensor:
            return torch.tensor(100.0)

        with pytest.raises(CircuitBreakerError, match="exceeds threshold"):
            bad_func()

    def test_trip_on_exceed_nan(self) -> None:
        @trip_on_exceed(max_threshold=10.0)
        def nan_func() -> torch.Tensor:
            return torch.tensor(float("nan"))

        with pytest.raises(CircuitBreakerError, match="exceeds threshold"):
            nan_func()

    def test_trip_on_exceed_ok(self) -> None:
        @trip_on_exceed(max_threshold=100.0)
        def good_func() -> torch.Tensor:
            return torch.tensor(50.0)

        assert good_func() == 50.0

    def test_trip_on_exceed_non_tensor(self) -> None:
        """Cover non-tensor return path (branch 318→328)."""

        @trip_on_exceed(max_threshold=10.0)
        def non_tensor_func() -> int:
            return 5

        assert non_tensor_func() == 5

    def test_trip_on_exceed_inf(self) -> None:
        """trip_on_exceed with inf value should trip."""

        @trip_on_exceed(max_threshold=10.0)
        def inf_func() -> torch.Tensor:
            return torch.tensor(float("inf"))

        with pytest.raises(CircuitBreakerError, match="exceeds threshold"):
            inf_func()


# ═══════════════════════════════════════════════════════════════════════════════
# Context Manager
# ═══════════════════════════════════════════════════════════════════════════════


class TestContextManager:
    def test_circuit_breaker_context(self) -> None:
        with circuit_breaker(name="test_ctx", max_threshold=100.0) as cb:
            assert cb.check(50.0) is True
            assert not cb.is_tripped

    def test_circuit_breaker_context_trips(self) -> None:
        with circuit_breaker(name="test_ctx", max_threshold=100.0) as cb:
            assert cb.check(50.0) is True
            assert cb.check(200.0) is False
            assert cb.is_tripped

    def test_context_exit_with_circuit_breaker_error(self) -> None:
        """Cover __exit__ with CircuitBreakerError (line 247)."""
        breaker = CircuitBreaker(name="test_ctx", max_threshold=100.0)
        ctx = _CircuitBreakerContext(breaker)

        # __exit__ with CircuitBreakerError should just return (propagate)
        result = ctx.__exit__(CircuitBreakerError, None, None)
        assert result is None  # returns None (falsy), so exception propagates

    def test_context_exit_with_other_error(self) -> None:
        """__exit__ with non-CircuitBreakerError should also return None."""
        breaker = CircuitBreaker(name="test_ctx", max_threshold=100.0)
        ctx = _CircuitBreakerContext(breaker)
        result = ctx.__exit__(ValueError, None, None)
        assert result is None  # returns None → exception propagates

    def test_context_exit_no_error(self) -> None:
        """__exit__ with no exception should work."""
        breaker = CircuitBreaker(name="test_ctx", max_threshold=100.0)
        ctx = _CircuitBreakerContext(breaker)
        result = ctx.__exit__(None, None, None)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# Specialised Guards
# ═══════════════════════════════════════════════════════════════════════════════


class TestTripOnEmpty:
    def test_none_data(self) -> None:
        err = trip_on_empty(None, name="batch")
        assert err is not None
        assert "None" in str(err)

    def test_empty_tensor(self) -> None:
        err = trip_on_empty(torch.tensor([]), name="batch")
        assert err is not None
        assert "0 elements" in str(err)

    def test_empty_list(self) -> None:
        err = trip_on_empty([], name="batch")
        assert err is not None
        assert "empty" in str(err)

    def test_empty_tuple(self) -> None:
        err = trip_on_empty((), name="batch")
        assert err is not None
        assert "empty" in str(err)

    def test_empty_dict(self) -> None:
        err = trip_on_empty({}, name="batch")
        assert err is not None
        assert "empty" in str(err)

    def test_non_empty_data(self) -> None:
        assert trip_on_empty(torch.tensor([1.0, 2.0])) is None
        assert trip_on_empty([1, 2, 3]) is None
        assert trip_on_empty({"a": 1}) is None

    def test_non_empty_scalar_tensor(self) -> None:
        assert trip_on_empty(torch.tensor(5.0)) is None

    def test_non_empty_dict(self) -> None:
        """Cover the elif branch for non-empty dict (403→410)."""
        result = trip_on_empty({"key": "value"}, name="test")
        assert result is None

    def test_non_collection_type(self) -> None:
        """Cover the branch where data is not None/tensor/list/tuple/dict (403→410)."""
        result = trip_on_empty("hello_string", name="test")
        assert result is None
        result = trip_on_empty(42)
        assert result is None
        result = trip_on_empty(object())
        assert result is None


class TestTripOnMemory:
    def test_no_gpu_returns_none(self) -> None:
        # When no GPU available, the guard should return None
        result = trip_on_memory(0.90)
        if not torch.cuda.is_available():
            assert result is None


class TestGradientCheck:
    def build_model_with_gradient(
        self, grad_value: float | None = 1.0
    ) -> torch.nn.Module:
        """Create a tiny model with a specific gradient on its parameters."""
        model = torch.nn.Linear(4, 2)
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()

        if grad_value is not None and not math.isnan(grad_value) and not math.isinf(grad_value):
            # Override gradients with desired value
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.fill_(grad_value)
        elif grad_value is None:
            # Remove gradients
            for p in model.parameters():
                p.grad = None

        return model

    def test_no_gradient_returns_none(self) -> None:
        model = torch.nn.Linear(4, 2)
        err = check_gradients(model)
        # No gradients → no error (gradients are None, no NaN/Inf to check)
        assert err is None

    def test_valid_gradients_ok(self) -> None:
        model = self.build_model_with_gradient(0.5)
        err = check_gradients(model)
        assert err is None

    def test_gradient_norm_exceeds(self) -> None:
        """Cover the total_norm > max_grad_norm branch (line 464)."""
        # Build model with very large gradient that exceeds the norm limit
        model = self.build_model_with_gradient(1000.0)
        err = check_gradients(model, max_grad_norm=0.01)
        assert err is not None
        assert "exceeds limit" in str(err)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires CUDA"
    )
    def test_gpu_memory_guard(self) -> None:
        """Only runs on GPU systems."""
        result = trip_on_memory(threshold_gb=0.99)  # 99% threshold — likely safe
        assert result is None or isinstance(result, CircuitBreakerError)


class TestGradientCheckEdgeCases:
    def test_nan_gradient_detected(self) -> None:
        model = torch.nn.Linear(4, 2)
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()

        # Inject NaN into gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.fill_(float("nan"))

        err = check_gradients(model, check_nan=True)
        assert err is not None
        assert "NaN gradient" in str(err)

    def test_inf_gradient_detected(self) -> None:
        model = torch.nn.Linear(4, 2)
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()

        # Inject Inf into gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.fill_(float("inf"))

        err = check_gradients(model, check_inf=True)
        assert err is not None
        assert "Inf gradient" in str(err)

    def test_gradient_check_skip_nan(self) -> None:
        """With check_nan=False, a NaN gradient should not be detected."""
        model = torch.nn.Linear(4, 2)
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()

        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.fill_(float("nan"))

        err = check_gradients(model, check_nan=False, check_inf=False, max_grad_norm=1e10)
        assert err is None


# ═══════════════════════════════════════════════════════════════════════════════
# BreakerRegistry
# ═══════════════════════════════════════════════════════════════════════════════


class TestBreakerRegistry:
    def test_register_and_get(self) -> None:
        registry = BreakerRegistry()
        cb = CircuitBreaker(name="loss", max_threshold=1e6)
        registry.register(cb)
        assert registry.get("loss") is cb
        assert registry["loss"] is cb

    def test_get_unknown_returns_none(self) -> None:
        registry = BreakerRegistry()
        assert registry.get("nonexistent") is None

    def test_get_unknown_raises_keyerror(self) -> None:
        registry = BreakerRegistry()
        with pytest.raises(KeyError):
            _ = registry["nonexistent"]

    def test_check_all(self) -> None:
        registry = BreakerRegistry()
        cb1 = CircuitBreaker(name="loss", max_threshold=1e6)
        cb2 = CircuitBreaker(name="grad_norm", max_threshold=500.0)
        registry.register(cb1)
        registry.register(cb2)

        results = registry.check_all({"loss": 5e5, "grad_norm": 100.0})
        assert results["loss"] is True
        assert results["grad_norm"] is True

    def test_check_all_trips(self) -> None:
        registry = BreakerRegistry()
        cb1 = CircuitBreaker(name="loss", max_threshold=1e6)
        registry.register(cb1)

        results = registry.check_all({"loss": 2e6})
        assert results["loss"] is False

    def test_check_all_unknown_name_skipped(self) -> None:
        """Cover branch 508→506: unknown breaker name not in registry is skipped."""
        registry = BreakerRegistry()
        cb = CircuitBreaker(name="loss", max_threshold=1e6)
        registry.register(cb)

        results = registry.check_all(
            {"loss": 5e5, "unknown_metric": 999.0}
        )
        # "unknown_metric" should be skipped; only "loss" in results
        assert "loss" in results
        assert "unknown_metric" not in results

    def test_all_safe(self) -> None:
        registry = BreakerRegistry()
        cb = CircuitBreaker(name="loss", max_threshold=1e6)
        registry.register(cb)
        assert registry.all_safe({"loss": 5e5}) is True
        assert registry.all_safe({"loss": 2e6}) is False

    def test_status_report(self) -> None:
        registry = BreakerRegistry()
        cb = CircuitBreaker(name="loss", max_threshold=1e6)
        registry.register(cb)
        report = registry.status_report()
        assert "loss" in report
        assert report["loss"]["state"] == "green"

    def test_reset_all(self) -> None:
        registry = BreakerRegistry()
        cb1 = CircuitBreaker(name="loss", max_threshold=1e6)
        cb2 = CircuitBreaker(name="grad_norm", max_threshold=500.0)
        registry.register(cb1)
        registry.register(cb2)

        cb1.check(2e6)  # trip loss
        assert cb1.is_tripped

        registry.reset_all()
        assert not cb1.is_tripped
        assert not cb2.is_tripped

    def test_register_replaces_existing(self) -> None:
        """Registering a new breaker with the same name replaces the old one."""
        registry = BreakerRegistry()
        cb1 = CircuitBreaker(name="loss", max_threshold=1e6)
        cb2 = CircuitBreaker(name="loss", max_threshold=2e6)
        registry.register(cb1)
        registry.register(cb2)
        assert registry.get("loss") is cb2
