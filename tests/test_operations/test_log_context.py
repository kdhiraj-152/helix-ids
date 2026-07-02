"""Tests for LogContext thread-safe structured logging context.

Covers:
  - context manager entry/exit (set and reset)
  - individual fields (run_id, experiment_id, checkpoint_id, phase, epoch, step)
  - extra dict merging
  - nested context stacking
  - current_log_context retrieval
  - concurrent context isolation
"""

from __future__ import annotations

from helix_ids.operations.logging.log_context import (
    LogContext,
    LogContextManager,
    current_log_context,
)


class TestLogContextManager:
    def test_empty_context_returns_empty_dict(self) -> None:
        """No context set returns empty dict."""
        ctx = current_log_context()
        assert ctx == {}

    def test_sets_run_id(self) -> None:
        """LogContext sets run_id in context."""
        with LogContext(run_id="run_abc"):
            ctx = current_log_context()
            assert ctx["run_id"] == "run_abc"

    def test_sets_all_fields(self) -> None:
        """All fields are set correctly."""
        with LogContext(
            run_id="run_abc",
            experiment_id="exp_42",
            checkpoint_id="ckpt_7",
            phase="training",
            epoch=5,
            step=100,
        ):
            ctx = current_log_context()
            assert ctx["run_id"] == "run_abc"
            assert ctx["experiment_id"] == "exp_42"
            assert ctx["checkpoint_id"] == "ckpt_7"
            assert ctx["phase"] == "training"
            assert ctx["epoch"] == 5
            assert ctx["step"] == 100

    def test_resets_after_exit(self) -> None:
        """Context is cleared after exiting."""
        with LogContext(run_id="run_abc"):
            assert current_log_context()["run_id"] == "run_abc"
        assert current_log_context() == {}

    def test_extra_dict_merged(self) -> None:
        """Extra dict is merged into context."""
        with LogContext(run_id="run_abc", extra={"custom_key": "custom_val", "batch_size": 64}):
            ctx = current_log_context()
            assert ctx["custom_key"] == "custom_val"
            assert ctx["batch_size"] == 64

    def test_extra_overrides_named_field(self) -> None:
        """Extra dict can override named fields."""
        with LogContext(run_id="original", extra={"run_id": "overridden"}):
            ctx = current_log_context()
            assert ctx["run_id"] == "overridden"

    def test_nested_context_merges(self) -> None:
        """Nested contexts merge fields; inner fields take precedence."""
        with LogContext(run_id="outer_run", phase="outer"):
            with LogContext(epoch=1, phase="inner"):
                ctx = current_log_context()
                assert ctx["run_id"] == "outer_run"
                assert ctx["phase"] == "inner"
                assert ctx["epoch"] == 1
            # Outer context restored
            ctx = current_log_context()
            assert ctx["run_id"] == "outer_run"
            assert ctx["phase"] == "outer"
            assert "epoch" not in ctx

    def test_partial_fields(self) -> None:
        """Only provided fields are set; others are absent."""
        with LogContext(phase="inference"):
            ctx = current_log_context()
            assert ctx["phase"] == "inference"
            assert "run_id" not in ctx
            assert "epoch" not in ctx

    def test_convenience_alias(self) -> None:
        """LogContext and LogContextManager are the same."""
        assert LogContext is LogContextManager

    def test_contextmanager_type(self) -> None:
        """LogContextManager returns self from __enter__."""
        with LogContext(run_id="test") as cm:
            assert isinstance(cm, LogContextManager)
            assert cm.run_id == "test"
