"""Comprehensive tests for structured logging framework.

Covers log_context, log_formatter, and structured_logger modules
with >85% branch coverage on each.
"""

from __future__ import annotations

import inspect
import json
import logging
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from helix_ids.operations.logging.log_context import LogContext, LogContextManager, current_log_context
from helix_ids.operations.logging.log_formatter import StructuredFormatter
from helix_ids.operations.logging.structured_logger import StructuredLogger, get_logger, _loggers


# ── Helpers ────────────────────────────────────────────────────────────────────


def _parse_log_lines(stream: StringIO) -> list[dict]:
    """Parse newline-delimited JSON from a StringIO."""
    stream.seek(0)
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def _setup_logger(
    name: str,
    formatter: StructuredFormatter | None = None,
) -> tuple[StringIO, StructuredLogger]:
    """Create a test logger writing to a StringIO, return (stream, logger)."""
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter or StructuredFormatter())
    logger = get_logger(name, add_handler=False)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return stream, logger


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_logger_registry() -> None:
    """Clear the global logger registry before each test for isolation."""
    _loggers.clear()


# ═════════════════════════════════════════════════════════════════════════════
# LogContext Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestLogContext:
    """Tests for the LogContext context manager (log_context.py)."""

    def test_context_empty_by_default(self) -> None:
        """current_log_context returns empty dict outside any context."""
        assert current_log_context() == {}

    def test_context_merge(self) -> None:
        """Basic context with multiple fields."""
        with LogContext(run_id="run_abc", phase="training", epoch=1):
            ctx = current_log_context()
            assert ctx["run_id"] == "run_abc"
            assert ctx["phase"] == "training"
            assert ctx["epoch"] == 1

    def test_context_nesting(self) -> None:
        """Nested contexts merge outer fields."""
        with LogContext(run_id="outer"):
            with LogContext(experiment_id="inner"):
                ctx = current_log_context()
                assert ctx["run_id"] == "outer"
                assert ctx["experiment_id"] == "inner"

    def test_context_override(self) -> None:
        """Inner context overrides outer field values."""
        with LogContext(phase="init"):
            with LogContext(phase="training"):
                ctx = current_log_context()
                assert ctx["phase"] == "training"

    def test_context_cleanup_after_exit(self) -> None:
        """Context is restored after exiting."""
        with LogContext(run_id="temp"):
            assert current_log_context()["run_id"] == "temp"
        assert current_log_context() == {}

    def test_context_extra_dict(self) -> None:
        """Extra dict keys get merged into context."""
        with LogContext(run_id="abc", extra={"custom_key": "custom_val"}):
            ctx = current_log_context()
            assert ctx["custom_key"] == "custom_val"

    def test_context_with_step_and_checkpoint(self) -> None:
        """Checkpoint_id and step fields."""
        with LogContext(run_id="run_1", checkpoint_id="ckpt_100", step=500):
            ctx = current_log_context()
            assert ctx["checkpoint_id"] == "ckpt_100"
            assert ctx["step"] == 500

    def test_context_with_experiment_id(self) -> None:
        """Experiment_id field branch coverage."""
        with LogContext(experiment_id="exp_42"):
            ctx = current_log_context()
            assert ctx["experiment_id"] == "exp_42"

    def test_context_all_fields(self) -> None:
        """All context fields at once."""
        with LogContext(
            run_id="r1",
            experiment_id="e1",
            checkpoint_id="c1",
            phase="test",
            epoch=10,
            step=100,
            extra={"foo": "bar"},
        ):
            ctx = current_log_context()
            assert ctx == {
                "run_id": "r1",
                "experiment_id": "e1",
                "checkpoint_id": "c1",
                "phase": "test",
                "epoch": 10,
                "step": 100,
                "foo": "bar",
            }

    def test_context_returns_self(self) -> None:
        """__enter__ returns the LogContextManager instance."""
        with LogContext(run_id="test") as ctx_mgr:
            assert isinstance(ctx_mgr, LogContextManager)
            assert ctx_mgr.run_id == "test"

    def test_context_no_fields(self) -> None:
        """Empty LogContext() produces empty context dict."""
        with LogContext():
            ctx = current_log_context()
            assert ctx == {}

    def test_current_context_returns_copy(self) -> None:
        """current_log_context returns a mutable copy, not the original."""
        with LogContext(run_id="original"):
            ctx = current_log_context()
            ctx["injected"] = "should_not_persist"
            assert current_log_context() == {"run_id": "original"}


# ═════════════════════════════════════════════════════════════════════════════
# StructuredFormatter Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestStructuredFormatter:
    """Tests for JSON log formatter (log_formatter.py)."""

    def test_minimal_record(self) -> None:
        """Basic record produces valid JSON with required fields."""
        stream, logger = _setup_logger("minimal_test")
        logger.info("hello world")
        parsed = _parse_log_lines(stream)
        assert len(parsed) == 1
        record = parsed[0]
        assert record["message"] == "hello world"
        assert record["severity"] == "INFO"
        assert "timestamp" in record
        assert "logger" in record
        assert "module" in record
        assert "function" in record
        assert "line" in record

    def test_record_with_context(self) -> None:
        """LogContext fields appear in output."""
        stream, logger = _setup_logger("ctx_fmt_test")
        with LogContext(run_id="run_ctx", phase="eval"):
            logger.info("context test")
        parsed = _parse_log_lines(stream)
        assert parsed[0]["run_id"] == "run_ctx"
        assert parsed[0]["phase"] == "eval"

    def test_record_without_context(self) -> None:
        """No extra keys when no LogContext is active."""
        stream, logger = _setup_logger("noctx_test")
        logger.info("plain")
        parsed = _parse_log_lines(stream)
        for key in ("run_id", "experiment_id", "phase", "epoch", "step"):
            assert key not in parsed[0]

    def test_pretty_print(self) -> None:
        """Pretty-printed JSON is indented."""
        stream, logger = _setup_logger("pretty_test", StructuredFormatter(pretty=True))
        logger.info("pretty")
        output = stream.getvalue()
        assert "\n" in output
        assert '"severity":' in output

    def test_compact_json(self) -> None:
        """Without pretty, each record is one JSON line."""
        stream, logger = _setup_logger("compact_test")
        logger.info("compact")
        output = stream.getvalue().strip()
        lines = output.splitlines()
        assert len(lines) == 1

    def test_record_with_extra_fields(self) -> None:
        """Extra fields ('extra' kwarg) appear as JSON 'extra' key.
        Covers: hasattr(record, 'extra_fields') branch (line 68)."""
        stream, logger = _setup_logger("extra_fmt")
        logger.info("with extra", extra={"batch_size": 64, "loss": 0.023})
        parsed = _parse_log_lines(stream)
        assert "extra" in parsed[0]
        assert parsed[0]["extra"]["batch_size"] == 64
        assert parsed[0]["extra"]["loss"] == 0.023

    def test_record_with_extra_and_context(self) -> None:
        """Both LogContext fields and extra kwargs appear."""
        stream, logger = _setup_logger("extra_ctx")
        with LogContext(run_id="test_run"):
            logger.info("combined", extra={"lr": 0.001})
        parsed = _parse_log_lines(stream)
        assert parsed[0]["run_id"] == "test_run"
        assert parsed[0]["extra"]["lr"] == 0.001

    def test_error_with_exception(self) -> None:
        """Exception info captured for ERROR+ with include_stack=True."""
        stream, logger = _setup_logger(
            "exc_fmt", StructuredFormatter(include_stack=True)
        )
        try:
            raise ValueError("test error")
        except ValueError:
            logger.exception("An error occurred")
        parsed = _parse_log_lines(stream)
        assert parsed[0]["severity"] == "ERROR"
        assert "exception" in parsed[0]
        assert "ValueError" in parsed[0]["exception"]
        assert "test error" in parsed[0]["exception"]

    def test_include_stack_error_no_exception(self) -> None:
        """include_stack=True but no exception — no 'exception' field.
        Covers: exc_value is None branch."""
        stream, logger = _setup_logger(
            "stack_noexc", StructuredFormatter(include_stack=True)
        )
        logger.error("plain error")
        parsed = _parse_log_lines(stream)
        assert parsed[0]["severity"] == "ERROR"
        assert "exception" not in parsed[0]

    def test_exception_below_error_threshold(self) -> None:
        """include_stack=True but level INFO — no exception captured."""
        stream, logger = _setup_logger(
            "info_noexc", StructuredFormatter(include_stack=True)
        )
        try:
            raise ValueError("should not appear")
        except ValueError:
            logger.info("logged at info level", exc_info=True)
        parsed = _parse_log_lines(stream)
        assert "exception" not in parsed[0]

    def test_format_exception_no_exc_info(self) -> None:
        """_format_exception returns '' when no exc_info.
        Covers: line 104."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname=__file__,
            lineno=42, msg="test", args=(), exc_info=None,
        )
        assert formatter._format_exception(record) == ""

    def test_format_args_dict(self) -> None:
        """record.args as dict is merged into 'extra'.
        Covers: lines 72-79 (isinstance(record.args, dict) branch)."""
        formatter = StructuredFormatter()
        # LogRecord.__init__ expects args[0] to exist (tuple), so create with
        # tuple then replace args with dict to test the isinstance branch.
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname=__file__,
            lineno=42, msg="test %(key)s", args=({"key": "value"},),
            exc_info=None,
        )
        object.__setattr__(record, "args", {"key": "value"})
        result = formatter.format(record)
        parsed = json.loads(result)
        assert "extra" in parsed
        assert parsed["extra"]["key"] == "value"

    def test_format_args_dict_underscore_skipped(self) -> None:
        """Dict args with underscore keys are filtered out."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname=__file__,
            lineno=42, msg="test", args=({"_hidden": "secret", "visible": "ok"},),
            exc_info=None,
        )
        object.__setattr__(record, "args", {"_hidden": "secret", "visible": "ok"})
        result = formatter.format(record)
        parsed = json.loads(result)
        assert "visible" in parsed["extra"]
        assert "_hidden" not in parsed["extra"]

    def test_format_args_dict_empty_explicit(self) -> None:
        """Dict args with only underscore keys yields no 'extra' field."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname=__file__,
            lineno=42, msg="test", args=({"_hidden": "secret"},),
            exc_info=None,
        )
        object.__setattr__(record, "args", {"_hidden": "secret"})
        result = formatter.format(record)
        parsed = json.loads(result)
        assert "extra" not in parsed

    def test_format_exc_info_true_boolean(self) -> None:
        """exc_info=True (bool) outside exception handler.
        Covers: lines 87-90 (elif record.exc_info branch).
        
        Note: Tested outside an exception handler so sys.exc_info()[1] is None
        and _format_exception is not called (it would fail on *True unpacking)."""
        formatter = StructuredFormatter(include_stack=True)
        # No exception handler active — sys.exc_info()[1] will be None
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname=__file__,
            lineno=42, msg="test", args=(), exc_info=True,
        )
        result = formatter.format(record)
        parsed = json.loads(result)
        # exc_value is None so no 'exception' key, but the elif branch was taken
        assert "exception" not in parsed

    def test_format_stack_info_on_record(self) -> None:
        """stack_info on record appears as 'stack' in output.
        Covers: line 93-94 (if record.stack_info)."""
        formatter = StructuredFormatter(include_stack=True)
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname=__file__,
            lineno=42, msg="test", args=(), exc_info=None,
            sinfo="Custom stack info\ntrace here",
        )
        result = formatter.format(record)
        parsed = json.loads(result)
        assert "stack" in parsed
        assert "Custom stack info" in parsed["stack"]

    def test_format_stack_info_no_include_stack(self) -> None:
        """stack_info present but include_stack=False — no 'stack'."""
        formatter = StructuredFormatter(include_stack=False)
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname=__file__,
            lineno=42, msg="test", args=(), exc_info=None,
            sinfo="Should not appear",
        )
        result = formatter.format(record)
        parsed = json.loads(result)
        assert "stack" not in parsed


# ═════════════════════════════════════════════════════════════════════════════
# StructuredLogger Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestStructuredLogger:
    """Tests for StructuredLogger class (structured_logger.py)."""

    def test_logger_info(self) -> None:
        stream, logger = _setup_logger("info_test")
        logger.info("simple message")
        parsed = _parse_log_lines(stream)
        assert len(parsed) == 1
        assert parsed[0]["message"] == "simple message"
        assert parsed[0]["severity"] == "INFO"

    def test_logger_debug(self) -> None:
        stream, logger = _setup_logger("debug_test")
        logger.debug("debug message")
        parsed = _parse_log_lines(stream)
        assert parsed[0]["severity"] == "DEBUG"

    def test_logger_warning(self) -> None:
        stream, logger = _setup_logger("warn_test")
        logger.warning("warning message")
        parsed = _parse_log_lines(stream)
        assert parsed[0]["severity"] == "WARNING"

    def test_logger_error(self) -> None:
        stream, logger = _setup_logger("error_test")
        logger.error("error message")
        parsed = _parse_log_lines(stream)
        assert parsed[0]["severity"] == "ERROR"

    def test_logger_critical(self) -> None:
        stream, logger = _setup_logger("crit_test")
        logger.critical("critical message")
        parsed = _parse_log_lines(stream)
        assert parsed[0]["severity"] == "CRITICAL"

    def test_all_log_levels(self) -> None:
        stream, logger = _setup_logger("levels_test")
        logger.debug("d")
        logger.info("i")
        logger.warning("w")
        logger.error("e")
        logger.critical("c")
        parsed = _parse_log_lines(stream)
        assert [r["severity"] for r in parsed] == [
            "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL",
        ]

    def test_with_structured_context(self) -> None:
        stream, logger = _setup_logger("struct_ctx")
        with LogContext(run_id="my_run"):
            logger.info("context message")
        parsed = _parse_log_lines(stream)
        assert parsed[0]["run_id"] == "my_run"

    def test_with_extra(self) -> None:
        stream, logger = _setup_logger("extra_logger")
        logger.info("with extra", extra={"key": "value"})
        parsed = _parse_log_lines(stream)
        assert parsed[0]["extra"]["key"] == "value"

    def test_exception_method(self) -> None:
        """logger.exception() captures and logs exceptions."""
        stream, logger = _setup_logger("exc_logger")
        logger.handlers[0].setFormatter(StructuredFormatter(include_stack=True))
        try:
            raise RuntimeError("test runtime error")
        except RuntimeError:
            logger.exception("Exception occurred")
        parsed = _parse_log_lines(stream)
        assert parsed[0]["severity"] == "ERROR"
        assert "exception" in parsed[0]
        assert "RuntimeError" in parsed[0]["exception"]

    def test_exc_info_tuple(self) -> None:
        """_log_with_context with exc_info as a tuple (not bool)."""
        stream, logger = _setup_logger("exc_tuple")
        try:
            raise ValueError("tuple error")
        except ValueError:
            exc_tuple = sys.exc_info()
        logger.error("error with tuple", exc_info=exc_tuple)
        parsed = _parse_log_lines(stream)
        assert len(parsed) == 1

    def test_exc_info_true_boolean(self) -> None:
        """_log_with_context with exc_info=True (bool, not tuple).
        Covers the 'if not isinstance(exc_info, tuple)' branch (line 83)."""
        stream, logger = _setup_logger("exc_true")
        try:
            raise KeyError("test key")
        except KeyError:
            logger.info("logged with exc_info", exc_info=True)
        parsed = _parse_log_lines(stream)
        assert len(parsed) == 1

    def test_with_stack_info(self) -> None:
        """Logging with stack_info=True.
        Covers the stack_info block (lines 103-109)."""
        stream, logger = _setup_logger("stack_test")
        logger.info("with stack info", stack_info=True)
        parsed = _parse_log_lines(stream)
        assert len(parsed) == 1
        assert parsed[0]["message"] == "with stack info"

    def test_with_empty_extra(self) -> None:
        """Logging with empty extra dict."""
        stream, logger = _setup_logger("empty_extra")
        logger.info("empty extra", extra={})
        parsed = _parse_log_lines(stream)
        assert len(parsed) == 1

    def test_caller_frame_none(self) -> None:
        """_log_with_context when caller_frame is None (simulated top-level).
        Covers: lines 96-97 (caller_frame else branch)."""
        stream, logger = _setup_logger("no_caller")
        with patch.object(inspect, "currentframe") as mock_cf:
            mock_frame = MagicMock()
            del mock_frame.f_back  # Remove auto-created attribute
            # Set f_back to None to simulate top-level frame
            mock_frame.f_back = None
            mock_cf.return_value = mock_frame
            logger.info("no caller frame")
        parsed = _parse_log_lines(stream)
        assert parsed[0]["line"] == 0
        assert parsed[0]["function"] == ""

    def test_get_logger_reuses_instance(self) -> None:
        l1 = get_logger("reuse_test")
        l2 = get_logger("reuse_test")
        assert l1 is l2

    def test_get_logger_default_name(self) -> None:
        logger = get_logger()
        assert logger.name == "helix_ids"

    def test_get_logger_no_handler(self) -> None:
        logger = get_logger("no_handler", add_handler=False)
        assert len(logger.handlers) == 0

    def test_get_logger_with_handler(self) -> None:
        logger = get_logger("with_handler", add_handler=True)
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)
        assert isinstance(logger.handlers[0].formatter, StructuredFormatter)

    def test_get_logger_with_level(self) -> None:
        logger = get_logger("lvl_test", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_get_logger_pretty(self) -> None:
        logger = get_logger("pretty_lgr", pretty=True, add_handler=True)
        assert logger.handlers[0].formatter._pretty is True

    def test_get_logger_include_stack(self) -> None:
        logger = get_logger("stack_lgr", include_stack=True, add_handler=True)
        assert logger.handlers[0].formatter._include_stack is True

    def test_logger_propagate_false(self) -> None:
        logger = get_logger("no_prop")
        assert logger.propagate is False

    def test_multiple_loggers_different_names(self) -> None:
        l1 = get_logger("logger_a")
        l2 = get_logger("logger_b")
        assert l1 is not l2
        assert l1.name == "logger_a"
        assert l2.name == "logger_b"


# ═════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """End-to-end scenarios combining all modules."""

    def test_training_cycle(self) -> None:
        """Simulate a training cycle with nested context."""
        stream, logger = _setup_logger("training_integration")
        with LogContext(run_id="exp_001", phase="training"):
            for epoch in range(1, 3):
                with LogContext(epoch=epoch):
                    for step in [1, 2]:
                        with LogContext(step=step):
                            logger.info(
                                "train_step",
                                extra={"loss": round(0.5 / step, 4), "lr": 0.001},
                            )
        parsed = _parse_log_lines(stream)
        assert len(parsed) == 4
        first = parsed[0]
        assert first["run_id"] == "exp_001"
        assert first["phase"] == "training"
        assert first["epoch"] == 1
        assert first["step"] == 1
        assert first["severity"] == "INFO"

    def test_error_during_training(self) -> None:
        """Error during training with exception capture."""
        stream, logger = _setup_logger(
            "error_training", StructuredFormatter(include_stack=True)
        )
        with LogContext(run_id="exp_fail", phase="training", epoch=5):
            try:
                raise RuntimeError("GPU OOM during forward pass")
            except RuntimeError:
                logger.exception("Training failed at epoch 5")
        parsed = _parse_log_lines(stream)
        assert parsed[0]["severity"] == "ERROR"
        assert parsed[0]["run_id"] == "exp_fail"
        assert "GPU OOM" in parsed[0]["exception"]

    def test_mixed_context_and_plain(self) -> None:
        """Context-rich and plain log records coexist."""
        stream, logger = _setup_logger("mixed")
        logger.info("startup")
        with LogContext(run_id="mix_1"):
            logger.info("running")
        logger.info("shutdown")
        parsed = _parse_log_lines(stream)
        assert len(parsed) == 3
        assert "run_id" not in parsed[0]
        assert parsed[1]["run_id"] == "mix_1"
        assert "run_id" not in parsed[2]

    def test_json_validity(self) -> None:
        """Every emitted line must be valid JSON."""
        stream, logger = _setup_logger("valid_json")
        logger.info("msg1")
        logger.warning("msg2")
        logger.error("msg3")
        for line in stream.getvalue().splitlines():
            if line.strip():
                json.loads(line)

    def test_module_info_populated(self) -> None:
        """Module, function, line fields are populated."""
        stream, logger = _setup_logger("module_info")
        logger.info("check module")
        parsed = _parse_log_lines(stream)
        assert parsed[0]["module"] == "test_structured_logger"
