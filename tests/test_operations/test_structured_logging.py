"""Tests for structured logging framework.

Covers log context, JSON formatting, structured emitter, and
integration scenarios.
"""

from __future__ import annotations

import json
import logging
from io import StringIO

import pytest

from helix_ids.operations.logging.log_context import LogContext, current_log_context
from helix_ids.operations.logging.log_formatter import StructuredFormatter
from helix_ids.operations.logging.structured_logger import (
    get_logger,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def log_stream() -> StringIO:
    """Return a StringIO that captures JSON log output."""
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(StructuredFormatter())
    logger = get_logger("test_logger", add_handler=False)
    # Re-register for isolation
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield stream
    logger.handlers.clear()


def _parse_log_lines(stream: StringIO) -> list[dict]:
    """Parse newline-delimited JSON from a StringIO."""
    stream.seek(0)
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


# ── LogContext Tests ──────────────────────────────────────────────────────────


class TestLogContext:
    def test_context_empty_by_default(self) -> None:
        assert current_log_context() == {}

    def test_context_merge(self) -> None:
        with LogContext(run_id="run_abc", phase="training", epoch=1):
            ctx = current_log_context()
            assert ctx["run_id"] == "run_abc"
            assert ctx["phase"] == "training"
            assert ctx["epoch"] == 1

    def test_context_nesting(self) -> None:
        with LogContext(run_id="outer"):
            with LogContext(experiment_id="inner"):
                ctx = current_log_context()
                assert ctx["run_id"] == "outer"
                assert ctx["experiment_id"] == "inner"

    def test_context_override(self) -> None:
        with LogContext(phase="init"):
            with LogContext(phase="training"):
                ctx = current_log_context()
                assert ctx["phase"] == "training"

    def test_context_cleanup_after_exit(self) -> None:
        with LogContext(run_id="temp"):
            assert current_log_context()["run_id"] == "temp"
        assert current_log_context() == {}

    def test_context_extra_dict(self) -> None:
        with LogContext(run_id="abc", extra={"custom_key": "custom_val"}):
            ctx = current_log_context()
            assert ctx["custom_key"] == "custom_val"

    def test_context_with_step_and_checkpoint(self) -> None:
        with LogContext(
            run_id="run_1",
            checkpoint_id="ckpt_100",
            step=500,
        ):
            ctx = current_log_context()
            assert ctx["checkpoint_id"] == "ckpt_100"
            assert ctx["step"] == 500


# ── StructuredFormatter Tests ────────────────────────────────────────────────


class TestStructuredFormatter:
    def test_minimal_record(self) -> None:
        """Verify a basic record produces valid JSON with required fields."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())
        logger = logging.getLogger("minimal_test")
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        logger.info("hello world")

        parsed = _parse_log_lines(stream)
        assert len(parsed) == 1
        record = parsed[0]
        assert record["message"] == "hello world"
        assert record["severity"] == "INFO"
        assert "timestamp" in record
        assert "logger" in record

    def test_record_with_context(self) -> None:
        """Verify LogContext fields appear in output."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())
        logger = get_logger("context_test", add_handler=False)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        with LogContext(run_id="run_ctx", phase="eval"):
            logger.info("context test")

        parsed = _parse_log_lines(stream)
        assert len(parsed) == 1
        record = parsed[0]
        assert record["run_id"] == "run_ctx"
        assert record["phase"] == "eval"

    def test_includes_module_info(self) -> None:
        """Verify module, function, line are populated."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())
        logger = logging.getLogger("module_test")
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        logger.info("check module")

        parsed = _parse_log_lines(stream)
        record = parsed[0]
        assert record["module"] == "test_structured_logging"
        assert record["function"] in ("test_includes_module_info", "info")

    def test_error_record_includes_exception(self) -> None:
        """Verify exception info is included for ERROR+ records."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter(include_stack=True))
        logger = logging.getLogger("error_test")
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        try:
            raise ValueError("test error")
        except ValueError:
            logger.exception("An error occurred")

        parsed = _parse_log_lines(stream)
        assert len(parsed) >= 1
        record = parsed[0]
        assert record["severity"] == "ERROR"
        assert "exception" in record
        assert "ValueError" in record["exception"]
        assert "test error" in record["exception"]

    def test_pretty_print(self) -> None:
        """Verify pretty-printed JSON is indented."""
        formatter = StructuredFormatter(pretty=True)
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)
        logger = logging.getLogger("pretty_test")
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        logger.info("pretty")

        output = stream.getvalue()
        assert "\n" in output
        assert '"severity":' in output


# ── StructuredLogger Tests ────────────────────────────────────────────────────


class TestStructuredLogger:
    def test_logger_info(self, log_stream: StringIO) -> None:
        logger = get_logger("struct_test", add_handler=False)
        logger.handlers.clear()
        logger.addHandler(logging.StreamHandler(log_stream))
        logger.setFormatter = lambda f: None  # type: ignore[attr-defined]
        logger.handlers[0].setFormatter(StructuredFormatter())
        logger.setLevel(logging.DEBUG)

        logger.info("simple message")

        parsed = _parse_log_lines(log_stream)
        assert len(parsed) == 1
        assert parsed[0]["message"] == "simple message"

    def test_logger_with_structured_context(self, log_stream: StringIO) -> None:
        logger = get_logger("struct_ctx_test", add_handler=False)
        logger.handlers.clear()
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        with LogContext(run_id="my_run"):
            logger.info("context message")

        parsed = _parse_log_lines(log_stream)
        assert parsed[0]["run_id"] == "my_run"

    def test_log_levels(self, log_stream: StringIO) -> None:
        logger = get_logger("level_test", add_handler=False)
        logger.handlers.clear()
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.debug("debug msg")
        logger.info("info msg")
        logger.warning("warn msg")
        logger.error("error msg")
        logger.critical("critical msg")

        parsed = _parse_log_lines(log_stream)
        severities = [r["severity"] for r in parsed]
        assert severities == ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    def test_get_logger_reuses_instance(self) -> None:
        logger1 = get_logger("reuse_test")
        logger2 = get_logger("reuse_test")
        assert logger1 is logger2

    def test_get_logger_no_handler(self) -> None:
        logger = get_logger("no_handler", add_handler=False)
        assert len(logger.handlers) == 0

    def test_logger_default_name(self) -> None:
        logger = get_logger()
        assert logger.name == "helix_ids"


# ── Integration Tests ─────────────────────────────────────────────────────────


class TestStructuredLoggingIntegration:
    """End-to-end scenario tests."""

    def test_training_cycle_logging(self, log_stream: StringIO) -> None:
        """Simulate a training cycle with run, epoch, and step context."""
        logger = get_logger("training", add_handler=False)
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(StructuredFormatter())
        logger.handlers.clear()
        logger.addHandler(handler)

        with LogContext(run_id="exp_001", phase="training"):
            for epoch in range(1, 3):  # 2 epochs
                with LogContext(epoch=epoch):
                    for step in [1, 2]:
                        with LogContext(step=step):
                            logger.info(
                                "train_step",
                                extra={"loss": round(0.5 / step, 4), "lr": 0.001},
                            )

        parsed = _parse_log_lines(log_stream)
        assert len(parsed) == 4  # 2 epochs × 2 steps

        # Verify first record structure
        first = parsed[0]
        assert first["run_id"] == "exp_001"
        assert first["phase"] == "training"
        assert first["epoch"] == 1
        assert first["step"] == 1
        assert first["severity"] == "INFO"
        assert "message" in first

    def test_error_during_training(self, log_stream: StringIO) -> None:
        """Simulate an error during training with exception capture."""
        logger = get_logger("error_training", add_handler=False)
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(StructuredFormatter(include_stack=True))
        logger.handlers.clear()
        logger.addHandler(handler)

        with LogContext(run_id="exp_fail", phase="training", epoch=5):
            try:
                raise RuntimeError("GPU OOM during forward pass")
            except RuntimeError:
                logger.exception("Training failed at epoch 5")

        parsed = _parse_log_lines(log_stream)
        assert len(parsed) >= 1
        rec = parsed[0]
        assert rec["severity"] == "ERROR"
        assert rec["run_id"] == "exp_fail"
        assert "GPU OOM" in rec["exception"]

    def test_mixed_context_and_plain_logging(self, log_stream: StringIO) -> None:
        """Ensure both context-rich and plain log records coexist."""
        logger = get_logger("mixed", add_handler=False)
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(StructuredFormatter())
        logger.handlers.clear()
        logger.addHandler(handler)

        # Plain log
        logger.info("startup")

        # Context-rich log
        with LogContext(run_id="mix_1"):
            logger.info("running")

        # Plain log after context exits
        logger.info("shutdown")

        parsed = _parse_log_lines(log_stream)
        assert len(parsed) == 3

        # First record has no context
        assert "run_id" not in parsed[0]
        # Second record has context
        assert parsed[1]["run_id"] == "mix_1"
        # Third record has no context again
        assert "run_id" not in parsed[2]

    def test_json_validity(self, log_stream: StringIO) -> None:
        """Every emitted line must be valid JSON."""
        logger = get_logger("valid_json", add_handler=False)
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(StructuredFormatter())
        logger.handlers.clear()
        logger.addHandler(handler)

        logger.info("msg1")
        logger.warning("msg2")
        logger.error("msg3")

        for line in log_stream.getvalue().splitlines():
            if line.strip():
                json.loads(line)  # raises if invalid
