"""Tests for StructuredFormatter JSON log formatter.

Covers:
  - basic JSON formatting of a LogRecord
  - structured context from LogContext
  - extra_fields attribute on record
  - include_stack functionality (error vs non-error levels)
  - pretty-print mode
  - exception formatting
  - edge cases (empty context, missing fields)
"""

from __future__ import annotations

import json
import logging
from unittest.mock import patch

import pytest

from helix_ids.operations.logging.log_context import LogContext, current_log_context
from helix_ids.operations.logging.log_formatter import StructuredFormatter


def _make_record(
    msg: str = "test message",
    level: int = logging.INFO,
    name: str = "test_logger",
    exc_info: tuple | None = None,
    stack_info: str | None = None,
    extra: dict | None = None,
) -> logging.LogRecord:
    """Create a minimal LogRecord for testing."""
    record = logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=42,
        msg=msg,
        args=(),
        exc_info=exc_info,
    )
    record.__dict__.setdefault("extra_fields", extra or {})
    if stack_info is not None:
        record.stack_info = stack_info
    return record


class TestStructuredFormatterBasic:
    def test_default_format_is_json(self) -> None:
        """formatter produces valid JSON."""
        fmt = StructuredFormatter()
        record = _make_record("hello")
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "hello"
        assert parsed["severity"] == "INFO"
        assert parsed["logger"] == "test_logger"
        assert "timestamp" in parsed
        assert "module" in parsed
        assert "function" in parsed
        assert "line" in parsed

    def test_timestamp_format(self) -> None:
        """Timestamp is ISO 8601 with Z suffix."""
        fmt = StructuredFormatter()
        record = _make_record()
        output = json.loads(fmt.format(record))
        ts = output["timestamp"]
        assert ts.endswith("Z")
        assert "T" in ts

    def test_severity_reflects_level(self) -> None:
        """Severity matches log level."""
        fmt = StructuredFormatter()
        for level, name in [
            (logging.DEBUG, "DEBUG"),
            (logging.INFO, "INFO"),
            (logging.WARNING, "WARNING"),
            (logging.ERROR, "ERROR"),
            (logging.CRITICAL, "CRITICAL"),
        ]:
            record = _make_record(level=level)
            output = json.loads(fmt.format(record))
            assert output["severity"] == name

    def test_extra_fields_in_output(self) -> None:
        """Extra fields appear in output under 'extra'."""
        fmt = StructuredFormatter()
        record = _make_record(extra={"batch_size": 64, "loss": 0.023})
        output = json.loads(fmt.format(record))
        assert output["extra"]["batch_size"] == 64
        assert output["extra"]["loss"] == 0.023

    def test_log_context_fields_injected(self) -> None:
        """Fields from LogContext are merged into output."""
        fmt = StructuredFormatter()
        with LogContext(run_id="run_abc", phase="training"):
            record = _make_record()
            output = json.loads(fmt.format(record))
            assert output["run_id"] == "run_abc"
            assert output["phase"] == "training"

    def test_empty_log_context(self) -> None:
        """No LogContext set does not inject unexpected fields."""
        fmt = StructuredFormatter()
        record = _make_record()
        output = json.loads(fmt.format(record))
        # Only base fields + extra
        for key in ("run_id", "phase", "epoch"):
            assert key not in output


class TestStructuredFormatterStack:
    def test_no_stack_by_default(self) -> None:
        """Stack info not included by default."""
        fmt = StructuredFormatter()
        record = _make_record(level=logging.ERROR, exc_info=(ValueError, ValueError("bad"), None))
        output = json.loads(fmt.format(record))
        assert "stack" not in output

    def test_stack_included_when_configured(self) -> None:
        """Stack info included for ERROR+ when include_stack=True."""
        fmt = StructuredFormatter(include_stack=True)
        record = _make_record(level=logging.ERROR, exc_info=(ValueError, ValueError("bad"), None))
        output = json.loads(fmt.format(record))
        assert "exception" in output
        assert "ValueError" in output["exception"]

    def test_no_stack_for_info_when_configured(self) -> None:
        """Stack info not included for INFO even when include_stack=True."""
        fmt = StructuredFormatter(include_stack=True)
        record = _make_record(level=logging.INFO)
        output = json.loads(fmt.format(record))
        assert "exception" not in output
        assert "stack" not in output

    def test_stack_info_field(self) -> None:
        """stack_info string is included for ERROR with include_stack."""
        fmt = StructuredFormatter(include_stack=True)
        record = _make_record(level=logging.ERROR, stack_info="Traceback...")
        output = json.loads(fmt.format(record))
        assert output["stack"] == "Traceback..."


class TestStructuredFormatterPretty:
    def test_pretty_indents(self) -> None:
        """Pretty mode indents JSON."""
        fmt = StructuredFormatter(pretty=True)
        record = _make_record()
        output = fmt.format(record)
        assert "\n" in output
        assert "  " in output

    def test_compact_no_indent(self) -> None:
        """Default compact mode has no indentation."""
        fmt = StructuredFormatter()
        record = _make_record()
        output = fmt.format(record)
        assert "\n" not in output


class TestStructuredFormatterEdgeCases:
    def test_empty_message(self) -> None:
        """Empty message is handled."""
        fmt = StructuredFormatter()
        record = _make_record(msg="")
        output = json.loads(fmt.format(record))
        assert output["message"] == ""

    def test_unicode_message(self) -> None:
        """Unicode message is handled."""
        fmt = StructuredFormatter()
        record = _make_record(msg="héllo wörld 🔥")
        output = json.loads(fmt.format(record))
        assert output["message"] == "héllo wörld 🔥"

    def test_exception_formatting(self) -> None:
        """_format_exception returns traceback string."""
        fmt = StructuredFormatter()
        try:
            raise RuntimeError("test error")
        except RuntimeError:
            import sys
            exc_info = sys.exc_info()
            record = _make_record(level=logging.ERROR, exc_info=exc_info)
            result = fmt._format_exception(record)
            assert "RuntimeError" in result
            assert "test error" in result

    def test_exception_formatting_no_exc(self) -> None:
        """_format_exception returns empty string when no exception."""
        fmt = StructuredFormatter()
        record = _make_record()
        result = fmt._format_exception(record)
        assert result == ""

    def test_record_args_as_dict(self) -> None:
        """Extra dict from record.args is merged."""
        fmt = StructuredFormatter()
        record = _make_record()
        record.args = {"custom_key": "custom_val"}
        output = json.loads(fmt.format(record))
        assert output["extra"]["custom_key"] == "custom_val"

    def test_record_args_skips_underscore_prefix(self) -> None:
        """Keys starting with _ are excluded from extra."""
        fmt = StructuredFormatter()
        record = _make_record()
        record.args = {"visible": 1, "_hidden": 2}
        output = json.loads(fmt.format(record))
        assert output["extra"]["visible"] == 1
        assert "_hidden" not in output["extra"]
