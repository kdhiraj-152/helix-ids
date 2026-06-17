"""JSON log formatter for structured logging.

Formats log records as newline-delimited JSON with rich structured context
including run_id, experiment_id, checkpoint_id, phase, epoch, step, and severity.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from .log_context import current_log_context


class StructuredFormatter(logging.Formatter):
    """Format log records as newline-delimited JSON.

    Each record includes:
      - timestamp (ISO 8601 with Z suffix)
      - severity (uppercase level name)
      - logger (name)
      - message (formatted %-style message)
      - module, function, line
      - structured fields from LogContext (run_id, experiment_id, etc.)

    Extra keyword arguments passed to the logger (logger.info("msg", extra={...}))
    are merged into the output dict.

    If *include_stack* is True, adds a "stack" field with traceback text
    for ERROR and CRITICAL records.
    """

    def __init__(
        self,
        *,
        include_stack: bool = False,
        pretty: bool = False,
    ):
        super().__init__()
        self._include_stack = include_stack
        self._pretty = pretty

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string."""
        timestamp = datetime.fromtimestamp(
            record.created, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Build the base payload
        payload: dict[str, Any] = {
            "timestamp": timestamp,
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Merge structured context from LogContext
        ctx = current_log_context()
        if ctx:
            payload.update(ctx)

        # Merge extra keyword args from the log call
        if hasattr(record, "extra_fields"):
            payload["extra"] = record.extra_fields

        # Include explicit extra dict from LogRecord
        if record.args and isinstance(record.args, dict):
            explicit_extra = {
                k: v
                for k, v in record.args.items()
                if not k.startswith("_")
            }
            if explicit_extra:
                payload.setdefault("extra", {}).update(explicit_extra)

        # Include stack trace for errors if configured
        if self._include_stack and record.levelno >= logging.ERROR:
            # exc_info can be True (bool) or (type, value, tb) tuple
            exc_value = None
            if isinstance(record.exc_info, (list, tuple)):
                exc_value = record.exc_info[1]
            elif record.exc_info:
                # True — try to get from sys
                import sys
                exc_value = sys.exc_info()[1]
            if exc_value:
                payload["exception"] = self._format_exception(record)
            if record.stack_info:
                payload["stack"] = record.stack_info

        indent = 2 if self._pretty else None
        return json.dumps(payload, default=str, indent=indent, sort_keys=False)

    def _format_exception(self, record: logging.LogRecord) -> str:
        """Format the exception info from a log record."""
        if record.exc_info:
            import traceback
            return "".join(traceback.format_exception(*record.exc_info))
        return ""
