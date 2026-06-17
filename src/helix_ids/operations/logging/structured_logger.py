"""Structured logger for production logging.

Provides a factory function ``get_logger`` that returns a logger pre-configured
with the StructuredFormatter. Emits JSON log records with automatic context
enrichment from LogContext.

Usage
-----
from helix_ids.operations.logging import get_logger, LogContext

logger = get_logger(__name__)

with LogContext(run_id="run_abc", phase="training", epoch=1):
    logger.info("Batch complete", extra={"batch_size": 64, "loss": 0.023})
    # Produces:
    # {"timestamp":"...","severity":"INFO","logger":"...","message":"Batch complete",
    #  "run_id":"run_abc","phase":"training","epoch":1,"extra":{"batch_size":64,"loss":0.023}}
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from logging import _ExcInfoType

from .log_context import current_log_context
from .log_formatter import StructuredFormatter

# Registry of created loggers for configuration
_loggers: dict[str, logging.Logger] = {}
_DEFAULT_LEVEL = logging.INFO


class StructuredLogger(logging.Logger):
    """A logger subclass that injects structured context into every record.

    Usage::

        logger = StructuredLogger("my_module")
        logger.info("Hello", extra={"count": 42})

    The *extra* dict is automatically captured as structured fields
    in the JSON output.
    """

    def _log_with_context(
        self,
        level: int,
        msg: object,
        args: tuple[object, ...] = (),
        exc_info: _ExcInfoType | None = None,
        extra: Mapping[str, object] | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
    ) -> None:
        """Internal: emit a log record at *level* with structured context.

        Parameters
        ----------
        level : int
            Logging level (e.g. ``logging.INFO``).
        msg : object
            Log message (converted to str via % formatting).
        args : tuple of object
            Positional arguments for %-formatting.
        exc_info : _ExcInfoType or None
            Exception info tuple, *True*, or *None*.
        extra : Mapping[str, object] or None
            Structured extra fields to include in JSON output.
        stack_info : bool
            If *True*, capture stack trace text.
        stacklevel : int
            Stack level for caller detection (currently unused).
        """
        import sys

        # Convert exc_info=True to sys.exc_info() like Logger._log does
        if exc_info:
            if not isinstance(exc_info, tuple):
                exc_info = sys.exc_info()

        # Collect top-of-stack caller info (module, line, function)
        f = inspect.currentframe()
        caller_frame: Any = None
        try:
            caller_frame = f.f_back if f else None
            caller_frame = caller_frame.f_back if caller_frame else None  # skip _log_with_context
            if caller_frame:
                line_no = caller_frame.f_lineno
                func_name = caller_frame.f_code.co_name
            else:
                line_no = 0
                func_name = ""
        finally:
            del f  # avoid reference cycles

        # Convert stack_info bool to sinfo str like Logger._log does
        sinfo: str | None = None
        if stack_info:
            import io
            import traceback

            sio = io.StringIO()
            traceback.print_stack(file=sio)
            sinfo = sio.getvalue()

        # Build the LogRecord using standard Python logging
        record = self.makeRecord(
            self.name,
            level,
            fn=caller_frame.f_code.co_filename if caller_frame else "",
            lno=line_no,
            msg=msg,
            args=args,
            exc_info=exc_info,  # type: ignore[arg-type]
            func=func_name,
            extra=extra or {},
            sinfo=sinfo,
        )

        # Store structured extra fields on the record for the formatter
        ctx = current_log_context()
        structured_extra: dict[str, object] = dict(extra) if extra else {}
        record.structured_extra = structured_extra  # type: ignore[attr-defined]
        record.extra_fields = structured_extra  # type: ignore[attr-defined]
        record.context = ctx  # type: ignore[attr-defined]

        self.handle(record)

    def debug(
        self,
        msg: object,
        *args: object,
        exc_info: _ExcInfoType = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        self._log_with_context(
            logging.DEBUG, msg, args,
            exc_info=exc_info, extra=extra,
            stack_info=stack_info, stacklevel=stacklevel,
        )

    def info(
        self,
        msg: object,
        *args: object,
        exc_info: _ExcInfoType = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        self._log_with_context(
            logging.INFO, msg, args,
            exc_info=exc_info, extra=extra,
            stack_info=stack_info, stacklevel=stacklevel,
        )

    def warning(
        self,
        msg: object,
        *args: object,
        exc_info: _ExcInfoType = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        self._log_with_context(
            logging.WARNING, msg, args,
            exc_info=exc_info, extra=extra,
            stack_info=stack_info, stacklevel=stacklevel,
        )

    def error(
        self,
        msg: object,
        *args: object,
        exc_info: _ExcInfoType = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        self._log_with_context(
            logging.ERROR, msg, args,
            exc_info=exc_info, extra=extra,
            stack_info=stack_info, stacklevel=stacklevel,
        )

    def critical(
        self,
        msg: object,
        *args: object,
        exc_info: _ExcInfoType = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        self._log_with_context(
            logging.CRITICAL, msg, args,
            exc_info=exc_info, extra=extra,
            stack_info=stack_info, stacklevel=stacklevel,
        )

    def exception(
        self,
        msg: object,
        *args: object,
        exc_info: _ExcInfoType = True,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        self._log_with_context(
            logging.ERROR, msg, args,
            exc_info=exc_info, extra=extra,
            stack_info=stack_info, stacklevel=stacklevel,
        )


def get_logger(
    name: str | None = None,
    *,
    level: int = _DEFAULT_LEVEL,
    include_stack: bool = False,
    pretty: bool = False,
    add_handler: bool = True,
) -> StructuredLogger:
    """Factory: get or create a structured JSON logger.

    Parameters
    ----------
    name : str or None
        Logger name (typically ``__name__``). Defaults to "helix_ids" if None.
    level : int
        Logging level (default: ``logging.INFO``).
    include_stack : bool
        If True, attach stack info to ERROR+ records (default: False).
    pretty : bool
        If True, pretty-print JSON (default: False).
    add_handler : bool
        If True (default), add a StreamHandler with StructuredFormatter.
        Set False to attach your own handler.

    Returns
    -------
    StructuredLogger
    """
    if name is None:
        name = "helix_ids"

    if name in _loggers:
        return _loggers[name]  # type: ignore[return-value]

    logger = StructuredLogger(name)
    logger.setLevel(level)

    if add_handler:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = StructuredFormatter(
            include_stack=include_stack,
            pretty=pretty,
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Prevent propagation to root logger (avoid double emission)
    logger.propagate = False

    _loggers[name] = logger
    return logger
