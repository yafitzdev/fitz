# fitz_sage/logging/logger.py
"""Logging for fitz-sage.

One entry point — ``get_logger`` returns a standard-library ``logging.Logger``.
``configure_logging`` installs a single root handler; call it once at an
application entrypoint (the CLI does). Libraries (the SDK) leave configuration
to the calling application.

Per-query correlation: ``set_query_context`` stashes a query id in a contextvar
that ``_QueryContextFilter`` injects into every record as ``query_id``, so the
formatter can show which query a log line belongs to.
"""

from __future__ import annotations

import contextvars
import logging
import sys

DEFAULT_FORMAT = "[%(levelname)s] %(name)s%(query_id)s — %(message)s"

_query_id: contextvars.ContextVar[str] = contextvars.ContextVar("fitz_query_id", default="")


def get_logger(name: str) -> logging.Logger:
    """Return a standard-library logger. The single fitz-sage logging entry point."""
    return logging.getLogger(name)


class _QueryContextFilter(logging.Filter):
    """Inject the current query id (if any) into each record for the formatter."""

    def filter(self, record: logging.LogRecord) -> bool:
        qid = _query_id.get()
        record.query_id = f" [q={qid}]" if qid else ""
        return True


def configure_logging(
    level: int = logging.WARNING,
    fmt: str = DEFAULT_FORMAT,
    stream=sys.stderr,
) -> None:
    """Install the root logging handler. Idempotent — safe to call repeatedly.

    Call once early at an application entrypoint (CLI / API server).
    """
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter(fmt))
        handler.addFilter(_QueryContextFilter())
        root.addHandler(handler)
    root.setLevel(level)


def set_query_context(query_id: str, collection: str | None = None, **kwargs) -> None:
    """Set the query id used to correlate log records emitted during a query."""
    _query_id.set(query_id)


def clear_query_context() -> None:
    """Clear the query id after a query completes."""
    _query_id.set("")
