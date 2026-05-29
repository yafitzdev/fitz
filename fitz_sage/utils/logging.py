# fitz_sage/utils/logging.py
"""
Structured logging utilities for consistent context across the platform.

Provides context-aware logging with automatic enrichment of fields like:
- query_id, collection_id, source_id for tracing
- operation, component for debugging
- user_id, org_id for multi-tenancy (future)
- latency, status for performance monitoring
"""

from __future__ import annotations

import contextvars
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict

# Context variables for request-scoped data
_query_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "query_context", default={}
)
_operation_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "operation_context", default={}
)


class StructuredLogger:
    """
    Logger wrapper that adds structured context to all messages.

    Usage:
        logger = StructuredLogger(__name__)

        # Add context for a block
        with logger.context(query_id="q123", collection="docs"):
            logger.info("Processing query")  # Includes context automatically

        # Add operation timing
        with logger.operation("bm25_search"):
            results = do_search()  # Logs start, end, duration automatically
    """

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.name = name

    def _get_context(self) -> Dict[str, Any]:
        """Merge all context sources."""
        context = {
            "component": self.name,
        }
        context.update(_query_context.get())
        context.update(_operation_context.get())
        return context

    def _format_message(self, msg: str, **kwargs) -> str:
        """Format message with context fields."""
        context = self._get_context()
        context.update(kwargs)

        # Build structured suffix
        fields = []
        for key, value in context.items():
            if value is not None:
                fields.append(f"{key}={value}")

        if fields:
            return f"{msg} [{', '.join(fields)}]"
        return msg

    def debug(self, msg: str, **kwargs):
        self.logger.debug(self._format_message(msg, **kwargs))

    def info(self, msg: str, **kwargs):
        self.logger.info(self._format_message(msg, **kwargs))

    def warning(self, msg: str, **kwargs):
        self.logger.warning(self._format_message(msg, **kwargs))

    def error(self, msg: str, exc_info=None, **kwargs):
        self.logger.error(self._format_message(msg, **kwargs), exc_info=exc_info)

    def exception(self, msg: str, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(self._format_message(msg, **kwargs))

    @contextmanager
    def context(self, **kwargs):
        """
        Add context fields for a block.

        Example:
            with logger.context(query_id="q123", user="alice"):
                logger.info("Processing")  # Has query_id and user fields
        """
        token = _query_context.set(_query_context.get() | kwargs)
        try:
            yield
        finally:
            _query_context.reset(token)

    @contextmanager
    def operation(self, name: str, **kwargs):
        """
        Track an operation with timing.

        Example:
            with logger.operation("db_query", table="chunks"):
                results = db.query(...)  # Logs start, duration, success/failure
        """
        start = time.time()
        op_context = {"operation": name, **kwargs}
        token = _operation_context.set(_operation_context.get() | op_context)

        self.info(f"Starting {name}", status="started")

        try:
            yield
            duration = time.time() - start
            self.info(f"Completed {name}", status="success", duration_ms=round(duration * 1000, 2))
        except Exception as e:
            duration = time.time() - start
            self.error(
                f"Failed {name}",
                status="failed",
                duration_ms=round(duration * 1000, 2),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise
        finally:
            _operation_context.reset(token)


def get_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger instance.

    This is the main entry point for getting loggers in fitz-sage.

    Args:
        name: Logger name (usually __name__)

    Returns:
        StructuredLogger instance with context capabilities
    """
    return StructuredLogger(name)


# Convenience functions for setting global context
def set_query_context(query_id: str, collection: str = None, **kwargs):
    """Set query-level context (lasts for entire query)."""
    context = {"query_id": query_id}
    if collection:
        context["collection"] = collection
    context.update(kwargs)
    _query_context.set(context)


def clear_query_context():
    """Clear query context (call after query completes)."""
    _query_context.set({})
