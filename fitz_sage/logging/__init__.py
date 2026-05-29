# fitz_sage/logging/__init__.py
"""Logging for fitz-sage: one stdlib ``get_logger`` plus ``configure_logging``."""

from fitz_sage.logging.logger import (
    clear_query_context,
    configure_logging,
    get_logger,
    set_query_context,
)

__all__ = [
    "get_logger",
    "configure_logging",
    "set_query_context",
    "clear_query_context",
]
