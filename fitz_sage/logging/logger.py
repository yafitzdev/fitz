# fitz_sage/logging/logger.py
"""
Unified logging setup for the entire Fitz project.

All modules in fitz_sage use:
    from fitz_sage.logging import get_logger
    logger = get_logger(__name__)

Why this works:
- ONE place for configuration (format, level, handlers)
- Log namespaces follow module paths automatically
- No duplicate setup between packages
- Structured context automatically added to all messages

For legacy compatibility, this module re-exports the structured logger.
"""

import logging
import sys

# Import our new structured logging
from fitz_sage.utils.logging import (
    StructuredLogger,
    clear_query_context,
)
from fitz_sage.utils.logging import get_logger as get_structured_logger
from fitz_sage.utils.logging import (
    set_query_context,
)

DEFAULT_FORMAT = "[%(levelname)s] %(name)s — %(message)s"


def configure_logging(
    level: int = logging.INFO,
    fmt: str = DEFAULT_FORMAT,
    stream=sys.stdout,
):
    """
    Configure root logging handler.

    Called once early in the application lifecycle (e.g., CLI entrypoint).
    Safe to call multiple times — handler duplication is prevented.
    """
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(handler)

    root.setLevel(level)


def get_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger instance.

    Example:
        logger = get_logger(__name__)

        # Use structured context
        with logger.context(query_id="q123"):
            logger.info("Processing query")  # Includes query_id

        # Track operations
        with logger.operation("bm25_search"):
            results = search()  # Logs timing automatically

    Returns:
        StructuredLogger with context capabilities
    """
    return get_structured_logger(name)


# Re-export for convenience
__all__ = [
    "configure_logging",
    "get_logger",
    "set_query_context",
    "clear_query_context",
    "StructuredLogger",
]
