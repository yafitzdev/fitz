# fitz_stack/logging.py
"""
Unified logging setup for the entire Fitz project.

All modules in fitz_ingest and fitz_rag use:
    from fitz_stack.logging import get_logger
    logger = get_logger(__name__)

Why this works:
- ONE place for configuration (format, level, handlers)
- Log namespaces follow module paths automatically
- No duplicate setup between packages
"""

import logging
import sys


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


def get_logger(name: str) -> logging.Logger:
    """
    Modules call this to get a logger.

    Example:
        logger = get_logger(__name__)

    Do NOT configure logging here — configuration happens in configure_logging().
    """
    return logging.getLogger(name)
