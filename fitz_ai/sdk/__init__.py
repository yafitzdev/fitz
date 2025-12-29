# fitz_ai/sdk/__init__.py
"""
Fitz SDK - Stateful Python interface for the Fitz RAG framework.

Provides a simple two-step API for ingesting documents and asking questions.

Examples:
    >>> from fitz_ai import fitz
    >>> f = fitz()
    >>> f.ingest("./docs")
    >>> answer = f.ask("What is quantum computing?")
    >>> print(answer.text)
"""

from .fitz import IngestStats, fitz

__all__ = ["fitz", "IngestStats"]
