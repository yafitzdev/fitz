# fitz_sage/sdk/__init__.py
"""
Fitz SDK - Stateful Python interface for Fitz retrieval.

Provides a simple API for pointing at documents, retrieving governed evidence,
and optionally asking for synthesized answers.

Examples:
    >>> from fitz_sage import fitz
    >>> f = fitz()
    >>> pack = f.evidence("What is quantum computing?", source="./docs")
    >>> print(pack.mode)
"""

from .fitz import fitz

__all__ = ["fitz"]
