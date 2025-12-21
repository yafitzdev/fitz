# fitz_ai/cli/__init__.py
"""
Fitz CLI v2 - Clean, minimal CLI.

Usage:
    fitz init              # Setup wizard
    fitz ingest ./docs     # Ingest documents
    fitz query "question"  # Query knowledge base
    fitz config            # Show configuration
    fitz doctor            # System diagnostics
"""

from fitz_ai.cli.cli import app

__all__ = ["app"]
