# fitz_ai/cli/__init__.py
"""
Fitz CLI - Clean, minimal CLI.

Usage:
    fitz init               # Setup wizard
    fitz ingest ./src       # Ingest documents
    fitz query "question"   # Query knowledge base
    fitz collections        # List/manage collections
    fitz config             # Show configuration
    fitz doctor             # System diagnostics
"""

from fitz_ai.cli.cli import app

__all__ = ["app"]
