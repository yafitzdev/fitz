# fitz_sage/cli/__init__.py
"""
Fitz CLI - Clean, minimal CLI.

Usage:
    fitz retrieve "question" --source ./docs         # Register + retrieve
    fitz retrieve "question"                         # Retrieve from an existing collection
    fitz collections                                 # List/manage collections
    fitz serve                                       # Start REST API

Config: .fitz/config.yaml (auto-created on first run)
"""

from fitz_sage.cli.cli import app  # noqa: E402

__all__ = ["app"]
