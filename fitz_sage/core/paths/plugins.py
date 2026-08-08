"""Managed model storage paths."""

from pathlib import Path


def user_home() -> Path:
    """Return the user-level Fitz model directory."""
    return Path.home() / ".fitz"
