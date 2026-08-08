"""Adapters between Fitz-Sage retrieval and managed model outputs."""

from fitz_sage.integrations.pyrrho import (
    answer_mode_from_pyrrho,
    create_pyrrho,
)

__all__ = ["answer_mode_from_pyrrho", "create_pyrrho"]
