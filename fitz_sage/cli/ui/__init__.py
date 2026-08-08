# fitz_sage/cli/ui/__init__.py
"""
CLI UI components.

Provides consistent styling and fallback for Rich-less environments.

Usage:
    from fitz_sage.cli.ui import ui, console, RICH

    ui.header("My Command")
    ui.success("Done!")
    name = ui.prompt_text("Enter name", default="default")
"""

from __future__ import annotations

from .console import RICH, console
from .display import display_answer, display_evidence_pack, display_sources
from .output import OutputMixin
from .prompts import PromptMixin


class UI(OutputMixin, PromptMixin):
    """
    Unified UI helpers with Rich fallback.

    All methods work with or without Rich installed.
    With Rich: colored output, panels, tables, progress bars.
    Without Rich: plain text fallback.
    """

    pass


# Singleton instance
ui = UI()

__all__ = [
    # Main UI
    "ui",
    "UI",
    # Console
    "console",
    "RICH",
    # Display functions
    "display_answer",
    "display_evidence_pack",
    "display_sources",
]
