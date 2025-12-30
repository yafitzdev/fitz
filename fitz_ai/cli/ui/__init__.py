# fitz_ai/cli/ui/__init__.py
"""
CLI UI components.

Provides consistent styling and fallback for Rich-less environments.

Usage:
    from fitz_ai.cli.ui import ui, console, RICH

    ui.header("My Command")
    ui.success("Done!")
    name = ui.prompt_text("Enter name", default="default")
    engine = ui.prompt_engine_selection(engines, descriptions, default)
"""

from __future__ import annotations

from .console import (
    RICH,
    Columns,
    Markdown,
    Panel,
    Progress,
    Syntax,
    Table,
    console,
)
from .display import display_answer, display_sources
from .engine_selection import EngineSelectionMixin
from .output import OutputMixin
from .progress import ProgressMixin
from .prompts import PromptMixin


def get_first_available(choices: list[str], fallback: str = "") -> str:
    """
    Get the first available choice from a list.

    Args:
        choices: List of available choices
        fallback: Fallback if no choices available

    Returns:
        First choice, or fallback if list is empty
    """
    if not choices:
        return fallback
    return choices[0]


class UI(OutputMixin, PromptMixin, ProgressMixin, EngineSelectionMixin):
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
    "display_sources",
    # Utilities
    "get_first_available",
    # Rich components (for advanced use)
    "Columns",
    "Panel",
    "Table",
    "Syntax",
    "Markdown",
    "Progress",
]
