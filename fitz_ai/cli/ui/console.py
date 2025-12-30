# fitz_ai/cli/ui/console.py
"""
Rich console setup and detection.

Provides the shared console instance and Rich availability flag.
"""

from __future__ import annotations

import sys

# Detect if we can use Unicode safely (not Windows legacy console)
CAN_USE_UNICODE = sys.platform != "win32" or sys.stdout.encoding.lower() in (
    "utf-8",
    "utf8",
)

# ASCII-safe alternatives for symbols
CHECK = "✓" if CAN_USE_UNICODE else "[OK]"
CROSS = "✗" if CAN_USE_UNICODE else "[X]"
WARN = "⚠" if CAN_USE_UNICODE else "[!]"
INFO = "ℹ" if CAN_USE_UNICODE else "[i]"
ARROW = "→" if CAN_USE_UNICODE else "->"

# Rich setup (optional dependency)
try:
    from rich.columns import Columns
    from rich.console import Console
    from rich.control import Control
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import Progress
    from rich.prompt import Confirm, IntPrompt, Prompt
    from rich.syntax import Syntax
    from rich.table import Table

    console = Console()
    RICH = True
except ImportError:
    console = None  # type: ignore
    RICH = False
    # Stubs for type checking
    Columns = None  # type: ignore
    Control = None  # type: ignore
    Panel = None  # type: ignore
    Table = None  # type: ignore
    Syntax = None  # type: ignore
    Markdown = None  # type: ignore
    Progress = None  # type: ignore
    Confirm = None  # type: ignore
    IntPrompt = None  # type: ignore
    Prompt = None  # type: ignore

__all__ = [
    # Symbols
    "CAN_USE_UNICODE",
    "CHECK",
    "CROSS",
    "WARN",
    "INFO",
    "ARROW",
    # Rich
    "RICH",
    "console",
    # Rich components
    "Columns",
    "Control",
    "Confirm",
    "IntPrompt",
    "Markdown",
    "Panel",
    "Progress",
    "Prompt",
    "Syntax",
    "Table",
]
