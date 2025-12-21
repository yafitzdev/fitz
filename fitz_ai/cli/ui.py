# fitz_ai/cli/ui.py
"""
Shared UI helpers for CLI v2 commands.

Provides consistent styling and fallback for Rich-less environments.

Usage:
    from fitz_ai.cli.ui import ui, console, RICH

    ui.header("My Command")
    ui.success("Done!")
    name = ui.prompt_text("Enter name", default="default")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

# =============================================================================
# Rich Setup (optional dependency)
# =============================================================================

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.prompt import Confirm, IntPrompt, Prompt
    from rich.syntax import Syntax
    from rich.table import Table

    console = Console()
    RICH = True
except ImportError:
    console = None  # type: ignore
    RICH = False
    # Stubs for type checking
    Panel = None  # type: ignore
    Table = None  # type: ignore
    Syntax = None  # type: ignore
    Markdown = None  # type: ignore
    Progress = None  # type: ignore


# =============================================================================
# UI Helper Class
# =============================================================================


class UI:
    """
    Unified UI helpers with Rich fallback.

    All methods work with or without Rich installed.
    With Rich: colored output, panels, tables, progress bars.
    Without Rich: plain text fallback.
    """

    # -------------------------------------------------------------------------
    # Output Methods
    # -------------------------------------------------------------------------

    def print(self, msg: str, style: str = "") -> None:
        """Print with optional Rich styling."""
        if RICH and style:
            console.print(f"[{style}]{msg}[/{style}]")
        else:
            print(msg)

    def header(self, title: str) -> None:
        """Print a command header."""
        if RICH:
            console.print(Panel.fit(f"[bold]{title}[/bold]", border_style="blue"))
        else:
            print(f"\n{'=' * 50}")
            print(title)
            print("=" * 50)

    def section(self, title: str) -> None:
        """Print a section header."""
        if RICH:
            console.print(f"\n[bold cyan]{title}[/bold cyan]")
        else:
            print(f"\n{title}")
            print("-" * len(title))

    def step(self, num: int, total: int, msg: str) -> None:
        """Print a step indicator."""
        if RICH:
            console.print(f"[bold blue][{num}/{total}][/bold blue] {msg}")
        else:
            print(f"[{num}/{total}] {msg}")

    def success(self, msg: str) -> None:
        """Print a success message."""
        if RICH:
            console.print(f"[green]✓[/green] {msg}")
        else:
            print(f"✓ {msg}")

    def error(self, msg: str) -> None:
        """Print an error message."""
        if RICH:
            console.print(f"[red]✗[/red] {msg}")
        else:
            print(f"✗ {msg}")

    def warning(self, msg: str, detail: str = "") -> None:
        """Print a warning message."""
        if RICH:
            detail_str = f" [dim]({detail})[/dim]" if detail else ""
            console.print(f"[yellow]⚠[/yellow] {msg}{detail_str}")
        else:
            detail_str = f" ({detail})" if detail else ""
            print(f"⚠ {msg}{detail_str}")

    def info(self, msg: str) -> None:
        """Print an info/dim message."""
        if RICH:
            console.print(f"[dim]{msg}[/dim]")
        else:
            print(f"  {msg}")

    def status(self, name: str, ok: bool, detail: str = "") -> None:
        """Print a status line (check/x with name and optional detail)."""
        if RICH:
            icon = "✓" if ok else "✗"
            color = "green" if ok else "red"
            detail_str = f" [dim]({detail})[/dim]" if detail else ""
            console.print(f"  [{color}]{icon}[/{color}] {name}{detail_str}")
        else:
            icon = "✓" if ok else "✗"
            detail_str = f" ({detail})" if detail else ""
            print(f"  {icon} {name}{detail_str}")

    def panel(self, content: str, title: str = "", style: str = "blue") -> None:
        """Print content in a panel."""
        if RICH:
            console.print(Panel(content, title=title, border_style=style))
        else:
            if title:
                print(f"\n--- {title} ---")
            print(content)
            if title:
                print("-" * (len(title) + 8))

    def rule(self, title: str = "") -> None:
        """Print a horizontal rule."""
        if RICH:
            console.rule(title)
        else:
            if title:
                print(f"\n--- {title} ---")
            else:
                print("-" * 50)

    # -------------------------------------------------------------------------
    # Prompt Methods
    # -------------------------------------------------------------------------

    def prompt_text(self, prompt: str, default: str = "") -> str:
        """Prompt for text input."""
        if RICH:
            if default:
                return Prompt.ask(prompt, default=default)
            else:
                return Prompt.ask(prompt)
        else:
            if default:
                response = input(f"{prompt} [{default}]: ").strip()
                return response if response else default
            else:
                return input(f"{prompt}: ").strip()

    def prompt_int(self, prompt: str, default: int) -> int:
        """Prompt for integer input."""
        if RICH:
            return IntPrompt.ask(prompt, default=default)
        else:
            while True:
                response = input(f"{prompt} [{default}]: ").strip()
                if not response:
                    return default
                try:
                    return int(response)
                except ValueError:
                    print("Please enter a valid number.")

    def prompt_choice(self, prompt: str, choices: list[str], default: str = "") -> str:
        """Prompt for choice from list."""
        if not default and choices:
            default = choices[0]

        if RICH:
            return Prompt.ask(prompt, choices=choices, default=default)
        else:
            choices_str = "/".join(choices)
            while True:
                response = input(f"{prompt} [{choices_str}] ({default}): ").strip()
                if not response:
                    return default
                if response in choices:
                    return response
                print(f"Choose from: {', '.join(choices)}")

    def prompt_confirm(self, prompt: str, default: bool = True) -> bool:
        """Prompt for yes/no confirmation."""
        if RICH:
            return Confirm.ask(prompt, default=default)
        else:
            yn = "Y/n" if default else "y/N"
            response = input(f"{prompt} [{yn}]: ").strip().lower()
            if not response:
                return default
            return response in ("y", "yes")

    def prompt_path(
        self,
        prompt: str,
        default: str = ".",
        must_exist: bool = True,
    ) -> Path:
        """Prompt for a path with optional validation."""
        while True:
            if RICH:
                path_str = Prompt.ask(prompt, default=default)
            else:
                response = input(f"{prompt} [{default}]: ").strip()
                path_str = response if response else default

            path = Path(path_str).expanduser().resolve()

            if not must_exist or path.exists():
                return path
            else:
                self.error(f"Path does not exist: {path}")
                if not self.prompt_confirm("Try again?", default=True):
                    raise typer.Exit(1)

    # -------------------------------------------------------------------------
    # Table Methods
    # -------------------------------------------------------------------------

    def table(
        self,
        headers: list[str],
        rows: list[list[str]],
        title: str = "",
    ) -> None:
        """Print a table."""
        if RICH:
            table = Table(title=title) if title else Table()
            for header in headers:
                table.add_column(header)
            for row in rows:
                table.add_row(*row)
            console.print(table)
        else:
            if title:
                print(f"\n{title}")

            # Simple text table
            col_widths = [len(h) for h in headers]
            for row in rows:
                for i, cell in enumerate(row):
                    col_widths[i] = max(col_widths[i], len(str(cell)))

            # Header
            header_line = "  ".join(
                h.ljust(col_widths[i]) for i, h in enumerate(headers)
            )
            print(header_line)
            print("-" * len(header_line))

            # Rows
            for row in rows:
                row_line = "  ".join(
                    str(c).ljust(col_widths[i]) for i, c in enumerate(row)
                )
                print(row_line)

    # -------------------------------------------------------------------------
    # Code/Syntax Display
    # -------------------------------------------------------------------------

    def syntax(self, code: str, language: str = "yaml") -> None:
        """Print syntax-highlighted code."""
        if RICH:
            console.print(Syntax(code, language, theme="monokai", line_numbers=True))
        else:
            print(code)

    def markdown(self, text: str) -> None:
        """Print markdown-formatted text."""
        if RICH:
            console.print(Markdown(text))
        else:
            print(text)

    # -------------------------------------------------------------------------
    # Progress
    # -------------------------------------------------------------------------

    def progress(
        self,
        description: str = "Working...",
        total: Optional[int] = None,
    ):
        """
        Create a progress context manager.

        Usage:
            with ui.progress("Processing", total=100) as update:
                for i in range(100):
                    do_work()
                    update(1)

        Returns a callable that advances the progress bar.
        """
        if RICH and total:
            return _RichProgress(description, total)
        else:
            return _PlainProgress(description, total)


class _RichProgress:
    """Rich progress bar context manager."""

    def __init__(self, description: str, total: int):
        self.description = description
        self.total = total
        self.progress = None
        self.task = None

    def __enter__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        )
        self.progress.__enter__()
        self.task = self.progress.add_task(self.description, total=self.total)
        return self._update

    def _update(self, advance: int = 1, description: str = None):
        if description:
            self.progress.update(self.task, description=description, advance=advance)
        else:
            self.progress.update(self.task, advance=advance)

    def __exit__(self, *args):
        self.progress.__exit__(*args)


class _PlainProgress:
    """Plain text progress fallback."""

    def __init__(self, description: str, total: Optional[int]):
        self.description = description
        self.total = total
        self.current = 0
        self.last_printed = 0

    def __enter__(self):
        print(f"{self.description}")
        return self._update

    def _update(self, advance: int = 1, description: str = None):
        self.current += advance
        # Print every 10% or every 20 items
        if self.total:
            pct = int(self.current * 100 / self.total)
            if pct >= self.last_printed + 10:
                print(f"  {self.current}/{self.total} ({pct}%)")
                self.last_printed = pct
        elif self.current % 20 == 0:
            print(f"  Processed {self.current}...")

    def __exit__(self, *args):
        if self.total:
            print(f"  {self.current}/{self.total} (100%)")


# =============================================================================
# Singleton Instance
# =============================================================================

ui = UI()

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ui",
    "console",
    "RICH",
    "UI",
    # Re-export Rich components for advanced use
    "Panel",
    "Table",
    "Syntax",
    "Markdown",
    "Progress",
]
