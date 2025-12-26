# fitz_ai/cli/ui.py
"""
Shared UI helpers for CLI v2 commands.

Provides consistent styling and fallback for Rich-less environments.

Usage:
    from fitz_ai.cli.ui import ui, console, RICH

    ui.header("My Command")
    ui.success("Done!")
    name = ui.prompt_text("Enter name", default="default")

    # Numbered choice selection
    choice = ui.prompt_numbered_choice("Select plugin", ["cohere", "local_ollama"], "cohere")
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
# Plugin Utilities
# =============================================================================


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

    def header(self, title: str, subtitle: str = "") -> None:
        """
        Print a command header.

        Creates a fitted box (minimal width) around the title for consistency
        across all CLI commands.

        Args:
            title: Main header title
            subtitle: Optional subtitle (dim text below title)
        """
        if RICH:
            if subtitle:
                content = f"[bold]{title}[/bold]\n[dim]{subtitle}[/dim]"
            else:
                content = f"[bold]{title}[/bold]"
            console.print(Panel.fit(content, border_style="blue"))
        else:
            print(f"\n{'=' * 50}")
            print(title)
            if subtitle:
                print(subtitle)
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
        """
        Print content in a panel (full-width).

        For fitted/minimal boxes, use header() instead.

        Args:
            content: Panel content
            title: Optional panel title
            style: Border style color
        """
        if RICH:
            console.print(Panel(content, title=title, border_style=style))
        else:
            if title:
                print(f"\n--- {title} ---")
            print(content)
            if title:
                print("-" * (len(title) + 8))

    def summary_panel(
        self, content: str, title: str = "", style: str = "green"
    ) -> None:
        """
        Print a summary panel (full-width, typically at end of command).

        Args:
            content: Panel content
            title: Panel title
            style: Border style color
        """
        if RICH:
            console.print(Panel(content, title=title, border_style=style))
        else:
            width = 60
            print("=" * width)
            if title:
                print(f" {title}")
                print("=" * width)
            print(content)
            print("=" * width)

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
            return Prompt.ask(prompt, default=default)
        else:
            response = input(f"{prompt} ({default}): ").strip()
            return response if response else default

    def prompt_int(self, prompt: str, default: int = 0) -> int:
        """Prompt for integer input."""
        if RICH:
            return IntPrompt.ask(prompt, default=default)
        else:
            response = input(f"{prompt} ({default}): ").strip()
            try:
                return int(response) if response else default
            except ValueError:
                return default

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

    def prompt_numbered_choice(
        self,
        prompt: str,
        choices: list[str],
        default: str = "",
    ) -> str:
        """
        Prompt for a numbered choice selection.

        The default option is always shown at position [1].

        Example output:
            Chat plugin:
              [1] cohere (default)
              [2] local_ollama
            Choice [1]:
            → cohere

        Args:
            prompt: Prompt text
            choices: List of choices
            default: Default choice (will be shown first)

        Returns:
            Selected choice string
        """
        if not choices:
            return default

        # Ensure default is in choices, fallback to first
        if default not in choices:
            default = choices[0]

        # Reorder choices: default first, then the rest in original order
        ordered_choices = [default] + [c for c in choices if c != default]

        # Print prompt and choices
        if RICH:
            console.print(f"  [bold]{prompt}:[/bold]")
            for i, choice in enumerate(ordered_choices, 1):
                if choice == default:
                    console.print(
                        f"    [cyan][{i}][/cyan] {choice} [dim](default)[/dim]"
                    )
                else:
                    console.print(f"    [cyan][{i}][/cyan] {choice}")
        else:
            print(f"  {prompt}:")
            for i, choice in enumerate(ordered_choices, 1):
                if choice == default:
                    print(f"    [{i}] {choice} (default)")
                else:
                    print(f"    [{i}] {choice}")

        # Get user input (default is always 1)
        while True:
            if RICH:
                response = Prompt.ask("  Choice", default="1")
            else:
                response = input("  Choice [1]: ").strip()
                if not response:
                    response = "1"

            try:
                idx = int(response)
                if 1 <= idx <= len(ordered_choices):
                    selected = ordered_choices[idx - 1]
                    if RICH:
                        console.print(f"  [dim]→ {selected}[/dim]")
                    return selected
                else:
                    if RICH:
                        console.print(
                            f"  [red]Please enter 1-{len(ordered_choices)}[/red]"
                        )
                    else:
                        print(f"  Please enter 1-{len(ordered_choices)}")
            except ValueError:
                if RICH:
                    console.print("  [red]Please enter a number[/red]")
                else:
                    print("  Please enter a number")

    def prompt_multi_select(
        self,
        prompt: str,
        choices: list[tuple[str, str]],
        defaults: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Prompt for multi-select with toggle.

        Example output:
            Select artifacts:
              [1] [x] navigation_index - File → purpose mapping
              [2] [x] interface_catalog - Protocols with implementations
              [3] [ ] architecture_narrative - High-level overview (requires LLM)
            Toggle (1-3), 'a' for all, 'n' for none, Enter to confirm:
            → navigation_index, interface_catalog

        Args:
            prompt: Prompt text
            choices: List of (name, description) tuples
            defaults: List of names to select by default (None = all selected)

        Returns:
            List of selected choice names
        """
        if not choices:
            return []

        # Initialize selection state
        names = [name for name, _ in choices]
        if defaults is None:
            selected = set(names)  # All selected by default
        else:
            selected = set(defaults)

        def _print_choices():
            if RICH:
                console.print(f"\n  [bold]{prompt}:[/bold]")
                for i, (name, desc) in enumerate(choices, 1):
                    check = "[green]x[/green]" if name in selected else " "
                    console.print(
                        f"    [cyan][{i}][/cyan] [{check}] {name} [dim]- {desc}[/dim]"
                    )
            else:
                print(f"\n  {prompt}:")
                for i, (name, desc) in enumerate(choices, 1):
                    check = "x" if name in selected else " "
                    print(f"    [{i}] [{check}] {name} - {desc}")

        # Interactive loop
        while True:
            _print_choices()

            if RICH:
                console.print(
                    "\n  [dim]Toggle (1-{0}), 'a' for all, 'n' for none, Enter to confirm[/dim]".format(
                        len(choices)
                    )
                )
                response = Prompt.ask("  ", default="").strip().lower()
            else:
                print(
                    f"\n  Toggle (1-{len(choices)}), 'a' for all, 'n' for none, Enter to confirm"
                )
                response = input("  : ").strip().lower()

            if response == "":
                # Confirm selection
                result = [name for name in names if name in selected]
                if RICH:
                    if result:
                        console.print(f"  [dim]→ {', '.join(result)}[/dim]")
                    else:
                        console.print("  [dim]→ (none)[/dim]")
                return result

            elif response == "a":
                # Select all
                selected = set(names)

            elif response == "n":
                # Select none
                selected = set()

            else:
                # Toggle individual items (comma-separated)
                for part in response.split(","):
                    part = part.strip()
                    try:
                        idx = int(part)
                        if 1 <= idx <= len(choices):
                            name = names[idx - 1]
                            if name in selected:
                                selected.remove(name)
                            else:
                                selected.add(name)
                    except ValueError:
                        pass  # Ignore invalid input

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
    # Plugin utilities
    "get_first_available",
    # Re-export Rich components for advanced use
    "Panel",
    "Table",
    "Syntax",
    "Markdown",
    "Progress",
]
