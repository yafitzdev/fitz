# fitz_ai/cli/ui/output.py
"""
Output methods for CLI display.

Provides styled output with Rich fallback support.
"""

from __future__ import annotations

from .console import CHECK, CROSS, RICH, WARN, Markdown, Panel, Syntax, Table, console


class OutputMixin:
    """Mixin providing output methods for the UI class."""

    def print(self, msg: str, style: str = "") -> None:
        """Print with optional Rich styling."""
        if RICH and style:
            console.print(f"[{style}]{msg}[/{style}]")
        else:
            print(msg)

    def header(self, title: str, subtitle: str = "") -> None:
        """
        Print a command header.

        Creates a fitted box around the title for consistency across all CLI commands.
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

    def step_done(self, num: int, total: int, msg: str) -> None:
        """Print a step indicator with inline checkmark (for quick completed steps)."""
        if RICH:
            console.print(f"[bold blue][{num}/{total}][/bold blue] {msg} [green]{CHECK}[/green]")
        else:
            print(f"[{num}/{total}] {msg} {CHECK}")

    def success(self, msg: str) -> None:
        """Print a success message."""
        if RICH:
            console.print(f"[green]{CHECK}[/green] {msg}")
        else:
            print(f"{CHECK} {msg}")

    def error(self, msg: str) -> None:
        """Print an error message."""
        if RICH:
            console.print(f"[red]{CROSS}[/red] {msg}")
        else:
            print(f"{CROSS} {msg}")

    def warning(self, msg: str, detail: str = "") -> None:
        """Print a warning message."""
        if RICH:
            detail_str = f" [dim]({detail})[/dim]" if detail else ""
            console.print(f"[yellow]{WARN}[/yellow] {msg}{detail_str}")
        else:
            detail_str = f" ({detail})" if detail else ""
            print(f"{WARN} {msg}{detail_str}")

    def info(self, msg: str) -> None:
        """Print an info/dim message."""
        if RICH:
            console.print(f"[dim]{msg}[/dim]")
        else:
            print(f"  {msg}")

    def status(self, name: str, ok: bool, detail: str = "") -> None:
        """Print a status line (check/x with name and optional detail)."""
        icon = CHECK if ok else CROSS
        if RICH:
            color = "green" if ok else "red"
            detail_str = f" [dim]({detail})[/dim]" if detail else ""
            console.print(f"  [{color}]{icon}[/{color}] {name}{detail_str}")
        else:
            detail_str = f" ({detail})" if detail else ""
            print(f"  {icon} {name}{detail_str}")

    def panel(self, content: str, title: str = "", style: str = "blue") -> None:
        """Print content in a panel (full-width)."""
        if RICH:
            console.print(Panel(content, title=title, border_style=style))
        else:
            if title:
                print(f"\n--- {title} ---")
            print(content)
            if title:
                print("-" * (len(title) + 8))

    def summary_panel(self, content: str, title: str = "", style: str = "green") -> None:
        """Print a summary panel (full-width, typically at end of command)."""
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

    def table(self, headers: list[str], rows: list[list[str]], title: str = "") -> None:
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
            header_line = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
            print(header_line)
            print("-" * len(header_line))

            # Rows
            for row in rows:
                row_line = "  ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(row))
                print(row_line)

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
