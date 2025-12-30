# fitz_ai/cli/ui/progress.py
"""
Progress indicators and spinners.

Provides progress bars and spinners with Rich fallback.
"""

from __future__ import annotations

from typing import Optional

from .console import RICH, console

# Import progress components directly from rich
try:
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )
except ImportError:
    pass


class RichSpinner:
    """Rich spinner context manager."""

    def __init__(self, message: str):
        self.message = message
        self.status = None

    def __enter__(self):
        from rich.status import Status

        self.status = Status(self.message, console=console, spinner="dots")
        self.status.__enter__()
        return self

    def __exit__(self, *args):
        self.status.__exit__(*args)


class PlainSpinner:
    """Plain text spinner fallback."""

    def __init__(self, message: str):
        self.message = message

    def __enter__(self):
        print(f"{self.message}", end="", flush=True)
        return self

    def __exit__(self, *args):
        print(" done")


class RichProgress:
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


class PlainProgress:
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


class ProgressMixin:
    """Mixin providing progress methods for the UI class."""

    def progress(self, description: str = "Working...", total: Optional[int] = None):
        """
        Create a progress context manager.

        Usage:
            with ui.progress("Processing", total=100) as update:
                for i in range(100):
                    do_work()
                    update(1)
        """
        if RICH and total:
            return RichProgress(description, total)
        else:
            return PlainProgress(description, total)

    def spinner(self, message: str = "Working..."):
        """
        Create a spinner context manager for indeterminate progress.

        Usage:
            with ui.spinner("Loading model..."):
                load_model()
        """
        if RICH:
            return RichSpinner(message)
        else:
            return PlainSpinner(message)
