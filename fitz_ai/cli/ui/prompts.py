# fitz_ai/cli/ui/prompts.py
"""
Prompt methods for user input.

Provides text, number, confirm, and choice prompts with Rich fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .console import ARROW, RICH, Confirm, IntPrompt, Prompt, console


class PromptMixin:
    """Mixin providing prompt methods for the UI class."""

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
        indent: bool = True,
    ) -> str:
        """
        Prompt for a numbered choice selection.

        The default option is always shown at position [1].
        """
        if not choices:
            return default

        # Ensure default is in choices, fallback to first
        if default not in choices:
            default = choices[0]

        # Reorder choices: default first, then the rest in original order
        ordered_choices = [default] + [c for c in choices if c != default]

        # Indentation
        p_indent = "  " if indent else ""
        c_indent = "  " if indent else ""

        # Print prompt and choices
        if RICH:
            console.print(f"{p_indent}[bold]{prompt}:[/bold]")
            for i, choice in enumerate(ordered_choices, 1):
                if choice == default:
                    console.print(
                        f"{p_indent}{c_indent}[cyan][{i}][/cyan] {choice} [dim](default)[/dim]"
                    )
                else:
                    console.print(f"{p_indent}{c_indent}[cyan][{i}][/cyan] {choice}")
        else:
            print(f"{p_indent}{prompt}:")
            for i, choice in enumerate(ordered_choices, 1):
                if choice == default:
                    print(f"{p_indent}{c_indent}[{i}] {choice} (default)")
                else:
                    print(f"{p_indent}{c_indent}[{i}] {choice}")

        # Get user input (default is always 1)
        while True:
            if RICH:
                response = Prompt.ask(f"{p_indent}Choice", default="1")
            else:
                response = input(f"{p_indent}Choice [1]: ").strip()
                if not response:
                    response = "1"

            try:
                idx = int(response)
                if 1 <= idx <= len(ordered_choices):
                    selected = ordered_choices[idx - 1]
                    if RICH:
                        console.print(f"{p_indent}[dim]{ARROW} {selected}[/dim]")
                    return selected
                else:
                    if RICH:
                        console.print(f"{p_indent}[red]Please enter 1-{len(ordered_choices)}[/red]")
                    else:
                        print(f"{p_indent}Please enter 1-{len(ordered_choices)}")
            except ValueError:
                if RICH:
                    console.print(f"{p_indent}[red]Please enter a number[/red]")
                else:
                    print(f"{p_indent}Please enter a number")

    def prompt_multi_select(
        self,
        prompt: str,
        choices: list[tuple[str, str]],
        defaults: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Prompt for multi-select with toggle.

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
                    console.print(f"    [cyan][{i}][/cyan] [{check}] {name} [dim]- {desc}[/dim]")
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
                print(f"\n  Toggle (1-{len(choices)}), 'a' for all, 'n' for none, Enter to confirm")
                response = input("  : ").strip().lower()

            if response == "":
                # Confirm selection
                result = [name for name in names if name in selected]
                if RICH:
                    if result:
                        console.print(f"  [dim]{ARROW} {', '.join(result)}[/dim]")
                    else:
                        console.print(f"  [dim]{ARROW} (none)[/dim]")
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
