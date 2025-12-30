# fitz_ai/cli/ui/engine_selection.py
"""
Engine selection with animated card UI.

Provides the engine selection prompt with horizontal cards.
"""

from __future__ import annotations

from .console import RICH, Columns, Control, Panel, Prompt, console


class EngineSelectionMixin:
    """Mixin providing engine selection for the UI class."""

    def prompt_engine_selection(
        self,
        engines: list[str],
        descriptions: dict[str, str],
        default: str,
    ) -> str:
        """
        Prompt for engine selection with horizontal cards.

        Displays each engine as a card with number and description.
        The default engine is highlighted and shown first.
        User can cycle through selections; Enter confirms.

        Args:
            engines: List of engine names
            descriptions: Dict mapping engine name to description
            default: Default engine name

        Returns:
            Selected engine name
        """
        if not engines:
            return default

        # Ensure default is valid
        if default not in engines:
            default = engines[0]

        # Reorder: default first, then the rest
        ordered_engines = [default] + [e for e in engines if e != default]

        if RICH:

            def _build_cards(selected_engine: str):
                """Build cards renderable."""
                cards = []
                for i, engine in enumerate(ordered_engines, 1):
                    desc = descriptions.get(engine, "")
                    # Truncate to fit horizontally (~20 chars)
                    if len(desc) > 20:
                        desc = desc[:17] + "..."

                    if engine == selected_engine:
                        content = f"[bold cyan][{i}][/bold cyan] [bold]{engine}[/bold]\n[dim]{desc}[/dim]\n[cyan](selected)[/cyan]"
                        style = "cyan"
                    else:
                        content = f"[dim][{i}][/dim] {engine}\n[dim]{desc}[/dim]"
                        style = "dim"

                    cards.append(Panel.fit(content, border_style=style, padding=(0, 1)))

                return Columns(cards, equal=False, expand=False)

            def _clear_and_redraw(lines_to_clear: int):
                """Clear previous output and prepare for redraw."""
                console.control(Control.move_to_column(0))
                for _ in range(lines_to_clear):
                    console.control(Control.move(0, -1))
                    console.print(" " * console.width, end="")
                    console.control(Control.move_to_column(0))

            # Show initial cards, get input, cycle until confirmed
            selected = default

            # Print cards initially
            console.print(_build_cards(selected))
            console.print()

            while True:
                response = Prompt.ask("Select engine, Enter to confirm", default="")

                # Empty input = confirm current selection
                if response == "":
                    # Clear the cards and prompt before returning
                    # Cards = 5 lines, blank = 1, prompt = 1 = 7 lines total
                    _clear_and_redraw(7)

                    # Print final cards (without prompt)
                    console.print(_build_cards(selected))
                    return selected

                try:
                    idx = int(response)
                    if 1 <= idx <= len(ordered_engines):
                        selected = ordered_engines[idx - 1]

                        # Clear previous cards and prompt
                        _clear_and_redraw(7)

                        # Print updated cards and continue loop
                        console.print(_build_cards(selected))
                        console.print()
                    else:
                        console.print(f"[red]Please enter 1-{len(ordered_engines)}[/red]")
                except ValueError:
                    console.print("[red]Please enter a number[/red]")
        else:
            # Plain text fallback
            print("\nAvailable engines:")
            for i, engine in enumerate(ordered_engines, 1):
                desc = descriptions.get(engine, "")
                default_marker = " (default)" if engine == default else ""
                print(f"  [{i}] {engine}{default_marker}")
                if desc:
                    print(f"      {desc}")

            while True:
                response = input("Select engine [1]: ").strip()
                if not response:
                    response = "1"
                try:
                    idx = int(response)
                    if 1 <= idx <= len(ordered_engines):
                        return ordered_engines[idx - 1]
                    else:
                        print(f"Please enter 1-{len(ordered_engines)}")
                except ValueError:
                    print("Please enter a number")
