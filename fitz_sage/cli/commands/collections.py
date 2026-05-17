# fitz_sage/cli/commands/collections.py
"""
Collection management command.

Usage:
    fitz collections   # Interactive mode
"""

from __future__ import annotations

from typing import Any

from fitz_sage.cli.ui import RICH, console, ui
from fitz_sage.logging.logger import get_logger
from fitz_sage.services import FitzService

logger = get_logger(__name__)


def _display_collections_table(collections: list[dict[str, Any]]) -> None:
    """Display collections in a table."""
    if RICH:
        from rich.table import Table

        table = Table(show_header=True, header_style="bold")
        table.add_column("#", style="dim", width=3)
        table.add_column("Collection", style="cyan")
        table.add_column("Items", justify="right")

        for i, coll in enumerate(collections, 1):
            table.add_row(
                str(i),
                coll["name"],
                str(coll.get("count", "?")),
            )

        console.print(table)
    else:
        for i, coll in enumerate(collections, 1):
            print(f"  {i}. {coll['name']} ({coll.get('count', '?')} items)")


def _display_collection_info(name: str, chunk_count: int, metadata: dict[str, Any]) -> None:
    """Display detailed collection info."""
    print()
    if RICH:
        from rich.panel import Panel
        from rich.table import Table

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold")
        table.add_column("Value", style="cyan")

        table.add_row("Name", name)
        table.add_row("Items", str(chunk_count))

        console.print(Panel(table, title=f"[bold]{name}[/bold]", border_style="blue"))
    else:
        print(f"  Name: {name}")
        print(f"  Items: {chunk_count}")


def _delete_table_registry(collection: str) -> None:
    """Delete table registry file associated with a collection."""
    from fitz_sage.core.paths import FitzPaths

    registry_path = FitzPaths.table_registry(collection)
    if registry_path.exists():
        try:
            registry_path.unlink()
            ui.info(f"Deleted table registry: {registry_path.name}")
        except Exception as e:
            logger.warning(f"Failed to delete table registry: {e}")


def command() -> None:
    """
    Manage collections.

    Interactive mode - browse, inspect, and delete collections.
    """
    ui.header("Collections", "Manage collections")

    service = FitzService()

    # =========================================================================
    # Step 1: List Collections
    # =========================================================================

    print()
    collection_infos = service.list_collections()

    if not collection_infos:
        ui.info("No collections found.")
        ui.info("Run 'fitz query \"question\" --source ./docs' to create one.")
        return

    # Convert to display format
    collections = [{"name": c.name, "count": c.chunk_count} for c in collection_infos]
    _display_collections_table(collections)
    print()

    # =========================================================================
    # Step 2: Select Collection
    # =========================================================================

    collection_names = [c["name"] for c in collections]
    selected_collection = ui.prompt_numbered_choice(
        "Select collection",
        collection_names + ["Exit"],
        collection_names[0],
    )

    if selected_collection == "Exit":
        return

    # =========================================================================
    # Step 3: Collection Menu
    # =========================================================================

    while True:
        # Get fresh info
        try:
            info = service.get_collection(selected_collection)
            chunk_count = info.chunk_count
            metadata = info.metadata
        except Exception:
            chunk_count = 0
            metadata = {}

        _display_collection_info(selected_collection, chunk_count, metadata)

        print()
        action = ui.prompt_numbered_choice(
            "Action",
            ["Delete collection", "Back to list", "Exit"],
            "Back to list",
        )

        if action == "Delete collection":
            ui.warning(f"This will delete '{selected_collection}' with {chunk_count} items.")

            if ui.prompt_confirm("Are you sure?", default=False):
                try:
                    service.delete_collection(selected_collection)
                    ui.success(f"Deleted '{selected_collection}'")

                    # Also delete associated table registry
                    _delete_table_registry(selected_collection)

                    return  # Exit after deletion
                except Exception as e:
                    ui.error(f"Failed to delete: {e}")
            else:
                ui.info("Cancelled.")

        elif action == "Back to list":
            # Refresh and show list again
            print()
            collection_infos = service.list_collections()
            if not collection_infos:
                ui.info("No collections remaining.")
                return

            collections = [{"name": c.name, "count": c.chunk_count} for c in collection_infos]
            _display_collections_table(collections)
            print()

            collection_names = [c["name"] for c in collections]
            selected_collection = ui.prompt_numbered_choice(
                "Select collection",
                collection_names + ["Exit"],
                collection_names[0],
            )

            if selected_collection == "Exit":
                return

        else:  # Exit
            return
