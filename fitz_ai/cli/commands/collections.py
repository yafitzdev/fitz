# fitz_ai/cli/commands/collections.py
"""
Collection management command.

Usage:
    fitz collections   # Interactive mode
"""

from __future__ import annotations

from typing import Any, Dict, List

import typer

from fitz_ai.cli.context import CLIContext
from fitz_ai.cli.ui import RICH, console, ui
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


def _get_vector_client(plugin_name: str, kwargs: dict = None):
    """Get vector DB client by plugin name."""
    from fitz_ai.vector_db.registry import get_vector_db_plugin

    try:
        return get_vector_db_plugin(plugin_name, **(kwargs or {}))
    except Exception as e:
        ui.error(f"Failed to connect to {plugin_name}: {e}")
        raise typer.Exit(1)


def _get_available_vector_dbs(ctx: CLIContext) -> List[Dict[str, Any]]:
    """Get list of available vector DBs from config and installed plugins."""
    from fitz_ai.vector_db.registry import available_vector_db_plugins

    # Get configured default
    configured = ctx.vector_db_plugin
    configured_kwargs = ctx.vector_db_kwargs

    # Get all available plugins
    available = available_vector_db_plugins()

    result = []
    for plugin in available:
        entry = {
            "name": plugin,
            "kwargs": configured_kwargs if plugin == configured else {},
            "is_configured": plugin == configured,
        }
        result.append(entry)

    # Sort: configured first, then alphabetically
    result.sort(key=lambda x: (not x["is_configured"], x["name"]))
    return result


def _display_collections_table(collections: List[Dict[str, Any]]) -> None:
    """Display collections in a table."""
    if RICH:
        from rich.table import Table

        table = Table(show_header=True, header_style="bold")
        table.add_column("#", style="dim", width=3)
        table.add_column("Collection", style="cyan")
        table.add_column("Chunks", justify="right")
        table.add_column("Status")

        for i, coll in enumerate(collections, 1):
            table.add_row(
                str(i),
                coll["name"],
                str(coll.get("count", "?")),
                coll.get("status", "ready"),
            )

        console.print(table)
    else:
        for i, coll in enumerate(collections, 1):
            print(f"  {i}. {coll['name']} ({coll.get('count', '?')} chunks)")


def _display_collection_info(name: str, stats: Dict[str, Any]) -> None:
    """Display detailed collection info."""
    print()
    if RICH:
        from rich.panel import Panel
        from rich.table import Table

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold")
        table.add_column("Value", style="cyan")

        table.add_row("Name", name)
        table.add_row("Chunks", str(stats.get("points_count", stats.get("vectors_count", "?"))))
        table.add_row("Vector Size", str(stats.get("vector_size", "?")))
        table.add_row("Status", stats.get("status", "ready"))

        console.print(Panel(table, title=f"[bold]{name}[/bold]", border_style="blue"))
    else:
        print(f"  Name: {name}")
        print(f"  Chunks: {stats.get('points_count', stats.get('vectors_count', '?'))}")
        print(f"  Vector Size: {stats.get('vector_size', '?')}")
        print(f"  Status: {stats.get('status', 'ready')}")


def _display_example_chunks(client: Any, collection: str, limit: int = 3) -> None:
    """Display example chunks from a collection."""
    print()
    ui.info(f"Example chunks from '{collection}':")
    print()

    try:
        # Use scroll to get sample chunks
        if hasattr(client, "scroll"):
            records, _ = client.scroll(collection=collection, limit=limit, offset=0)
            for i, record in enumerate(records, 1):
                payload = record.payload if hasattr(record, "payload") else {}
                content = payload.get("content", payload.get("text", ""))
                doc_id = payload.get("doc_id", record.id if hasattr(record, "id") else "?")

                # Truncate content
                if len(content) > 200:
                    content = content[:200] + "..."

                if RICH:
                    from rich.panel import Panel

                    console.print(
                        Panel(
                            content or "[dim]No content[/dim]",
                            title=f"[bold]#{i}[/bold] {doc_id}",
                            border_style="dim",
                        )
                    )
                else:
                    print(f"  #{i} [{doc_id}]")
                    print(f"     {content}")
                    print()
        else:
            ui.warning("This vector DB doesn't support browsing chunks.")
    except Exception as e:
        ui.error(f"Failed to fetch chunks: {e}")


def command() -> None:
    """
    Manage vector database collections.

    Interactive mode - browse, inspect, and delete collections.
    """
    ui.header("Collections", "Manage vector database collections")

    ctx = CLIContext.load_or_none()
    if ctx is None:
        ui.error("No config found. Run 'fitz init' or 'fitz quickstart' first.")
        raise typer.Exit(1)

    # =========================================================================
    # Step 1: Select Vector DB (if multiple available)
    # =========================================================================

    available_dbs = _get_available_vector_dbs(ctx)

    if not available_dbs:
        ui.error("No vector DB plugins found.")
        raise typer.Exit(1)

    if len(available_dbs) == 1:
        selected_db = available_dbs[0]
        ui.info(f"Vector DB: {selected_db['name']}")
    else:
        # Multiple DBs available - let user choose
        print()
        db_names = [
            f"{db['name']} {'(configured)' if db['is_configured'] else ''}" for db in available_dbs
        ]
        selected_name = ui.prompt_numbered_choice(
            "Select vector database",
            db_names,
            db_names[0],  # Default to configured one
        )
        # Find the selected DB
        selected_idx = db_names.index(selected_name)
        selected_db = available_dbs[selected_idx]

    # Connect to selected DB
    client = _get_vector_client(selected_db["name"], selected_db["kwargs"])

    # =========================================================================
    # Step 2: List Collections
    # =========================================================================

    print()
    collections_list = client.list_collections()

    if not collections_list:
        ui.info("No collections found.")
        ui.info("Run 'fitz ingest' to create one.")
        return

    # Get stats for each collection
    collections = []
    for name in collections_list:
        try:
            stats = client.get_collection_stats(name)
            count = stats.get("points_count", stats.get("vectors_count", "?"))
            status = stats.get("status", "ready")
        except Exception:
            count = "?"
            status = "unknown"

        collections.append(
            {
                "name": name,
                "count": count,
                "status": status,
            }
        )

    _display_collections_table(collections)
    print()

    # =========================================================================
    # Step 3: Select Collection
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
    # Step 4: Collection Menu
    # =========================================================================

    while True:
        # Get fresh stats
        try:
            stats = client.get_collection_stats(selected_collection)
        except Exception:
            stats = {}

        _display_collection_info(selected_collection, stats)

        print()
        action = ui.prompt_numbered_choice(
            "Action",
            ["Show example chunks", "Delete collection", "Back to list", "Exit"],
            "Show example chunks",
        )

        if action == "Show example chunks":
            _display_example_chunks(client, selected_collection)
            print()
            ui.prompt_text("Press Enter to continue", "")

        elif action == "Delete collection":
            count = stats.get("points_count", stats.get("vectors_count", "?"))
            ui.warning(f"This will delete '{selected_collection}' with {count} chunks.")

            if ui.prompt_confirm("Are you sure?", default=False):
                try:
                    deleted = client.delete_collection(selected_collection)
                    ui.success(f"Deleted '{selected_collection}' ({deleted} chunks)")
                    return  # Exit after deletion
                except Exception as e:
                    ui.error(f"Failed to delete: {e}")
            else:
                ui.info("Cancelled.")

        elif action == "Back to list":
            # Refresh and show list again
            print()
            collections_list = client.list_collections()
            if not collections_list:
                ui.info("No collections remaining.")
                return

            collections = []
            for name in collections_list:
                try:
                    stats = client.get_collection_stats(name)
                    count = stats.get("points_count", stats.get("vectors_count", "?"))
                    status = stats.get("status", "ready")
                except Exception:
                    count = "?"
                    status = "unknown"
                collections.append({"name": name, "count": count, "status": status})

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
