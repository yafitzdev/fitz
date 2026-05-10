# fitz_sage/cli/commands/query.py
"""
Query command - Engine-agnostic query interface.

Uses the default engine (set via 'fitz engine'). Override with --engine.

Usage:
    fitz query "What is RAG?"                      # Query existing collection
    fitz query "What is RAG?" --source ./docs      # Point at docs first, then query
    fitz query -c my_collection                    # Specify collection
    fitz query --engine custom                     # Use a custom engine
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from fitz_sage.cli.ui import RICH, console, display_answer, ui
from fitz_sage.logging.logger import get_logger
from fitz_sage.runtime import create_engine, get_default_engine, get_engine_registry
from fitz_sage.services import FitzService
from fitz_sage.services.fitz_service import CollectionNotFoundError, QueryError

logger = get_logger(__name__)


# =============================================================================
# Helpers
# =============================================================================


def _get_root_cause(exc: Exception) -> str:
    """
    Extract the root cause message from a chain of exceptions.

    Walks the __cause__ chain to find the most specific error message,
    filtering out generic wrapper messages like "Retrieval failed".
    """
    messages = []
    current = exc

    while current is not None:
        msg = str(current)
        # Skip generic wrapper messages
        if msg and msg not in ("Retrieval failed", "Retrieval failed: Retrieval failed"):
            messages.append(msg)
        current = current.__cause__

    # Return the most specific (deepest) meaningful message
    if messages:
        # Prefer the deepest cause that has useful info
        for msg in reversed(messages):
            if "timed out" in msg.lower() or "error" in msg.lower() or len(msg) > 20:
                return msg
        return messages[-1]

    return str(exc)


def _apply_chat_overrides(
    engine_name: str,
    endpoint: Optional[str],
    model: Optional[str],
    api_key_env: Optional[str],
):
    """
    Build a config object with CLI chat overrides applied.

    Returns ``None`` when no overrides are set (the engine then loads
    its own config via the registry). Returns an engine-specific
    config when ``--endpoint`` / ``--model`` / ``--api-key-env`` are
    supplied so the engine binds chat to the user's endpoint without
    needing a YAML edit.

    Currently supports the ``fitz_krag`` engine; other engines fall
    through to the unmodified default config and the flags are a
    no-op (with a UI warning).
    """
    if endpoint is None and model is None and api_key_env is None:
        return None

    if engine_name != "fitz_krag":
        ui.warning(
            f"--endpoint / --model / --api-key-env are not supported "
            f"for engine '{engine_name}'. Ignoring."
        )
        return None

    from fitz_sage.runtime.registry import get_engine_registry

    registry = get_engine_registry()
    base_config = registry.load_config("fitz_krag", None)

    # base_config is a FitzKragConfig (pydantic). Build a mutated copy.
    overrides: Dict[str, Any] = {}

    if endpoint is not None:
        overrides["chat_base_url"] = endpoint

    if model is not None:
        # Apply spec to all three tiers — typical for a single
        # llama-server with one loaded model.
        spec = f"endpoint/{model}"
        overrides["chat_fast"] = spec
        overrides["chat_balanced"] = spec
        overrides["chat_smart"] = spec
    elif endpoint is not None:
        # User gave --endpoint but no --model. Reuse the configured
        # chat_smart model name, but route it via the new endpoint.
        # If the existing spec isn't endpoint/<model>, switch it.
        existing = base_config.chat_smart
        if "/" in existing:
            _, model_name = existing.split("/", 1)
        else:
            model_name = existing
        spec = f"endpoint/{model_name}"
        overrides["chat_fast"] = spec
        overrides["chat_balanced"] = spec
        overrides["chat_smart"] = spec

    if api_key_env is not None:
        overrides["chat_api_key_env"] = api_key_env

    return base_config.model_copy(update=overrides)


# =============================================================================
# Main Command
# =============================================================================


def command(
    question: Optional[str] = None,
    source: Optional[Path] = None,
    collection: Optional[str] = None,
    engine: Optional[str] = None,
    chat: bool = False,
    endpoint: Optional[str] = None,
    model: Optional[str] = None,
    api_key_env: Optional[str] = None,
) -> None:
    """Query your knowledge base."""
    # =========================================================================
    # First-run setup (auto-detect providers if no config exists)
    # =========================================================================

    from fitz_sage.core.firstrun import needs_firstrun, run_firstrun_setup

    # Skip first-run if user supplied --endpoint — they're explicitly
    # configuring at the CLI, no detection needed.
    if needs_firstrun() and endpoint is None:
        if not run_firstrun_setup():
            raise typer.Exit(1)

    # =========================================================================
    # Engine selection (use default if not specified)
    # =========================================================================

    registry = get_engine_registry()

    if engine is None:
        engine = get_default_engine()
    elif engine not in registry.list():
        ui.error(f"Unknown engine: '{engine}'. Available: {', '.join(registry.list())}")
        raise typer.Exit(1)

    # =========================================================================
    # Capabilities-based routing
    # =========================================================================

    caps = registry.get_capabilities(engine)

    # Engines that need documents loaded first can't do standalone queries
    if caps.requires_documents_at_query:
        _show_documents_required_message(engine, caps)
        raise typer.Exit(0)

    # Validate --source path if provided
    if source is not None and not source.exists():
        ui.error(f"Path does not exist: {source}")
        raise typer.Exit(1)

    # Build CLI override config (None if no flags set).
    override_config = _apply_chat_overrides(engine, endpoint, model, api_key_env)

    # Engines with persistent ingest support
    if caps.supports_persistent_ingest:
        _run_persistent_ingest_query(
            question,
            source,
            collection,
            engine,
            chat=chat,
            override_config=override_config,
        )
    # Engines with collection support use FitzService
    elif caps.supports_collections:
        _run_collection_query(question, collection, engine, override_config=override_config)
    else:
        # Engines without collections use generic runtime path
        _run_generic_query(question, engine, override_config=override_config)


def _show_documents_required_message(engine_name: str, caps) -> None:
    """Show message for engines that require documents at query time."""
    if caps.cli_query_message:
        ui.warning(caps.cli_query_message)
    else:
        ui.warning(f"Engine '{engine_name}' requires documents to be loaded first.")
        ui.info(f"Use 'fitz query \"question\" --source <folder> --engine {engine_name}' instead.")
    print()
    ui.info("Or use the Python API:")
    print()
    print("  from fitz_sage.runtime import create_engine")
    print("  from fitz_sage.core import Query")
    print()
    print(f"  engine = create_engine('{engine_name}')")
    print("  engine.add_documents(['doc1...', 'doc2...'])")
    print("  answer = engine.answer(Query(text='your question'))")
    print()


def _select_collection(service: FitzService, requested: Optional[str]) -> str:
    """Select collection interactively or use requested."""
    collections = service.list_collections()

    if not collections:
        print()
        ui.warning("No collections found in vector database.")
        ui.info("Run 'fitz query \"question\" --source ./docs' to get started.")
        raise typer.Exit(0)

    collection_names = [c.name for c in collections]

    if requested is not None:
        if requested not in collection_names:
            print()
            ui.warning(f"Collection '{requested}' not found.")
            ui.info(f"Available collections: {', '.join(collection_names)}")
            ui.info(
                "Use -c <collection> to specify another, or use --source to register documents."
            )
            raise typer.Exit(0)
        return requested

    # Auto-select if only one
    if len(collection_names) == 1:
        return collection_names[0]

    # Prompt user
    print()
    return ui.prompt_numbered_choice("Collection", collection_names, collection_names[0])


def _run_persistent_ingest_query(
    question: Optional[str],
    source: Optional[Path],
    collection: Optional[str],
    engine_name: str,
    *,
    chat: bool = False,
    override_config: Any = None,
) -> None:
    """Run query using an engine with persistent ingest support."""

    # If --source provided, point first (auto-creates collection)
    if source is not None:
        collection = collection or "default"
    else:
        # No source — need an existing collection
        registry = get_engine_registry()
        collections = registry.get_list_collections(engine_name)

        if not collections:
            ui.warning("No collections found.")
            ui.info("Run 'fitz query \"question\" --source ./docs' to get started.")
            raise typer.Exit(0)

        if collection is None:
            if len(collections) == 1:
                collection = collections[0]
            else:
                print()
                collection = ui.prompt_numbered_choice("Collection", collections, collections[0])
        elif collection not in collections:
            ui.error(f"Collection '{collection}' not found. Available: {', '.join(collections)}")
            raise typer.Exit(1)

    # Prompt for question if not provided (and not chat mode)
    if not chat:
        if question is None:
            question_text = ui.prompt_text("Question")
        else:
            question_text = question

    print()

    # Create engine and load/point collection
    try:
        import time

        wall_start = time.perf_counter()

        engine_instance = create_engine(engine_name, config=override_config)
        t_engine = time.perf_counter() - wall_start

        # Point at source if provided
        if source is not None:
            t0 = time.perf_counter()
            ui.info(f"Registering {source}...")
            manifest = engine_instance.point(
                source, collection, start_worker=False, progress=ui.info
            )
            t_point = time.perf_counter() - t0
            ui.info(f"Registered {len(manifest.entries())} files")
        else:
            t0 = time.perf_counter()
            ui.info(f"Loading collection '{collection}'...")
            engine_instance.load(collection)
            t_point = time.perf_counter() - t0

        if chat:
            _chat_loop(engine_instance, collection)
        else:
            from fitz_sage.core import Query

            query = Query(text=question_text)
            answer = engine_instance.answer(query, progress=ui.info)
            wall_total = time.perf_counter() - wall_start
            ui.info(
                f"Total wall-clock: {wall_total:.1f}s "
                f"(engine={t_engine:.1f}s, register={t_point:.1f}s)"
            )
            display_answer(answer)
    except Exception as e:
        # Show clean error message, full traceback only at debug level
        ui.error(f"Query failed: {_get_root_cause(e)}")
        logger.debug("Query error", exc_info=True)
        raise typer.Exit(1)


def _run_collection_query(
    question: Optional[str],
    collection: Optional[str],
    engine_name: str,
    *,
    override_config: Any = None,
) -> None:
    """Run query using FitzService for fitz_krag engine."""
    if override_config is not None:
        ui.warning(
            "--endpoint / --model overrides are not yet plumbed through "
            "FitzService; ignoring. Use --source to use overrides via "
            "the persistent-ingest path."
        )
    service = FitzService()

    # Collection selection
    selected_collection = _select_collection(service, collection)

    # Prompt for question if not provided
    if question is None:
        question_text = ui.prompt_text("Question")
    else:
        question_text = question

    # Display info
    print()
    ui.info(f"Engine: {engine_name} | Collection: {selected_collection}")
    print()

    # Execute query via FitzService
    try:
        answer = service.query(
            question=question_text,
            collection=selected_collection,
            engine=engine_name,
        )
        display_answer(answer)
    except CollectionNotFoundError as e:
        ui.error(f"Collection '{e.collection}' not found.")
        ui.info("Run 'fitz collections' to see available collections.")
        raise typer.Exit(1)
    except QueryError as e:
        # Show clean error message, full traceback only at debug level
        ui.error(f"Query failed: {_get_root_cause(e)}")
        logger.debug("Query error", exc_info=True)
        raise typer.Exit(1)


def _run_generic_query(
    question: Optional[str],
    engine_name: str,
    *,
    override_config: Any = None,
) -> None:
    """Run query using generic runtime path."""
    # Prompt for question if not provided
    if question is None:
        question_text = ui.prompt_text("Question")
    else:
        question_text = question

    print()
    ui.info(f"Using engine: {engine_name}")
    print()

    try:
        engine_instance = create_engine(engine_name, config=override_config)
        from fitz_sage.core import Query

        query = Query(text=question_text)
        answer = engine_instance.answer(query)
        display_answer(answer)
    except Exception as e:
        # Show clean error message, full traceback only at debug level
        ui.error(f"Query failed: {_get_root_cause(e)}")
        logger.debug("Query error", exc_info=True)
        raise typer.Exit(1)


# =============================================================================
# Chat Mode
# =============================================================================


def _chat_loop(engine: Any, collection: str) -> None:
    """Interactive chat loop with conversation history."""
    from fitz_sage.core import Query

    if RICH:
        console.print(
            f"\n[bold green]Chat started[/bold green] with collection: [cyan]{collection}[/cyan]"
        )
        console.print("[dim]Type 'exit' or 'quit' to end. Press Ctrl+C to interrupt.[/dim]\n")
    else:
        print(f"\nChat started with collection: {collection}")
        print("Type 'exit' or 'quit' to end. Press Ctrl+C to interrupt.\n")

    history: List[Dict[str, str]] = []

    try:
        while True:
            try:
                if RICH:
                    from rich.prompt import Prompt

                    user_input = Prompt.ask("\n[bold green]You[/bold green]")
                else:
                    user_input = input("\nYou: ").strip()
            except EOFError:
                break

            if user_input.lower() in ("exit", "quit", "q"):
                break

            if not user_input.strip():
                continue

            try:
                answer = engine.answer(Query(text=user_input), progress=ui.info)
            except Exception as e:
                ui.error(f"Query failed: {e}")
                logger.debug("Query error", exc_info=True)
                continue

            # Display response
            if RICH:
                from rich.markdown import Markdown
                from rich.padding import Padding
                from rich.panel import Panel

                console.print()
                panel = Panel(
                    Markdown(answer.text),
                    title="[bold cyan]Assistant[/bold cyan]",
                    title_align="left",
                    border_style="cyan",
                    padding=(0, 1),
                )
                console.print(Padding(panel, (0, 0, 0, 12)))
            else:
                print(f"\nAssistant: {answer.text}")

            # Show sources
            if answer.provenance:
                from fitz_sage.cli.ui import display_sources
                from fitz_sage.core.chunk import Chunk

                chunks = [
                    Chunk(
                        content=p.excerpt or "",
                        metadata={"source": p.source_id},
                    )
                    for p in answer.provenance
                    if p.excerpt
                ]
                if chunks:
                    display_sources(chunks, indent=12)

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": answer.text})

    except KeyboardInterrupt:
        pass

    if RICH:
        console.print("\n[dim]Chat ended. Goodbye![/dim]")
    else:
        print("\nChat ended. Goodbye!")
