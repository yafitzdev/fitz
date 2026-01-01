# fitz_ai/cli/commands/query.py
"""
Query command - Engine-agnostic query interface.

Usage:
    fitz query                          # Interactive mode (fitz_rag)
    fitz query "What is RAG?"           # Direct query
    fitz query -c my_collection         # Specify collection
    fitz query --engine clara           # Use CLaRa engine
"""

from __future__ import annotations

from typing import Optional

import typer

from fitz_ai.cli.context import CLIContext
from fitz_ai.cli.ui import display_answer, ui
from fitz_ai.core import Query
from fitz_ai.logging.logger import get_logger
from fitz_ai.runtime import create_engine, get_default_engine, get_engine_registry, list_engines

logger = get_logger(__name__)


# =============================================================================
# Main Command
# =============================================================================


def command(
    question: Optional[str] = typer.Argument(
        None,
        help="Question to ask (will prompt if not provided).",
    ),
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        "-c",
        help="Collection to query (fitz_rag only).",
    ),
    engine: Optional[str] = typer.Option(
        None,
        "--engine",
        "-e",
        help="Engine to use. Will prompt if not specified.",
    ),
) -> None:
    """
    Query your knowledge base.

    Run without arguments for interactive mode:
        fitz query

    Or ask directly:
        fitz query "What is RAG?"

    Specify engine:
        fitz query "question" --engine clara

    Specify a collection (fitz_rag only):
        fitz query "question" -c my_collection
    """
    # =========================================================================
    # Header
    # =========================================================================

    ui.header("Fitz Query", "Query your knowledge base")

    # =========================================================================
    # Engine selection
    # =========================================================================

    available_engines = list_engines()
    registry = get_engine_registry()

    if engine is None:
        # Prompt for engine selection with cards
        print()
        engine_descriptions = registry.list_with_descriptions()
        default_engine_name = get_default_engine()
        engine = ui.prompt_engine_selection(
            engines=available_engines,
            descriptions=engine_descriptions,
            default=default_engine_name,
        )
    elif engine not in available_engines:
        ui.error(f"Unknown engine: '{engine}'. Available: {', '.join(available_engines)}")
        raise typer.Exit(1)

    # =========================================================================
    # Capabilities-based routing
    # =========================================================================

    registry = get_engine_registry()
    caps = registry.get_capabilities(engine)

    # Engines that need documents loaded first can't do standalone queries
    if caps.requires_documents_at_query:
        _show_documents_required_message(engine, caps)
        raise typer.Exit(0)

    # Engines with persistent ingest support (graphrag, clara)
    if caps.supports_persistent_ingest:
        _run_persistent_ingest_query(question, collection, engine)
    # Engines with collection support use collection-based query
    elif caps.supports_collections:
        _run_collection_query(question, collection, engine)
    else:
        # Engines without collections use generic runtime path
        _run_generic_query(question, engine)


def _show_documents_required_message(engine_name: str, caps) -> None:
    """Show message for engines that require documents at query time."""
    if caps.cli_query_message:
        ui.warning(caps.cli_query_message)
    else:
        ui.warning(f"Engine '{engine_name}' requires documents to be loaded first.")
        ui.info(
            f"Use 'fitz quickstart <folder> \"question\" --engine {engine_name}' for one-off queries."
        )
    print()
    ui.info("Or use the Python API:")
    print()
    print("  from fitz_ai.runtime import create_engine")
    print("  from fitz_ai.core import Query")
    print()
    print(f"  engine = create_engine('{engine_name}')")
    print("  engine.add_documents(['doc1...', 'doc2...'])")
    print("  answer = engine.answer(Query(text='your question'))")
    print()


def _run_persistent_ingest_query(
    question: Optional[str], collection: Optional[str], engine_name: str
) -> None:
    """Run query using an engine with persistent ingest support (graphrag, clara)."""

    # Get list of available collections via registry (no hardcoded imports)
    registry = get_engine_registry()
    collections = registry.get_list_collections(engine_name)

    if not collections:
        ui.warning(f"No {engine_name} collections found.")
        ui.info(f"Run 'fitz ingest ./docs -e {engine_name}' first to ingest documents.")
        raise typer.Exit(0)

    # Collection selection
    if collection is None:
        if len(collections) == 1:
            collection = collections[0]
            ui.info(f"Using collection: {collection}")
        else:
            print()
            collection = ui.prompt_numbered_choice("Collection", collections, collections[0])
    elif collection not in collections:
        ui.error(f"Collection '{collection}' not found. Available: {', '.join(collections)}")
        raise typer.Exit(1)

    # Prompt for question if not provided
    if question is None:
        question_text = ui.prompt_text("Question")
    else:
        question_text = question

    print()
    ui.info(f"Engine: {engine_name} | Collection: {collection}")
    print()

    # Create engine, load collection, and query
    try:
        engine_instance = create_engine(engine_name)

        ui.info("Loading collection...")
        engine_instance.load(collection)

        ui.info("Querying...")
        query = Query(text=question_text)
        answer = engine_instance.answer(query)
        display_answer(answer)
    except Exception as e:
        ui.error(f"Query failed: {e}")
        logger.exception("Query error")
        raise typer.Exit(1)


def _run_collection_query(
    question: Optional[str], collection: Optional[str], engine_name: str
) -> None:
    """Run query using an engine with collection support."""

    # Load config via CLIContext
    ctx = CLIContext.load_or_none()
    if ctx is None or ctx.typed_config is None:
        ui.error("No config found. Run 'fitz init' first.")
        raise typer.Exit(1)

    typed_config = ctx.typed_config

    # Prompt for question if not provided
    if question is None:
        question_text = ui.prompt_text("Question")
    else:
        question_text = question

    # Collection selection
    if collection:
        typed_config.retrieval.collection = collection
    else:
        collections = ctx.get_collections()
        if collections and len(collections) > 1:
            print()
            selected = ui.prompt_numbered_choice(
                "Collection", collections, ctx.retrieval_collection
            )
            typed_config.retrieval.collection = selected
        elif collections:
            typed_config.retrieval.collection = collections[0]

    # Display info - all the ugly dict.get() chains are now just ctx properties
    print()
    ui.info(ctx.info_line())
    print()

    # Execute query using runtime
    try:
        engine_instance = create_engine(engine_name, config=typed_config)
        query = Query(text=question_text)
        answer = engine_instance.answer(query)
        display_answer(answer)
    except Exception as e:
        ui.error(f"Query failed: {e}")
        logger.exception("Query error")
        raise typer.Exit(1)


def _run_generic_query(question: Optional[str], engine_name: str) -> None:
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
        engine_instance = create_engine(engine_name)
        query = Query(text=question_text)
        answer = engine_instance.answer(query)
        display_answer(answer)
    except Exception as e:
        ui.error(f"Query failed: {e}")
        logger.exception("Query error")
        raise typer.Exit(1)
