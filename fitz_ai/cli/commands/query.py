# fitz_ai/cli/commands/query.py
"""
Query command - Engine-agnostic query interface.

Usage:
    fitz query                          # Interactive mode (classic_rag)
    fitz query "What is RAG?"           # Direct query
    fitz query -c my_collection         # Specify collection
    fitz query --engine clara           # Use CLaRa engine
"""

from __future__ import annotations

from typing import Optional

import typer

from fitz_ai.cli.ui import ui
from fitz_ai.cli.ui_display import display_answer
from fitz_ai.cli.utils import get_collections, load_classic_rag_config
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
        help="Collection to query (classic_rag only).",
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

    Specify a collection (classic_rag only):
        fitz query "question" -c my_collection
    """
    # =========================================================================
    # Engine selection
    # =========================================================================

    available_engines = list_engines()
    registry = get_engine_registry()

    if engine is None:
        # Prompt for engine selection
        ui.header("Fitz Query")
        print()

        # Build choices with descriptions
        engine_info = registry.list_with_descriptions()
        choices = []
        for name in available_engines:
            desc = engine_info.get(name, "")
            if len(desc) > 60:
                desc = desc[:57] + "..."
            choices.append(f"{name} - {desc}" if desc else name)

        # Get default engine from config (prompt_numbered_choice puts default at position 1)
        default_engine_name = get_default_engine()
        default_choice = next((c for c in choices if c.startswith(default_engine_name)), choices[0])
        selected = ui.prompt_numbered_choice("Engine", choices, default_choice)
        engine = selected.split(" - ")[0]
        print()
    elif engine not in available_engines:
        ui.error(f"Unknown engine: '{engine}'. Available: {', '.join(available_engines)}")
        raise typer.Exit(1)

    # =========================================================================
    # Header (with engine)
    # =========================================================================

    ui.header("Fitz Query", f"Engine: {engine}")

    # =========================================================================
    # Capabilities-based routing
    # =========================================================================

    registry = get_engine_registry()
    caps = registry.get_capabilities(engine)

    # Engines that need documents loaded first can't do standalone queries
    if caps.requires_documents_at_query:
        _show_documents_required_message(engine, caps)
        raise typer.Exit(0)

    # Engines with collection support use collection-based query
    if caps.supports_collections:
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
        ui.info(f"Use 'fitz quickstart <folder> \"question\" --engine {engine_name}' for one-off queries.")
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


def _run_collection_query(
    question: Optional[str], collection: Optional[str], engine_name: str
) -> None:
    """Run query using an engine with collection support."""

    # Load config
    raw_config, typed_config = load_classic_rag_config()
    if typed_config is None:
        ui.error("No config found. Run 'fitz init' first.")
        raise typer.Exit(1)

    default_collection = typed_config.retrieval.collection

    # Prompt for question if not provided
    if question is None:
        question_text = ui.prompt_text("Question")
    else:
        question_text = question

    # Collection selection
    if collection:
        typed_config.retrieval.collection = collection
    else:
        collections = get_collections(raw_config)
        if collections and len(collections) > 1:
            print()
            selected = ui.prompt_numbered_choice("Collection", collections, default_collection)
            typed_config.retrieval.collection = selected
        elif collections:
            typed_config.retrieval.collection = collections[0]

    # Display info
    display_collection = typed_config.retrieval.collection
    display_retrieval = typed_config.retrieval.plugin_name

    chat_plugin = raw_config.get("chat", {}).get("plugin_name", "?")
    chat_model = raw_config.get("chat", {}).get("kwargs", {}).get("model", "")
    display_chat = f"{chat_plugin} ({chat_model})" if chat_model else chat_plugin

    embedding_plugin = raw_config.get("embedding", {}).get("plugin_name", "?")
    embedding_model = raw_config.get("embedding", {}).get("kwargs", {}).get("model", "")
    display_embedding = (
        f"{embedding_plugin} ({embedding_model})" if embedding_model else embedding_plugin
    )

    display_rerank = None
    if raw_config.get("rerank", {}).get("enabled"):
        rerank_plugin = raw_config.get("rerank", {}).get("plugin_name", "?")
        rerank_model = raw_config.get("rerank", {}).get("kwargs", {}).get("model", "")
        display_rerank = f"{rerank_plugin} ({rerank_model})" if rerank_model else rerank_plugin

    print()
    info_parts = [
        f"Collection: {display_collection}",
        f"Retrieval: {display_retrieval}",
        f"Chat: {display_chat}",
        f"Embedding: {display_embedding}",
    ]
    if display_rerank:
        info_parts.append(f"Rerank: {display_rerank}")

    ui.info(" | ".join(info_parts))
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
