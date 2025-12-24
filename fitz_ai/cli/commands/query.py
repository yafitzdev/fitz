# fitz_ai/cli/commands/query.py
"""
Query command.

Usage:
    fitz query                     # Interactive mode
    fitz query "What is RAG?"      # Direct query
    fitz query -c my_collection    # Specify collection
"""

from __future__ import annotations

from typing import List, Optional

import typer

from fitz_ai.cli.ui import ui
from fitz_ai.cli.ui_display import display_answer
from fitz_ai.core.config import ConfigNotFoundError, load_config_dict
from fitz_ai.core.paths import FitzPaths
from fitz_ai.engines.classic_rag.config import ClassicRagConfig, load_config
from fitz_ai.engines.classic_rag.pipeline.engine import RAGPipeline
from fitz_ai.logging.logger import get_logger
from fitz_ai.vector_db.registry import get_vector_db_plugin

logger = get_logger(__name__)


# =============================================================================
# Config Loading
# =============================================================================


def _load_config_safe() -> tuple[dict, ClassicRagConfig]:
    """Load config or exit with helpful message."""
    try:
        config_path = FitzPaths.config()
        raw_config = load_config_dict(config_path)
        typed_config = load_config(config_path)
        return raw_config, typed_config
    except ConfigNotFoundError:
        ui.error("No config found. Run 'fitz init' first.")
        raise typer.Exit(1)
    except Exception as e:
        ui.error(f"Failed to load config: {e}")
        raise typer.Exit(1)


def _get_collections(raw_config: dict) -> List[str]:
    """Get list of collections from vector DB."""
    try:
        vdb_plugin = raw_config.get("vector_db", {}).get("plugin_name", "qdrant")
        vdb_kwargs = raw_config.get("vector_db", {}).get("kwargs", {})
        vdb = get_vector_db_plugin(vdb_plugin, **vdb_kwargs)
        return sorted(vdb.list_collections())
    except Exception:
        return []


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
        help="Collection to query (uses config default if not specified).",
    ),
) -> None:
    """
    Query your knowledge base.

    Run without arguments for interactive mode:
        fitz query

    Or ask directly:
        fitz query "What is RAG?"

    Specify a collection:
        fitz query "question" -c my_collection
    """
    # =========================================================================
    # Header
    # =========================================================================

    ui.header("Fitz Query", "Retrieve answers from your knowledge base")

    # =========================================================================
    # Load config
    # =========================================================================

    raw_config, typed_config = _load_config_safe()
    default_collection = typed_config.retrieval.collection

    # =========================================================================
    # Interactive prompts
    # =========================================================================

    # Prompt for question if not provided
    if question is None:
        question_text = ui.prompt_text("Question")
    else:
        question_text = question

    # Collection selection
    if collection:
        typed_config.retrieval.collection = collection
    else:
        collections = _get_collections(raw_config)
        if collections and len(collections) > 1:
            print()
            selected = ui.prompt_numbered_choice(
                "Collection", collections, default_collection
            )
            typed_config.retrieval.collection = selected
        elif collections:
            typed_config.retrieval.collection = collections[0]

    # Get display info
    display_collection = typed_config.retrieval.collection
    display_retrieval = typed_config.retrieval.plugin_name

    # Chat
    chat_plugin = raw_config.get("chat", {}).get("plugin_name", "?")
    chat_model = raw_config.get("chat", {}).get("kwargs", {}).get("model", "")
    display_chat = f"{chat_plugin} ({chat_model})" if chat_model else chat_plugin

    # Embedding
    embedding_plugin = raw_config.get("embedding", {}).get("plugin_name", "?")
    embedding_model = raw_config.get("embedding", {}).get("kwargs", {}).get("model", "")
    display_embedding = (
        f"{embedding_plugin} ({embedding_model})"
        if embedding_model
        else embedding_plugin
    )

    # Rerank
    display_rerank = None
    if raw_config.get("rerank", {}).get("enabled"):
        rerank_plugin = raw_config.get("rerank", {}).get("plugin_name", "?")
        rerank_model = raw_config.get("rerank", {}).get("kwargs", {}).get("model", "")
        display_rerank = (
            f"{rerank_plugin} ({rerank_model})" if rerank_model else rerank_plugin
        )

    # =========================================================================
    # Show query info
    # =========================================================================

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

    # =========================================================================
    # Execute query
    # =========================================================================

    try:
        pipeline = RAGPipeline.from_config(typed_config)
        answer = pipeline.run(question_text)
        display_answer(answer)
    except Exception as e:
        ui.error(f"Query failed: {e}")
        logger.exception("Query error")
        raise typer.Exit(1)
