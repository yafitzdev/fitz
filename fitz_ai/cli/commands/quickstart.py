# fitz_ai/cli/commands/quickstart.py
"""
Quickstart command - Zero-friction RAG in one command.

Usage:
    fitz quickstart                                    # Interactive prompts
    fitz quickstart ./docs "What is the refund policy?" # Direct

This command is a thin wrapper that:
1. Prompts for API key if not set
2. Creates config file if not exists (like silent `fitz init -y`)
3. Ingests documents (calls existing ingest logic)
4. Runs query (calls existing query logic)

No separate code path - just sugar on top of the normal flow.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import typer

from fitz_ai.cli.ui import ui
from fitz_ai.cli.ui_display import display_answer
from fitz_ai.core.paths import FitzPaths
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Main Command
# =============================================================================


def command(
        source: Optional[Path] = typer.Argument(
            None,
            help="Path to documents (file or directory). Will prompt if not provided.",
        ),
        question: Optional[str] = typer.Argument(
            None,
            help="Question to ask about your documents. Will prompt if not provided.",
        ),
        collection: str = typer.Option(
            "quickstart",
            "--collection",
            "-c",
            help="Collection name for vector storage",
        ),
        verbose: bool = typer.Option(
            False,
            "--verbose",
            "-v",
            help="Show detailed progress",
        ),
) -> None:
    """
    One-command RAG: ingest docs and ask a question.

    This command sets up everything automatically on first run.

    Examples:
        fitz quickstart                                      # Interactive
        fitz quickstart ./docs "What is the refund policy?"  # Direct
        fitz quickstart ./contracts "What are the payment terms?" -v
    """
    # =========================================================================
    # Header
    # =========================================================================

    ui.header("Fitz Quickstart")

    # =========================================================================
    # Prompt for source if not provided
    # =========================================================================

    if source is None:
        source = ui.prompt_path("Path to documents", ".")

    # Validate source exists
    if not source.exists():
        ui.error(f"Path does not exist: {source}")
        raise typer.Exit(1)

    # =========================================================================
    # Prompt for question if not provided
    # =========================================================================

    if question is None:
        question = ui.prompt_text("Question to ask")

    if not question.strip():
        ui.error("Question cannot be empty")
        raise typer.Exit(1)

    # =========================================================================
    # Step 1: Ensure API Key
    # =========================================================================

    api_key = _ensure_api_key()
    if not api_key:
        raise typer.Exit(1)

    # =========================================================================
    # Step 2: Ensure Config Exists (silent fitz init)
    # =========================================================================

    config_path = FitzPaths.config()

    if not config_path.exists():
        if verbose:
            ui.info("Creating default configuration...")

        _create_default_config(config_path)

        if verbose:
            ui.success(f"Config created at {config_path}")
    else:
        if verbose:
            ui.info(f"Using existing config: {config_path}")

    # =========================================================================
    # Step 3: Ingest Documents
    # =========================================================================

    ui.step(1, 3, f"Ingesting documents from {source}...")

    try:
        stats = _run_ingestion(
            source=source,
            collection=collection,
            verbose=verbose,
        )
        ui.success(f"Ingested {stats['documents']} documents ({stats['chunks']} chunks)")

    except Exception as e:
        ui.error(f"Ingestion failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        raise typer.Exit(1)

    # =========================================================================
    # Step 4: Run Query
    # =========================================================================

    ui.step(2, 3, "Searching and generating answer...")

    try:
        answer = _run_query(
            question=question,
            collection=collection,
            verbose=verbose,
        )

    except Exception as e:
        ui.error(f"Query failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        raise typer.Exit(1)

    # =========================================================================
    # Step 5: Display Answer (using shared display function)
    # =========================================================================

    ui.step(3, 3, "Done!")

    display_answer(answer, show_sources=True)

    # =========================================================================
    # Step 6: Next Steps
    # =========================================================================

    ui.info("Your documents are now indexed. Run more queries with:")
    ui.info(f"  fitz query \"your question\" --collection {collection}")


# =============================================================================
# API Key Handling
# =============================================================================


def _ensure_api_key() -> Optional[str]:
    """
    Check for Cohere API key, prompt if missing.

    Returns the API key or None if user declined.
    """
    api_key = os.getenv("COHERE_API_KEY")

    if api_key:
        masked = f"{api_key[:8]}..." if len(api_key) > 8 else "***"
        ui.success(f"Using Cohere API key: {masked}")
        return api_key

    # No key found - prompt user
    print()
    ui.warning("No API key found")
    print()
    print("  Fitz needs an API key for embeddings and chat.")
    print("  Get a free key at: https://dashboard.cohere.com/api-keys")
    print()
    print("  It takes 30 seconds and no credit card is required.")
    print()

    try:
        api_key = typer.prompt("Paste your Cohere API key")
    except typer.Abort:
        return None

    if not api_key or len(api_key) < 10:
        ui.error("Invalid API key")
        return None

    # Set for this session
    os.environ["COHERE_API_KEY"] = api_key

    # Offer to save
    print()
    save = typer.confirm("Save to shell config for future sessions?", default=True)

    if save:
        _save_api_key_to_shell(api_key)
    else:
        ui.info("To set permanently, add to your shell config:")
        ui.info(f'  export COHERE_API_KEY="{api_key}"')

    print()
    return api_key


def _save_api_key_to_shell(api_key: str) -> None:
    """Save API key to user's shell config file."""
    home = Path.home()

    # Try zshrc first (more common on macOS), then bashrc
    rc_files = [home / ".zshrc", home / ".bashrc"]

    for rc_file in rc_files:
        if rc_file.exists():
            content = rc_file.read_text()

            # Check if already present
            if "COHERE_API_KEY" in content:
                ui.info(f"COHERE_API_KEY already in {rc_file.name}")
                return

            # Append to file
            with open(rc_file, "a") as f:
                f.write(f'\n# Added by fitz quickstart\nexport COHERE_API_KEY="{api_key}"\n')

            ui.success(f"Saved to ~/{rc_file.name}")
            ui.info(f"Run `source ~/{rc_file.name}` or restart your terminal")
            return

    # No rc file found - create .bashrc
    bashrc = home / ".bashrc"
    with open(bashrc, "w") as f:
        f.write(f'# Created by fitz quickstart\nexport COHERE_API_KEY="{api_key}"\n')

    ui.success("Created ~/.bashrc with API key")


# =============================================================================
# Config Generation
# =============================================================================


def _create_default_config(config_path: Path) -> None:
    """
    Create a default config file for quickstart.

    Uses Cohere for everything (embedding, chat, rerank) and local FAISS.
    This is the same as running `fitz init -y` with these providers available.
    """
    config_content = """\
# Fitz RAG Configuration
# Generated by: fitz quickstart
# Edit with: fitz config --edit

# Chat (LLM for answering questions)
chat:
  plugin_name: cohere
  kwargs:
    model: command-a-03-2025
    temperature: 0.2

# Embedding (text to vectors)
embedding:
  plugin_name: cohere
  kwargs:
    model: embed-english-v3.0

# Vector Database (local FAISS - no external service needed)
vector_db:
  plugin_name: local_faiss
  kwargs: {}

# Retrieval
retrieval:
  plugin_name: dense
  collection: quickstart
  top_k: 5

# Reranker
rerank:
  enabled: true
  plugin_name: cohere
  kwargs:
    model: rerank-v3.5

# RGS (Retrieval-Guided Synthesis)
rgs:
  enable_citations: true
  strict_grounding: true
  max_chunks: 8

# Logging
logging:
  level: INFO
"""

    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Write config
    config_path.write_text(config_content)


# =============================================================================
# Ingestion (reuses existing ingest logic)
# =============================================================================


def _run_ingestion(
        source: Path,
        collection: str,
        verbose: bool = False,
) -> dict:
    """
    Run document ingestion using the existing ingest pipeline.

    Returns dict with 'documents' and 'chunks' counts.
    """
    from fitz_ai.core.config import load_config_dict
    from fitz_ai.ingest.chunking.engine import ChunkingEngine
    from fitz_ai.ingest.config.schema import ChunkerConfig
    from fitz_ai.ingest.ingestion.engine import IngestionEngine
    from fitz_ai.ingest.ingestion.registry import get_ingest_plugin
    from fitz_ai.llm.registry import get_llm_plugin
    from fitz_ai.vector_db.registry import get_vector_db_plugin
    from fitz_ai.vector_db.writer import VectorDBWriter

    # Load config
    config_path = FitzPaths.config()
    config = load_config_dict(config_path)

    # Get plugin configs
    embedding_config = config.get("embedding", {})
    vector_db_config = config.get("vector_db", {})

    # Step 1: Read documents
    if verbose:
        ui.info("Reading documents...")

    IngestPluginCls = get_ingest_plugin("local")
    ingest_plugin = IngestPluginCls()
    ingest_engine = IngestionEngine(plugin=ingest_plugin, kwargs={})
    raw_docs = list(ingest_engine.run(str(source)))

    if not raw_docs:
        raise ValueError(f"No documents found in {source}")

    if verbose:
        ui.info(f"Found {len(raw_docs)} documents")

    # Step 2: Chunk documents
    if verbose:
        ui.info("Chunking documents...")

    # Simple chunker only accepts chunk_size (no overlap parameter)
    chunker_config = ChunkerConfig(
        plugin_name="simple",
        kwargs={"chunk_size": 1000},
    )
    chunking_engine = ChunkingEngine.from_config(chunker_config)

    chunks: List = []
    for raw_doc in raw_docs:
        doc_chunks = chunking_engine.run(raw_doc)
        chunks.extend(doc_chunks)

    if not chunks:
        raise ValueError("No chunks created from documents")

    if verbose:
        ui.info(f"Created {len(chunks)} chunks")

    # Step 3: Embed chunks
    if verbose:
        ui.info("Generating embeddings...")

    # get_llm_plugin returns an instance, not a class
    embedder = get_llm_plugin(
        plugin_type="embedding",
        plugin_name=embedding_config.get("plugin_name", "cohere"),
        **embedding_config.get("kwargs", {}),
    )

    vectors = []
    for chunk in chunks:
        vec = embedder.embed(chunk.content)
        vectors.append(vec)

    # Step 4: Store in vector DB
    if verbose:
        ui.info("Storing vectors...")

    vdb_plugin = get_vector_db_plugin(
        vector_db_config.get("plugin_name", "local_faiss"),
        **vector_db_config.get("kwargs", {}),
    )

    writer = VectorDBWriter(client=vdb_plugin)
    writer.upsert(collection=collection, chunks=chunks, vectors=vectors)

    return {
        "documents": len(raw_docs),
        "chunks": len(chunks),
    }


# =============================================================================
# Query (reuses existing query/pipeline logic)
# =============================================================================


def _run_query(
        question: str,
        collection: str,
        verbose: bool = False,
):
    """
    Run a query using the existing RAG pipeline.

    Returns an RGSAnswer object.
    """
    from fitz_ai.engines.classic_rag.config import load_config
    from fitz_ai.engines.classic_rag.pipeline.engine import RAGPipeline

    # Load typed config
    config_path = FitzPaths.config()
    typed_config = load_config(config_path)

    # Override collection
    typed_config.retrieval.collection = collection

    if verbose:
        ui.info(f"Querying collection: {collection}")

    # Create pipeline and run
    pipeline = RAGPipeline.from_config(typed_config)
    answer = pipeline.run(question)

    return answer