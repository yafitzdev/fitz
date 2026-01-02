# fitz_ai/cli/commands/quickstart.py
"""
Quickstart command - Zero-friction RAG in one command.

Usage:
    fitz quickstart                                    # Interactive prompts
    fitz quickstart ./docs "What is the refund policy?" # Direct

This command is a thin wrapper that:
1. Prompts for API key if not set
2. Creates config file if not exists (like silent `fitz init -y`)
3. Ingests documents into FAISS
4. Runs query

No separate code path - just sugar on top of the normal flow.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import typer

from fitz_ai.cli.ui import display_answer, ui
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

    Examples:
        fitz quickstart                                      # Interactive
        fitz quickstart ./docs "What is the refund policy?"  # Direct
    """
    ui.header("Fitz Quickstart", "Zero-friction RAG")

    # =========================================================================
    # Prompt for source if not provided
    # =========================================================================

    if source is None:
        print()
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
    # Run quickstart with fitz_rag + FAISS
    # =========================================================================

    _run_quickstart(source, question, collection, verbose)


# =============================================================================
# Quickstart Implementation
# =============================================================================


def _run_quickstart(source: Path, question: str, collection: str, verbose: bool) -> None:
    """Run quickstart with fitz_rag engine and FAISS."""
    engine_config_path = FitzPaths.engine_config("fitz_rag")

    # =========================================================================
    # Step 1: Select Provider and Ensure API Key
    # =========================================================================

    # Always prompt for provider selection
    provider = _select_provider()
    if provider is None:
        raise typer.Exit(1)

    # Ensure API key for selected provider
    if provider != "ollama":
        api_key = _ensure_api_key(provider)
        if not api_key:
            raise typer.Exit(1)

    # Create config for selected provider
    _create_provider_config(engine_config_path, provider)

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
        hierarchy_info = (
            f", {stats['hierarchy_summaries']} summaries"
            if stats.get("hierarchy_summaries")
            else ""
        )
        ui.success(
            f"Ingested {stats['documents']} documents ({stats['chunks']} chunks{hierarchy_info})"
        )

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
    ui.info(f'  fitz query "your question" --collection {collection}')


# =============================================================================
# Provider Selection
# =============================================================================

# Provider configurations
PROVIDERS = {
    "cohere": {
        "name": "Cohere",
        "env_var": "COHERE_API_KEY",
        "description": "Free tier available, best for RAG (has reranking)",
        "signup_url": "https://dashboard.cohere.com/api-keys",
    },
    "openai": {
        "name": "OpenAI",
        "env_var": "OPENAI_API_KEY",
        "description": "Most popular, requires payment",
        "signup_url": "https://platform.openai.com/api-keys",
    },
    "ollama": {
        "name": "Ollama (Local)",
        "env_var": None,
        "description": "Free, runs locally, no API key needed",
        "signup_url": None,
    },
}


def _select_provider() -> Optional[str]:
    """Prompt user to select an LLM provider."""
    print()

    # Check which providers have API keys already set
    available = []
    for key, info in PROVIDERS.items():
        env_var = info["env_var"]
        has_key = env_var is None or os.getenv(env_var)
        available.append((key, info, has_key))

    # Show options
    ui.info("Select your LLM provider:")
    print()

    for i, (key, info, has_key) in enumerate(available, 1):
        status = " (API key found)" if has_key and info["env_var"] else ""
        print(f"  [{i}] {info['name']}{status}")
        print(f"      {info['description']}")
        print()

    # Get choice
    try:
        choice = ui.prompt_int("Choice", default=1)
        idx = choice - 1
        if 0 <= idx < len(available):
            selected = available[idx][0]
            ui.success(f"Selected: {PROVIDERS[selected]['name']}")
            return selected
        else:
            ui.error("Invalid choice")
            return None
    except (ValueError, KeyboardInterrupt):
        return None


# =============================================================================
# API Key Handling
# =============================================================================


def _ensure_api_key(provider: str) -> Optional[str]:
    """
    Check for API key for the given provider, prompt if missing.

    Returns the API key or None if user declined.
    """
    info = PROVIDERS.get(provider, {})
    env_var = info.get("env_var")

    if not env_var:
        return "local"  # Ollama doesn't need a key

    api_key = os.getenv(env_var)

    if api_key:
        masked = f"{api_key[:8]}..." if len(api_key) > 8 else "***"
        ui.success(f"Using API key ({env_var}): {masked}")
        return api_key

    # No key found - prompt user
    print()
    ui.warning(f"No API key found ({env_var})")
    print()
    print(f"  Get your API key at: {info.get('signup_url', 'provider website')}")
    print()

    try:
        api_key = typer.prompt(f"Paste your API key ({env_var})")
    except typer.Abort:
        return None

    if not api_key or len(api_key) < 10:
        ui.error("Invalid API key")
        return None

    # Set for this session
    os.environ[env_var] = api_key

    # Offer to save
    print()
    save = typer.confirm("Save to shell config for future sessions?", default=True)

    if save:
        _save_api_key_to_shell(api_key, env_var)
    else:
        ui.info("To set permanently, add to your shell config:")
        ui.info(f'  export {env_var}="{api_key}"')

    print()
    return api_key


def _save_api_key_to_shell(api_key: str, env_var: str) -> None:
    """Save API key to user's shell config file."""
    home = Path.home()

    # Try zshrc first (more common on macOS), then bashrc
    rc_files = [home / ".zshrc", home / ".bashrc"]

    for rc_file in rc_files:
        if rc_file.exists():
            content = rc_file.read_text()

            # Check if already present
            if env_var in content:
                ui.info(f"{env_var} already in {rc_file.name}")
                return

            # Append to file
            with open(rc_file, "a") as f:
                f.write(f'\n# Added by fitz quickstart\nexport {env_var}="{api_key}"\n')

            ui.success(f"Saved to ~/{rc_file.name}")
            ui.info(f"Run `source ~/{rc_file.name}` or restart your terminal")
            return

    # No rc file found - create .bashrc
    bashrc = home / ".bashrc"
    with open(bashrc, "w") as f:
        f.write(f'# Created by fitz quickstart\nexport {env_var}="{api_key}"\n')

    ui.success("Created ~/.bashrc with API key")


# =============================================================================
# Config Generation
# =============================================================================

# Provider-specific configurations
PROVIDER_CONFIGS = {
    "cohere": {
        "chat": {
            "plugin_name": "cohere",
            "kwargs": {
                "models": {"smart": "command-a-03-2025", "fast": "command-r7b-12-2024"},
                "temperature": 0.2,
            },
        },
        "embedding": {"plugin_name": "cohere", "kwargs": {"model": "embed-english-v3.0"}},
        "rerank": {"enabled": True, "plugin_name": "cohere", "kwargs": {"model": "rerank-v3.5"}},
    },
    "openai": {
        "chat": {
            "plugin_name": "openai",
            "kwargs": {"models": {"smart": "gpt-4o", "fast": "gpt-4o-mini"}, "temperature": 0.2},
        },
        "embedding": {"plugin_name": "openai", "kwargs": {"model": "text-embedding-3-small"}},
        "rerank": {"enabled": False},  # OpenAI doesn't have reranking
    },
    "ollama": {
        "chat": {
            "plugin_name": "local_ollama",
            "kwargs": {"models": {"smart": "llama3.2", "fast": "llama3.2"}},
        },
        "embedding": {"plugin_name": "local_ollama", "kwargs": {"model": "nomic-embed-text"}},
        "rerank": {"enabled": False},  # Ollama doesn't have reranking
    },
}


def _create_provider_config(config_path: Path, provider: str) -> None:
    """
    Create a config file for the selected provider.

    Uses the selected provider for chat/embedding and local FAISS for vectors.
    """
    import yaml

    provider_cfg = PROVIDER_CONFIGS.get(provider, PROVIDER_CONFIGS["cohere"])

    config = {
        "chat": provider_cfg["chat"],
        "embedding": provider_cfg["embedding"],
        "vector_db": {"plugin_name": "local_faiss", "kwargs": {}},
        "retrieval": {"plugin_name": "dense", "collection": "quickstart", "top_k": 5},
        "rerank": provider_cfg["rerank"],
        "rgs": {"enable_citations": True, "strict_grounding": True, "max_chunks": 8},
        "logging": {"level": "INFO"},
    }

    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Write config with header
    header = f"""\
# Fitz RAG Configuration
# Generated by: fitz quickstart
# Provider: {provider}
# Edit with: fitz config --edit

"""
    config_path.write_text(header + yaml.dump(config, default_flow_style=False, sort_keys=False))


# =============================================================================
# Ingestion (reuses existing ingest logic)
# =============================================================================


def _run_ingestion(
    source: Path,
    collection: str,
    verbose: bool = False,
) -> dict:
    """
    Run document ingestion with hierarchical summaries.

    Returns dict with 'documents', 'chunks', and 'hierarchy_summaries' counts.
    """
    from fitz_ai.core.config import load_config_dict
    from fitz_ai.engines.fitz_rag.config import (
        ChunkingRouterConfig,
        ExtensionChunkerConfig,
    )
    from fitz_ai.engines.fitz_rag.config import load_config_dict as load_default_config_dict
    from fitz_ai.ingestion.chunking.engine import ChunkingEngine
    from fitz_ai.ingestion.enrichment.config import HierarchyConfig
    from fitz_ai.ingestion.enrichment.hierarchy.enricher import HierarchyEnricher
    from fitz_ai.ingestion.reader.engine import IngestionEngine
    from fitz_ai.ingestion.reader.registry import get_ingest_plugin
    from fitz_ai.llm.registry import get_llm_plugin
    from fitz_ai.vector_db.registry import get_vector_db_plugin
    from fitz_ai.vector_db.writer import VectorDBWriter

    # Load config from engine-specific path
    config_path = FitzPaths.engine_config("fitz_rag")
    config = load_config_dict(config_path)

    # Get plugin configs
    embedding_config = config.get("embedding", {})
    vector_db_config = config.get("vector_db", {})
    chat_config = config.get("chat", {})

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

    # Load chunking config from user config, fall back to package defaults
    chunking_cfg = config.get("chunking") or config.get("ingest", {}).get("chunking")
    if chunking_cfg:
        default_cfg = chunking_cfg.get("default", {})
        chunking_config = ChunkingRouterConfig(
            default=ExtensionChunkerConfig(
                plugin_name=default_cfg.get("plugin_name", "recursive"),
                kwargs=default_cfg.get("kwargs", {}),
            ),
            by_extension={
                ext: ExtensionChunkerConfig(
                    plugin_name=ext_cfg.get("plugin_name", "simple"),
                    kwargs=ext_cfg.get("kwargs", {}),
                )
                for ext, ext_cfg in chunking_cfg.get("by_extension", {}).items()
            },
            warn_on_fallback=chunking_cfg.get("warn_on_fallback", False),
        )
    else:
        # Fall back to package defaults from default.yaml
        default_config = load_default_config_dict()
        default_ingest = default_config.get("ingest", {})
        default_chunking = default_ingest.get("chunking", {}).get("default", {})
        chunking_config = ChunkingRouterConfig(
            default=ExtensionChunkerConfig(
                plugin_name=default_chunking.get("plugin_name", "recursive"),
                kwargs=default_chunking.get("kwargs", {}),
            ),
            warn_on_fallback=False,
        )
    chunking_engine = ChunkingEngine.from_config(chunking_config)

    chunks: List = []
    for raw_doc in raw_docs:
        doc_chunks = chunking_engine.run(raw_doc)
        chunks.extend(doc_chunks)

    if not chunks:
        raise ValueError("No chunks created from documents")

    if verbose:
        ui.info(f"Created {len(chunks)} chunks")

    # Step 3: Hierarchical enrichment (group summaries + corpus summary)
    if verbose:
        ui.info("Generating hierarchical summaries...")

    # Get chat client for summarization (use fast tier)
    chat_client = get_llm_plugin(
        plugin_type="chat",
        plugin_name=chat_config.get("plugin_name", "cohere"),
        tier="fast",
        **chat_config.get("kwargs", {}),
    )

    # Create hierarchy enricher with simple mode defaults
    hierarchy_config = HierarchyConfig(enabled=True, group_by="source")
    hierarchy_enricher = HierarchyEnricher(config=hierarchy_config, chat_client=chat_client)

    # Enrich chunks (adds L1 summaries as metadata, returns chunks + L2 corpus summary)
    original_chunk_count = len(chunks)
    chunks = hierarchy_enricher.enrich(chunks)
    hierarchy_summaries = len(chunks) - original_chunk_count

    if verbose:
        ui.info(f"Generated {hierarchy_summaries} hierarchy summary chunks")

    # Step 4: Embed chunks
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
        "chunks": original_chunk_count,
        "hierarchy_summaries": hierarchy_summaries,
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
    from fitz_ai.engines.fitz_rag.config import load_config
    from fitz_ai.engines.fitz_rag.pipeline.engine import RAGPipeline

    # Load typed config from engine-specific path
    config_path = FitzPaths.engine_config("fitz_rag")
    typed_config = load_config(config_path)

    # Override collection
    typed_config.retrieval.collection = collection

    if verbose:
        ui.info(f"Querying collection: {collection}")

    # Create pipeline and run
    pipeline = RAGPipeline.from_config(typed_config)
    answer = pipeline.run(question)

    return answer
