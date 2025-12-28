# fitz_ai/cli/commands/init.py
"""
Init command - Interactive setup wizard.

Usage:
    fitz init              # Interactive wizard
    fitz init -y           # Auto-detect and use defaults
    fitz init --show       # Preview config without saving
"""

from __future__ import annotations

import typer

from fitz_ai.cli.ui import RICH, console, get_first_available, ui
from fitz_ai.core.registry import (
    available_chunking_plugins,
    available_llm_plugins,
    available_retrieval_plugins,
    available_vector_db_plugins,
)
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Default Config Loading
# =============================================================================


def _load_default_config() -> dict:
    """Load the default configuration from default.yaml."""
    from fitz_ai.engines.classic_rag.config import load_config_dict

    return load_config_dict()


# =============================================================================
# System Detection
# =============================================================================


def detect_system():
    """Detect all available services and API keys."""
    from fitz_ai.core.detect import detect_system_status

    return detect_system_status()


# =============================================================================
# Plugin Filtering
# =============================================================================


def _filter_available_plugins(plugins: list[str], plugin_type: str, system) -> list[str]:
    """Filter plugins to only those that are available."""
    available = []

    for plugin in plugins:
        plugin_lower = plugin.lower()

        # Ollama plugins require Ollama
        if "ollama" in plugin_lower:
            if system.ollama.available:
                available.append(plugin)
            continue

        # Qdrant requires Qdrant
        if "qdrant" in plugin_lower:
            if system.qdrant.available:
                available.append(plugin)
            continue

        # FAISS requires faiss
        if "faiss" in plugin_lower:
            if system.faiss.available:
                available.append(plugin)
            continue

        # API-based plugins require API keys
        if "cohere" in plugin_lower:
            if system.api_keys.get("cohere", type("", (), {"available": False})).available:
                available.append(plugin)
            continue

        if "openai" in plugin_lower or "azure" in plugin_lower:
            if system.api_keys.get("openai", type("", (), {"available": False})).available:
                available.append(plugin)
            continue

        if "anthropic" in plugin_lower:
            if system.api_keys.get("anthropic", type("", (), {"available": False})).available:
                available.append(plugin)
            continue

        # Default: include unknown plugins
        available.append(plugin)

    return available


# =============================================================================
# Model Selection Helpers
# =============================================================================


def _get_default_model(plugin_type: str, plugin_name: str) -> str:
    """Get the default model for a plugin."""
    defaults = {
        "chat": {
            "cohere": "command-a-03-2025",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-sonnet-4-20250514",
            "local_ollama": "llama3.2",
        },
        "embedding": {
            "cohere": "embed-english-v3.0",
            "openai": "text-embedding-3-small",
            "local_ollama": "nomic-embed-text",
        },
        "rerank": {
            "cohere": "rerank-v3.5",
        },
    }
    return defaults.get(plugin_type, {}).get(plugin_name, "")


def _prompt_model(plugin_type: str, plugin_name: str) -> str:
    """Prompt for model selection with smart default."""
    default_model = _get_default_model(plugin_type, plugin_name)

    if not default_model:
        return ""

    return ui.prompt_text(f"  Model for {plugin_name}", default_model)


# =============================================================================
# Config Generation
# =============================================================================


def _generate_config(
    *,
    chat: str,
    chat_model: str,
    embedding: str,
    embedding_model: str,
    vector_db: str,
    retrieval: str,
    rerank: str | None,
    rerank_model: str,
    qdrant_host: str,
    qdrant_port: int,
    # Chunking config
    chunker: str,
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    """Generate the config YAML string."""
    # Build chat kwargs
    chat_kwargs = ""
    if chat_model:
        chat_kwargs = f"\n    model: {chat_model}\n    temperature: 0.2"

    # Build embedding kwargs
    embedding_kwargs = ""
    if embedding_model:
        embedding_kwargs = f"\n    model: {embedding_model}"

    # Build vector DB kwargs
    vdb_kwargs = ""
    if vector_db == "qdrant":
        vdb_kwargs = f'\n    host: "{qdrant_host}"\n    port: {qdrant_port}'

    # Build rerank section
    if rerank:
        rerank_kwargs = " {}"
        if rerank_model:
            rerank_kwargs = f"\n    model: {rerank_model}"
        rerank_section = f"""
# Reranker (used by retrieval plugins that support reranking)
rerank:
  enabled: true
  plugin_name: {rerank}
  kwargs:{rerank_kwargs}
"""
    else:
        rerank_section = """
# Reranker (no reranker available)
rerank:
  enabled: false
"""

    # Build chunking section
    chunking_section = f"""
# Chunking (document splitting for ingestion)
chunking:
  default:
    plugin_name: {chunker}
    kwargs:
      chunk_size: {chunk_size}
      chunk_overlap: {chunk_overlap}
  by_extension: {{}}
"""

    return f"""# Fitz RAG Configuration
# Generated by: fitz init

# Chat (LLM for answering questions)
chat:
  plugin_name: {chat}
  kwargs:{chat_kwargs}

# Embedding (text to vectors)
embedding:
  plugin_name: {embedding}
  kwargs:{embedding_kwargs}

# Vector Database
vector_db:
  plugin_name: {vector_db}
  kwargs:{vdb_kwargs if vdb_kwargs else " {}"}

# Retrieval (YAML-based plugin)
retrieval:
  plugin_name: {retrieval}
  collection: default
  top_k: 5
{rerank_section}{chunking_section}
# RGS (Retrieval-Guided Synthesis)
rgs:
  enable_citations: true
  strict_grounding: true
  max_chunks: 8

# Logging
logging:
  level: INFO
"""


# =============================================================================
# Main Command
# =============================================================================


def command(
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        "-y",
        help="Use detected defaults without prompting.",
    ),
    show_config: bool = typer.Option(
        False,
        "--show",
        "-s",
        help="Preview config without saving.",
    ),
) -> None:
    """
    Initialize Fitz with an interactive setup wizard.

    Detects available providers (API keys, Ollama, Qdrant) and
    creates a working configuration file.

    Examples:
        fitz init           # Interactive wizard
        fitz init -y        # Auto-detect and use defaults
        fitz init --show    # Preview config without saving
    """
    from fitz_ai.core.paths import FitzPaths

    # =========================================================================
    # Header
    # =========================================================================

    ui.header("Fitz Init", "Let's configure your RAG pipeline")

    # =========================================================================
    # Detect System
    # =========================================================================

    ui.section("Detecting System")

    system = detect_system()

    # Show detection results
    ui.status(
        "Ollama",
        system.ollama.available,
        (
            f"{system.ollama.host}:{system.ollama.port}"
            if system.ollama.available
            else system.ollama.details
        ),
    )
    ui.status(
        "Qdrant",
        system.qdrant.available,
        (
            f"{system.qdrant.host}:{system.qdrant.port}"
            if system.qdrant.available
            else system.qdrant.details
        ),
    )
    ui.status("FAISS", system.faiss.available)

    for name, key_status in system.api_keys.items():
        ui.status(name.capitalize(), key_status.available)

    # =========================================================================
    # Discover Plugins
    # =========================================================================

    all_chat = available_llm_plugins("chat")
    all_embedding = available_llm_plugins("embedding")
    all_rerank = available_llm_plugins("rerank")
    all_vector_db = available_vector_db_plugins()
    all_retrieval = available_retrieval_plugins()
    all_chunkers = available_chunking_plugins()

    # Filter to available only
    avail_chat = _filter_available_plugins(all_chat, "chat", system)
    avail_embedding = _filter_available_plugins(all_embedding, "embedding", system)
    avail_rerank = _filter_available_plugins(all_rerank, "rerank", system)
    avail_vector_db = _filter_available_plugins(all_vector_db, "vector_db", system)
    avail_retrieval = all_retrieval
    avail_chunkers = all_chunkers if all_chunkers else ["simple"]

    # =========================================================================
    # Validate Minimum Requirements
    # =========================================================================

    if not avail_chat:
        ui.error("No chat plugins available!")
        ui.info("Set an API key (COHERE_API_KEY, OPENAI_API_KEY) or start Ollama.")
        raise typer.Exit(1)

    if not avail_embedding:
        ui.error("No embedding plugins available!")
        ui.info("Set an API key (COHERE_API_KEY, OPENAI_API_KEY) or start Ollama.")
        raise typer.Exit(1)

    if not avail_vector_db:
        ui.error("No vector database available!")
        ui.info("Start Qdrant or install FAISS (pip install faiss-cpu).")
        raise typer.Exit(1)

    # =========================================================================
    # Select Plugins
    # =========================================================================

    # Load defaults from default.yaml (single source of truth)
    default_config = _load_default_config()

    # Plugin defaults from default.yaml
    default_chat = default_config.get("chat", {}).get("plugin_name", "cohere")
    default_embedding = default_config.get("embedding", {}).get("plugin_name", "cohere")
    default_vector_db = default_config.get("vector_db", {}).get("plugin_name", "local_faiss")
    default_retrieval = default_config.get("retrieval", {}).get("plugin_name", "dense")
    default_rerank = default_config.get("rerank", {}).get("plugin_name", "cohere")

    # Chunking defaults from default.yaml
    default_ingest = default_config.get("ingest", {})
    default_chunking = default_ingest.get("chunking", {}).get("default", {})
    default_chunker = default_chunking.get("plugin_name", "recursive")
    default_chunk_size = default_chunking.get("kwargs", {}).get("chunk_size", 1000)
    default_chunk_overlap = default_chunking.get("kwargs", {}).get("chunk_overlap", 200)

    # Helper to get default if available, otherwise first available
    def _get_default_or_first(choices: list[str], default: str) -> str:
        if default in choices:
            return default
        return get_first_available(choices, default)

    if non_interactive:
        # Use defaults from default.yaml if available
        chat_choice = _get_default_or_first(avail_chat, default_chat)
        chat_model = _get_default_model("chat", chat_choice)
        embedding_choice = _get_default_or_first(avail_embedding, default_embedding)
        embedding_model = _get_default_model("embedding", embedding_choice)
        vector_db_choice = _get_default_or_first(avail_vector_db, default_vector_db)
        retrieval_choice = _get_default_or_first(avail_retrieval, default_retrieval)
        rerank_choice = (
            _get_default_or_first(avail_rerank, default_rerank) if avail_rerank else None
        )
        rerank_model = _get_default_model("rerank", rerank_choice) if rerank_choice else ""
        # Chunking defaults from default.yaml
        chunker_choice = default_chunker
        chunk_size = default_chunk_size
        chunk_overlap = default_chunk_overlap

    else:
        # Interactive selection
        ui.section("Configuration")

        # Chat
        chat_choice = ui.prompt_numbered_choice(
            "Chat plugin", avail_chat, _get_default_or_first(avail_chat, default_chat)
        )
        chat_model = _prompt_model("chat", chat_choice)

        # Embedding
        print()
        embedding_choice = ui.prompt_numbered_choice(
            "Embedding plugin",
            avail_embedding,
            _get_default_or_first(avail_embedding, default_embedding),
        )
        embedding_model = _prompt_model("embedding", embedding_choice)

        # Vector DB
        print()
        vector_db_choice = ui.prompt_numbered_choice(
            "Vector database",
            avail_vector_db,
            _get_default_or_first(avail_vector_db, default_vector_db),
        )

        # Retrieval
        print()
        retrieval_choice = ui.prompt_numbered_choice(
            "Retrieval strategy",
            avail_retrieval,
            _get_default_or_first(avail_retrieval, default_retrieval),
        )

        # Rerank (optional)
        rerank_choice = None
        rerank_model = ""
        if avail_rerank:
            print()
            rerank_choice = ui.prompt_numbered_choice(
                "Rerank plugin",
                avail_rerank,
                _get_default_or_first(avail_rerank, default_rerank),
            )
            rerank_model = _prompt_model("rerank", rerank_choice)
        else:
            ui.info("Rerank: not available (no rerank plugins detected)")

        # Chunking (defaults from default.yaml)
        print()
        ui.print("Chunking configuration:", "bold")
        chunker_choice = ui.prompt_numbered_choice(
            "Default chunker", avail_chunkers, default_chunker
        )
        chunk_size = ui.prompt_int("Chunk size", default_chunk_size)
        chunk_overlap = ui.prompt_int("Chunk overlap", default_chunk_overlap)

    # =========================================================================
    # Generate Config
    # =========================================================================

    qdrant_host = system.qdrant.host if system.qdrant.available else "localhost"
    qdrant_port = system.qdrant.port if system.qdrant.available else 6333

    config_yaml = _generate_config(
        chat=chat_choice,
        chat_model=chat_model,
        embedding=embedding_choice,
        embedding_model=embedding_model,
        vector_db=vector_db_choice,
        retrieval=retrieval_choice,
        rerank=rerank_choice,
        rerank_model=rerank_model,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        chunker=chunker_choice,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # =========================================================================
    # Show Config
    # =========================================================================

    ui.section("Generated Configuration")
    ui.syntax(config_yaml, "yaml")

    if show_config:
        return

    # =========================================================================
    # Save Config
    # =========================================================================

    ui.section("Saving")

    fitz_dir = FitzPaths.workspace()
    fitz_config = FitzPaths.config()

    # Confirm overwrite if exists
    if fitz_config.exists() and not non_interactive:
        if not ui.prompt_confirm("Config exists. Overwrite?", default=False):
            ui.warning("Aborted.")
            raise typer.Exit(0)

    fitz_dir.mkdir(parents=True, exist_ok=True)
    fitz_config.write_text(config_yaml)
    ui.success(f"Saved to {fitz_config}")

    # =========================================================================
    # Next Steps
    # =========================================================================

    ui.section("Done!")

    if RICH:
        console.print(
            """
[green]Your configuration is ready![/green]

Next steps:
  [cyan]fitz ingest ./docs[/cyan]          # Ingest documents
  [cyan]fitz query "your question"[/cyan]  # Query knowledge base
  [cyan]fitz doctor[/cyan]                 # Verify setup
"""
        )
    else:
        print(
            """
Your configuration is ready!

Next steps:
  fitz ingest ./docs          # Ingest documents
  fitz query "your question"  # Query knowledge base
  fitz doctor                 # Verify setup
"""
        )
