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

from fitz_ai.cli.context import CLIContext
from fitz_ai.cli.ui import RICH, console, get_first_available, ui
from fitz_ai.core.registry import (
    available_chunking_plugins,
    available_llm_plugins,
    available_retrieval_plugins,
    available_vector_db_plugins,
)
from fitz_ai.logging.logger import get_logger
from fitz_ai.runtime import get_default_engine, get_engine_registry

logger = get_logger(__name__)


# =============================================================================
# Default Config Loading
# =============================================================================


def _load_default_config() -> dict:
    """Load the default configuration from default.yaml."""
    from fitz_ai.engines.fitz_rag.config import load_config_dict

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


def _get_default_model(plugin_type: str, plugin_name: str, tier: str = "smart") -> str:
    """Get the default model for a plugin.

    Args:
        plugin_type: Type of plugin (chat, embedding, rerank)
        plugin_name: Name of the plugin (cohere, openai, etc.)
        tier: Model tier for chat plugins ("smart" or "fast")
    """
    defaults = {
        "chat_smart": {
            "cohere": "command-a-03-2025",
            "openai": "gpt-4o",
            "anthropic": "claude-sonnet-4-20250514",
            "local_ollama": "llama3.2",
        },
        "chat_fast": {
            "cohere": "command-r7b-12-2024",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-haiku-3-5-20241022",
            "local_ollama": "llama3.2:1b",
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
    if plugin_type == "chat":
        key = f"chat_{tier}"
        return defaults.get(key, {}).get(plugin_name, "")
    return defaults.get(plugin_type, {}).get(plugin_name, "")


def _prompt_model(plugin_type: str, plugin_name: str, tier: str = "smart") -> str:
    """Prompt for model selection with smart default.

    Args:
        plugin_type: Type of plugin (chat, embedding, rerank)
        plugin_name: Name of the plugin
        tier: Model tier for chat plugins ("smart" or "fast")
    """
    default_model = _get_default_model(plugin_type, plugin_name, tier)

    if not default_model:
        return ""

    if plugin_type == "chat":
        tier_label = "smart" if tier == "smart" else "fast"
        return ui.prompt_text(f"  {tier_label.capitalize()} model for {plugin_name}", default_model)

    return ui.prompt_text(f"  Model for {plugin_name}", default_model)


# =============================================================================
# Config Generation
# =============================================================================


def _generate_global_config(default_engine: str) -> str:
    """Generate the global config YAML with just default_engine."""
    return f"""# Fitz Global Configuration
# Generated by: fitz init

# Default engine for CLI commands
default_engine: {default_engine}
"""


def _generate_fitz_rag_config(
    *,
    chat: str,
    chat_model_smart: str,
    chat_model_fast: str,
    embedding: str,
    embedding_model: str,
    rerank: str | None,
    rerank_model: str,
    vector_db: str,
    retrieval: str,
    qdrant_host: str,
    qdrant_port: int,
    # Chunking config
    chunker: str,
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    """Generate the Fitz RAG config YAML string."""
    # Build chat kwargs with smart/fast models
    chat_kwargs = ""
    if chat_model_smart or chat_model_fast:
        chat_kwargs = "\n    models:"
        if chat_model_smart:
            chat_kwargs += f"\n      smart: {chat_model_smart}"
        if chat_model_fast:
            chat_kwargs += f"\n      fast: {chat_model_fast}"
        chat_kwargs += "\n    temperature: 0.2"

    # Build embedding kwargs
    embedding_kwargs = ""
    if embedding_model:
        embedding_kwargs = f"\n    model: {embedding_model}"

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

    # Build vector DB kwargs
    vdb_kwargs = ""
    if vector_db == "qdrant":
        vdb_kwargs = f'\n    host: "{qdrant_host}"\n    port: {qdrant_port}'

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

    return f"""# Fitz RAG Engine Configuration
# Generated by: fitz init

# Chat (LLM for answering questions)
# smart: Best quality for user-facing responses (queries)
# fast: Best speed for background tasks (enrichment, summaries)
chat:
  plugin_name: {chat}
  kwargs:{chat_kwargs}

# Embedding (text to vectors)
embedding:
  plugin_name: {embedding}
  kwargs:{embedding_kwargs}
{rerank_section}
# Vector Database
vector_db:
  plugin_name: {vector_db}
  kwargs:{vdb_kwargs if vdb_kwargs else " {}"}

# Retrieval (YAML-based plugin)
retrieval:
  plugin_name: {retrieval}
  collection: default
  top_k: 5
{chunking_section}
# RGS (Retrieval-Guided Synthesis)
rgs:
  enable_citations: true
  strict_grounding: true
  max_chunks: 8

# Logging
logging:
  level: INFO
"""


def _copy_engine_default_config(engine_name: str, registry) -> str | None:
    """
    Get the content of an engine's default config file.

    Uses the registry to discover the engine's default config path,
    then reads and returns its contents. This enables auto-discovery
    of engine configs without hardcoding engine names.

    Args:
        engine_name: Name of the engine (e.g., 'graphrag', 'clara')
        registry: The engine registry instance

    Returns:
        Config file contents as string, or None if no default config
    """
    default_path = registry.get_default_config_path(engine_name)
    if default_path is None or not default_path.exists():
        return None

    return default_path.read_text(encoding="utf-8")


# =============================================================================
# GraphRAG Config Generation
# =============================================================================


def _generate_graphrag_config(
    *,
    llm_provider: str,
    embedding_provider: str,
    storage_backend: str,
) -> str:
    """Generate GraphRAG config YAML string."""
    return f"""# GraphRAG Engine Configuration
# Generated by: fitz init

graphrag:

  # ===========================================================================
  # LLM Provider
  # ===========================================================================
  # Model tiers (smart/fast) are defined in the plugin YAML.

  llm_provider: {llm_provider}

  # ===========================================================================
  # Embedding Provider
  # ===========================================================================

  embedding_provider: {embedding_provider}

  # ===========================================================================
  # Extraction
  # ===========================================================================

  extraction:
    max_entities_per_chunk: 20
    max_relationships_per_chunk: 30
    entity_types:
      - person
      - organization
      - location
      - event
      - concept
      - technology
      - product
      - date
    relationship_types: []
    include_descriptions: true

  # ===========================================================================
  # Community Detection
  # ===========================================================================
  # Algorithms: "louvain" (fast) or "leiden" (requires leidenalg package)

  community:
    algorithm: louvain
    resolution: 1.0
    min_community_size: 2
    max_hierarchy_levels: 2

  # ===========================================================================
  # Search
  # ===========================================================================
  # Modes: "local" (entity-focused), "global" (community-based), "hybrid"

  search:
    default_mode: local
    local_top_k: 10
    global_top_k: 5
    include_relationships: true
    max_context_tokens: 4000

  # ===========================================================================
  # Storage
  # ===========================================================================
  # Backends: "memory" (default) or "file"

  storage:
    backend: {storage_backend}
    storage_path: null

  # ===========================================================================
  # Ingestion
  # ===========================================================================
  # Chunker defaults are defined in each plugin.

  ingest:
    ingester:
      plugin_name: local

    chunking:
      default:
        plugin_name: recursive
      by_extension:
        .py:
          plugin_name: python_code
        .md:
          plugin_name: markdown
        .pdf:
          plugin_name: pdf_sections
      warn_on_fallback: true

    collection: default
"""


# =============================================================================
# Clara Config Generation
# =============================================================================


def _generate_clara_config(
    *,
    model_variant: str,
    device: str,
    compression_rate: int,
) -> str:
    """Generate Clara config YAML string."""
    # Map variant to model path
    variant_models = {
        "e2e": "apple/CLaRa-7B-E2E",
        "instruct": "apple/CLaRa-7B-Instruct",
        "base": "apple/CLaRa-7B-Base",
    }
    model_path = variant_models.get(model_variant, "apple/CLaRa-7B-E2E")

    return f"""# Clara Engine Configuration
# Generated by: fitz init

clara:

  # ===========================================================================
  # Model
  # ===========================================================================
  # HuggingFace model variants:
  #   - apple/CLaRa-7B-Base (compression only)
  #   - apple/CLaRa-7B-Instruct (instruction-tuned)
  #   - apple/CLaRa-7B-E2E (full retrieval + generation)

  model:
    model_name_or_path: "{model_path}"
    variant: "{model_variant}"
    device: "{device}"
    torch_dtype: "bfloat16"
    trust_remote_code: true
    load_in_8bit: false
    load_in_4bit: false

  # ===========================================================================
  # Compression
  # ===========================================================================
  # Compression rate options: 4, 16, 32, 64, 128
  # Higher = smaller but may lose information
  # Note: CLaRa compresses whole documents, not chunks

  compression:
    compression_rate: {compression_rate}
    doc_max_length: 256
    num_memory_tokens: null

  # ===========================================================================
  # Retrieval
  # ===========================================================================

  retrieval:
    top_k: 5
    candidate_pool_size: 20
    differentiable_topk: false

  # ===========================================================================
  # Generation
  # ===========================================================================

  generation:
    max_new_tokens: 256
    temperature: 0.7
    top_p: 0.9
    do_sample: true

  # ===========================================================================
  # Knowledge Base
  # ===========================================================================

  knowledge_base_path: null
  cache_compressed_docs: true

  # ===========================================================================
  # Ingestion
  # ===========================================================================
  # CLaRa compresses whole documents (not chunks).
  # Parser converts files to text before compression.

  ingest:
    ingester:
      plugin_name: local

    collection: default
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
    # Select Default Engine
    # =========================================================================

    ctx = CLIContext.load()
    registry = get_engine_registry()

    if non_interactive:
        default_engine = get_default_engine()
    else:
        ui.section("Engine Setup")
        default_engine = ctx.select_engine()

    # If user selected a non-collection engine, handle engine-specific config
    caps = registry.get_capabilities(default_engine)
    if not caps.supports_collections:
        global_config_yaml = _generate_global_config(default_engine)
        engine_config_yaml = None

        # =====================================================================
        # GraphRAG Configuration
        # =====================================================================
        if default_engine == "graphrag":
            # Detect available providers
            ui.section("Detecting System")
            system = detect_system()

            # Show detection results
            for name, key_status in system.api_keys.items():
                ui.status(name.capitalize(), key_status.available)
            ui.status("Ollama", system.ollama.available)

            # Build available provider lists
            llm_providers = []
            embedding_providers = []

            if system.api_keys.get("cohere", type("", (), {"available": False})).available:
                llm_providers.append("cohere")
                embedding_providers.append("cohere")
            if system.api_keys.get("openai", type("", (), {"available": False})).available:
                llm_providers.append("openai")
                embedding_providers.append("openai")
            if system.api_keys.get("anthropic", type("", (), {"available": False})).available:
                llm_providers.append("anthropic")
            if system.ollama.available:
                llm_providers.append("ollama")
                embedding_providers.append("ollama")

            if not llm_providers:
                ui.error("No LLM providers available!")
                ui.info("Set an API key (COHERE_API_KEY, OPENAI_API_KEY) or start Ollama.")
                raise typer.Exit(1)

            if not embedding_providers:
                ui.error("No embedding providers available!")
                ui.info("Set an API key (COHERE_API_KEY, OPENAI_API_KEY) or start Ollama.")
                raise typer.Exit(1)

            if non_interactive:
                llm_choice = llm_providers[0]
                embedding_choice = embedding_providers[0]
                storage_choice = "memory"
            else:
                ui.section("Configuration")

                llm_choice = ui.prompt_numbered_choice(
                    "LLM provider", llm_providers, llm_providers[0]
                )
                print()
                embedding_choice = ui.prompt_numbered_choice(
                    "Embedding provider", embedding_providers, embedding_providers[0]
                )
                print()
                storage_backends = ["memory", "file"]
                storage_choice = ui.prompt_numbered_choice(
                    "Storage backend", storage_backends, "memory"
                )

            engine_config_yaml = _generate_graphrag_config(
                llm_provider=llm_choice,
                embedding_provider=embedding_choice,
                storage_backend=storage_choice,
            )

        # =====================================================================
        # Clara Configuration
        # =====================================================================
        elif default_engine == "clara":
            if non_interactive:
                variant_choice = "e2e"
                device_choice = "cuda"
                compression_choice = 16
            else:
                ui.section("Configuration")

                # Model variant
                variant_descs = [
                    "e2e - Full retrieval + generation (recommended)",
                    "instruct - Instruction-tuned",
                    "base - Compression only",
                ]
                selected = ui.prompt_numbered_choice(
                    "Model variant", variant_descs, variant_descs[0]
                )
                variant_choice = selected.split(" - ")[0]

                # Device
                print()
                devices = ["cuda", "cpu"]
                device_choice = ui.prompt_numbered_choice("Device", devices, "cuda")

                # Compression rate
                print()
                rate_descs = [
                    "4 - Lowest compression, highest quality",
                    "16 - Balanced (recommended)",
                    "32 - Higher compression",
                    "64 - High compression",
                    "128 - Maximum compression",
                ]
                selected = ui.prompt_numbered_choice("Compression rate", rate_descs, rate_descs[1])
                compression_choice = int(selected.split(" - ")[0])

            engine_config_yaml = _generate_clara_config(
                model_variant=variant_choice,
                device=device_choice,
                compression_rate=compression_choice,
            )

        # =====================================================================
        # Unknown Engine - Copy Default
        # =====================================================================
        else:
            engine_config_yaml = _copy_engine_default_config(default_engine, registry)

        ui.section("Generated Configuration")
        ui.print("Global config (.fitz/config.yaml):", "bold")
        ui.syntax(global_config_yaml, "yaml")

        if engine_config_yaml:
            print()
            ui.print(f"Engine config (.fitz/config/{default_engine}.yaml):", "bold")
            ui.syntax(engine_config_yaml, "yaml")

        if show_config:
            return

        ui.section("Saving")
        fitz_dir = FitzPaths.workspace()
        fitz_config = FitzPaths.config()
        fitz_dir.mkdir(parents=True, exist_ok=True)
        fitz_config.write_text(global_config_yaml)
        ui.success(f"Saved global config to {fitz_config}")

        if engine_config_yaml:
            engine_config_path = FitzPaths.engine_config(default_engine)
            FitzPaths.ensure_config_dir()
            engine_config_path.write_text(engine_config_yaml)
            ui.success(f"Saved engine config to {engine_config_path}")

        ui.section("Done!")
        ui.info(
            f"Use 'fitz quickstart <folder> \"question\" --engine {default_engine}' to get started."
        )
        return

    # =========================================================================
    # Detect System (for fitz_rag plugins)
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
        chat_model_smart = _get_default_model("chat", chat_choice, "smart")
        chat_model_fast = _get_default_model("chat", chat_choice, "fast")
        embedding_choice = _get_default_or_first(avail_embedding, default_embedding)
        embedding_model = _get_default_model("embedding", embedding_choice)
        rerank_choice = (
            _get_default_or_first(avail_rerank, default_rerank) if avail_rerank else None
        )
        rerank_model = _get_default_model("rerank", rerank_choice) if rerank_choice else ""
        vector_db_choice = _get_default_or_first(avail_vector_db, default_vector_db)
        retrieval_choice = _get_default_or_first(avail_retrieval, default_retrieval)
        # Chunking defaults from default.yaml
        chunker_choice = default_chunker
        chunk_size = default_chunk_size
        chunk_overlap = default_chunk_overlap

    else:
        # Interactive selection
        ui.section("Configuration")

        # Chat plugin with smart/fast model selection
        chat_choice = ui.prompt_numbered_choice(
            "Chat plugin", avail_chat, _get_default_or_first(avail_chat, default_chat)
        )
        chat_model_smart = _prompt_model("chat", chat_choice, "smart")
        chat_model_fast = _prompt_model("chat", chat_choice, "fast")

        # Embedding
        print()
        embedding_choice = ui.prompt_numbered_choice(
            "Embedding plugin",
            avail_embedding,
            _get_default_or_first(avail_embedding, default_embedding),
        )
        embedding_model = _prompt_model("embedding", embedding_choice)

        # Rerank (optional) - comes after embedding
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
            print()
            ui.info("Rerank: not available (no rerank plugins detected)")

        # Vector DB - comes after rerank
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

    # Generate global config (just default_engine)
    global_config_yaml = _generate_global_config(default_engine)

    # Generate engine-specific config
    # fitz_rag uses interactive plugin selection, other engines use their default.yaml
    if default_engine == "fitz_rag":
        engine_config_yaml = _generate_fitz_rag_config(
            chat=chat_choice,
            chat_model_smart=chat_model_smart,
            chat_model_fast=chat_model_fast,
            embedding=embedding_choice,
            embedding_model=embedding_model,
            rerank=rerank_choice,
            rerank_model=rerank_model,
            vector_db=vector_db_choice,
            retrieval=retrieval_choice,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            chunker=chunker_choice,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        # For any other engine, use auto-discovery to copy default.yaml
        engine_config_yaml = _copy_engine_default_config(default_engine, registry)
        if engine_config_yaml is None:
            ui.warning(f"No default config found for engine '{default_engine}'")
            engine_config_yaml = f"# {default_engine} configuration\n# Edit as needed\n"

    # =========================================================================
    # Show Config
    # =========================================================================

    ui.section("Generated Configuration")
    ui.print("Global config (.fitz/config.yaml):", "bold")
    ui.syntax(global_config_yaml, "yaml")
    print()
    ui.print(f"Engine config (.fitz/config/{default_engine}.yaml):", "bold")
    ui.syntax(engine_config_yaml, "yaml")

    if show_config:
        return

    # =========================================================================
    # Save Config
    # =========================================================================

    ui.section("Saving")

    fitz_dir = FitzPaths.workspace()
    fitz_config = FitzPaths.config()
    engine_config_path = FitzPaths.engine_config(default_engine)

    # Confirm overwrite if exists
    if (fitz_config.exists() or engine_config_path.exists()) and not non_interactive:
        if not ui.prompt_confirm("Config exists. Overwrite?", default=False):
            ui.warning("Aborted.")
            raise typer.Exit(0)

    fitz_dir.mkdir(parents=True, exist_ok=True)
    fitz_config.write_text(global_config_yaml)
    ui.success(f"Saved global config to {fitz_config}")

    FitzPaths.ensure_config_dir()
    engine_config_path.write_text(engine_config_yaml)
    ui.success(f"Saved engine config to {engine_config_path}")

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
