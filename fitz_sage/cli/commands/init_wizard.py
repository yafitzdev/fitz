# fitz_sage/cli/commands/init_wizard.py
"""
Init wizard - main orchestration for interactive setup.
"""

from __future__ import annotations

import typer

from fitz_sage.cli.context import CLIContext
from fitz_sage.cli.ui import RICH, console, ui
from fitz_sage.core.registry import (
    available_llm_plugins,
    available_vector_db_plugins,
)
from fitz_sage.runtime import get_default_engine, get_engine_registry

from .init_config import (
    copy_engine_default_config,
    generate_fitz_krag_config,
    generate_global_config,
)
from .init_detector import detect_system, filter_available_plugins
from .init_models import get_default_model

# Preferred plugin order: first available = default (selected on Enter).
# Ollama first because it's local, private, and free.
_PLUGIN_PREFERENCE: dict[str, list[str]] = {
    "chat": ["ollama", "cohere", "openai", "anthropic"],
    "embedding": ["ollama", "cohere", "openai"],
    "rerank": ["ollama", "cohere"],
    "vision": ["ollama", "cohere", "openai", "anthropic"],
    "vector_db": ["pgvector"],
}


def _order_by_preference(available: list[str], plugin_type: str) -> list[str]:
    """Reorder available plugins by preference. First item = default."""
    preference = _PLUGIN_PREFERENCE.get(plugin_type, [])
    ordered = [p for p in preference if p in available]
    remaining = [p for p in available if p not in ordered]
    return ordered + remaining


def _select_plugins(system, non_interactive: bool) -> dict:
    """Run shared plugin selection (chat, embedding, rerank, vision, vector_db).

    Returns:
        Dict with keys: chat, chat_model_smart, chat_model_fast, chat_model_balanced,
        embedding, embedding_model, rerank, rerank_model, vector_db,
        vision, vision_model.
    """
    # Discover plugins
    all_chat = available_llm_plugins("chat")
    all_embedding = available_llm_plugins("embedding")
    all_rerank = available_llm_plugins("rerank")
    all_vision = available_llm_plugins("vision")
    all_vector_db = available_vector_db_plugins()

    # Filter to available, then order by preference (first = default)
    avail_chat = _order_by_preference(filter_available_plugins(all_chat, "chat", system), "chat")
    avail_embedding = _order_by_preference(
        filter_available_plugins(all_embedding, "embedding", system), "embedding"
    )
    avail_rerank = _order_by_preference(
        filter_available_plugins(all_rerank, "rerank", system), "rerank"
    )
    avail_vision = _order_by_preference(
        filter_available_plugins(all_vision, "vision", system), "vision"
    )
    avail_vector_db = _order_by_preference(
        filter_available_plugins(all_vector_db, "vector_db", system), "vector_db"
    )

    # Validate minimum requirements
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
        ui.info("Ensure pgvector packages are installed: pip install psycopg pgvector pgserver")
        raise typer.Exit(1)

    if non_interactive:
        chat_choice = avail_chat[0]
        chat_model_smart = get_default_model("chat", chat_choice, "smart")
        chat_model_fast = get_default_model("chat", chat_choice, "fast")
        chat_model_balanced = get_default_model("chat", chat_choice, "balanced")
        embedding_choice = avail_embedding[0]
        embedding_model = get_default_model("embedding", embedding_choice)
        rerank_choice = None
        rerank_model = ""
        vector_db_choice = avail_vector_db[0]
        vision_choice = avail_vision[0] if avail_vision else None
        vision_model = get_default_model("vision", vision_choice) if vision_choice else ""
    else:
        # Interactive selection (first item in each list is the default)
        ui.section("Configuration")

        chat_choice = ui.prompt_numbered_choice("Chat plugin", avail_chat, avail_chat[0])
        chat_model_smart = get_default_model("chat", chat_choice, "smart")
        chat_model_fast = get_default_model("chat", chat_choice, "fast")
        chat_model_balanced = get_default_model("chat", chat_choice, "balanced")

        print()
        embedding_choice = ui.prompt_numbered_choice(
            "Embedding plugin", avail_embedding, avail_embedding[0]
        )
        embedding_model = get_default_model("embedding", embedding_choice)

        rerank_choice = None
        rerank_model = ""
        print()
        rerank_options = ["None (no reranking)"] + avail_rerank
        selected = ui.prompt_numbered_choice(
            "Rerank plugin (optional, improves retrieval accuracy)",
            rerank_options,
            rerank_options[0],
        )
        if selected != "None (no reranking)":
            rerank_choice = selected
            rerank_model = get_default_model("rerank", rerank_choice)

        vision_choice = None
        vision_model = ""
        if avail_vision:
            print()
            vision_choice = ui.prompt_numbered_choice(
                "Vision plugin (for image/figure descriptions)",
                avail_vision,
                avail_vision[0],
            )
            vision_model = get_default_model("vision", vision_choice)
        else:
            print()
            ui.info("Vision: not available (no vision plugins detected)")

        vector_db_choice = avail_vector_db[0]

    return {
        "chat": chat_choice,
        "chat_model_smart": chat_model_smart,
        "chat_model_fast": chat_model_fast,
        "chat_model_balanced": chat_model_balanced,
        "embedding": embedding_choice,
        "embedding_model": embedding_model,
        "rerank": rerank_choice,
        "rerank_model": rerank_model,
        "vector_db": vector_db_choice,
        "vision": vision_choice,
        "vision_model": vision_model,
    }


def _run_fitz_krag_wizard(system, non_interactive: bool) -> str:
    """Run FitzKRAG-specific configuration wizard.

    Returns:
        Generated FitzKRAG config YAML string.
    """
    plugins = _select_plugins(system, non_interactive)

    return generate_fitz_krag_config(
        chat=plugins["chat"],
        chat_model_smart=plugins["chat_model_smart"],
        chat_model_fast=plugins["chat_model_fast"],
        chat_model_balanced=plugins["chat_model_balanced"],
        embedding=plugins["embedding"],
        embedding_model=plugins["embedding_model"],
        rerank=plugins["rerank"],
        rerank_model=plugins["rerank_model"],
        vector_db=plugins["vector_db"],
        vision=plugins["vision"],
        vision_model=plugins["vision_model"],
    )


def _save_configs(
    global_config_yaml: str,
    engine_config_yaml: str | None,
    default_engine: str,
    non_interactive: bool,
) -> None:
    """Save configuration files to disk."""
    from fitz_sage.core.paths import FitzPaths

    fitz_dir = FitzPaths.workspace()
    fitz_config = FitzPaths.config()

    # Confirm overwrite if exists
    if fitz_config.exists() and not non_interactive:
        if not ui.prompt_confirm("Config exists. Overwrite?", default=False):
            ui.warning("Aborted.")
            raise typer.Exit(0)

    fitz_dir.mkdir(parents=True, exist_ok=True)

    # Write engine config as the main config.yaml (flat format)
    if engine_config_yaml:
        fitz_config.write_text(engine_config_yaml)
        ui.success(f"Saved config to {fitz_config}")
    else:
        fitz_config.write_text(global_config_yaml)
        ui.success(f"Saved global config to {fitz_config}")


def _show_next_steps() -> None:
    """Display next steps after successful configuration."""
    if RICH:
        console.print(
            """
[green]Your configuration is ready![/green]

Next steps:
  [cyan]fitz query "your question" --source ./docs[/cyan]  # Register + query
  [cyan]fitz query "your question"[/cyan]                  # Query existing collection
  [cyan]fitz config --doctor[/cyan]                        # Verify setup
"""
        )
    else:
        print(
            """
Your configuration is ready!

Next steps:
  fitz query "your question" --source ./docs  # Register + query
  fitz query "your question"                  # Query existing collection
  fitz config --doctor                        # Verify setup
"""
        )


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

    Detects available providers (API keys, Ollama, pgvector) and
    creates a working configuration file.

    Examples:
        fitz init           # Interactive wizard
        fitz init -y        # Auto-detect and use defaults
        fitz init --show    # Preview config without saving
    """
    # Header
    ui.header("Fitz Init", "Let's configure your RAG pipeline")

    # Select default engine
    ctx = CLIContext.load()
    registry = get_engine_registry()

    if non_interactive:
        default_engine = get_default_engine()
    else:
        ui.section("Engine Setup")
        default_engine = ctx.select_engine()

    # Check if engine supports collections (FitzRAG does, others may not)
    caps = registry.get_capabilities(default_engine)

    if not caps.supports_collections:
        # Non-collection engines - use default config from registry
        global_config_yaml = generate_global_config(default_engine)
        engine_config_yaml = copy_engine_default_config(default_engine, registry)

        # Show config
        ui.section("Generated Configuration")
        ui.print("Global config (.fitz/config.yaml):", "bold")
        ui.syntax(global_config_yaml, "yaml")

        if engine_config_yaml:
            print()
            ui.print(f"Engine config (.fitz/config/{default_engine}.yaml):", "bold")
            ui.syntax(engine_config_yaml, "yaml")

        if show_config:
            return

        # Save
        ui.section("Saving")
        _save_configs(global_config_yaml, engine_config_yaml, default_engine, non_interactive)

        ui.section("Done!")
        ui.info(
            f"Use 'fitz quickstart <folder> \"question\" --engine {default_engine}' to get started."
        )
        return

    # FitzRAG and other collection-based engines
    ui.section("Detecting System")
    system = detect_system()

    # Show detection results
    ui.status(
        "OpenAI-compatible endpoint",
        system.llm_endpoint.available,
        (
            f"{system.llm_endpoint.base_url} — {system.llm_endpoint.details}"
            if system.llm_endpoint.available
            else system.llm_endpoint.details
        ),
    )
    ui.status("pgvector", system.pgvector.available, system.pgvector.details)

    for name, key_status in system.api_keys.items():
        ui.status(name.capitalize(), key_status.available)

    # Generate configs
    global_config_yaml = generate_global_config(default_engine)

    if default_engine == "fitz_krag":
        engine_config_yaml = _run_fitz_krag_wizard(system, non_interactive)
    else:
        # For any other engine, use auto-discovery
        engine_config_yaml = copy_engine_default_config(default_engine, registry)
        if engine_config_yaml is None:
            ui.warning(f"No default config found for engine '{default_engine}'")
            engine_config_yaml = f"# {default_engine} configuration\n# Edit as needed\n"

    # Show config
    ui.section("Generated Configuration")
    ui.print("Config (.fitz/config.yaml):", "bold")
    ui.syntax(engine_config_yaml, "yaml")

    if show_config:
        return

    # Save
    ui.section("Saving")
    _save_configs(global_config_yaml, engine_config_yaml, default_engine, non_interactive)

    # Done
    ui.section("Done!")
    _show_next_steps()
