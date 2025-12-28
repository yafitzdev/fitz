# fitz_ai/cli/commands/ingest.py
"""
Document ingestion with incremental (diff) support.

Usage:
    fitz ingest              # Interactive mode
    fitz ingest ./src        # Ingest specific directory
    fitz ingest ./src -y     # Non-interactive with defaults
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import typer

from fitz_ai.cli.ui import RICH, console, ui
from fitz_ai.core.config import ConfigNotFoundError, load_config_dict
from fitz_ai.core.paths import FitzPaths
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Config Loading
# =============================================================================


def _load_config() -> dict:
    """Load config or exit with helpful message."""
    try:
        return load_config_dict(FitzPaths.config())
    except ConfigNotFoundError:
        ui.error("No config found. Run 'fitz init' first.")
        raise typer.Exit(1)


def _get_collections(config: dict) -> List[str]:
    """Get list of existing collections from vector DB."""
    from fitz_ai.vector_db.registry import get_vector_db_plugin

    try:
        vdb_plugin = config.get("vector_db", {}).get("plugin_name", "qdrant")
        vdb_kwargs = config.get("vector_db", {}).get("kwargs", {})
        vdb = get_vector_db_plugin(vdb_plugin, **vdb_kwargs)
        return sorted(vdb.list_collections())
    except Exception:
        return []


def _suggest_collection_name(source: str) -> str:
    """Suggest a collection name from source path."""
    from pathlib import Path

    path = Path(source).resolve()
    # Use folder name, sanitized
    name = path.name if path.is_dir() else path.parent.name
    # Replace spaces/special chars with underscores
    return name.replace(" ", "_").replace("-", "_").lower()


def _is_code_project(source: str) -> bool:
    """Check if source contains code files (suggests artifacts would be useful)."""
    from pathlib import Path

    code_extensions = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java"}
    path = Path(source)

    if path.is_file():
        return path.suffix in code_extensions

    # Check first 100 files for code
    count = 0
    for f in path.rglob("*"):
        if f.is_file() and f.suffix in code_extensions:
            return True
        count += 1
        if count > 100:
            break
    return False


def _get_available_artifacts(has_llm: bool = False) -> List[tuple]:
    """
    Get available artifact plugins as (name, description) tuples.

    Args:
        has_llm: Whether an LLM client is available (enables LLM-requiring artifacts)

    Returns:
        List of (name, description) tuples for available artifacts
    """
    from fitz_ai.ingest.enrichment.artifacts.registry import get_artifact_registry

    registry = get_artifact_registry()
    result = []

    for name in registry.list_plugin_names():
        info = registry.get_plugin(name)
        if info is None:
            continue

        # Skip LLM-requiring artifacts if no LLM available
        if info.requires_llm and not has_llm:
            desc = f"{info.description} (requires LLM)"
        else:
            desc = info.description

        result.append((name, desc))

    return result


def _parse_artifact_selection(
    artifacts_arg: Optional[str], available: List[str]
) -> Optional[List[str]]:
    """
    Parse the --artifacts argument.

    Args:
        artifacts_arg: The --artifacts argument value
        available: List of available artifact names

    Returns:
        List of selected artifact names, or None if should prompt interactively
    """
    if artifacts_arg is None:
        return None  # Interactive selection

    artifacts_arg = artifacts_arg.strip().lower()

    if artifacts_arg == "all":
        return available
    elif artifacts_arg == "none":
        return []
    else:
        # Comma-separated list
        requested = [a.strip() for a in artifacts_arg.split(",")]
        # Filter to valid names
        return [a for a in requested if a in available]


def _build_chunking_router_config(config: dict):
    """
    Build ChunkingRouterConfig from fitz.yaml config.

    Expected config structure:
        chunking:
          default:
            plugin_name: simple
            kwargs:
              chunk_size: 1000
              chunk_overlap: 0
          by_extension:
            .md:
              plugin_name: markdown
              kwargs: {...}
          warn_on_fallback: true
    """
    from fitz_ai.engines.classic_rag.config import (
        ChunkingRouterConfig,
        ExtensionChunkerConfig,
    )

    chunking = config.get("chunking", {})

    # Build default config
    default_cfg = chunking.get("default", {})
    default = ExtensionChunkerConfig(
        plugin_name=default_cfg.get("plugin_name", "simple"),
        kwargs=default_cfg.get("kwargs", {"chunk_size": 1000, "chunk_overlap": 0}),
    )

    # Build per-extension configs
    by_extension = {}
    for ext, ext_cfg in chunking.get("by_extension", {}).items():
        by_extension[ext] = ExtensionChunkerConfig(
            plugin_name=ext_cfg.get("plugin_name", "simple"),
            kwargs=ext_cfg.get("kwargs", {}),
        )

    return ChunkingRouterConfig(
        default=default,
        by_extension=by_extension,
        warn_on_fallback=chunking.get("warn_on_fallback", True),
    )


# =============================================================================
# Adapter Classes
# =============================================================================


class ParserAdapter:
    """Adapts ingestion plugin to Parser protocol."""

    def __init__(self, plugin):
        self._plugin = plugin

    def parse(self, path: str) -> str:
        """Parse a file and return its text content."""
        docs = list(self._plugin.ingest(path, kwargs={}))
        if not docs:
            return ""
        return docs[0].content


class VectorDBWriterAdapter:
    """Adapts vector DB client to VectorDBWriter protocol."""

    def __init__(self, client):
        self._client = client

    def upsert(
        self, collection: str, points: List[Dict[str, Any]], defer_persist: bool = False
    ) -> None:
        """Upsert points into collection."""
        # Try to pass defer_persist if supported
        try:
            self._client.upsert(collection, points, defer_persist=defer_persist)
        except TypeError:
            self._client.upsert(collection, points)

    def flush(self) -> None:
        """Flush any pending writes to disk."""
        if hasattr(self._client, "flush"):
            self._client.flush()


# =============================================================================
# Main Command
# =============================================================================


def command(
    source: Optional[str] = typer.Argument(
        None,
        help="Source path (file or directory). Prompts if not provided.",
    ),
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        "-c",
        help="Collection name. Uses config default if not provided.",
    ),
    non_interactive: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Non-interactive mode, use defaults.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-ingest all files, ignoring state.",
    ),
    artifacts: Optional[str] = typer.Option(
        None,
        "--artifacts",
        "-a",
        help="Artifacts to generate: 'all', 'none', or comma-separated list (e.g. 'navigation_index,interface_catalog'). Interactive selection if not provided.",
    ),
    hierarchy: bool = typer.Option(
        False,
        "--hierarchy",
        "-H",
        help="Enable hierarchical summaries (group + corpus level). Zero-config, uses smart defaults.",
    ),
) -> None:
    """
    Ingest documents into the vector database.

    Interactive mode guides you through source, collection, and artifacts.
    Incremental by default - only processes new/changed files.

    Examples:
        fitz ingest                  # Interactive mode (prompts for artifacts)
        fitz ingest ./src            # Specify source, interactive artifacts
        fitz ingest ./src -a all     # All applicable artifacts
        fitz ingest ./src -a none    # No artifacts
        fitz ingest ./src -a navigation_index,interface_catalog  # Specific artifacts
        fitz ingest ./src -f         # Force re-ingest
        fitz ingest ./src -a all -y  # Non-interactive with all artifacts
        fitz ingest ./docs --hierarchy  # Enable hierarchical summaries (zero-config)
    """
    from fitz_ai.ingest.chunking.router import ChunkingRouter
    from fitz_ai.ingest.diff import run_diff_ingest
    from fitz_ai.ingest.ingestion.registry import get_ingest_plugin
    from fitz_ai.ingest.state import IngestStateManager
    from fitz_ai.llm.registry import get_llm_plugin
    from fitz_ai.vector_db.registry import get_vector_db_plugin

    # =========================================================================
    # Load config
    # =========================================================================

    config = _load_config()

    embedding_plugin = config.get("embedding", {}).get("plugin_name", "cohere")
    embedding_model = (
        config.get("embedding", {}).get("kwargs", {}).get("model", "embed-english-v3.0")
    )
    embedding_id = f"{embedding_plugin}:{embedding_model}"

    vector_db_plugin = config.get("vector_db", {}).get("plugin_name", "qdrant")
    default_collection = config.get("retrieval", {}).get("collection", "default")

    # Get chunking config from fitz.yaml
    chunking_config = config.get("chunking", {})
    default_chunker = chunking_config.get("default", {}).get("plugin_name", "simple")
    chunk_size = chunking_config.get("default", {}).get("kwargs", {}).get("chunk_size", 1000)
    chunk_overlap = chunking_config.get("default", {}).get("kwargs", {}).get("chunk_overlap", 0)

    # =========================================================================
    # Header
    # =========================================================================

    ui.header("Fitz Ingest", "Feed your documents into the vector database")
    if force:
        ui.warning("Force mode: will re-ingest all files")
    else:
        ui.info("Incremental mode: skipping unchanged files")
    ui.info(f"Embedding: {embedding_id}")
    ui.info(f"Vector DB: {vector_db_plugin}")
    ui.info(f"Chunking: {default_chunker} (size={chunk_size}, overlap={chunk_overlap})")
    if hierarchy:
        ui.info("Hierarchy: enabled (group + corpus summaries)")

    print()

    # =========================================================================
    # Interactive Prompts
    # =========================================================================

    if non_interactive:
        if source is None:
            ui.error("Source path required in non-interactive mode.")
            ui.info("Usage: fitz ingest ./docs -y")
            raise typer.Exit(1)

        if collection is None:
            collection = default_collection

        ui.info(f"Source: {source}")
        ui.info(f"Collection: {collection}")

    else:
        # 1. Source path
        if source is None:
            source = ui.prompt_path("Source path", ".")

        # 2. Collection - smart selection
        if collection is None:
            existing_collections = _get_collections(config)

            if existing_collections:
                # Collections exist - let user choose or create new
                print()
                choices = existing_collections + ["+ Create new"]
                selected = ui.prompt_numbered_choice(
                    "Collection", choices, default_collection, indent=False
                )
                if selected == "+ Create new":
                    suggested = _suggest_collection_name(source)
                    collection = ui.prompt_text("Collection name", suggested)
                else:
                    collection = selected
            else:
                # No collections exist - prompt for name
                suggested = _suggest_collection_name(source)
                collection = ui.prompt_text("Collection name", suggested)

        # 3. Artifacts - interactive multi-select for code projects
        if artifacts is None and _is_code_project(source):
            # Check if user has a chat LLM configured
            has_chat_llm = bool(config.get("chat", {}).get("plugin_name"))
            available_artifacts = _get_available_artifacts(has_llm=has_chat_llm)
            if available_artifacts:
                # Default to all artifacts if LLM is available, otherwise structural only
                if has_chat_llm:
                    defaults = [name for name, _ in available_artifacts]
                else:
                    defaults = [
                        name for name, desc in available_artifacts if "requires LLM" not in desc
                    ]
                selected_artifacts = ui.prompt_multi_select(
                    "Select artifacts to generate",
                    available_artifacts,
                    defaults=defaults,
                )
                artifacts = ",".join(selected_artifacts) if selected_artifacts else "none"

        # 4. Hierarchy - prompt for hierarchical summaries (requires LLM)
        if not hierarchy:
            has_chat_llm = bool(config.get("chat", {}).get("plugin_name"))
            if has_chat_llm:
                hierarchy = ui.prompt_confirm(
                    "Enable hierarchical summaries? (synthesizes insights across documents)",
                    default=False,
                )

    print()

    # =========================================================================
    # Initialize components
    # =========================================================================

    # Determine total steps based on whether artifacts are enabled
    has_artifacts = artifacts != "none" and (artifacts is not None or _is_code_project(source))
    total_steps = 4 if has_artifacts else 3

    ui.step(1, total_steps, "Initializing...")

    try:
        # State manager
        state_manager = IngestStateManager()
        state_manager.load()

        # Ingest plugin (for parsing)
        IngestPluginCls = get_ingest_plugin("local")
        ingest_plugin = IngestPluginCls()
        parser = ParserAdapter(ingest_plugin)

        # Build chunking router from config
        router_config = _build_chunking_router_config(config)
        chunking_router = ChunkingRouter.from_config(router_config)

        ui.info(f"Router: {chunking_router}")

        # Embedder
        embedding_kwargs = config.get("embedding", {}).get("kwargs", {})
        embedder = get_llm_plugin(
            plugin_type="embedding", plugin_name=embedding_plugin, **embedding_kwargs
        )

        # Vector DB writer
        vector_client = get_vector_db_plugin(vector_db_plugin)
        writer = VectorDBWriterAdapter(vector_client)

        # Enrichment pipeline (from config, CLI selection can override)
        enrichment_pipeline = None
        enrichment_cfg = config.get("enrichment", {})

        # Parse artifact selection from CLI or config
        available_artifact_names = [name for name, _ in _get_available_artifacts(has_llm=False)]
        selected_artifacts = _parse_artifact_selection(artifacts, available_artifact_names)

        # If artifacts were selected (via CLI or interactive), enable enrichment
        if selected_artifacts is not None and len(selected_artifacts) > 0:
            enrichment_cfg["enabled"] = True
            enrichment_cfg["artifacts"] = {"enabled": selected_artifacts}
            # Note: Summaries are opt-in (default False) - user can enable in config
            # Now that we use batch summarization (20 chunks per LLM call), it's efficient
        elif selected_artifacts is not None and len(selected_artifacts) == 0:
            # Explicitly disabled via 'none'
            enrichment_cfg["enabled"] = False

        # Handle --hierarchy flag (zero-config hierarchical summaries)
        if hierarchy:
            enrichment_cfg["enabled"] = True
            if "hierarchy" not in enrichment_cfg:
                enrichment_cfg["hierarchy"] = {}
            enrichment_cfg["hierarchy"]["enabled"] = True
            # Uses smart defaults: group_by="source", default prompts

        if enrichment_cfg.get("enabled", False):
            from pathlib import Path

            from fitz_ai.ingest.enrichment import EnrichmentConfig, EnrichmentPipeline
            from fitz_ai.ingest.enrichment.artifacts.registry import (
                get_artifact_registry,
            )

            enrichment_config = EnrichmentConfig.from_dict(enrichment_cfg)

            # Check if any selected artifact requires LLM or if hierarchy is enabled
            chat_client = None
            needs_llm = False

            # Check artifacts that require LLM
            if selected_artifacts:
                registry = get_artifact_registry()
                needs_llm = any(
                    registry.get_plugin(name) and registry.get_plugin(name).requires_llm
                    for name in selected_artifacts
                )

            # Hierarchy always requires LLM for summarization
            if hierarchy:
                needs_llm = True

            if needs_llm:
                # Get chat client from config - use "fast" tier for enrichment tasks
                chat_plugin = config.get("chat", {}).get("plugin_name", "cohere")
                chat_kwargs = config.get("chat", {}).get("kwargs", {})
                # Include user's model overrides if present
                chat_models = config.get("chat", {}).get("models")
                if chat_models:
                    chat_kwargs["models"] = chat_models
                chat_client = get_llm_plugin(
                    plugin_type="chat", plugin_name=chat_plugin, tier="fast", **chat_kwargs
                )
                model_name = chat_client.params.get("model", "unknown")
                if hierarchy:
                    ui.info(f"Chat LLM: {chat_plugin}:{model_name} (fast tier for summaries)")
                else:
                    ui.info(f"Chat LLM: {chat_plugin}:{model_name} (fast tier for artifacts)")

            enrichment_pipeline = EnrichmentPipeline(
                config=enrichment_config,
                project_root=Path(source).resolve(),
                chat_client=chat_client,
            )

            # Show which artifacts will be generated
            plugins = enrichment_pipeline.get_applicable_artifact_plugins()
            plugin_names = [p.name for p in plugins]
            if plugin_names:
                ui.info(f"Artifacts: {', '.join(plugin_names)}")

    except Exception as e:
        ui.error(f"Failed to initialize: {e}")
        raise typer.Exit(1)

    ui.success("Initialized")

    # =========================================================================
    # Generate artifacts (if enabled)
    # =========================================================================

    current_step = 1
    artifacts_generated = 0
    artifact_errors: List[str] = []

    if has_artifacts and enrichment_pipeline is not None:
        current_step += 1
        ui.step(current_step, total_steps, "Generating artifacts...")

        from fitz_ai.ingest.diff.executor import DiffIngestExecutor

        # Create executor for artifact generation
        executor = DiffIngestExecutor(
            state_manager=state_manager,
            vector_db_writer=writer,
            embedder=embedder,
            parser=parser,
            chunking_router=chunking_router,
            collection=collection,
            embedding_id=embedding_id,
            enrichment_pipeline=enrichment_pipeline,
        )

        try:
            if RICH:
                from rich.progress import (
                    BarColumn,
                    Progress,
                    SpinnerColumn,
                    TaskProgressColumn,
                    TextColumn,
                )

                progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TextColumn("[dim]{task.fields[name]}[/dim]"),
                    console=console,
                )

                task_id = None

                def on_artifact_progress(current: int, total: int, name: str):
                    nonlocal task_id
                    if task_id is None:
                        task_id = progress.add_task("Artifacts", total=total, name="")
                    progress.update(task_id, completed=current, name=name if name != "Done" else "")

                with progress:
                    artifacts_generated, artifact_errors = executor.ingest_artifacts(
                        on_progress=on_artifact_progress
                    )
            else:
                artifacts_generated, artifact_errors = executor.ingest_artifacts()

            if artifacts_generated > 0:
                ui.success(f"Generated {artifacts_generated} artifacts")
            else:
                ui.info("No artifacts generated")

        except Exception as e:
            ui.error(f"Artifact generation failed: {e}")
            artifact_errors.append(str(e))

    # =========================================================================
    # Run diff ingestion
    # =========================================================================

    current_step += 1
    ui.step(current_step, total_steps, "Ingesting files...")

    try:
        # Set up progress tracking
        if RICH:
            from pathlib import Path as PathLib

            from rich.progress import (
                BarColumn,
                Progress,
                SpinnerColumn,
                TaskProgressColumn,
                TextColumn,
            )

            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("[dim]{task.fields[file]}[/dim]"),
                console=console,
            )

            task_id = None

            def on_progress(current: int, total: int, file_path: str):
                nonlocal task_id
                if task_id is None:
                    task_id = progress.add_task("Ingesting", total=total, file="")
                file_name = PathLib(file_path).name if file_path != "Done" else ""
                progress.update(task_id, completed=current, file=file_name)

            with progress:
                summary = run_diff_ingest(
                    source=source,
                    state_manager=state_manager,
                    vector_db_writer=writer,
                    embedder=embedder,
                    parser=parser,
                    chunking_router=chunking_router,
                    collection=collection,
                    embedding_id=embedding_id,
                    enrichment_pipeline=enrichment_pipeline,
                    force=force,
                    on_progress=on_progress,
                    skip_artifacts=has_artifacts,  # Skip if we already did them
                )
        else:
            # Simple text progress for non-Rich mode
            last_pct = -1

            def on_progress(current: int, total: int, file_path: str):
                nonlocal last_pct
                if total == 0:
                    return
                pct = int(current * 100 / total)
                if pct >= last_pct + 10 or current == total:
                    print(f"  {current}/{total} ({pct}%)")
                    last_pct = pct

            summary = run_diff_ingest(
                source=source,
                state_manager=state_manager,
                vector_db_writer=writer,
                embedder=embedder,
                parser=parser,
                chunking_router=chunking_router,
                collection=collection,
                embedding_id=embedding_id,
                enrichment_pipeline=enrichment_pipeline,
                force=force,
                on_progress=on_progress,
                skip_artifacts=has_artifacts,  # Skip if we already did them
            )

        # Add artifact stats from separate step
        summary.artifacts_generated = artifacts_generated
        summary.errors += len(artifact_errors)
        summary.error_details.extend(artifact_errors)

    except Exception as e:
        ui.error(f"Ingestion failed: {e}")
        logger.exception("Ingestion error")
        raise typer.Exit(1)

    # =========================================================================
    # Summary
    # =========================================================================

    current_step += 1
    ui.step(current_step, total_steps, "Complete!")
    print()

    if RICH:
        from rich.table import Table

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="bold")
        table.add_column("Value", style="cyan")

        table.add_row("Scanned", str(summary.scanned))
        table.add_row("Ingested", str(summary.ingested))
        table.add_row("Skipped", str(summary.skipped))
        table.add_row("Marked deleted", str(summary.marked_deleted))
        if summary.artifacts_generated > 0:
            table.add_row("Artifacts", str(summary.artifacts_generated))
        if summary.hierarchy_summaries > 0:
            table.add_row("Hierarchy summaries", str(summary.hierarchy_summaries))
        table.add_row("Errors", str(summary.errors))
        table.add_row("Duration", f"{summary.duration_seconds:.1f}s")

        console.print(table)
    else:
        print(f"  Scanned: {summary.scanned}")
        print(f"  Ingested: {summary.ingested}")
        print(f"  Skipped: {summary.skipped}")
        print(f"  Marked deleted: {summary.marked_deleted}")
        if summary.artifacts_generated > 0:
            print(f"  Artifacts: {summary.artifacts_generated}")
        if summary.hierarchy_summaries > 0:
            print(f"  Hierarchy summaries: {summary.hierarchy_summaries}")
        print(f"  Errors: {summary.errors}")
        print(f"  Duration: {summary.duration_seconds:.1f}s")

    if summary.errors > 0:
        print()
        ui.warning(f"{summary.errors} errors occurred:")
        for err in summary.error_details[:5]:
            ui.info(f"  • {err}")
        if len(summary.error_details) > 5:
            ui.info(f"  ... and {len(summary.error_details) - 5} more")

    print()
    ui.success(f"Documents ingested into collection '{collection}'")
