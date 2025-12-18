# fitz/cli/quickstart.py

"""
Quickstart command: Run end-to-end test.

This command:
1. Checks system for required dependencies
2. Creates sample documents about RAG
3. Ingests them into a vector database
4. Runs a test query to verify everything works
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import typer

# Import plugin registries for discovery
from fitz.llm.registry import available_llm_plugins
from fitz.vector_db.registry import available_vector_db_plugins
from fitz.core.detect import detect_all

# Rich for pretty output (optional, falls back gracefully)
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


# =============================================================================
# Sample Documents
# =============================================================================

SAMPLE_DOCUMENTS = {
    "what_is_rag.txt": """# What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with language model generation. The system first retrieves relevant documents from a knowledge base using semantic search, then uses those documents as context for generating accurate responses.

Key benefits of RAG:
- Reduces hallucinations by grounding responses in retrieved facts
- Enables access to up-to-date information not in training data
- Provides source attribution for answers
- Allows knowledge base updates without retraining

RAG is particularly useful for question-answering systems, chatbots, and any application requiring factual accuracy with source citations.
""",
    "vector_databases.txt": """# Vector Databases

Vector databases are specialized storage systems designed to store and query high-dimensional embedding vectors. They enable semantic search by converting text into numerical vectors and finding similar vectors using distance metrics like cosine similarity.

Popular vector databases include:
- Qdrant: High-performance vector search engine
- Pinecone: Managed vector database service
- Weaviate: Open-source vector search engine
- Milvus: Scalable vector database
- Chroma: Lightweight vector store

These databases are essential components in modern RAG systems, enabling fast similarity search across millions of documents.
""",
    "building_rag_pipeline.txt": """# Building a RAG Pipeline

A production-ready RAG pipeline typically consists of several components:

## Core Components

### 1. Document Ingestion
- Load documents from files, APIs, or databases
- Split into manageable chunks (typically 200-500 tokens)
- Preserve metadata for filtering and attribution

### 2. Embedding Generation
- Convert text chunks to vector embeddings
- Use models like OpenAI's text-embedding-3-small or open-source alternatives
- Batch processing for efficiency

### 3. Vector Storage
- Store embeddings with their metadata
- Index for fast similarity search
- Support filtering by metadata

### 4. Retrieval
- Embed the user's query
- Find top-k similar chunks
- Optional: Rerank results for better relevance

### 5. Generation
- Build a prompt with retrieved context
- Use an LLM to generate a grounded answer
- Include citations to sources

## Best Practices

- Chunk documents with overlap to preserve context
- Use reranking for improved retrieval quality
- Implement strict grounding to reduce hallucination
- Monitor and evaluate retrieval quality
""",
}


# =============================================================================
# Helper Functions
# =============================================================================


def print_step(step: int, total: int, message: str) -> None:
    """Print a step indicator."""
    if RICH_AVAILABLE:
        console.print(f"\n[bold blue][{step}/{total}][/bold blue] {message}")
    else:
        print(f"\n[{step}/{total}] {message}")


def print_success(message: str) -> None:
    """Print a success message."""
    if RICH_AVAILABLE:
        console.print(f"  [green]✓[/green] {message}")
    else:
        print(f"  ✓ {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    if RICH_AVAILABLE:
        console.print(f"  [red]✗[/red] {message}")
    else:
        print(f"  ✗ {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    if RICH_AVAILABLE:
        console.print(f"  [yellow]⚠[/yellow] {message}")
    else:
        print(f"  ⚠ {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    if RICH_AVAILABLE:
        console.print(f"  [dim]{message}[/dim]")
    else:
        print(f"    {message}")


# =============================================================================
# System Check (PLUGIN-AGNOSTIC!)
# =============================================================================


def check_system() -> tuple[bool, dict]:
    """
    Check system requirements and available providers using plugin registries.

    Returns (success, details) where details contains:
    - llm_provider: name of available LLM provider
    - embedding_provider: name of available embedding provider
    - vector_db: name of available vector DB
    """
    details = {
        "llm_provider": None,
        "embedding_provider": None,
        "vector_db": None,
        "issues": [],
    }

    # Check Python version
    major, minor = sys.version_info[:2]
    if major < 3 or minor < 10:
        details["issues"].append(f"Python 3.10+ required (you have {major}.{minor})")
    else:
        print_success(f"Python {major}.{minor}")

    # Get system status
    system = detect_all()

    # Discover available plugins from registries
    available_chat = available_llm_plugins("chat")
    available_embeddings = available_llm_plugins("embedding")
    available_vector_dbs = available_vector_db_plugins()

    # Find first available chat plugin
    for plugin in available_chat:
        if plugin in ["ollama", "local"] and system.ollama.available:
            details["llm_provider"] = plugin
            print_success(f"{plugin.capitalize()} is running (local)")
            break
        elif plugin in system.api_keys and system.api_keys[plugin].available:
            details["llm_provider"] = plugin
            print_success(f"{plugin.capitalize()} API key found")
            break

    if not details["llm_provider"]:
        details["issues"].append("No LLM provider found. Set an API key or install Ollama.")

    # Find first available embedding plugin
    for plugin in available_embeddings:
        if plugin in ["ollama", "local"] and system.ollama.available:
            details["embedding_provider"] = plugin
            break
        elif plugin in system.api_keys and system.api_keys[plugin].available:
            details["embedding_provider"] = plugin
            break

    if not details["embedding_provider"]:
        details["issues"].append("No embedding provider found")

    # Find first available vector database
    for plugin in available_vector_dbs:
        if plugin in ["local-faiss", "faiss"] and system.faiss.available:
            details["vector_db"] = plugin
            print_success("FAISS available (local vector DB)")
            break
        elif plugin == "qdrant" and system.qdrant.available:
            details["vector_db"] = plugin
            details["qdrant_host"] = system.qdrant.host or "localhost"
            details["qdrant_port"] = system.qdrant.port or 6333
            print_success(f"Qdrant available at {details['qdrant_host']}:{details['qdrant_port']}")
            break

    if not details["vector_db"]:
        details["issues"].append("No vector database found. Install faiss-cpu or start Qdrant.")

    success = len(details["issues"]) == 0
    return success, details


# =============================================================================
# Sample Documents
# =============================================================================


def create_sample_documents(base_dir: Path) -> list[Path]:
    """Create sample documents in the given directory."""
    base_dir.mkdir(parents=True, exist_ok=True)

    created = []
    for filename, content in SAMPLE_DOCUMENTS.items():
        file_path = base_dir / filename
        file_path.write_text(content, encoding="utf-8")
        created.append(file_path)

    return created


# =============================================================================
# Config Generation (PLUGIN-AGNOSTIC!)
# =============================================================================


def generate_quickstart_config(
    llm_provider: str,
    embedding_provider: str,
    vector_db: str,
    qdrant_host: str = "localhost",
    qdrant_port: str = "6333",
) -> str:
    """
    Generate a minimal config for quickstart using dynamic plugin config.
    """
    # Chat config (using 'chat' not 'llm')
    chat_config = f"""chat:
  plugin_name: {llm_provider}
  kwargs:
    temperature: 0.2"""

    # Embedding config
    embedding_config = f"""embedding:
  plugin_name: {embedding_provider}
  kwargs: {{}}"""

    # Vector DB config
    if vector_db == "qdrant":
        vector_db_config = f"""vector_db:
  plugin_name: qdrant
  kwargs:
    host: "{qdrant_host}"
    port: {qdrant_port}"""
    else:
        vector_db_config = f"""vector_db:
  plugin_name: {vector_db}
  kwargs: {{}}"""

    config = f"""# Fitz Quickstart Configuration
# Auto-generated by: fitz quickstart

{chat_config}

{embedding_config}

{vector_db_config}

retriever:
  plugin_name: dense
  collection: quickstart
  top_k: 3

rgs:
  enable_citations: true
  strict_grounding: true
  max_chunks: 5
"""
    return config


# =============================================================================
# Ingestion
# =============================================================================


def run_ingestion(
    docs_path: Path,
    collection: str,
    embedding_plugin: str,
    vector_db_plugin: str,
) -> tuple[bool, int]:
    """
    Run document ingestion.

    Returns (success, num_chunks).
    """
    try:
        from fitz.ingest.pipeline.ingestion_pipeline import IngestionPipeline
        from fitz.ingest.config.schema import IngestConfig, IngesterConfig, ChunkerConfig
        from fitz.llm.registry import get_llm_plugin
        from fitz.vector_db.registry import get_vector_db_plugin
        from fitz.vector_db.writer import VectorDBWriter

        # Create config
        config = IngestConfig(
            ingester=IngesterConfig(plugin_name="local", kwargs={}),
            chunker=ChunkerConfig(plugin_name="simple", kwargs={"chunk_size": 1000}),
            collection=collection,
        )

        # Create embedder - get_llm_plugin returns an INSTANCE directly
        embedder = get_llm_plugin(plugin_name=embedding_plugin, plugin_type="embedding")

        # Create vector DB writer - get_vector_db_plugin returns an INSTANCE directly
        vector_db = get_vector_db_plugin(vector_db_plugin)
        writer = VectorDBWriter(client=vector_db)

        # Create pipeline
        pipeline = IngestionPipeline(
            config=config,
            writer=writer,
            embedder=embedder,
        )

        # Run ingestion
        num_chunks = pipeline.run(source=str(docs_path))

        return True, num_chunks

    except Exception as e:
        print_error(f"Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return False, 0


# =============================================================================
# Query Test
# =============================================================================


def run_test_query(collection: str, config_path: Path) -> tuple[bool, str, list[str]]:
    """
    Run a test query through the pipeline.

    Returns (success, answer_text, sources).
    """
    try:
        from fitz.engines.classic_rag.pipeline.pipeline.engine import create_pipeline_from_yaml

        pipeline = create_pipeline_from_yaml(str(config_path))

        query = "What is RAG and how does it work?"
        result = pipeline.run(query)

        answer_text = getattr(result, "answer", "") or getattr(result, "text", "") or str(result)

        # Extract sources
        sources = []
        citations = getattr(result, "citations", None) or getattr(result, "sources", None) or []
        for c in citations[:3]:
            source_id = getattr(c, "source_id", None) or getattr(c, "label", None) or "?"
            sources.append(source_id)

        return True, answer_text, sources

    except Exception as e:
        print_error(f"Query failed: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e), []


# =============================================================================
# Main Command
# =============================================================================


def command(
    clean: bool = typer.Option(
        False,
        "--clean",
        help="Remove existing quickstart data before running",
    ),
    skip_query: bool = typer.Option(
        False,
        "--skip-query",
        help="Skip the test query (useful for CI without LLM)",
    ),
) -> None:
    """
    Run a complete end-to-end test of Fitz.

    This command:
    1. Checks your system for required dependencies
    2. Creates sample documents about RAG
    3. Ingests them into a vector database
    4. Runs a test query to verify everything works

    Perfect for new users to verify their setup!

    Examples:
        fitz quickstart           # Run full quickstart
        fitz quickstart --clean   # Clean and re-run
    """
    # Import FitzPaths for consistent path management
    from fitz.core.paths import FitzPaths

    TOTAL_STEPS = 5 if not skip_query else 4

    # Header
    if RICH_AVAILABLE:
        console.print(
            Panel.fit(
                "[bold]🚀 Fitz Quickstart[/bold]\n" "Let's verify your setup works end-to-end!",
                border_style="blue",
            )
        )
    else:
        print("\n" + "=" * 60)
        print("🚀 Fitz Quickstart")
        print("Let's verify your setup works end-to-end!")
        print("=" * 60)

    # Use FitzPaths for all paths
    quickstart_dir = FitzPaths.quickstart_docs()
    config_path = FitzPaths.quickstart_config()
    collection = "quickstart"

    # Clean if requested
    if clean:
        if quickstart_dir.exists():
            shutil.rmtree(quickstart_dir)
        if config_path.exists():
            config_path.unlink()
        # Also clean vector DB for quickstart collection
        vector_db_path = FitzPaths.vector_db()
        if vector_db_path.exists():
            shutil.rmtree(vector_db_path)
        print_info("Cleaned previous quickstart data")

    # =========================================================================
    # Step 1: Check System
    # =========================================================================
    print_step(1, TOTAL_STEPS, "Checking system...")

    success, details = check_system()

    if not success:
        print()
        for issue in details["issues"]:
            print_error(issue)
        print()
        if RICH_AVAILABLE:
            console.print("[red]Cannot continue - please fix the issues above.[/red]")
            console.print("\nRun [bold]fitz doctor[/bold] for detailed diagnostics.")
        else:
            print("Cannot continue - please fix the issues above.")
            print("\nRun 'fitz doctor' for detailed diagnostics.")
        raise typer.Exit(code=1)

    llm_provider = details["llm_provider"]
    embedding_provider = details["embedding_provider"]
    vector_db = details["vector_db"]

    # =========================================================================
    # Step 2: Create Sample Documents
    # =========================================================================
    print_step(2, TOTAL_STEPS, "Creating sample documents...")

    created_files = create_sample_documents(quickstart_dir)
    print_success(f"Created {len(created_files)} sample docs in .fitz/quickstart_docs/")

    for f in created_files:
        print_info(f"  • {f.name}")

    # =========================================================================
    # Step 3: Generate Config
    # =========================================================================
    print_step(3, TOTAL_STEPS, "Generating configuration...")

    config = generate_quickstart_config(
        llm_provider,
        embedding_provider,
        vector_db,
        qdrant_host=details.get("qdrant_host", "localhost"),
        qdrant_port=details.get("qdrant_port", "6333"),
    )
    FitzPaths.ensure_workspace()
    config_path.write_text(config)
    print_success("Config saved to .fitz/quickstart_config.yaml")
    print_info(f"  LLM: {llm_provider}")
    print_info(f"  Embedding: {embedding_provider}")
    print_info(f"  Vector DB: {vector_db}")

    # =========================================================================
    # Step 4: Ingest Documents
    # =========================================================================
    print_step(4, TOTAL_STEPS, "Ingesting documents...")

    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Ingesting...", total=None)
            success, num_chunks = run_ingestion(
                quickstart_dir, collection, embedding_provider, vector_db
            )
            progress.update(task, completed=True)
    else:
        success, num_chunks = run_ingestion(
            quickstart_dir, collection, embedding_provider, vector_db
        )

    if not success:
        print_error("Ingestion failed!")
        raise typer.Exit(code=1)

    print_success(f"Ingested {num_chunks} chunks into '{collection}' collection")

    # =========================================================================
    # Step 5: Run Test Query
    # =========================================================================
    if not skip_query:
        print_step(5, TOTAL_STEPS, "Running test query...")

        test_query = "What is RAG and how does it work?"
        print_info(f'Query: "{test_query}"')
        print()

        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Generating answer...", total=None)
                success, answer, sources = run_test_query(collection, config_path)
                progress.update(task, completed=True)
        else:
            success, answer, sources = run_test_query(collection, config_path)

        if not success:
            print_error("Query failed!")
            print_warning("The system is partially working (ingestion OK, query failed)")
            raise typer.Exit(code=1)

        # Display answer
        if RICH_AVAILABLE:
            console.print(
                Panel(
                    answer[:500] + ("..." if len(answer) > 500 else ""),
                    title="[green]Answer[/green]",
                    border_style="green",
                )
            )
            if sources:
                console.print(f"  [dim]Sources: {', '.join(sources)}[/dim]")
        else:
            print()
            print("  Answer:")
            print("  " + "-" * 50)
            print("  " + answer[:500] + ("..." if len(answer) > 500 else ""))
            print("  " + "-" * 50)
            if sources:
                print(f"  Sources: {', '.join(sources)}")

    # =========================================================================
    # Success!
    # =========================================================================
    print()

    if RICH_AVAILABLE:
        success_message = """
[bold green]✅ Fitz is working![/bold green]

You can now:
  • Query the quickstart collection:
    [cyan]fitz-pipeline query "your question" --collection quickstart[/cyan]

  • Ingest your own documents:
    [cyan]fitz-ingest run ./your_docs --collection my_knowledge[/cyan]

  • View your configuration:
    [cyan]fitz-pipeline config show[/cyan]

  • Run diagnostics:
    [cyan]fitz doctor[/cyan]
"""
        console.print(Panel(success_message, border_style="green"))
    else:
        print("=" * 60)
        print("✅ Fitz is working!")
        print("=" * 60)
        print()
        print("You can now:")
        print('  • Query: fitz-pipeline query "your question" --collection quickstart')
        print("  • Ingest: fitz-ingest run ./your_docs --collection my_knowledge")
        print("  • Config: fitz-pipeline config show")
        print("  • Diagnose: fitz doctor")
        print()


if __name__ == "__main__":
    typer.run(command)