# fitz/cli/quickstart.py
"""
Quickstart command for Fitz CLI.

One command to verify everything works end-to-end:
1. Check system requirements
2. Create sample documents
3. Ingest them
4. Run a test query
5. Show success!

This gives new users confidence that Fitz is working correctly.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import typer

# Rich for pretty output (optional)
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
# Sample Documents (Built-in knowledge base about RAG)
# =============================================================================

SAMPLE_DOCUMENTS = {
    "what_is_rag.txt": """# What is RAG (Retrieval-Augmented Generation)?

RAG (Retrieval-Augmented Generation) is an AI architecture that combines information 
retrieval with language model generation. It was introduced by Facebook AI Research 
in 2020 to address the limitation of LLMs having static, outdated knowledge.

## How RAG Works

1. **Query Processing**: The user's question is converted into a vector embedding
2. **Retrieval**: Similar documents are found using vector similarity search
3. **Context Building**: Retrieved documents are formatted as context
4. **Generation**: An LLM generates an answer grounded in the retrieved context

## Benefits of RAG

- **Reduced Hallucination**: Answers are grounded in actual documents
- **Up-to-date Information**: Knowledge base can be updated without retraining
- **Source Attribution**: Responses can cite their sources
- **Domain Adaptation**: Works with any document collection
""",
    "vector_databases.txt": """# Vector Databases for RAG

Vector databases are specialized databases designed to store and query 
high-dimensional vector embeddings. They are essential for RAG systems.

## Popular Vector Databases

- **Qdrant**: Open-source, Rust-based, excellent filtering support
- **Pinecone**: Managed service, easy to use, good for production
- **Milvus**: Open-source, highly scalable, GPU acceleration
- **FAISS**: Facebook's library, great for local development
- **Weaviate**: GraphQL API, hybrid search capabilities

## Key Concepts

- **Embedding**: A numerical vector representation of text (e.g., 768 or 1536 dimensions)
- **Similarity Search**: Finding vectors closest to a query vector
- **HNSW**: Hierarchical Navigable Small World - efficient approximate search algorithm
- **Cosine Similarity**: Common metric for comparing embedding vectors
""",
    "building_rag_pipeline.txt": """# Building a RAG Pipeline

A RAG pipeline consists of several components working together.

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
# System Check
# =============================================================================


def check_system() -> tuple[bool, dict]:
    """
    Check system requirements and available providers.

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

    # Check for LLM providers (in order of preference for quickstart)
    if os.getenv("COHERE_API_KEY"):
        details["llm_provider"] = "cohere"
        details["embedding_provider"] = "cohere"
        print_success("Cohere API key found")
    elif os.getenv("OPENAI_API_KEY"):
        details["llm_provider"] = "openai"
        details["embedding_provider"] = "openai"
        print_success("OpenAI API key found")
    elif os.getenv("ANTHROPIC_API_KEY"):
        details["llm_provider"] = "anthropic"
        # Anthropic doesn't have embeddings, need alternative
        if os.getenv("COHERE_API_KEY"):
            details["embedding_provider"] = "cohere"
        elif os.getenv("OPENAI_API_KEY"):
            details["embedding_provider"] = "openai"
        else:
            details["embedding_provider"] = "local"
        print_success("Anthropic API key found")
    else:
        # Check Ollama for local
        try:
            import httpx

            response = httpx.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                details["llm_provider"] = "local"
                details["embedding_provider"] = "local"
                print_success("Ollama is running (local)")
        except Exception:
            pass

    if not details["llm_provider"]:
        details["issues"].append(
            "No LLM provider found. Set COHERE_API_KEY, OPENAI_API_KEY, "
            "or install Ollama for local use."
        )

    if not details["embedding_provider"]:
        details["issues"].append("No embedding provider found")

    # Check for vector database (prefer Qdrant, fall back to FAISS)
    qdrant_available = False
    try:
        import httpx

        host = os.getenv("QDRANT_HOST", "localhost")
        port = os.getenv("QDRANT_PORT", "6333")
        response = httpx.get(f"http://{host}:{port}/collections", timeout=2)
        if response.status_code == 200:
            details["vector_db"] = "qdrant"
            details["qdrant_host"] = host
            details["qdrant_port"] = port
            qdrant_available = True
            print_success(f"Qdrant available at {host}:{port}")
    except Exception:
        pass

    if not qdrant_available:
        try:
            import faiss

            details["vector_db"] = "local-faiss"
            print_success("FAISS available (local vector DB)")
        except ImportError:
            pass

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
# Config Generation
# =============================================================================


def generate_quickstart_config(
    llm_provider: str,
    embedding_provider: str,
    vector_db: str,
    qdrant_host: str = "localhost",
    qdrant_port: str = "6333",
) -> str:
    """
    Generate a minimal config for quickstart.

    Note: FAISS no longer needs dim - it auto-detects on first upsert.
    """

    # LLM configs
    llm_configs = {
        "cohere": """llm:
  plugin_name: cohere
  kwargs:
    model: command-r-08-2024
    temperature: 0.2""",
        "openai": """llm:
  plugin_name: openai
  kwargs:
    model: gpt-4o-mini
    temperature: 0.2""",
        "anthropic": """llm:
  plugin_name: anthropic
  kwargs:
    model: claude-sonnet-4-20250514
    temperature: 0.2""",
        "local": """llm:
  plugin_name: local
  kwargs:
    model: llama3.2:1b""",
    }

    # Embedding configs
    embedding_configs = {
        "cohere": """embedding:
  plugin_name: cohere
  kwargs:
    model: embed-english-v3.0""",
        "openai": """embedding:
  plugin_name: openai
  kwargs:
    model: text-embedding-3-small""",
        "local": """embedding:
  plugin_name: local
  kwargs:
    model: nomic-embed-text""",
    }

    # Vector DB configs
    # Note: FAISS uses FitzPaths.vector_db() by default, no need to specify path
    vector_db_configs = {
        "qdrant": f"""vector_db:
  plugin_name: qdrant
  kwargs:
    host: "{qdrant_host}"
    port: {qdrant_port}""",
        "local-faiss": """vector_db:
  plugin_name: local-faiss
  kwargs: {}""",
    }

    config = f"""# Fitz Quickstart Configuration
# Auto-generated by: fitz quickstart

{llm_configs.get(llm_provider, llm_configs["cohere"])}

{embedding_configs.get(embedding_provider, embedding_configs["cohere"])}

{vector_db_configs.get(vector_db, vector_db_configs["local-faiss"])}

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
        from fitz.engines.classic_rag.models.chunk import Chunk
        from fitz.ingest.ingestion.engine import IngestionEngine
        from fitz.ingest.ingestion.registry import get_ingest_plugin
        from fitz.llm.embedding.engine import EmbeddingEngine
        from fitz.llm.registry import get_llm_plugin
        from fitz.vector_db.registry import get_vector_db_plugin
        from fitz.vector_db.writer import VectorDBWriter

        # Step 1: Ingest documents
        IngestPluginCls = get_ingest_plugin("local")
        ingest_plugin = IngestPluginCls()
        ingest_engine = IngestionEngine(plugin=ingest_plugin, kwargs={})
        raw_docs = list(ingest_engine.run(str(docs_path)))

        # Step 2: Convert to chunks
        chunks = []
        for i, doc in enumerate(raw_docs):
            content = getattr(doc, "content", "") or getattr(doc, "text", "") or ""
            path = getattr(doc, "path", f"doc_{i}")

            # Simple paragraph-based chunking
            paragraphs = [
                p.strip() for p in content.split("\n\n") if p.strip() and len(p.strip()) > 50
            ]

            for j, para in enumerate(paragraphs):
                chunks.append(
                    Chunk(
                        id=f"{Path(path).stem}:{j}",
                        doc_id=str(path),
                        chunk_index=j,
                        content=para,
                        metadata={"source": str(path)},
                    )
                )

        if not chunks:
            return False, 0

        # Step 3: Generate embeddings
        EmbedPluginCls = get_llm_plugin(plugin_name=embedding_plugin, plugin_type="embedding")
        embed_engine = EmbeddingEngine(EmbedPluginCls())

        vectors = []
        for chunk in chunks:
            vec = embed_engine.embed(chunk.content)
            vectors.append(vec)

        # Step 4: Store in vector DB
        # All vector DBs now have the same interface - no special handling needed!
        VectorDBPluginCls = get_vector_db_plugin(vector_db_plugin)
        vdb_client = VectorDBPluginCls()

        writer = VectorDBWriter(client=vdb_client)
        writer.upsert(collection=collection, chunks=chunks, vectors=vectors)

        return True, len(chunks)

    except Exception as e:
        print_error(f"Ingestion failed: {e}")
        import traceback

        traceback.print_exc()
        return False, 0


# =============================================================================
# Query Test
# =============================================================================


def run_test_query(collection: str, config_path: Path) -> tuple[bool, str, list]:
    """
    Run a test query.

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
