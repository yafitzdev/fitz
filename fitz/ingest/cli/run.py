"""
Run command: Ingest documents into vector database.

Usage:
    fitz-ingest run ./documents --collection my_docs
    fitz-ingest run ./documents --collection my_docs --ingest-plugin local
    fitz-ingest run ./documents --collection my_docs --embedding-plugin cohere
"""
from pathlib import Path
from typing import Any, Iterable

import typer

from fitz.core.llm.embedding.engine import EmbeddingEngine
from fitz.core.llm.registry import get_llm_plugin
from fitz.core.logging.logger import get_logger
from fitz.core.logging.tags import CHUNKING, CLI, EMBEDDING, INGEST, VECTOR_DB
from fitz.core.models.chunk import Chunk
from fitz.core.vector_db.registry import get_vector_db_plugin
from fitz.core.vector_db.writer import VectorDBWriter
from fitz.ingest.ingestion.engine import IngestionEngine
from fitz.ingest.ingestion.registry import get_ingest_plugin

logger = get_logger(__name__)


def _raw_to_chunks(raw_docs: Iterable[Any]) -> list[Chunk]:
    """Convert raw documents to canonical chunks."""
    chunks: list[Chunk] = []

    for i, doc in enumerate(raw_docs):
        meta = dict(getattr(doc, "metadata", None) or {})
        path = getattr(doc, "path", None)
        if path:
            meta.setdefault("path", str(path))

        # Be robust across RawDocument variants: some use .content, some .text
        content = getattr(doc, "content", None)
        if content is None:
            content = getattr(doc, "text", None)
        content = content or ""

        chunks.append(
            Chunk(
                id=f"{meta.get('path', 'doc')}:{i}",
                doc_id=str(meta.get("path") or meta.get("doc_id") or "unknown"),
                chunk_index=i,
                content=content,
                metadata=meta,
            )
        )

    return chunks


def command(
        source: Path = typer.Argument(
            ...,
            help="Source to ingest (file or directory, depending on ingest plugin).",
        ),
        collection: str = typer.Option(
            ...,
            "--collection",
            "-c",
            help="Target vector DB collection name.",
        ),
        ingest_plugin: str = typer.Option(
            "local",
            "--ingest-plugin",
            "-i",
            help="Ingestion plugin name (as registered in ingest.ingestion.registry).",
        ),
        embedding_plugin: str = typer.Option(
            "cohere",
            "--embedding-plugin",
            "-e",
            help="Embedding plugin name (registered in core.llm.registry).",
        ),
        vector_db_plugin: str = typer.Option(
            "qdrant",
            "--vector-db-plugin",
            "-v",
            help="Vector DB plugin name (registered in core.vector_db.registry).",
        ),
) -> None:
    """
    Ingest documents into a vector database.

    This command performs the complete ingestion pipeline:
    1. Ingest documents from source (file or directory)
    2. Convert to chunks
    3. Generate embeddings
    4. Store in vector database

    Examples:
        # Ingest local documents with default plugins
        fitz-ingest run ./docs --collection my_knowledge

        # Use specific plugins
        fitz-ingest run ./docs --collection my_docs \\
            --ingest-plugin local \\
            --embedding-plugin openai \\
            --vector-db-plugin qdrant
    """
    # Validate source path
    if not source.exists():
        typer.echo(f"ERROR: source does not exist: {source}")
        raise typer.Exit(code=1)

    if not (source.is_file() or source.is_dir()):
        typer.echo(f"ERROR: source is not file or directory: {source}")
        raise typer.Exit(code=1)

    logger.info(
        f"{CLI}{INGEST} Starting ingestion: source='{source}' → collection='{collection}' "
        f"(ingest='{ingest_plugin}', embedding='{embedding_plugin}', vdb='{vector_db_plugin}')"
    )

    # 1) Ingestion → RawDocument
    typer.echo(f"[1/4] Ingesting documents from {source}...")
    IngestPluginCls = get_ingest_plugin(ingest_plugin)
    ingest_plugin_obj = IngestPluginCls()
    ingest_engine = IngestionEngine(plugin=ingest_plugin_obj, kwargs={})

    raw_docs = list(ingest_engine.run(str(source)))
    logger.info(f"{INGEST} Ingested {len(raw_docs)} raw documents")
    typer.echo(f"  ✓ Ingested {len(raw_docs)} documents")

    # 2) RawDocument → canonical Chunk (1:1 fallback)
    typer.echo("[2/4] Converting to chunks...")
    chunks = _raw_to_chunks(raw_docs)
    logger.info(f"{CHUNKING} Produced {len(chunks)} chunks (1:1 raw→chunk)")
    typer.echo(f"  ✓ Created {len(chunks)} chunks")

    # 3) Embedding
    typer.echo(f"[3/4] Generating embeddings with '{embedding_plugin}'...")
    EmbedPluginCls = get_llm_plugin(plugin_name=embedding_plugin, plugin_type="embedding")
    embed_engine = EmbeddingEngine(EmbedPluginCls())
    vectors = [embed_engine.embed(c.content) for c in chunks]
    logger.info(f"{EMBEDDING} Embedded {len(vectors)} chunks using '{embedding_plugin}'")
    typer.echo(f"  ✓ Generated {len(vectors)} embeddings")

    # 4) Vector DB upsert
    typer.echo(f"[4/4] Writing to vector database '{vector_db_plugin}'...")
    VectorDBPluginCls = get_vector_db_plugin(vector_db_plugin)
    vdb_client = VectorDBPluginCls()

    writer = VectorDBWriter(client=vdb_client)
    writer.upsert(collection=collection, chunks=chunks, vectors=vectors)

    logger.info(f"{VECTOR_DB} Upserted {len(chunks)} chunks into collection='{collection}'")

    typer.echo()
    typer.echo("=" * 60)
    typer.echo("✓ Ingestion complete!")
    typer.echo("=" * 60)
    typer.echo(f"Documents:  {len(raw_docs)}")
    typer.echo(f"Chunks:     {len(chunks)}")
    typer.echo(f"Collection: {collection}")
    typer.echo()