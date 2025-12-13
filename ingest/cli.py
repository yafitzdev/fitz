# ingest/cli.py
"""
Command-line ingestion tool for fitz-ingest.

High-level flow:
    Ingestion → (optional) Chunking → Embedding → Vector DB

This CLI stays strictly on *core* contracts:
- IngestionEngine produces RawDocument
- We convert RawDocument -> core.models.chunk.Chunk (canonical)
- EmbeddingPlugin.embed(text)->list[float]
- VectorDBWriter.upsert(collection, chunks, vectors) into a VectorDB client that exposes upsert(collection, points)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any

import typer

from core.logging.logger import get_logger
from core.logging.tags import CLI, INGEST, CHUNKING, VECTOR_DB, EMBEDDING

from core.llm.embedding.engine import EmbeddingEngine
from core.llm.registry import get_llm_plugin

from core.models.chunk import Chunk
from core.vector_db.registry import get_vector_db_plugin
from core.vector_db.writer import VectorDBWriter

from ingest.ingestion.engine import IngestionEngine
from ingest.ingestion.registry import get_ingest_plugin

app = typer.Typer(help="Ingestion CLI for fitz-ingest")
logger = get_logger(__name__)


def _raw_to_chunks(raw_docs: Iterable[Any]) -> list[Chunk]:
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


@app.command("run")
def run(
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
    # Manual validation (so pytest CliRunner doesn't depend on Typer Path validation modes)
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
    IngestPluginCls = get_ingest_plugin(ingest_plugin)
    ingest_plugin_obj = IngestPluginCls()
    ingest_engine = IngestionEngine(plugin=ingest_plugin_obj, kwargs={})

    raw_docs = list(ingest_engine.run(str(source)))
    logger.info(f"{INGEST} Ingested {len(raw_docs)} raw documents")

    # 2) RawDocument → canonical Chunk (1:1 fallback)
    chunks = _raw_to_chunks(raw_docs)
    logger.info(f"{CHUNKING} Produced {len(chunks)} chunks (1:1 raw→chunk)")

    # 3) Embedding
    EmbedPluginCls = get_llm_plugin(plugin_name=embedding_plugin, plugin_type="embedding")
    embed_engine = EmbeddingEngine(EmbedPluginCls())
    vectors = [embed_engine.embed(c.content) for c in chunks]
    logger.info(f"{EMBEDDING} Embedded {len(vectors)} chunks using '{embedding_plugin}'")

    # 4) Vector DB upsert
    VectorDBPluginCls = get_vector_db_plugin(vector_db_plugin)
    vdb_client = VectorDBPluginCls()

    writer = VectorDBWriter(client=vdb_client)
    writer.upsert(collection=collection, chunks=chunks, vectors=vectors)

    logger.info(f"{VECTOR_DB} Upserted {len(chunks)} chunks into collection='{collection}'")
    typer.echo(
        f"OK: ingested {len(raw_docs)} documents → upserted {len(chunks)} chunks into '{collection}'."
    )


if __name__ == "__main__":
    app()
