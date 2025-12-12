"""
Command-line ingestion tool for fitz_ingest.

High-level flow:

    Ingester → (optional) Chunker → Embedding → Vector DB

- Ingester turns your raw source (files, etc.) into RawDocument objects
- Chunker turns RawDocument → Chunk
- Embedding + VectorDBWriter turn Chunk → vector records in your collection
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import typer

from ingest.ingester.engine import Ingester
from ingest.ingester.base import RawDocument

from rag.core import Chunk

from core.logging.logger import get_logger
from core.logging.tags import CLI, INGEST, CHUNKING, VECTOR_DB, EMBEDDING

from core.llm.registry import get_llm_plugin
from core.llm.embedding.engine import EmbeddingEngine

from core.vector_db.registry import get_vector_db_plugin
from core.vector_db.writer import VectorDBWriter

app = typer.Typer(help="Ingestion CLI for fitz-ingest")
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raw_to_chunks(raw_docs: Iterable[RawDocument]) -> List[Chunk]:
    """
    Very simple 1:1 RawDocument → Chunk fallback.

    If you want full-blown chunking, you can plug in ChunkingEngine here later.
    For now we keep this CLI minimal and rely on the ingestion plugins
    to provide good RawDocuments.
    """
    chunks: List[Chunk] = []
    for doc in raw_docs:
        meta = dict(getattr(doc, "metadata", None) or {})
        # Common provenance fields (if present on RawDocument)
        source = getattr(doc, "source", None)
        path = getattr(doc, "path", None)

        if source is not None:
            meta.setdefault("source", source)
        if path is not None:
            meta.setdefault("path", path)

        chunks.append(
            Chunk(
                id=None,
                text=doc.text,
                metadata=meta,
                score=None,
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("run")
def run(
    source: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
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
        help="Ingest plugin name (as registered in fitz_ingest.ingester.registry).",
    ),
    embedding_plugin: str = typer.Option(
        "cohere",
        "--embedding-plugin",
        "-e",
        help="LLM embedding plugin name (registered in fitz_stack.llm.registry).",
    ),
    vector_db_plugin: str = typer.Option(
        "qdrant",
        "--vector-db-plugin",
        "-v",
        help="Vector DB plugin name (registered in fitz_stack.vector_db.registry).",
    ),
) -> None:
    """
    Ingest → (simple) chunk → embed → write into a vector DB collection.

    All heavy lifting is delegated to:
    - Ingester
    - EmbeddingEngine
    - VectorDBWriter
    - Vector DB plugin (e.g. Qdrant)
    """
    logger.info(
        f"{CLI}{INGEST} Starting ingestion: source='{source}' → collection='{collection}' "
        f"(ingest_plugin='{ingest_plugin}', embedding='{embedding_plugin}', vdb='{vector_db_plugin}')"
    )

    # ------------------------------------------------------------------
    # 1) Ingest → RawDocument
    # ------------------------------------------------------------------
    ingester = Ingester(plugin_name=ingest_plugin, config={})
    raw_docs = list(ingester.run(str(source)))
    logger.info(f"{INGEST} Ingested {len(raw_docs)} raw documents")

    # ------------------------------------------------------------------
    # 2) Simple RawDocument → Chunk conversion
    #    (slot where you can plug ChunkingEngine later)
    # ------------------------------------------------------------------
    chunks = _raw_to_chunks(raw_docs)
    logger.info(f"{CHUNKING} Produced {len(chunks)} chunks (1:1 raw→chunk)")

    # ------------------------------------------------------------------
    # 3) EmbeddingEngine from LLM registry
    # ------------------------------------------------------------------
    EmbedPluginCls = get_llm_plugin(embedding_plugin, plugin_type="embedding")
    # CohereEmbeddingClient (and other plugins) handle API keys / models via env
    embed_plugin = EmbedPluginCls()
    embed_engine = EmbeddingEngine(embed_plugin)
    logger.info(f"{EMBEDDING} Using embedding plugin='{embedding_plugin}'")

    # ------------------------------------------------------------------
    # 4) Vector DB plugin + writer
    # ------------------------------------------------------------------
    VectorDBPluginCls = get_vector_db_plugin(vector_db_plugin)
    vectordb = VectorDBPluginCls()
    writer = VectorDBWriter(embedder=embed_engine, vectordb=vectordb)

    written = writer.write(collection=collection, chunks=chunks)
    logger.info(f"{VECTOR_DB} Wrote {written} chunks into collection='{collection}'")

    typer.echo(
        f"Ingested {len(raw_docs)} documents → wrote {written} chunks into collection '{collection}'."
    )


if __name__ == "__main__":
    app()
