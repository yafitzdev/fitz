"""
Command-line ingestion tool for fitz-ingest.

High-level flow:

    Ingestion → (optional) Chunking → Embedding → Vector DB

- Ingester turns your raw source (files, etc.) into RawDocument objects
- Chunking (here: trivial 1:1 fallback) turns RawDocument → Chunk
- Embedding + VectorDBWriter turn Chunk → vector records in your collection
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any

import typer

from ingest.ingestion.engine import Ingester
from ingest.ingestion.base import RawDocument

from core.logging.logger import get_logger
from core.logging.tags import CLI, INGEST, CHUNKING, VECTOR_DB, EMBEDDING

from core.llm.registry import get_llm_plugin
from core.llm.embedding.engine import EmbeddingEngine

from core.vector_db.registry import get_vector_db_plugin
from core.vector_db.writer import VectorDBWriter

app = typer.Typer(help="Ingestion CLI for fitz-ingest")
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal DTOs
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """
    Minimal chunk DTO for ingestion CLI.

    This intentionally does NOT depend on RAG / retrieval modules.
    """
    id: Optional[str]
    text: str
    metadata: Dict[str, Any]
    score: Optional[float] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raw_to_chunks(raw_docs: Iterable[RawDocument]) -> List[Chunk]:
    """
    Very simple 1:1 RawDocument → Chunk fallback.

    This keeps the CLI lightweight and avoids pulling in ChunkingEngine
    or retrieval-layer abstractions.
    """
    chunks: List[Chunk] = []

    for doc in raw_docs:
        meta = dict(getattr(doc, "metadata", None) or {})

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
    Ingest → simple chunk → embed → write into a vector DB collection.
    """
    logger.info(
        f"{CLI}{INGEST} Starting ingestion: source='{source}' → collection='{collection}' "
        f"(ingest='{ingest_plugin}', embedding='{embedding_plugin}', vdb='{vector_db_plugin}')"
    )

    # ------------------------------------------------------------------
    # 1) Ingestion → RawDocument
    # ------------------------------------------------------------------
    ingester = Ingester(plugin_name=ingest_plugin, config={})
    raw_docs = list(ingester.run(str(source)))
    logger.info(f"{INGEST} Ingested {len(raw_docs)} raw documents")

    # ------------------------------------------------------------------
    # 2) RawDocument → Chunk (1:1 fallback)
    # ------------------------------------------------------------------
    chunks = _raw_to_chunks(raw_docs)
    logger.info(f"{CHUNKING} Produced {len(chunks)} chunks (1:1 raw→chunk)")

    # ------------------------------------------------------------------
    # 3) Embedding engine
    # ------------------------------------------------------------------
    EmbedPluginCls = get_llm_plugin(embedding_plugin, plugin_type="embedding")
    embed_plugin = EmbedPluginCls()
    embed_engine = EmbeddingEngine(embed_plugin)
    logger.info(f"{EMBEDDING} Using embedding plugin='{embedding_plugin}'")

    # ------------------------------------------------------------------
    # 4) Vector DB writer
    # ------------------------------------------------------------------
    VectorDBPluginCls = get_vector_db_plugin(vector_db_plugin)
    vectordb = VectorDBPluginCls()

    writer = VectorDBWriter(
        embedder=embed_engine,
        vectordb=vectordb,
    )

    written = writer.write(collection=collection, chunks=chunks)
    logger.info(f"{VECTOR_DB} Wrote {written} chunks into collection='{collection}'")

    typer.echo(
        f"Ingested {len(raw_docs)} documents → wrote {written} chunks into collection '{collection}'."
    )


if __name__ == "__main__":
    app()
