"""
Command-line ingestion tool for fitz_ingest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qdrant_client import QdrantClient

from fitz_ingest.chunker.engine import ChunkingEngine
from fitz_ingest.chunker.plugins.simple import SimpleChunker
from fitz_ingest.ingester.engine import IngestionEngine

from fitz_ingest.exceptions.base import IngestionError
from fitz_ingest.exceptions.config import IngestionConfigError
from fitz_ingest.exceptions.vector import IngestionVectorError
from fitz_ingest.exceptions.chunking import IngestionChunkingError


# ============================================================
# Chunker plugin registry
# ============================================================
CHUNKER_REGISTRY = {
    "simple": SimpleChunker,
    # future plugins go here
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="fitz_ingest CLI")

    parser.add_argument("--path", type=str, required=True,
                        help="File or directory to ingest")
    parser.add_argument("--collection", type=str, required=True,
                        help="Qdrant collection")
    parser.add_argument("--vector-size", type=int, required=True,
                        help="Vector dimension (e.g., 1536)")
    parser.add_argument("--host", type=str, default="localhost",
                        help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333,
                        help="Qdrant port")

    parser.add_argument("--chunker", type=str, default="simple",
                        help="Chunker plugin to use")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="Chunk size for simple chunker")

    return parser


# ============================================================
# Helper: ingest folder
# ============================================================
def ingest_directory(root: Path, engine: IngestionEngine) -> None:
    """
    Recursively walk a directory and ingest all files.
    """
    for path in root.rglob("*"):
        if path.is_file():
            engine.ingest_file(path)


# ============================================================
# Main
# ============================================================
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        raise IngestionConfigError(f"Path does not exist: {target}")

    # -------------------------
    # Qdrant connection
    # -------------------------
    try:
        client = QdrantClient(host=args.host, port=args.port)
    except Exception as e:
        raise IngestionVectorError(
            f"Failed to connect to Qdrant at {args.host}:{args.port}"
        ) from e

    # -------------------------
    # Load chunker plugin
    # -------------------------
    plugin_name = args.chunker.lower()
    if plugin_name not in CHUNKER_REGISTRY:
        raise IngestionConfigError(
            f"Unknown chunker plugin '{plugin_name}'. "
            f"Available: {', '.join(CHUNKER_REGISTRY.keys())}"
        )

    PluginClass = CHUNKER_REGISTRY[plugin_name]

    try:
        if plugin_name == "simple":
            plugin = PluginClass(chunk_size=args.chunk_size)
        else:
            plugin = PluginClass()
    except Exception as e:
        raise IngestionConfigError(
            f"Failed to initialize chunker plugin '{plugin_name}': {e}"
        ) from e

    chunker_engine = ChunkingEngine(plugin)

    # -------------------------
    # Build ingestion engine
    # -------------------------
    engine = IngestionEngine(
        client=client,
        collection=args.collection,
        vector_size=args.vector_size,
        chunker_engine=chunker_engine,
        embedder=None,      # Feature not implemented yet
    )

    # -------------------------
    # Run ingestion
    # -------------------------
    try:
        if target.is_file():
            engine.ingest_file(target)
        else:
            ingest_directory(target, engine)

    except (IngestionConfigError,
            IngestionChunkingError,
            IngestionVectorError) as e:
        raise
    except Exception as e:
        raise IngestionError(f"Unexpected error during ingestion: {e}") from e

    print(f"✓ Ingestion complete → collection='{args.collection}'")


if __name__ == "__main__":
    main()
