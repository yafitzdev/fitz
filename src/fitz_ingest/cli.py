from __future__ import annotations

"""
fitz_ingest CLI

Pure ingestion + chunking + output.

Steps:
1. Load ingestion plugin (default: local filesystem)
2. Read documents (RawDocument)
3. Chunk them via ChunkingEngine
4. Print or write chunks as JSONL

NO vector DBs
NO embeddings
NO indexing
"""

import argparse
import json
from pathlib import Path

from fitz_ingest.ingester.engine import Ingester
from fitz_ingest.chunker.engine import ChunkingEngine
from fitz_ingest.chunker.plugins.simple import SimpleChunker

from fitz_ingest.exceptions.base import IngestionError
from fitz_ingest.exceptions.config import IngestionConfigError
from fitz_ingest.exceptions.chunking import IngestionChunkingError


# -------------------------------------------------------------
# Chunker registry
# -------------------------------------------------------------
CHUNKER_REGISTRY = {
    "simple": SimpleChunker,
    # future chunkers can be added here
}


# -------------------------------------------------------------
# Argument parser
# -------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="fitz_ingest CLI")

    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="File or directory to ingest",
    )

    parser.add_argument(
        "--chunker",
        type=str,
        default="simple",
        help="Chunker plugin name",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size for 'simple' chunker",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL file. If omitted, output prints to stdout.",
    )

    # Ingestion plugin selection (future-proof)
    parser.add_argument(
        "--ingest-plugin",
        type=str,
        default="local",
        help="Ingestion plugin to use (default: local filesystem)",
    )

    return parser


# -------------------------------------------------------------
# Directory ingestion helper
# -------------------------------------------------------------
def ingest_directory(root: Path, ingester: Ingester):
    """
    ingester.run(path) returns an iterable of RawDocuments.
    """
    for path in root.rglob("*"):
        if path.is_file():
            for doc in ingester.run(str(path)):
                yield doc


# -------------------------------------------------------------
# Main CLI
# -------------------------------------------------------------
def main():
    parser = build_parser()
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        raise IngestionConfigError(f"Path does not exist: {target}")

    # -------------------------
    # Load ingestion plugin
    # -------------------------
    ingester = Ingester(plugin_name=args.ingest_plugin, config={})

    # -------------------------
    # Load chunker plugin
    # -------------------------
    plugin_name = args.chunker.lower()
    if plugin_name not in CHUNKER_REGISTRY:
        raise IngestionConfigError(
            f"Unknown chunker plugin '{plugin_name}'. "
            f"Available: {', '.join(CHUNKER_REGISTRY.keys())}"
        )

    ChunkerClass = CHUNKER_REGISTRY[plugin_name]

    try:
        if plugin_name == "simple":
            chunker_plugin = ChunkerClass(chunk_size=args.chunk_size)
        else:
            chunker_plugin = ChunkerClass()
    except Exception as e:
        raise IngestionConfigError(f"Failed to initialize chunker: {e}") from e

    chunker_engine = ChunkingEngine(chunker_plugin)

    # -------------------------
    # Run ingestion
    # -------------------------
    try:
        if target.is_file():
            docs = list(ingester.run(str(target)))
        else:
            docs = list(ingest_directory(target, ingester))
    except (IngestionConfigError, IngestionChunkingError) as e:
        raise
    except Exception as e:
        raise IngestionError(f"Unexpected error during ingestion: {e}") from e

    # -------------------------
    # Chunking step
    # -------------------------
    chunks = []
    for doc in docs:
        try:
            chunks.extend(chunker_engine.run(doc))
        except Exception as e:
            raise IngestionChunkingError(f"Chunking failed for {doc.path}: {e}") from e

    # -------------------------
    # Output
    # -------------------------
    if args.output:
        out_path = Path(args.output)
        with out_path.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c) + "\n")
        print(f"✓ Wrote {len(chunks)} chunks → {args.output}")
    else:
        for c in chunks:
            print(json.dumps(c))


if __name__ == "__main__":
    main()
