"""Internal subprocess worker for hard-crash ingestion recovery tests."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from fitz_sage.core.paths import FitzPaths
from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
from fitz_sage.runtime import create_engine
from fitz_sage.storage.sqlite import SqliteConnectionManager

CRASH_EXIT_CODE = 86
_INDEXED_PROGRESS_RE = re.compile(r"^Indexed \d+/\d+:")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    FitzPaths.set_workspace(args.workspace.resolve())
    SqliteConnectionManager.reset_instance()
    indexed = 0

    def progress(message: str) -> None:
        nonlocal indexed
        if not _INDEXED_PROGRESS_RE.match(message):
            return
        indexed += 1
        if args.crash_after_indexed and indexed >= args.crash_after_indexed:
            os._exit(CRASH_EXIT_CODE)

    try:
        engine = create_engine(
            "fitz_krag",
            config=FitzKragConfig(collection=args.collection, parser=args.parser),
        )
        engine.load(args.collection)
        engine.point(
            args.source.resolve(),
            collection=args.collection,
            start_worker=False,
            progress=progress,
        )
        status = dict(engine.indexing_status())
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        return 0 if status.get("query_ready") else 1
    finally:
        SqliteConnectionManager.reset_instance()
        FitzPaths.reset()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument(
        "--parser",
        choices=("cpu", "docling", "docling_vision", "glm_ocr"),
        default="cpu",
    )
    parser.add_argument("--crash-after-indexed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
