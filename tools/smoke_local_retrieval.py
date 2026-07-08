"""Smoke-test the local CPU retrieval stack.

This is an integration smoke, not a quality benchmark. It verifies that the
standard local path can initialize and execute:

- managed Qwen ONNX GenAI enrichment and query keywords
- ONNX reranking
- ONNX Pyrrho governance

Run from the repository root after installing fitz-sage dependencies:

    python tools/smoke_local_retrieval.py
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

from fitz_sage import Query
from fitz_sage.engines.fitz_krag.config import FitzKragConfig
from fitz_sage.engines.fitz_krag.engine import FitzKragEngine
from fitz_sage.llm.providers.onnx_chat import OnnxChat


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local CPU retrieval smoke.")
    parser.add_argument(
        "--docs",
        type=Path,
        default=None,
        help="Existing document directory. Defaults to a generated temporary corpus.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Collection name. Defaults to a unique smoke collection.",
    )
    parser.add_argument(
        "--governance",
        default="pyrrho",
        help="Governance spec. Defaults to the managed Pyrrho package.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the generated temporary corpus on disk.",
    )
    args = parser.parse_args()

    temp_root: Path | None = None
    docs = args.docs
    if docs is None:
        temp_root = Path(tempfile.mkdtemp(prefix="fitz-local-smoke-"))
        docs = temp_root / "docs"
        _write_smoke_docs(docs)

    collection = args.collection or f"local_cpu_smoke_{int(time.time())}"
    config = FitzKragConfig(
        collection=collection,
        governance=args.governance,
        synthesizer=None,
        query_intelligence=None,
    )

    try:
        qwen_info = OnnxChat().model_info()
        print(
            "managed_qwen=" f"{qwen_info.repo_id} {qwen_info.onnx_file} {qwen_info.revision[:12]}"
        )

        progress: list[str] = []
        engine = FitzKragEngine(config)
        engine.point(docs, progress=progress.append, start_worker=False)

        t0 = time.perf_counter()
        engine.continue_indexing()
        print(f"index_seconds={time.perf_counter() - t0:.2f}")

        for message in progress:
            if "Qwen" in message or "Managed Qwen" in message:
                print(f"progress={message}")

        for name, text in _smoke_queries().items():
            pack = engine.evidence(Query(text=text), top_k=4)
            files = [item.file_path for item in pack.items]
            print(f"{name}={pack.mode.value};items={len(pack.items)};files={files}")

        return 0
    finally:
        if temp_root is not None and not args.keep_temp:
            shutil.rmtree(temp_root, ignore_errors=True)
        elif temp_root is not None:
            print(f"temp_corpus={temp_root}")


def _write_smoke_docs(docs: Path) -> None:
    """Create a tiny corpus with enough contrast for retrieval smoke checks."""
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "refund_policy.md").write_text(
        "\n".join(
            [
                "# Refund Policy",
                "Customers may request a refund within 30 days of purchase.",
                "Refunds after 30 days require manager approval.",
            ]
        ),
        encoding="utf-8",
    )
    (docs / "legacy_refund_note.md").write_text(
        "\n".join(
            [
                "# Legacy Refund Note",
                "The legacy 2021 handbook allowed refunds within 14 days only.",
                "This note is obsolete and retained for audit history.",
            ]
        ),
        encoding="utf-8",
    )
    (docs / "ops_handbook.md").write_text(
        "\n".join(
            [
                "# Operations Handbook",
                "Escalated refund requests go to the support manager.",
                "Shipping incidents use ticket code OPS-17.",
            ]
        ),
        encoding="utf-8",
    )


def _smoke_queries() -> dict[str, str]:
    """Return stable queries that exercise sufficient, conflict, and gap surfaces."""
    return {
        "refund_window": "What is the current refund window?",
        "refund_conflict": "Is the refund window 14 days or 30 days?",
        "missing_policy": "What is the vacation carryover policy?",
    }


if __name__ == "__main__":
    raise SystemExit(main())
