from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fitz.core.models.chunk import Chunk
from fitz.core.vector_db.writer import VectorDBWriter
from fitz.pipeline.context.pipeline import ContextPipeline
from fitz.pipeline.generation.rgs import RGS, RGSConfig
from fitz.retrieval.plugins.dense import DenseRetrievalPlugin, RetrieverCfg


def read_text_files(root: Path) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".txt", ".md"}:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        doc_id = str(p.relative_to(root)).replace("\\", "/")
        docs.append((doc_id, text))
    return docs


def naive_chunks(docs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for doc_id, text in docs:
        parts = [t.strip() for t in text.split("\n\n") if t.strip()]
        for i, part in enumerate(parts):
            out.append(
                {
                    "id": f"{doc_id}:{i}",
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "content": part,
                    "metadata": {"file": doc_id},
                }
            )
    return out


def embed_hash(text: str, dim: int = 48) -> list[float]:
    # Deterministic toy embedding (good enough for a local smoke test)
    vec = [0.0] * dim
    for i, ch in enumerate(text.encode("utf-8", errors="ignore")):
        vec[i % dim] += (ch % 13) - 6
    # normalize
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class Hit:
    id: str
    payload: dict[str, Any]
    score: float


class InMemoryVectorDB:
    def __init__(self) -> None:
        self._points: list[dict[str, Any]] = []

    def upsert(self, collection: str, points: list[dict[str, Any]]) -> None:
        # replace by id
        by_id = {p["id"]: p for p in self._points}
        for p in points:
            by_id[p["id"]] = p
        self._points = list(by_id.values())

    def search(
        self, collection_name: str, query_vector: list[float], limit: int, with_payload: bool = True
    ):
        scored: list[Hit] = []
        for p in self._points:
            score = cosine(query_vector, p["vector"])
            scored.append(Hit(id=str(p["id"]), payload=dict(p["payload"]), score=score))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:limit]


class DummyEmbedder:
    def embed(self, text: str) -> list[float]:
        return embed_hash(text)


def main() -> int:
    root = Path(r"C:\Users\yanfi\Downloads\test_data")
    if not root.exists():
        raise SystemExit(f"Folder not found: {root}")

    docs = read_text_files(root)
    print(f"[smoke] docs: {len(docs)}")
    if not docs:
        print("[smoke] No .txt/.md found. Add a .txt test file or extend read_text_files().")
        return 0

    raw = naive_chunks(docs)
    print(f"[smoke] raw chunks: {len(raw)}")

    ctx_pipe = ContextPipeline(max_chars=1200)
    processed: list[Chunk] = ctx_pipe.process(raw)
    print(f"[smoke] processed chunks: {len(processed)}")
    for c in processed[:3]:
        print(f"  - {c.id} doc={c.doc_id} len={len(c.content)}")

    # write to vector db
    vectordb = InMemoryVectorDB()
    writer = VectorDBWriter(client=vectordb)
    vectors = [embed_hash(c.content) for c in processed]
    writer.upsert(collection="local", chunks=processed, vectors=vectors)
    print(f"[smoke] upserted: {len(processed)}")

    # retrieve
    retr = DenseRetrievalPlugin(
        client=vectordb,
        retriever_cfg=RetrieverCfg(collection="local", top_k=5),
        embedder=DummyEmbedder(),
        rerank_engine=None,
    )

    query = "What is this folder about?"
    hits = retr.retrieve(query)
    print(f"[smoke] retrieved: {len(hits)}")
    for h in hits:
        print(f"  - score={h.metadata.get('score'):.4f} doc={h.doc_id} id={h.id}")

    # build RGS prompt
    rgs = RGS(RGSConfig(enable_citations=True, max_chunks=5))
    prompt = rgs.build_prompt(query, hits)
    print("\n=== SYSTEM PROMPT ===\n")
    print(prompt.system)
    print("\n=== USER PROMPT ===\n")
    print(prompt.user)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
