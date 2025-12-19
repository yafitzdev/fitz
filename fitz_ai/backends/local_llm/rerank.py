# fitz/backends/local_llm/rerank.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import RERANK

from .embedding import LocalEmbedder

logger = get_logger(__name__)


@dataclass(frozen=True)
class LocalRerankerConfig:
    """
    Baseline local reranker.

    Uses cosine similarity between local embeddings of query and document text.
    Deterministic and dependency-free.
    """

    top_k: int = 10


class LocalReranker:
    def __init__(self, embedder: LocalEmbedder, cfg: LocalRerankerConfig | None = None) -> None:
        self._emb = embedder
        self._cfg = cfg or LocalRerankerConfig()

    def rerank(self, query: str, candidates: Sequence[str]) -> List[Tuple[int, float]]:
        """
        Returns list of (index, score) sorted descending.
        """
        logger.info(f"{RERANK} Using local reranking (baseline quality)")

        if not candidates:
            return []

        qv = self._emb.embed_texts([query])[0]
        dvs = self._emb.embed_texts(candidates)

        scored: list[tuple[int, float]] = []
        for i, dv in enumerate(dvs):
            scored.append((i, _cosine(qv, dv)))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[: self._cfg.top_k]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    # Assumes both are normalized (they are).
    s = 0.0
    n = min(len(a), len(b))
    for i in range(n):
        s += a[i] * b[i]
    return float(s)
