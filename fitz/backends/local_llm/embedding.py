# fitz/backends/local_llm/embedding.py
from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Iterable, List

from fitz.core.logging.logger import get_logger
from fitz.core.logging.tags import EMBEDDING

logger = get_logger(__name__)


@dataclass(frozen=True)
class LocalEmbedderConfig:
    """
    Deterministic local embedding fallback.

    This is intentionally “shitty but servicable”:
      - no external deps
      - stable across machines
      - good enough to verify vector_db + retrieval wiring

    Dimension should be kept small-ish to avoid inflating storage.
    """

    dim: int = 384
    seed: int = 0


class LocalEmbedder:
    """
    Deterministic hash-embedding backend.

    This is *not* semantic. It's a stable baseline for pipeline bootstrapping.
    """

    def __init__(self, cfg: LocalEmbedderConfig | None = None) -> None:
        self._cfg = cfg or LocalEmbedderConfig()

    @property
    def dim(self) -> int:
        return self._cfg.dim

    def embed_texts(self, texts: Iterable[str]) -> List[list[float]]:
        logger.info(f"{EMBEDDING} Using local embeddings (baseline quality)")
        return [_hash_embed(t or "", dim=self._cfg.dim, seed=self._cfg.seed) for t in texts]


def _hash_embed(text: str, *, dim: int, seed: int) -> list[float]:
    # Deterministic: use blake2b over (seed + text) to generate enough bytes.
    # Then map bytes to floats in [-1, 1], and L2-normalize.
    msg = f"{seed}\n{text}".encode("utf-8", errors="ignore")

    # Need dim floats; generate 2 bytes each -> dim*2 bytes minimum
    out = bytearray()
    ctr = 0
    while len(out) < dim * 2:
        h = blake2b(msg + ctr.to_bytes(4, "little"), digest_size=32)
        out.extend(h.digest())
        ctr += 1

    vec: list[float] = []
    for i in range(dim):
        b0 = out[2 * i]
        b1 = out[2 * i + 1]
        u16 = (b0 << 8) | b1  # 0..65535
        x = (u16 / 32767.5) - 1.0  # ~[-1,1]
        vec.append(x)

    # L2 normalize
    norm = 0.0
    for x in vec:
        norm += x * x
    norm = norm ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec
