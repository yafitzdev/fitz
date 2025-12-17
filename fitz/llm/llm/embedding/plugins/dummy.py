from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class DummyEmbeddingClient:
    """
    Deterministic embedding plugin for tests.
    """

    dim: int = 10

    def embed(self, text: str) -> List[float]:
        # Deterministic based on hash(text)
        base = abs(hash(text)) % 997
        return [(base + i) / 997.0 for i in range(self.dim)]
