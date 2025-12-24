# fitz_ai/engines/classic_rag/pipeline/context/pipeline.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fitz_ai.engines.classic_rag.exceptions import PipelineError
from fitz_ai.engines.classic_rag.models.chunk import Chunk

from .steps.dedupe import DedupeStep
from .steps.group import GroupByDocumentStep
from .steps.merge import MergeAdjacentStep
from .steps.normalize import ChunkDict, NormalizeStep
from .steps.pack import PackWindowStep


def _chunkdict_to_chunk(d: ChunkDict) -> Chunk:
    return Chunk(
        id=str(d["id"]),
        doc_id=str(d["doc_id"]),
        chunk_index=int(d["chunk_index"]),
        content=str(d["content"]),
        metadata=dict(d.get("metadata") or {}),
    )


def _is_vip(ch: ChunkDict) -> bool:
    """Check if chunk has VIP status (score=1.0, bypasses packing limits)."""
    meta = ch.get("metadata", {})
    return meta.get("rerank_score") == 1.0 or meta.get("score") == 1.0


@dataclass
class ContextPipeline:
    """
    Step-based chunk pre-processing pipeline.

    Steps:
        normalize → dedupe → group → merge → pack

    VIP chunks (score=1.0) are always included and excluded from packing.
    Only regular chunks are subject to max_chars packing.

    Output:
        list[Chunk]
    """

    max_chars: int = 8000  # Budget for regular chunks only (VIP excluded)

    normalize_step: NormalizeStep = field(default_factory=NormalizeStep)
    dedupe_step: DedupeStep = field(default_factory=DedupeStep)
    group_step: GroupByDocumentStep = field(default_factory=GroupByDocumentStep)
    merge_step: MergeAdjacentStep = field(default_factory=MergeAdjacentStep)
    pack_step: PackWindowStep = field(default_factory=PackWindowStep)

    def process(self, chunks: list[Any], max_chars: int | None = None) -> list[Chunk]:
        if max_chars is None:
            max_chars = self.max_chars

        try:
            norm = self.normalize_step(chunks)

            # Separate VIP from regular chunks
            # VIP chunks are always included (score=1.0 means "always include")
            vip: list[ChunkDict] = []
            regular: list[ChunkDict] = []
            for ch in norm:
                if _is_vip(ch):
                    vip.append(ch)
                else:
                    regular.append(ch)

            # Process regular chunks through dedupe → group → merge → pack
            deduped = self.dedupe_step(regular)
            grouped: dict[str, list[ChunkDict]] = self.group_step(deduped)

            merged_per_doc: list[ChunkDict] = []
            for _, doc_chunks in grouped.items():
                merged_per_doc.extend(self.merge_step(doc_chunks))

            packed = self.pack_step(merged_per_doc, max_chars=max_chars)

            # Combine: VIP first, then packed regular chunks
            result = vip + packed
            return [_chunkdict_to_chunk(d) for d in result]

        except Exception as exc:
            raise PipelineError(f"Failed context pipeline: {exc}") from exc
