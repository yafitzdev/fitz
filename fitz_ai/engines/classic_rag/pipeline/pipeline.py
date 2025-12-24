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


@dataclass
class ContextPipeline:
    """
    Step-based chunk pre-processing pipeline.

    Steps:
        normalize → dedupe → group → merge → pack

    Artifacts (is_artifact=True) are always included and excluded from packing.
    Only regular chunks are subject to max_chars packing.

    Output:
        list[Chunk]
    """

    max_chars: int = 8000  # Budget for regular chunks only (artifacts excluded)

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

            # Separate artifacts from regular chunks
            # Artifacts are always included (they have score=1.0 for a reason)
            artifacts: list[ChunkDict] = []
            regular: list[ChunkDict] = []
            for ch in norm:
                if ch.get("metadata", {}).get("is_artifact"):
                    artifacts.append(ch)
                else:
                    regular.append(ch)

            # Process regular chunks through dedupe → group → merge → pack
            deduped = self.dedupe_step(regular)
            grouped: dict[str, list[ChunkDict]] = self.group_step(deduped)

            merged_per_doc: list[ChunkDict] = []
            for _, doc_chunks in grouped.items():
                merged_per_doc.extend(self.merge_step(doc_chunks))

            packed = self.pack_step(merged_per_doc, max_chars=max_chars)

            # Combine: artifacts first, then packed regular chunks
            result = artifacts + packed
            return [_chunkdict_to_chunk(d) for d in result]

        except Exception as exc:
            raise PipelineError(f"Failed context pipeline: {exc}") from exc
