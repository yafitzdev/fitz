from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Dict

from .steps.normalize import NormalizeStep, _to_chunk_dict, ChunkDict
from .steps.dedupe import DedupeStep
from .steps.group import GroupByDocumentStep
from .steps.merge import MergeAdjacentStep
from .steps.pack import PackWindowStep
from .steps.render_markdown import RenderMarkdownStep

from fitz_rag.exceptions.pipeline import PipelineError


@dataclass
class ContextPipeline:
    """
    Modern step-based context builder.

    This is the ONLY entry point for building context.
    No legacy backward-compatibility layers remain.
    No old build_context() helper exists.

    Pipeline steps:
        normalize → dedupe → group → merge → pack → render
    """

    max_chars: int = 6000

    # Step instances
    normalize_step: NormalizeStep = field(default_factory=NormalizeStep)
    dedupe_step: DedupeStep = field(default_factory=DedupeStep)
    group_step: GroupByDocumentStep = field(default_factory=GroupByDocumentStep)
    merge_step: MergeAdjacentStep = field(default_factory=MergeAdjacentStep)
    pack_step: PackWindowStep = field(default_factory=PackWindowStep)
    render_step: RenderMarkdownStep = field(default_factory=RenderMarkdownStep)

    # --------------------------------------------------------
    # Main pipeline entry
    # --------------------------------------------------------
    def build(self, chunks: List[Any], max_chars: int | None = None) -> str:
        """
        Execute the full context-building pipeline.

        Parameters
        ----------
        chunks : list[Any]
            Retrieved raw chunks of any supported structure.
        max_chars : int | None
            Overrides the pipeline default.

        Returns
        -------
        str
            Markdown-formatted final context.
        """
        try:
            # 1. Normalize
            norm = self.normalize_step(chunks)

            # 2. Deduplicate
            deduped = self.dedupe_step(norm)

            # 3. Group by document
            grouped: Dict[str, List[ChunkDict]] = self.group_step(deduped)

            # 4. Merge adjacent per document
            merged_per_doc: List[ChunkDict] = []
            for _, file_chunks in grouped.items():
                merged_per_doc.extend(self.merge_step(file_chunks))

            # 5. Pack into window
            packed = self.pack_step(merged_per_doc, max_chars=max_chars)

            # 6. Render markdown
            out = self.render_step(packed)
            return out

        except Exception as e:
            raise PipelineError(f"Failed context pipeline: {e}") from e
