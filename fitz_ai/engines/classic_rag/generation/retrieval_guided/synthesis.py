# fitz_ai/engines/classic_rag/generation/retrieval_guided/synthesis.py
"""
Retrieval-Guided Synthesis (RGS) - Generates grounded answers from retrieved chunks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fitz_ai.engines.classic_rag.exceptions import PipelineError, RGSGenerationError
from fitz_ai.engines.classic_rag.generation.prompting.assembler import PromptAssembler
from fitz_ai.engines.classic_rag.generation.prompting.profiles import PromptProfile
from fitz_ai.engines.classic_rag.generation.prompting.slots import PromptSlots

if TYPE_CHECKING:
    from fitz_ai.core.answer_mode import AnswerMode


def _get_attr(obj: Any, *keys: str, default: Any = None) -> Any:
    """Get attribute from dict or object, trying multiple keys in order."""
    is_dict = isinstance(obj, dict)
    for key in keys:
        val = obj.get(key) if is_dict else getattr(obj, key, None)
        if val is not None:
            return val
    return default


@dataclass
class RGSConfig:
    """Configuration for Retrieval-Guided Synthesis."""

    enable_citations: bool = True
    strict_grounding: bool = True
    answer_style: str | None = None
    max_chunks: int | None = 8
    max_answer_chars: int | None = None
    include_query_in_context: bool = True
    source_label_prefix: str = "S"
    prompt_config: dict[str, str] | None = None


@dataclass
class RGSPrompt:
    """Structured prompt for RGS."""

    system: str
    user: str


@dataclass
class RGSSourceRef:
    """Reference to a source chunk."""

    source_id: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str | None = None
    content: str | None = None


@dataclass
class RGSAnswer:
    """Structured answer from RGS."""

    answer: str
    sources: list[RGSSourceRef] = field(default_factory=list)
    mode: "AnswerMode | None" = None
    """
    Epistemic posture of the answer.

    Set by the pipeline based on constraint evaluation.
    None if constraints were not applied.
    """


# Type alias for chunk-like inputs
ChunkInput = dict[str, Any] | Any


class RGS:
    """
    Retrieval-Guided Synthesis engine.

    Builds prompts and structures answers from retrieved chunks.
    """

    def __init__(self, config: RGSConfig | None = None):
        self.config = config or RGSConfig()
        self._assembler = PromptAssembler(
            defaults=PromptSlots(),
            overrides=self.config.prompt_config,
            profile=PromptProfile.RAG_USER,
        )

    def build_prompt(self, query: str, chunks: Sequence[ChunkInput]) -> RGSPrompt:
        """Build system and user prompts from query and chunks."""
        try:
            limited = self._limit_chunks(chunks)
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(query, limited)
            return RGSPrompt(system=system_prompt, user=user_prompt)
        except Exception as e:
            raise RGSGenerationError("Failed to build RGS prompt") from e

    def build_answer(
        self,
        raw_answer: str,
        chunks: Sequence[ChunkInput],
        mode: "AnswerMode | None" = None,
    ) -> RGSAnswer:
        """Structure raw LLM output into RGSAnswer with source references."""
        try:
            limited = self._limit_chunks(chunks)
            sources: list[RGSSourceRef] = []

            for idx, chunk in enumerate(limited, start=1):
                cid = self._get_chunk_id(chunk, idx)
                meta = self._get_chunk_metadata(chunk)
                doc_id = self._get_chunk_doc_id(chunk)
                content = self._get_chunk_content(chunk)

                sources.append(
                    RGSSourceRef(
                        source_id=cid,
                        index=idx,
                        metadata=meta,
                        doc_id=doc_id,
                        content=content,
                    )
                )

            return RGSAnswer(answer=raw_answer, sources=sources, mode=mode)
        except Exception as e:
            raise PipelineError("Failed to build RGS answer") from e

    def _limit_chunks(self, chunks: Sequence[ChunkInput]) -> list[ChunkInput]:
        """Limit chunks to max_chunks config."""
        if self.config.max_chunks is None:
            return list(chunks)
        return list(chunks[: self.config.max_chunks])

    def _build_system_prompt(self) -> str:
        """Build the system prompt."""
        return self._assembler.build_system(
            enable_citations=self.config.enable_citations,
            strict_grounding=self.config.strict_grounding,
            answer_style=self.config.answer_style,
            source_label_prefix=self.config.source_label_prefix,
            max_answer_chars=self.config.max_answer_chars,
        )

    def _build_user_prompt(self, query: str, chunks: Sequence[ChunkInput]) -> str:
        """Build the user prompt with context and query."""
        context_items = []

        for idx, chunk in enumerate(chunks, start=1):
            content = self._get_chunk_content(chunk)
            label = f"[{self.config.source_label_prefix}{idx}]"
            context_items.append(f"{label}\n{content}")

        return self._assembler.build_user(
            query=query,
            context_items=context_items,
            include_query_in_context=self.config.include_query_in_context,
            user_instructions=None,
        )

    def _get_chunk_content(self, chunk: ChunkInput) -> str:
        """Extract content from chunk-like object."""
        return str(_get_attr(chunk, "content", "text", default="") or "")

    def _get_chunk_id(self, chunk: ChunkInput, index: int) -> str:
        """Extract or generate chunk ID."""
        chunk_id = _get_attr(chunk, "id", "chunk_id")
        return str(chunk_id) if chunk_id is not None else f"chunk_{index}"

    def _get_chunk_doc_id(self, chunk: ChunkInput) -> str:
        """Extract document ID from chunk-like object."""
        return str(
            _get_attr(
                chunk,
                "doc_id",
                "document_id",
                "source_file",
                "source",
                default="unknown",
            )
        )

    def _get_chunk_metadata(self, chunk: ChunkInput) -> dict[str, Any]:
        """Extract metadata from chunk-like object."""
        meta = _get_attr(chunk, "metadata", default={})
        return dict(meta) if isinstance(meta, Mapping) else {}


__all__ = [
    "RGS",
    "RGSConfig",
    "RGSPrompt",
    "RGSAnswer",
    "RGSSourceRef",
]
