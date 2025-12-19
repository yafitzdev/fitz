# fitz/engines/classic_rag/generation/retrieval_guided/synthesis.py
"""
Retrieval-Guided Synthesis (RGS) - Generates grounded answers from retrieved chunks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from fitz_ai.engines.classic_rag.exceptions import PipelineError, RGSGenerationError
from fitz_ai.engines.classic_rag.generation.prompting.assembler import PromptAssembler
from fitz_ai.engines.classic_rag.generation.prompting.profiles import PromptProfile
from fitz_ai.engines.classic_rag.generation.prompting.slots import PromptSlots


@dataclass
class RGSConfig:
    """Configuration for Retrieval-Guided Synthesis."""

    enable_citations: bool = True
    strict_grounding: bool = True
    answer_style: Optional[str] = None
    max_chunks: Optional[int] = 8
    max_answer_chars: Optional[int] = None
    include_query_in_context: bool = True
    source_label_prefix: str = "S"
    prompt_config: Optional[Dict[str, str]] = None


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
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RGSAnswer:
    """Structured answer from RGS."""

    answer: str
    sources: List[RGSSourceRef] = field(default_factory=list)


# Type alias for chunk-like inputs
ChunkInput = Union[Dict[str, Any], Any]


class RGS:
    """
    Retrieval-Guided Synthesis engine.

    Builds prompts and structures answers from retrieved chunks.
    """

    def __init__(self, config: Optional[RGSConfig] = None):
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

    def build_answer(self, raw_answer: str, chunks: Sequence[ChunkInput]) -> RGSAnswer:
        """Structure raw LLM output into RGSAnswer with source references."""
        try:
            limited = self._limit_chunks(chunks)
            sources: List[RGSSourceRef] = []

            for idx, chunk in enumerate(limited, start=1):
                cid = self._get_chunk_id(chunk, idx)
                meta = self._get_chunk_metadata(chunk)
                sources.append(RGSSourceRef(source_id=cid, index=idx, metadata=meta))

            return RGSAnswer(answer=raw_answer, sources=sources)
        except Exception as e:
            raise PipelineError("Failed to build RGS answer") from e

    def _limit_chunks(self, chunks: Sequence[ChunkInput]) -> List[ChunkInput]:
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
        if isinstance(chunk, dict):
            return str(chunk.get("content") or chunk.get("text") or "")
        return str(getattr(chunk, "content", None) or getattr(chunk, "text", "") or "")

    def _get_chunk_id(self, chunk: ChunkInput, index: int) -> str:
        """Extract or generate chunk ID."""
        if isinstance(chunk, dict):
            return str(chunk.get("id") or chunk.get("chunk_id") or f"chunk_{index}")
        return str(
            getattr(chunk, "id", None) or getattr(chunk, "chunk_id", None) or f"chunk_{index}"
        )

    def _get_chunk_metadata(self, chunk: ChunkInput) -> Dict[str, Any]:
        """Extract metadata from chunk-like object."""
        if isinstance(chunk, dict):
            meta = chunk.get("metadata", {})
            return dict(meta) if isinstance(meta, Mapping) else {}
        meta = getattr(chunk, "metadata", {})
        return dict(meta) if isinstance(meta, Mapping) else {}


__all__ = [
    "RGS",
    "RGSConfig",
    "RGSPrompt",
    "RGSAnswer",
    "RGSSourceRef",
]
