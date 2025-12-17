# pipeline/generation/synthesis.py
"""
Retrieval-Guided Synthesis (RGS)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

from fitz.generation.exceptions.base import GenerationError, RGSGenerationError
from fitz.generation.prompting import (
    PromptAssembler,
    PromptConfig,
    PromptProfile,
    PromptSlots,
)


@runtime_checkable
class SupportsRGSChunk(Protocol):
    id: str
    content: str
    metadata: Dict[str, Any]


ChunkInput = Union[SupportsRGSChunk, Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class RGSConfig:
    enable_citations: bool = True
    strict_grounding: bool = True
    answer_style: str | None = None
    max_chunks: int | None = 8
    max_answer_chars: int | None = None
    include_query_in_context: bool = True
    source_label_prefix: str = "S"

    prompt_config: PromptConfig | None = None


@dataclass(frozen=True, slots=True)
class RGSSourceRef:
    source_id: str
    index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RGSAnswer:
    answer: str
    sources: List[RGSSourceRef]


@dataclass(frozen=True, slots=True)
class RGSPrompt:
    system: str
    user: str


class RGS:
    def __init__(self, config: RGSConfig | None = None) -> None:
        self.config: RGSConfig = config or RGSConfig()
        self._assembler = PromptAssembler(
            defaults=PromptSlots(),
            overrides=self.config.prompt_config,
            profile=PromptProfile.RAG_USER,
        )

    def build_prompt(self, query: str, chunks: Sequence[ChunkInput]) -> RGSPrompt:
        try:
            limited = self._limit_chunks(chunks)
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(query, limited)
            return RGSPrompt(system=system_prompt, user=user_prompt)
        except Exception as e:
            raise GenerationError("Failed to build RGS answer") from e

    def build_answer(self, raw_answer: str, chunks: Sequence[ChunkInput]) -> RGSAnswer:
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
        if self.config.max_chunks is None:
            return list(chunks)
        return list(chunks[: self.config.max_chunks])

    def _build_system_prompt(self) -> str:
        return self._assembler.build_system(
            strict_grounding=self.config.strict_grounding,
            enable_citations=self.config.enable_citations,
            source_label_prefix=self.config.source_label_prefix,
            answer_style=self.config.answer_style,
            max_answer_chars=self.config.max_answer_chars,
        )

    def _build_user_prompt(self, query: str, chunks: Sequence[ChunkInput]) -> str:
        context_items: list[str] = []
        prefix = self.config.source_label_prefix

        for idx, chunk in enumerate(chunks, start=1):
            label = f"[{prefix}{idx}]" if self.config.enable_citations else ""
            content = self._get_chunk_content(chunk).strip()
            metadata = self._get_chunk_metadata(chunk)

            if metadata:
                meta_str = self._format_metadata(metadata)
                header = f"{label} (metadata: {meta_str})" if label else f"(metadata: {meta_str})"
            else:
                header = label

            if header:
                item = self._assembler.format_context_item(header=header, content=content)
            else:
                item = self._assembler.format_context_item(header="", content=content).lstrip("\n")

            context_items.append(item)

        return self._assembler.build_user(
            query=query,
            context_items=context_items,
            include_query_in_context=self.config.include_query_in_context,
            user_instructions=(
                self.config.prompt_config.user_instructions
                if self.config.prompt_config is not None
                else None
            ),
        )

    @staticmethod
    def _get_chunk_id(chunk: ChunkInput, fallback_index: int) -> str:
        if hasattr(chunk, "id") and isinstance(getattr(chunk, "id"), str):
            return getattr(chunk, "id")

        if isinstance(chunk, Mapping) and isinstance(chunk.get("id"), str):
            return chunk["id"]

        return f"chunk_{fallback_index}"

    @staticmethod
    def _get_chunk_content(chunk: ChunkInput) -> str:
        if hasattr(chunk, "content") and isinstance(getattr(chunk, "content"), str):
            return getattr(chunk, "content")

        if isinstance(chunk, Mapping) and isinstance(chunk.get("content"), str):
            return chunk["content"]

        raise ValueError("Chunk missing required 'content' field")

    @staticmethod
    def _get_chunk_metadata(chunk: ChunkInput) -> Dict[str, Any]:
        if hasattr(chunk, "metadata") and isinstance(getattr(chunk, "metadata"), Mapping):
            return dict(getattr(chunk, "metadata"))

        if isinstance(chunk, Mapping) and isinstance(chunk.get("metadata"), Mapping):
            return dict(chunk["metadata"])

        return {}

    @staticmethod
    def _format_metadata(metadata: Mapping[str, Any], max_items: int = 3) -> str:
        items = list(metadata.items())[:max_items]
        parts = [f"{k}={v!r}" for k, v in items]
        if len(metadata) > max_items:
            parts.append("...")
        return ", ".join(parts)
