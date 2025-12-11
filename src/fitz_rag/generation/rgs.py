"""
Retrieval-Guided Synthesis (RGS)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Protocol, Union

from fitz_rag.exceptions.pipeline import RGSGenerationError, PipelineError


class SupportsRGSChunk(Protocol):
    id: str
    content: str
    metadata: Dict[str, Any]


ChunkInput = Union[SupportsRGSChunk, Mapping[str, Any]]


@dataclass
class RGSConfig:
    enable_citations: bool = True
    strict_grounding: bool = True
    answer_style: Optional[str] = None
    max_chunks: Optional[int] = 8
    max_answer_chars: Optional[int] = None
    include_query_in_context: bool = True
    source_label_prefix: str = "S"


@dataclass
class RGSSourceRef:
    source_id: str
    index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RGSAnswer:
    answer: str
    sources: List[RGSSourceRef]


@dataclass
class RGSPrompt:
    system: str
    user: str


class RGS:
    def __init__(self, config: Optional[RGSConfig] = None) -> None:
        self.config: RGSConfig = config or RGSConfig()

    # ------------------------------------------------------------------
    def build_prompt(
        self,
        query: str,
        chunks: Sequence[ChunkInput],
    ) -> RGSPrompt:
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
    ) -> RGSAnswer:
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

    # ------------------------------------------------------------------
    def _limit_chunks(self, chunks: Sequence[ChunkInput]) -> List[ChunkInput]:
        if self.config.max_chunks is None:
            return list(chunks)
        return list(chunks[: self.config.max_chunks])

    def _build_system_prompt(self) -> str:
        parts = [
            "You are a retrieval-grounded assistant.",
            "You must answer ONLY using the provided context snippets.",
        ]

        if self.config.strict_grounding:
            parts.append(
                "If the answer is not contained in the context, say "
                "\"I don't know based on the provided information.\""
            )

        if self.config.enable_citations:
            prefix = self.config.source_label_prefix
            parts.append(f"Use citations like [{prefix}1], [{prefix}2].")

        if self.config.answer_style:
            parts.append(f"Preferred style: {self.config.answer_style}.")

        if self.config.max_answer_chars:
            parts.append(f"Limit your answer to ~{self.config.max_answer_chars} characters.")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    def _build_user_prompt(
        self,
        query: str,
        chunks: Sequence[ChunkInput],
    ) -> str:
        if not chunks:
            return (
                "No context snippets available.\n\n"
                f"User question:\n{query}\n"
                "Explain that you cannot answer based on missing context."
            )

        prefix = self.config.source_label_prefix
        context_lines: List[str] = []

        for idx, chunk in enumerate(chunks, start=1):

            # Build label ONLY when citations enabled
            if self.config.enable_citations:
                label = f"[{prefix}{idx}]"
            else:
                label = ""

            content = self._get_chunk_content(chunk)
            metadata = self._get_chunk_metadata(chunk)

            # Header formatting
            if metadata:
                if label:
                    header = f"{label} (metadata: {self._format_metadata(metadata)})"
                else:
                    header = f"(metadata: {self._format_metadata(metadata)})"
            else:
                header = label

            context_lines.append(header)
            context_lines.append(content.strip())
            context_lines.append("")

        context_block = "\n".join(context_lines).rstrip()

        user_parts = [
            "You are given the following context snippets:",
            "",
            context_block,
            "",
        ]

        if self.config.include_query_in_context:
            user_parts.extend(["User question:", query.strip()])
        else:
            user_parts.append("Answer the question using ONLY the context above.")

        return "\n".join(user_parts)

    # ------------------------------------------------------------------
    @staticmethod
    def _get_chunk_id(chunk: ChunkInput, fallback_index: int) -> str:
        if hasattr(chunk, "id") and isinstance(getattr(chunk, "id"), str):
            return getattr(chunk, "id")

        if isinstance(chunk, Mapping) and isinstance(chunk.get("id"), str):
            return chunk["id"]

        return f"chunk_{fallback_index}"

    @staticmethod
    def _get_chunk_content(chunk: ChunkInput) -> str:
        if hasattr(chunk, "text") and isinstance(getattr(chunk, "text"), str):
            return getattr(chunk, "text")

        if hasattr(chunk, "content") and isinstance(getattr(chunk, "content"), str):
            return getattr(chunk, "content")

        if isinstance(chunk, Mapping):
            if isinstance(chunk.get("text"), str):
                return chunk["text"]
            if isinstance(chunk.get("content"), str):
                return chunk["content"]

        raise ValueError("Chunk missing usable text field")

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
