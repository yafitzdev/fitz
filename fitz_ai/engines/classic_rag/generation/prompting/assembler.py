# pipeline/generation/prompting/assembler.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from fitz_ai.engines.classic_rag.generation.prompting.profiles import PromptProfile


@dataclass(frozen=True, slots=True)
class PromptConfig:
    """
    User overrides for prompt slot texts.

    Rules:
    - None => use defaults
    - provided str => replace exactly
    """

    system_base: str | None = None
    system_grounding: str | None = None
    system_safety: str | None = None

    context_header: str | None = None
    context_item: str | None = None

    user_instructions: str | None = None


@dataclass(frozen=True, slots=True)
class PromptAssembler:
    defaults: Any  # PromptSlots
    overrides: PromptConfig | None = None
    profile: PromptProfile = PromptProfile.RAG_USER

    def slot(self, name: str) -> str:
        # NOTE: profile is currently a no-op (RAG_USER only)
        if self.overrides is None:
            return str(getattr(self.defaults, name))

        value = getattr(self.overrides, name)
        if value is None:
            return str(getattr(self.defaults, name))
        return str(value)

    def build_system(
        self,
        *,
        strict_grounding: bool,
        enable_citations: bool,
        source_label_prefix: str,
        answer_style: str | None,
        max_answer_chars: int | None,
    ) -> str:
        parts: list[str] = [
            self.slot("system_base"),
            self.slot("system_grounding"),
        ]

        if strict_grounding:
            parts.append(self.slot("system_safety"))

        if enable_citations:
            parts.append(f"Use citations like [{source_label_prefix}1], [{source_label_prefix}2].")

        if answer_style:
            parts.append(f"Preferred style: {answer_style}.")

        if max_answer_chars:
            parts.append(f"Limit your answer to ~{max_answer_chars} characters.")

        return "\n".join(parts)

    def build_user(
        self,
        *,
        query: str,
        context_items: Iterable[str],
        include_query_in_context: bool,
        user_instructions: str | None,
    ) -> str:
        items = list(context_items)
        if not items:
            return (
                "No context snippets available.\n\n"
                f"User question:\n{query}\n"
                "Explain that you cannot answer based on missing context."
            )

        parts: list[str] = [
            self.slot("context_header"),
            "",
            "\n".join(items).rstrip(),
            "",
        ]

        instructions = (
            user_instructions if user_instructions is not None else self.slot("user_instructions")
        )
        if instructions and instructions.strip():
            parts.extend([instructions.strip(), ""])

        if include_query_in_context:
            parts.extend(["User question:", query.strip()])
        else:
            parts.append("Answer the question using ONLY the context above.")

        return "\n".join(parts)

    def format_context_item(self, *, header: str, content: str) -> str:
        tpl = self.slot("context_item")
        return tpl.format(header=header, content=content)
