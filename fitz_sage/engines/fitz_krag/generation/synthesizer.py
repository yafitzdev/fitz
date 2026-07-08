# fitz_sage/engines/fitz_krag/generation/synthesizer.py
"""
Code synthesizer — generates grounded answers from code context.

Adapts the RGS pattern for address-based provenance. Answers include
[S1], [S2] citation markers mapping to file:line source blocks.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from fitz_sage.core import Answer, GenerationError, Provenance
from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.engines.fitz_krag.types import ReadResult
from fitz_sage.governance.instructions import get_mode_instruction

if TYPE_CHECKING:
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.llm.providers.base import ChatProvider

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_GROUNDED = """\
You are a code-aware assistant. Answer questions using ONLY the provided source \
code and documentation context. Cite sources using [S1], [S2], etc. markers that \
correspond to the numbered source blocks.

Rules:
- Only use information from the provided context
- Cite specific sources with [S1], [S2] markers
- If the context does not contain relevant information to answer the question, \
respond with "No information found" or "Unable to find relevant information"
- Reference specific files and line numbers when helpful
- Be precise about what the code does, not what it might do"""

SYSTEM_PROMPT_OPEN = """\
You are a code-aware assistant. Answer questions using the provided source code \
and documentation context as primary references. Cite sources using [S1], [S2], \
etc. markers. You may supplement with general knowledge when the context is \
insufficient, but clearly indicate when doing so. If the context does not contain \
relevant information, respond with "No information found"."""

CONCISE_ANSWER_INSTRUCTION = """\
Keep the answer concise and evidence-first. Start with the direct answer, then \
add only the source-backed details needed to justify it. Prefer 1-3 short \
paragraphs or a compact bullet list."""

SHORT_FACTUAL_ANSWER_INSTRUCTION = """\
This is a specific factual question. Answer in 1-3 sentences unless the sources \
show multiple conflicting facts. Do not write background explanation before the \
answer."""

SHORT_FACTUAL_MAX_TOKENS = 192


class CodeSynthesizer:
    """Generates grounded answers from code context."""

    def __init__(self, chat: "ChatProvider", config: "FitzKragConfig"):
        self._chat = chat
        self._config = config

    def generate(
        self,
        query: str,
        context: str,
        results: list[ReadResult],
        answer_mode: AnswerMode = AnswerMode.SUFFICIENT,
        gap_context: dict | None = None,
        conflict_context: dict | None = None,
    ) -> Answer:
        """
        Generate an answer grounded in the provided code context.

        Args:
            query: User's question
            context: Assembled context string from ContextAssembler
            results: ReadResults used to build provenance
            answer_mode: Epistemic posture controlling answer framing
            gap_context: Gap analysis for actionable INSUFFICIENT messages
            conflict_context: Conflict details for actionable DISPUTED messages

        Returns:
            Answer with text, provenance, and metadata
        """
        # INSUFFICIENT: generate actionable diagnostic instead of generic refusal
        if answer_mode == AnswerMode.INSUFFICIENT:
            provenance = self._build_provenance(results)
            return Answer(
                text=self._build_insufficient_message(query, gap_context),
                provenance=provenance,
                mode=answer_mode,
                metadata={
                    "engine": "fitz_krag",
                    "query": query,
                    "sources_count": len(results),
                    "answer_mode": answer_mode.value,
                    "gap_context": gap_context or {},
                },
            )

        system_prompt = (
            SYSTEM_PROMPT_GROUNDED if self._config.strict_grounding else SYSTEM_PROMPT_OPEN
        )

        # Prepend mode instruction, enriched with conflict details for DISPUTED
        mode_instruction = get_mode_instruction(answer_mode)
        if answer_mode == AnswerMode.DISPUTED and conflict_context:
            mode_instruction = self._build_disputed_instruction(conflict_context)
        style_instruction = self._build_style_instruction(query)
        system_prompt = f"{mode_instruction}\n\n{style_instruction}\n\n{system_prompt}"

        user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
        max_tokens = self._max_tokens_for_query(query)

        try:
            raw_answer = self._chat.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise GenerationError(f"LLM generation failed: {e}") from e

        provenance = self._build_provenance(results)

        metadata = {
            "engine": "fitz_krag",
            "query": query,
            "sources_count": len(results),
            "answer_mode": answer_mode.value,
        }
        if conflict_context:
            metadata["conflict_context"] = conflict_context

        return Answer(
            text=raw_answer,
            provenance=provenance,
            mode=answer_mode,
            metadata=metadata,
        )

    def _build_style_instruction(self, query: str) -> str:
        """Return synthesis style guidance for the query shape."""
        if self._is_short_factual_query(query):
            return f"{CONCISE_ANSWER_INSTRUCTION}\n\n{SHORT_FACTUAL_ANSWER_INSTRUCTION}"
        return CONCISE_ANSWER_INSTRUCTION

    def _max_tokens_for_query(self, query: str) -> int:
        """Return the output cap for the query shape."""
        configured_max = int(self._config.max_answer_tokens)
        if not self._is_short_factual_query(query):
            return configured_max

        configured_short = int(
            getattr(self._config, "short_answer_tokens", SHORT_FACTUAL_MAX_TOKENS)
        )
        return min(configured_max, configured_short)

    @staticmethod
    def _is_short_factual_query(query: str) -> bool:
        """Detect narrow factual questions that should get compact synthesis."""
        normalized = re.sub(r"\s+", " ", query.strip().lower())
        if not normalized:
            return False

        broad_cues = (
            "architecture",
            "compare",
            "design",
            "explain",
            "how does",
            "overview",
            "summarize",
            "why",
        )
        if any(cue in normalized for cue in broad_cues):
            return False

        factual_prefixes = (
            "what ",
            "which ",
            "who ",
            "when ",
            "where ",
            "how many ",
            "how much ",
        )
        return normalized.startswith(factual_prefixes) and len(normalized.split()) <= 18

    def _build_disputed_instruction(self, conflict_context: dict) -> str:
        """
        Build an enriched DISPUTED mode instruction that tells the LLM
        exactly WHAT conflicts so it can address the disagreement specifically.
        """
        source_a = conflict_context.get("source_a", "Source A")
        source_b = conflict_context.get("source_b", "Source B")
        excerpt_a = conflict_context.get("excerpt_a", "")[:200]
        excerpt_b = conflict_context.get("excerpt_b", "")[:200]

        return (
            "Sources disagree on this topic. Present BOTH perspectives clearly "
            "and do not assert one view as correct.\n\n"
            f"Specifically, there is a conflict between:\n"
            f'- {source_a}: "{excerpt_a}..."\n'
            f'- {source_b}: "{excerpt_b}..."\n\n'
            "Explain what each source says, where they disagree, "
            "and note that the user should verify which is current/authoritative."
        )

    def _build_insufficient_message(self, query: str, gap_context: dict | None) -> str:
        """
        Build an actionable INSUFFICIENT message that explains WHY the system
        can't answer and WHAT the user can do about it.

        Instead of "I don't have enough information", tells the user:
        1. What was searched for
        2. What related topics exist in the corpus
        3. What documents to add to fill the gap
        """
        if not gap_context:
            return "The available information does not allow a definitive answer."

        lines = ["I don't have enough information to answer this question."]

        # Governance reasons (why constraints fired)
        reasons = gap_context.get("governance_reasons", ())
        if reasons:
            lines.append("")
            lines.append("Why:")
            for reason in reasons[:3]:
                lines.append(f"  - {reason}")

        # Related topics that DO exist in the corpus
        related = gap_context.get("related_topics", [])
        if related:
            lines.append("")
            lines.append("Related topics in the knowledge base:")
            for topic in related[:5]:
                name = topic.get("name", "")
                mentions = topic.get("mentions", 0)
                topic_type = topic.get("type", "")
                suffix = f" ({topic_type})" if topic_type else ""
                lines.append(f"  - {name}{suffix} — {mentions} mentions")

        # Top corpus entities (when no related topics found, show what IS available)
        if not related:
            top = gap_context.get("top_corpus_topics", [])
            if top:
                lines.append("")
                lines.append("Topics available in the knowledge base:")
                for topic in top[:5]:
                    lines.append(
                        f"  - {topic.get('name', '')} ({topic.get('mentions', 0)} mentions)"
                    )

        # Suggestions for what to add
        lines.append("")
        lines.append("To answer this question, consider adding:")
        corpus_size = gap_context.get("corpus_document_count", 0)
        if corpus_size == 0:
            lines.append("  - Documents or code files covering this topic")
            lines.append(
                "  - Use fitz_sage.query('your question', source='./path') to query a directory"
            )
        else:
            lines.append("  - Documents covering the specific topic of this question")
            lines.append("  - More detailed documentation or examples on this subject")

        return "\n".join(lines)

    def _build_provenance(self, results: list[ReadResult]) -> list[Provenance]:
        """Build provenance list from read results."""
        provenance = []
        for r in results:
            # Skip context-type expansions (imports, class headers) from provenance
            if r.metadata.get("context_type"):
                continue

            metadata = {
                "kind": r.address.kind.value,
                "location": r.address.location,
                "file_path": r.file_path,
            }
            if r.line_range:
                metadata["line_range"] = list(r.line_range)
            metadata.update({k: v for k, v in r.address.metadata.items() if k != "context_type"})

            provenance.append(
                Provenance(
                    source_id=r.address.source_id,
                    excerpt=r.content[:200] if r.content else "",
                    metadata=metadata,
                )
            )
        return provenance
