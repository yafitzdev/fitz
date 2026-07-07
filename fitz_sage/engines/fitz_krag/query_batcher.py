# fitz_sage/engines/fitz_krag/query_batcher.py
"""
Batched query intelligence — combines analysis, detection, and rewriting
into a single LLM call to avoid model-swap overhead on local providers.

On ollama, three sequential LLM calls take 60-90s due to model swapping.
One batched call takes ~20s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fitz_sage.core.json_utils import parse_llm_json
from fitz_sage.engines.fitz_krag.query_analyzer import (
    QueryAnalysis,
    parse_analysis_dict,
)
from fitz_sage.retrieval.detection.detectors.expansion import expand_terms
from fitz_sage.retrieval.detection.modules import distribute_to_modules
from fitz_sage.retrieval.rewriter.rewriter import parse_rewrite_dict
from fitz_sage.retrieval.rewriter.types import RewriteResult

if TYPE_CHECKING:
    from fitz_sage.llm.factory import ChatFactory
    from fitz_sage.retrieval.detection.modules.base import DetectionModule
    from fitz_sage.retrieval.detection.protocol import DetectionCategory, DetectionResult

_PROMPT_TEMPLATE = """Analyze this search query. Return a single JSON object with {section_list}.
{history_section}
Query: "{query}"

Return this exact JSON structure:
{{
{sections_json}
}}

{sections_instructions}
Return JSON only, no markdown."""

_ANALYSIS_JSON = """\
  "analysis": {{
    "primary_type": "code" | "documentation" | "general" | "cross" | "data",
    "confidence": 0.0-1.0,
    "entities": [],
    "refined_query": "cleaned query text"
  }}"""

_ANALYSIS_INSTRUCTIONS = """\
## analysis
- "code": References functions, classes, methods, implementations
- "documentation": References document sections, specs, procedures
- "data": CSV/spreadsheet/SQL data queries
- "general": Overview questions, summaries
- "cross": Both code and documentation
- "entities": Specific symbol names or section titles mentioned
- "refined_query": Rewrite query to be more specific for search"""

_REWRITING_JSON = """\
  "rewriting": {{
    "rewritten_query": "improved query for retrieval",
    "rewrite_type": "none|clarity|retrieval|decomposition|combined",
    "confidence": 0.0-1.0,
    "is_compound": true/false,
    "decomposed_queries": [],
    "is_ambiguous": true/false,
    "disambiguated_queries": []
  }}"""

_REWRITING_INSTRUCTIONS = """\
## rewriting
- If no rewrite needed, set rewrite_type "none" and return the original query
- Fix typos, remove filler words, simplify complex phrasing
- Convert questions to statement form: "What is X?" -> "X definition overview"
- If the query covers multiple topics or several distinct points, set
  is_compound=true and provide focused decomposed_queries
- Resolve pronouns if conversation history is present"""

_EXTENDED_JSON = """\
  "extended": {{
    "specificity": "broad" | "moderate" | "narrow",
    "answer_type": "factual" | "procedural" | "comparative" | "exploratory",
    "domain": "general" | "technical" | "legal" | "financial" | "medical"
  }}"""

_EXTENDED_INSTRUCTIONS = """\
## extended
- specificity: "broad" for overview/survey questions, "narrow" for specific fact/symbol lookup, "moderate" otherwise
- answer_type: what kind of answer the user expects
- domain: primary domain vocabulary of the query"""

_KEYWORDS_JSON = """\
  "keywords": []"""

_KEYWORDS_INSTRUCTIONS = """\
## keywords
- Produce 5-10 concrete retrieval keywords or short search phrases.
- Use real synonyms, acronym expansions, sibling concepts, and domain vocabulary.
- Do not copy schema placeholders such as "term", "keyword", or "actual phrase".
- Avoid repeating the exact query unless it is itself a useful search phrase."""

_KEYWORD_PLACEHOLDERS = {
    "...",
    "actual phrase",
    "keyword",
    "keywords",
    "phrase",
    "search phrase",
    "term",
    "terms",
}


@dataclass
class BatchResult:
    """Result from a batched query intelligence call."""

    analysis: QueryAnalysis | None = None
    detection_results: dict["DetectionCategory", "DetectionResult"] | None = None
    rewrite_result: RewriteResult | None = None
    extended_signals: dict[str, Any] | None = None
    keywords: list[str] = field(default_factory=list)


@dataclass
class QueryBatcher:
    """Batches analysis + detection + rewriting into a single LLM call."""

    chat_factory: "ChatFactory"
    detection_modules: list["DetectionModule"] = field(default_factory=list)

    def batch_classify(
        self,
        query: str,
        *,
        include_analysis: bool = True,
        include_detection: bool = True,
        include_rewriting: bool = True,
        include_extended: bool = False,
        include_keywords: bool = True,
        conversation_context: Any = None,
    ) -> BatchResult:
        """Run analysis + detection + rewriting + keywords in one LLM call.

        Args:
            query: User query text.
            include_analysis: Include query type classification.
            include_detection: Include detection modules.
            include_rewriting: Include query rewriting.
            include_extended: Include extended advisory signals (specificity, domain, etc.).
            include_keywords: Include semantic keyword expansion.
            conversation_context: Optional ConversationContext for rewriting.

        Returns:
            BatchResult with per-section results (None for excluded sections).
        """
        active_modules = list(self.detection_modules) if include_detection else []
        if include_detection and not active_modules:
            include_detection = False

        prompt = self._build_prompt(
            query,
            include_analysis=include_analysis,
            include_detection=include_detection,
            active_modules=active_modules,
            include_rewriting=include_rewriting,
            include_extended=include_extended,
            include_keywords=include_keywords,
            conversation_context=conversation_context,
        )

        chat = self.chat_factory("fast")
        response = chat.chat([{"role": "user", "content": prompt}])
        raw = parse_llm_json(response)
        if not isinstance(raw, dict):
            raise ValueError("batched query intelligence returned non-object JSON")

        return self._distribute(
            raw,
            query,
            include_analysis=include_analysis,
            include_detection=include_detection,
            active_modules=active_modules,
            include_rewriting=include_rewriting,
            include_extended=include_extended,
            include_keywords=include_keywords,
        )

    def _build_prompt(
        self,
        query: str,
        *,
        include_analysis: bool,
        include_detection: bool,
        active_modules: list["DetectionModule"],
        include_rewriting: bool,
        include_extended: bool = False,
        include_keywords: bool = False,
        conversation_context: Any = None,
    ) -> str:
        """Build the combined prompt from active sections."""
        section_names = []
        json_parts = []
        instruction_parts = []

        if include_analysis:
            section_names.append("analysis")
            json_parts.append(_ANALYSIS_JSON)
            instruction_parts.append(_ANALYSIS_INSTRUCTIONS)

        if include_detection and active_modules:
            section_names.append("detection")
            fragments = [m.prompt_fragment() for m in active_modules]
            combined = ",\n    ".join(fragments)
            json_parts.append(f'  "detection": {{\n    {combined}\n  }}')
            instruction_parts.append(
                "## detection\n"
                "Only set detected=true when the query CLEARLY matches. Default to false."
            )

        if include_rewriting:
            section_names.append("rewriting")
            json_parts.append(_REWRITING_JSON)
            instruction_parts.append(_REWRITING_INSTRUCTIONS)

        if include_keywords:
            section_names.append("keywords")
            json_parts.append(_KEYWORDS_JSON)
            instruction_parts.append(_KEYWORDS_INSTRUCTIONS)

        if include_extended:
            section_names.append("extended")
            json_parts.append(_EXTENDED_JSON)
            instruction_parts.append(_EXTENDED_INSTRUCTIONS)

        if include_rewriting and (include_analysis or include_detection or include_keywords):
            instruction_parts.append(
                "## ordering\n"
                "First determine the rewritten query, then base analysis, "
                "detection, and keywords on that rewritten intent."
            )

        history_section = ""
        if conversation_context and hasattr(conversation_context, "format_for_prompt"):
            if not conversation_context.is_empty():
                history_section = (
                    f"\n## Conversation History\n{conversation_context.format_for_prompt()}\n"
                )

        section_list = " and ".join(section_names)

        return _PROMPT_TEMPLATE.format(
            query=query,
            section_list=section_list,
            history_section=history_section,
            sections_json=",\n".join(json_parts),
            sections_instructions="\n\n".join(instruction_parts),
        )

    def _distribute(
        self,
        raw: dict[str, Any],
        query: str,
        *,
        include_analysis: bool,
        include_detection: bool,
        active_modules: list["DetectionModule"],
        include_rewriting: bool,
        include_extended: bool = False,
        include_keywords: bool = False,
    ) -> BatchResult:
        """Distribute parsed JSON to per-section parsers.

        The caller already chose to use an LLM-backed batcher. Missing sections
        or parser failures are treated as model output errors instead of
        silently converting the call into deterministic query prep.
        """
        result = BatchResult()

        if include_analysis:
            analysis_data = raw.get("analysis")
            if not isinstance(analysis_data, dict):
                raise ValueError("batched query intelligence missing `analysis` object")
            try:
                result.analysis = parse_analysis_dict(analysis_data, query)
            except Exception as e:
                raise ValueError("batched query intelligence returned invalid `analysis`") from e

        if include_detection and active_modules:
            detection_data = raw.get("detection")
            if not isinstance(detection_data, dict):
                raise ValueError("batched query intelligence missing `detection` object")
            try:
                result.detection_results = distribute_to_modules(detection_data, active_modules)
            except Exception as e:
                raise ValueError("batched query intelligence returned invalid `detection`") from e

        if include_rewriting:
            rewrite_data = raw.get("rewriting")
            if not isinstance(rewrite_data, dict):
                raise ValueError("batched query intelligence missing `rewriting` object")
            try:
                result.rewrite_result = parse_rewrite_dict(rewrite_data, query)
            except Exception as e:
                raise ValueError("batched query intelligence returned invalid `rewriting`") from e

        if include_extended:
            extended_data = raw.get("extended")
            if not isinstance(extended_data, dict):
                raise ValueError("batched query intelligence missing `extended` object")
            result.extended_signals = extended_data

        if include_keywords:
            llm_keywords: list[str] = []
            kw_data = raw.get("keywords")
            if not isinstance(kw_data, list):
                raise ValueError("batched query intelligence missing `keywords` array")
            llm_keywords = [
                value
                for value in (str(k).strip() for k in kw_data if isinstance(k, str))
                if value and value.lower() not in _KEYWORD_PLACEHOLDERS
            ]
            # Fuse deterministic dict expansion (synonyms / acronyms) — always
            # available, independent of whether the LLM section parsed.
            seen: set[str] = set()
            merged: list[str] = []
            for term in (*llm_keywords, *expand_terms(query)):
                low = term.lower()
                if low not in seen:
                    seen.add(low)
                    merged.append(term)
            result.keywords = merged

        return result
