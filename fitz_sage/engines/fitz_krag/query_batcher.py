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

from fitz_sage.core.exceptions import QueryIntelligenceError
from fitz_sage.core.json_utils import parse_llm_json
from fitz_sage.engines.fitz_krag.query_analyzer import (
    QueryAnalysis,
    parse_analysis_dict,
)
from fitz_sage.retrieval.detection.modules import distribute_to_modules
from fitz_sage.retrieval.rewriter.rewriter import parse_rewrite_dict
from fitz_sage.retrieval.rewriter.types import RewriteResult

if TYPE_CHECKING:
    from fitz_sage.llm.factory import ChatFactory
    from fitz_sage.retrieval.detection.modules.base import DetectionModule
    from fitz_sage.retrieval.detection.protocol import DetectionCategory, DetectionResult


@dataclass(frozen=True)
class _QuerySectionSpec:
    """Prompt contract for one query-intelligence JSON section."""

    name: str
    json_fragment: str
    instructions: str


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
- Return every keyword as a separate quoted JSON array item; never combine
  multiple keywords into one comma-separated string.
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
SEMANTIC_KEYWORD_MAX_ITEMS = 10
SEMANTIC_KEYWORD_MAX_TOKENS = 128

_ANALYSIS_SPEC = _QuerySectionSpec("analysis", _ANALYSIS_JSON, _ANALYSIS_INSTRUCTIONS)
_REWRITING_SPEC = _QuerySectionSpec("rewriting", _REWRITING_JSON, _REWRITING_INSTRUCTIONS)
_EXTENDED_SPEC = _QuerySectionSpec("extended", _EXTENDED_JSON, _EXTENDED_INSTRUCTIONS)
_KEYWORDS_SPEC = _QuerySectionSpec("keywords", _KEYWORDS_JSON, _KEYWORDS_INSTRUCTIONS)


def _active_section_specs(
    *,
    include_analysis: bool,
    include_detection: bool,
    active_modules: list["DetectionModule"],
    include_rewriting: bool,
    include_extended: bool,
    include_keywords: bool,
) -> list[_QuerySectionSpec]:
    """Return prompt sections in the order expected by the batcher contract."""
    specs: list[_QuerySectionSpec] = []
    if include_analysis:
        specs.append(_ANALYSIS_SPEC)
    if include_detection and active_modules:
        fragments = [module.prompt_fragment() for module in active_modules]
        combined = ",\n    ".join(fragments)
        specs.append(
            _QuerySectionSpec(
                "detection",
                f'  "detection": {{\n    {combined}\n  }}',
                (
                    "## detection\n"
                    "Only set detected=true when the query CLEARLY matches. Default to false."
                ),
            )
        )
    if include_rewriting:
        specs.append(_REWRITING_SPEC)
    if include_keywords:
        specs.append(_KEYWORDS_SPEC)
    if include_extended:
        specs.append(_EXTENDED_SPEC)
    return specs


def _required_object(raw: dict[str, Any], section: str) -> dict[str, Any]:
    """Return a required JSON object section or raise a typed query error."""
    data = raw.get(section)
    if not isinstance(data, dict):
        raise QueryIntelligenceError(f"batched query intelligence missing `{section}` object")
    return data


def _required_array(raw: dict[str, Any], section: str) -> list[Any]:
    """Return a required JSON array section or raise a typed query error."""
    data = raw.get(section)
    if not isinstance(data, list):
        raise QueryIntelligenceError(f"batched query intelligence missing `{section}` array")
    return data


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
    max_tokens: int | None = None

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

        try:
            chat = self.chat_factory("fast")
            options = {"max_tokens": self.max_tokens} if self.max_tokens is not None else {}
            response = chat.chat([{"role": "user", "content": prompt}], **options)
        except QueryIntelligenceError:
            raise
        except Exception as e:
            raise QueryIntelligenceError(f"query intelligence provider failed: {e}") from e

        raw = parse_llm_json(response)
        if not isinstance(raw, dict):
            raise QueryIntelligenceError("batched query intelligence returned non-object JSON")

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
        specs = _active_section_specs(
            include_analysis=include_analysis,
            include_detection=include_detection,
            active_modules=active_modules,
            include_rewriting=include_rewriting,
            include_extended=include_extended,
            include_keywords=include_keywords,
        )
        section_names = [spec.name for spec in specs]
        json_parts = [spec.json_fragment for spec in specs]
        instruction_parts = [spec.instructions for spec in specs]

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
            analysis_data = _required_object(raw, "analysis")
            try:
                result.analysis = parse_analysis_dict(analysis_data, query)
            except Exception as e:
                raise QueryIntelligenceError(
                    "batched query intelligence returned invalid `analysis`"
                ) from e

        if include_detection and active_modules:
            detection_data = _required_object(raw, "detection")
            try:
                result.detection_results = distribute_to_modules(detection_data, active_modules)
            except Exception as e:
                raise QueryIntelligenceError(
                    "batched query intelligence returned invalid `detection`"
                ) from e

        if include_rewriting:
            rewrite_data = _required_object(raw, "rewriting")
            try:
                result.rewrite_result = parse_rewrite_dict(rewrite_data, query)
            except Exception as e:
                raise QueryIntelligenceError(
                    "batched query intelligence returned invalid `rewriting`"
                ) from e

        if include_extended:
            result.extended_signals = _required_object(raw, "extended")

        if include_keywords:
            kw_data = _required_array(raw, "keywords")
            result.keywords = [
                value
                for value in (str(k).strip() for k in kw_data if isinstance(k, str))
                if value and value.lower() not in _KEYWORD_PLACEHOLDERS
            ][:SEMANTIC_KEYWORD_MAX_ITEMS]

        return result
