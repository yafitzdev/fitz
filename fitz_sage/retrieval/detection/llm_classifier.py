# fitz_sage/retrieval/detection/llm_classifier.py
"""
LLM-based query classification using detection modules.

Combines all module prompt fragments into a single LLM call,
then distributes results to each module for parsing.

Similar to the enrichment bus, but for query-time classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fitz_sage.core.json_utils import parse_llm_json
from fitz_sage.llm.factory import ChatFactory, ModelTier
from fitz_sage.logging.logger import get_logger

if TYPE_CHECKING:
    from .modules.base import DetectionModule
    from .protocol import DetectionCategory, DetectionResult

logger = get_logger(__name__)

PROMPT_HEADER = """Classify this search query. Return JSON only.

Query: "{query}"

Return this exact structure:
{{
  {module_fragments}
}}

Only set detected=true when the query CLEARLY matches the criteria. Default to detected=false unless there is explicit evidence."""


@dataclass
class LLMClassifier:
    """
    Module-based LLM classifier.

    Combines all module prompt fragments into one LLM call,
    then distributes results to each module for parsing.
    """

    chat_factory: ChatFactory
    modules: list["DetectionModule"] = field(default_factory=list)

    # Tier for classification (developer decision)
    TIER_CLASSIFY: ModelTier = "fast"

    def __post_init__(self):
        """Load default modules if none provided."""
        if not self.modules:
            from .modules import DEFAULT_MODULES

            self.modules = list(DEFAULT_MODULES)

    def classify(self, query: str) -> dict["DetectionCategory", "DetectionResult[Any]"]:
        """
        Classify query using all detection modules in one LLM call.

        Args:
            query: User's query string

        Returns:
            Dict mapping DetectionCategory to DetectionResult
        """
        prompt = self._build_prompt(query)

        try:
            chat = self.chat_factory(self.TIER_CLASSIFY)
            response = chat.chat([{"role": "user", "content": prompt}])
            raw_results = parse_llm_json(response)
            return self._distribute_to_modules(raw_results)
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return self._empty_results()

    def _build_prompt(self, query: str) -> str:
        """Build the combined prompt from all detection modules."""
        fragments = [m.prompt_fragment() for m in self.modules]
        combined = ",\n  ".join(fragments)
        return PROMPT_HEADER.format(query=query, module_fragments=combined)

    def _distribute_to_modules(
        self, raw_results: dict[str, Any]
    ) -> dict["DetectionCategory", "DetectionResult[Any]"]:
        """Distribute parsed results to each detection module."""
        return distribute_to_modules(raw_results, self.modules)

    def _empty_results(self) -> dict["DetectionCategory", "DetectionResult[Any]"]:
        """Return not-detected results for all detection modules."""
        return {module.category: module.not_detected() for module in self.modules}


def distribute_to_modules(
    raw_results: dict[str, Any],
    modules: "list[DetectionModule]",
) -> dict["DetectionCategory", "DetectionResult[Any]"]:
    """Distribute parsed JSON results to detection modules.

    Used by both LLMClassifier and QueryBatcher.
    """
    results = {}
    for module in modules:
        module_data = raw_results.get(module.json_key, {})
        if not isinstance(module_data, dict):
            module_data = {}
        results[module.category] = module.parse_result(module_data)
    return results
