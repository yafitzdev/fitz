# fitz_sage/engines/fitz_krag/retrieval/multihop.py
"""
Multi-hop retrieval controller for KRAG.

Loops a `RetrievalPass` (retrieve -> rerank -> read): after each pass the
pyrrho governance classifier judges whether the accumulated evidence is
enough. SUFFICIENT / DISPUTED -> stop; INSUFFICIENT -> extract a bridge
question and run another pass. The pass already reranks, so every hop's
candidates get the cross-encoder — multi-hop is purely a loop on top.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.core.json_utils import parse_llm_json
from fitz_sage.engines.fitz_krag.types import ReadResult

if TYPE_CHECKING:
    from fitz_sage.engines.fitz_krag.retrieval.retrieval_pass import RetrievalPass
    from fitz_sage.governance import Pyrrho
    from fitz_sage.llm.factory import ChatFactory

logger = logging.getLogger(__name__)


class KragHopController:
    """
    Multi-hop retrieval: loop a `RetrievalPass`, pyrrho-gated.

    Iterates: run a pass -> evaluate sufficiency (pyrrho) -> extract a
    bridge question -> run another pass with the bridge query.
    """

    def __init__(
        self,
        retrieval_pass: "RetrievalPass",
        chat_factory: "ChatFactory",
        governance: "Pyrrho | None" = None,
        max_hops: int = 2,
    ):
        self._pass = retrieval_pass
        self._chat_factory = chat_factory
        self._governance = governance
        self._max_hops = max_hops

    def execute(
        self,
        query: str,
        profile: Any = None,
    ) -> list[ReadResult]:
        """
        Run iterative multi-hop retrieval.

        Args:
            query: Original user query
            profile: RetrievalProfile with pre-computed gates and signals

        Returns:
            Accumulated read results across all hops
        """
        all_results: list[ReadResult] = []
        seen: set[tuple[str, str]] = set()
        current_query = query

        for hop in range(self._max_hops):
            results = self._pass.run(current_query, profile, exclude=seen)
            if not results:
                break

            for r in results:
                seen.add((r.address.source_id, r.address.location))
            all_results.extend(results)

            # Evaluate sufficiency — pyrrho verdict, no chat call.
            if self._is_sufficient(query, all_results):
                logger.debug(f"Multi-hop: sufficient evidence at hop {hop + 1}")
                break

            # Extract bridge questions to fill the remaining gap.
            bridge_questions = self._extract_bridge(query, all_results)
            if not bridge_questions:
                logger.debug(f"Multi-hop: no bridge questions at hop {hop + 1}")
                break

            current_query = bridge_questions[0]
            logger.debug(f"Multi-hop: bridge query = '{current_query[:80]}'")

        return all_results

    def _is_sufficient(self, query: str, results: list[ReadResult]) -> bool:
        """Decide whether to stop hopping, using the pyrrho governance verdict.

        SUFFICIENT or DISPUTED -> stop (evidence is enough, or the sources
        disagree and more retrieval will not resolve it); INSUFFICIENT -> keep
        hopping. With no classifier wired (governance disabled) there is no
        cheap sufficiency signal, so the loop relies on bridge extraction
        and max_hops to terminate.
        """
        if not results or self._governance is None:
            return False
        decision = self._governance.decide(query, results)
        return decision.mode is not AnswerMode.INSUFFICIENT

    def _extract_bridge(self, query: str, results: list[ReadResult]) -> list[str]:
        """Generate bridge questions to fill evidence gaps."""
        context = self._build_context(results)
        prompt = (
            "You're helping answer a question. The current evidence is missing information.\n\n"
            f"Original question: {query}\n\n"
            f"Current evidence:\n{context}\n\n"
            "What specific follow-up question would help find the missing information?\n"
            'Return ONLY a JSON array: ["query1", "query2"]\n'
            "If no clear gaps, return: []"
        )

        try:
            chat = self._chat_factory("fast")
            response = chat.chat([{"role": "user", "content": prompt}])
            parsed = parse_llm_json(response, as_array=True)
            return [str(q) for q in parsed[:2] if isinstance(q, str) and q.strip()]
        except Exception as e:
            logger.warning(f"Bridge extraction failed: {e}")

        return []

    def _build_context(self, results: list[ReadResult], max_chars: int = 5000) -> str:
        """Build context string from read results."""
        parts: list[str] = []
        total = 0
        for r in results:
            content = r.content[:500]
            if total + len(content) > max_chars:
                break
            parts.append(f"[{r.file_path}] {content}")
            total += len(content) + 20
        return "\n\n".join(parts)
