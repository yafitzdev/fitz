# tests/unit/test_krag_reranker.py
"""
Unit tests for AddressReranker.

Tests cross-encoder reranking of KRAG addresses: reranking when addresses
reach min_addresses, passthrough truncation when below, and error handling.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fitz_sage.engines.fitz_krag.retrieval.reranker import AddressReranker
from fitz_sage.engines.fitz_krag.types import Address, AddressKind

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _addr(
    location: str = "mod.func",
    score: float = 0.5,
    source_id: str = "src",
    summary: str = "does something",
) -> Address:
    """Build an Address."""
    return Address(
        kind=AddressKind.SYMBOL,
        source_id=source_id,
        location=location,
        summary=summary,
        score=score,
    )


def _make_rerank_result(index: int, score: float) -> MagicMock:
    """Create a mock RerankResult with index and score attributes."""
    result = MagicMock()
    result.index = index
    result.score = score
    return result


# ---------------------------------------------------------------------------
# TestAddressReranker
# ---------------------------------------------------------------------------


class TestAddressReranker:
    """Tests for AddressReranker.rerank()."""

    def test_default_reranks_two_or_more_addresses(self):
        """The product default reranks as soon as there is a real choice."""
        reranker_provider = MagicMock(name="reranker")
        reranker_provider.rerank.return_value = [
            _make_rerank_result(index=1, score=0.95),
            _make_rerank_result(index=0, score=0.85),
        ]
        addresses = [
            _addr(location="func_a", score=0.8),
            _addr(location="func_b", score=0.7),
        ]

        reranker = AddressReranker(reranker=reranker_provider, k=2)

        result = reranker.rerank("query", addresses)

        reranker_provider.rerank.assert_called_once()
        assert [address.location for address in result] == ["func_b", "func_a"]

    def test_reranks_when_addresses_ge_min_addresses(self):
        """Reranking is applied when address count >= min_addresses."""
        reranker_provider = MagicMock(name="reranker")

        # 5 addresses, min_addresses=5 -> reranking triggers
        addresses = [
            _addr(location=f"func_{i}", score=0.5 + i * 0.1, summary=f"Summary {i}")
            for i in range(5)
        ]

        # Reranker returns top-3 in reverse order by score
        reranker_provider.rerank.return_value = [
            _make_rerank_result(index=4, score=0.95),
            _make_rerank_result(index=3, score=0.85),
            _make_rerank_result(index=2, score=0.75),
        ]

        reranker = AddressReranker(
            reranker=reranker_provider,
            k=3,
            min_addresses=5,
        )

        result = reranker.rerank("what does func_4 do?", addresses)

        # Reranker called with summaries and top_n
        reranker_provider.rerank.assert_called_once()
        call_args = reranker_provider.rerank.call_args
        assert call_args[0][0] == "what does func_4 do?"
        assert len(call_args[0][1]) == 5  # all documents
        assert call_args[1]["top_n"] == 3

        # Returns 3 reranked addresses
        assert len(result) == 3

        # Scores come from the reranker, not originals
        assert result[0].score == 0.95
        assert result[1].score == 0.85
        assert result[2].score == 0.75

        # Address identity preserved
        assert result[0].location == "func_4"
        assert result[1].location == "func_3"
        assert result[2].location == "func_2"

    def test_returns_first_k_when_addresses_lt_min_addresses(self):
        """When fewer than min_addresses, skip reranking and return first k."""
        reranker_provider = MagicMock(name="reranker")

        # 3 addresses, min_addresses=5 -> skip reranking
        addresses = [
            _addr(location="func_a", score=0.9),
            _addr(location="func_b", score=0.8),
            _addr(location="func_c", score=0.7),
        ]

        reranker = AddressReranker(
            reranker=reranker_provider,
            k=2,
            min_addresses=5,
        )

        result = reranker.rerank("query", addresses)

        # Reranker NOT called
        reranker_provider.rerank.assert_not_called()

        # Returns first k addresses (truncated, original order)
        assert len(result) == 2
        assert result[0].location == "func_a"
        assert result[1].location == "func_b"

    def test_returns_all_when_fewer_than_k(self):
        """When addresses < k and < min_addresses, return all addresses."""
        reranker_provider = MagicMock(name="reranker")

        addresses = [_addr(location="only_one", score=0.5)]

        reranker = AddressReranker(
            reranker=reranker_provider,
            k=10,
            min_addresses=20,
        )

        result = reranker.rerank("query", addresses)

        reranker_provider.rerank.assert_not_called()
        assert len(result) == 1
        assert result[0].location == "only_one"

    def test_handles_reranker_error_gracefully(self):
        """When reranker raises, returns first k from original list."""
        reranker_provider = MagicMock(name="reranker")
        reranker_provider.rerank.side_effect = RuntimeError("API timeout")

        addresses = [_addr(location=f"func_{i}", score=0.9 - i * 0.1) for i in range(25)]

        reranker = AddressReranker(
            reranker=reranker_provider,
            k=5,
            min_addresses=20,
        )

        result = reranker.rerank("query", addresses)

        # Reranker was called (and failed)
        reranker_provider.rerank.assert_called_once()

        # Falls back to first k addresses
        assert len(result) == 5
        assert result[0].location == "func_0"
        assert result[4].location == "func_4"

    def test_rerank_uses_summary_or_location_as_document(self):
        """Reranker receives summary text; falls back to location if no summary."""
        reranker_provider = MagicMock(name="reranker")
        reranker_provider.rerank.return_value = [
            _make_rerank_result(index=0, score=0.9),
        ]

        addr_with_summary = _addr(location="mod.func", summary="Handles user auth")
        addr_no_summary = Address(
            kind=AddressKind.SYMBOL,
            source_id="src",
            location="mod.other",
            summary="",
            score=0.5,
        )

        addresses = [addr_with_summary, addr_no_summary]

        reranker = AddressReranker(
            reranker=reranker_provider,
            k=1,
            min_addresses=2,
        )

        reranker.rerank("query", addresses)

        call_args = reranker_provider.rerank.call_args
        documents = call_args[0][1]

        # First document uses summary, second falls back to location
        assert documents[0] == "Handles user auth"
        assert documents[1] == "mod.other"

    def test_rerank_includes_agentic_file_text(self):
        """File-level agentic candidates include their loaded text for reranking."""
        reranker_provider = MagicMock(name="reranker")
        reranker_provider.rerank.return_value = [
            _make_rerank_result(index=0, score=0.9),
        ]
        address = Address(
            kind=AddressKind.FILE,
            source_id="doc",
            location="product_roadmap.md",
            summary="Document: Roadmap, Metrics",
            score=0.5,
            metadata={"text": "Launch plan and customer feedback trends."},
        )

        reranker = AddressReranker(
            reranker=reranker_provider,
            k=1,
            min_addresses=1,
        )

        reranker.rerank("What are the key facts?", [address])

        documents = reranker_provider.rerank.call_args.args[1]
        assert documents == [
            "Document: Roadmap, Metrics\nLaunch plan and customer feedback trends."
        ]

    def test_rerank_uses_query_relevant_window_from_late_file_text(self):
        """Late literal evidence must reach the bounded cross-encoder input."""
        reranker_provider = MagicMock(name="reranker")
        reranker_provider.rerank.return_value = [
            _make_rerank_result(index=0, score=0.9),
        ]
        late_evidence = "RUN_9 final verdict is FAIL with ERR_RETAINED."
        address = Address(
            kind=AddressKind.FILE,
            source_id="report",
            location="long_report.txt",
            summary="Long test report",
            score=0.5,
            metadata={
                "text": ("Routine setup: no final verdict is recorded. " * 120) + late_evidence
            },
        )

        reranker = AddressReranker(
            reranker=reranker_provider,
            k=1,
            min_addresses=1,
        )
        reranker.rerank("What is the final verdict for RUN_9?", [address])

        document = reranker_provider.rerank.call_args.args[1][0]
        assert late_evidence in document
        assert len(document) <= len(address.summary) + 1 + 1200

    def test_cold_long_section_does_not_repeat_its_fallback_prefix(self):
        """Cold sections use their heading plus excerpt as the scoring surface."""
        reranker_provider = MagicMock(name="reranker")
        reranker_provider.rerank.return_value = [
            _make_rerank_result(index=0, score=0.9),
        ]
        address = Address(
            kind=AddressKind.SECTION,
            source_id="report",
            location="Results > Retained seal",
            summary="Retained seal: " + ("routine setup " * 25),
            score=0.5,
            metadata={
                "rerank_heading": "Results > Retained seal",
                "rerank_text": "RUN_77 retained seal is KAPPA-END.",
            },
        )
        reranker = AddressReranker(
            reranker=reranker_provider,
            k=1,
            min_addresses=1,
        )

        reranker.rerank("What is the retained seal for RUN_77?", [address])

        expected_document = "Results > Retained seal\nRUN_77 retained seal is KAPPA-END."
        assert reranker_provider.rerank.call_args.args[1] == [expected_document]
        assert "documents" not in reranker.last_trace
        assert "input" not in reranker.last_trace
        assert reranker.last_trace["document_characters"] == [len(expected_document)]

    def test_reranked_addresses_preserve_kind_and_metadata(self):
        """Reranked addresses retain kind, source_id, location, summary, metadata."""
        reranker_provider = MagicMock(name="reranker")

        original = Address(
            kind=AddressKind.SECTION,
            source_id="doc.md",
            location="Setup Guide",
            summary="How to set up the project",
            score=0.3,
            metadata={"section_id": "s1", "level": 2},
        )

        reranker_provider.rerank.return_value = [
            _make_rerank_result(index=0, score=0.99),
        ]

        # Need enough addresses to trigger reranking
        filler = [_addr(location=f"filler_{i}") for i in range(19)]

        reranker = AddressReranker(
            reranker=reranker_provider,
            k=1,
            min_addresses=20,
        )

        result = reranker.rerank("setup", [original] + filler)

        assert len(result) == 1
        assert result[0].kind == AddressKind.SECTION
        assert result[0].source_id == "doc.md"
        assert result[0].location == "Setup Guide"
        assert result[0].summary == "How to set up the project"
        assert result[0].score == 0.99
        assert result[0].metadata == {"section_id": "s1", "level": 2}

    def test_empty_addresses_returns_empty(self):
        """Reranking an empty list returns an empty list."""
        reranker_provider = MagicMock(name="reranker")

        reranker = AddressReranker(
            reranker=reranker_provider,
            k=5,
            min_addresses=3,
        )

        result = reranker.rerank("query", [])

        reranker_provider.rerank.assert_not_called()
        assert result == []
