# tests/unit/test_krag_enricher.py
"""
Unit tests for KragEnricher.

KragEnricher: batch LLM enrichment for KRAG symbols and sections,
extracting keywords and entities in-place.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from fitz_sage.core import KnowledgeError
from fitz_sage.engines.fitz_krag.ingestion.enricher import KragEnricher

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chat(response: str | None = None, side_effect=None) -> MagicMock:
    """Create a mock ChatProvider."""
    chat = MagicMock(name="chat")
    if side_effect is not None:
        chat.chat.side_effect = side_effect
    elif response is not None:
        chat.chat.return_value = response
    return chat


def _make_enrichment_response(items: list[dict]) -> str:
    """Build a JSON array string matching the LLM enrichment response format."""
    return json.dumps(items)


def _symbol_dicts(n: int = 2) -> list[dict]:
    """Create n minimal symbol dicts."""
    return [
        {
            "name": f"func_{i}",
            "kind": "function",
            "summary": f"Does thing {i}",
        }
        for i in range(n)
    ]


def _section_dicts(n: int = 2) -> list[dict]:
    """Create n minimal section dicts."""
    return [
        {
            "title": f"Section {i}",
            "content": f"Content for section {i}",
            "summary": f"Summary of section {i}",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# TestEnrichSymbols
# ---------------------------------------------------------------------------


class TestEnrichSymbols:
    """Tests for enrich_symbols."""

    def test_adds_keywords_and_entities(self):
        """enrich_symbols adds keywords and entities to symbol dicts in-place."""
        enrichments = [
            {
                "keywords": ["auth", "login"],
                "entities": [{"name": "PostgreSQL", "type": "technology"}],
            },
            {
                "keywords": ["hash", "sha256"],
                "entities": [{"name": "SHA-256", "type": "algorithm"}],
            },
        ]
        chat = _make_chat(response=_make_enrichment_response(enrichments))
        enricher = KragEnricher(chat, batch_size=15)
        symbols = _symbol_dicts(2)

        enricher.enrich_symbols(symbols)

        assert symbols[0]["keywords"] == ["auth", "login"]
        assert symbols[0]["entities"] == [{"name": "PostgreSQL", "type": "technology"}]
        assert symbols[1]["keywords"] == ["hash", "sha256"]
        assert symbols[1]["entities"] == [{"name": "SHA-256", "type": "algorithm"}]

    def test_keyword_strategy_only_sets_keywords(self):
        """enrich_symbol_keywords runs the query-ready keyword strategy only."""
        enrichments = [{"keywords": ["auth", "login"]}]
        chat = _make_chat(response=_make_enrichment_response(enrichments))
        enricher = KragEnricher(chat, batch_size=15)
        symbols = _symbol_dicts(1)

        enricher.enrich_symbol_keywords(symbols)

        assert symbols[0]["keywords"] == ["auth", "login"]
        assert "entities" not in symbols[0]
        messages = chat.chat.call_args.args[0]
        assert "Extract retrieval keywords" in messages[0]["content"]

    def test_modifies_in_place(self):
        """enrich_symbols modifies the original list objects, not copies."""
        enrichments = [
            {"keywords": ["kw1"], "entities": []},
        ]
        chat = _make_chat(response=_make_enrichment_response(enrichments))
        enricher = KragEnricher(chat, batch_size=15)
        symbols = _symbol_dicts(1)
        original_ref = symbols[0]

        enricher.enrich_symbols(symbols)

        assert original_ref is symbols[0]
        assert original_ref["keywords"] == ["kw1"]


# ---------------------------------------------------------------------------
# TestEnrichSections
# ---------------------------------------------------------------------------


class TestEnrichSections:
    """Tests for enrich_sections."""

    def test_adds_keywords_and_entities(self):
        """enrich_sections adds keywords and entities to section dicts in-place."""
        enrichments = [
            {
                "keywords": ["introduction", "overview"],
                "entities": [{"name": "KRAG", "type": "system"}],
            },
            {
                "keywords": ["setup", "config"],
                "entities": [],
            },
        ]
        chat = _make_chat(response=_make_enrichment_response(enrichments))
        enricher = KragEnricher(chat, batch_size=15)
        sections = _section_dicts(2)

        enricher.enrich_sections(sections)

        assert sections[0]["keywords"] == ["introduction", "overview"]
        assert sections[0]["entities"] == [{"name": "KRAG", "type": "system"}]
        assert sections[1]["keywords"] == ["setup", "config"]
        assert sections[1]["entities"] == []

    def test_entity_strategy_preserves_existing_keywords(self):
        """enrich_section_entities does not erase query-ready keywords."""
        enrichments = [
            {
                "entities": [{"name": "KRAG", "type": "system"}],
                "temporal": {"dates": ["2026-06-02"], "versions": [], "refs": []},
            }
        ]
        chat = _make_chat(response=_make_enrichment_response(enrichments))
        enricher = KragEnricher(chat, batch_size=15)
        sections = _section_dicts(1)
        sections[0]["keywords"] = ["retrieval"]

        enricher.enrich_section_entities(sections)

        assert sections[0]["keywords"] == ["retrieval"]
        assert sections[0]["entities"] == [{"name": "KRAG", "type": "system"}]
        assert sections[0]["metadata"]["temporal"]["dates"] == ["2026-06-02"]

    def test_preserves_exact_identifiers_when_model_misses_them(self):
        """Ticket-like IDs are added deterministically even if the model omits them."""
        enrichments = [{"keywords": ["timeout"], "entities": []}]
        chat = _make_chat(response=_make_enrichment_response(enrichments))
        enricher = KragEnricher(chat, batch_size=15)
        sections = [
            {
                "title": "Sprint 47 Incident Notes",
                "content": "Sprint 47 failed because test case TC-4812 timed out.",
                "summary": None,
            }
        ]

        enricher.enrich_sections(sections)

        assert sections[0]["keywords"] == ["timeout", "TC-4812"]

    def test_derive_section_entities_does_not_call_llm(self):
        """Progressive doc entity linking should not run a second generation pass."""
        chat = _make_chat()
        enricher = KragEnricher(chat, batch_size=15)
        sections = [
            {
                "title": "Customer Contract",
                "content": "Acme Corp signed contract TC-1000 for the Berlin rollout.",
                "summary": None,
                "keywords": ["contract"],
            }
        ]

        enricher.derive_section_entities(sections)

        chat.chat.assert_not_called()
        assert {"name": "TC-1000", "type": "identifier"} in sections[0]["entities"]
        assert {"name": "Acme Corp", "type": "entity"} in sections[0]["entities"]
        assert sections[0]["keywords"] == ["contract"]


# ---------------------------------------------------------------------------
# TestBatchProcessing
# ---------------------------------------------------------------------------


class TestBatchProcessing:
    """Tests for batch splitting and LLM call orchestration."""

    def test_batch_size_controls_splitting(self):
        """batch_size=3 splits 7 symbols into 3 LLM calls (3+3+1)."""
        enrichments_3 = [{"keywords": [f"kw{i}"], "entities": []} for i in range(3)]
        enrichments_1 = [{"keywords": ["kw_last"], "entities": []}]
        chat = _make_chat()
        chat.chat.side_effect = [
            _make_enrichment_response(enrichments_3),
            _make_enrichment_response(enrichments_3),
            _make_enrichment_response(enrichments_1),
        ]
        enricher = KragEnricher(chat, batch_size=3)
        symbols = _symbol_dicts(7)

        enricher.enrich_symbols(symbols)

        assert chat.chat.call_count == 3
        # Verify all symbols got keywords
        assert all("keywords" in s for s in symbols)
        assert symbols[0]["keywords"] == ["kw0"]
        assert symbols[6]["keywords"] == ["kw_last"]

    def test_single_batch_when_count_less_than_batch_size(self):
        """2 items with batch_size=15 -> single LLM call."""
        enrichments = [{"keywords": ["a"], "entities": []}, {"keywords": ["b"], "entities": []}]
        chat = _make_chat(response=_make_enrichment_response(enrichments))
        enricher = KragEnricher(chat, batch_size=15)
        symbols = _symbol_dicts(2)

        enricher.enrich_symbols(symbols)

        chat.chat.assert_called_once()

    def test_prompt_treats_multiline_content_as_one_item(self):
        """Multiline content is explicitly fenced so Qwen does not split it into items."""
        chat = _make_chat(response=_make_enrichment_response([{"keywords": ["queries"]}]))
        enricher = KragEnricher(chat, batch_size=15)
        sections = [
            {
                "title": "Queries",
                "content": '"Question one?"\n"Question two?"\n"Question three?"',
                "summary": None,
            }
        ]

        enricher.enrich_sections(sections)

        messages = chat.chat.call_args.args[0]
        kwargs = chat.chat.call_args.kwargs
        assert "exactly 1 item block(s)" in messages[0]["content"]
        assert "multiple lines, bullets, sentences, or questions" in messages[0]["content"]
        assert "at most 8 keywords" in messages[0]["content"]
        assert "Never repeat a keyword or entity name" in messages[0]["content"]
        assert "v2.3" not in messages[0]["content"]
        assert "latest" not in messages[0]["content"]
        assert '<item index="1">' in messages[1]["content"]
        assert "</item>" in messages[1]["content"]
        assert kwargs["max_tokens"] >= 256
        assert kwargs["temperature"] == 0

    def test_empty_list_no_llm_call(self):
        """Empty symbol list makes no LLM calls."""
        chat = _make_chat()
        enricher = KragEnricher(chat, batch_size=15)

        enricher.enrich_symbols([])

        chat.chat.assert_not_called()


# ---------------------------------------------------------------------------
# TestGracefulFallback
# ---------------------------------------------------------------------------


class TestRequiredEnrichmentFailures:
    """Tests for enrichment failure handling."""

    def test_llm_exception_raises(self):
        """When LLM raises, enrichment fails instead of storing empty metadata."""
        chat = _make_chat(side_effect=RuntimeError("LLM unreachable"))
        enricher = KragEnricher(chat, batch_size=15)
        symbols = _symbol_dicts(2)

        with pytest.raises(KnowledgeError, match="Required enrichment batch failed"):
            enricher.enrich_symbols(symbols)

        for s in symbols:
            assert "keywords" not in s
            assert "entities" not in s

    def test_malformed_json_uses_deterministic_fallback(self):
        """When model JSON is unusable, exact grounded identifiers still index."""
        chat = _make_chat(response="This is not JSON at all")
        enricher = KragEnricher(chat, batch_size=15)
        symbols = [
            {
                "name": "OAuthClientV2",
                "kind": "class",
                "summary": "Handles TC-4812 retries for Acme Corp in v2.3.",
            }
        ]

        enricher.enrich_symbols(symbols)

        assert chat.chat.call_count == 2
        assert symbols[0]["keywords"] == ["TC-4812", "v2.3"]
        assert {"name": "TC-4812", "type": "identifier"} in symbols[0]["entities"]
        assert {"name": "v2.3", "type": "identifier"} in symbols[0]["entities"]
        assert {"name": "Acme Corp", "type": "entity"} in symbols[0]["entities"]

    def test_retries_once_when_first_response_is_invalid_json(self):
        """A malformed first response gets one strict retry before fallback."""
        good_response = _make_enrichment_response([{"keywords": ["retry"], "entities": []}])
        chat = _make_chat()
        chat.chat.side_effect = ["truncated json", good_response]
        enricher = KragEnricher(chat, batch_size=15)
        symbols = _symbol_dicts(1)

        enricher.enrich_symbols(symbols)

        assert symbols[0]["keywords"] == ["retry"]
        assert chat.chat.call_count == 2
        retry_messages = chat.chat.call_args.args[0]
        assert "Retry the same enrichment" in retry_messages[-1]["content"]

    def test_partial_batch_failure(self):
        """First batch succeeds, second fails; ingestion raises before completion."""
        good_response = _make_enrichment_response([{"keywords": ["good"], "entities": []}])
        chat = _make_chat()
        chat.chat.side_effect = [
            good_response,
            RuntimeError("timeout"),
        ]
        enricher = KragEnricher(chat, batch_size=1)
        symbols = _symbol_dicts(2)

        with pytest.raises(KnowledgeError, match="Required enrichment batch failed"):
            enricher.enrich_symbols(symbols)

        assert symbols[0]["keywords"] == ["good"]
        assert "keywords" not in symbols[1]
        assert "entities" not in symbols[1]

    def test_sections_raise_on_failure(self):
        """Sections also fail closed when LLM enrichment fails."""
        chat = _make_chat(side_effect=RuntimeError("API error"))
        enricher = KragEnricher(chat, batch_size=15)
        sections = _section_dicts(2)

        with pytest.raises(KnowledgeError, match="Required enrichment batch failed"):
            enricher.enrich_sections(sections)

    def test_response_with_code_fence(self):
        """LLM response wrapped in markdown code fence is parsed correctly."""
        enrichments = [{"keywords": ["fenced"], "entities": []}]
        response = f"```json\n{json.dumps(enrichments)}\n```"
        chat = _make_chat(response=response)
        enricher = KragEnricher(chat, batch_size=15)
        symbols = _symbol_dicts(1)

        enricher.enrich_symbols(symbols)

        assert symbols[0]["keywords"] == ["fenced"]

    def test_single_object_response_for_single_item(self):
        """Single-item batches accept a JSON object when the shape is valid."""
        response = json.dumps({"keywords": ["single"], "entities": []})
        chat = _make_chat(response=response)
        enricher = KragEnricher(chat, batch_size=15)
        symbols = _symbol_dicts(1)

        enricher.enrich_symbols(symbols)

        assert symbols[0]["keywords"] == ["single"]

    def test_truncated_single_item_response_is_salvaged(self):
        """A runaway single-item response still contributes complete model fields."""
        response = (
            '[{"keywords": ["Name", "Name", "Geburtsdatum"], "entities": ['
            '{"name": "Yan Isa Fitzner", "type": "Person"}, '
            '{"name": "Kainzenbadstraße", "type": "Address"}, '
            '{"name": "Yan Isa Fitzner", "type": "Person"}, '
            '{"name": "unterminated", "type": '
        )
        chat = _make_chat(response=response)
        enricher = KragEnricher(chat, batch_size=15)
        symbols = _symbol_dicts(1)

        enricher.enrich_symbols(symbols)

        assert symbols[0]["keywords"] == ["Name", "Geburtsdatum"]
        assert symbols[0]["entities"] == [
            {"name": "Yan Isa Fitzner", "type": "Person"},
            {"name": "Kainzenbadstraße", "type": "Address"},
        ]

    def test_truncated_keywords_array_is_salvaged(self):
        """A runaway keywords array still contributes complete model strings."""
        response = '[{"keywords": ["Energiepreis", "Energiepreis", "Stromtarif", ' '"unterminated'
        chat = _make_chat(response=response)
        enricher = KragEnricher(chat, batch_size=15)
        symbols = _symbol_dicts(1)

        enricher.enrich_symbols(symbols)

        assert symbols[0]["keywords"] == ["Energiepreis", "Stromtarif"]
        assert symbols[0]["entities"] == []

    def test_wrapped_array_response(self):
        """Provider wrappers that return an object containing results are parsed."""
        response = json.dumps({"results": [{"keywords": ["wrapped"], "entities": []}]})
        chat = _make_chat(response=response)
        enricher = KragEnricher(chat, batch_size=15)
        symbols = _symbol_dicts(1)

        enricher.enrich_symbols(symbols)

        assert symbols[0]["keywords"] == ["wrapped"]
