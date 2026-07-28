import json
from unittest.mock import MagicMock

import pytest

from fitz_sage.core.exceptions import EnrichmentError
from fitz_sage.engines.fitz_krag.ingestion.enricher import KragEnricher


def _symbol() -> dict:
    return {
        "id": "s1",
        "name": "PaymentGateway",
        "kind": "class",
        "metadata": {},
    }


def test_symbol_enrichment_extracts_entities_and_temporal_metadata() -> None:
    chat = MagicMock()
    chat.chat.return_value = json.dumps(
        [
            {
                "entities": [{"name": "PaymentGateway", "type": "component"}],
                "temporal": {"dates": [], "versions": ["v2.4"], "refs": []},
            }
        ]
    )
    symbol = _symbol()

    KragEnricher(chat, batch_size=1).enrich_symbol_entities([symbol])

    assert symbol["entities"] == [{"name": "PaymentGateway", "type": "component"}]
    assert symbol["metadata"]["temporal"]["versions"] == ["v2.4"]
    prompt = chat.chat.call_args.args[0][0]["content"]
    assert "entities and temporal references" in prompt
    assert "keywords" not in prompt


def test_model_entities_are_merged_with_exact_identifiers() -> None:
    chat = MagicMock()
    chat.chat.return_value = json.dumps([{"entities": [], "temporal": {}}])
    symbol = _symbol()
    symbol["name"] = "ATX-123"

    KragEnricher(chat, batch_size=1).enrich_symbol_entities([symbol])

    assert symbol["entities"] == [{"name": "ATX-123", "type": "identifier"}]


def test_identifier_extraction_does_not_normalize_separators() -> None:
    sections = [
        {"title": "ATX-123", "content": "ATX_123 and ATX 123", "entities": []}
    ]

    KragEnricher(MagicMock()).derive_section_entities(sections)

    names = [entity["name"] for entity in sections[0]["entities"]]
    assert "ATX-123" in names
    assert "ATX_123" in names
    assert "ATX123" not in names


def test_batches_preserve_item_order() -> None:
    chat = MagicMock()
    chat.chat.side_effect = [
        json.dumps([{"entities": [{"name": "A", "type": "x"}], "temporal": {}}]),
        json.dumps([{"entities": [{"name": "B", "type": "x"}], "temporal": {}}]),
    ]
    symbols = [_symbol(), {**_symbol(), "id": "s2", "name": "Other"}]

    KragEnricher(chat, batch_size=1).enrich_symbol_entities(symbols)

    assert symbols[0]["entities"][0]["name"] == "A"
    assert symbols[1]["entities"][0]["name"] == "B"


def test_invalid_json_is_retried_once() -> None:
    chat = MagicMock()
    chat.chat.side_effect = [
        "not json",
        json.dumps([{"entities": [{"name": "Retry", "type": "component"}]}]),
    ]
    symbol = _symbol()

    KragEnricher(chat, batch_size=1).enrich_symbol_entities([symbol])

    assert chat.chat.call_count == 2
    assert symbol["entities"][0]["name"] == "Retry"


def test_two_invalid_responses_raise_enrichment_error() -> None:
    chat = MagicMock()
    chat.chat.side_effect = ["bad", "still bad"]

    with pytest.raises(EnrichmentError):
        KragEnricher(chat, batch_size=1).enrich_symbol_entities([_symbol()])
