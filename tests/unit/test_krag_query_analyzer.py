# tests/unit/test_krag_query_analyzer.py
"""Unit tests for fitz_krag query analyzer types."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from fitz_sage.engines.fitz_krag.query_analyzer import (
    QueryAnalysis,
    QueryType,
    _parse_query_type,
)

# ---------------------------------------------------------------------------
# TestQueryType
# ---------------------------------------------------------------------------


class TestQueryType:
    """Tests for the QueryType enum."""

    def test_query_type_values(self) -> None:
        """Enum values match expected strings."""
        assert QueryType.CODE.value == "code"
        assert QueryType.DOCUMENTATION.value == "documentation"
        assert QueryType.GENERAL.value == "general"
        assert QueryType.CROSS.value == "cross"
        assert QueryType.DATA.value == "data"

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("code", QueryType.CODE),
            ("documentation", QueryType.DOCUMENTATION),
            ("general", QueryType.GENERAL),
            ("cross", QueryType.CROSS),
        ],
    )
    def test_parse_query_type_valid(self, value: str, expected: QueryType) -> None:
        """Each valid lowercase value maps to the correct enum member."""
        assert _parse_query_type(value) is expected

    def test_parse_query_type_invalid(self) -> None:
        """Unknown string falls back to GENERAL."""
        assert _parse_query_type("nonexistent") is QueryType.GENERAL
        assert _parse_query_type("") is QueryType.GENERAL
        assert _parse_query_type("foo_bar") is QueryType.GENERAL

    @pytest.mark.parametrize("raw", ["CODE", "Code", "CoDe", "DOCUMENTATION"])
    def test_parse_query_type_case(self, raw: str) -> None:
        """Parsing is case-insensitive (value.lower() inside _parse_query_type)."""
        result = _parse_query_type(raw)
        assert result is QueryType(raw.lower())


# ---------------------------------------------------------------------------
# TestQueryAnalysis
# ---------------------------------------------------------------------------


class TestQueryAnalysis:
    """Tests for the QueryAnalysis dataclass and strategy_weights property."""

    def test_strategy_weights_code(self) -> None:
        """CODE type produces code-heavy weights."""
        analysis = QueryAnalysis(primary_type=QueryType.CODE)
        assert analysis.strategy_weights == {
            "code": 0.83,
            "section": 0.11,
            "table": 0.06,
        }

    def test_strategy_weights_documentation(self) -> None:
        """DOCUMENTATION type produces section-heavy weights."""
        analysis = QueryAnalysis(primary_type=QueryType.DOCUMENTATION)
        assert analysis.strategy_weights == {
            "code": 0.11,
            "section": 0.83,
            "table": 0.06,
        }

    def test_strategy_weights_general(self) -> None:
        """GENERAL type produces balanced weights."""
        analysis = QueryAnalysis(primary_type=QueryType.GENERAL)
        assert analysis.strategy_weights == {
            "code": 0.38,
            "section": 0.39,
            "table": 0.23,
        }

    def test_strategy_weights_cross(self) -> None:
        """CROSS type produces code/section balanced weights with table."""
        analysis = QueryAnalysis(primary_type=QueryType.CROSS)
        assert analysis.strategy_weights == {
            "code": 0.44,
            "section": 0.44,
            "table": 0.12,
        }

    def test_strategy_weights_data(self) -> None:
        """DATA type produces table-heavy weights with section floor for hybrid retrieval."""
        analysis = QueryAnalysis(primary_type=QueryType.DATA)
        assert analysis.strategy_weights == {
            "code": 0.06,
            "section": 0.17,
            "table": 0.77,
        }

    def test_strategy_weights_returns_copy(self) -> None:
        """strategy_weights returns a new dict each call (not the internal one)."""
        analysis = QueryAnalysis(primary_type=QueryType.CODE)
        w1 = analysis.strategy_weights
        w2 = analysis.strategy_weights
        assert w1 == w2
        assert w1 is not w2

    def test_frozen(self) -> None:
        """QueryAnalysis is frozen -- fields cannot be reassigned."""
        analysis = QueryAnalysis(primary_type=QueryType.CODE, confidence=0.9)
        with pytest.raises(FrozenInstanceError):
            analysis.primary_type = QueryType.GENERAL  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            analysis.confidence = 0.1  # type: ignore[misc]

    def test_defaults(self) -> None:
        """Default field values are sane."""
        analysis = QueryAnalysis(primary_type=QueryType.GENERAL)
        assert analysis.secondary_type is None
        assert analysis.confidence == 0.5
        assert analysis.entities == ()
        assert analysis.refined_query == ""
