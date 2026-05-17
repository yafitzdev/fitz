# tests/unit/property/test_expansion_detector.py
"""
Property-based tests for expand_terms().

Tests pure, deterministic properties of query term expansion logic.
Target: fitz_sage/retrieval/detection/detectors/expansion.py
"""

import pytest
from hypothesis import given, settings

from fitz_sage.retrieval.detection.detectors.expansion import (
    ACRONYMS,
    SYNONYMS,
    expand_terms,
)

from .strategies import (
    non_empty_text,
    query_text,
    query_with_acronym,
    query_with_synonym,
)

pytestmark = pytest.mark.property


class TestExpandTermsIdempotence:
    """Test that expand_terms() returns identical results on repeated calls."""

    @given(query=query_text())
    def test_expand_terms_idempotent(self, query: str):
        """expand_terms(query) returns identical results on repeated calls."""
        result1 = expand_terms(query)
        result2 = expand_terms(query)
        assert result1 == result2

    @given(query=non_empty_text(min_size=1, max_size=200))
    def test_expand_terms_idempotent_arbitrary_text(self, query: str):
        """expand_terms() is idempotent with arbitrary text."""
        assert expand_terms(query) == expand_terms(query)


class TestExpandTermsReturnType:
    """Test that expand_terms() always returns a list of strings."""

    @given(query=query_text())
    def test_returns_list(self, query: str):
        """expand_terms() always returns a list."""
        result = expand_terms(query)
        assert isinstance(result, list)

    @given(query=non_empty_text(min_size=1, max_size=200))
    def test_all_elements_are_strings(self, query: str):
        """All elements in the result are strings."""
        result = expand_terms(query)
        for term in result:
            assert isinstance(term, str)


class TestExpandTermsNoDuplicates:
    """Test that expand_terms() produces no duplicate terms."""

    @given(query=query_text())
    def test_no_duplicate_terms(self, query: str):
        """expand_terms() never returns duplicate terms."""
        result = expand_terms(query)
        assert len(result) == len(set(result))

    @given(query=query_with_synonym())
    def test_no_duplicate_terms_with_synonyms(self, query: str):
        """No duplicates even when synonyms are present."""
        result = expand_terms(query)
        assert len(result) == len(set(result))

    @given(query=query_with_acronym())
    def test_no_duplicate_terms_with_acronyms(self, query: str):
        """No duplicates even when acronyms are present."""
        result = expand_terms(query)
        assert len(result) == len(set(result))


class TestExpandTermsSynonyms:
    """Test that expand_terms() correctly expands known synonyms."""

    def test_known_synonym_produces_terms(self):
        """A query with a known synonym word returns expansion terms."""
        result = expand_terms("delete the file")
        assert len(result) > 0
        # "delete" maps to ["remove", "erase"]
        assert any(t in result for t in ["remove", "erase"])

    def test_all_synonyms_produce_terms(self):
        """Every word in SYNONYMS produces at least one expansion term."""
        for word in SYNONYMS:
            result = expand_terms(word)
            assert len(result) >= 1, f"'{word}' should produce expansion terms"

    @given(query=query_with_synonym())
    def test_synonym_query_produces_terms(self, query: str):
        """A query containing a known synonym word produces at least one term."""
        # The query_with_synonym strategy guarantees a synonym word is present
        result = expand_terms(query)
        assert len(result) >= 1


class TestExpandTermsAcronyms:
    """Test that expand_terms() correctly expands known acronyms."""

    def test_known_acronym_produces_expansion(self):
        """A query with a known acronym returns its expansion."""
        result = expand_terms("api endpoint")
        assert len(result) > 0
        # "api" maps to "application programming interface" in ACRONYMS
        assert "application programming interface" in result

    def test_all_acronyms_produce_terms(self):
        """Every acronym in ACRONYMS produces at least one expansion term."""
        for acronym in ACRONYMS:
            result = expand_terms(acronym)
            assert len(result) >= 1, f"'{acronym}' should produce expansion terms"

    @given(query=query_with_acronym())
    def test_acronym_query_produces_terms(self, query: str):
        """A query containing a known acronym produces at least one term."""
        result = expand_terms(query)
        assert len(result) >= 1


class TestExpandTermsUnknownWords:
    """Test that expand_terms() returns empty list for unknown words."""

    @given(query=query_text())
    @settings(max_examples=50)
    def test_unknown_words_produce_no_terms(self, query: str):
        """Words not in SYNONYMS or ACRONYMS produce no expansion terms."""
        words = set(query.lower().split())
        has_known_word = any(w in SYNONYMS or w in ACRONYMS for w in words)

        result = expand_terms(query)

        if not has_known_word:
            assert result == [], f"No known words but got terms for: {query!r}"

    def test_completely_unknown_query_returns_empty(self):
        """A query with no dictionary words returns an empty list."""
        result = expand_terms("xyzzy qwerty frobnicate")
        assert result == []


class TestExpandTermsDictionaryConsistency:
    """Test that expand_terms() uses SYNONYMS and ACRONYMS correctly."""

    def test_synonym_terms_are_from_dictionary(self):
        """Synonym expansion terms are values from the SYNONYMS dict."""
        result = expand_terms("delete")
        # All returned terms for "delete" should be from its synonym list
        for term in result:
            # Could be synonym expansion or acronym expansion
            assert any(term in vals for vals in SYNONYMS.values()) or term in ACRONYMS.values()

    def test_acronym_terms_are_from_dictionary(self):
        """Acronym expansion terms are values from the ACRONYMS dict."""
        result = expand_terms("db")
        assert "database" in result or "datastore" in result
