# tests/unit/test_vocabulary.py
"""
Tests for fitz_sage.retrieval.vocabulary module.

Tests cover:
1. Keyword model - serialization, matching, variations
2. Variation generation - category-specific expansions
3. KeywordDetector - pattern-based detection from chunks
4. VocabularyStore - persistence, merging, CRUD operations
5. KeywordMatcher - query matching, chunk filtering
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from fitz_sage.retrieval.vocabulary.detector import (
    DetectorPattern,
    KeywordDetector,
    suggest_keywords,
)
from fitz_sage.retrieval.vocabulary.matcher import (
    KeywordMatcher,
    create_matcher_from_store,
)
from fitz_sage.retrieval.vocabulary.models import (
    Keyword,
    VocabularyConfig,
    VocabularyMetadata,
)
from fitz_sage.retrieval.vocabulary.store import VocabularyStore
from fitz_sage.retrieval.vocabulary.variations import (
    generate_variations,
    normalize_for_matching,
)

# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@dataclass
class MockChunk:
    """Mock chunk for testing."""

    content: str
    doc_id: str = "test_doc"
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def make_keyword(
    keyword_id: str,
    category: str = "testcase",
    match: list[str] | None = None,
    occurrences: int = 1,
    user_defined: bool = False,
) -> Keyword:
    """Helper to create a Keyword."""
    return Keyword(
        id=keyword_id,
        category=category,
        match=match or [keyword_id],
        occurrences=occurrences,
        user_defined=user_defined,
    )


# ---------------------------------------------------------------------------
# Tests for models.py
# ---------------------------------------------------------------------------


class TestKeyword:
    """Tests for Keyword dataclass."""

    def test_keyword_creation(self):
        """Test basic keyword creation."""
        kw = Keyword(
            id="TC-1001",
            category="testcase",
            match=["TC-1001", "tc-1001", "TC_1001"],
            occurrences=5,
        )

        assert kw.id == "TC-1001"
        assert kw.category == "testcase"
        assert len(kw.match) == 3
        assert kw.occurrences == 5
        assert not kw.user_defined

    def test_to_dict(self):
        """Test serialization to dict."""
        kw = Keyword(
            id="TC-1001",
            category="testcase",
            match=["TC-1001", "tc-1001"],
            occurrences=3,
            first_seen="doc.md",
            user_defined=True,
            auto_generated=["TC-1001"],
        )

        data = kw.to_dict()

        assert data["id"] == "TC-1001"
        assert data["category"] == "testcase"
        assert data["match"] == ["TC-1001", "tc-1001"]
        assert data["occurrences"] == 3
        assert data["first_seen"] == "doc.md"
        assert data["user_defined"] is True
        assert data["_auto_generated"] == ["TC-1001"]

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "id": "JIRA-123",
            "category": "ticket",
            "match": ["JIRA-123", "jira-123"],
            "occurrences": 10,
            "first_seen": "report.md",
            "user_defined": False,
            "_auto_generated": ["JIRA-123"],
        }

        kw = Keyword.from_dict(data)

        assert kw.id == "JIRA-123"
        assert kw.category == "ticket"
        assert kw.match == ["JIRA-123", "jira-123"]
        assert kw.occurrences == 10
        assert kw.first_seen == "report.md"
        assert not kw.user_defined
        assert kw.auto_generated == ["JIRA-123"]

    def test_from_dict_minimal(self):
        """Test deserialization with minimal data."""
        data = {"id": "v1.0.0", "category": "version"}

        kw = Keyword.from_dict(data)

        assert kw.id == "v1.0.0"
        assert kw.category == "version"
        assert kw.match == []
        assert kw.occurrences == 1
        assert kw.first_seen is None
        assert not kw.user_defined

    def test_add_variation(self):
        """Test adding a variation."""
        kw = make_keyword("TC-1001", match=["TC-1001"])

        kw.add_variation("tc_1001")
        assert "tc_1001" in kw.match

        # Adding duplicate should not add again
        kw.add_variation("tc_1001")
        assert kw.match.count("tc_1001") == 1

    def test_matches_text(self):
        """Test text matching."""
        kw = Keyword(
            id="TC-1001",
            category="testcase",
            match=["TC-1001", "tc-1001", "TC_1001"],
        )

        assert kw.matches_text("This is about TC-1001")
        assert kw.matches_text("Check tc-1001 status")
        assert kw.matches_text("TC_1001 passed")
        assert not kw.matches_text("This is about TC-1002")

    def test_matches_text_case_insensitive(self):
        """Test case-insensitive matching."""
        kw = Keyword(id="TC-1001", category="testcase", match=["TC-1001"])

        assert kw.matches_text("tc-1001 is failing")
        assert kw.matches_text("TC-1001 is passing")


class TestVocabularyConfig:
    """Tests for VocabularyConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VocabularyConfig()

        assert config.enabled is True
        assert config.detect_on_ingest is True
        assert config.min_occurrences == 1
        assert "testcase" in config.categories
        assert "ticket" in config.categories
        assert "version" in config.categories

    def test_custom_config(self):
        """Test custom configuration."""
        config = VocabularyConfig(
            enabled=False,
            min_occurrences=3,
            categories=["testcase", "ticket"],
        )

        assert config.enabled is False
        assert config.min_occurrences == 3
        assert len(config.categories) == 2


class TestVocabularyMetadata:
    """Tests for VocabularyMetadata dataclass."""

    def test_default_metadata(self):
        """Test default metadata."""
        meta = VocabularyMetadata()

        assert meta.source_docs == 0
        assert meta.auto_detected == 0
        assert meta.user_modified == 0
        assert isinstance(meta.generated, datetime)

    def test_to_dict(self):
        """Test serialization."""
        meta = VocabularyMetadata(
            generated=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            source_docs=10,
            auto_detected=50,
            user_modified=5,
        )

        data = meta.to_dict()

        assert "2025-01-01" in data["generated"]
        assert data["source_docs"] == 10
        assert data["auto_detected"] == 50
        assert data["user_modified"] == 5

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "generated": "2025-01-01T12:00:00+00:00",
            "source_docs": 20,
            "auto_detected": 100,
            "user_modified": 10,
        }

        meta = VocabularyMetadata.from_dict(data)

        assert meta.source_docs == 20
        assert meta.auto_detected == 100
        assert meta.user_modified == 10


# ---------------------------------------------------------------------------
# Tests for variations.py
# ---------------------------------------------------------------------------


class TestGenerateVariations:
    """Tests for generate_variations function."""

    def test_basic_variations(self):
        """Test basic separator and case variations."""
        variations = generate_variations("TC-1001", "testcase")

        assert "TC-1001" in variations  # Original
        assert "tc-1001" in variations  # Lowercase
        assert "TC_1001" in variations  # Underscore
        assert "TC1001" in variations  # No separator

    def test_testcase_variations(self):
        """Test testcase-specific variations."""
        variations = generate_variations("TC-1001", "testcase")

        assert "testcase 1001" in variations
        assert "test case 1001" in variations
        assert "tc1001" in variations
        assert "TC1001" in variations

    def test_ticket_variations(self):
        """Test ticket-specific variations."""
        variations = generate_variations("JIRA-4521", "ticket")

        assert "JIRA-4521" in variations
        assert "jira-4521" in variations
        assert "JIRA4521" in variations
        assert "jira4521" in variations
        assert "JIRA 4521" in variations

    def test_version_variations(self):
        """Test version-specific variations."""
        variations = generate_variations("v2.0.1", "version")

        assert "v2.0.1" in variations
        assert "V2.0.1" in variations
        assert "2.0.1" in variations
        assert "version 2.0.1" in variations

    def test_pull_request_variations(self):
        """Test pull request-specific variations."""
        variations = generate_variations("PR-123", "pull_request")

        assert "PR-123" in variations
        assert "PR 123" in variations
        assert "PR#123" in variations
        assert "pr-123" in variations
        assert "pull request 123" in variations

    def test_person_variations(self):
        """Test person name variations."""
        variations = generate_variations("John Smith", "person")

        assert "John Smith" in variations
        assert "john smith" in variations
        assert "J. Smith" in variations
        assert "jsmith" in variations
        assert "john.smith" in variations

    def test_no_empty_variations(self):
        """Test that empty variations are filtered out."""
        variations = generate_variations("TC-1", "testcase")

        assert "" not in variations
        assert all(v for v in variations)


class TestNormalizeForMatching:
    """Tests for normalize_for_matching function."""

    def test_lowercase(self):
        """Test lowercasing."""
        assert normalize_for_matching("TC-1001") == "tc 1001"

    def test_separator_normalization(self):
        """Test separator normalization to spaces."""
        assert normalize_for_matching("TC_1001") == "tc 1001"
        assert normalize_for_matching("TC-1001") == "tc 1001"
        assert normalize_for_matching("TC 1001") == "tc 1001"

    def test_multiple_separators(self):
        """Test multiple consecutive separators."""
        assert normalize_for_matching("TC--1001") == "tc 1001"
        assert normalize_for_matching("TC__1001") == "tc 1001"

    def test_trim(self):
        """Test whitespace trimming."""
        assert normalize_for_matching("  TC-1001  ") == "tc 1001"


# ---------------------------------------------------------------------------
# Tests for detector.py
# ---------------------------------------------------------------------------


class TestDetectorPattern:
    """Tests for DetectorPattern dataclass."""

    def test_compile(self):
        """Test pattern compilation."""
        pattern = DetectorPattern(
            pattern=r"\b(TC[_\-]?\d+)\b",
            category="testcase",
            description="Test case IDs",
        )

        compiled = pattern.compile()
        matches = compiled.findall("Check TC-1001 and TC_1002")

        assert len(matches) == 2
        assert "TC-1001" in matches
        assert "TC_1002" in matches


class TestKeywordDetector:
    """Tests for KeywordDetector class."""

    def test_detect_testcases(self):
        """Test detecting test case IDs."""
        chunks = [
            MockChunk("TC-1001 failed during testing"),
            MockChunk("TC-1002 passed successfully"),
        ]

        detector = KeywordDetector()
        keywords = detector.detect_from_chunks(chunks)

        keyword_ids = [k.id for k in keywords]
        assert "TC-1001" in keyword_ids
        assert "TC-1002" in keyword_ids

    def test_detect_tickets(self):
        """Test detecting ticket IDs."""
        chunks = [
            MockChunk("Fixed in JIRA-4521"),
            MockChunk("See also BUG-789 for details"),
        ]

        detector = KeywordDetector()
        keywords = detector.detect_from_chunks(chunks)

        keyword_ids = [k.id for k in keywords]
        assert "JIRA-4521" in keyword_ids
        assert "BUG-789" in keyword_ids

    def test_detect_versions(self):
        """Test detecting version numbers."""
        chunks = [
            MockChunk("Released in v2.0.1"),
            MockChunk("Updated to v2.0.1 from v1.9.0"),
        ]

        detector = KeywordDetector()
        keywords = detector.detect_from_chunks(chunks)

        keyword_ids = [k.id.lower() for k in keywords]
        assert "v2.0.1" in keyword_ids

    def test_detect_pull_requests(self):
        """Test detecting PR IDs."""
        chunks = [
            MockChunk("Merged in PR#145"),
            MockChunk("See PR-146 for fix"),
        ]

        detector = KeywordDetector()
        keywords = detector.detect_from_chunks(chunks)

        keyword_ids = [k.id for k in keywords]
        assert any("145" in kid for kid in keyword_ids)
        assert any("146" in kid for kid in keyword_ids)

    def test_occurrences_counted(self):
        """Test that occurrences are counted correctly."""
        # Use custom pattern to avoid double-counting from multiple patterns
        custom_pattern = DetectorPattern(
            pattern=r"\b(ITEM-\d+)\b",
            category="item",
            description="Item IDs",
        )

        chunks = [
            MockChunk("ITEM-1001 failed"),
            MockChunk("ITEM-1001 still failing"),
            MockChunk("ITEM-1001 now passing"),
        ]

        detector = KeywordDetector(patterns=[custom_pattern])
        keywords = detector.detect_from_chunks(chunks)

        item1001 = next(k for k in keywords if k.id == "ITEM-1001")
        assert item1001.occurrences == 3

    def test_min_occurrences_filter(self):
        """Test minimum occurrences filtering."""
        # Use custom pattern to avoid double-counting from multiple patterns
        custom_pattern = DetectorPattern(
            pattern=r"\b(ITEM-\d+)\b",
            category="item",
            description="Item IDs",
        )

        chunks = [
            MockChunk("ITEM-1001 mentioned once"),
            MockChunk("ITEM-1002 mentioned twice"),
            MockChunk("ITEM-1002 mentioned again"),
        ]

        detector = KeywordDetector(patterns=[custom_pattern], min_occurrences=2)
        keywords = detector.detect_from_chunks(chunks)

        keyword_ids = [k.id for k in keywords]
        assert "ITEM-1001" not in keyword_ids  # Only 1 occurrence
        assert "ITEM-1002" in keyword_ids  # 2 occurrences

    def test_first_seen_tracked(self):
        """Test that first_seen source is tracked."""
        chunks = [
            MockChunk("TC-1001 here", doc_id="first_doc.md"),
            MockChunk("TC-1001 also here", doc_id="second_doc.md"),
        ]

        detector = KeywordDetector()
        keywords = detector.detect_from_chunks(chunks)

        tc1001 = next(k for k in keywords if k.id == "TC-1001")
        assert tc1001.first_seen == "first_doc.md"

    def test_detect_from_text(self):
        """Test detecting from raw text."""
        detector = KeywordDetector()
        keywords = detector.detect_from_text(
            "TC-1001 and JIRA-123 are related",
            source="notes.md",
        )

        keyword_ids = [k.id for k in keywords]
        assert "TC-1001" in keyword_ids
        assert "JIRA-123" in keyword_ids

    def test_variations_generated(self):
        """Test that variations are generated for detected keywords."""
        chunks = [MockChunk("TC-1001 found")]

        detector = KeywordDetector()
        keywords = detector.detect_from_chunks(chunks)

        tc1001 = next(k for k in keywords if k.id == "TC-1001")
        assert len(tc1001.match) > 1  # Should have variations
        assert "tc-1001" in tc1001.match or "tc 1001" in tc1001.match

    def test_empty_chunks(self):
        """Test with empty chunks list."""
        detector = KeywordDetector()
        keywords = detector.detect_from_chunks([])

        assert keywords == []

    def test_no_matches(self):
        """Test with content that has no matches."""
        chunks = [MockChunk("Just regular text here")]

        detector = KeywordDetector()
        keywords = detector.detect_from_chunks(chunks)

        assert keywords == []

    def test_custom_patterns(self):
        """Test with custom patterns."""
        custom_pattern = DetectorPattern(
            pattern=r"\b(CUSTOM-\d+)\b",
            category="custom",
            description="Custom IDs",
        )

        chunks = [MockChunk("See CUSTOM-999 for details")]

        detector = KeywordDetector(patterns=[custom_pattern])
        keywords = detector.detect_from_chunks(chunks)

        assert len(keywords) == 1
        assert keywords[0].id == "CUSTOM-999"
        assert keywords[0].category == "custom"


class TestSuggestKeywords:
    """Tests for suggest_keywords function."""

    def test_suggest_with_min_occurrences(self):
        """Test suggesting keywords with minimum occurrences."""
        # Note: suggest_keywords uses default patterns which may match
        # the same content multiple times. Using tickets that won't
        # trigger overlapping patterns.
        chunks = [
            MockChunk("JIRA-1001 once"),
            MockChunk("JIRA-1002 twice"),
            MockChunk("JIRA-1002 again"),
        ]

        suggestions = suggest_keywords(chunks, min_occurrences=2)

        keyword_ids = [k.id for k in suggestions]
        assert "JIRA-1002" in keyword_ids
        assert "JIRA-1001" not in keyword_ids


# ---------------------------------------------------------------------------
# Tests for store.py
# ---------------------------------------------------------------------------


class TestVocabularyStore:
    """Tests for VocabularyStore class (PostgreSQL-based)."""

    def test_save_and_load(self):
        """Test basic save and load."""
        store = VocabularyStore(collection="test_save_and_load")
        store.clear()  # Ensure clean state

        keywords = [
            make_keyword("TC-1001", match=["TC-1001", "tc-1001"]),
            make_keyword("JIRA-123", category="ticket", match=["JIRA-123"]),
        ]

        store.save(keywords)
        loaded = store.load()

        assert len(loaded) == 2
        # PostgreSQL sorts by category, id so order may differ
        ids = {k.id for k in loaded}
        assert "TC-1001" in ids
        assert "JIRA-123" in ids

    def test_exists(self):
        """Test exists check."""
        store = VocabularyStore(collection="test_exists")
        store.clear()  # Ensure clean state

        assert not store.exists()

        store.save([make_keyword("TC-1001")])

        assert store.exists()

    def test_load_nonexistent(self):
        """Test loading from empty collection."""
        store = VocabularyStore(collection="test_load_nonexistent")
        store.clear()  # Ensure clean state

        keywords = store.load()

        assert keywords == []

    def test_load_with_metadata(self):
        """Test loading with metadata."""
        store = VocabularyStore(collection="test_load_with_metadata")
        store.clear()  # Ensure clean state

        keywords = [make_keyword("TC-1001")]
        metadata = VocabularyMetadata(source_docs=10, auto_detected=5)

        store.save(keywords, metadata)
        loaded_kw, loaded_meta = store.load_with_metadata()

        assert len(loaded_kw) == 1
        assert loaded_meta is not None
        assert loaded_meta.source_docs == 10
        assert loaded_meta.auto_detected == 5

    def test_merge_and_save_new_keywords(self):
        """Test merging new keywords."""
        store = VocabularyStore(collection="test_merge_and_save_new")
        store.clear()  # Ensure clean state

        # Initial save
        store.save([make_keyword("TC-1001", match=["TC-1001"])])

        # Merge new
        new_keywords = [
            make_keyword("TC-1002", match=["TC-1002"]),
        ]

        merged = store.merge_and_save(new_keywords, source_docs=2)

        assert len(merged) == 1  # Only new keywords in merge result
        loaded = store.load()
        # TC-1001 was not re-detected, so it's removed unless user_defined
        assert any(k.id == "TC-1002" for k in loaded)

    def test_merge_preserves_user_defined(self):
        """Test that user-defined keywords are preserved."""
        store = VocabularyStore(collection="test_merge_preserves_user")
        store.clear()  # Ensure clean state

        # Save user-defined keyword
        user_kw = make_keyword("CUSTOM-001", user_defined=True)
        store.save([user_kw])

        # Merge with new auto-detected (doesn't include CUSTOM-001)
        new_keywords = [make_keyword("TC-1001")]

        store.merge_and_save(new_keywords, source_docs=1)
        loaded = store.load()

        # User-defined should be preserved
        assert any(k.id == "CUSTOM-001" for k in loaded)
        assert any(k.id == "TC-1001" for k in loaded)

    def test_merge_preserves_user_variations(self):
        """Test that user-added variations are preserved."""
        store = VocabularyStore(collection="test_merge_preserves_vars")
        store.clear()  # Ensure clean state

        # Save with auto-generated variations
        kw = Keyword(
            id="TC-1001",
            category="testcase",
            match=["TC-1001", "tc-1001", "user-added-variation"],
            auto_generated=["TC-1001", "tc-1001"],
        )
        store.save([kw])

        # Merge with re-detected keyword
        new_kw = Keyword(
            id="TC-1001",
            category="testcase",
            match=["TC-1001", "tc-1001", "new-auto"],
            auto_generated=["TC-1001", "tc-1001", "new-auto"],
        )

        store.merge_and_save([new_kw], source_docs=1)
        loaded = store.load()

        tc1001 = next(k for k in loaded if k.id == "TC-1001")
        # User variation should be preserved
        assert "user-added-variation" in tc1001.match

    def test_add_keyword(self):
        """Test adding a single keyword."""
        store = VocabularyStore(collection="test_add_keyword")
        store.clear()  # Ensure clean state

        store.add_keyword(make_keyword("TC-1001"))

        loaded = store.load()
        assert len(loaded) == 1
        assert loaded[0].id == "TC-1001"
        assert loaded[0].user_defined is True

    def test_add_keyword_duplicate(self):
        """Test adding duplicate keyword (should not add)."""
        store = VocabularyStore(collection="test_add_keyword_dup")
        store.clear()  # Ensure clean state
        store.save([make_keyword("TC-1001")])

        store.add_keyword(make_keyword("TC-1001"))

        loaded = store.load()
        assert len(loaded) == 1

    def test_add_variation(self):
        """Test adding variation to existing keyword."""
        store = VocabularyStore(collection="test_add_variation")
        store.clear()  # Ensure clean state
        store.save([make_keyword("TC-1001", match=["TC-1001"])])

        result = store.add_variation("TC-1001", "testcase-1001")

        assert result is True
        loaded = store.load()
        assert "testcase-1001" in loaded[0].match

    def test_add_variation_not_found(self):
        """Test adding variation to nonexistent keyword."""
        store = VocabularyStore(collection="test_add_var_notfound")
        store.clear()  # Ensure clean state

        result = store.add_variation("TC-9999", "variation")

        assert result is False

    def test_remove_keyword(self):
        """Test removing a keyword."""
        store = VocabularyStore(collection="test_remove_keyword")
        store.clear()  # Ensure clean state
        store.save(
            [
                make_keyword("TC-1001"),
                make_keyword("TC-1002"),
            ]
        )

        result = store.remove_keyword("TC-1001")

        assert result is True
        loaded = store.load()
        assert len(loaded) == 1
        assert loaded[0].id == "TC-1002"

    def test_remove_keyword_not_found(self):
        """Test removing nonexistent keyword."""
        store = VocabularyStore(collection="test_remove_kw_notfound")
        store.clear()  # Ensure clean state
        store.save([make_keyword("TC-1001")])

        result = store.remove_keyword("TC-9999")

        assert result is False

    def test_get_by_category(self):
        """Test filtering by category."""
        store = VocabularyStore(collection="test_get_by_category")
        store.clear()  # Ensure clean state
        store.save(
            [
                make_keyword("TC-1001", category="testcase"),
                make_keyword("JIRA-123", category="ticket"),
                make_keyword("TC-1002", category="testcase"),
            ]
        )

        testcases = store.get_by_category("testcase")

        assert len(testcases) == 2
        assert all(k.category == "testcase" for k in testcases)

    def test_get_categories(self):
        """Test getting all categories."""
        store = VocabularyStore(collection="test_get_categories")
        store.clear()  # Ensure clean state
        store.save(
            [
                make_keyword("TC-1001", category="testcase"),
                make_keyword("JIRA-123", category="ticket"),
                make_keyword("v1.0.0", category="version"),
            ]
        )

        categories = store.get_categories()

        assert "testcase" in categories
        assert "ticket" in categories
        assert "version" in categories

    def test_collection_isolation(self):
        """Test that collections are isolated from each other."""
        store1 = VocabularyStore(collection="test_isolation_1")
        store2 = VocabularyStore(collection="test_isolation_2")
        store1.clear()
        store2.clear()

        store1.save([make_keyword("TC-1001")])
        store2.save([make_keyword("JIRA-123", category="ticket")])

        loaded1 = store1.load()
        loaded2 = store2.load()

        assert len(loaded1) == 1
        assert loaded1[0].id == "TC-1001"
        assert len(loaded2) == 1
        assert loaded2[0].id == "JIRA-123"


# ---------------------------------------------------------------------------
# Tests for matcher.py
# ---------------------------------------------------------------------------


class TestKeywordMatcher:
    """Tests for KeywordMatcher class."""

    def test_find_in_query_single(self):
        """Test finding a single keyword in query."""
        keywords = [
            Keyword(id="TC-1001", category="testcase", match=["TC-1001", "tc-1001"]),
        ]

        matcher = KeywordMatcher(keywords)
        found = matcher.find_in_query("What happened with TC-1001?")

        assert len(found) == 1
        assert found[0].id == "TC-1001"

    def test_find_in_query_multiple(self):
        """Test finding multiple keywords in query."""
        keywords = [
            Keyword(id="TC-1001", category="testcase", match=["TC-1001"]),
            Keyword(id="JIRA-123", category="ticket", match=["JIRA-123"]),
        ]

        matcher = KeywordMatcher(keywords)
        found = matcher.find_in_query("Is TC-1001 related to JIRA-123?")

        assert len(found) == 2
        keyword_ids = [k.id for k in found]
        assert "TC-1001" in keyword_ids
        assert "JIRA-123" in keyword_ids

    def test_find_in_query_variation(self):
        """Test finding keyword via variation."""
        keywords = [
            Keyword(
                id="TC-1001",
                category="testcase",
                match=["TC-1001", "tc 1001", "testcase 1001"],
            ),
        ]

        matcher = KeywordMatcher(keywords)

        found = matcher.find_in_query("What is testcase 1001?")
        assert len(found) == 1
        assert found[0].id == "TC-1001"

    def test_find_in_query_case_insensitive(self):
        """Test case-insensitive matching."""
        keywords = [
            Keyword(id="TC-1001", category="testcase", match=["TC-1001"]),
        ]

        matcher = KeywordMatcher(keywords)
        found = matcher.find_in_query("what about tc-1001?")

        assert len(found) == 1

    def test_find_in_query_no_match(self):
        """Test query with no matching keywords."""
        keywords = [
            Keyword(id="TC-1001", category="testcase", match=["TC-1001"]),
        ]

        matcher = KeywordMatcher(keywords)
        found = matcher.find_in_query("What is the weather?")

        assert len(found) == 0

    def test_filter_chunks_with_keyword(self):
        """Test filtering chunks by keyword."""
        keywords = [
            Keyword(id="TC-1001", category="testcase", match=["TC-1001", "tc-1001"]),
        ]

        chunks = [
            MockChunk("TC-1001 failed"),
            MockChunk("TC-1002 passed"),
            MockChunk("No test case here"),
        ]

        matcher = KeywordMatcher(keywords)
        filtered = matcher.filter_chunks("What happened with TC-1001?", chunks)

        assert len(filtered) == 1
        assert "TC-1001" in filtered[0].content

    def test_filter_chunks_no_keyword_in_query(self):
        """Test that all chunks returned when no keyword in query."""
        keywords = [
            Keyword(id="TC-1001", category="testcase", match=["TC-1001"]),
        ]

        chunks = [
            MockChunk("Some content"),
            MockChunk("More content"),
        ]

        matcher = KeywordMatcher(keywords)
        filtered = matcher.filter_chunks("General question", chunks)

        assert len(filtered) == 2  # All chunks returned

    def test_filter_chunks_multiple_keywords(self):
        """Test filtering with multiple keywords (AND logic)."""
        keywords = [
            Keyword(id="TC-1001", category="testcase", match=["TC-1001"]),
            Keyword(id="JIRA-123", category="ticket", match=["JIRA-123"]),
        ]

        chunks = [
            MockChunk("TC-1001 is related to JIRA-123"),  # Has both
            MockChunk("TC-1001 standalone"),  # Has only TC-1001
            MockChunk("JIRA-123 standalone"),  # Has only JIRA-123
            MockChunk("Unrelated content"),  # Has neither
        ]

        matcher = KeywordMatcher(keywords)
        filtered = matcher.filter_chunks(
            "How are TC-1001 and JIRA-123 related?",
            chunks,
        )

        assert len(filtered) == 1
        assert "TC-1001" in filtered[0].content
        assert "JIRA-123" in filtered[0].content

    def test_chunk_matches_any(self):
        """Test checking if chunk matches any keyword."""
        keywords = [
            Keyword(id="TC-1001", category="testcase", match=["TC-1001"]),
            Keyword(id="TC-1002", category="testcase", match=["TC-1002"]),
        ]

        matcher = KeywordMatcher(keywords)

        assert matcher.chunk_matches_any(MockChunk("TC-1001 here"))
        assert matcher.chunk_matches_any(MockChunk("TC-1002 here"))
        assert not matcher.chunk_matches_any(MockChunk("TC-1003 here"))

    def test_get_matching_keywords(self):
        """Test getting all keywords that match in a chunk."""
        keywords = [
            Keyword(id="TC-1001", category="testcase", match=["TC-1001"]),
            Keyword(id="TC-1002", category="testcase", match=["TC-1002"]),
            Keyword(id="JIRA-123", category="ticket", match=["JIRA-123"]),
        ]

        matcher = KeywordMatcher(keywords)
        chunk = MockChunk("TC-1001 and TC-1002 are affected")

        matched = matcher.get_matching_keywords(chunk)

        assert len(matched) == 2
        keyword_ids = [k.id for k in matched]
        assert "TC-1001" in keyword_ids
        assert "TC-1002" in keyword_ids
        assert "JIRA-123" not in keyword_ids

    def test_empty_keywords(self):
        """Test matcher with no keywords."""
        matcher = KeywordMatcher([])

        found = matcher.find_in_query("TC-1001 question")
        assert found == []

        chunks = [MockChunk("TC-1001 content")]
        filtered = matcher.filter_chunks("TC-1001?", chunks)
        assert filtered == chunks  # All returned when no keywords


class TestCreateMatcherFromStore:
    """Tests for create_matcher_from_store function."""

    def test_create_from_existing_store(self, tmp_path: Path, monkeypatch):
        """Test creating matcher from existing store."""
        from fitz_sage.core import paths

        monkeypatch.setattr(paths.FitzPaths, "workspace", classmethod(lambda cls: tmp_path))

        # Create store with keywords
        store = VocabularyStore(collection="test_collection")
        store.save(
            [
                Keyword(id="TC-1001", category="testcase", match=["TC-1001"]),
            ]
        )

        matcher = create_matcher_from_store(collection="test_collection")

        assert matcher is not None
        found = matcher.find_in_query("TC-1001?")
        assert len(found) == 1

    def test_create_from_nonexistent_store(self, tmp_path: Path, monkeypatch):
        """Test that None is returned when store doesn't exist."""
        from fitz_sage.core import paths

        monkeypatch.setattr(paths.FitzPaths, "workspace", classmethod(lambda cls: tmp_path))

        matcher = create_matcher_from_store(collection="nonexistent")

        assert matcher is None

    def test_create_from_empty_store(self, tmp_path: Path, monkeypatch):
        """Test that None is returned when store is empty."""
        from fitz_sage.core import paths

        monkeypatch.setattr(paths.FitzPaths, "workspace", classmethod(lambda cls: tmp_path))

        store = VocabularyStore(collection="empty_collection")
        store.save([])

        matcher = create_matcher_from_store(collection="empty_collection")

        assert matcher is None
