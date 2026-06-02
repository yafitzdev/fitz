# tests/unit/test_agentic_search.py
"""
Unit tests for AgenticSearchStrategy — LLM-driven file selection from manifest.

Tests BM25 pre-filter scoring, path-match fallback, address creation
for code/doc/generic files, and LLM response parsing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

from fitz_sage.engines.fitz_krag.retrieval.strategies.agentic_search import (
    _BM25_PREFILTER_THRESHOLD,
    AgenticSearchStrategy,
    _entry_to_text,
    _tokenize,
)
from fitz_sage.engines.fitz_krag.types import AddressKind

# ---------------------------------------------------------------------------
# Helpers: lightweight stand-ins for ManifestEntry, ManifestSymbol, ManifestHeading
# ---------------------------------------------------------------------------


@dataclass
class _FakeSymbol:
    name: str
    qualified_name: str
    kind: str
    signature: str | None
    start_line: int
    end_line: int


@dataclass
class _FakeHeading:
    title: str
    level: int


@dataclass
class _FakeEntry:
    file_id: str
    rel_path: str
    abs_path: str
    content_hash: str
    file_type: str
    size_bytes: int
    state: str = "registered"
    symbols: list = field(default_factory=list)
    headings: list = field(default_factory=list)
    priority: int = 4
    last_queried_at: float | None = None


def _code_entry(rel_path: str = "src/auth.py", symbols: list | None = None) -> _FakeEntry:
    syms = symbols or [
        _FakeSymbol("login", "auth.login", "function", "def login(user, pw)", 10, 25),
        _FakeSymbol("AuthManager", "auth.AuthManager", "class", "class AuthManager", 30, 80),
    ]
    return _FakeEntry(
        file_id="id-auth",
        rel_path=rel_path,
        abs_path=f"/project/{rel_path}",
        content_hash="abc123",
        file_type=".py",
        size_bytes=2048,
        symbols=syms,
    )


def _doc_entry(rel_path: str = "docs/guide.md", headings: list | None = None) -> _FakeEntry:
    hdgs = headings or [
        _FakeHeading("Getting Started", 1),
        _FakeHeading("Installation", 2),
        _FakeHeading("Configuration", 2),
    ]
    return _FakeEntry(
        file_id="id-guide",
        rel_path=rel_path,
        abs_path=f"/project/{rel_path}",
        content_hash="def456",
        file_type=".md",
        size_bytes=4096,
        headings=hdgs,
    )


def _generic_entry(rel_path: str = "data/config.yaml") -> _FakeEntry:
    return _FakeEntry(
        file_id="id-config",
        rel_path=rel_path,
        abs_path=f"/project/{rel_path}",
        content_hash="ghi789",
        file_type=".yaml",
        size_bytes=512,
    )


# ---------------------------------------------------------------------------
# Tests: _tokenize helper
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_basic(self):
        assert _tokenize("hello world") == ["hello", "world"]

    def test_splits_path_chars(self):
        tokens = _tokenize("src/auth/manager.py")
        assert "src" in tokens
        assert "auth" in tokens
        assert "manager" in tokens

    def test_drops_single_char(self):
        tokens = _tokenize("a b cc dd")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "cc" in tokens
        assert "dd" in tokens


# ---------------------------------------------------------------------------
# Tests: _entry_to_text helper
# ---------------------------------------------------------------------------


class TestEntryToText:
    def test_code_entry_includes_symbols(self):
        entry = _code_entry()
        text = _entry_to_text(entry)
        assert "login" in text
        assert "auth.login" in text
        assert "AuthManager" in text

    def test_doc_entry_includes_headings(self):
        entry = _doc_entry()
        text = _entry_to_text(entry)
        assert "Getting Started" in text
        assert "Installation" in text

    def test_always_includes_path(self):
        entry = _generic_entry()
        text = _entry_to_text(entry)
        assert entry.rel_path in text


# ---------------------------------------------------------------------------
# Tests: BM25 pre-filter
# ---------------------------------------------------------------------------


class TestBM25Prefilter:
    def _make_strategy(self) -> AgenticSearchStrategy:
        manifest = MagicMock()
        config = MagicMock()
        return AgenticSearchStrategy(
            manifest=manifest,
            source_dir=Path("/project"),
            chat_factory=MagicMock(),
            config=config,
        )

    def test_scores_rank_relevant_higher(self):
        """Entries with query-matching tokens should score higher."""
        strategy = self._make_strategy()

        relevant = _code_entry(
            "src/auth.py",
            symbols=[
                _FakeSymbol("login", "auth.login", "function", "def login(user)", 1, 10),
            ],
        )
        irrelevant = _code_entry(
            "src/math_utils.py",
            symbols=[
                _FakeSymbol("add", "math_utils.add", "function", "def add(a, b)", 1, 5),
            ],
        )

        result = strategy._bm25_prefilter([irrelevant, relevant], "authentication login")
        # Relevant entry should come first
        assert result[0].rel_path == "src/auth.py"

    def test_returns_max_threshold_entries(self):
        """Should return at most _BM25_PREFILTER_THRESHOLD entries."""
        strategy = self._make_strategy()
        entries = [_generic_entry(f"file_{i}.txt") for i in range(100)]
        result = strategy._bm25_prefilter(entries, "test query")
        assert len(result) <= _BM25_PREFILTER_THRESHOLD

    def test_empty_query_returns_first_n(self):
        """Empty query tokens should return first N entries unchanged."""
        strategy = self._make_strategy()
        entries = [_generic_entry(f"f{i}.txt") for i in range(60)]
        # "a" will be dropped (single char) — effectively empty
        result = strategy._bm25_prefilter(entries, "a")
        assert len(result) == _BM25_PREFILTER_THRESHOLD


class TestNoLlmRetrieval:
    def test_allow_llm_false_uses_bm25_without_chat(self):
        """No-chat retrieval should select from manifest with deterministic scoring."""
        entries = [_generic_entry(f"docs/config_{i}.md") for i in range(20)]
        manifest = MagicMock()
        manifest.entries.return_value = {entry.rel_path: entry for entry in entries}
        chat_factory = MagicMock(side_effect=AssertionError("chat called"))
        strategy = AgenticSearchStrategy(
            manifest=manifest,
            source_dir=Path("/project"),
            chat_factory=chat_factory,
            config=MagicMock(),
        )
        strategy._read_from_disk = MagicMock(return_value="configuration guide")

        result = strategy.retrieve("config", limit=3, allow_llm=False)

        assert len(result) == 3
        chat_factory.assert_not_called()

    def test_query_ready_files_are_not_agentic_candidates(self):
        """Once keyword indexing is usable, agentic fallback leaves the file alone."""
        not_ready = _generic_entry("docs/raw.md")
        ready = _generic_entry("docs/ready.md")
        ready.state = "query_ready"
        manifest = MagicMock()
        manifest.entries.return_value = {
            not_ready.rel_path: not_ready,
            ready.rel_path: ready,
        }
        strategy = AgenticSearchStrategy(
            manifest=manifest,
            source_dir=Path("/project"),
            chat_factory=None,
            config=MagicMock(),
        )
        strategy._read_from_disk = MagicMock(return_value="raw content")

        result = strategy.retrieve("raw ready", limit=10, allow_llm=False)

        assert len(result) == 1
        assert result[0].location == "docs/raw.md"


# ---------------------------------------------------------------------------
# Tests: path-match fallback
# ---------------------------------------------------------------------------


class TestPathMatchFallback:
    def _make_strategy(self) -> AgenticSearchStrategy:
        return AgenticSearchStrategy(
            manifest=MagicMock(),
            source_dir=Path("/project"),
            chat_factory=MagicMock(),
            config=MagicMock(),
        )

    def test_finds_matching_paths(self):
        strategy = self._make_strategy()
        entries = [
            _code_entry("src/auth/handler.py"),
            _code_entry("src/database/pool.py"),
            _code_entry("src/auth/middleware.py"),
        ]
        result = strategy._path_match_files(entries, "How does auth work?")
        assert "src/auth/handler.py" in result
        assert "src/auth/middleware.py" in result
        assert "src/database/pool.py" not in result

    def test_skips_stopwords(self):
        """Stopwords like 'what', 'the', 'how' should not trigger path matches."""
        strategy = self._make_strategy()
        entries = [_code_entry("src/what_handler.py")]
        result = strategy._path_match_files(entries, "what is this")
        # "what" and "this" are stopwords, "is" is a stopword — no valid terms
        assert result == []

    def test_skips_short_terms(self):
        """Terms with <= 2 characters should be skipped."""
        strategy = self._make_strategy()
        entries = [_code_entry("src/db.py")]
        result = strategy._path_match_files(entries, "db query")
        # "db" is only 2 chars — skipped. "query" should not match "db.py"
        assert "src/db.py" not in result


# ---------------------------------------------------------------------------
# Tests: Address creation
# ---------------------------------------------------------------------------


class TestAddressCreation:
    def _make_strategy(self) -> AgenticSearchStrategy:
        return AgenticSearchStrategy(
            manifest=MagicMock(),
            source_dir=Path("/project"),
            chat_factory=MagicMock(),
            config=MagicMock(),
        )

    def test_code_file_creates_symbol_addresses(self):
        """Code files with symbols should produce one SYMBOL address per symbol."""
        strategy = self._make_strategy()
        entry = _code_entry()
        addresses = strategy._create_addresses(entry, "def login(): pass")

        assert len(addresses) == 2
        assert all(a.kind == AddressKind.SYMBOL for a in addresses)
        assert addresses[0].metadata["name"] == "login"
        assert addresses[1].metadata["name"] == "AuthManager"
        assert all(a.metadata.get("agentic") is True for a in addresses)

    def test_symbol_address_has_line_info(self):
        """Symbol addresses should carry start_line and end_line metadata."""
        strategy = self._make_strategy()
        entry = _code_entry()
        addresses = strategy._create_addresses(entry, "code")

        assert addresses[0].metadata["start_line"] == 10
        assert addresses[0].metadata["end_line"] == 25
        assert addresses[0].location == "src/auth.py:10"

    def test_doc_file_creates_file_address(self):
        """Doc files with headings should produce one FILE address with heading summary."""
        strategy = self._make_strategy()
        entry = _doc_entry()
        content = "# Guide\nSome content"
        addresses = strategy._create_addresses(entry, content)

        assert len(addresses) == 1
        assert addresses[0].kind == AddressKind.FILE
        assert "Getting Started" in addresses[0].summary
        assert "Installation" in addresses[0].summary
        assert addresses[0].metadata["text"] == content

    def test_generic_file_creates_file_address(self):
        """Files without symbols or headings produce a single FILE address."""
        strategy = self._make_strategy()
        entry = _generic_entry()
        content = "key: value"
        addresses = strategy._create_addresses(entry, content)

        assert len(addresses) == 1
        assert addresses[0].kind == AddressKind.FILE
        assert entry.rel_path in addresses[0].summary
        assert addresses[0].metadata["text"] == content


# ---------------------------------------------------------------------------
# Tests: LLM selection parsing
# ---------------------------------------------------------------------------


class TestLLMSelectionParsing:
    def _make_strategy(self) -> AgenticSearchStrategy:
        return AgenticSearchStrategy(
            manifest=MagicMock(),
            source_dir=Path("/project"),
            chat_factory=MagicMock(),
            config=MagicMock(),
        )

    def test_parses_json_array(self):
        """Normal JSON array response is parsed correctly."""
        strategy = self._make_strategy()
        chat_mock = MagicMock()
        chat_mock.chat.return_value = '["src/auth.py", "src/db.py"]'
        strategy._chat_factory = MagicMock(return_value=chat_mock)

        result = strategy._llm_select_files("manifest text", "auth query")
        assert result == ["src/auth.py", "src/db.py"]

    def test_parses_json_with_surrounding_text(self):
        """JSON array embedded in explanatory text should still parse."""
        strategy = self._make_strategy()
        chat_mock = MagicMock()
        chat_mock.chat.return_value = (
            "Based on the query, here are the relevant files:\n"
            '["src/auth.py", "src/models/user.py"]\n'
            "These files contain authentication logic."
        )
        strategy._chat_factory = MagicMock(return_value=chat_mock)

        result = strategy._llm_select_files("manifest", "auth")
        assert result == ["src/auth.py", "src/models/user.py"]

    def test_returns_empty_on_exception(self):
        """LLM errors should return empty list, not crash."""
        strategy = self._make_strategy()
        chat_mock = MagicMock()
        chat_mock.chat.side_effect = RuntimeError("LLM unavailable")
        strategy._chat_factory = MagicMock(return_value=chat_mock)

        result = strategy._llm_select_files("manifest", "query")
        assert result == []

    def test_returns_empty_on_invalid_json(self):
        """Malformed JSON should return empty list."""
        strategy = self._make_strategy()
        chat_mock = MagicMock()
        chat_mock.chat.return_value = "I cannot determine which files are relevant."
        strategy._chat_factory = MagicMock(return_value=chat_mock)

        result = strategy._llm_select_files("manifest", "query")
        assert result == []

    def test_filters_non_string_entries(self):
        """Non-string entries in the JSON array should be filtered out."""
        strategy = self._make_strategy()
        chat_mock = MagicMock()
        chat_mock.chat.return_value = '["src/auth.py", 42, null, "src/db.py", ""]'
        strategy._chat_factory = MagicMock(return_value=chat_mock)

        result = strategy._llm_select_files("manifest", "query")
        # 42, null, and "" should be filtered
        assert result == ["src/auth.py", "src/db.py"]
