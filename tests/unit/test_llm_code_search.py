# tests/unit/test_llm_code_search.py
"""Tests for LLM structural code search strategy."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from fitz_sage.engines.fitz_krag.retrieval.strategies.llm_code_search import (
    LlmCodeSearchStrategy,
    ManifestBuilder,
)
from fitz_sage.engines.fitz_krag.types import AddressKind

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MANIFEST_DATA = [
    {
        "raw_file_id": "f1",
        "path": "app/models.py",
        "symbols": [
            {
                "name": "User",
                "qualified_name": "app.models.User",
                "kind": "class",
                "signature": "class User:",
                "start_line": 1,
                "end_line": 30,
                "imports": ["sqlalchemy"],
            },
            {
                "name": "__init__",
                "qualified_name": "app.models.User.__init__",
                "kind": "method",
                "signature": "def __init__(self, name: str) -> None",
                "start_line": 5,
                "end_line": 10,
                "imports": [],
            },
        ],
    },
    {
        "raw_file_id": "f2",
        "path": "app/views.py",
        "symbols": [
            {
                "name": "get_users",
                "qualified_name": "app.views.get_users",
                "kind": "function",
                "signature": "def get_users(db: Session) -> list[User]",
                "start_line": 1,
                "end_line": 15,
                "imports": ["app.models"],
            },
        ],
    },
    {
        "raw_file_id": "f3",
        "path": "app/serializers.py",
        "symbols": [
            {
                "name": "UserSerializer",
                "qualified_name": "app.serializers.UserSerializer",
                "kind": "class",
                "signature": "class UserSerializer:",
                "start_line": 1,
                "end_line": 20,
                "imports": ["app.models"],
            },
        ],
    },
    {
        "raw_file_id": "f4",
        "path": "lib/utils.py",
        "symbols": [
            {
                "name": "slugify",
                "qualified_name": "lib.utils.slugify",
                "kind": "function",
                "signature": "def slugify(text: str) -> str",
                "start_line": 1,
                "end_line": 5,
                "imports": [],
            },
        ],
    },
]


def _make_symbol_store(manifest_data=None):
    store = MagicMock()
    store.get_structural_manifest.return_value = (
        manifest_data if manifest_data is not None else MANIFEST_DATA
    )
    return store


def _make_import_store(reverse_counts=None, imports_by_file=None):
    store = MagicMock()
    store.get_reverse_counts.return_value = reverse_counts or {}
    if imports_by_file:
        store.get_imports.side_effect = lambda fid: imports_by_file.get(fid, [])
    else:
        store.get_imports.return_value = []
    return store


def _make_chat_response(search_terms=None, files=None):
    """Build a combined JSON response matching the new prompt format."""
    if files is None:
        files = ["app/models.py"]
    obj = {"search_terms": search_terms or [], "files": files}
    return json.dumps(obj)


def _make_chat_factory(response=None):
    if response is None:
        response = _make_chat_response()
    chat = MagicMock()
    chat.chat.return_value = response
    factory = MagicMock(return_value=chat)
    return factory


def _make_config():
    config = MagicMock()
    config.code_search_mode = "auto"
    return config


def _make_fallback():
    from fitz_sage.engines.fitz_krag.types import Address

    fallback = MagicMock()
    fallback.retrieve.return_value = [
        Address(
            kind=AddressKind.SYMBOL,
            source_id="f1",
            location="app.models.User",
            summary="class User",
            score=0.5,
        )
    ]
    return fallback


# ---------------------------------------------------------------------------
# ManifestBuilder tests
# ---------------------------------------------------------------------------


class TestManifestBuilder:
    def test_build_returns_text_and_file_data(self):
        symbol_store = _make_symbol_store()
        import_store = _make_import_store()
        builder = ManifestBuilder(symbol_store, import_store)

        text, file_data = builder.build()

        assert "## app/models.py" in text
        assert "## app/views.py" in text
        assert "app/models.py" in file_data
        assert "app/views.py" in file_data
        assert file_data["app/models.py"]["raw_file_id"] == "f1"

    def test_build_includes_classes_and_functions(self):
        symbol_store = _make_symbol_store()
        import_store = _make_import_store()
        builder = ManifestBuilder(symbol_store, import_store)

        text, _ = builder.build()

        assert "classes:" in text
        assert "User" in text
        assert "functions:" in text
        assert "get_users" in text

    def test_build_includes_imports(self):
        symbol_store = _make_symbol_store()
        import_store = _make_import_store()
        builder = ManifestBuilder(symbol_store, import_store)

        text, _ = builder.build()

        assert "imports:" in text
        assert "sqlalchemy" in text

    def test_build_caches_on_same_symbol_count(self):
        symbol_store = _make_symbol_store()
        import_store = _make_import_store()
        builder = ManifestBuilder(symbol_store, import_store)

        text1, _ = builder.build()
        text2, _ = builder.build()

        assert text1 == text2
        assert symbol_store.get_structural_manifest.call_count == 2

    def test_truncation_strips_imports_first(self):
        symbol_store = _make_symbol_store()
        import_store = _make_import_store(reverse_counts={"f1": 5, "f2": 1})
        builder = ManifestBuilder(symbol_store, import_store, max_chars=100)

        text, _ = builder.build()

        assert "## app/models.py" in text
        assert "## app/views.py" in text

    def test_empty_manifest(self):
        symbol_store = _make_symbol_store(manifest_data=[])
        import_store = _make_import_store()
        builder = ManifestBuilder(symbol_store, import_store)

        text, file_data = builder.build()

        assert text == ""
        assert file_data == {}

    def test_extract_return_type(self):
        assert ManifestBuilder._extract_return_type("def foo() -> str") == "str"
        assert ManifestBuilder._extract_return_type("def foo()") == "None"

    def test_extract_params(self):
        assert ManifestBuilder._extract_params("def foo(a: int, b: str)") == "a: int, b: str"
        assert ManifestBuilder._extract_params("def foo(self, a: int)") == "a: int"
        assert ManifestBuilder._extract_params("def foo(self)") == ""


# ---------------------------------------------------------------------------
# LlmCodeSearchStrategy tests
# ---------------------------------------------------------------------------


class TestLlmCodeSearchStrategy:
    def test_retrieve_returns_file_addresses(self):
        response = _make_chat_response(
            search_terms=["User", "model"],
            files=["app/models.py"],
        )
        strategy = LlmCodeSearchStrategy(
            _make_symbol_store(),
            _make_import_store(),
            _make_chat_factory(response),
            _make_config(),
            _make_fallback(),
        )
        addresses = strategy.retrieve("What is User?", limit=10)

        assert len(addresses) > 0
        assert all(a.kind == AddressKind.FILE for a in addresses)

    def test_selected_files_score_1_0(self):
        response = _make_chat_response(files=["app/models.py", "app/views.py"])
        strategy = LlmCodeSearchStrategy(
            _make_symbol_store(),
            _make_import_store(),
            _make_chat_factory(response),
            _make_config(),
            _make_fallback(),
        )
        addresses = strategy.retrieve("How does the app work?", limit=20)

        selected = [a for a in addresses if a.metadata.get("origin") == "selected"]
        assert len(selected) == 2
        assert all(a.score == 1.0 for a in selected)

    def test_import_expanded_files_score_0_9(self):
        response = _make_chat_response(files=["app/models.py"])
        strategy = LlmCodeSearchStrategy(
            _make_symbol_store(),
            _make_import_store(
                imports_by_file={"f1": [{"target_file_id": "f2", "target_module": "app.views"}]}
            ),
            _make_chat_factory(response),
            _make_config(),
            _make_fallback(),
        )
        addresses = strategy.retrieve("What is User?", limit=20)

        import_addrs = [a for a in addresses if a.metadata.get("origin") == "import"]
        assert len(import_addrs) >= 1
        assert all(a.score == 0.9 for a in import_addrs)

    def test_neighbor_expanded_files_score_0_8(self):
        """Files in the same directory as selected files get neighbor score."""
        response = _make_chat_response(files=["app/models.py"])
        strategy = LlmCodeSearchStrategy(
            _make_symbol_store(),
            _make_import_store(),
            _make_chat_factory(response),
            _make_config(),
            _make_fallback(),
        )
        addresses = strategy.retrieve("What is User?", limit=20)

        # app/views.py and app/serializers.py are neighbors of app/models.py
        neighbor_addrs = [a for a in addresses if a.metadata.get("origin") == "neighbor"]
        assert len(neighbor_addrs) >= 1
        assert all(a.score == 0.8 for a in neighbor_addrs)

    def test_neighbor_expansion_skips_large_directories(self):
        """Directories with >10 new siblings are skipped."""
        # Create manifest with 12 files in same directory
        many_files = [
            {
                "raw_file_id": f"f{i}",
                "path": f"bigpkg/mod{i}.py",
                "symbols": [
                    {
                        "name": f"func{i}",
                        "qualified_name": f"bigpkg.mod{i}.func{i}",
                        "kind": "function",
                        "signature": f"def func{i}() -> None",
                        "start_line": 1,
                        "end_line": 5,
                        "imports": [],
                    }
                ],
            }
            for i in range(12)
        ]
        response = _make_chat_response(files=["bigpkg/mod0.py"])
        strategy = LlmCodeSearchStrategy(
            _make_symbol_store(many_files),
            _make_import_store(),
            _make_chat_factory(response),
            _make_config(),
            _make_fallback(),
        )
        addresses = strategy.retrieve("What does func0 do?", limit=20)

        # Only the selected file — neighbors skipped because >10 siblings
        assert len(addresses) == 1
        assert addresses[0].metadata["origin"] == "selected"

    def test_retrieve_expands_imports(self):
        response = _make_chat_response(files=["app/models.py"])
        strategy = LlmCodeSearchStrategy(
            _make_symbol_store(),
            _make_import_store(
                imports_by_file={"f1": [{"target_file_id": "f2", "target_module": "app.views"}]}
            ),
            _make_chat_factory(response),
            _make_config(),
            _make_fallback(),
        )
        addresses = strategy.retrieve("What is User?", limit=20)

        source_ids = {a.source_id for a in addresses}
        assert "f1" in source_ids
        assert "f2" in source_ids

    def test_fallback_on_llm_failure(self):
        chat = MagicMock()
        chat.chat.side_effect = RuntimeError("LLM unavailable")
        chat_factory = MagicMock(return_value=chat)

        fallback = _make_fallback()
        strategy = LlmCodeSearchStrategy(
            _make_symbol_store(),
            _make_import_store(),
            chat_factory,
            _make_config(),
            fallback,
        )
        addresses = strategy.retrieve("What is User?", limit=10)

        assert len(addresses) == 1
        assert addresses[0].source_id == "f1"
        fallback.retrieve.assert_called_once()

    def test_fallback_on_empty_response(self):
        response = _make_chat_response(files=[])
        fallback = _make_fallback()
        strategy = LlmCodeSearchStrategy(
            _make_symbol_store(),
            _make_import_store(),
            _make_chat_factory(response),
            _make_config(),
            fallback,
        )
        addresses = strategy.retrieve("What is User?", limit=10)

        assert len(addresses) == 1
        fallback.retrieve.assert_called_once()

    def test_fallback_on_invalid_json(self):
        fallback = _make_fallback()
        strategy = LlmCodeSearchStrategy(
            _make_symbol_store(),
            _make_import_store(),
            _make_chat_factory("not json at all"),
            _make_config(),
            fallback,
        )
        addresses = strategy.retrieve("What is User?", limit=10)

        assert len(addresses) == 1
        fallback.retrieve.assert_called_once()

    def test_retrieve_respects_limit(self):
        response = _make_chat_response(files=["app/models.py", "app/views.py"])
        strategy = LlmCodeSearchStrategy(
            _make_symbol_store(),
            _make_import_store(),
            _make_chat_factory(response),
            _make_config(),
            _make_fallback(),
        )
        addresses = strategy.retrieve("What is User?", limit=1)

        assert len(addresses) <= 1

    def test_forwards_raw_store(self):
        strategy = LlmCodeSearchStrategy(
            _make_symbol_store(),
            _make_import_store(),
            _make_chat_factory(),
            _make_config(),
            _make_fallback(),
        )
        strategy._raw_store = "raw_store"

        assert strategy._raw_store == "raw_store"

    def test_fallback_receives_detection(self):
        chat = MagicMock()
        chat.chat.side_effect = RuntimeError("fail")
        chat_factory = MagicMock(return_value=chat)

        fallback = _make_fallback()
        strategy = LlmCodeSearchStrategy(
            _make_symbol_store(),
            _make_import_store(),
            chat_factory,
            _make_config(),
            fallback,
        )

        detection = MagicMock()

        strategy.retrieve("query", limit=5, detection=detection)

        fallback.retrieve.assert_called_once_with("query", 5, detection=detection)

    def test_parses_combined_json_response(self):
        """New format: LLM returns {search_terms, files}."""
        response = json.dumps(
            {
                "search_terms": ["user", "model", "ORM"],
                "files": ["app/models.py", "app/views.py"],
            }
        )
        strategy = LlmCodeSearchStrategy(
            _make_symbol_store(),
            _make_import_store(),
            _make_chat_factory(response),
            _make_config(),
            _make_fallback(),
        )
        addresses = strategy.retrieve("How does User model work?", limit=20)

        source_ids = {a.source_id for a in addresses}
        assert "f1" in source_ids
        assert "f2" in source_ids

    def test_parses_plain_array_fallback(self):
        """Backward compat: LLM returns plain JSON array."""
        strategy = LlmCodeSearchStrategy(
            _make_symbol_store(),
            _make_import_store(),
            _make_chat_factory('["app/models.py", "app/views.py"]'),
            _make_config(),
            _make_fallback(),
        )
        addresses = strategy.retrieve("How does get_users work?", limit=20)

        source_ids = {a.source_id for a in addresses}
        assert "f1" in source_ids
        assert "f2" in source_ids

    def test_parses_json_in_markdown_code_block(self):
        """LLM may wrap JSON in markdown code blocks."""
        response = '```json\n{"search_terms": ["user"], "files": ["app/models.py"]}\n```'
        strategy = LlmCodeSearchStrategy(
            _make_symbol_store(),
            _make_import_store(),
            _make_chat_factory(response),
            _make_config(),
            _make_fallback(),
        )
        addresses = strategy.retrieve("What is User?", limit=20)

        assert any(a.source_id == "f1" for a in addresses)
