# tests/unit/test_krag_entity_graph.py
"""
Unit tests for entity graph integration in KRAG.

The ingestion tests exercise the *live* path: the background worker schedules
``core.enrich_file``, which extracts entities and adds them to the
EntityGraphStore. This is what ``engine.point()`` runs in production.

Tests that:
- enrich_file adds extracted entities to the EntityGraphStore
- symbols/sections without entities are skipped
- entity graph errors fail gracefully
- CodeExpander._add_entity_related finds related symbols
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fitz_sage.engines.fitz_krag.retrieval.expander import CodeExpander
from fitz_sage.engines.fitz_krag.types import Address, AddressKind, ReadResult

# ---------------------------------------------------------------------------
# Helpers — ingestion core
# ---------------------------------------------------------------------------


def _make_core(
    *,
    entity_graph_store: MagicMock | None = None,
):
    """Create a KragIngestPipeline core with a mocked connection manager."""
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.engines.fitz_krag.ingestion.pipeline import KragIngestPipeline

    config = FitzKragConfig(collection="test_col")
    return KragIngestPipeline(
        config=config,
        chat=MagicMock(),
        connection_manager=MagicMock(),
        collection="test_col",
        entity_graph_store=entity_graph_store,
    )


def _enricher_stamping(entity_sets: list[list[dict]]) -> MagicMock:
    """Enricher stub: stamps the i-th entity set onto the i-th symbol dict."""
    enricher = MagicMock()

    def _stamp(dicts):
        for i, d in enumerate(dicts):
            d["keywords"] = []
            d["entities"] = list(entity_sets[i % len(entity_sets)])

    enricher.enrich_symbols.side_effect = _stamp
    enricher.enrich_sections.side_effect = _stamp
    return enricher


# ---------------------------------------------------------------------------
# TestEnrichEntityGraphIntegration — produced by core.enrich_file
# ---------------------------------------------------------------------------


class TestEnrichEntityGraphIntegration:
    """Tests that enrich_file populates the entity graph during ingestion."""

    def test_enrich_file_populates_entity_graph(self):
        """enrich_file adds each symbol's extracted entities to the graph store."""
        entity_store = MagicMock()
        core = _make_core(entity_graph_store=entity_store)
        core._enricher = _enricher_stamping(
            [
                [
                    {"name": "PostgreSQL", "type": "technology"},
                    {"name": "auth_handler", "type": "function"},
                ],
                [{"name": "Redis", "type": "technology"}],
            ]
        )
        core._symbol_store = MagicMock()
        core._symbol_store.get_by_file.return_value = [
            {"id": "sym-001", "keywords": [], "entities": []},
            {"id": "sym-002", "keywords": [], "entities": []},
        ]

        core.enrich_file("file-1", ".py")

        assert entity_store.add_chunk_entities.call_count == 2

        first_call = entity_store.add_chunk_entities.call_args_list[0]
        assert first_call[0][0] == "sym-001"
        assert first_call[0][1] == [
            ("PostgreSQL", "technology"),
            ("auth_handler", "function"),
        ]

        second_call = entity_store.add_chunk_entities.call_args_list[1]
        assert second_call[0][0] == "sym-002"
        assert second_call[0][1] == [("Redis", "technology")]

    def test_enrich_file_skips_symbols_without_entities(self):
        """A symbol with no extracted entities is not added to the graph."""
        entity_store = MagicMock()
        core = _make_core(entity_graph_store=entity_store)
        core._enricher = _enricher_stamping(
            [
                [{"name": "PostgreSQL", "type": "technology"}],
                [],  # sym-002 has no entities — should be skipped
            ]
        )
        core._symbol_store = MagicMock()
        core._symbol_store.get_by_file.return_value = [
            {"id": "sym-001", "keywords": [], "entities": []},
            {"id": "sym-002", "keywords": [], "entities": []},
        ]

        core.enrich_file("file-1", ".py")

        entity_store.add_chunk_entities.assert_called_once()
        assert entity_store.add_chunk_entities.call_args[0][0] == "sym-001"

    def test_graceful_failure_on_entity_graph_errors(self):
        """enrich_file catches entity graph errors without crashing."""
        entity_store = MagicMock()
        entity_store.add_chunk_entities.side_effect = RuntimeError("DB connection lost")
        core = _make_core(entity_graph_store=entity_store)
        core._enricher = _enricher_stamping([[{"name": "PostgreSQL", "type": "technology"}]])
        core._symbol_store = MagicMock()
        core._symbol_store.get_by_file.return_value = [
            {"id": "sym-001", "keywords": [], "entities": []},
        ]

        # Should not raise
        core.enrich_file("file-1", ".py")

    def test_link_doc_entities_uses_deterministic_derivation(self):
        """Progressive doc entity linking should not run Qwen entity generation."""
        entity_store = MagicMock()
        core = _make_core(entity_graph_store=entity_store)
        sections = [
            {
                "id": "sec-001",
                "keywords": ["contract"],
                "entities": [],
                "metadata": {},
            }
        ]

        def _derive(items):
            items[0]["entities"] = [{"name": "TC-1000", "type": "identifier"}]

        core._enricher = MagicMock()
        core._enricher.derive_section_entities.side_effect = _derive
        core._section_store = MagicMock()
        core._section_store.get_by_file.return_value = sections

        core.link_entities_file("file-1", ".md")

        core._enricher.derive_section_entities.assert_called_once_with(sections)
        core._enricher.enrich_section_entities.assert_not_called()
        core._section_store.update_enrichment_by_file.assert_called_once_with("file-1", sections)
        entity_store.add_chunk_entities.assert_called_once_with(
            "sec-001",
            [("TC-1000", "identifier")],
        )


# ---------------------------------------------------------------------------
# Helpers — CodeExpander
# ---------------------------------------------------------------------------

RAW_FILE_CONTENT = (
    "import os\n"
    "import sys\n"
    "\n"
    "def func():\n"
    "    return 42\n"
    "\n"
    "def other():\n"
    "    pass\n"
)


def _make_raw_store(files: dict[str, dict] | None = None) -> MagicMock:
    store = MagicMock()
    if files is None:
        files = {
            "file1": {"path": "module.py", "content": RAW_FILE_CONTENT},
        }
    store.get.side_effect = lambda sid: files.get(sid)
    return store


def _make_config(
    max_expansion_depth: int = 1,
    include_class_context: bool = False,
    max_reference_expansions: int = 0,
    include_import_summaries: bool = False,
    max_import_expansions: int = 0,
) -> MagicMock:
    config = MagicMock()
    config.max_expansion_depth = max_expansion_depth
    config.include_class_context = include_class_context
    config.max_reference_expansions = max_reference_expansions
    config.include_import_summaries = include_import_summaries
    config.max_import_expansions = max_import_expansions
    return config


def _make_symbol_address(
    source_id: str = "file1",
    location: str = "mod.func",
    symbol_id: str = "sym-1",
    kind: str = "function",
    qualified_name: str = "mod.func",
    start_line: int = 4,
    end_line: int = 6,
    score: float = 0.9,
) -> Address:
    return Address(
        kind=AddressKind.SYMBOL,
        source_id=source_id,
        location=location,
        summary=f"Symbol {location}",
        score=score,
        metadata={
            "start_line": start_line,
            "end_line": end_line,
            "kind": kind,
            "qualified_name": qualified_name,
            "symbol_id": symbol_id,
        },
    )


def _make_read_result(
    symbol_id: str = "sym-1",
    source_id: str = "file1",
    file_path: str = "module.py",
    content: str = "def func():\n    return 42",
) -> ReadResult:
    addr = _make_symbol_address(source_id=source_id, symbol_id=symbol_id)
    return ReadResult(
        address=addr,
        content=content,
        file_path=file_path,
        line_range=(4, 6),
    )


def _make_entity_graph_store(
    related_ids: list[str] | None = None,
    side_effect=None,
) -> MagicMock:
    store = MagicMock()
    if side_effect:
        store.get_related_chunks.side_effect = side_effect
    else:
        store.get_related_chunks.return_value = related_ids or []
    return store


def _make_expander(
    entity_graph_store: MagicMock | None = None,
    raw_files: dict[str, dict] | None = None,
    symbol_get_return: dict | None = None,
) -> CodeExpander:
    """Create a CodeExpander with entity graph support."""
    raw_store = _make_raw_store(raw_files)
    symbol_store = MagicMock()
    symbol_store.search_by_name.return_value = []
    symbol_store.get_by_file.return_value = []
    symbol_store.get.return_value = symbol_get_return
    import_store = MagicMock()
    import_store.get_imports.return_value = []
    config = _make_config()

    expander = CodeExpander(raw_store, symbol_store, import_store, config)
    if entity_graph_store is not None:
        expander._entity_graph_store = entity_graph_store
    return expander


# ---------------------------------------------------------------------------
# TestExpanderEntityRelated
# ---------------------------------------------------------------------------


class TestExpanderEntityRelated:
    """Tests for CodeExpander._add_entity_related."""

    def test_finds_related_symbols(self):
        """Entity graph returns related symbol IDs; expander fetches and appends them."""
        entity_store = _make_entity_graph_store(related_ids=["sym-related"])
        related_symbol = {
            "id": "sym-related",
            "name": "related_func",
            "qualified_name": "mod.related_func",
            "kind": "function",
            "raw_file_id": "file1",
            "start_line": 7,
            "end_line": 8,
        }
        expander = _make_expander(
            entity_graph_store=entity_store,
            symbol_get_return=related_symbol,
        )

        original = _make_read_result(symbol_id="sym-1")
        expanded = expander.expand([original])

        # Original + imports + entity-related
        entity_results = [r for r in expanded if r.metadata.get("context_type") == "entity_related"]
        assert len(entity_results) == 1
        assert entity_results[0].address.location == "mod.related_func"
        assert entity_results[0].address.metadata["symbol_id"] == "sym-related"

    def test_skips_already_present_ids(self):
        """Entity-related IDs that are already in expanded results are not added again."""
        entity_store = _make_entity_graph_store(related_ids=["sym-1"])
        expander = _make_expander(entity_graph_store=entity_store)

        original = _make_read_result(symbol_id="sym-1")
        expanded = expander.expand([original])

        entity_results = [r for r in expanded if r.metadata.get("context_type") == "entity_related"]
        assert len(entity_results) == 0

    def test_skipped_when_no_entity_graph_store(self):
        """No entity_graph_store -> entity expansion step is skipped entirely."""
        expander = _make_expander(entity_graph_store=None)

        original = _make_read_result()
        expanded = expander.expand([original])

        entity_results = [r for r in expanded if r.metadata.get("context_type") == "entity_related"]
        assert len(entity_results) == 0

    def test_graceful_failure_on_entity_graph_error(self):
        """Exception in entity graph store does not crash the expander."""
        entity_store = _make_entity_graph_store(side_effect=RuntimeError("connection timeout"))
        expander = _make_expander(entity_graph_store=entity_store)

        original = _make_read_result()

        # Should not raise; returns at least the original results
        expanded = expander.expand([original])
        assert len(expanded) >= 1
        assert expanded[0].content == original.content

    def test_no_related_ids_returns_unchanged(self):
        """Entity graph returning empty list does not add any results."""
        entity_store = _make_entity_graph_store(related_ids=[])
        expander = _make_expander(entity_graph_store=entity_store)

        original = _make_read_result()
        expanded = expander.expand([original])

        entity_results = [r for r in expanded if r.metadata.get("context_type") == "entity_related"]
        assert len(entity_results) == 0
