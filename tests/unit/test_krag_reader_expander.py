# tests/unit/test_krag_reader_expander.py
"""
Unit tests for ContentReader and CodeExpander.

ContentReader: reads raw file content for addresses, extracts line ranges.
CodeExpander: enriches read results with contextual code (imports, class headers).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fitz_sage.engines.fitz_krag.retrieval.expander import CodeExpander
from fitz_sage.engines.fitz_krag.retrieval.reader import ContentReader
from fitz_sage.engines.fitz_krag.types import Address, AddressKind, ReadResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_symbol_address(
    source_id: str = "file1",
    location: str = "mod.func",
    start_line: int = 4,
    end_line: int = 6,
    kind: str = "function",
    qualified_name: str = "mod.func",
    score: float = 0.9,
    symbol_id: str | None = None,
) -> Address:
    metadata: dict = {
        "start_line": start_line,
        "end_line": end_line,
        "kind": kind,
        "qualified_name": qualified_name,
    }
    if symbol_id is not None:
        metadata["symbol_id"] = symbol_id
    return Address(
        kind=AddressKind.SYMBOL,
        source_id=source_id,
        location=location,
        summary=f"Symbol {location}",
        score=score,
        metadata=metadata,
    )


def _make_file_address(
    source_id: str = "file1",
    location: str = "module.py",
) -> Address:
    return Address(
        kind=AddressKind.FILE,
        source_id=source_id,
        location=location,
        summary=f"File {location}",
    )


def _make_section_address(
    section_id: str = "sec-001",
    source_id: str = "file1",
) -> Address:
    return Address(
        kind=AddressKind.SECTION,
        source_id=source_id,
        location="Section A",
        summary="Section about A",
        metadata={"section_id": section_id},
    )


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
    """Create a mock RawFileStore.

    ``files`` maps source_id -> {"path": ..., "content": ...}.
    """
    store = MagicMock()
    if files is None:
        files = {
            "file1": {
                "path": "module.py",
                "content": RAW_FILE_CONTENT,
            }
        }
    store.get.side_effect = lambda sid: files.get(sid)
    return store


def _make_section_store(sections: dict[str, dict] | None = None) -> MagicMock:
    store = MagicMock()
    if sections is None:
        sections = {}
    store.get.side_effect = lambda sid: sections.get(sid)
    return store


def _make_config(
    max_expansion_depth: int = 1,
    include_class_context: bool = True,
    max_reference_expansions: int = 3,
    include_import_summaries: bool = True,
    max_import_expansions: int = 5,
    include_section_context: bool = True,
) -> MagicMock:
    config = MagicMock()
    config.max_expansion_depth = max_expansion_depth
    config.include_class_context = include_class_context
    config.max_reference_expansions = max_reference_expansions
    config.include_import_summaries = include_import_summaries
    config.max_import_expansions = max_import_expansions
    config.include_section_context = include_section_context
    return config


# ===========================================================================
# TestContentReader
# ===========================================================================


class TestContentReader:
    """Tests for ContentReader."""

    def test_read_symbol(self):
        """Reads symbol content by extracting the correct line range."""
        raw_store = _make_raw_store()
        reader = ContentReader(raw_store)

        # Lines 4-5 of RAW_FILE_CONTENT: "def func():" and "    return 42"
        addr = _make_symbol_address(start_line=4, end_line=5, location="mod.func")
        results = reader.read([addr], limit=10)

        assert len(results) == 1
        r = results[0]
        assert r.content == "def func():\n    return 42"
        assert r.file_path == "module.py"
        assert r.line_range == (4, 5)
        assert r.address is addr

    def test_read_symbol_missing_file(self):
        """When raw_store returns None the address is skipped."""
        raw_store = _make_raw_store(files={})  # empty -> always None
        reader = ContentReader(raw_store)

        addr = _make_symbol_address(source_id="missing")
        results = reader.read([addr], limit=10)

        assert results == []

    def test_read_file(self):
        """Reads full file content for a FILE address."""
        raw_store = _make_raw_store()
        reader = ContentReader(raw_store)

        addr = _make_file_address(source_id="file1", location="module.py")
        results = reader.read([addr], limit=10)

        assert len(results) == 1
        assert results[0].content == RAW_FILE_CONTENT
        assert results[0].file_path == "module.py"
        assert results[0].line_range is None

    def test_read_section(self):
        """Reads section content from the section store with metadata."""
        section_data = {
            "sec-001": {
                "content": "Section body text.",
                "title": "Introduction",
                "level": 2,
                "page_start": 1,
                "page_end": 3,
            }
        }
        raw_store = _make_raw_store()
        section_store = _make_section_store(section_data)
        reader = ContentReader(raw_store, section_store=section_store)

        addr = _make_section_address(section_id="sec-001", source_id="file1")
        results = reader.read([addr], limit=10)

        assert len(results) == 1
        r = results[0]
        assert r.content == "Section body text."
        assert r.file_path == "module.py"
        assert r.metadata["section_title"] == "Introduction"
        assert r.metadata["section_level"] == 2
        assert r.metadata["page_start"] == 1
        assert r.metadata["page_end"] == 3

    def test_read_section_no_store(self):
        """When section_store is None, section addresses return None."""
        reader = ContentReader(_make_raw_store(), section_store=None)

        addr = _make_section_address()
        results = reader.read([addr], limit=10)

        assert results == []

    def test_read_section_missing_id(self):
        """Missing section_id in metadata results in None."""
        section_store = _make_section_store({"sec-001": {"content": "x"}})
        reader = ContentReader(_make_raw_store(), section_store=section_store)

        addr = Address(
            kind=AddressKind.SECTION,
            source_id="file1",
            location="Section B",
            summary="no id",
            metadata={},  # no section_id key
        )
        results = reader.read([addr], limit=10)

        assert results == []

    def test_read_limits_results(self):
        """Only reads up to `limit` addresses."""
        raw_store = _make_raw_store()
        reader = ContentReader(raw_store)

        addrs = [
            _make_symbol_address(start_line=4, end_line=6),
            _make_file_address(),
        ]
        results = reader.read(addrs, limit=1)

        assert len(results) == 1
        # Only the first address should have been processed
        assert results[0].address is addrs[0]

    def test_read_skips_failed(self):
        """Failed reads are skipped; successful ones are still returned."""
        raw_store = _make_raw_store()
        reader = ContentReader(raw_store)

        addrs = [
            _make_symbol_address(source_id="missing"),  # will fail
            _make_symbol_address(source_id="file1", start_line=4, end_line=5),  # will succeed
        ]
        results = reader.read(addrs, limit=10)

        assert len(results) == 1
        assert results[0].content == "def func():\n    return 42"


# ===========================================================================
# TestCodeExpander
# ===========================================================================


CLASS_FILE_CONTENT = (
    "import os\n"
    "from pathlib import Path\n"
    "\n"
    "class MyClass:\n"
    '    """A sample class."""\n'
    "\n"
    "    def __init__(self):\n"
    "        self.x = 1\n"
    "\n"
    "    def do_thing(self):\n"
    "        return self.x\n"
)


NO_IMPORT_CONTENT = "# Just a comment\n" "\n" "def standalone():\n" "    return True\n"


class TestCodeExpander:
    """Tests for CodeExpander."""

    def _make_expander(
        self,
        raw_files: dict[str, dict] | None = None,
        symbol_results: list[dict] | None = None,
        file_symbols: list[dict] | None = None,
        import_edges: list[dict] | None = None,
        max_expansion_depth: int = 1,
        include_class_context: bool = True,
        max_reference_expansions: int = 3,
        include_import_summaries: bool = True,
        max_import_expansions: int = 5,
    ) -> CodeExpander:
        if raw_files is None:
            raw_files = {
                "file1": {
                    "path": "module.py",
                    "content": RAW_FILE_CONTENT,
                }
            }
        raw_store = _make_raw_store(raw_files)
        symbol_store = MagicMock()
        symbol_store.search_by_name.return_value = symbol_results or []
        symbol_store.get_by_file.return_value = file_symbols or []
        symbol_store.get.return_value = None
        import_store = MagicMock()
        import_store.get_imports.return_value = import_edges or []
        config = _make_config(
            max_expansion_depth=max_expansion_depth,
            include_class_context=include_class_context,
            max_reference_expansions=max_reference_expansions,
            include_import_summaries=include_import_summaries,
            max_import_expansions=max_import_expansions,
        )
        return CodeExpander(raw_store, symbol_store, import_store, config)

    def _symbol_read_result(
        self,
        file_path: str = "module.py",
        content: str = "def func():\n    return 42",
        source_id: str = "file1",
        start_line: int = 4,
        end_line: int = 6,
        kind: str = "function",
        qualified_name: str = "mod.func",
        symbol_id: str | None = None,
    ) -> ReadResult:
        addr = _make_symbol_address(
            source_id=source_id,
            start_line=start_line,
            end_line=end_line,
            kind=kind,
            qualified_name=qualified_name,
            symbol_id=symbol_id,
        )
        return ReadResult(
            address=addr,
            content=content,
            file_path=file_path,
            line_range=(start_line, end_line),
        )

    def test_expand_adds_imports(self):
        """Symbol result triggers addition of file-level imports."""
        expander = self._make_expander()
        result = self._symbol_read_result()

        expanded = expander.expand([result])

        # Original + imports
        assert len(expanded) == 2
        import_result = expanded[1]
        assert import_result.metadata.get("context_type") == "imports"
        assert "import os" in import_result.content
        assert "import sys" in import_result.content
        assert import_result.file_path == "module.py"

    def test_expand_no_imports(self):
        """File with no imports produces no additional results."""
        raw_files = {
            "file1": {
                "path": "no_imports.py",
                "content": NO_IMPORT_CONTENT,
            }
        }
        expander = self._make_expander(raw_files=raw_files)
        result = self._symbol_read_result(file_path="no_imports.py")

        expanded = expander.expand([result])

        assert len(expanded) == 1
        assert expanded[0] is result

    def test_expand_skips_non_symbol(self):
        """Non-SYMBOL addresses (e.g. SECTION) are not expanded."""
        expander = self._make_expander()
        addr = _make_section_address()
        section_result = ReadResult(
            address=addr,
            content="Section text",
            file_path="doc.md",
        )

        expanded = expander.expand([section_result])

        assert len(expanded) == 1
        assert expanded[0] is section_result

    def test_expand_class_context_for_method(self):
        """Method symbols get a class header added."""
        raw_files = {
            "file1": {
                "path": "module.py",
                "content": CLASS_FILE_CONTENT,
            }
        }
        class_symbol = {
            "qualified_name": "module.MyClass",
            "kind": "class",
            "raw_file_id": "file1",
            "start_line": 4,
            "end_line": 12,
        }
        expander = self._make_expander(
            raw_files=raw_files,
            symbol_results=[class_symbol],
        )
        result = self._symbol_read_result(
            content="    def do_thing(self):\n        return self.x",
            start_line=10,
            end_line=12,
            kind="method",
            qualified_name="module.MyClass.do_thing",
        )

        expanded = expander.expand([result])

        # Original + imports + class header = 3
        class_headers = [r for r in expanded if r.metadata.get("context_type") == "class_header"]
        assert len(class_headers) == 1
        header = class_headers[0]
        assert "class MyClass:" in header.content
        assert header.file_path == "module.py"
        assert header.line_range == (4, 8)  # start_line to min(start+5, end_line)

    def test_expand_no_class_context_for_function(self):
        """Function (not method) symbols do not get a class header."""
        raw_files = {
            "file1": {
                "path": "module.py",
                "content": CLASS_FILE_CONTENT,
            }
        }
        expander = self._make_expander(
            raw_files=raw_files,
            symbol_results=[],
        )
        result = self._symbol_read_result(
            kind="function",
            qualified_name="mod.func",
        )

        expanded = expander.expand([result])

        class_headers = [r for r in expanded if r.metadata.get("context_type") == "class_header"]
        assert class_headers == []

    def test_expand_deduplicates(self):
        """Same imports block is not added twice for two symbols in same file."""
        expander = self._make_expander()
        result1 = self._symbol_read_result(start_line=4, end_line=6, qualified_name="mod.func")
        result2 = self._symbol_read_result(
            start_line=7,
            end_line=9,
            qualified_name="mod.other",
            content="def other():\n    pass",
            source_id="file1",
        )

        expanded = expander.expand([result1, result2])

        import_results = [r for r in expanded if r.metadata.get("context_type") == "imports"]
        # The import block duplication is caught either by the inline
        # check in _add_file_imports or by _deduplicate at the end.
        assert len(import_results) == 1

    def test_expand_disabled_when_depth_zero(self):
        """max_expansion_depth=0 returns read results as-is."""
        expander = self._make_expander(max_expansion_depth=0)
        result = self._symbol_read_result()

        expanded = expander.expand([result])

        assert expanded == [result]
        assert len(expanded) == 1

    def test_expand_class_context_disabled(self):
        """include_class_context=False prevents class header expansion."""
        raw_files = {
            "file1": {
                "path": "module.py",
                "content": CLASS_FILE_CONTENT,
            }
        }
        class_symbol = {
            "qualified_name": "module.MyClass",
            "kind": "class",
            "raw_file_id": "file1",
            "start_line": 4,
            "end_line": 12,
        }
        expander = self._make_expander(
            raw_files=raw_files,
            symbol_results=[class_symbol],
            include_class_context=False,
        )
        result = self._symbol_read_result(
            content="    def do_thing(self):\n        return self.x",
            start_line=10,
            end_line=12,
            kind="method",
            qualified_name="module.MyClass.do_thing",
        )

        expanded = expander.expand([result])

        class_headers = [r for r in expanded if r.metadata.get("context_type") == "class_header"]
        assert class_headers == []

    # ------------------------------------------------------------------
    # Same-file reference expansion tests
    # ------------------------------------------------------------------

    def test_expand_same_file_references(self):
        """Referenced helper function from same file is added with context_type='reference'."""
        file_symbols = [
            {
                "id": "s1",
                "name": "main_fn",
                "qualified_name": "mod.main_fn",
                "kind": "function",
                "start_line": 4,
                "end_line": 6,
                "references": ["helper"],
                "raw_file_id": "file1",
                "signature": "def main_fn()",
                "summary": "Main function",
                "metadata": {},
            },
            {
                "id": "s2",
                "name": "helper",
                "qualified_name": "mod.helper",
                "kind": "function",
                "start_line": 7,
                "end_line": 8,
                "references": [],
                "raw_file_id": "file1",
                "signature": "def helper()",
                "summary": "Helper function",
                "metadata": {},
            },
        ]
        expander = self._make_expander(file_symbols=file_symbols)
        result = self._symbol_read_result(symbol_id="s1", qualified_name="mod.main_fn")

        expanded = expander.expand([result])

        refs = [r for r in expanded if r.metadata.get("context_type") == "reference"]
        assert len(refs) == 1
        assert "def other():" in refs[0].content or "helper" in refs[0].address.location
        assert refs[0].address.metadata["qualified_name"] == "mod.helper"

    def test_expand_same_file_refs_capped(self):
        """5 referenced symbols but cap=2, only 2 added."""
        file_symbols = [
            {
                "id": "s1",
                "name": "main_fn",
                "qualified_name": "mod.main_fn",
                "kind": "function",
                "start_line": 4,
                "end_line": 6,
                "references": ["a", "b", "c", "d", "e"],
                "raw_file_id": "file1",
                "signature": "def main_fn()",
                "summary": "Main",
                "metadata": {},
            },
        ]
        # Add 5 symbols that match the references
        for i, name in enumerate(["a", "b", "c", "d", "e"]):
            file_symbols.append(
                {
                    "id": f"s{i + 2}",
                    "name": name,
                    "qualified_name": f"mod.{name}",
                    "kind": "function",
                    "start_line": 7 + i * 2,
                    "end_line": 8 + i * 2,
                    "references": [],
                    "raw_file_id": "file1",
                    "signature": f"def {name}()",
                    "summary": f"Function {name}",
                    "metadata": {},
                }
            )
        expander = self._make_expander(file_symbols=file_symbols, max_reference_expansions=2)
        result = self._symbol_read_result(symbol_id="s1", qualified_name="mod.main_fn")

        expanded = expander.expand([result])

        refs = [r for r in expanded if r.metadata.get("context_type") == "reference"]
        assert len(refs) == 2

    def test_expand_same_file_refs_skip_self(self):
        """Symbol doesn't reference itself even if its name appears in references."""
        file_symbols = [
            {
                "id": "s1",
                "name": "recursive",
                "qualified_name": "mod.recursive",
                "kind": "function",
                "start_line": 4,
                "end_line": 6,
                "references": ["recursive"],
                "raw_file_id": "file1",
                "signature": "def recursive()",
                "summary": "Recursive fn",
                "metadata": {},
            },
        ]
        expander = self._make_expander(file_symbols=file_symbols)
        result = self._symbol_read_result(symbol_id="s1", qualified_name="mod.recursive")

        expanded = expander.expand([result])

        refs = [r for r in expanded if r.metadata.get("context_type") == "reference"]
        assert len(refs) == 0

    def test_expand_same_file_refs_disabled(self):
        """max_reference_expansions=0 skips reference expansion entirely."""
        file_symbols = [
            {
                "id": "s1",
                "name": "main_fn",
                "qualified_name": "mod.main_fn",
                "kind": "function",
                "start_line": 4,
                "end_line": 6,
                "references": ["helper"],
                "raw_file_id": "file1",
                "signature": "def main_fn()",
                "summary": "Main",
                "metadata": {},
            },
            {
                "id": "s2",
                "name": "helper",
                "qualified_name": "mod.helper",
                "kind": "function",
                "start_line": 7,
                "end_line": 8,
                "references": [],
                "raw_file_id": "file1",
                "signature": "def helper()",
                "summary": "Helper",
                "metadata": {},
            },
        ]
        expander = self._make_expander(file_symbols=file_symbols, max_reference_expansions=0)
        result = self._symbol_read_result(symbol_id="s1", qualified_name="mod.main_fn")

        expanded = expander.expand([result])

        refs = [r for r in expanded if r.metadata.get("context_type") == "reference"]
        assert len(refs) == 0

    # ------------------------------------------------------------------
    # Import summary expansion tests
    # ------------------------------------------------------------------

    def test_expand_import_summaries(self):
        """Resolved import edge produces summary context block."""
        target_file_symbols = [
            {
                "id": "ts1",
                "name": "helper_fn",
                "qualified_name": "utils.helper_fn",
                "kind": "function",
                "start_line": 1,
                "end_line": 5,
                "references": [],
                "raw_file_id": "f2",
                "signature": "def helper_fn()",
                "metadata": {},
            },
        ]
        import_edges = [
            {
                "source_file_id": "file1",
                "target_module": "utils",
                "target_file_id": "f2",
                "import_names": ["helper_fn"],
            },
        ]
        expander = self._make_expander(import_edges=import_edges)
        # Override get_by_file to return target symbols for f2
        expander._symbol_store.get_by_file.side_effect = lambda fid: (
            target_file_symbols if fid == "f2" else []
        )
        result = self._symbol_read_result(symbol_id="s1")

        expanded = expander.expand([result])

        summaries = [r for r in expanded if r.metadata.get("context_type") == "import_summaries"]
        assert len(summaries) == 1
        assert "utils.helper_fn" in summaries[0].content
        assert "function helper_fn" in summaries[0].content

    def test_expand_import_summaries_unresolved_skipped(self):
        """target_file_id=None edges are skipped (stdlib/third-party)."""
        import_edges = [
            {
                "source_file_id": "file1",
                "target_module": "os",
                "target_file_id": None,
                "import_names": ["path"],
            },
        ]
        expander = self._make_expander(import_edges=import_edges)
        result = self._symbol_read_result(symbol_id="s1")

        expanded = expander.expand([result])

        summaries = [r for r in expanded if r.metadata.get("context_type") == "import_summaries"]
        assert len(summaries) == 0

    def test_expand_import_summaries_disabled(self):
        """include_import_summaries=False skips import summary expansion."""
        import_edges = [
            {
                "source_file_id": "file1",
                "target_module": "utils",
                "target_file_id": "f2",
                "import_names": ["helper_fn"],
            },
        ]
        expander = self._make_expander(import_edges=import_edges, include_import_summaries=False)
        result = self._symbol_read_result(symbol_id="s1")

        expanded = expander.expand([result])

        summaries = [r for r in expanded if r.metadata.get("context_type") == "import_summaries"]
        assert len(summaries) == 0

    def test_expand_import_summaries_capped(self):
        """Respects max_import_expansions limit."""
        target_file_symbols = [
            {
                "id": f"ts{i}",
                "name": f"fn{i}",
                "qualified_name": f"utils.fn{i}",
                "kind": "function",
                "start_line": i,
                "end_line": i + 2,
                "references": [],
                "raw_file_id": "f2",
                "signature": f"def fn{i}()",
                "summary": f"Function {i}",
                "metadata": {},
            }
            for i in range(10)
        ]
        import_edges = [
            {
                "source_file_id": "file1",
                "target_module": "utils",
                "target_file_id": "f2",
                "import_names": [],  # empty = all
            },
        ]
        expander = self._make_expander(import_edges=import_edges, max_import_expansions=2)
        expander._symbol_store.get_by_file.side_effect = lambda fid: (
            target_file_symbols if fid == "f2" else []
        )
        result = self._symbol_read_result(symbol_id="s1")

        expanded = expander.expand([result])

        summaries = [r for r in expanded if r.metadata.get("context_type") == "import_summaries"]
        assert len(summaries) == 1
        # Only 2 lines despite 10 available symbols
        lines = [line for line in summaries[0].content.splitlines() if line.startswith("- ")]
        assert len(lines) == 2


# ===========================================================================
# TestContentReader — Section context tests
# ===========================================================================


class TestContentReaderSectionContext:
    """Tests for section breadcrumb and child TOC in ContentReader."""

    def test_read_section_with_breadcrumb(self):
        """Parent chain rendered as breadcrumb prefix."""
        section_data = {
            "sec-root": {
                "id": "sec-root",
                "content": "Root content",
                "title": "Introduction",
                "level": 1,
                "page_start": 1,
                "page_end": 5,
                "parent_section_id": None,
            },
            "sec-child": {
                "id": "sec-child",
                "content": "Child content",
                "title": "Background",
                "level": 2,
                "page_start": 2,
                "page_end": 4,
                "parent_section_id": "sec-root",
            },
            "sec-leaf": {
                "id": "sec-leaf",
                "content": "Leaf body text.",
                "title": "Details",
                "level": 3,
                "page_start": 3,
                "page_end": 4,
                "parent_section_id": "sec-child",
            },
        }
        raw_store = _make_raw_store()
        section_store = _make_section_store(section_data)
        section_store.get_children.return_value = []
        config = _make_config(include_section_context=True)
        reader = ContentReader(raw_store, section_store=section_store, config=config)

        addr = _make_section_address(section_id="sec-leaf", source_id="file1")
        results = reader.read([addr], limit=10)

        assert len(results) == 1
        r = results[0]
        assert r.content.startswith("[Introduction > Background]")
        assert "Leaf body text." in r.content
        assert r.metadata["breadcrumb"] == "Introduction > Background"

    def test_read_section_with_child_toc(self):
        """Children titles appended as subsection list."""
        section_data = {
            "sec-parent": {
                "id": "sec-parent",
                "content": "Parent body.",
                "title": "Chapter 1",
                "level": 1,
                "page_start": 1,
                "page_end": 10,
                "parent_section_id": None,
            },
        }
        child_sections = [
            {"title": "Section 1.1"},
            {"title": "Section 1.2"},
            {"title": "Section 1.3"},
        ]
        raw_store = _make_raw_store()
        section_store = _make_section_store(section_data)
        section_store.get_children.return_value = child_sections
        config = _make_config(include_section_context=True)
        reader = ContentReader(raw_store, section_store=section_store, config=config)

        addr = _make_section_address(section_id="sec-parent", source_id="file1")
        results = reader.read([addr], limit=10)

        assert len(results) == 1
        r = results[0]
        assert "Subsections:" in r.content
        assert "  - Section 1.1" in r.content
        assert "  - Section 1.2" in r.content
        assert "  - Section 1.3" in r.content
        assert r.metadata["child_count"] == 3

    def test_read_section_context_disabled(self):
        """include_section_context=False returns plain content."""
        section_data = {
            "sec-001": {
                "content": "Plain content.",
                "title": "Section A",
                "level": 1,
                "page_start": 1,
                "page_end": 5,
                "parent_section_id": None,
            },
        }
        raw_store = _make_raw_store()
        section_store = _make_section_store(section_data)
        config = _make_config(include_section_context=False)
        reader = ContentReader(raw_store, section_store=section_store, config=config)

        addr = _make_section_address(section_id="sec-001", source_id="file1")
        results = reader.read([addr], limit=10)

        assert len(results) == 1
        r = results[0]
        assert r.content == "Plain content."
        assert "breadcrumb" not in r.metadata
        assert "child_count" not in r.metadata

    def test_read_section_without_config_uses_plain_content(self):
        """ContentReader without config returns plain section content."""
        section_data = {
            "sec-001": {
                "content": "Section body text.",
                "title": "Introduction",
                "level": 2,
                "page_start": 1,
                "page_end": 3,
                "parent_section_id": "sec-root",
            },
        }
        raw_store = _make_raw_store()
        section_store = _make_section_store(section_data)
        reader = ContentReader(raw_store, section_store=section_store)

        addr = _make_section_address(section_id="sec-001", source_id="file1")
        results = reader.read([addr], limit=10)

        assert len(results) == 1
        r = results[0]
        assert r.content == "Section body text."
        assert "breadcrumb" not in r.metadata


# ===========================================================================
# TestContentReader — Table tests
# ===========================================================================


class TestContentReaderTable:
    """Tests for table address reading in ContentReader."""

    def _make_table_address(
        self,
        table_index_id: str = "rec-001",
        table_id: str = "tbl_abc",
        source_id: str = "file1",
        name: str = "Sales Data",
    ) -> Address:
        return Address(
            kind=AddressKind.TABLE,
            source_id=source_id,
            location=name,
            summary=f"Table {name}",
            metadata={
                "table_index_id": table_index_id,
                "table_id": table_id,
                "name": name,
                "columns": ["product", "revenue"],
                "row_count": 100,
            },
        )

    def test_read_table_address(self):
        """Reader returns schema + sample for TABLE address."""
        raw_store = _make_raw_store()
        table_store = MagicMock()
        table_store.get.return_value = {
            "id": "rec-001",
            "raw_file_id": "file1",
            "table_id": "tbl_abc",
            "name": "Sales Data",
            "columns": ["product", "revenue"],
            "row_count": 100,
            "summary": "Sales records",
            "metadata": {},
        }
        reader = ContentReader(raw_store, table_store=table_store)

        addr = self._make_table_address()
        results = reader.read([addr], limit=10)

        assert len(results) == 1
        r = results[0]
        assert "Table: Sales Data" in r.content
        assert "product, revenue" in r.content
        assert "Row count: 100" in r.content
        assert r.file_path == "module.py"
        assert r.metadata["table_id"] == "tbl_abc"

    def test_read_table_uses_bm25_matched_rows_instead_of_prefix_sample(self):
        """Concrete row hits should survive the address-to-evidence read."""
        raw_store = _make_raw_store()
        table_store = MagicMock()
        table_store.get.return_value = {
            "id": "rec-001",
            "raw_file_id": "file1",
            "table_id": "tbl_abc",
            "name": "Sales Data",
            "columns": ["product", "revenue"],
            "row_count": 900,
            "summary": "Sales records",
            "metadata": {},
        }
        sqlite_store = MagicMock()
        sqlite_store.get_rows_by_numbers.return_value = (
            ["product", "revenue"],
            [["Meridian", "85000"]],
        )
        reader = ContentReader(
            raw_store,
            table_store=table_store,
            sqlite_table_store=sqlite_store,
        )
        addr = self._make_table_address()
        addr.metadata["row_search"] = {"row_numbers": [640]}

        result = reader.read([addr], limit=10)[0]

        assert "Matched data:" in result.content
        assert "Meridian" in result.content
        assert result.metadata["matched_row_numbers"] == [640]
        sqlite_store.get_rows_by_numbers.assert_called_once_with(
            "tbl_abc",
            [640],
            limit=20,
        )

    def test_read_table_no_store(self):
        """Graceful fallback when table store unavailable."""
        raw_store = _make_raw_store()
        reader = ContentReader(raw_store, table_store=None)

        addr = self._make_table_address()
        results = reader.read([addr], limit=10)

        assert results == []

    def test_read_table_missing_index_id(self):
        """Missing table_index_id in metadata returns None."""
        raw_store = _make_raw_store()
        table_store = MagicMock()
        reader = ContentReader(raw_store, table_store=table_store)

        addr = Address(
            kind=AddressKind.TABLE,
            source_id="file1",
            location="Some Table",
            summary="A table",
            metadata={},  # no table_index_id
        )
        results = reader.read([addr], limit=10)

        assert results == []
