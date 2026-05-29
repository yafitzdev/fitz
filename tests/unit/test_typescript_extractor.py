# tests/unit/test_typescript_extractor.py
"""Tests for TypeScript/JavaScript extractor — tree-sitter mocked for unit tests."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock tree-sitter modules so tests run without optional dependency
# ---------------------------------------------------------------------------
_mock_ts = MagicMock()
_mock_ts_lang = MagicMock()
_orig_ts = sys.modules.get("tree_sitter")
_orig_ts_lang = sys.modules.get("tree_sitter_typescript")
sys.modules["tree_sitter"] = _mock_ts
sys.modules["tree_sitter_typescript"] = _mock_ts_lang

import fitz_sage.engines.fitz_krag.ingestion.strategies.typescript as _ts_mod  # noqa: E402
from fitz_sage.engines.fitz_krag.ingestion.code_utils import (  # noqa: E402
    node_text,
    path_to_module,
)
from fitz_sage.engines.fitz_krag.ingestion.strategies.typescript import (  # noqa: E402
    TypeScriptIngestStrategy,
    _extract_import,
    _walk_node,
)

_TS_EXTS = (".ts", ".tsx", ".js", ".jsx")


def _ts_path_to_module(path: str) -> str:
    return path_to_module(path, _TS_EXTS, "/index")


# ---------------------------------------------------------------------------
# MockNode — lightweight stand-in for tree-sitter Node
# ---------------------------------------------------------------------------
class MockNode:
    """Mock tree-sitter node for testing."""

    def __init__(self, type_, text="", start=(0, 0), end=(0, 0), children=None, fields=None):
        self.type = type_
        self.text = text.encode("utf-8") if isinstance(text, str) else text
        self.start_point = start
        self.end_point = end
        self.children = children or []
        self._fields = fields or {}

    def child_by_field_name(self, name):
        return self._fields.get(name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# Reference the actual Parser mock the module imported (avoids cross-test mock issues)
_parser_mock = _ts_mod.Parser


def _make_tree(root_children):
    """Build a mock tree whose root_node has the given children."""
    root = MockNode("program", children=root_children)
    _parser_mock.return_value.parse.return_value.root_node = root
    return root


# ---------------------------------------------------------------------------
# Tests: path / text helpers
# ---------------------------------------------------------------------------
class TestHelpers:
    def test_path_to_module_ts(self):
        assert _ts_path_to_module("src/utils.ts") == "src.utils"

    def test_path_to_module_tsx(self):
        assert _ts_path_to_module("components/App.tsx") == "components.App"

    def test_path_to_module_js(self):
        assert _ts_path_to_module("lib/index.js") == "lib"

    def test_path_to_module_backslash(self):
        assert _ts_path_to_module("src\\utils.ts") == "src.utils"

    def test_node_text_bytes(self):
        node = MockNode("id", b"hello")
        assert node_text(node) == "hello"

    def test_node_text_none(self):
        assert node_text(None) == ""


# ---------------------------------------------------------------------------
# Tests: function extraction
# ---------------------------------------------------------------------------
class TestFunctionExtraction:
    def test_extracts_function(self):
        name = MockNode("identifier", "greet", (0, 9), (0, 14))
        params = MockNode("formal_parameters", "(name: string)", (0, 14), (0, 28))
        func = MockNode(
            "function_declaration",
            "function greet(name: string) { }",
            (0, 0),
            (0, 32),
            children=[name, params],
            fields={"name": name, "parameters": params},
        )

        source = 'function greet(name: string) { return "hi"; }'
        symbols = []
        imports = []
        _walk_node(
            MockNode("program", children=[func]), source.splitlines(), "mod", symbols, imports
        )

        assert len(symbols) == 1
        assert symbols[0].name == "greet"
        assert symbols[0].kind == "function"
        assert symbols[0].qualified_name == "mod.greet"
        assert "function greet" in symbols[0].signature

    def test_arrow_function_from_lexical(self):
        arrow = MockNode(
            "arrow_function", "() => {}", (0, 20), (0, 28), fields={"parameters": None}
        )
        name = MockNode("identifier", "handler", (0, 6), (0, 13))
        declarator = MockNode(
            "variable_declarator",
            "handler = () => {}",
            (0, 6),
            (0, 28),
            children=[name, arrow],
            fields={"name": name, "value": arrow},
        )
        lexical = MockNode(
            "lexical_declaration",
            "const handler = () => {}",
            (0, 0),
            (0, 28),
            children=[declarator],
        )

        source = "const handler = () => {}"
        symbols = []
        imports = []
        _walk_node(
            MockNode("program", children=[lexical]), source.splitlines(), "mod", symbols, imports
        )

        assert len(symbols) == 1
        assert symbols[0].name == "handler"
        assert symbols[0].kind == "function"

    def test_constant_from_lexical(self):
        value = MockNode("number", "42", (0, 16), (0, 18))
        name = MockNode("identifier", "MAX_SIZE", (0, 6), (0, 14))
        declarator = MockNode(
            "variable_declarator",
            "MAX_SIZE = 42",
            (0, 6),
            (0, 18),
            children=[name, value],
            fields={"name": name, "value": value},
        )
        lexical = MockNode(
            "lexical_declaration",
            "const MAX_SIZE = 42",
            (0, 0),
            (0, 18),
            children=[declarator],
        )

        source = "const MAX_SIZE = 42"
        symbols = []
        imports = []
        _walk_node(
            MockNode("program", children=[lexical]), source.splitlines(), "mod", symbols, imports
        )

        assert len(symbols) == 1
        assert symbols[0].name == "MAX_SIZE"
        assert symbols[0].kind == "constant"


# ---------------------------------------------------------------------------
# Tests: class extraction
# ---------------------------------------------------------------------------
class TestClassExtraction:
    def test_extracts_class(self):
        name = MockNode("identifier", "UserService", (0, 6), (0, 17))
        body = MockNode("class_body", "{}", (0, 18), (0, 20), children=[])
        cls = MockNode(
            "class_declaration",
            "class UserService {}",
            (0, 0),
            (0, 20),
            children=[name, body],
            fields={"name": name, "body": body},
        )

        source = "class UserService {}"
        symbols = []
        imports = []
        _walk_node(
            MockNode("program", children=[cls]), source.splitlines(), "mod", symbols, imports
        )

        assert len(symbols) == 1
        assert symbols[0].name == "UserService"
        assert symbols[0].kind == "class"

    def test_extracts_class_with_method(self):
        cls_name = MockNode("identifier", "Server", (0, 6), (0, 12))
        method_name = MockNode("identifier", "start", (1, 2), (1, 7))
        method_params = MockNode("formal_parameters", "()", (1, 7), (1, 9))
        method = MockNode(
            "method_definition",
            "start() {}",
            (1, 2),
            (1, 12),
            children=[method_name, method_params],
            fields={"name": method_name, "parameters": method_params},
        )
        body = MockNode("class_body", "{ start() {} }", (0, 13), (2, 1), children=[method])
        cls = MockNode(
            "class_declaration",
            "class Server { start() {} }",
            (0, 0),
            (2, 1),
            children=[cls_name, body],
            fields={"name": cls_name, "body": body},
        )

        source = "class Server {\n  start() {}\n}"
        symbols = []
        imports = []
        _walk_node(
            MockNode("program", children=[cls]), source.splitlines(), "mod", symbols, imports
        )

        assert len(symbols) == 2
        assert symbols[0].name == "Server"
        assert symbols[1].name == "start"
        assert symbols[1].kind == "method"
        assert symbols[1].qualified_name == "mod.Server.start"


# ---------------------------------------------------------------------------
# Tests: interface + type alias
# ---------------------------------------------------------------------------
class TestInterfaceAndType:
    def test_extracts_interface(self):
        name = MockNode("identifier", "Config", (0, 10), (0, 16))
        iface = MockNode(
            "interface_declaration",
            "interface Config { port: number }",
            (0, 0),
            (0, 33),
            children=[name],
            fields={"name": name},
        )

        source = "interface Config { port: number }"
        symbols = []
        imports = []
        _walk_node(
            MockNode("program", children=[iface]), source.splitlines(), "mod", symbols, imports
        )

        assert len(symbols) == 1
        assert symbols[0].name == "Config"
        assert symbols[0].kind == "interface"

    def test_extracts_type_alias(self):
        name = MockNode("identifier", "ID", (0, 5), (0, 7))
        alias = MockNode(
            "type_alias_declaration",
            "type ID = string",
            (0, 0),
            (0, 16),
            children=[name],
            fields={"name": name},
        )

        source = "type ID = string"
        symbols = []
        imports = []
        _walk_node(
            MockNode("program", children=[alias]), source.splitlines(), "mod", symbols, imports
        )

        assert len(symbols) == 1
        assert symbols[0].name == "ID"
        assert symbols[0].kind == "type"


# ---------------------------------------------------------------------------
# Tests: imports
# ---------------------------------------------------------------------------
class TestImports:
    def test_extracts_named_import(self):
        spec_name = MockNode("identifier", "useState", (0, 10), (0, 18))
        spec = MockNode(
            "import_specifier",
            "useState",
            (0, 10),
            (0, 18),
            children=[spec_name],
            fields={"name": spec_name},
        )
        named = MockNode("named_imports", "{ useState }", (0, 9), (0, 21), children=[spec])
        clause = MockNode("import_clause", "{ useState }", (0, 7), (0, 21), children=[named])
        source_mod = MockNode("string", "'react'", (0, 27), (0, 34))
        imp = MockNode(
            "import_statement",
            "import { useState } from 'react'",
            (0, 0),
            (0, 34),
            children=[clause, source_mod],
            fields={"source": source_mod},
        )

        edge = _extract_import(imp)
        assert edge is not None
        assert edge.target_module == "react"
        assert "useState" in edge.import_names

    def test_no_source_returns_none(self):
        imp = MockNode("import_statement", "import x", fields={})
        assert _extract_import(imp) is None


# ---------------------------------------------------------------------------
# Tests: export wrapping
# ---------------------------------------------------------------------------
class TestExportStatement:
    def test_export_wraps_function(self):
        name = MockNode("identifier", "fetchData", (0, 25), (0, 34))
        params = MockNode("formal_parameters", "()", (0, 34), (0, 36))
        func = MockNode(
            "function_declaration",
            "function fetchData() {}",
            (0, 16),
            (0, 38),
            children=[name, params],
            fields={"name": name, "parameters": params},
        )
        export = MockNode(
            "export_statement",
            "export function fetchData() {}",
            (0, 0),
            (0, 38),
            children=[func],
        )

        source = "export function fetchData() {}"
        symbols = []
        imports = []
        _walk_node(
            MockNode("program", children=[export]), source.splitlines(), "mod", symbols, imports
        )

        assert len(symbols) == 1
        assert symbols[0].name == "fetchData"


# ---------------------------------------------------------------------------
# Tests: full extract() integration
# ---------------------------------------------------------------------------
class TestExtractIntegration:
    def test_extract_returns_ingest_result(self):
        name = MockNode("identifier", "main", (0, 9), (0, 13))
        params = MockNode("formal_parameters", "()", (0, 13), (0, 15))
        func = MockNode(
            "function_declaration",
            "function main() {}",
            (0, 0),
            (0, 18),
            children=[name, params],
            fields={"name": name, "parameters": params},
        )
        _make_tree([func])

        strategy = TypeScriptIngestStrategy()
        result = strategy.extract("function main() {}", "src/app.ts")

        assert len(result.symbols) == 1
        assert result.symbols[0].name == "main"
        assert result.symbols[0].qualified_name == "src.app.main"

    def test_extract_empty_source(self):
        _make_tree([])
        strategy = TypeScriptIngestStrategy()
        result = strategy.extract("", "empty.ts")
        assert result.symbols == []
        assert result.imports == []

    def test_extract_parse_error_returns_empty(self):
        _parser_mock.return_value.parse.side_effect = RuntimeError("parse failed")
        strategy = TypeScriptIngestStrategy()
        result = strategy.extract("invalid{{{", "bad.ts")
        assert result.symbols == []
        # Reset side effect
        _parser_mock.return_value.parse.side_effect = None
