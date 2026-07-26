# tests/unit/test_go_extractor.py
"""Tests for Go extractor — tree-sitter mocked for unit tests."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock tree-sitter modules so tests run without optional dependency
# ---------------------------------------------------------------------------
_mock_ts = MagicMock()
_mock_ts_go = MagicMock()
sys.modules.setdefault("tree_sitter", _mock_ts)
sys.modules.setdefault("tree_sitter_go", _mock_ts_go)

import fitz_sage.engines.fitz_krag.ingestion.strategies.go as _go_mod  # noqa: E402
from fitz_sage.engines.fitz_krag.ingestion.code_utils import (  # noqa: E402
    node_text,
    path_to_module,
)
from fitz_sage.engines.fitz_krag.ingestion.strategies.go import (  # noqa: E402
    GoIngestStrategy,
    _extract_imports,
    _extract_package,
    _extract_receiver_type,
    _regex_fallback,
)


# ---------------------------------------------------------------------------
# MockNode
# ---------------------------------------------------------------------------
class MockNode:
    def __init__(self, type_, text="", start=(0, 0), end=(0, 0), children=None, fields=None):
        self.type = type_
        self.text = text.encode("utf-8") if isinstance(text, str) else text
        self.start_point = start
        self.end_point = end
        self.children = children or []
        self._fields = fields or {}

    def child_by_field_name(self, name):
        return self._fields.get(name)


# Reference the actual Parser mock the module imported
_parser_mock = _go_mod.Parser


def _make_tree(root_children):
    root = MockNode("source_file", children=root_children)
    _parser_mock.return_value.parse.return_value.root_node = root
    return root


# ---------------------------------------------------------------------------
# Tests: helpers
# ---------------------------------------------------------------------------
class TestHelpers:
    def test_path_to_module(self):
        assert path_to_module("cmd/server/main.go", (".go",)) == "cmd.server.main"

    def test_path_to_module_backslash(self):
        assert path_to_module("pkg\\handler.go", (".go",)) == "pkg.handler"

    def test_node_text_bytes(self):
        assert node_text(MockNode("id", b"hello")) == "hello"

    def test_node_text_none(self):
        assert node_text(None) == ""

    def test_extract_package(self):
        pkg_id = MockNode("package_identifier", "main", (0, 8), (0, 12))
        pkg = MockNode("package_clause", "package main", (0, 0), (0, 12), children=[pkg_id])
        root = MockNode("source_file", children=[pkg])
        assert _extract_package(root) == "main"

    def test_extract_package_missing(self):
        assert _extract_package(MockNode("source_file", children=[])) is None

    def test_extract_receiver_type_pointer(self):
        node = MockNode("parameter_list", "(s *Server)", (0, 0), (0, 11))
        assert _extract_receiver_type(node) == "Server"

    def test_extract_receiver_type_value(self):
        node = MockNode("parameter_list", "(s Server)", (0, 0), (0, 10))
        assert _extract_receiver_type(node) == "Server"

    def test_extract_receiver_type_none(self):
        assert _extract_receiver_type(None) is None


# ---------------------------------------------------------------------------
# Tests: function extraction
# ---------------------------------------------------------------------------
class TestFunctionExtraction:
    def test_extracts_function(self):
        name = MockNode("identifier", "main", (0, 5), (0, 9))
        params = MockNode("parameter_list", "()", (0, 9), (0, 11))
        func = MockNode(
            "function_declaration",
            "func main() {}",
            (0, 0),
            (0, 14),
            children=[name, params],
            fields={"name": name, "parameters": params},
        )
        _make_tree([func])

        strategy = GoIngestStrategy()
        result = strategy.extract("func main() {}", "main.go")

        assert len(result.symbols) == 1
        assert result.symbols[0].name == "main"
        assert result.symbols[0].kind == "function"
        assert "func main" in result.symbols[0].signature

    def test_function_with_return_type(self):
        name = MockNode("identifier", "Add", (0, 5), (0, 8))
        params = MockNode("parameter_list", "(a, b int)", (0, 8), (0, 18))
        ret = MockNode("type_identifier", "int", (0, 19), (0, 22))
        func = MockNode(
            "function_declaration",
            "func Add(a, b int) int { return a + b }",
            (0, 0),
            (0, 39),
            children=[name, params, ret],
            fields={"name": name, "parameters": params, "result": ret},
        )
        _make_tree([func])

        strategy = GoIngestStrategy()
        result = strategy.extract("func Add(a, b int) int { return a + b }", "math.go")

        assert result.symbols[0].name == "Add"
        assert "int" in result.symbols[0].signature


# ---------------------------------------------------------------------------
# Tests: method extraction (with receiver)
# ---------------------------------------------------------------------------
class TestMethodExtraction:
    def test_extracts_method_with_pointer_receiver(self):
        name = MockNode("identifier", "Handle", (0, 21), (0, 27))
        receiver = MockNode("parameter_list", "(s *Server)", (0, 5), (0, 16))
        params = MockNode("parameter_list", "(req Request)", (0, 27), (0, 40))
        method = MockNode(
            "method_declaration",
            "func (s *Server) Handle(req Request) {}",
            (0, 0),
            (0, 41),
            children=[receiver, name, params],
            fields={"name": name, "receiver": receiver, "parameters": params},
        )
        _make_tree([method])

        strategy = GoIngestStrategy()
        source = "func (s *Server) Handle(req Request) {}"
        result = strategy.extract(source, "server.go")

        assert len(result.symbols) == 1
        sym = result.symbols[0]
        assert sym.name == "Handle"
        assert sym.kind == "method"
        assert "Server" in sym.qualified_name


# ---------------------------------------------------------------------------
# Tests: type declarations
# ---------------------------------------------------------------------------
class TestTypeExtraction:
    def test_extracts_struct(self):
        type_name = MockNode("type_identifier", "Server", (0, 5), (0, 11))
        struct_type = MockNode("struct_type", "struct { port int }", (0, 12), (0, 31))
        type_spec = MockNode(
            "type_spec",
            "Server struct { port int }",
            (0, 5),
            (0, 31),
            children=[type_name, struct_type],
            fields={"name": type_name, "type": struct_type},
        )
        type_decl = MockNode(
            "type_declaration",
            "type Server struct { port int }",
            (0, 0),
            (0, 31),
            children=[type_spec],
        )
        _make_tree([type_decl])

        strategy = GoIngestStrategy()
        result = strategy.extract("type Server struct { port int }", "server.go")

        assert len(result.symbols) == 1
        assert result.symbols[0].name == "Server"
        assert result.symbols[0].kind == "struct"
        assert "type Server struct" in result.symbols[0].signature

    def test_extracts_interface(self):
        type_name = MockNode("type_identifier", "Handler", (0, 5), (0, 12))
        iface_type = MockNode("interface_type", "interface { Handle() }", (0, 13), (0, 35))
        type_spec = MockNode(
            "type_spec",
            "Handler interface { Handle() }",
            (0, 5),
            (0, 35),
            children=[type_name, iface_type],
            fields={"name": type_name, "type": iface_type},
        )
        type_decl = MockNode(
            "type_declaration",
            "type Handler interface { Handle() }",
            (0, 0),
            (0, 35),
            children=[type_spec],
        )
        _make_tree([type_decl])

        strategy = GoIngestStrategy()
        result = strategy.extract("type Handler interface { Handle() }", "handler.go")

        assert len(result.symbols) == 1
        assert result.symbols[0].name == "Handler"
        assert result.symbols[0].kind == "interface"


# ---------------------------------------------------------------------------
# Tests: const / var
# ---------------------------------------------------------------------------
class TestConstVar:
    def test_extracts_const(self):
        const_name = MockNode("identifier", "MaxRetries", (0, 6), (0, 16))
        const_spec = MockNode(
            "const_spec",
            "MaxRetries = 3",
            (0, 6),
            (0, 20),
            children=[const_name],
            fields={"name": const_name},
        )
        const_decl = MockNode(
            "const_declaration",
            "const MaxRetries = 3",
            (0, 0),
            (0, 20),
            children=[const_spec],
        )
        _make_tree([const_decl])

        strategy = GoIngestStrategy()
        result = strategy.extract("const MaxRetries = 3", "config.go")

        assert len(result.symbols) == 1
        assert result.symbols[0].name == "MaxRetries"
        assert result.symbols[0].kind == "constant"

    def test_extracts_var(self):
        var_name = MockNode("identifier", "logger", (0, 4), (0, 10))
        var_spec = MockNode(
            "var_spec",
            "logger = log.New()",
            (0, 4),
            (0, 22),
            children=[var_name],
            fields={"name": var_name},
        )
        var_decl = MockNode(
            "var_declaration",
            "var logger = log.New()",
            (0, 0),
            (0, 22),
            children=[var_spec],
        )
        _make_tree([var_decl])

        strategy = GoIngestStrategy()
        result = strategy.extract("var logger = log.New()", "main.go")

        assert len(result.symbols) == 1
        assert result.symbols[0].name == "logger"
        assert result.symbols[0].kind == "variable"


# ---------------------------------------------------------------------------
# Tests: imports
# ---------------------------------------------------------------------------
class TestImports:
    def test_single_import(self):
        path = MockNode("interpreted_string_literal", '"fmt"', (0, 7), (0, 12))
        spec = MockNode(
            "import_spec",
            '"fmt"',
            (0, 7),
            (0, 12),
            children=[path],
            fields={"path": path},
        )
        imp = MockNode(
            "import_declaration",
            'import "fmt"',
            (0, 0),
            (0, 12),
            children=[spec],
        )

        edges = _extract_imports(imp)
        assert len(edges) == 1
        assert edges[0].target_module == "fmt"
        assert edges[0].import_names == ["fmt"]

    def test_grouped_imports(self):
        path1 = MockNode("interpreted_string_literal", '"fmt"', (1, 1), (1, 6))
        spec1 = MockNode(
            "import_spec", '"fmt"', (1, 1), (1, 6), children=[path1], fields={"path": path1}
        )
        path2 = MockNode("interpreted_string_literal", '"net/http"', (2, 1), (2, 11))
        spec2 = MockNode(
            "import_spec", '"net/http"', (2, 1), (2, 11), children=[path2], fields={"path": path2}
        )
        spec_list = MockNode(
            "import_spec_list",
            '(\n"fmt"\n"net/http"\n)',
            (0, 7),
            (3, 1),
            children=[spec1, spec2],
        )
        imp = MockNode(
            "import_declaration",
            'import (\n"fmt"\n"net/http"\n)',
            (0, 0),
            (3, 1),
            children=[spec_list],
        )

        edges = _extract_imports(imp)
        assert len(edges) == 2
        assert edges[0].target_module == "fmt"
        assert edges[1].target_module == "net/http"
        assert edges[1].import_names == ["http"]


# ---------------------------------------------------------------------------
# Tests: full extract() integration
# ---------------------------------------------------------------------------
class TestExtractIntegration:
    def test_extract_with_package(self):
        pkg_id = MockNode("package_identifier", "server", (0, 8), (0, 14))
        pkg = MockNode("package_clause", "package server", (0, 0), (0, 14), children=[pkg_id])
        name = MockNode("identifier", "New", (1, 5), (1, 8))
        params = MockNode("parameter_list", "()", (1, 8), (1, 10))
        func = MockNode(
            "function_declaration",
            "func New() *Server {}",
            (1, 0),
            (1, 21),
            children=[name, params],
            fields={"name": name, "parameters": params},
        )
        _make_tree([pkg, func])

        strategy = GoIngestStrategy()
        result = strategy.extract("package server\nfunc New() *Server {}", "server.go")

        assert len(result.symbols) == 1
        assert result.symbols[0].qualified_name == "server.New"

    def test_extract_empty(self):
        _make_tree([])
        strategy = GoIngestStrategy()
        result = strategy.extract("", "empty.go")
        assert result.symbols == []

    def test_extract_parse_error(self):
        _parser_mock.return_value.parse.side_effect = RuntimeError("parse failed")
        strategy = GoIngestStrategy()
        result = strategy.extract("{{invalid}}", "bad.go")
        assert result.symbols == []
        _parser_mock.return_value.parse.side_effect = None

    def test_regex_fallback_preserves_doc_comment_and_complete_body(self):
        source = """// AuthMiddleware validates Bearer tokens.
// It reads the Authorization header.
func AuthMiddleware(next Handler) Handler {
    if next == nil {
        return defaultHandler
    }
    return next
}
"""

        result = _regex_fallback(source, "server.go")
        symbol = result.symbols[0]

        assert symbol.start_line == 1
        assert symbol.end_line == 8
        assert "Bearer" in symbol.docstring
        assert "Authorization" in symbol.docstring
