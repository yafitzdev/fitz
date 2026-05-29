# tests/unit/test_java_extractor.py
"""Tests for Java extractor — tree-sitter mocked for unit tests."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock tree-sitter modules so tests run without optional dependency
# ---------------------------------------------------------------------------
_mock_ts = MagicMock()
_mock_ts_java = MagicMock()
sys.modules.setdefault("tree_sitter", _mock_ts)
sys.modules.setdefault("tree_sitter_java", _mock_ts_java)

import fitz_sage.engines.fitz_krag.ingestion.strategies.java as _java_mod  # noqa: E402
from fitz_sage.engines.fitz_krag.ingestion.code_utils import (  # noqa: E402
    node_text,
    path_to_module,
)
from fitz_sage.engines.fitz_krag.ingestion.strategies.java import (  # noqa: E402
    JavaIngestStrategy,
    _extract_import,
    _extract_package,
    _walk_node,
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
_parser_mock = _java_mod.Parser


def _make_tree(root_children):
    _parser_mock.return_value.parse.return_value.root_node = MockNode(
        "program", children=root_children
    )


# ---------------------------------------------------------------------------
# Tests: helpers
# ---------------------------------------------------------------------------
class TestHelpers:
    def test_path_to_module(self):
        assert path_to_module("com/example/App.java", (".java",)) == "com.example.App"

    def test_path_to_module_backslash(self):
        assert path_to_module("com\\example\\App.java", (".java",)) == "com.example.App"

    def test_node_text_bytes(self):
        assert node_text(MockNode("id", b"hello")) == "hello"

    def test_node_text_none(self):
        assert node_text(None) == ""

    def test_extract_package(self):
        pkg_text = MockNode("package_declaration", "package com.example;", (0, 0), (0, 20))
        root = MockNode("program", children=[pkg_text])
        assert _extract_package(root) == "com.example"

    def test_extract_package_missing(self):
        root = MockNode("program", children=[])
        assert _extract_package(root) is None


# ---------------------------------------------------------------------------
# Tests: class extraction
# ---------------------------------------------------------------------------
class TestClassExtraction:
    def test_extracts_class(self):
        name = MockNode("identifier", "UserService", (0, 13), (0, 24))
        body = MockNode("class_body", "{}", (0, 25), (0, 27), children=[])
        cls = MockNode(
            "class_declaration",
            "public class UserService {}",
            (0, 0),
            (0, 27),
            children=[name, body],
            fields={"name": name, "body": body},
        )

        source = "public class UserService {}"
        symbols = []
        imports = []
        _walk_node(
            MockNode("program", children=[cls]),
            source.splitlines(),
            "com.example",
            symbols,
            imports,
        )

        assert len(symbols) == 1
        assert symbols[0].name == "UserService"
        assert symbols[0].kind == "class"
        assert symbols[0].qualified_name == "com.example.UserService"

    def test_class_with_superclass(self):
        name = MockNode("identifier", "Admin", (0, 13), (0, 18))
        superclass = MockNode("type_identifier", "User", (0, 27), (0, 31))
        body = MockNode("class_body", "{}", (0, 32), (0, 34), children=[])
        cls = MockNode(
            "class_declaration",
            "public class Admin extends User {}",
            (0, 0),
            (0, 34),
            children=[name, superclass, body],
            fields={"name": name, "superclass": superclass, "body": body},
        )

        source = "public class Admin extends User {}"
        symbols = []
        imports = []
        _walk_node(
            MockNode("program", children=[cls]), source.splitlines(), "pkg", symbols, imports
        )

        assert symbols[0].name == "Admin"
        assert "extends User" in symbols[0].signature

    def test_class_with_method(self):
        cls_name = MockNode("identifier", "Calculator", (0, 13), (0, 23))
        method_name = MockNode("identifier", "add", (1, 15), (1, 18))
        ret_type = MockNode("integral_type", "int", (1, 11), (1, 14))
        params = MockNode("formal_parameters", "(int a, int b)", (1, 18), (1, 32))
        method = MockNode(
            "method_declaration",
            "public int add(int a, int b) { return a + b; }",
            (1, 4),
            (1, 51),
            children=[ret_type, method_name, params],
            fields={"name": method_name, "type": ret_type, "parameters": params},
        )
        body = MockNode("class_body", "{ ... }", (0, 24), (2, 1), children=[method])
        cls = MockNode(
            "class_declaration",
            "public class Calculator { ... }",
            (0, 0),
            (2, 1),
            children=[cls_name, body],
            fields={"name": cls_name, "body": body},
        )

        source = "public class Calculator {\n    public int add(int a, int b) { return a + b; }\n}"
        symbols = []
        imports = []
        _walk_node(
            MockNode("program", children=[cls]), source.splitlines(), "pkg", symbols, imports
        )

        assert len(symbols) == 2
        assert symbols[0].name == "Calculator"
        assert symbols[1].name == "add"
        assert symbols[1].kind == "method"
        assert symbols[1].qualified_name == "pkg.Calculator.add"
        assert "int" in symbols[1].signature


# ---------------------------------------------------------------------------
# Tests: interface + enum + record
# ---------------------------------------------------------------------------
class TestOtherTypes:
    def test_extracts_interface(self):
        name = MockNode("identifier", "Serializable", (0, 10), (0, 22))
        body = MockNode("interface_body", "{}", (0, 23), (0, 25), children=[])
        iface = MockNode(
            "interface_declaration",
            "interface Serializable {}",
            (0, 0),
            (0, 25),
            children=[name, body],
            fields={"name": name, "body": body},
        )

        source = "interface Serializable {}"
        symbols = []
        imports = []
        _walk_node(
            MockNode("program", children=[iface]), source.splitlines(), "pkg", symbols, imports
        )

        assert len(symbols) == 1
        assert symbols[0].name == "Serializable"
        assert symbols[0].kind == "interface"

    def test_extracts_enum(self):
        name = MockNode("identifier", "Color", (0, 5), (0, 10))
        enum = MockNode(
            "enum_declaration",
            "enum Color { RED, GREEN, BLUE }",
            (0, 0),
            (0, 31),
            children=[name],
            fields={"name": name},
        )

        source = "enum Color { RED, GREEN, BLUE }"
        symbols = []
        imports = []
        _walk_node(
            MockNode("program", children=[enum]), source.splitlines(), "pkg", symbols, imports
        )

        assert len(symbols) == 1
        assert symbols[0].name == "Color"
        assert symbols[0].kind == "enum"

    def test_extracts_record(self):
        name = MockNode("identifier", "Point", (0, 7), (0, 12))
        rec = MockNode(
            "record_declaration",
            "record Point(int x, int y) {}",
            (0, 0),
            (0, 29),
            children=[name],
            fields={"name": name},
        )

        source = "record Point(int x, int y) {}"
        symbols = []
        imports = []
        _walk_node(
            MockNode("program", children=[rec]), source.splitlines(), "pkg", symbols, imports
        )

        assert len(symbols) == 1
        assert symbols[0].name == "Point"
        assert symbols[0].kind == "record"


# ---------------------------------------------------------------------------
# Tests: constructor + field
# ---------------------------------------------------------------------------
class TestConstructorAndField:
    def test_extracts_constructor(self):
        cls_name = MockNode("identifier", "Server", (0, 13), (0, 19))
        ctor_name = MockNode("identifier", "Server", (1, 11), (1, 17))
        ctor_params = MockNode("formal_parameters", "(int port)", (1, 17), (1, 27))
        ctor = MockNode(
            "constructor_declaration",
            "public Server(int port) { this.port = port; }",
            (1, 4),
            (1, 50),
            children=[ctor_name, ctor_params],
            fields={"name": ctor_name, "parameters": ctor_params},
        )
        body = MockNode("class_body", "{ ... }", (0, 20), (2, 1), children=[ctor])
        cls = MockNode(
            "class_declaration",
            "public class Server { ... }",
            (0, 0),
            (2, 1),
            children=[cls_name, body],
            fields={"name": cls_name, "body": body},
        )

        source = "public class Server {\n    public Server(int port) { this.port = port; }\n}"
        symbols = []
        imports = []
        _walk_node(
            MockNode("program", children=[cls]), source.splitlines(), "pkg", symbols, imports
        )

        assert symbols[1].name == "Server"
        assert symbols[1].kind == "constructor"

    def test_extracts_field(self):
        cls_name = MockNode("identifier", "Config", (0, 13), (0, 19))
        field_name = MockNode("identifier", "timeout", (1, 16), (1, 23))
        var_decl = MockNode(
            "variable_declarator",
            "timeout",
            (1, 16),
            (1, 23),
            children=[field_name],
            fields={"name": field_name},
        )
        field = MockNode(
            "field_declaration",
            "private int timeout;",
            (1, 4),
            (1, 24),
            children=[var_decl],
        )
        body = MockNode("class_body", "{ ... }", (0, 20), (2, 1), children=[field])
        cls = MockNode(
            "class_declaration",
            "public class Config { ... }",
            (0, 0),
            (2, 1),
            children=[cls_name, body],
            fields={"name": cls_name, "body": body},
        )

        source = "public class Config {\n    private int timeout;\n}"
        symbols = []
        imports = []
        _walk_node(
            MockNode("program", children=[cls]), source.splitlines(), "pkg", symbols, imports
        )

        assert len(symbols) == 2
        assert symbols[1].name == "timeout"
        assert symbols[1].kind == "field"


# ---------------------------------------------------------------------------
# Tests: imports
# ---------------------------------------------------------------------------
class TestImports:
    def test_extracts_import(self):
        imp = MockNode(
            "import_declaration",
            "import com.example.Foo;",
            (0, 0),
            (0, 23),
        )
        edge = _extract_import(imp)
        assert edge is not None
        assert edge.target_module == "com.example"
        assert edge.import_names == ["Foo"]

    def test_extracts_wildcard_import(self):
        imp = MockNode(
            "import_declaration",
            "import com.example.*;",
            (0, 0),
            (0, 21),
        )
        edge = _extract_import(imp)
        assert edge is not None
        assert edge.target_module == "com.example"
        assert edge.import_names == []

    def test_extracts_static_import(self):
        imp = MockNode(
            "import_declaration",
            "import static org.junit.Assert.assertEquals;",
            (0, 0),
            (0, 45),
        )
        edge = _extract_import(imp)
        assert edge is not None
        assert edge.target_module == "org.junit.Assert"
        assert edge.import_names == ["assertEquals"]

    def test_non_import_returns_none(self):
        imp = MockNode("import_declaration", "not an import", (0, 0), (0, 13))
        assert _extract_import(imp) is None


# ---------------------------------------------------------------------------
# Tests: full extract() integration
# ---------------------------------------------------------------------------
class TestExtractIntegration:
    def test_extract_with_package(self):
        pkg = MockNode("package_declaration", "package com.example;", (0, 0), (0, 20))
        name = MockNode("identifier", "App", (1, 13), (1, 16))
        body = MockNode("class_body", "{}", (1, 17), (1, 19), children=[])
        cls = MockNode(
            "class_declaration",
            "public class App {}",
            (1, 0),
            (1, 19),
            children=[name, body],
            fields={"name": name, "body": body},
        )
        _make_tree([pkg, cls])

        strategy = JavaIngestStrategy()
        result = strategy.extract(
            "package com.example;\npublic class App {}", "com/example/App.java"
        )

        assert len(result.symbols) == 1
        assert result.symbols[0].qualified_name == "com.example.App"

    def test_extract_empty(self):
        _make_tree([])
        strategy = JavaIngestStrategy()
        result = strategy.extract("", "Empty.java")
        assert result.symbols == []

    def test_extract_parse_error(self):
        _parser_mock.return_value.parse.side_effect = RuntimeError("parse failed")
        strategy = JavaIngestStrategy()
        result = strategy.extract("{{invalid}}", "Bad.java")
        assert result.symbols == []
        _parser_mock.return_value.parse.side_effect = None
