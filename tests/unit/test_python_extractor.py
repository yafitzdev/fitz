# tests/unit/test_python_extractor.py
"""Tests for Python AST extraction strategy."""

import pytest

from fitz_sage.engines.fitz_krag.ingestion.strategies.python_code import (
    PythonCodeIngestStrategy,
)


@pytest.fixture
def strategy():
    return PythonCodeIngestStrategy()


def _one(symbols, kind):
    """The single symbol of the given kind — test sources have exactly one."""
    matches = [s for s in symbols if s.kind == kind]
    assert len(matches) == 1, f"expected exactly one {kind}, got {len(matches)}"
    return matches[0]


class TestContentTypes:
    def test_handles_python(self, strategy):
        assert ".py" in strategy.content_types()


class TestFunctionExtraction:
    def test_simple_function(self, strategy):
        source = '''def hello():
    """Say hello."""
    return "world"
'''
        result = strategy.extract(source, "test.py")
        sym = _one(result.symbols, "function")
        assert sym.name == "hello"
        assert sym.kind == "function"
        assert sym.start_line == 1
        assert sym.end_line == 3
        assert "def hello()" in sym.signature

    def test_function_with_args(self, strategy):
        source = "def add(a: int, b: int) -> int:\n    return a + b\n"
        result = strategy.extract(source, "math.py")
        sym = _one(result.symbols, "function")
        assert "a: int" in sym.signature
        assert "b: int" in sym.signature
        assert "-> int" in sym.signature

    def test_async_function(self, strategy):
        source = "async def fetch(url: str) -> str:\n    pass\n"
        result = strategy.extract(source, "net.py")
        sym = _one(result.symbols, "function")
        assert sym.name == "fetch"
        assert "async def" in sym.signature

    def test_function_with_defaults(self, strategy):
        source = "def greet(name: str, greeting: str = 'hello') -> str:\n    return f'{greeting} {name}'\n"
        result = strategy.extract(source, "test.py")
        sym = _one(result.symbols, "function")
        assert "greeting: str=..." in sym.signature


class TestClassExtraction:
    def test_simple_class(self, strategy):
        source = '''class Dog:
    """A dog."""
    def __init__(self, name):
        self.name = name

    def bark(self):
        return "woof"
'''
        result = strategy.extract(source, "animals.py")
        symbols = result.symbols
        # Class + __init__ + bark
        assert len([s for s in symbols if s.kind != "module"]) == 3
        class_sym = [s for s in symbols if s.kind == "class"][0]
        assert class_sym.name == "Dog"
        methods = [s for s in symbols if s.kind == "method"]
        assert len(methods) == 2
        names = {m.name for m in methods}
        assert "__init__" in names
        assert "bark" in names

    def test_class_with_bases(self, strategy):
        source = "class MyEngine(KnowledgeEngine):\n    pass\n"
        result = strategy.extract(source, "engine.py")
        class_sym = _one(result.symbols, "class")
        assert "KnowledgeEngine" in class_sym.signature

    def test_method_qualified_name(self, strategy):
        source = "class Foo:\n    def bar(self):\n        pass\n"
        result = strategy.extract(source, "foo.py")
        method = [s for s in result.symbols if s.kind == "method"][0]
        assert "foo.Foo.bar" in method.qualified_name


class TestConstantExtraction:
    def test_upper_case_constant(self, strategy):
        source = "MAX_RETRIES = 3\nTIMEOUT = 30\n"
        result = strategy.extract(source, "config.py")
        constants = [s for s in result.symbols if s.kind == "constant"]
        assert len(constants) == 2
        names = {c.name for c in constants}
        assert "MAX_RETRIES" in names
        assert "TIMEOUT" in names

    def test_lower_case_ignored(self, strategy):
        source = "my_var = 42\n"
        result = strategy.extract(source, "test.py")
        constants = [s for s in result.symbols if s.kind == "constant"]
        assert len(constants) == 0


class TestImportExtraction:
    def test_import_statement(self, strategy):
        source = "import os\nimport sys\n"
        result = strategy.extract(source, "test.py")
        assert len(result.imports) == 2
        modules = {imp.target_module for imp in result.imports}
        assert "os" in modules
        assert "sys" in modules

    def test_from_import(self, strategy):
        source = "from pathlib import Path\n"
        result = strategy.extract(source, "test.py")
        assert len(result.imports) == 1
        assert result.imports[0].target_module == "pathlib"
        assert "Path" in result.imports[0].import_names

    def test_multiple_from_import(self, strategy):
        source = "from typing import Any, Dict, List\n"
        result = strategy.extract(source, "test.py")
        imp = result.imports[0]
        assert "Any" in imp.import_names
        assert "Dict" in imp.import_names
        assert "List" in imp.import_names

    def test_relative_import_single_dot(self, strategy):
        """``from .models import User`` in ``src/utils/helpers.py`` resolves to ``src.utils.models``."""
        source = "from .models import User\n"
        result = strategy.extract(source, "src/utils/helpers.py")
        assert len(result.imports) == 1
        assert result.imports[0].target_module == "src.utils.models"
        assert "User" in result.imports[0].import_names

    def test_relative_import_double_dot(self, strategy):
        """``from ..core import Base`` in ``src/utils/helpers.py`` resolves to ``src.core``."""
        source = "from ..core import Base\n"
        result = strategy.extract(source, "src/utils/helpers.py")
        assert len(result.imports) == 1
        assert result.imports[0].target_module == "src.core"
        assert "Base" in result.imports[0].import_names

    def test_relative_import_in_init(self, strategy):
        """``from .helpers import foo`` in ``src/utils/__init__.py`` resolves to ``src.utils.helpers``."""
        source = "from .helpers import foo\n"
        result = strategy.extract(source, "src/utils/__init__.py")
        assert len(result.imports) == 1
        assert result.imports[0].target_module == "src.utils.helpers"
        assert "foo" in result.imports[0].import_names

    def test_relative_import_no_module(self, strategy):
        """``from . import models`` (module=None) resolves using import names."""
        source = "from . import models\n"
        result = strategy.extract(source, "src/utils/helpers.py")
        assert len(result.imports) == 1
        assert result.imports[0].target_module == "src.utils"
        assert "models" in result.imports[0].import_names

    def test_relative_import_beyond_root(self, strategy):
        """Relative import that goes above root is skipped."""
        source = "from ....way.too.high import X\n"
        result = strategy.extract(source, "pkg/mod.py")
        assert len(result.imports) == 0

    def test_absolute_import_unchanged(self, strategy):
        """Absolute imports still work as before."""
        source = "from fitz_sage.core import Query\n"
        result = strategy.extract(source, "src/engine.py")
        assert len(result.imports) == 1
        assert result.imports[0].target_module == "fitz_sage.core"


class TestSyntaxError:
    def test_invalid_syntax_uses_regex_fallback(self, strategy):
        source = "def broken(:\n    pass\n"
        result = strategy.extract(source, "broken.py")
        # Regex fallback extracts function signature even from invalid syntax
        assert len(result.symbols) == 1
        assert result.symbols[0].name == "broken"
        assert len(result.imports) == 0


class TestModulePath:
    def test_path_to_module(self, strategy):
        source = "def func(): pass\n"
        result = strategy.extract(source, "fitz_sage/engines/krag/engine.py")
        sym = _one(result.symbols, "function")
        assert sym.qualified_name == "fitz_sage.engines.krag.engine.func"

    def test_init_path(self, strategy):
        source = "X = 1\n"
        # __init__.py should not include __init__ in the module name
        strategy.extract(source, "pkg/__init__.py")
        # Constants with single char are not UPPER_CASE per regex
        # Use a proper constant name
        source2 = "MAX = 1\n"
        result2 = strategy.extract(source2, "pkg/__init__.py")
        sym = _one(result2.symbols, "constant")
        assert sym.qualified_name == "pkg.MAX"


class TestComplexExtraction:
    def test_full_module(self, strategy):
        source = '''"""Module docstring."""

import logging
from pathlib import Path

MAX_SIZE = 1024

logger = logging.getLogger(__name__)


def helper(x):
    """Help function."""
    return x * 2


class Processor:
    """Processes things."""

    def __init__(self, config):
        self.config = config

    def run(self, data):
        return helper(data)
'''
        result = strategy.extract(source, "processor.py")
        names = {s.name for s in result.symbols}
        assert "MAX_SIZE" in names
        assert "helper" in names
        assert "Processor" in names
        assert "__init__" in names
        assert "run" in names
        assert len(result.imports) == 2  # logging, pathlib

    def test_references_extracted(self, strategy):
        source = "def process(data):\n    result = transform(data)\n    return result\n"
        result = strategy.extract(source, "test.py")
        sym = _one(result.symbols, "function")
        assert "transform" in sym.references


class TestModuleExtraction:
    def test_module_symbol_carries_docstring(self, strategy):
        source = '"""What this module does."""\n\ndef f():\n    pass\n'
        result = strategy.extract(source, "pkg/thing.py")
        mod = _one(result.symbols, "module")
        assert mod.name == "thing"
        assert mod.qualified_name == "pkg.thing"
        assert mod.docstring == "What this module does."

    def test_module_symbol_without_docstring(self, strategy):
        source = "def f():\n    pass\n"
        result = strategy.extract(source, "pkg/thing.py")
        mod = _one(result.symbols, "module")
        assert mod.docstring == ""
