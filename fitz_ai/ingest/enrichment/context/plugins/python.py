# fitz_ai/ingest/enrichment/context/plugins/python.py
"""
Python-specific context builder plugin.

Builds rich context for Python files including:
- Imports (what this file depends on)
- Exports (classes, functions, constants defined)
- Used by (what files import this one)
- Module docstring
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

from fitz_ai.ingest.enrichment.base import (
    CodeEnrichmentContext,
    ContentType,
    EnrichmentContext,
)

logger = logging.getLogger(__name__)

plugin_name = "python"
plugin_type = "context"
supported_extensions = {".py", ".pyw"}


DEFAULT_EXCLUDES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    "dist",
    "build",
    "node_modules",
    ".tox",
    ".eggs",
}


@dataclass
class FileAnalysis:
    """Analysis results for a single Python file."""

    path: str
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    docstring: str | None = None


class PythonProjectAnalyzer:
    """
    Analyzes a Python project to build import/export graphs.

    This analyzer:
    1. Scans all Python files in the project
    2. Extracts imports and exports from each file
    3. Builds a reverse lookup (used_by) for each file
    """

    def __init__(self, root: Path, excludes: Set[str] | None = None):
        self._root = Path(root).resolve()
        self._excludes = excludes or DEFAULT_EXCLUDES
        self._file_analyses: Dict[str, FileAnalysis] = {}
        self._import_graph: Dict[str, Set[str]] = {}
        self._used_by: Dict[str, Set[str]] = {}
        self._analyzed = False

    def analyze(self) -> None:
        """Analyze all Python files in the project."""
        if self._analyzed:
            return

        logger.info(f"Analyzing Python project: {self._root}")

        for py_file in self._iter_python_files():
            try:
                analysis = self._analyze_file(py_file)
                self._file_analyses[str(py_file)] = analysis
            except Exception as e:
                logger.debug(f"Failed to analyze {py_file}: {e}")

        self._build_graphs()
        self._analyzed = True
        logger.info(f"Analyzed {len(self._file_analyses)} Python files")

    def _iter_python_files(self):
        """Iterate over all Python files in the project."""
        for py_file in self._root.rglob("*.py"):
            rel = py_file.relative_to(self._root)
            if any(part in self._excludes for part in rel.parts):
                continue
            yield py_file

    def _analyze_file(self, file_path: Path) -> FileAnalysis:
        """Analyze a single Python file."""
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(file_path))

        imports = self._extract_imports(tree, file_path)
        exports = self._extract_exports(tree)
        docstring = ast.get_docstring(tree)

        return FileAnalysis(
            path=str(file_path),
            imports=imports,
            exports=exports,
            docstring=docstring,
        )

    def _extract_imports(self, tree: ast.Module, file_path: Path) -> List[str]:
        """Extract import statements from AST."""
        imports: List[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                level = node.level

                if level > 0:
                    resolved = self._resolve_relative_import(file_path, module, level)
                    if resolved:
                        imports.append(resolved)
                elif module:
                    imports.append(module)

        return imports

    def _resolve_relative_import(
        self, file_path: Path, module: str, level: int
    ) -> str | None:
        """Resolve a relative import to an absolute module name."""
        try:
            rel_path = file_path.relative_to(self._root)
            parts = list(rel_path.parts)

            if parts and parts[-1].endswith(".py"):
                if parts[-1] == "__init__.py":
                    parts = parts[:-1]
                else:
                    parts[-1] = parts[-1][:-3]

            if level > 0:
                parts = parts[:-level] if len(parts) >= level else []

            if module:
                parts.extend(module.split("."))

            return ".".join(parts) if parts else None
        except Exception:
            return None

    def _extract_exports(self, tree: ast.Module) -> List[str]:
        """Extract exported symbols from AST."""
        exports: List[str] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                exports.append(f"class {node.name}")
            elif isinstance(node, ast.FunctionDef):
                if not node.name.startswith("_"):
                    exports.append(f"def {node.name}")
            elif isinstance(node, ast.AsyncFunctionDef):
                if not node.name.startswith("_"):
                    exports.append(f"async def {node.name}")
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        if name.isupper() or name == "__all__":
                            exports.append(f"const {name}")

        return exports

    def _build_graphs(self) -> None:
        """Build import graph and used_by reverse lookup."""
        module_to_file: Dict[str, str] = {}
        for file_path in self._file_analyses:
            module_name = self._file_to_module(file_path)
            if module_name:
                module_to_file[module_name] = file_path

        for file_path, analysis in self._file_analyses.items():
            imported_files: Set[str] = set()

            for imp in analysis.imports:
                if imp in module_to_file:
                    imported_files.add(module_to_file[imp])
                else:
                    for mod, path in module_to_file.items():
                        if mod.startswith(imp + ".") or imp.startswith(mod + "."):
                            imported_files.add(path)
                            break

            self._import_graph[file_path] = imported_files

            for imported in imported_files:
                if imported not in self._used_by:
                    self._used_by[imported] = set()
                self._used_by[imported].add(file_path)

    def _file_to_module(self, file_path: str) -> str | None:
        """Convert file path to module name."""
        try:
            path = Path(file_path)
            rel = path.relative_to(self._root)
            parts = list(rel.parts)

            if parts[-1] == "__init__.py":
                parts = parts[:-1]
            else:
                parts[-1] = parts[-1].removesuffix(".py")

            return ".".join(parts) if parts else None
        except Exception:
            return None

    def get_analysis(self, file_path: str) -> FileAnalysis | None:
        """Get analysis for a specific file."""
        if not self._analyzed:
            self.analyze()
        return self._file_analyses.get(str(Path(file_path).resolve()))

    def get_used_by(self, file_path: str) -> List[Tuple[str, str]]:
        """Get list of files that import the given file."""
        if not self._analyzed:
            self.analyze()

        resolved = str(Path(file_path).resolve())
        importers = self._used_by.get(resolved, set())

        result: List[Tuple[str, str]] = []
        for importer in importers:
            role = self._infer_role(importer)
            result.append((importer, role))

        return sorted(result)

    def _infer_role(self, file_path: str) -> str:
        """Infer the role of a file based on its path."""
        path = Path(file_path)
        parts = path.parts
        name = path.stem

        if "test" in parts or name.startswith("test_"):
            return "test"
        if "cli" in parts or "commands" in parts:
            return "CLI"
        if "api" in parts or "routes" in parts:
            return "API"
        if "examples" in parts:
            return "example"
        if name == "__main__":
            return "entrypoint"

        return path.parent.name if path.parent.name else "module"


class Builder:
    """
    Builds enrichment context for Python files.

    Uses a PythonProjectAnalyzer to provide rich context including
    imports, exports, and reverse dependency information.
    """

    supported_extensions = supported_extensions

    def __init__(self, analyzer: PythonProjectAnalyzer | None = None):
        self._analyzer = analyzer

    def build(self, file_path: str, content: str) -> EnrichmentContext:
        """Build enrichment context for a Python file."""
        analysis = None
        used_by: List[Tuple[str, str]] = []

        if self._analyzer:
            analysis = self._analyzer.get_analysis(file_path)
            used_by = self._analyzer.get_used_by(file_path)

        if analysis is None:
            analysis = self._quick_analyze(file_path, content)

        return CodeEnrichmentContext(
            file_path=file_path,
            content_type=ContentType.PYTHON,
            language="python",
            imports=analysis.imports,
            exports=analysis.exports,
            used_by=used_by,
            docstring=analysis.docstring,
        )

    def _quick_analyze(self, file_path: str, content: str) -> FileAnalysis:
        """Quick analysis when no pre-built analyzer is available."""
        try:
            tree = ast.parse(content, filename=file_path)

            imports: List[str] = []
            exports: List[str] = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    exports.append(f"class {node.name}")
                elif isinstance(node, ast.FunctionDef):
                    if not node.name.startswith("_"):
                        exports.append(f"def {node.name}")

            docstring = ast.get_docstring(tree)

            return FileAnalysis(
                path=file_path,
                imports=imports,
                exports=exports,
                docstring=docstring,
            )
        except SyntaxError:
            return FileAnalysis(path=file_path)
