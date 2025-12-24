# fitz_ai/ingest/enrichment/artifacts/analyzer.py
"""
Project analyzer for artifact generation.

Analyzes a Python project to extract:
- File structure and metadata
- Classes, functions, protocols
- Pydantic models
- Import relationships

This analysis is shared across all artifact generators.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set

from fitz_ai.ingest.enrichment.artifacts.base import (
    FileInfo,
    ProjectAnalysis,
)

logger = logging.getLogger(__name__)

# Directories to exclude from analysis
DEFAULT_EXCLUDES = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".idea", ".vscode",
    "dist", "build", "node_modules", ".tox", ".eggs",
}


@dataclass
class ExtractedClass:
    """Extracted class information."""
    name: str
    file_path: str
    bases: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    docstring: str | None = None
    is_protocol: bool = False
    is_pydantic: bool = False


@dataclass
class ExtractedFunction:
    """Extracted function information."""
    name: str
    file_path: str
    signature: str = ""
    docstring: str | None = None
    is_async: bool = False


class ProjectAnalyzer:
    """
    Analyzes a Python project for artifact generation.

    Usage:
        analyzer = ProjectAnalyzer(Path("/path/to/project"))
        analysis = analyzer.analyze()
    """

    def __init__(
        self,
        root: Path,
        excludes: Set[str] | None = None,
    ):
        self._root = Path(root).resolve()
        self._excludes = excludes or DEFAULT_EXCLUDES

    def analyze(self) -> ProjectAnalysis:
        """Analyze the project and return structured data."""
        logger.info(f"Analyzing project: {self._root}")

        analysis = ProjectAnalysis(root=self._root)

        # Collect all Python files
        py_files = list(self._iter_python_files())
        logger.debug(f"Found {len(py_files)} Python files")

        # Analyze each file
        for py_file in py_files:
            try:
                file_info = self._analyze_file(py_file)
                analysis.files.append(file_info)

                # Extract classes, functions, etc.
                self._extract_definitions(py_file, analysis)
            except Exception as e:
                logger.debug(f"Failed to analyze {py_file}: {e}")

        # Build import graph
        analysis.import_graph = self._build_import_graph(analysis.files)

        # Extract package names
        analysis.packages = self._extract_packages()

        logger.info(
            f"Analysis complete: {len(analysis.files)} files, "
            f"{len(analysis.classes)} classes, {len(analysis.protocols)} protocols"
        )

        return analysis

    def _iter_python_files(self):
        """Iterate over all Python files in the project."""
        for py_file in self._root.rglob("*.py"):
            rel = py_file.relative_to(self._root)
            if any(part in self._excludes for part in rel.parts):
                continue
            yield py_file

    def _analyze_file(self, path: Path) -> FileInfo:
        """Analyze a single Python file."""
        rel_path = str(path.relative_to(self._root))
        content = path.read_text(encoding="utf-8")

        imports: List[str] = []
        exports: List[str] = []
        docstring: str | None = None

        try:
            tree = ast.parse(content, filename=str(path))
            docstring = ast.get_docstring(tree)

            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            # Extract top-level exports
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    exports.append(f"class {node.name}")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not node.name.startswith("_"):
                        exports.append(f"def {node.name}")

        except SyntaxError:
            pass

        return FileInfo(
            path=str(path),
            relative_path=rel_path,
            extension=path.suffix,
            size_bytes=path.stat().st_size,
            imports=imports,
            exports=exports,
            docstring=docstring,
        )

    def _extract_definitions(self, path: Path, analysis: ProjectAnalysis) -> None:
        """Extract class and function definitions from a file."""
        content = path.read_text(encoding="utf-8")
        rel_path = str(path.relative_to(self._root))

        try:
            tree = ast.parse(content, filename=str(path))
        except SyntaxError:
            return

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                cls_info = self._extract_class(node, rel_path)

                if cls_info["is_protocol"]:
                    analysis.protocols.append(cls_info)
                elif cls_info["is_pydantic"]:
                    analysis.models.append(cls_info)
                else:
                    analysis.classes.append(cls_info)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_"):
                    func_info = self._extract_function(node, rel_path)
                    analysis.functions.append(func_info)

    def _extract_class(self, node: ast.ClassDef, file_path: str) -> Dict[str, Any]:
        """Extract class information."""
        bases = []
        is_protocol = False
        is_pydantic = False

        for base in node.bases:
            base_name = self._get_name(base)
            bases.append(base_name)
            if base_name == "Protocol":
                is_protocol = True
            if base_name in ("BaseModel", "BaseSettings"):
                is_pydantic = True

        # Check decorators for Protocol
        for decorator in node.decorator_list:
            dec_name = self._get_name(decorator)
            if dec_name == "runtime_checkable":
                is_protocol = True

        methods = []
        fields = []

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not item.name.startswith("_") or item.name in ("__init__", "__call__"):
                    sig = self._get_function_signature(item)
                    methods.append({"name": item.name, "signature": sig})

            # Extract Pydantic fields
            if is_pydantic and isinstance(item, ast.AnnAssign):
                if isinstance(item.target, ast.Name):
                    field_name = item.target.id
                    field_type = self._get_annotation_str(item.annotation)
                    fields.append({"name": field_name, "type": field_type})

        return {
            "name": node.name,
            "file_path": file_path,
            "bases": bases,
            "methods": methods,
            "fields": fields,
            "docstring": ast.get_docstring(node),
            "is_protocol": is_protocol,
            "is_pydantic": is_pydantic,
        }

    def _extract_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, file_path: str) -> Dict[str, Any]:
        """Extract function information."""
        return {
            "name": node.name,
            "file_path": file_path,
            "signature": self._get_function_signature(node),
            "docstring": ast.get_docstring(node),
            "is_async": isinstance(node, ast.AsyncFunctionDef),
        }

    def _get_function_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """Get function signature as string."""
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {self._get_annotation_str(arg.annotation)}"
            args.append(arg_str)

        sig = f"({', '.join(args)})"

        if node.returns:
            sig += f" -> {self._get_annotation_str(node.returns)}"

        return sig

    def _get_annotation_str(self, node: ast.expr | None) -> str:
        """Convert annotation AST node to string."""
        if node is None:
            return "Any"
        return self._get_name(node)

    def _get_name(self, node: ast.expr) -> str:
        """Get name from various AST node types."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return f"{self._get_name(node.value)}[{self._get_name(node.slice)}]"
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Tuple):
            return ", ".join(self._get_name(e) for e in node.elts)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return f"{self._get_name(node.left)} | {self._get_name(node.right)}"
        elif isinstance(node, ast.Call):
            return self._get_name(node.func)
        else:
            return "..."

    def _build_import_graph(self, files: List[FileInfo]) -> Dict[str, List[str]]:
        """Build a graph of file imports."""
        graph: Dict[str, List[str]] = {}
        for file_info in files:
            graph[file_info.relative_path] = file_info.imports
        return graph

    def _extract_packages(self) -> List[str]:
        """Extract top-level package names."""
        packages = []
        for item in self._root.iterdir():
            if item.is_dir() and (item / "__init__.py").exists():
                packages.append(item.name)
        return packages


__all__ = ["ProjectAnalyzer", "ExtractedClass", "ExtractedFunction"]
