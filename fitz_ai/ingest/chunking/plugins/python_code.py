# fitz_ai/ingest/chunking/plugins/python_code.py
"""
Python code-aware chunker.

Splits Python files by logical units:
- Classes (with their methods)
- Functions (standalone)
- Module-level code blocks

Preserves:
- Docstrings with their associated code
- Decorator chains
- Import blocks

Chunker ID format: "python_code:{max_chunk_size}:{include_imports}"
Example: "python_code:2000:1"
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from fitz_ai.engines.classic_rag.models.chunk import Chunk


@dataclass
class PythonCodeChunker:
    """
    Python code-aware chunker using AST parsing.

    Strategy:
    1. Parse Python AST to find classes and functions
    2. Each class/function becomes a chunk
    3. Module-level code is grouped into preamble
    4. Large classes can be split by method
    5. Imports are optionally included in each chunk for context

    Example:
        >>> chunker = PythonCodeChunker(max_chunk_size=2000)
        >>> chunker.chunker_id
        'python_code:2000:1'
    """

    plugin_name: str = field(default="python_code", repr=False)
    supported_extensions: list[str] = field(
        default_factory=lambda: [".py", ".pyw", ".pyi"], repr=False
    )
    max_chunk_size: int = 2000
    include_imports: bool = True
    split_classes_by_method: bool = True
    min_chunk_size: int = 10  # Very low default to not filter out small functions

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.max_chunk_size < 100:
            raise ValueError(f"max_chunk_size must be >= 100, got {self.max_chunk_size}")
        if self.min_chunk_size < 1:
            raise ValueError(f"min_chunk_size must be >= 1, got {self.min_chunk_size}")

    @property
    def chunker_id(self) -> str:
        """Unique identifier for this chunker configuration."""
        include_flag = 1 if self.include_imports else 0
        return f"{self.plugin_name}:{self.max_chunk_size}:{include_flag}"

    def _get_source_segment(
        self,
        source_lines: List[str],
        node: ast.AST,
        include_decorators: bool = True,
    ) -> str:
        """Extract source code for an AST node."""
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            return ""

        start_line = node.lineno - 1

        # Include decorators if present
        if include_decorators and hasattr(node, "decorator_list") and node.decorator_list:
            first_decorator = node.decorator_list[0]
            if hasattr(first_decorator, "lineno"):
                start_line = first_decorator.lineno - 1

        end_line = node.end_lineno

        return "\n".join(source_lines[start_line:end_line])

    def _extract_imports(self, tree: ast.AST, source_lines: List[str]) -> str:
        """Extract all import statements from the module."""
        imports: List[str] = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                segment = self._get_source_segment(source_lines, node, include_decorators=False)
                if segment:
                    imports.append(segment)

        return "\n".join(imports)

    def _get_docstring(self, node: ast.AST) -> Optional[str]:
        """Extract docstring from a node if present."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            return ast.get_docstring(node)
        return None

    def _extract_class_methods(
        self,
        class_node: ast.ClassDef,
        source_lines: List[str],
    ) -> List[Tuple[str, str, Optional[str]]]:
        """
        Extract methods from a class.

        Returns list of (method_name, method_source, docstring).
        """
        methods: List[Tuple[str, str, Optional[str]]] = []

        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                source = self._get_source_segment(source_lines, node)
                docstring = self._get_docstring(node)
                methods.append((node.name, source, docstring))

        return methods

    def _chunk_large_class(
        self,
        class_name: str,
        class_source: str,
        class_node: ast.ClassDef,
        source_lines: List[str],
        imports: str,
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Split a large class into chunks by method.

        Returns list of (chunk_name, chunk_content, extra_meta).
        """
        chunks: List[Tuple[str, str, Dict[str, Any]]] = []

        # Get class header (decorators + class line + docstring)
        class_start = class_node.lineno - 1
        if class_node.decorator_list:
            first_dec = class_node.decorator_list[0]
            if hasattr(first_dec, "lineno"):
                class_start = first_dec.lineno - 1

        # Find where the class body starts
        class_docstring = self._get_docstring(class_node)
        header_lines: List[str] = []

        for i, line in enumerate(source_lines[class_start:], start=class_start):
            header_lines.append(line)
            # Check if we've passed the class definition line
            if line.strip().endswith(":"):
                # Include docstring if present
                if class_docstring:
                    # Find docstring in source
                    for j in range(i + 1, min(i + 10, len(source_lines))):
                        doc_line = source_lines[j]
                        header_lines.append(doc_line)
                        if doc_line.strip().endswith('"""') or doc_line.strip().endswith("'''"):
                            if '"""' in doc_line or "'''" in doc_line:
                                break
                break

        class_header = "\n".join(header_lines)

        # Extract methods
        methods = self._extract_class_methods(class_node, source_lines)

        if not methods:
            # No methods, return class as single chunk
            content = imports + "\n\n" + class_source if imports else class_source
            return [(f"class {class_name}", content, {"type": "class"})]

        # Group methods into chunks
        current_methods: List[str] = []
        current_size = len(class_header) + len(imports) + 10

        for method_name, method_source, docstring in methods:
            method_size = len(method_source)

            if current_size + method_size > self.max_chunk_size and current_methods:
                # Emit current chunk
                content_parts = [imports] if imports else []
                content_parts.append(class_header)
                content_parts.extend(current_methods)
                content = "\n\n".join(filter(None, content_parts))

                chunk_name = f"class {class_name} (Part {len(chunks) + 1})"
                chunks.append((chunk_name, content, {"type": "class", "partial": True}))

                current_methods = [method_source]
                current_size = len(class_header) + len(imports) + method_size + 10
            else:
                current_methods.append(method_source)
                current_size += method_size + 2

        # Emit final chunk
        if current_methods:
            content_parts = [imports] if imports else []
            content_parts.append(class_header)
            content_parts.extend(current_methods)
            content = "\n\n".join(filter(None, content_parts))

            chunk_name = (
                f"class {class_name}"
                if len(chunks) == 0
                else f"class {class_name} (Part {len(chunks) + 1})"
            )
            chunks.append((chunk_name, content, {"type": "class", "partial": len(chunks) > 0}))

        return chunks

    def _parse_and_chunk(
        self,
        text: str,
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Parse Python code and extract chunks.

        Returns list of (chunk_name, chunk_content, extra_meta).
        """
        chunks: List[Tuple[str, str, Dict[str, Any]]] = []

        try:
            tree = ast.parse(text)
        except SyntaxError:
            # Fall back to returning entire text as single chunk
            if len(text.strip()) >= self.min_chunk_size:
                return [("module", text.strip(), {"type": "unparseable"})]
            return [("module", text.strip(), {"type": "unparseable"})] if text.strip() else []

        source_lines = text.splitlines()
        imports = self._extract_imports(tree, source_lines) if self.include_imports else ""

        # Track which lines we've processed
        processed_ranges: List[Tuple[int, int]] = []

        # Process top-level definitions
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Skip imports (already extracted)
                if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                    processed_ranges.append((node.lineno, node.end_lineno))
                continue

            if isinstance(node, ast.ClassDef):
                class_source = self._get_source_segment(source_lines, node)
                class_name = node.name
                docstring = self._get_docstring(node)

                # Check if class is too large
                if self.split_classes_by_method and len(class_source) > self.max_chunk_size:
                    class_chunks = self._chunk_large_class(
                        class_name, class_source, node, source_lines, imports
                    )
                    chunks.extend(class_chunks)
                else:
                    content = imports + "\n\n" + class_source if imports else class_source
                    meta = {"type": "class"}
                    if docstring:
                        meta["docstring"] = docstring[:200]
                    chunks.append((f"class {class_name}", content, meta))

                if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                    processed_ranges.append((node.lineno, node.end_lineno))

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_source = self._get_source_segment(source_lines, node)
                func_name = node.name
                docstring = self._get_docstring(node)

                content = imports + "\n\n" + func_source if imports else func_source
                meta = {"type": "function"}
                if docstring:
                    meta["docstring"] = docstring[:200]

                # Async indicator
                if isinstance(node, ast.AsyncFunctionDef):
                    meta["async"] = True

                chunks.append((f"def {func_name}", content, meta))

                if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                    processed_ranges.append((node.lineno, node.end_lineno))

        # Collect module-level code that wasn't part of classes/functions
        module_code_lines: List[str] = []
        for i, line in enumerate(source_lines, start=1):
            in_processed = any(start <= i <= end for start, end in processed_ranges)
            if not in_processed and line.strip():
                module_code_lines.append(line)

        if module_code_lines:
            module_code = "\n".join(module_code_lines)
            # Only include if substantial
            if len(module_code.strip()) >= self.min_chunk_size:
                chunks.insert(0, ("module_level", module_code, {"type": "module_level"}))

        return chunks

    def chunk_text(self, text: str, base_meta: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk Python code by classes and functions.

        Args:
            text: Python source code to chunk.
            base_meta: Base metadata to include in each chunk.

        Returns:
            List of Chunk objects.
        """
        if not text or not text.strip():
            return []

        doc_id = str(base_meta.get("doc_id") or base_meta.get("source_file") or "unknown")

        # Parse and extract chunks
        raw_chunks = self._parse_and_chunk(text)

        chunks: List[Chunk] = []
        chunk_index = 0

        for chunk_name, content, extra_meta in raw_chunks:
            if not content.strip():
                continue

            chunk_meta = dict(base_meta)
            chunk_meta["code_element"] = chunk_name
            chunk_meta.update(extra_meta)

            chunk_id = f"{doc_id}:{chunk_index}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    content=content,
                    metadata=chunk_meta,
                )
            )
            chunk_index += 1

        return chunks


__all__ = ["PythonCodeChunker"]
