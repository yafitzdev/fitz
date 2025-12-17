# tools/contract_map/imports.py
"""Import graph analysis and layering violation detection.

This module distinguishes between:
- Module-level imports (always executed when module loads)
- Lazy imports (inside functions/methods, only executed when called)

Only module-level imports are considered for architecture violations,
since lazy imports don't create circular dependencies or architectural coupling.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple

from .common import (
    DEFAULT_LAYOUT_EXCLUDES,
    REPO_ROOT,
    ImportEdge,
    ImportGraph,
    iter_python_files,
    module_name_from_path,
    toplevel,
)


def resolve_from_import(*, current_module: str, module: str | None, level: int) -> str | None:
    """Resolve a relative import to an absolute module name."""
    if level <= 0:
        return module

    cur_parts = current_module.split(".")
    drop = max(level, 1)
    base_parts = cur_parts[:-drop]
    if not base_parts:
        return module

    if module:
        return ".".join(base_parts + module.split("."))
    return ".".join(base_parts)


def normalize_root(name: str | None) -> str | None:
    """
    Normalize a module name to its architectural domain root.

    Example:
    - fitz.engines.classic_rag.retrieval.runtime.engine -> retrieval
    - fitz.engines.classic_rag.pipeline.pipeline.engine -> pipeline
    """
    if not name:
        return None
    return toplevel(name)


def _is_inside_function(node: ast.AST, tree: ast.Module) -> bool:
    """
    Check if an import node is inside a function or method.

    Returns True if the import is "lazy" (inside a function/method),
    False if it's at module level.
    """
    # Build parent map
    parent_map: Dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[child] = parent

    # Walk up the tree to see if we're inside a function
    current = node
    while current in parent_map:
        current = parent_map[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return True

    return False


def _extract_imports(
    tree: ast.Module,
    current_module: str,
    *,
    include_lazy: bool = True,
) -> List[Tuple[str, bool]]:
    """
    Extract all imports from an AST tree.

    Args:
        tree: Parsed AST module
        current_module: Name of the current module (for relative imports)
        include_lazy: If True, include imports inside functions

    Returns:
        List of (module_name, is_lazy) tuples
    """
    imports: List[Tuple[str, bool]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            is_lazy = _is_inside_function(node, tree)
            if not include_lazy and is_lazy:
                continue
            for alias in node.names:
                imports.append((alias.name, is_lazy))

        elif isinstance(node, ast.ImportFrom):
            is_lazy = _is_inside_function(node, tree)
            if not include_lazy and is_lazy:
                continue
            target = resolve_from_import(
                current_module=current_module,
                module=node.module,
                level=int(getattr(node, "level", 0) or 0),
            )
            if target:
                imports.append((target, is_lazy))

    return imports


def build_import_graph(root: Path, *, excludes: set[str]) -> ImportGraph:
    """
    Build an import graph by analyzing all Python files.

    This function distinguishes between module-level and lazy imports.
    Only module-level imports are counted for violation detection.
    """
    # Track both module-level and lazy imports separately
    module_level_counts: Dict[Tuple[str, str], int] = {}
    lazy_counts: Dict[Tuple[str, str], int] = {}

    for file in iter_python_files(root, excludes=excludes):
        mod = module_name_from_path(file)
        src = normalize_root(mod)
        if not src:
            continue

        try:
            text = file.read_text(encoding="utf-8")
        except Exception:
            continue

        try:
            tree = ast.parse(text, filename=str(file))
        except SyntaxError:
            continue

        # Extract imports with lazy detection
        imports = _extract_imports(tree, mod or "", include_lazy=True)

        for import_name, is_lazy in imports:
            dst = normalize_root(import_name)
            if not dst or dst == src:
                continue

            key = (src, dst)
            if is_lazy:
                lazy_counts[key] = lazy_counts.get(key, 0) + 1
            else:
                module_level_counts[key] = module_level_counts.get(key, 0) + 1

    # Combine for total edge counts (for display)
    all_counts: Dict[Tuple[str, str], int] = {}
    for key, count in module_level_counts.items():
        all_counts[key] = all_counts.get(key, 0) + count
    for key, count in lazy_counts.items():
        all_counts[key] = all_counts.get(key, 0) + count

    edges = [ImportEdge(src=k[0], dst=k[1], count=v) for k, v in all_counts.items()]
    edges.sort(key=lambda e: (-e.count, e.src, e.dst))

    # Only check MODULE-LEVEL imports for violations
    # Lazy imports (inside functions) are allowed and don't create architectural coupling
    violations: List[str] = []
    for (src, dst), count in module_level_counts.items():
        if src == "core" and dst in {"pipeline", "ingest"}:
            violations.append(f"VIOLATION: core imports {dst} at module level ({count}x)")
        if src == "ingest" and dst == "pipeline":
            violations.append(f"VIOLATION: ingest imports pipeline at module level ({count}x)")

    # Add info about lazy imports that would have been violations
    lazy_would_violate: List[str] = []
    for (src, dst), count in lazy_counts.items():
        if src == "core" and dst in {"pipeline", "ingest"}:
            lazy_would_violate.append(f"(lazy/OK) core imports {dst} inside functions ({count}x)")
        if src == "ingest" and dst == "pipeline":
            lazy_would_violate.append(
                f"(lazy/OK) ingest imports pipeline inside functions ({count}x)"
            )

    return ImportGraph(
        edges=edges, violations=sorted(violations), lazy_ok=sorted(lazy_would_violate)
    )


def render_import_graph_section(import_graph: ImportGraph | None) -> str:
    """Render the Import Graph section of the report."""
    lines = ["## Import Graph"]

    if not import_graph:
        lines.append("- (not computed)")
        lines.append("")
        return "\n".join(lines)

    if import_graph.violations:
        for v in import_graph.violations:
            lines.append(f"- **ERROR**: {v}")
    else:
        lines.append("- (no module-level layering violations detected)")

    # Show lazy imports that are OK
    if hasattr(import_graph, "lazy_ok") and import_graph.lazy_ok:
        lines.append("")
        lines.append("Lazy imports (inside functions, no violation):")
        for info in import_graph.lazy_ok:
            lines.append(f"- {info}")

    lines.append("")

    lines.append("Top edges:")
    for e in import_graph.edges[:20]:
        lines.append(f"- `{e.src} -> {e.dst}`: {e.count}x")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    print("Building import graph...")
    graph = build_import_graph(REPO_ROOT, excludes=DEFAULT_LAYOUT_EXCLUDES)

    print(f"Found {len(graph.edges)} edges")
    print(f"Found {len(graph.violations)} module-level violations")
    if hasattr(graph, "lazy_ok"):
        print(f"Found {len(graph.lazy_ok)} lazy imports (OK)")

    print("\n" + "=" * 80)
    print(render_import_graph_section(graph))
