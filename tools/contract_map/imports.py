# tools/contract_map/imports.py
"""Import graph analysis and layering violation detection."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Tuple

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


def build_import_graph(root: Path, *, excludes: set[str]) -> ImportGraph:
    """Build an import graph by analyzing all Python files."""
    edge_counts: Dict[Tuple[str, str], int] = {}

    for file in iter_python_files(root, excludes=excludes):
        mod = module_name_from_path(file)
        src = toplevel(mod)
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

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dst = toplevel(alias.name)
                    if not dst or dst == src:
                        continue
                    edge_counts[(src, dst)] = edge_counts.get((src, dst), 0) + 1

            elif isinstance(node, ast.ImportFrom):
                target = resolve_from_import(
                    current_module=mod or "",
                    module=node.module,
                    level=int(getattr(node, "level", 0) or 0),
                )
                dst = toplevel(target)
                if not dst or dst == src:
                    continue
                edge_counts[(src, dst)] = edge_counts.get((src, dst), 0) + 1

    edges = [ImportEdge(src=k[0], dst=k[1], count=v) for k, v in edge_counts.items()]
    edges.sort(key=lambda e: (-e.count, e.src, e.dst))

    violations: List[str] = []
    for e in edges:
        if e.src == "core" and e.dst in {"pipeline", "ingest"}:
            violations.append(f"VIOLATION: core imports {e.dst} ({e.count}x)")
        if e.src == "ingest" and e.dst == "pipeline":
            violations.append(f"VIOLATION: ingest imports pipeline ({e.count}x)")

    return ImportGraph(edges=edges, violations=sorted(violations))


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
        lines.append("- (no layering violations detected)")
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
    print(f"Found {len(graph.violations)} violations")

    print("\n" + "=" * 80)
    print(render_import_graph_section(graph))
