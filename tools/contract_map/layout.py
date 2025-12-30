# tools/contract_map/layout.py
"""Project layout tree generation and rendering."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List

from .common import DEFAULT_LAYOUT_EXCLUDES, REPO_ROOT, should_exclude_path


def get_classes_for_file(path: Path) -> list[str]:
    """Extract class names from a Python file using AST."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except Exception:
        return []

    return sorted(node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef))


def build_layout_tree(root: Path, *, max_depth: int | None, excludes: set[str]) -> Dict[str, Any]:
    """Build a hierarchical tree structure of the project layout."""
    tree: Dict[str, Any] = {}

    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if should_exclude_path(rel, excludes):
            continue
        if max_depth is not None and len(rel.parts) > max_depth:
            continue

        node = tree
        parts = rel.parts
        for i, part in enumerate(parts):
            is_last = i == len(parts) - 1
            if is_last:
                if p.is_dir():
                    key = f"{part}/"
                    node = node.setdefault(key, {})  # type: ignore[assignment]
                else:
                    node.setdefault(part, None)
            else:
                key = f"{part}/"
                node = node.setdefault(key, {})  # type: ignore[assignment]

    return tree


def render_layout_tree(
    tree: Dict[str, Any],
    prefix: str = "",
    *,
    root: Path,
) -> List[str]:
    """Render a layout tree with Unicode box-drawing characters."""
    lines: List[str] = []

    entries = sorted(
        tree.items(),
        key=lambda kv: (0 if kv[0].endswith("/") else 1, kv[0]),
    )

    for idx, (name, child) in enumerate(entries):
        last = idx == len(entries) - 1
        connector = "└── " if last else "├── "

        label = name

        # Annotate .py files inline with their classes
        if not name.endswith("/") and name.endswith(".py"):
            file_path = root / name
            classes = get_classes_for_file(file_path)

            if classes:
                if len(classes) == 1:
                    label = f"{name} (class: {classes[0]})"
                else:
                    label = f"{name} (classes: {', '.join(classes)})"

        lines.append(f"{prefix}{connector}{label}")

        if isinstance(child, dict) and child:
            extension = "    " if last else "│   "
            lines.extend(
                render_layout_tree(
                    child,
                    prefix=prefix + extension,
                    root=root / name.rstrip("/"),
                )
            )

    return lines


def render_layout_section(*, layout_depth: int | None) -> str:
    """Render the Project Layout section of the report."""
    lines = ["## Project Layout"]
    lines.append(f"- root: `{REPO_ROOT.name}`")
    lines.append(f"- excludes: `{sorted(DEFAULT_LAYOUT_EXCLUDES)}`")
    lines.append(f"- max_depth: `{layout_depth if layout_depth is not None else 'unlimited'}`")
    lines.append("")

    layout_tree = build_layout_tree(
        REPO_ROOT, max_depth=layout_depth, excludes=DEFAULT_LAYOUT_EXCLUDES
    )

    lines.append("```")
    lines.append(f"{REPO_ROOT.name}/")
    lines.extend(render_layout_tree(layout_tree, prefix="", root=REPO_ROOT))
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    print("Generating project layout tree...")
    print("\n" + "=" * 80)
    print(render_layout_section(layout_depth=3))
