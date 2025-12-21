# tools/contract_map/any_analysis.py
"""
Enhanced Any type analysis for contract map.

Categorizes Any usage into:
- Legitimate (should keep)
- Lazy (should fix)
- Converter (needs Protocol)
- Reflection (acceptable in tools)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set

from .common import DEFAULT_LAYOUT_EXCLUDES, REPO_ROOT


@dataclass
class AnyOccurrence:
    """Single occurrence of Any in code."""

    file: str
    line_num: int
    line: str
    category: str
    context: str


@dataclass
class AnyAnalysis:
    """Comprehensive Any type analysis."""

    total_count: int
    by_category: dict[str, int] = field(default_factory=dict)
    by_file: dict[str, int] = field(default_factory=dict)
    occurrences: List[AnyOccurrence] = field(default_factory=list)
    hotspots: List[tuple[str, int]] = field(default_factory=list)  # (file, count)


# Patterns for categorizing Any usage
LEGITIMATE_PATTERNS = {
    "plugin_kwargs": re.compile(r"kwargs:\s*dict\[str,\s*Any\]"),
    "metadata": re.compile(r"metadata:\s*[Dd]ict\[str,\s*Any\]"),
    "llm_messages": re.compile(r"messages:\s*list\[dict\[str,\s*Any\]\]"),
    "config_presets": re.compile(r"presets:\s*dict\[str,\s*dict\[str,\s*Any\]\]"),
}

LAZY_PATTERNS = {
    "type_erasure": re.compile(r"Type\[Any\]"),
    "return_any": re.compile(r"->\s*Any:"),
    "param_any": re.compile(r"\(\s*\w+:\s*Any\s*[,\)]"),
    "dict_any": re.compile(r"dict\[str,\s*Any\]"),  # Generic, needs context
}

CONVERTER_PATTERNS = {
    "chunk_like": re.compile(r"chunk_like:\s*Any"),
    "chunk_input": re.compile(r"ChunkInput.*=.*Any"),
    "to_chunk": re.compile(r"def _to_chunk"),
}

REFLECTION_PATTERNS = {
    "inspect_module": re.compile(r"module.*:.*Any"),
    "extract_fields": re.compile(r"extract.*model_cls.*Type\[Any\]"),
    "getattr_obj": re.compile(r"getattr\(.*obj.*Any"),
}


def categorize_any_occurrence(file: str, line: str) -> str:
    """Categorize a single Any occurrence."""
    # Check if in reflection/tool code
    if "tools/contract_map" in file:
        for pattern in REFLECTION_PATTERNS.values():
            if pattern.search(line):
                return "reflection"

    # Check legitimate patterns
    for name, pattern in LEGITIMATE_PATTERNS.items():
        if pattern.search(line):
            return f"legitimate:{name}"

    # Check converter patterns
    for pattern in CONVERTER_PATTERNS.values():
        if pattern.search(line):
            return "converter"

    # Check lazy patterns
    if LAZY_PATTERNS["type_erasure"].search(line):
        return "lazy:type_erasure"
    if LAZY_PATTERNS["return_any"].search(line):
        return "lazy:return_type"
    if LAZY_PATTERNS["param_any"].search(line):
        return "lazy:parameter"

    # Check for dict[str, Any] in specific contexts
    if LAZY_PATTERNS["dict_any"].search(line):
        # Config helpers that aren't legitimate
        if "def _" in line and "config" in file:
            return "lazy:config_helper"
        # Otherwise might be legitimate
        if "kwargs" in line.lower() or "metadata" in line.lower():
            return "legitimate:dict_any"

    return "other"


def extract_context(file_path: Path, line_num: int) -> str:
    """Extract context around a line (function or class name)."""
    try:
        lines = file_path.read_text().splitlines()
        # Look backwards for function or class definition
        for i in range(max(0, line_num - 10), line_num):
            line = lines[i].strip()
            if line.startswith("def ") or line.startswith("class "):
                return line.split("(")[0].replace("def ", "").replace("class ", "")
    except Exception:
        pass
    return "unknown"


def analyze_any_usage(
    root: Path,
    *,
    excludes: Set[str] = None,
) -> AnyAnalysis:
    """Comprehensive Any usage analysis."""
    if excludes is None:
        excludes = DEFAULT_LAYOUT_EXCLUDES

    analysis = AnyAnalysis(total_count=0, by_category={}, by_file={})

    for py_file in root.rglob("*.py"):
        # Skip excluded paths
        rel_path = py_file.relative_to(root)
        if any(part in excludes for part in rel_path.parts):
            continue

        try:
            content = py_file.read_text(encoding="utf-8")
        except Exception:
            continue

        lines = content.splitlines()
        file_count = 0

        for line_num, line in enumerate(lines, start=1):
            # Skip import lines
            if line.strip().startswith(("import ", "from ")):
                continue

            # Count Any occurrences
            any_count = (
                line.count(" Any")
                + line.count("Any]")
                + line.count("Any,")
                + line.count("Any)")
                + line.count("Any:")
            )

            if any_count > 0:
                file_count += any_count
                analysis.total_count += any_count

                # Categorize
                category = categorize_any_occurrence(str(rel_path), line)

                # Track by category
                analysis.by_category[category] = (
                    analysis.by_category.get(category, 0) + any_count
                )

                # Store occurrence
                context = extract_context(py_file, line_num - 1)
                analysis.occurrences.append(
                    AnyOccurrence(
                        file=str(rel_path),
                        line_num=line_num,
                        line=line.strip(),
                        category=category,
                        context=context,
                    )
                )

        if file_count > 0:
            analysis.by_file[str(rel_path)] = file_count

    # Compute hotspots (files with most Any usage)
    analysis.hotspots = sorted(
        analysis.by_file.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    return analysis


def render_any_analysis_section(analysis: AnyAnalysis) -> str:
    """Render the Any Analysis section."""
    lines = ["## Any Type Analysis"]
    lines.append("")

    # Summary
    lines.append(f"**Total Any mentions: `{analysis.total_count}`**")
    lines.append("")

    # Breakdown by category
    lines.append("### Breakdown by Category")
    lines.append("")

    # Group categories
    legitimate = sum(
        count
        for cat, count in analysis.by_category.items()
        if cat.startswith("legitimate")
    )
    lazy = sum(
        count for cat, count in analysis.by_category.items() if cat.startswith("lazy")
    )
    converter = analysis.by_category.get("converter", 0)
    reflection = analysis.by_category.get("reflection", 0)
    other = analysis.by_category.get("other", 0)

    lines.append(f"- **Legitimate (Keep)**: `{legitimate}` mentions")
    lines.append("  - Plugin kwargs, metadata, LLM messages")
    for cat, count in sorted(analysis.by_category.items()):
        if cat.startswith("legitimate"):
            subcategory = cat.split(":")[-1] if ":" in cat else cat
            lines.append(f"    - {subcategory}: {count}")

    lines.append(f"- **Lazy (Should Fix)**: `{lazy}` mentions")
    lines.append("  - Type erasure, return types, parameters")
    for cat, count in sorted(analysis.by_category.items()):
        if cat.startswith("lazy"):
            subcategory = cat.split(":")[-1] if ":" in cat else cat
            lines.append(f"    - {subcategory}: {count}")

    lines.append(f"- **Converter Functions**: `{converter}` mentions")
    lines.append("  - Needs Protocol-based typing")

    lines.append(f"- **Reflection/Tools**: `{reflection}` mentions")
    lines.append("  - Acceptable for introspection code")

    lines.append(f"- **Other**: `{other}` mentions")
    lines.append("")

    # Hotspots
    lines.append("### Any Usage Hotspots")
    lines.append("")
    lines.append("Files with most Any mentions:")
    lines.append("")
    for file, count in analysis.hotspots:
        lines.append(f"- `{file}`: {count} mentions")
    lines.append("")

    # Recommendations
    lines.append("### Recommendations")
    lines.append("")

    if lazy > 0:
        lines.append(
            f"1. **Fix Lazy Typing** ({lazy} mentions): Type erasure and lazy parameters"
        )
        lines.append("   - Replace `Type[Any]` with specific types")
        lines.append("   - Replace `-> Any:` with concrete return types")
        lines.append("   - Replace parameter `Any` with `object` or specific types")

    if converter > 0:
        lines.append(
            f"2. **Add Protocols** ({converter} mentions): Converter functions"
        )
        lines.append("   - Create `@runtime_checkable` Protocols")
        lines.append(
            "   - Replace `chunk_like: Any` with `chunk_like: ChunkLike | dict`"
        )

    lines.append(
        f"3. **Keep Legitimate** ({legitimate} mentions): Config, metadata, messages"
    )
    lines.append("   - These are correctly typed")
    lines.append("")

    # Progress tracking
    total_fixable = lazy + converter
    percentage_fixable = (
        (total_fixable / analysis.total_count * 100) if analysis.total_count > 0 else 0
    )

    lines.append("### Improvement Potential")
    lines.append("")
    lines.append(f"- **Total Any**: {analysis.total_count}")
    lines.append(f"- **Fixable**: {total_fixable} ({percentage_fixable:.1f}%)")
    lines.append(
        f"- **Should Keep**: {legitimate + reflection} ({(legitimate + reflection) / analysis.total_count * 100:.1f}%)"
    )
    lines.append("")

    return "\n".join(lines)


def render_any_details_section(analysis: AnyAnalysis, *, max_items: int = 20) -> str:
    """Render detailed Any occurrences section."""
    lines = ["## Any Usage Details (Top 20)"]
    lines.append("")

    # Group by category
    by_category: dict[str, List[AnyOccurrence]] = {}
    for occ in analysis.occurrences:
        if occ.category not in by_category:
            by_category[occ.category] = []
        by_category[occ.category].append(occ)

    # Show lazy ones first (most important to fix)
    priority_categories = [
        ("lazy:type_erasure", "Type Erasure (Fix First)"),
        ("lazy:return_type", "Return Type Any"),
        ("lazy:parameter", "Parameter Any"),
        ("converter", "Converter Functions"),
    ]

    for category, title in priority_categories:
        if category not in by_category:
            continue

        occurrences = by_category[category][:max_items]
        if not occurrences:
            continue

        lines.append(f"### {title}")
        lines.append("")

        for occ in occurrences:
            lines.append(f"**{occ.file}:{occ.line_num}** (in `{occ.context}`)")
            lines.append("```python")
            lines.append(occ.line)
            lines.append("```")
            lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    print("Analyzing Any usage...")
    analysis = analyze_any_usage(REPO_ROOT)

    print("\n" + "=" * 80)
    print(render_any_analysis_section(analysis))
    print("\n" + "=" * 80)
    print(render_any_details_section(analysis))
