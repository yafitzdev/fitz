# tools/contract_map/__main__.py
"""
Main entry point for contract map generation.
Combines all sections into a comprehensive report.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

from tools.contract_map.common import (
    REPO_ROOT, DEFAULT_LAYOUT_EXCLUDES, ContractMap, HealthIssue
)
from tools.contract_map.models import extract_models, extract_protocols, render_models_section, render_protocols_section
from tools.contract_map.registries import extract_registries, render_registries_section
from tools.contract_map.layout import render_layout_section
from tools.contract_map.imports import build_import_graph, render_import_graph_section
from tools.contract_map.discovery import scan_all_discoveries, render_discovery_section
from tools.contract_map.analysis import (
    discover_entrypoints, compute_hotspots, compute_stats,
    compute_config_surface, compute_invariants,
    render_entrypoints_section, render_hotspots_section,
    render_config_surface_section, render_invariants_section,
    render_stats_section
)


def check_registry_health(cm: ContractMap) -> None:
    """Check if registries are populated and add health issues."""

    def registry_nonempty(name_contains: str) -> bool:
        for r in cm.registries:
            if name_contains in r.name and r.plugins:
                return True
        return False

    if not registry_nonempty("LLM_REGISTRY"):
        cm.health.append(
            HealthIssue(
                level="ERROR",
                message="LLM registry appears empty. Likely cause: discovery not triggered or plugin imports failing.",
            )
        )

    if not registry_nonempty("RETRIEVER_REGISTRY"):
        cm.health.append(
            HealthIssue(
                level="ERROR",
                message="Retriever registry appears empty. Likely cause: rag.retrieval.plugins package missing/empty or import failures.",
            )
        )

    if cm.import_failures:
        cm.health.append(
            HealthIssue(
                level="WARN",
                message=f"{len(cm.import_failures)} import/discovery failures detected (see Import Failures section).",
            )
        )


def build_contract_map(*, verbose: bool, layout_depth: int | None) -> ContractMap:
    """Build the complete contract map by running all extraction steps."""
    cm = ContractMap(
        meta={
            "python": sys.version.split()[0],
            "repo_root": str(REPO_ROOT),
            "cwd": str(Path.cwd()),
        }
    )

    # Extract all components
    extract_models(cm, verbose=verbose)
    extract_protocols(cm, verbose=verbose)
    extract_registries(cm, verbose=verbose)

    # Check health
    check_registry_health(cm)

    # Build graphs and analysis
    cm.import_graph = build_import_graph(REPO_ROOT, excludes=DEFAULT_LAYOUT_EXCLUDES)
    cm.entrypoints = discover_entrypoints(REPO_ROOT, excludes=DEFAULT_LAYOUT_EXCLUDES)
    cm.discovery = scan_all_discoveries()
    cm.hotspots = compute_hotspots(REPO_ROOT, excludes=DEFAULT_LAYOUT_EXCLUDES)
    cm.config_surface = compute_config_surface(cm, excludes=DEFAULT_LAYOUT_EXCLUDES)
    cm.invariants = compute_invariants(cm)
    cm.stats = compute_stats(REPO_ROOT, excludes=DEFAULT_LAYOUT_EXCLUDES)

    return cm


def render_meta_section(cm: ContractMap) -> str:
    """Render the Meta section."""
    lines = ["## Meta"]
    for k in sorted(cm.meta.keys()):
        lines.append(f"- `{k}`: `{cm.meta[k]}`")
    lines.append("")
    return "\n".join(lines)


def render_health_section(cm: ContractMap) -> str:
    """Render the Health section."""
    if not cm.health:
        return ""

    lines = ["## Health"]
    for h in cm.health:
        lines.append(f"- **{h.level}**: {h.message}")
    lines.append("")
    return "\n".join(lines)


def render_import_failures_section(cm: ContractMap, *, verbose: bool) -> str:
    """Render the Import Failures section."""
    if not cm.import_failures:
        return ""

    lines = ["## Import Failures"]
    for f in cm.import_failures:
        lines.append(f"- `{f.module}`: {f.error}")
        if verbose and f.traceback:
            lines.append("")
            lines.append("```")
            lines.append(f.traceback.rstrip())
            lines.append("```")
    lines.append("")
    return "\n".join(lines)


def fix_unicode_rendering(text: str) -> str:
    """
    Fix Unicode box-drawing characters that may not render properly.
    This ensures the tree structure displays correctly across different terminals.
    """
    # The original characters should already be correct UTF-8,
    # but if you encounter rendering issues, you can add replacements here
    # For example:
    # text = text.replace('â"‚', '│')
    # text = text.replace('â"œ', '├')
    # text = text.replace('â""', '└')
    # text = text.replace('â"€', '─')

    # For now, we'll leave them as-is since they should work
    return text


def render_markdown(cm: ContractMap, *, verbose: bool, layout_depth: int | None) -> str:
    """Render the complete contract map as Markdown."""
    sections: List[str] = []

    # Title
    sections.append("# Contract Map")
    sections.append("")

    # Combine all sections
    sections.append(render_meta_section(cm))
    sections.append(render_layout_section(layout_depth=layout_depth))
    sections.append(render_import_graph_section(cm.import_graph))
    sections.append(render_entrypoints_section(cm.entrypoints))
    sections.append(render_discovery_section(cm.discovery))

    health = render_health_section(cm)
    if health:
        sections.append(health)

    failures = render_import_failures_section(cm, verbose=verbose)
    if failures:
        sections.append(failures)

    sections.append(render_config_surface_section(cm.config_surface))
    sections.append(render_invariants_section(cm.invariants))
    sections.append(render_stats_section(cm.stats))
    sections.append(render_hotspots_section(cm.hotspots))
    sections.append(render_models_section(cm))
    sections.append(render_protocols_section(cm))
    sections.append(render_registries_section(cm))

    text = "\n".join(sections).rstrip() + "\n"

    # Fix any Unicode rendering issues
    text = fix_unicode_rendering(text)

    return text


def render_json(cm: ContractMap) -> str:
    """Render the contract map as JSON."""
    payload = asdict(cm)
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate a contract map from the current codebase."
    )
    parser.add_argument(
        "--format",
        choices=["md", "json"],
        default="md",
        help="Output format"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Write output to a file (otherwise prints to stdout)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include tracebacks for import/discovery failures"
    )
    parser.add_argument(
        "--fail-on-errors",
        action="store_true",
        help="Exit non-zero if any ERROR health issues exist"
    )
    parser.add_argument(
        "--layout-depth",
        type=int,
        default=None,
        help="Max depth for Project Layout tree"
    )

    args = parser.parse_args(argv)

    print("Building contract map...", file=sys.stderr)
    cm = build_contract_map(verbose=args.verbose, layout_depth=args.layout_depth)

    print("Rendering output...", file=sys.stderr)
    if args.format == "json":
        text = render_json(cm)
    else:
        text = render_markdown(cm, verbose=args.verbose, layout_depth=args.layout_depth)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"Wrote output to {out_path}", file=sys.stderr)
    else:
        sys.stdout.write(text)

    if args.fail_on_errors and any(h.level == "ERROR" for h in cm.health):
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
