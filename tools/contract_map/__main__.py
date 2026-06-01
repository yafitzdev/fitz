# tools/contract_map/__main__.py
"""
Main entry point for contract map generation.
Combines all sections into a comprehensive report.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is in path for direct execution
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Dict, List

from tools.contract_map.analysis import (
    compute_config_surface,
    compute_hotspots,
    compute_invariants,
    compute_stats,
    discover_entrypoints,
    render_any_breakdown_section,
    render_config_surface_section,
    render_entrypoints_section,
    render_exception_analysis_section,
    render_hotspots_section,
    render_invariants_section,
    render_stats_section,
)
from tools.contract_map.architecture import ROLE_RULES, RoleResolver, load_architecture_mapping
from tools.contract_map.common import (
    DEFAULT_LAYOUT_EXCLUDES,
    REPO_ROOT,
    ContractMap,
    HealthIssue,
    ImportEdge,
)
from tools.contract_map.discovery import render_discovery_section, scan_all_discoveries
from tools.contract_map.imports import build_import_graph, render_import_graph_section
from tools.contract_map.layout import render_layout_section
from tools.contract_map.models import (
    extract_models,
    extract_protocols,
    render_models_section,
    render_protocols_section,
)
from tools.contract_map.registries import extract_registries, render_registries_section


@dataclass(frozen=True)
class ArchitectureViolation:
    """Represents a single architecture rule violation."""

    src_role: str
    dst_role: str
    src_module: str
    dst_module: str
    count: int


def check_discovery_health(cm: ContractMap) -> None:
    """
    Check if plugin discovery found expected plugins.

    Uses cm.discovery (the scan results) rather than cm.registries
    to avoid false positives from lazy initialization.
    """
    if not cm.discovery:
        return

    # Check for discovery failures
    total_failures = sum(len(r.failures) for r in cm.discovery)
    if total_failures > 0:
        cm.health.append(
            HealthIssue(
                level="WARN",
                message=f"{total_failures} plugin discovery failure(s) detected (see Discovery Report).",
            )
        )

    # Check for duplicates
    total_duplicates = sum(len(r.duplicates) for r in cm.discovery)
    if total_duplicates > 0:
        cm.health.append(
            HealthIssue(
                level="WARN",
                message=f"{total_duplicates} duplicate plugin name(s) detected (see Discovery Report).",
            )
        )

    # Check all discovered namespaces for empty plugin lists
    for report in cm.discovery:
        if not report.plugins_found and not report.failures:
            cm.health.append(
                HealthIssue(
                    level="WARN",
                    message=f"No plugins discovered in {report.namespace}",
                )
            )

    # Check for import failures recorded during extraction
    if cm.import_failures:
        cm.health.append(
            HealthIssue(
                level="WARN",
                message=f"{len(cm.import_failures)} import/discovery failures detected (see Import Failures section).",
            )
        )


def detect_architecture_violations(cm: ContractMap) -> List[ArchitectureViolation]:
    """
    Detect architecture violations from the import graph.

    Returns:
        List of ArchitectureViolation objects
    """
    violations: List[ArchitectureViolation] = []

    edges = architecture_edges(cm)
    if not edges:
        return violations

    try:
        resolver = RoleResolver()
    except Exception:
        # If we can't load architecture contracts, return empty
        return violations

    # Check each module-level import edge. Lazy imports are reported separately
    # by the import graph because they do not create import-time coupling.
    for edge in edges:
        importer_role = resolver.resolve_role(edge.src)
        imported_role = resolver.resolve_role(edge.dst)

        # Skip unknown roles (external deps, etc)
        if importer_role == "unknown" or imported_role == "unknown":
            continue

        # Skip self-imports (same role)
        if importer_role == imported_role:
            continue

        # Check if import is allowed
        if not resolver.is_allowed(importer_role, imported_role):
            violations.append(
                ArchitectureViolation(
                    src_role=importer_role,
                    dst_role=imported_role,
                    src_module=edge.src,
                    dst_module=edge.dst,
                    count=edge.count,
                )
            )

    return violations


def validate_architecture_contracts(cm: ContractMap) -> List[str]:
    """
    Validate import graph against architecture contracts.

    Returns:
        List of violation messages (warnings, not errors)
    """
    violations: List[str] = []

    edges = architecture_edges(cm)
    if not edges:
        return violations

    try:
        resolver = RoleResolver()
    except Exception as e:
        violations.append(f"Failed to load architecture contracts: {e}")
        return violations

    # Check each module-level import edge.
    for edge in edges:
        importer_role = resolver.resolve_role(edge.src)
        imported_role = resolver.resolve_role(edge.dst)

        # Skip unknown roles (external deps, etc)
        if importer_role == "unknown" or imported_role == "unknown":
            continue

        # Check if import is allowed
        if not resolver.is_allowed(importer_role, imported_role):
            violations.append(
                f"`{edge.src}` (role: {importer_role}) imports `{edge.dst}` (role: {imported_role}) — "
                f"violates architecture contract ({edge.count}x occurrences)"
            )

    return violations


def architecture_edges(cm: ContractMap) -> List[ImportEdge]:
    """Return module-level import edges used for architecture validation."""
    if not cm.import_graph:
        return []
    if cm.import_graph.module_edges:
        return cm.import_graph.module_edges
    return cm.import_graph.edges


def has_contract_failures(cm: ContractMap) -> bool:
    """Return True when --fail-on-errors should exit non-zero."""
    import_violations = bool(cm.import_graph and cm.import_graph.violations)
    arch_violations = bool(validate_architecture_contracts(cm))
    health_errors = any(issue.level == "ERROR" for issue in cm.health)
    return import_violations or arch_violations or health_errors


def build_contract_map(*, verbose: bool, layout_depth: int | None) -> ContractMap:
    """Build the complete contract map by running all extraction steps."""
    cm = ContractMap(
        meta={
            "python": sys.version.split()[0],
            "repo_root": REPO_ROOT.name,
            "cwd": Path.cwd().name,
        }
    )

    # Extract all components
    extract_models(cm, verbose=verbose)
    extract_protocols(cm, verbose=verbose)
    extract_registries(cm, verbose=verbose)

    # Build graphs and analysis
    cm.import_graph = build_import_graph(REPO_ROOT, excludes=DEFAULT_LAYOUT_EXCLUDES)
    cm.entrypoints = discover_entrypoints(REPO_ROOT, excludes=DEFAULT_LAYOUT_EXCLUDES)
    cm.discovery = scan_all_discoveries()
    cm.hotspots = compute_hotspots(REPO_ROOT, excludes=DEFAULT_LAYOUT_EXCLUDES)
    cm.config_surface = compute_config_surface(cm, excludes=DEFAULT_LAYOUT_EXCLUDES)
    cm.invariants = compute_invariants(cm)
    cm.stats = compute_stats(REPO_ROOT, excludes=DEFAULT_LAYOUT_EXCLUDES)

    # Check health AFTER discovery is populated
    check_discovery_health(cm)

    # Validate architecture contracts
    arch_violations = validate_architecture_contracts(cm)
    if arch_violations:
        cm.health.append(
            HealthIssue(
                level="WARN",
                message=f"{len(arch_violations)} architecture contract violation(s) detected (see Architecture Violations section).",
            )
        )

    return cm


def render_meta_section(cm: ContractMap) -> str:
    """Render the Meta section."""
    from tools.contract_map.common import render_section

    return render_section("Meta", sorted(cm.meta.items()), fmt=lambda kv: f"- `{kv[0]}`: `{kv[1]}`")


def render_health_section(cm: ContractMap) -> str:
    """Render the Health section."""
    if not cm.health:
        return ""
    from tools.contract_map.common import render_section

    return render_section("Health", cm.health, fmt=lambda h: f"- **{h.level}**: {h.message}")


def render_import_failures_section(cm: ContractMap, *, verbose: bool) -> str:
    """Render the Import Failures section."""
    if not cm.import_failures:
        return ""
    lines = ["## Import Failures"]
    for f in cm.import_failures:
        lines.append(f"- `{f.module}`: {f.error}")
        if verbose and f.traceback:
            lines.extend(["", "```", f.traceback.rstrip(), "```"])
    lines.append("")
    return "\n".join(lines)


def render_architecture_section() -> str:
    """
    Render the Architecture Contracts section.

    Shows:
    - Role mappings
    - Import rules
    - Validation behavior
    """
    lines = ["## Architecture Contracts"]
    lines.append("")
    lines.append("Architectural boundaries are enforced via role-based import rules.")
    lines.append("")

    # Show role mappings
    lines.append("### Role Mappings")
    lines.append("")
    mapping = load_architecture_mapping()
    package_mappings = {k: v for k, v in mapping.items() if "." in k}
    for package in sorted(package_mappings.keys()):
        role = package_mappings[package]
        lines.append(f"- `{package}` → **{role}**")
    lines.append("")

    # Show role rules
    lines.append("### Role Import Rules")
    lines.append("")
    for role in sorted(ROLE_RULES.values(), key=lambda r: r.name):
        if role.may_import is None:
            lines.append(f"- **{role.name}**: may compose any project layer")
            continue
        allowed = ", ".join(f"`{r}`" for r in sorted(role.may_import))
        lines.append(f"- **{role.name}**: may import {allowed}")
    lines.append("")

    # Validation results
    lines.append("### Validation")
    lines.append("")
    lines.append("Architecture validation runs on module-level import graph edges.")
    lines.append("Lazy imports are reported by the import graph but do not fail the contract.")
    lines.append("")

    return "\n".join(lines)


def render_architecture_violations_section(cm: ContractMap) -> str:
    """
    Render the Architecture Violations section.

    Shows concrete violations grouped by role pairs.
    """
    lines = ["## Architecture Violations"]
    lines.append("")

    violations = detect_architecture_violations(cm)

    if not violations:
        lines.append("No architecture violations detected.")
        lines.append("")
        return "\n".join(lines)

    # Group by role pair
    grouped: Dict[tuple[str, str], List[ArchitectureViolation]] = {}
    for v in violations:
        key = (v.src_role, v.dst_role)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(v)

    # Render each group
    for src_role, dst_role in sorted(grouped.keys()):
        group = grouped[(src_role, dst_role)]
        lines.append(f"### `{src_role}` → `{dst_role}` ({len(group)} violation(s))")
        lines.append("")

        for v in sorted(group, key=lambda x: (x.src_module, x.dst_module)):
            lines.append(f"- `{v.src_module}` imports `{v.dst_module}` ({v.count}x)")

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
    sections.append(render_architecture_section())
    sections.append(render_architecture_violations_section(cm))
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
    sections.append(render_any_breakdown_section(cm.stats))
    sections.append(render_exception_analysis_section(cm.stats))
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
        "-v",
        "--verbose",
        action="store_true",
        help="Include detailed traceback in failures",
    )
    parser.add_argument(
        "--layout-depth",
        type=int,
        default=None,
        help="Max depth for project layout (default: unlimited)",
    )
    parser.add_argument(
        "--fail-on-errors",
        action="store_true",
        help="Exit non-zero on architecture violations or ERROR health issues",
    )

    args = parser.parse_args(argv)

    # Ensure UTF-8 output on Windows
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    print("Building contract map...", file=sys.stderr)
    cm = build_contract_map(verbose=args.verbose, layout_depth=args.layout_depth)

    print("Rendering output...", file=sys.stderr)
    text = render_markdown(cm, verbose=args.verbose, layout_depth=args.layout_depth)

    print(text)

    if args.fail_on_errors and has_contract_failures(cm):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
