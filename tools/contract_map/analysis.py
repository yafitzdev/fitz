# tools/contract_map/analysis.py
"""Code analysis: hotspots, config surface, stats, entrypoints, and invariants."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import tomllib  # py3.11+
except Exception:
    tomllib = None  # type: ignore[assignment]

from .common import (
    DEFAULT_LAYOUT_EXCLUDES,
    PKG,
    REPO_ROOT,
    CodeStats,
    ConfigSurface,
    ContractMap,
    Entrypoint,
    Hotspot,
    iter_python_files,
)


def read_pyproject() -> dict[str, Any] | None:
    """Read pyproject.toml if available."""
    pyproject = REPO_ROOT / "pyproject.toml"
    if not pyproject.exists():
        return None
    if tomllib is None:
        return None
    try:
        return tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except Exception:
        return None


def discover_entrypoints(root: Path, *, excludes: set[str]) -> List[Entrypoint]:
    """Discover entrypoints from pyproject.toml and __main__.py files."""
    eps: List[Entrypoint] = []

    data = read_pyproject()
    if data:
        proj = data.get("project") or {}
        scripts = proj.get("scripts") or {}
        if isinstance(scripts, dict):
            for name, target in sorted(scripts.items()):
                if isinstance(target, str):
                    eps.append(Entrypoint(kind="console_script", name=name, target=target))

        tool = data.get("tool") or {}
        poetry = tool.get("poetry") or {}
        poetry_scripts = poetry.get("scripts") or {}
        if isinstance(poetry_scripts, dict):
            for name, target in sorted(poetry_scripts.items()):
                if isinstance(target, str):
                    eps.append(Entrypoint(kind="poetry_script", name=name, target=target))

    for p in iter_python_files(root, excludes=excludes):
        rel = p.relative_to(root)
        if p.name == "__main__.py":
            eps.append(Entrypoint(kind="module_main", name=str(rel.parent), target=str(rel)))
        if p.name == "cli.py":
            eps.append(Entrypoint(kind="cli_module", name=str(rel.parent), target=str(rel)))

    eps.sort(key=lambda e: (e.kind, e.name, e.target))
    return eps


def find_default_yamls(root: Path, *, excludes: set[str]) -> List[str]:
    """Find all default.yaml files."""
    from .common import should_exclude_path

    out: List[str] = []
    for p in root.rglob("default.yaml"):
        rel = p.relative_to(root)
        if should_exclude_path(rel, excludes):
            continue
        out.append(str(rel))
    return sorted(out)


def list_loader_modules() -> List[str]:
    """List known config loader modules (auto-discovered)."""
    import importlib

    out: List[str] = []
    for pkg in PKG.config_loader_packages:
        # Try both {pkg}.loader and {pkg} directly
        for mod_name in (f"{pkg}.loader", pkg):
            try:
                mod = importlib.import_module(mod_name)
                # Check if it has a load function
                if any(hasattr(mod, fn) for fn in ("load_config", "load")):
                    out.append(mod_name)
                    break
            except Exception:
                continue
    return sorted(set(out))


def find_load_callsites(root: Path, *, excludes: set[str]) -> List[str]:
    """Find files that call config loading functions."""
    needles = (
        "load_config(",
        "load_rag_config(",
        "load_ingest_config(",
        "load_fitz_config(",
    )
    hits: List[str] = []

    for p in iter_python_files(root, excludes=excludes):
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue

        if any(n in text for n in needles):
            hits.append(str(p.relative_to(root)))

    return sorted(set(hits))


def compute_hotspots(root: Path, *, excludes: set[str]) -> List[Hotspot]:
    """Identify plugin hotspots (interfaces with many implementations)."""
    import importlib

    from tools.contract_map.discovery import scan_discovery

    impl: Dict[str, List[str]] = {}
    consumers: Dict[str, List[str]] = {}

    # Scan for YAML-based plugins from registries
    for reg_module in PKG.registry_modules:
        try:
            mod = importlib.import_module(reg_module)

            # LLM-style registry
            if hasattr(mod, "available_llm_plugins"):
                available_fn = getattr(mod, "available_llm_plugins")
                for ptype in ("chat", "embedding", "rerank"):
                    iface = f"{ptype.title()}Plugin"
                    try:
                        impl[iface] = [f"{p} (YAML)" for p in available_fn(ptype)]
                    except Exception:
                        impl[iface] = []

            # Vector DB-style registry
            if hasattr(mod, "available_vector_db_plugins"):
                available_fn = getattr(mod, "available_vector_db_plugins")
                try:
                    impl["VectorDBPlugin"] = [f"{p} (YAML)" for p in available_fn()]
                except Exception:
                    impl["VectorDBPlugin"] = []

        except Exception:
            continue

    # Scan Python plugins (auto-discovered)
    for ns, iface in PKG.python_plugin_namespaces:
        rep = scan_discovery(ns, note="hotspot scan")
        impl[iface] = rep.plugins_found

    # Consumer patterns (auto-discovered)
    patterns = PKG.consumer_patterns

    for p in iter_python_files(root, excludes=excludes):
        rel = str(p.relative_to(root))
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue

        for iface, pats in patterns.items():
            if any(x in text for x in pats):
                consumers.setdefault(iface, []).append(rel)

    out: List[Hotspot] = []
    for iface in sorted(patterns.keys()):
        out.append(
            Hotspot(
                interface=iface,
                implementations=impl.get(iface, []),
                consumers=sorted(set(consumers.get(iface, []))),
            )
        )
    return out


def compute_stats(root: Path, *, excludes: set[str]) -> CodeStats:
    """Compute code statistics."""
    py_files = 0
    total_lines = 0
    todo_fixme = 0
    any_mentions = 0
    module_sizes: List[Tuple[int, str]] = []

    for p in iter_python_files(root, excludes=excludes):
        py_files += 1
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue

        lines = text.splitlines()
        n = len(lines)
        total_lines += n
        module_sizes.append((n, str(p.relative_to(root))))
        todo_fixme += sum(1 for line in lines if "TODO" in line or "FIXME" in line)
        any_mentions += (
            text.count(" Any")
            + text.count("Any]")
            + text.count("Any,")
            + text.count("Any)")
            + text.count("Any:")
        )

    module_sizes.sort(key=lambda t: (-t[0], t[1]))
    largest = [f"{n} lines: {path}" for n, path in module_sizes[:10]]
    return CodeStats(
        py_files=py_files,
        total_lines=total_lines,
        todo_fixme=todo_fixme,
        any_mentions=any_mentions,
        largest_modules=largest,
    )


def compute_config_surface(cm: ContractMap, *, excludes: set[str]) -> ConfigSurface:
    """Compute the configuration surface area."""
    config_models = [f"{m.module}.{m.name}" for m in cm.models if ".config.schema" in m.module]
    default_yamls = find_default_yamls(REPO_ROOT, excludes=excludes)
    loaders = list_loader_modules()
    load_callsites = find_load_callsites(REPO_ROOT, excludes=excludes)
    return ConfigSurface(
        config_models=sorted(config_models),
        default_yamls=default_yamls,
        loaders=loaders,
        load_callsites=load_callsites,
    )


def compute_invariants(cm: ContractMap) -> List[str]:
    """Extract runtime invariants from the contract map."""
    inv: List[str] = []

    for m in cm.models:
        if m.module == "core.models.chunk" and m.name == "Chunk":
            req = [f.name for f in m.fields if f.required]
            inv.append(f"Chunk required fields: {', '.join(req)}")

    for p in cm.protocols:
        if p.name in {
            "EmbeddingPlugin",
            "RerankPlugin",
            "ChatPlugin",
            "VectorDBPlugin",
        }:
            for meth in p.methods:
                if meth.returns:
                    inv.append(f"{p.name}.{meth.name} returns {meth.returns}")

    for r in cm.registries:
        if r.name in {
            "LLM_REGISTRY",
            "RETRIEVER_REGISTRY",
            "CHUNKER_REGISTRY",
            "REGISTRY",
        }:
            inv.append(f"{r.module}.{r.name} plugins: {len(r.plugins)}")

    return inv


def render_entrypoints_section(entrypoints: List[Entrypoint]) -> str:
    """Render the Entrypoints section."""
    lines = ["## Entrypoints"]
    if not entrypoints:
        lines.append("- (none detected)")
    else:
        for ep in entrypoints:
            lines.append(f"- `{ep.kind}`: `{ep.name}` -> `{ep.target}`")
    lines.append("")
    return "\n".join(lines)


def render_hotspots_section(hotspots: List[Hotspot]) -> str:
    """Render the Contract Hotspots section."""
    lines = ["## Contract Hotspots"]
    for h in hotspots:
        lines.append(f"### `{h.interface}`")
        lines.append("- implementations:")
        if h.implementations:
            for i in h.implementations:
                lines.append(f"  - `{i}`")
        else:
            lines.append("  - (none)")
        lines.append("- consumers:")
        if h.consumers:
            for c in h.consumers:
                lines.append(f"  - `{c}`")
        else:
            lines.append("  - (none)")
        lines.append("")
    return "\n".join(lines)


def render_config_surface_section(cs: ConfigSurface | None) -> str:
    """Render the Config Surface section."""
    lines = ["## Config Surface"]
    if not cs:
        lines.append("- (not computed)")
        lines.append("")
        return "\n".join(lines)

    lines.append("- config models:")
    for m in cs.config_models:
        lines.append(f"  - `{m}`")
    lines.append("- default yamls:")
    for y in cs.default_yamls:
        lines.append(f"  - `{y}`")
    lines.append("- loaders:")
    for loader in cs.loaders:
        lines.append(f"  - `{loader}`")
    lines.append("- load callsites:")
    for c in cs.load_callsites:
        lines.append(f"  - `{c}`")
    lines.append("")
    return "\n".join(lines)


def render_invariants_section(invariants: List[str]) -> str:
    """Render the Runtime Invariants section."""
    lines = ["## Runtime Invariants"]
    if invariants:
        for inv in invariants:
            lines.append(f"- {inv}")
    else:
        lines.append("- (none)")
    lines.append("")
    return "\n".join(lines)


def render_stats_section(stats: CodeStats | None) -> str:
    """Render the Code Stats section."""
    lines = ["## Code Stats"]
    if not stats:
        lines.append("- (not computed)")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"- python_files: `{stats.py_files}`")
    lines.append(f"- total_lines: `{stats.total_lines}`")
    lines.append(f"- TODO/FIXME lines: `{stats.todo_fixme}`")
    lines.append(f"- 'Any' mentions: `{stats.any_mentions}`")
    lines.append("- largest modules:")
    for m in stats.largest_modules:
        lines.append(f"  - {m}")
    lines.append("")
    return "\n".join(lines)


def analyze_any_breakdown(root, excludes):
    """Quick Any breakdown - categorize by pattern."""

    categories = {
        "legitimate_kwargs": 0,
        "legitimate_metadata": 0,
        "legitimate_messages": 0,
        "lazy_type": 0,
        "lazy_return": 0,
        "lazy_param": 0,
        "other": 0,
    }

    for py_file in root.rglob("*.py"):
        rel = py_file.relative_to(root)
        if any(part in excludes for part in rel.parts):
            continue

        try:
            for line in py_file.read_text().splitlines():
                if "import " in line:
                    continue

                # Categorize
                if "kwargs: dict[str, Any]" in line or "kwargs: Dict[str, Any]" in line:
                    categories["legitimate_kwargs"] += 1
                elif "metadata: dict[str, Any]" in line or "metadata: Dict[str, Any]" in line:
                    categories["legitimate_metadata"] += 1
                elif "messages: list[dict[str, Any]]" in line:
                    categories["legitimate_messages"] += 1
                elif "Type[Any]" in line:
                    categories["lazy_type"] += 1
                elif "-> Any:" in line:
                    categories["lazy_return"] += 1
                elif ": Any" in line and "def " in line:
                    categories["lazy_param"] += 1
                elif " Any" in line or "Any]" in line or "Any," in line:
                    categories["other"] += 1
        except Exception:
            pass

    return categories


def render_any_breakdown_section(stats):
    """Render Any breakdown section."""
    if not stats:
        return ""

    cats = analyze_any_breakdown(REPO_ROOT, DEFAULT_LAYOUT_EXCLUDES)

    lines = ["## Any Usage Breakdown"]
    lines.append("")
    lines.append(f"**Total: `{stats.any_mentions}`**")
    lines.append("")
    lines.append("### By Category")
    lines.append("")

    legit = cats["legitimate_kwargs"] + cats["legitimate_metadata"] + cats["legitimate_messages"]
    lazy = cats["lazy_type"] + cats["lazy_return"] + cats["lazy_param"]

    lines.append(f"- **Legitimate (Keep)**: ~{legit}")
    lines.append(f"  - kwargs: {cats['legitimate_kwargs']}")
    lines.append(f"  - metadata: {cats['legitimate_metadata']}")
    lines.append(f"  - messages: {cats['legitimate_messages']}")
    lines.append("")
    lines.append(f"- **Lazy (Fix)**: ~{lazy}")
    lines.append(f"  - Type[Any]: {cats['lazy_type']}")
    lines.append(f"  - return Any: {cats['lazy_return']}")
    lines.append(f"  - param Any: {cats['lazy_param']}")
    lines.append("")
    lines.append(f"- **Other**: ~{cats['other']}")
    lines.append("")

    fixable_pct = (lazy / stats.any_mentions * 100) if stats.any_mentions > 0 else 0
    lines.append(f"**Fixable**: ~{lazy}/{stats.any_mentions} ({fixable_pct:.1f}%)")
    lines.append("")

    return "\n".join(lines)


def analyze_exception_patterns(root, excludes):
    """Analyze exception handling patterns in the codebase."""

    patterns = {
        "bare_except_continue": 0,
        "bare_except_pass": 0,
        "logged_exceptions": 0,
        "reraise_exceptions": 0,
    }

    problem_files = []

    for py_file in root.rglob("*.py"):
        rel = py_file.relative_to(root)
        if any(part in excludes for part in rel.parts):
            continue

        try:
            content = py_file.read_text()
            lines = content.splitlines()

            for i, line in enumerate(lines):
                # Check for bare except patterns
                if "except Exception:" in line or "except:" in line:
                    # Look at next line
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()

                        if next_line == "continue":
                            patterns["bare_except_continue"] += 1
                            problem_files.append((str(rel), i + 1, "silent continue"))
                        elif next_line == "pass":
                            patterns["bare_except_pass"] += 1
                            # Don't flag if in __del__ (cleanup is OK)
                            if i > 5 and "def __del__" not in "".join(lines[i - 5 : i]):
                                problem_files.append((str(rel), i + 1, "silent pass"))

                # Check for good patterns
                if "except" in line and "as e:" in line:
                    if i + 1 < len(lines):
                        next_line = lines[i + 1]
                        if "log" in next_line.lower():
                            patterns["logged_exceptions"] += 1
                        elif "raise" in next_line:
                            patterns["reraise_exceptions"] += 1

        except Exception:
            pass

    return patterns, problem_files


def render_exception_analysis_section(stats):
    """Render exception handling analysis section."""
    if not stats:
        return ""

    patterns, problems = analyze_exception_patterns(REPO_ROOT, DEFAULT_LAYOUT_EXCLUDES)

    lines = ["## Exception Handling Analysis"]
    lines.append("")

    total_bare = patterns["bare_except_continue"] + patterns["bare_except_pass"]
    good = patterns["logged_exceptions"] + patterns["reraise_exceptions"]

    lines.append(f"**Total bare exceptions**: `{total_bare}`")
    lines.append(f"**Good exception handling**: `{good}`")
    lines.append("")

    lines.append("### Patterns Found")
    lines.append("")
    lines.append(f"- **Silent failures** (`except: continue`): {patterns['bare_except_continue']}")
    lines.append(f"- **Silent ignores** (`except: pass`): {patterns['bare_except_pass']}")
    lines.append(f"- **Logged exceptions**: {patterns['logged_exceptions']}")
    lines.append(f"- **Re-raised exceptions**: {patterns['reraise_exceptions']}")
    lines.append("")

    if problems:
        lines.append("### Issues to Fix")
        lines.append("")
        # Show first 10
        for file, line_num, issue in problems[:10]:
            lines.append(f"- `{file}:{line_num}` - {issue}")

        if len(problems) > 10:
            lines.append(f"- ... and {len(problems) - 10} more")
        lines.append("")

    if total_bare > 0:
        lines.append("### Recommendation")
        lines.append("")
        lines.append(f"Fix {total_bare} silent exception(s) by adding logging:")
        lines.append("```python")
        lines.append("# Instead of:")
        lines.append("except Exception:")
        lines.append("    continue")
        lines.append("")
        lines.append("# Use:")
        lines.append("except Exception as e:")
        lines.append('    logger.warning(f"Failed: {e}")')
        lines.append("    continue")
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    from .common import ContractMap

    print("Computing analysis metrics...")

    print("\n1. Entrypoints:")
    eps = discover_entrypoints(REPO_ROOT, excludes=DEFAULT_LAYOUT_EXCLUDES)
    print(f"   Found {len(eps)} entrypoints")

    print("\n2. Hotspots:")
    hotspots = compute_hotspots(REPO_ROOT, excludes=DEFAULT_LAYOUT_EXCLUDES)
    print(f"   Found {len(hotspots)} hotspot interfaces")

    print("\n3. Stats:")
    stats = compute_stats(REPO_ROOT, excludes=DEFAULT_LAYOUT_EXCLUDES)
    print(f"   Scanned {stats.py_files} Python files, {stats.total_lines} lines")

    print("\n" + "=" * 80)
    print(render_entrypoints_section(eps))
    print(render_stats_section(stats))
