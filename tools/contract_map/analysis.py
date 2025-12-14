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
    REPO_ROOT,
    CodeStats,
    ConfigSurface,
    ContractMap,
    Entrypoint,
    Hotspot,
    iter_python_files,
)
from .discovery import scan_discovery


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
    """List known config loader modules."""
    import importlib

    out: List[str] = []
    for pkg in ("core.config", "fitz.pipeline.config", "fitz.ingest.config"):
        mod = f"{pkg}.loader"
        try:
            importlib.import_module(mod)
            out.append(mod)
        except Exception:
            continue
    return sorted(out)


def find_load_callsites(root: Path, *, excludes: set[str]) -> List[str]:
    """Find files that call config loading functions."""
    needles = ("load_config(", "load_rag_config(", "load_ingest_config(", "load_fitz_config(")
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
    impl: Dict[str, List[str]] = {}
    consumers: Dict[str, List[str]] = {}

    expected = [
        ("fitz.core.llm.chat.plugins", "ChatPlugin"),
        ("fitz.core.llm.embedding.plugins", "EmbeddingPlugin"),
        ("fitz.core.llm.rerank.plugins", "RerankPlugin"),
        ("fitz.core.vector_db.plugins", "VectorDBPlugin"),
        ("fitz.retrieval.plugins", "RetrievalPlugin"),
        ("fitz.pipeline.pipeline.plugins", "PipelinePlugin"),
        ("fitz.ingest.chunking.plugins", "ChunkerPlugin"),
        ("fitz.ingest.ingestion.plugins", "IngestPlugin"),
    ]
    for ns, iface in expected:
        rep = scan_discovery(ns, note="hotspot scan")
        impl[iface] = rep.plugins_found

    patterns = {
        "ChatPlugin": ("fitz.core.llm.chat", 'plugin_type="chat"', "plugin_type='chat'"),
        "EmbeddingPlugin": (
            "fitz.core.llm.embedding",
            'plugin_type="embedding"',
            "plugin_type='embedding'",
        ),
        "RerankPlugin": ("fitz.core.llm.rerank", 'plugin_type="rerank"', "plugin_type='rerank'"),
        "VectorDBPlugin": ("core.vector_db", 'plugin_type="vector_db"', "plugin_type='vector_db'"),
        "RetrievalPlugin": ("fitz.retrieval.registry", "get_retriever_plugin(", "RetrieverEngine.from_name("),
        "PipelinePlugin": ("fitz.pipeline.pipeline", "get_pipeline_plugin(", "available_pipeline_plugins("),
        "ChunkerPlugin": ("fitz.ingest.chunking", "get_chunker_plugin(", "ChunkingEngine"),
        "IngestPlugin": ("fitz.ingest.ingestion", "get_ingest_plugin(", "IngestionEngine"),
    }

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
        if p.name in {"EmbeddingPlugin", "RerankPlugin", "ChatPlugin", "VectorDBPlugin"}:
            for meth in p.methods:
                if meth.returns:
                    inv.append(f"{p.name}.{meth.name} returns {meth.returns}")

    for r in cm.registries:
        if r.name in {"LLM_REGISTRY", "RETRIEVER_REGISTRY", "CHUNKER_REGISTRY", "REGISTRY"}:
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
    for l in cs.loaders:
        lines.append(f"  - `{l}`")
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
