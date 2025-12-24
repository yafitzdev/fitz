# tools/contract_map/discovery.py
"""
Plugin discovery scanning for contract map.

Scans all plugin namespaces and reports what's discovered,
including any failures or duplicates.

NOTE: LLM plugins (chat, embedding, rerank) and Vector DB plugins use YAML files,
      not Python modules. We scan for .yaml files instead.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from tools.contract_map.common import REPO_ROOT, DiscoveryReport


def _simple_plugin_id(cls) -> str:
    return f"{cls.__module__}.{cls.__name__}"


# ---------------------------------------------------------------------------
# YAML Plugin Discovery
# ---------------------------------------------------------------------------


def scan_yaml_plugins(plugin_dir: str, plugin_type: str) -> DiscoveryReport:
    """
    Scan for YAML-based plugins (LLM and Vector DB).

    Args:
        plugin_dir: Directory path like "fitz_ai/llm/chat" or "fitz_ai/vector_db/plugins"
        plugin_type: Type identifier for note

    Returns:
        DiscoveryReport with YAML plugins found
    """
    import sys

    # Find the fitz_ai package root
    fitz_locations = [p for p in sys.path if Path(p).name in ("fitz_ai", "src", "")]

    yaml_path = None
    for loc in fitz_locations:
        candidate = Path(loc) / plugin_dir.replace(".", "/")
        if candidate.exists():
            yaml_path = candidate
            break

    # Try relative to cwd as fallback
    if yaml_path is None:
        yaml_path = REPO_ROOT / plugin_dir.replace(".", "/")

    if not yaml_path.exists():
        return DiscoveryReport(
            namespace=plugin_dir,
            note=f"{plugin_type} plugins (YAML discovery)",
            modules_scanned=0,
            plugins_found=[],
            failures=[f"{plugin_dir}: Directory not found"],
            duplicates=[],
        )

    # Scan for .yaml files
    yaml_files = list(yaml_path.glob("*.yaml"))
    plugin_names = sorted([f.stem for f in yaml_files if not f.stem.startswith("_")])

    return DiscoveryReport(
        namespace=plugin_dir,
        note=f"{plugin_type} plugins (YAML discovery)",
        modules_scanned=len(yaml_files),
        plugins_found=[f"{name} (YAML)" for name in plugin_names],
        failures=[],
        duplicates=[],
    )


# ---------------------------------------------------------------------------
# Discovery Predicates (for Python-based plugins)
# ---------------------------------------------------------------------------

# Map namespace -> (predicate, plugin_id_fn, allow_reexport)
PLUGIN_PREDICATES: Dict[str, Tuple[Callable, Callable, bool]] = {
    # --- pipeline (under engines/classic_rag) ---
    "fitz_ai.engines.classic_rag.pipeline.plugins": (
        lambda cls: (
            isinstance(getattr(cls, "plugin_name", None), str)
            and callable(getattr(cls, "build", None))
        ),
        _simple_plugin_id,
        False,
    ),
    # --- chunking (under ingest) ---
    "fitz_ai.ingest.chunking.plugins": (
        lambda cls: (
            isinstance(getattr(cls, "plugin_name", None), str)
            and callable(getattr(cls, "chunk_text", None))
        ),
        _simple_plugin_id,
        False,
    ),
    # --- ingestion (under ingest) ---
    "fitz_ai.ingest.ingestion.plugins": (
        lambda cls: (
            isinstance(getattr(cls, "plugin_name", None), str)
            and callable(getattr(cls, "ingest", None))
        ),
        _simple_plugin_id,
        False,
    ),
}


def scan_discovery(namespace: str, note: str) -> DiscoveryReport:
    """
    Scan a plugin namespace for classes matching a predicate.

    Args:
        namespace: Package name to scan (e.g., "fitz_ai.ingest.chunking.plugins")
        note: Human-readable description

    Returns:
        DiscoveryReport with plugins discovered, failures, and duplicates
    """
    if namespace not in PLUGIN_PREDICATES:
        return DiscoveryReport(
            namespace=namespace,
            note=note,
            modules_scanned=0,
            plugins_found=[],
            failures=[f"{namespace}: No predicate configured"],
            duplicates=[],
        )

    predicate, plugin_id, allow_reexport = PLUGIN_PREDICATES[namespace]

    try:
        package = importlib.import_module(namespace)
    except ImportError as exc:
        return DiscoveryReport(
            namespace=namespace,
            note=note,
            modules_scanned=0,
            plugins_found=[],
            failures=[f"{namespace}: {type(exc).__name__}: {exc}"],
            duplicates=[],
        )

    package_path = getattr(package, "__path__", None)
    if not package_path:
        return DiscoveryReport(
            namespace=namespace,
            note=note,
            modules_scanned=0,
            plugins_found=[],
            failures=[f"{namespace}: No __path__ (not a package)"],
            duplicates=[],
        )

    found: Dict[str, str] = {}
    failures: List[str] = []
    duplicates: List[str] = []
    modules_scanned = 0

    for mod_info in pkgutil.iter_modules(package_path, prefix=f"{namespace}."):
        if mod_info.ispkg:
            continue

        modules_scanned += 1
        mod_name = mod_info.name

        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:
            failures.append(f"{mod_name}: {type(exc).__name__}: {exc}")
            continue

        actual_name = getattr(mod, "__name__", mod_name)
        for obj in vars(mod).values():
            if not isinstance(obj, type):
                continue

            # Check module match - skip if class is from different module
            # UNLESS allow_reexport is True for this namespace
            obj_module = getattr(obj, "__module__", None)
            if not allow_reexport and obj_module != actual_name:
                continue

            if not predicate(obj):
                continue

            name = getattr(obj, "plugin_name")
            pid = plugin_id(obj)

            existing = found.get(name)
            if existing and existing != pid:
                duplicates.append(f"{name!r}: {existing} vs {pid}")
            else:
                found[name] = pid

    return DiscoveryReport(
        namespace=namespace,
        note=note,
        modules_scanned=modules_scanned,
        plugins_found=[f"{k} -> {found[k]}" for k in sorted(found)],
        failures=sorted(failures),
        duplicates=sorted(duplicates),
    )


def scan_all_discoveries() -> List[DiscoveryReport]:
    """Scan all declared plugin namespaces."""
    return [
        # LLM plugins - YAML-based
        scan_yaml_plugins("fitz_ai/llm/chat", "LLM chat"),
        scan_yaml_plugins("fitz_ai/llm/embedding", "LLM embedding"),
        scan_yaml_plugins("fitz_ai/llm/rerank", "LLM rerank"),
        # Vector DB plugins - YAML-based
        scan_yaml_plugins("fitz_ai/vector_db/plugins", "Vector DB"),
        # Retrieval plugins - YAML-based (not Python)
        scan_yaml_plugins(
            "fitz_ai/engines/classic_rag/retrieval/plugins", "RAG retrieval"
        ),
        # Pipeline plugins - Python-based
        scan_discovery(
            "fitz_ai.engines.classic_rag.pipeline.plugins",
            "RAG pipeline plugins (Python discovery)",
        ),
        # Ingest plugins - Python-based
        scan_discovery(
            "fitz_ai.ingest.chunking.plugins",
            "Ingest chunking plugins (Python discovery)",
        ),
        scan_discovery(
            "fitz_ai.ingest.ingestion.plugins",
            "Ingest ingestion plugins (Python discovery)",
        ),
    ]


def render_discovery_section(reports: List[DiscoveryReport]) -> str:
    """Render the Discovery Report section."""
    lines = ["## Discovery Report"]

    for r in reports:
        lines.append(f"### `{r.namespace}`")
        if r.note:
            lines.append(f"- {r.note}")
        lines.append(f"- modules_scanned: `{r.modules_scanned}`")

        if r.plugins_found:
            lines.append("- plugins:")
            for p in r.plugins_found:
                lines.append(f"  - `{p}`")

        if r.failures:
            lines.append("- failures:")
            for f in r.failures:
                lines.append(f"  - `{f}`")

        if r.duplicates:
            lines.append("- duplicates:")
            for d in r.duplicates:
                lines.append(f"  - `{d}`")

        lines.append("")

    return "\n".join(lines)
