# tools/contract_map/discovery.py
"""Plugin discovery and namespace scanning."""
from __future__ import annotations

import importlib
import pkgutil
from typing import Callable, Dict, List

from .common import DiscoveryReport


def plugin_predicate_for_namespace(namespace: str):
    """Return predicates to identify plugins in a namespace."""
    if namespace.startswith("fitz.core.llm."):
        expected = namespace.split(".", 3)[2]  # chat|embedding|rerank
        method = {"chat": "chat", "embedding": "embed", "rerank": "rerank"}.get(expected)

        def is_plugin(cls: type) -> bool:
            if not isinstance(cls, type):
                return False
            if not isinstance(getattr(cls, "plugin_name", None), str):
                return False
            if getattr(cls, "plugin_type", None) != expected:
                return False
            return callable(getattr(cls, method, None)) if method else False

        def plugin_id(cls: type) -> str:
            return f"{cls.__module__}.{cls.__name__}"

        return is_plugin, plugin_id

    if namespace == "fitz.core.vector_db.plugins":

        def is_plugin(cls: type) -> bool:
            if not isinstance(getattr(cls, "plugin_name", None), str):
                return False
            if getattr(cls, "plugin_type", None) != "vector_db":
                return False
            return callable(getattr(cls, "search", None))

        def plugin_id(cls: type) -> str:
            return f"{cls.__module__}.{cls.__name__}"

        return is_plugin, plugin_id

    if namespace == "fitz.retrieval.plugins":

        def is_plugin(cls: type) -> bool:
            if not isinstance(getattr(cls, "plugin_name", None), str):
                return False
            return callable(getattr(cls, "retrieve", None))

        def plugin_id(cls: type) -> str:
            return f"{cls.__module__}.{cls.__name__}"

        return is_plugin, plugin_id

    if namespace == "fitz.pipeline.pipeline.plugins":

        def is_plugin(cls: type) -> bool:
            if not isinstance(getattr(cls, "plugin_name", None), str):
                return False
            return callable(getattr(cls, "build", None))

        def plugin_id(cls: type) -> str:
            return f"{cls.__module__}.{cls.__name__}"

        return is_plugin, plugin_id

    if namespace == "fitz.ingest.chunking.plugins":

        def is_plugin(cls: type) -> bool:
            if not isinstance(getattr(cls, "plugin_name", None), str):
                return False
            return callable(getattr(cls, "chunk_text", None))

        def plugin_id(cls: type) -> str:
            return f"{cls.__module__}.{cls.__name__}"

        return is_plugin, plugin_id

    if namespace == "fitz.ingest.ingestion.plugins":

        def is_plugin(cls: type) -> bool:
            if not isinstance(getattr(cls, "plugin_name", None), str):
                return False
            return callable(getattr(cls, "ingest", None))

        def plugin_id(cls: type) -> str:
            return f"{cls.__module__}.{cls.__name__}"

        return is_plugin, plugin_id

    def is_plugin(_: type) -> bool:
        return False

    def plugin_id(cls: type) -> str:
        return f"{cls.__module__}.{cls.__name__}"

    return is_plugin, plugin_id


def scan_discovery(namespace: str, note: str) -> DiscoveryReport:
    """Scan a namespace for plugins using discovery predicates."""
    failures: List[str] = []
    duplicates: List[str] = []
    found: Dict[str, str] = {}
    modules_scanned = 0

    try:
        pkg = importlib.import_module(namespace)
    except Exception as exc:
        return DiscoveryReport(
            namespace=namespace,
            note=note,
            modules_scanned=0,
            plugins_found=[],
            failures=[f"{namespace}: {type(exc).__name__}: {exc}"],
            duplicates=[],
        )

    pkg_path = getattr(pkg, "__path__", None)
    if pkg_path is None:
        return DiscoveryReport(
            namespace=namespace,
            note=note,
            modules_scanned=0,
            plugins_found=[],
            failures=[],
            duplicates=[],
        )

    is_plugin, plugin_id = plugin_predicate_for_namespace(namespace)

    for mod_info in pkgutil.iter_modules(pkg_path):
        modules_scanned += 1
        mod_name = f"{namespace}.{mod_info.name}"
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:
            failures.append(f"{mod_name}: {type(exc).__name__}: {exc}")
            continue

        mod_name_actual = getattr(mod, "__name__", mod_name)
        for obj in vars(mod).values():
            if not isinstance(obj, type):
                continue
            if getattr(obj, "__module__", None) != mod_name_actual:
                continue
            if not is_plugin(obj):
                continue

            pn = getattr(obj, "plugin_name")
            pid = plugin_id(obj)
            existing = found.get(pn)
            if existing is not None and existing != pid:
                duplicates.append(f"{pn!r}: {existing} vs {pid}")
            else:
                found[pn] = pid

    plugins_found = [f"{name} -> {found[name]}" for name in sorted(found.keys())]
    return DiscoveryReport(
        namespace=namespace,
        note=note,
        modules_scanned=modules_scanned,
        plugins_found=plugins_found,
        failures=sorted(failures),
        duplicates=sorted(duplicates),
    )


def scan_all_discoveries() -> List[DiscoveryReport]:
    """Scan all known plugin namespaces."""
    return [
        scan_discovery("fitz.core.llm.chat.plugins", "LLM chat plugins (Option A discovery)"),
        scan_discovery("fitz.core.llm.embedding.plugins", "LLM embedding plugins (Option A discovery)"),
        scan_discovery("fitz.core.llm.rerank.plugins", "LLM rerank plugins (Option A discovery)"),
        scan_discovery("fitz.core.vector_db.plugins", "Vector DB plugins (Option A discovery)"),
        scan_discovery("fitz.retrieval.plugins", "RAG retriever plugins (Option A discovery)"),
        scan_discovery("fitz.pipeline.pipeline.plugins", "RAG pipeline plugins (Option A discovery)"),
        scan_discovery("fitz.ingest.chunking.plugins", "Ingest chunking plugins (Option A discovery)"),
        scan_discovery("fitz.ingest.ingestion.plugins", "Ingest ingestion plugins (Option A discovery)"),
    ]


def render_discovery_section(discovery: List[DiscoveryReport]) -> str:
    """Render the Discovery Report section."""
    lines = ["## Discovery Report"]

    for rep in discovery:
        lines.append(f"### `{rep.namespace}`")
        lines.append(f"- {rep.note}")
        lines.append(f"- modules_scanned: `{rep.modules_scanned}`")

        if rep.duplicates:
            for d in rep.duplicates:
                lines.append(f"- **ERROR** duplicate: {d}")

        if rep.failures:
            for f in rep.failures[:10]:
                lines.append(f"- **WARN** import failure: {f}")
            if len(rep.failures) > 10:
                lines.append(f"- **WARN** ... {len(rep.failures) - 10} more failures omitted")

        if rep.plugins_found:
            lines.append("- plugins:")
            for p in rep.plugins_found:
                lines.append(f"  - `{p}`")
        else:
            lines.append("- plugins: (none)")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    print("Running plugin discovery...")
    reports = scan_all_discoveries()

    total_plugins = sum(len(r.plugins_found) for r in reports)
    print(f"Found {total_plugins} plugins across {len(reports)} namespaces")

    print("\n" + "=" * 80)
    print(render_discovery_section(reports))
