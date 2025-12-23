# tools/contract_map/registries.py
"""Extract plugin registries from the codebase."""

from __future__ import annotations

import traceback
from typing import Iterable, List

from .common import ContractMap, ImportFailure, RegistryContract, safe_import


def maybe_call(
    cm: ContractMap, module_obj: object, fn_name: str, *, verbose: bool
) -> None:
    """Try to call a function on a module, logging failures."""
    fn = getattr(module_obj, fn_name, None)
    if callable(fn):
        try:
            fn()
        except Exception as exc:
            tb = traceback.format_exc() if verbose else None
            cm.import_failures.append(
                ImportFailure(
                    module=getattr(module_obj, "__name__", "<module>"),
                    error=f"Discovery call failed: {fn_name}(): {type(exc).__name__}: {exc}",
                    traceback=tb,
                )
            )


def extract_registry_plugins(
    cm: ContractMap,
    module_name: str,
    *,
    dict_attr: str,
    discover_fns: Iterable[str] = (),
    note: str | None = None,
    verbose: bool,
) -> RegistryContract | None:
    """Extract plugins from a registry dict."""
    mod = safe_import(cm, module_name, verbose=verbose)
    if mod is None:
        return None

    for fn in discover_fns:
        maybe_call(cm, mod, fn, verbose=verbose)

    reg = getattr(mod, dict_attr, None)
    if not isinstance(reg, dict):
        return None

    plugins = sorted(str(k) for k in reg.keys())
    return RegistryContract(
        module=module_name, name=dict_attr, plugins=plugins, note=note
    )


def extract_pipeline_registry(
    cm: ContractMap,
    module_name: str,
    *,
    list_fn: str,
    note: str | None,
    verbose: bool,
) -> RegistryContract | None:
    """Extract plugins from a registry that uses a list function."""
    mod = safe_import(cm, module_name, verbose=verbose)
    if mod is None:
        return None

    fn = getattr(mod, list_fn, None)
    if not callable(fn):
        return None

    try:
        plugins = fn()
    except Exception as exc:
        tb = traceback.format_exc() if verbose else None
        cm.import_failures.append(
            ImportFailure(
                module=module_name,
                error=f"{list_fn}() failed: {type(exc).__name__}: {exc}",
                traceback=tb,
            )
        )
        return None

    if not isinstance(plugins, list):
        return None

    return RegistryContract(
        module=module_name,
        name=list_fn,
        plugins=sorted(str(x) for x in plugins),
        note=note,
    )


def extract_llm_registry(
    cm: ContractMap, module_name: str, *, verbose: bool
) -> List[RegistryContract]:
    """
    Extract LLM registry with multiple plugin types.

    NOTE: LLM registry now uses YAML-based discovery.
    We call available_llm_plugins() to get the real plugin list.
    """
    out: List[RegistryContract] = []
    mod = safe_import(cm, module_name, verbose=verbose)
    if mod is None:
        return out

    # Try to get available_llm_plugins function
    available_fn = getattr(mod, "available_llm_plugins", None)
    if not callable(available_fn):
        return out

    # Query each plugin type
    for plugin_type in ["chat", "embedding", "rerank"]:
        try:
            plugins = available_fn(plugin_type)
            if isinstance(plugins, list):
                out.append(
                    RegistryContract(
                        module=module_name,
                        name=f"LLM_REGISTRY[{plugin_type!r}]",
                        plugins=sorted(str(p) for p in plugins),
                        note=f"YAML-based discovery (fitz_ai/llm/{plugin_type}/*.yaml)",
                    )
                )
        except Exception as exc:
            tb = traceback.format_exc() if verbose else None
            cm.import_failures.append(
                ImportFailure(
                    module=module_name,
                    error=f"available_llm_plugins('{plugin_type}') failed: {type(exc).__name__}: {exc}",
                    traceback=tb,
                )
            )

    # Flatten all plugins for summary view
    flat: list[str] = []
    for contract in out:
        flat.extend(contract.plugins)

    if flat:
        out.append(
            RegistryContract(
                module=module_name,
                name="LLM_REGISTRY",
                plugins=sorted(set(flat)),
                note="Central LLM plugin registry (flattened view, YAML-based)",
            )
        )

    return out


def extract_vector_db_registry(
    cm: ContractMap, module_name: str, *, verbose: bool
) -> RegistryContract | None:
    """
    Extract Vector DB registry.

    NOTE: Vector DB registry now uses YAML-based discovery.
    We call available_vector_db_plugins() to get the real plugin list.
    """
    mod = safe_import(cm, module_name, verbose=verbose)
    if mod is None:
        return None

    # Try to get available_vector_db_plugins function
    available_fn = getattr(mod, "available_vector_db_plugins", None)
    if not callable(available_fn):
        return None

    try:
        plugins = available_fn()
        if isinstance(plugins, list):
            return RegistryContract(
                module=module_name,
                name="VECTOR_DB_REGISTRY",
                plugins=sorted(str(p) for p in plugins),
                note="YAML-based discovery (fitz_ai/vector_db/plugins/*.yaml)",
            )
    except Exception as exc:
        tb = traceback.format_exc() if verbose else None
        cm.import_failures.append(
            ImportFailure(
                module=module_name,
                error=f"available_vector_db_plugins() failed: {type(exc).__name__}: {exc}",
                traceback=tb,
            )
        )

    return None


def extract_registries(cm: ContractMap, *, verbose: bool) -> None:
    """Extract all registries from the codebase."""
    # LLM registry (chat, embedding, rerank) - YAML-based
    cm.registries.extend(
        extract_llm_registry(cm, "fitz_ai.llm.registry", verbose=verbose)
    )

    # Vector DB registry (separate from LLM) - YAML-based
    vdb = extract_vector_db_registry(cm, "fitz_ai.vector_db.registry", verbose=verbose)
    if vdb:
        cm.registries.append(vdb)

    # Chunker registry - Python-based
    cr = extract_registry_plugins(
        cm,
        "fitz_ai.ingest.chunking.registry",
        dict_attr="CHUNKER_REGISTRY",
        discover_fns=("_auto_discover",),
        note="Lazy discovery over ingest.chunking.plugins.*",
        verbose=verbose,
    )
    if cr:
        cm.registries.append(cr)

    # Ingestion registry - Python-based
    ir = extract_registry_plugins(
        cm,
        "fitz_ai.ingest.ingestion.registry",
        dict_attr="REGISTRY",
        discover_fns=("_auto_discover",),
        note="Lazy discovery over ingest.ingestion.plugins.*",
        verbose=verbose,
    )
    if ir:
        cm.registries.append(ir)

    # Pipeline registry - Python-based
    # Fixed: was "pipeline.pipeline.registry", now "pipeline.registry"
    pr = extract_pipeline_registry(
        cm,
        "fitz_ai.engines.classic_rag.pipeline.registry",
        list_fn="available_pipeline_plugins",
        note="Lazy discovery over pipeline.plugins.*",
        verbose=verbose,
    )
    if pr:
        cm.registries.append(pr)

    cm.registries.sort(key=lambda r: (r.module, r.name))


def render_registries_section(cm: ContractMap) -> str:
    """Render the Registries section of the report."""
    lines = ["## Registries"]

    for r in cm.registries:
        lines.append(f"### `{r.module}.{r.name}`")
        if r.note:
            lines.append(f"- {r.note}")
        if r.plugins:
            for p in r.plugins:
                lines.append(f"- `{p}`")
        else:
            lines.append("- (empty)")
        lines.append("")

    return "\n".join(lines)