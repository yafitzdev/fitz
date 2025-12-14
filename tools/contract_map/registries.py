# tools/contract_map/registries.py
"""Extract plugin registries from the codebase."""
from __future__ import annotations

import traceback
from typing import Iterable, List

from .common import ContractMap, RegistryContract, ImportFailure, safe_import


def maybe_call(cm: ContractMap, module_obj: object, fn_name: str, *, verbose: bool) -> None:
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
    return RegistryContract(module=module_name, name=dict_attr, plugins=plugins, note=note)


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


def extract_llm_registry(cm: ContractMap, module_name: str, *, verbose: bool) -> List[RegistryContract]:
    """Extract LLM registry with multiple plugin types."""
    out: List[RegistryContract] = []
    mod = safe_import(cm, module_name, verbose=verbose)
    if mod is None:
        return out

    maybe_call(cm, mod, "_auto_discover", verbose=verbose)

    reg = getattr(mod, "LLM_REGISTRY", None)
    if not isinstance(reg, dict):
        return out

    for plugin_type in sorted(reg.keys()):
        bucket = reg.get(plugin_type)
        if not isinstance(bucket, dict):
            continue
        plugins = sorted(str(k) for k in bucket.keys())
        out.append(
            RegistryContract(
                module=module_name,
                name=f"LLM_REGISTRY[{plugin_type!r}]",
                plugins=plugins,
                note="Lazy discovery over core.llm*/plugins and core.vector_db.plugins",
            )
        )

    flat: list[str] = []
    for bucket in reg.values():
        if isinstance(bucket, dict):
            flat.extend(str(k) for k in bucket.keys())
    out.append(
        RegistryContract(
            module=module_name,
            name="LLM_REGISTRY",
            plugins=sorted(set(flat)),
            note="Central plugin registry (flattened view)",
        )
    )
    return out


def extract_registries(cm: ContractMap, *, verbose: bool) -> None:
    """Extract all registries from the codebase."""
    cm.registries.extend(extract_llm_registry(cm, "core.llm.registry", verbose=verbose))

    rr = extract_registry_plugins(
        cm,
        "rag.retrieval.registry",
        dict_attr="RETRIEVER_REGISTRY",
        discover_fns=("_auto_discover",),
        note="Lazy discovery over rag.retrieval.plugins.*",
        verbose=verbose,
    )
    if rr:
        cm.registries.append(rr)

    cr = extract_registry_plugins(
        cm,
        "ingest.chunking.registry",
        dict_attr="CHUNKER_REGISTRY",
        discover_fns=("_auto_discover",),
        note="Lazy discovery over ingest.chunking.plugins.*",
        verbose=verbose,
    )
    if cr:
        cm.registries.append(cr)

    ir = extract_registry_plugins(
        cm,
        "ingest.ingestion.registry",
        dict_attr="REGISTRY",
        discover_fns=("_auto_discover",),
        note="Lazy discovery over ingest.ingestion.plugins.*",
        verbose=verbose,
    )
    if ir:
        cm.registries.append(ir)

    pr = extract_pipeline_registry(
        cm,
        "rag.pipeline.registry",
        list_fn="available_pipeline_plugins",
        note="Lazy discovery over rag.pipeline.plugins.*",
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
        if not r.plugins:
            lines.append("- (no plugins found)")
            lines.append("")
            continue
        for name in r.plugins:
            lines.append(f"- `{name}`")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    from .common import ContractMap

    cm = ContractMap(meta={"test": "registries_extraction"})

    print("Extracting registries...")
    extract_registries(cm, verbose=True)
    print(f"Found {len(cm.registries)} registries")

    print("\n" + "=" * 80)
    print(render_registries_section(cm))
