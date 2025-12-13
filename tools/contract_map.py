# tools/contract_map.py
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type, get_type_hints

try:
    from typing import get_args, get_origin
except ImportError:  # pragma: no cover
    get_args = None  # type: ignore[assignment]
    get_origin = None  # type: ignore[assignment]

try:
    from pydantic import BaseModel
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore[assignment]


def _ensure_repo_root_on_syspath() -> Path:
    """
    When running as: python tools/contract_map.py
    sys.path[0] is tools/, not repo root. Fix deterministically.
    """
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


REPO_ROOT = _ensure_repo_root_on_syspath()


@dataclass(frozen=True, slots=True)
class ImportFailure:
    module: str
    error: str
    traceback: str | None = None


@dataclass(frozen=True, slots=True)
class ModelField:
    name: str
    type: str
    required: bool
    default: str | None = None


@dataclass(frozen=True, slots=True)
class ModelContract:
    module: str
    name: str
    fields: List[ModelField] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class MethodContract:
    name: str
    signature: str
    returns: str | None


@dataclass(frozen=True, slots=True)
class ProtocolContract:
    module: str
    name: str
    methods: List[MethodContract] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RegistryContract:
    module: str
    name: str
    plugins: List[str] = field(default_factory=list)
    note: str | None = None


@dataclass(frozen=True, slots=True)
class HealthIssue:
    level: str  # "WARN" | "ERROR"
    message: str


@dataclass(frozen=True, slots=True)
class ContractMap:
    meta: Dict[str, Any] = field(default_factory=dict)
    import_failures: List[ImportFailure] = field(default_factory=list)
    health: List[HealthIssue] = field(default_factory=list)
    models: List[ModelContract] = field(default_factory=list)
    protocols: List[ProtocolContract] = field(default_factory=list)
    registries: List[RegistryContract] = field(default_factory=list)


def _fmt_type(tp: Any) -> str:
    if tp is None:
        return "None"

    if get_origin is not None and get_args is not None:
        origin = get_origin(tp)
        if origin is None:
            return getattr(tp, "__name__", None) or str(tp)

        args = get_args(tp)
        origin_name = getattr(origin, "__name__", None) or str(origin)
        if args:
            return f"{origin_name}[{', '.join(_fmt_type(a) for a in args)}]"
        return origin_name

    return getattr(tp, "__name__", None) or str(tp)


def _is_pydantic_model(obj: Any) -> bool:
    try:
        return isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel
    except Exception:
        return False


def _extract_pydantic_fields(model_cls: Type[Any]) -> List[ModelField]:
    fields: List[ModelField] = []

    # pydantic v2
    if hasattr(model_cls, "model_fields"):
        mf: Dict[str, Any] = getattr(model_cls, "model_fields")
        for name in sorted(mf.keys()):
            f = mf[name]
            ann = getattr(f, "annotation", None)
            required = bool(getattr(f, "is_required", False))
            default = getattr(f, "default", None)
            fields.append(
                ModelField(
                    name=name,
                    type=_fmt_type(ann),
                    required=required,
                    default=None if required else repr(default),
                )
            )
        return fields

    # pydantic v1
    if hasattr(model_cls, "__fields__"):
        ff: Dict[str, Any] = getattr(model_cls, "__fields__")
        for name in sorted(ff.keys()):
            f = ff[name]
            ann = getattr(f, "type_", None)
            required = bool(getattr(f, "required", False))
            default = getattr(f, "default", None)
            fields.append(
                ModelField(
                    name=name,
                    type=_fmt_type(ann),
                    required=required,
                    default=None if required else repr(default),
                )
            )
        return fields

    hints = getattr(model_cls, "__annotations__", {}) or {}
    for name in sorted(hints.keys()):
        fields.append(ModelField(name=name, type=_fmt_type(hints[name]), required=True, default=None))
    return fields


def _looks_like_protocol(obj: Any) -> bool:
    if not isinstance(obj, type):
        return False
    return bool(getattr(obj, "_is_protocol", False))


def _extract_protocol_methods(proto_cls: Type[Any]) -> List[MethodContract]:
    methods: List[MethodContract] = []

    for name, member in inspect.getmembers(proto_cls):
        if name.startswith("_"):
            continue
        if not inspect.isfunction(member) and not inspect.ismethod(member):
            continue
        if getattr(member, "__qualname__", "").split(".")[0] != proto_cls.__name__:
            continue

        try:
            sig = str(inspect.signature(member))
        except Exception:
            sig = "(...)"

        returns: str | None = None
        try:
            hints = get_type_hints(member, include_extras=True)
            if "return" in hints:
                returns = _fmt_type(hints["return"])
        except Exception:
            returns = None

        methods.append(MethodContract(name=name, signature=sig, returns=returns))

    methods.sort(key=lambda m: m.name)
    return methods


def _safe_import(cm: ContractMap, module: str, *, verbose: bool) -> object | None:
    try:
        return importlib.import_module(module)
    except Exception as exc:
        tb = traceback.format_exc() if verbose else None
        cm.import_failures.append(ImportFailure(module=module, error=f"{type(exc).__name__}: {exc}", traceback=tb))
        return None


def _maybe_call(cm: ContractMap, module_obj: object, fn_name: str, *, verbose: bool) -> None:
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


def _extract_registry_plugins(
    cm: ContractMap,
    module_name: str,
    *,
    dict_attr: str,
    discover_fns: Iterable[str] = (),
    note: str | None = None,
    verbose: bool,
) -> RegistryContract | None:
    mod = _safe_import(cm, module_name, verbose=verbose)
    if mod is None:
        return None

    for fn in discover_fns:
        _maybe_call(cm, mod, fn, verbose=verbose)

    reg = getattr(mod, dict_attr, None)
    if not isinstance(reg, dict):
        return None

    plugins = sorted(str(k) for k in reg.keys())
    return RegistryContract(module=module_name, name=dict_attr, plugins=plugins, note=note)


def _extract_pipeline_registry(
    cm: ContractMap,
    module_name: str,
    *,
    list_fn: str,
    note: str | None,
    verbose: bool,
) -> RegistryContract | None:
    mod = _safe_import(cm, module_name, verbose=verbose)
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


def _extract_llm_registry(
    cm: ContractMap,
    module_name: str,
    *,
    verbose: bool,
) -> List[RegistryContract]:
    out: List[RegistryContract] = []
    mod = _safe_import(cm, module_name, verbose=verbose)
    if mod is None:
        return out

    _maybe_call(cm, mod, "_auto_discover", verbose=verbose)

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

    flat = []
    for bucket in reg.values():
        if isinstance(bucket, dict):
            flat.extend(bucket.keys())
    out.append(
        RegistryContract(
            module=module_name,
            name="LLM_REGISTRY",
            plugins=sorted(str(x) for x in set(flat)),
            note="Central plugin registry (flattened view)",
        )
    )
    return out


def build_contract_map(*, verbose: bool) -> ContractMap:
    cm = ContractMap(
        meta={
            "python": sys.version.split()[0],
            "repo_root": str(REPO_ROOT),
            "cwd": str(Path.cwd()),
        }
    )

    model_modules = [
        "core.config.schema",
        "rag.models.chunk",
        "rag.models.document",
        "rag.config.schema",
        "ingest.config.schema",
        "ingest.ingestion.base",
    ]
    for mod_name in model_modules:
        mod = _safe_import(cm, mod_name, verbose=verbose)
        if mod is None:
            continue
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if getattr(obj, "__module__", None) != mod_name:
                continue

            if _is_pydantic_model(obj):
                cm.models.append(
                    ModelContract(
                        module=mod_name,
                        name=obj.__name__,
                        fields=_extract_pydantic_fields(obj),
                    )
                )
                continue

            if obj.__name__ in {"RawDocument"} and hasattr(obj, "__annotations__"):
                hints = obj.__annotations__ or {}
                fields = [ModelField(name=k, type=_fmt_type(hints[k]), required=True) for k in sorted(hints.keys())]
                cm.models.append(ModelContract(module=mod_name, name=obj.__name__, fields=fields))

    cm.models.sort(key=lambda m: (m.module, m.name))

    protocol_modules = [
        "core.llm.chat.base",
        "core.llm.embedding.base",
        "core.llm.rerank.base",
        "core.vector_db.base",
        "rag.retrieval.base",
        "rag.pipeline.base",
        "ingest.chunking.base",
        "ingest.ingestion.base",
    ]
    for mod_name in protocol_modules:
        mod = _safe_import(cm, mod_name, verbose=verbose)
        if mod is None:
            continue
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if getattr(obj, "__module__", None) != mod_name:
                continue
            if _looks_like_protocol(obj):
                cm.protocols.append(
                    ProtocolContract(
                        module=mod_name,
                        name=obj.__name__,
                        methods=_extract_protocol_methods(obj),
                    )
                )

    cm.protocols.sort(key=lambda p: (p.module, p.name))

    cm.registries.extend(_extract_llm_registry(cm, "core.llm.registry", verbose=verbose))

    rr = _extract_registry_plugins(
        cm,
        "rag.retrieval.registry",
        dict_attr="RETRIEVER_REGISTRY",
        discover_fns=("_auto_discover",),
        note="Lazy discovery over rag.retrieval.plugins.*",
        verbose=verbose,
    )
    if rr:
        cm.registries.append(rr)

    cr = _extract_registry_plugins(
        cm,
        "ingest.chunking.registry",
        dict_attr="CHUNKER_REGISTRY",
        discover_fns=("_auto_discover",),
        note="Lazy discovery over ingest.chunking.plugins.*",
        verbose=verbose,
    )
    if cr:
        cm.registries.append(cr)

    ir = _extract_registry_plugins(
        cm,
        "ingest.ingestion.registry",
        dict_attr="REGISTRY",
        discover_fns=("_auto_discover",),
        note="Lazy discovery over ingest.ingestion.plugins.*",
        verbose=verbose,
    )
    if ir:
        cm.registries.append(ir)

    pr = _extract_pipeline_registry(
        cm,
        "rag.pipeline.registry",
        list_fn="available_pipeline_plugins",
        note="Lazy discovery over rag.pipeline.plugins.*",
        verbose=verbose,
    )
    if pr:
        cm.registries.append(pr)

    cm.registries.sort(key=lambda r: (r.module, r.name))

    def _registry_nonempty(name_contains: str) -> bool:
        for r in cm.registries:
            if name_contains in r.name and r.plugins:
                return True
        return False

    if not _registry_nonempty("LLM_REGISTRY"):
        cm.health.append(
            HealthIssue(
                level="ERROR",
                message="LLM registry appears empty. Likely cause: discovery not triggered or plugin imports failing.",
            )
        )

    if not _registry_nonempty("RETRIEVER_REGISTRY"):
        cm.health.append(
            HealthIssue(
                level="ERROR",
                message="Retriever registry appears empty. Likely cause: rag.retrieval.plugins package missing/empty or import failures.",
            )
        )

    if not _registry_nonempty("CHUNKER_REGISTRY"):
        cm.health.append(
            HealthIssue(
                level="WARN",
                message="Chunker registry appears empty. If intentional, ignore; otherwise check ingest.chunking.plugins.*",
            )
        )

    if not _registry_nonempty("REGISTRY"):
        cm.health.append(
            HealthIssue(
                level="WARN",
                message="Ingestion registry appears empty. Likely cause: ingest.ingestion.plugins package missing/empty or import failures.",
            )
        )

    if cm.import_failures:
        cm.health.append(
            HealthIssue(
                level="WARN",
                message=f"{len(cm.import_failures)} import/discovery failures detected (see Import Failures section).",
            )
        )

    if any("cannot import name 'VectorRecord'" in f.error for f in cm.import_failures):
        cm.health.append(
            HealthIssue(
                level="ERROR",
                message=(
                    "Some modules still import VectorRecord from core.vector_db.base. "
                    "Canonical name is SearchResult. Replace VectorRecord imports/usages."
                ),
            )
        )

    return cm


def render_markdown(cm: ContractMap, *, verbose: bool) -> str:
    lines: List[str] = []
    lines.append("# Contract Map")
    lines.append("")
    lines.append("## Meta")
    for k in sorted(cm.meta.keys()):
        lines.append(f"- `{k}`: `{cm.meta[k]}`")
    lines.append("")

    if cm.health:
        lines.append("## Health")
        for h in cm.health:
            lines.append(f"- **{h.level}**: {h.message}")
        lines.append("")

    if cm.import_failures:
        lines.append("## Import Failures")
        for f in cm.import_failures:
            lines.append(f"- `{f.module}`: {f.error}")
            if verbose and f.traceback:
                lines.append("")
                lines.append("```")
                lines.append(f.traceback.rstrip())
                lines.append("```")
        lines.append("")

    lines.append("## Models")
    for m in cm.models:
        lines.append(f"### `{m.module}.{m.name}`")
        if not m.fields:
            lines.append("- (no fields found)")
            lines.append("")
            continue
        for f in m.fields:
            req = "required" if f.required else "optional"
            default = "" if f.default is None else f" (default={f.default})"
            lines.append(f"- `{f.name}`: `{f.type}` — {req}{default}")
        lines.append("")

    lines.append("## Protocols")
    for p in cm.protocols:
        lines.append(f"### `{p.module}.{p.name}`")
        if not p.methods:
            lines.append("- (no methods found)")
            lines.append("")
            continue
        for m in p.methods:
            ret = f" -> {m.returns}" if m.returns else ""
            lines.append(f"- `{m.name}{m.signature}`{ret}")
        lines.append("")

    lines.append("## Registries")
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

    return "\n".join(lines).rstrip() + "\n"


def render_json(cm: ContractMap) -> str:
    payload = asdict(cm)
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a contract map from the current codebase.")
    parser.add_argument("--format", choices=["md", "json"], default="md", help="Output format")
    parser.add_argument("--out", type=str, default=None, help="Write output to a file (otherwise prints to stdout)")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include tracebacks for import/discovery failures",
    )
    parser.add_argument(
        "--fail-on-errors",
        action="store_true",
        help="Exit non-zero if any ERROR health issues exist",
    )

    args = parser.parse_args(argv)

    cm = build_contract_map(verbose=args.verbose)

    if args.format == "json":
        text = render_json(cm)
    else:
        text = render_markdown(cm, verbose=args.verbose)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)

    if args.fail_on_errors and any(h.level == "ERROR" for h in cm.health):
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
