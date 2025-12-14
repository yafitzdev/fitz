# tools/contract_map.py
from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, get_type_hints

try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

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
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


REPO_ROOT = _ensure_repo_root_on_syspath()

_DEFAULT_LAYOUT_EXCLUDES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    "dist",
    "build",
    "node_modules",
}

_TOPLEVEL_PACKAGES = ("core", "rag", "ingest", "tools")


@dataclass(slots=True)
class ImportFailure:
    module: str
    error: str
    traceback: str | None = None


@dataclass(slots=True)
class ModelField:
    name: str
    type: str
    required: bool
    default: str | None = None


@dataclass(slots=True)
class ModelContract:
    module: str
    name: str
    fields: List[ModelField] = field(default_factory=list)


@dataclass(slots=True)
class MethodContract:
    name: str
    signature: str
    returns: str | None


@dataclass(slots=True)
class ProtocolContract:
    module: str
    name: str
    methods: List[MethodContract] = field(default_factory=list)


@dataclass(slots=True)
class RegistryContract:
    module: str
    name: str
    plugins: List[str] = field(default_factory=list)
    note: str | None = None


@dataclass(slots=True)
class HealthIssue:
    level: str  # "WARN" | "ERROR"
    message: str


@dataclass(slots=True)
class Entrypoint:
    kind: str
    name: str
    target: str


@dataclass(slots=True)
class ImportEdge:
    src: str
    dst: str
    count: int


@dataclass(slots=True)
class ImportGraph:
    edges: List[ImportEdge] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)


@dataclass(slots=True)
class DiscoveryReport:
    namespace: str
    note: str
    modules_scanned: int
    plugins_found: List[str]
    failures: List[str]
    duplicates: List[str]


@dataclass(slots=True)
class Hotspot:
    interface: str
    implementations: List[str]
    consumers: List[str]


@dataclass(slots=True)
class ConfigSurface:
    config_models: List[str]
    default_yamls: List[str]
    loaders: List[str]
    load_callsites: List[str]


@dataclass(slots=True)
class CodeStats:
    py_files: int
    total_lines: int
    todo_fixme: int
    any_mentions: int
    largest_modules: List[str]


@dataclass(slots=True)
class ContractMap:
    meta: Dict[str, Any] = field(default_factory=dict)
    import_failures: List[ImportFailure] = field(default_factory=list)
    health: List[HealthIssue] = field(default_factory=list)

    models: List[ModelContract] = field(default_factory=list)
    protocols: List[ProtocolContract] = field(default_factory=list)
    registries: List[RegistryContract] = field(default_factory=list)

    import_graph: ImportGraph | None = None
    entrypoints: List[Entrypoint] = field(default_factory=list)
    discovery: List[DiscoveryReport] = field(default_factory=list)
    hotspots: List[Hotspot] = field(default_factory=list)
    config_surface: ConfigSurface | None = None
    invariants: List[str] = field(default_factory=list)
    stats: CodeStats | None = None


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

    if hasattr(model_cls, "model_fields"):  # pydantic v2
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

    if hasattr(model_cls, "__fields__"):  # pydantic v1
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
        fields.append(
            ModelField(name=name, type=_fmt_type(hints[name]), required=True, default=None)
        )
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
        cm.import_failures.append(
            ImportFailure(module=module, error=f"{type(exc).__name__}: {exc}", traceback=tb)
        )
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
    cm: ContractMap, module_name: str, *, verbose: bool
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


def _should_exclude_path(rel: Path, excludes: set[str]) -> bool:
    return any(part in excludes for part in rel.parts)


def _build_layout_tree(root: Path, *, max_depth: int | None, excludes: set[str]) -> Dict[str, Any]:
    tree: Dict[str, Any] = {}

    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if _should_exclude_path(rel, excludes):
            continue
        if max_depth is not None and len(rel.parts) > max_depth:
            continue

        node = tree
        parts = rel.parts
        for i, part in enumerate(parts):
            is_last = i == len(parts) - 1
            if is_last:
                if p.is_dir():
                    key = f"{part}/"
                    node = node.setdefault(key, {})  # type: ignore[assignment]
                else:
                    node.setdefault(part, None)
            else:
                key = f"{part}/"
                node = node.setdefault(key, {})  # type: ignore[assignment]

    return tree


def _classes_for_file(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    return sorted(node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef))


def _get_classes_for_file(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    return sorted(node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef))


def _render_layout_tree(
    tree: Dict[str, Any],
    prefix: str = "",
    *,
    root: Path,
) -> List[str]:
    lines: List[str] = []

    entries = sorted(
        tree.items(),
        key=lambda kv: (0 if kv[0].endswith("/") else 1, kv[0]),
    )

    for idx, (name, child) in enumerate(entries):
        last = idx == len(entries) - 1
        connector = "└── " if last else "├── "

        label = name

        # 🔹 annotate .py files inline
        if not name.endswith("/") and name.endswith(".py"):
            file_path = root / name
            classes = _get_classes_for_file(file_path)

            if classes:
                if len(classes) == 1:
                    label = f"{name} (class: {classes[0]})"
                else:
                    label = f"{name} (classes: {', '.join(classes)})"

        lines.append(f"{prefix}{connector}{label}")

        if isinstance(child, dict) and child:
            extension = "    " if last else "│   "
            lines.extend(
                _render_layout_tree(
                    child,
                    prefix=prefix + extension,
                    root=root / name.rstrip("/"),
                )
            )

    return lines


def _extract_classes_from_file(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    return sorted(node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef))


def _build_class_layout_tree(root: Path, *, excludes: set[str]) -> dict[str, Any]:
    tree: dict[str, Any] = {}

    for p in _iter_python_files(root, excludes=excludes):
        rel = p.relative_to(root)

        node = tree
        for part in rel.parts[:-1]:
            node = node.setdefault(f"{part}/", {})  # type: ignore[assignment]

        classes = _extract_classes_from_file(p)
        if classes:
            node[p.name] = classes

    return tree


def _render_class_layout_tree(tree: dict[str, Any], prefix: str = "") -> list[str]:
    lines: list[str] = []
    entries = sorted(tree.items(), key=lambda kv: kv[0])

    for idx, (name, child) in enumerate(entries):
        last = idx == len(entries) - 1
        connector = "└── " if last else "├── "
        lines.append(f"{prefix}{connector}{name}")

        extension = "    " if last else "│   "

        if isinstance(child, dict):
            lines.extend(_render_class_layout_tree(child, prefix + extension))
        elif isinstance(child, list):
            for i, cls in enumerate(child):
                cls_last = i == len(child) - 1
                cls_connector = "└── " if cls_last else "├── "
                lines.append(f"{prefix}{extension}{cls_connector}{cls}")

    return lines


def _iter_python_files(root: Path, *, excludes: set[str]) -> Iterable[Path]:
    for p in root.rglob("*.py"):
        rel = p.relative_to(root)
        if _should_exclude_path(rel, excludes):
            continue
        yield p


def _module_name_from_path(path: Path) -> str | None:
    try:
        rel = path.relative_to(REPO_ROOT)
    except Exception:
        return None

    parts = list(rel.parts)
    if not parts:
        return None

    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].removesuffix(".py")

    if not parts:
        return None

    if parts[0] not in _TOPLEVEL_PACKAGES:
        return None

    return ".".join(parts)


def _toplevel(pkg: str | None) -> str | None:
    if not pkg:
        return None
    top = pkg.split(".", 1)[0]
    return top if top in _TOPLEVEL_PACKAGES else None


def _resolve_from_import(*, current_module: str, module: str | None, level: int) -> str | None:
    if level <= 0:
        return module

    cur_parts = current_module.split(".")
    drop = max(level, 1)
    base_parts = cur_parts[:-drop]
    if not base_parts:
        return module

    if module:
        return ".".join(base_parts + module.split("."))
    return ".".join(base_parts)


def _build_import_graph(root: Path, *, excludes: set[str]) -> ImportGraph:
    edge_counts: Dict[Tuple[str, str], int] = {}

    for file in _iter_python_files(root, excludes=excludes):
        mod = _module_name_from_path(file)
        src = _toplevel(mod)
        if not src:
            continue

        try:
            text = file.read_text(encoding="utf-8")
        except Exception:
            continue

        try:
            tree = ast.parse(text, filename=str(file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dst = _toplevel(alias.name)
                    if not dst or dst == src:
                        continue
                    edge_counts[(src, dst)] = edge_counts.get((src, dst), 0) + 1

            elif isinstance(node, ast.ImportFrom):
                target = _resolve_from_import(
                    current_module=mod or "",
                    module=node.module,
                    level=int(getattr(node, "level", 0) or 0),
                )
                dst = _toplevel(target)
                if not dst or dst == src:
                    continue
                edge_counts[(src, dst)] = edge_counts.get((src, dst), 0) + 1

    edges = [ImportEdge(src=k[0], dst=k[1], count=v) for k, v in edge_counts.items()]
    edges.sort(key=lambda e: (-e.count, e.src, e.dst))

    violations: List[str] = []
    for e in edges:
        if e.src == "core" and e.dst in {"rag", "ingest"}:
            violations.append(f"VIOLATION: core imports {e.dst} ({e.count}x)")
        if e.src == "ingest" and e.dst == "rag":
            violations.append(f"VIOLATION: ingest imports rag ({e.count}x)")

    return ImportGraph(edges=edges, violations=sorted(violations))


def _read_pyproject() -> dict[str, Any] | None:
    pyproject = REPO_ROOT / "pyproject.toml"
    if not pyproject.exists():
        return None
    if tomllib is None:
        return None
    try:
        return tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except Exception:
        return None


def _discover_entrypoints(root: Path, *, excludes: set[str]) -> List[Entrypoint]:
    eps: List[Entrypoint] = []

    data = _read_pyproject()
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

    for p in _iter_python_files(root, excludes=excludes):
        rel = p.relative_to(root)
        if p.name == "__main__.py":
            eps.append(Entrypoint(kind="module_main", name=str(rel.parent), target=str(rel)))
        if p.name == "cli.py":
            eps.append(Entrypoint(kind="cli_module", name=str(rel.parent), target=str(rel)))

    eps.sort(key=lambda e: (e.kind, e.name, e.target))
    return eps


def _find_default_yamls(root: Path, *, excludes: set[str]) -> List[str]:
    out: List[str] = []
    for p in root.rglob("default.yaml"):
        rel = p.relative_to(root)
        if _should_exclude_path(rel, excludes):
            continue
        out.append(str(rel))
    return sorted(out)


def _list_loader_modules() -> List[str]:
    out: List[str] = []
    for pkg in ("core.config", "fitz.rag.config", "fitz.ingest.config"):
        mod = f"{pkg}.loader"
        try:
            importlib.import_module(mod)
            out.append(mod)
        except Exception:
            continue
    return sorted(out)


def _find_load_callsites(root: Path, *, excludes: set[str]) -> List[str]:
    needles = ("load_config(", "load_rag_config(", "load_ingest_config(", "load_fitz_config(")
    hits: List[str] = []

    for p in _iter_python_files(root, excludes=excludes):
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue

        if any(n in text for n in needles):
            hits.append(str(p.relative_to(root)))

    return sorted(set(hits))


def _plugin_predicate_for_namespace(namespace: str):
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

    if namespace == "core.vector_db.plugins":

        def is_plugin(cls: type) -> bool:
            if not isinstance(getattr(cls, "plugin_name", None), str):
                return False
            if getattr(cls, "plugin_type", None) != "vector_db":
                return False
            return callable(getattr(cls, "search", None))

        def plugin_id(cls: type) -> str:
            return f"{cls.__module__}.{cls.__name__}"

        return is_plugin, plugin_id

    if namespace == "fitz.rag.retrieval.plugins":

        def is_plugin(cls: type) -> bool:
            if not isinstance(getattr(cls, "plugin_name", None), str):
                return False
            return callable(getattr(cls, "retrieve", None))

        def plugin_id(cls: type) -> str:
            return f"{cls.__module__}.{cls.__name__}"

        return is_plugin, plugin_id

    if namespace == "fitz.rag.pipeline.plugins":

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


def _scan_discovery(namespace: str, note: str) -> DiscoveryReport:
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

    import pkgutil

    is_plugin, plugin_id = _plugin_predicate_for_namespace(namespace)

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


def _compute_hotspots(root: Path, *, excludes: set[str]) -> List[Hotspot]:
    impl: Dict[str, List[str]] = {}
    consumers: Dict[str, List[str]] = {}

    expected = [
        ("fitz.core.llm.chat.plugins", "ChatPlugin"),
        ("fitz.core.llm.embedding.plugins", "EmbeddingPlugin"),
        ("fitz.core.llm.rerank.plugins", "RerankPlugin"),
        ("core.vector_db.plugins", "VectorDBPlugin"),
        ("fitz.rag.retrieval.plugins", "RetrievalPlugin"),
        ("fitz.rag.pipeline.plugins", "PipelinePlugin"),
        ("fitz.ingest.chunking.plugins", "ChunkerPlugin"),
        ("fitz.ingest.ingestion.plugins", "IngestPlugin"),
    ]
    for ns, iface in expected:
        rep = _scan_discovery(ns, note="hotspot scan")
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
        "RetrievalPlugin": ("fitz.rag.retrieval", "get_retriever_plugin(", "RetrieverEngine.from_name("),
        "PipelinePlugin": ("fitz.rag.pipeline", "get_pipeline_plugin(", "available_pipeline_plugins("),
        "ChunkerPlugin": ("fitz.ingest.chunking", "get_chunker_plugin(", "ChunkingEngine"),
        "IngestPlugin": ("fitz.ingest.ingestion", "get_ingest_plugin(", "IngestionEngine"),
    }

    for p in _iter_python_files(root, excludes=excludes):
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


def _compute_stats(root: Path, *, excludes: set[str]) -> CodeStats:
    py_files = 0
    total_lines = 0
    todo_fixme = 0
    any_mentions = 0
    module_sizes: List[Tuple[int, str]] = []

    for p in _iter_python_files(root, excludes=excludes):
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


def _compute_config_surface(cm: ContractMap, *, excludes: set[str]) -> ConfigSurface:
    config_models = [f"{m.module}.{m.name}" for m in cm.models if ".config.schema" in m.module]
    default_yamls = _find_default_yamls(REPO_ROOT, excludes=excludes)
    loaders = _list_loader_modules()
    load_callsites = _find_load_callsites(REPO_ROOT, excludes=excludes)
    return ConfigSurface(
        config_models=sorted(config_models),
        default_yamls=default_yamls,
        loaders=loaders,
        load_callsites=load_callsites,
    )


def _compute_invariants(cm: ContractMap) -> List[str]:
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


def build_contract_map(*, verbose: bool, layout_depth: int | None) -> ContractMap:
    cm = ContractMap(
        meta={
            "python": sys.version.split()[0],
            "repo_root": str(REPO_ROOT),
            "cwd": str(Path.cwd()),
        }
    )

    model_modules = [
        "core.config.schema",
        "core.models.chunk",
        "core.models.document",
        "fitz.rag.config.schema",
        "fitz.ingest.config.schema",
        "fitz.ingest.ingestion.base",
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
                fields = [
                    ModelField(name=k, type=_fmt_type(hints[k]), required=True)
                    for k in sorted(hints.keys())
                ]
                cm.models.append(ModelContract(module=mod_name, name=obj.__name__, fields=fields))

    cm.models.sort(key=lambda m: (m.module, m.name))

    protocol_modules = [
        "fitz.core.llm.chat.base",
        "fitz.core.llm.embedding.base",
        "fitz.core.llm.rerank.base",
        "core.vector_db.base",
        "fitz.rag.retrieval.base",
        "fitz.rag.pipeline.base",
        "fitz.ingest.chunking.base",
        "fitz.ingest.ingestion.base",
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

    cm.registries.extend(_extract_llm_registry(cm, "fitz.core.llm.registry", verbose=verbose))

    rr = _extract_registry_plugins(
        cm,
        "fitz.rag.retrieval.registry",
        dict_attr="RETRIEVER_REGISTRY",
        discover_fns=("_auto_discover",),
        note="Lazy discovery over rag.retrieval.plugins.*",
        verbose=verbose,
    )
    if rr:
        cm.registries.append(rr)

    cr = _extract_registry_plugins(
        cm,
        "fitz.ingest.chunking.registry",
        dict_attr="CHUNKER_REGISTRY",
        discover_fns=("_auto_discover",),
        note="Lazy discovery over ingest.chunking.plugins.*",
        verbose=verbose,
    )
    if cr:
        cm.registries.append(cr)

    ir = _extract_registry_plugins(
        cm,
        "fitz.ingest.ingestion.registry",
        dict_attr="REGISTRY",
        discover_fns=("_auto_discover",),
        note="Lazy discovery over ingest.ingestion.plugins.*",
        verbose=verbose,
    )
    if ir:
        cm.registries.append(ir)

    pr = _extract_pipeline_registry(
        cm,
        "fitz.rag.pipeline.registry",
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

    if cm.import_failures:
        cm.health.append(
            HealthIssue(
                level="WARN",
                message=f"{len(cm.import_failures)} import/discovery failures detected (see Import Failures section).",
            )
        )

    cm.import_graph = _build_import_graph(REPO_ROOT, excludes=_DEFAULT_LAYOUT_EXCLUDES)
    cm.entrypoints = _discover_entrypoints(REPO_ROOT, excludes=_DEFAULT_LAYOUT_EXCLUDES)

    cm.discovery = [
        _scan_discovery("fitz.core.llm.chat.plugins", "LLM chat plugins (Option A discovery)"),
        _scan_discovery("fitz.core.llm.embedding.plugins", "LLM embedding plugins (Option A discovery)"),
        _scan_discovery("fitz.core.llm.rerank.plugins", "LLM rerank plugins (Option A discovery)"),
        _scan_discovery("core.vector_db.plugins", "Vector DB plugins (Option A discovery)"),
        _scan_discovery("fitz.rag.retrieval.plugins", "RAG retriever plugins (Option A discovery)"),
        _scan_discovery("fitz.rag.pipeline.plugins", "RAG pipeline plugins (Option A discovery)"),
        _scan_discovery("fitz.ingest.chunking.plugins", "Ingest chunking plugins (Option A discovery)"),
        _scan_discovery(
            "fitz.ingest.ingestion.plugins", "Ingest ingestion plugins (Option A discovery)"
        ),
    ]

    cm.hotspots = _compute_hotspots(REPO_ROOT, excludes=_DEFAULT_LAYOUT_EXCLUDES)
    cm.config_surface = _compute_config_surface(cm, excludes=_DEFAULT_LAYOUT_EXCLUDES)
    cm.invariants = _compute_invariants(cm)
    cm.stats = _compute_stats(REPO_ROOT, excludes=_DEFAULT_LAYOUT_EXCLUDES)

    return cm


def render_markdown(cm: ContractMap, *, verbose: bool, layout_depth: int | None) -> str:
    lines: List[str] = []
    lines.append("# Contract Map")
    lines.append("")

    lines.append("## Meta")
    for k in sorted(cm.meta.keys()):
        lines.append(f"- `{k}`: `{cm.meta[k]}`")
    lines.append("")

    lines.append("## Project Layout")
    lines.append(f"- root: `{REPO_ROOT}`")
    lines.append(f"- excludes: `{sorted(_DEFAULT_LAYOUT_EXCLUDES)}`")
    lines.append(f"- max_depth: `{layout_depth if layout_depth is not None else 'unlimited'}`")
    lines.append("")
    layout_tree = _build_layout_tree(
        REPO_ROOT, max_depth=layout_depth, excludes=_DEFAULT_LAYOUT_EXCLUDES
    )
    lines.append("```")
    lines.append(f"{REPO_ROOT.name}/")
    lines.extend(_render_layout_tree(layout_tree, prefix="", root=REPO_ROOT))
    lines.append("```")
    lines.append("")

    if cm.import_graph:
        lines.append("## Import Graph")
        if cm.import_graph.violations:
            for v in cm.import_graph.violations:
                lines.append(f"- **ERROR**: {v}")
        else:
            lines.append("- (no layering violations detected)")
        lines.append("")
        lines.append("Top edges:")
        for e in cm.import_graph.edges[:20]:
            lines.append(f"- `{e.src} -> {e.dst}`: {e.count}x")
        lines.append("")

    lines.append("## Entrypoints")
    if not cm.entrypoints:
        lines.append("- (none detected)")
    else:
        for ep in cm.entrypoints:
            lines.append(f"- `{ep.kind}`: `{ep.name}` -> `{ep.target}`")
    lines.append("")

    lines.append("## Discovery Report")
    for rep in cm.discovery:
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

    if cm.config_surface:
        cs = cm.config_surface
        lines.append("## Config Surface")
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

    lines.append("## Runtime Invariants")
    if cm.invariants:
        for inv in cm.invariants:
            lines.append(f"- {inv}")
    else:
        lines.append("- (none)")
    lines.append("")

    if cm.stats:
        s = cm.stats
        lines.append("## Code Stats")
        lines.append(f"- python_files: `{s.py_files}`")
        lines.append(f"- total_lines: `{s.total_lines}`")
        lines.append(f"- TODO/FIXME lines: `{s.todo_fixme}`")
        lines.append(f"- 'Any' mentions: `{s.any_mentions}`")
        lines.append("- largest modules:")
        for m in s.largest_modules:
            lines.append(f"  - {m}")
        lines.append("")

    lines.append("## Contract Hotspots")
    for h in cm.hotspots:
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
    parser = argparse.ArgumentParser(
        description="Generate a contract map from the current codebase."
    )
    parser.add_argument("--format", choices=["md", "json"], default="md", help="Output format")
    parser.add_argument(
        "--out", type=str, default=None, help="Write output to a file (otherwise prints to stdout)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Include tracebacks for import/discovery failures"
    )
    parser.add_argument(
        "--fail-on-errors",
        action="store_true",
        help="Exit non-zero if any ERROR health issues exist",
    )
    parser.add_argument(
        "--layout-depth", type=int, default=None, help="Max depth for Project Layout tree"
    )

    args = parser.parse_args(argv)

    cm = build_contract_map(verbose=args.verbose, layout_depth=args.layout_depth)

    if args.format == "json":
        text = render_json(cm)
    else:
        text = render_markdown(cm, verbose=args.verbose, layout_depth=args.layout_depth)

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
