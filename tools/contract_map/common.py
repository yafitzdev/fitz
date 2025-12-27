# tools/contract_map/common.py
"""Shared data structures, utilities, and constants for contract map generation."""

from __future__ import annotations

import importlib
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    from typing import get_args, get_origin
except ImportError:
    get_args = None  # type: ignore[assignment]
    get_origin = None  # type: ignore[assignment]

try:
    from pydantic import BaseModel
except Exception:
    BaseModel = object  # type: ignore[assignment]


# PKG (AutoDiscoveredConfig) is initialized after REPO_ROOT below


# ============================================================================
# Repo / package discovery
# ============================================================================


def _ensure_repo_root_on_syspath() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


REPO_ROOT = _ensure_repo_root_on_syspath()

# Now initialize PKG with auto-discovery
from tools.contract_map.autodiscover import AutoDiscoveredConfig

PKG = AutoDiscoveredConfig(REPO_ROOT)


def _discover_fitz_subpackages() -> set[str]:
    """
    Discover logical top-level packages inside the main package directory.

    Example:
        fitz_ai/
          core/
          pipeline/
          ingest/
          retrieval/

    -> {"core", "pipeline", "ingest", "retrieval"}
    """
    fitz_dir = REPO_ROOT / PKG.name
    if not fitz_dir.is_dir():
        return set()

    out: set[str] = set()
    for p in fitz_dir.iterdir():
        if p.is_dir() and (p / "__init__.py").exists():
            out.add(p.name)
    return out


FITZ_SUBPACKAGES = _discover_fitz_subpackages()

TOPLEVEL_PACKAGES = {"tools"} | FITZ_SUBPACKAGES

DEFAULT_LAYOUT_EXCLUDES = {
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


# ============================================================================
# Data Structures
# ============================================================================


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
    """
    Import graph with violation detection.

    Attributes:
        edges: All import edges (both module-level and lazy)
        violations: Module-level imports that violate architecture rules
        lazy_ok: Lazy imports (inside functions) that would violate rules
                 if they were at module level, but are OK because they're lazy
    """

    edges: List[ImportEdge] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    lazy_ok: List[str] = field(default_factory=list)


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


# ============================================================================
# Utility Functions
# ============================================================================


def fmt_type(tp: Any) -> str:
    """Format a type annotation as a string."""
    if tp is None:
        return "None"

    if get_origin is not None and get_args is not None:
        origin = get_origin(tp)
        if origin is None:
            return getattr(tp, "__name__", None) or str(tp)

        args = get_args(tp)
        origin_name = getattr(origin, "__name__", None) or str(origin)
        if args:
            return f"{origin_name}[{', '.join(fmt_type(a) for a in args)}]"
        return origin_name

    return getattr(tp, "__name__", None) or str(tp)


def is_pydantic_model(obj: Any) -> bool:
    """Check if an object is a Pydantic model class."""
    try:
        return isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel
    except Exception:
        return False


def safe_import(cm: ContractMap, module: str, *, verbose: bool) -> object | None:
    """Safely import a module, logging failures to the ContractMap."""
    try:
        return importlib.import_module(module)
    except Exception as exc:
        tb = traceback.format_exc() if verbose else None
        cm.import_failures.append(
            ImportFailure(module=module, error=f"{type(exc).__name__}: {exc}", traceback=tb)
        )
        return None


def should_exclude_path(rel: Path, excludes: set[str]) -> bool:
    """Check if a path should be excluded based on exclude patterns."""
    return any(part in excludes for part in rel.parts)


def iter_python_files(root: Path, *, excludes: set[str]) -> Iterable[Path]:
    """Iterate over all Python files in a directory tree."""
    for p in root.rglob("*.py"):
        rel = p.relative_to(root)
        if should_exclude_path(rel, excludes):
            continue
        yield p


def module_name_from_path(path: Path) -> str | None:
    """Convert a file path to a Python module name."""
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

    # tools.*
    if parts[0] == "tools":
        return ".".join(parts)

    # <package>.<subpkg>.*
    if parts[0] == PKG.name and len(parts) > 1 and parts[1] in FITZ_SUBPACKAGES:
        return ".".join(parts)

    return None


def toplevel(pkg: str | None) -> str | None:
    """Extract the top-level package name."""
    if not pkg:
        return None

    pkg_prefix = f"{PKG.name}."
    if pkg.startswith(pkg_prefix):
        parts = pkg.split(".")
        if len(parts) >= 2:
            return parts[1]

    top = pkg.split(".", 1)[0]
    return top if top in TOPLEVEL_PACKAGES else None
