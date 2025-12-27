# tools/contract_map/autodiscover.py
"""
Auto-discovery of package structure, plugins, registries, models, and protocols.

This module eliminates hardcoded paths by scanning the codebase to find:
- Main package name
- YAML-based plugin directories
- Python-based plugin namespaces
- Registry modules
- Model modules (Pydantic BaseModel subclasses)
- Protocol modules (typing.Protocol subclasses)
- Consumer patterns for hotspot detection
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple


def find_main_package(repo_root: Path) -> str | None:
    """
    Find the main Python package in the repo.

    Looks for a directory with __init__.py that's not tests/tools/examples.
    """
    exclude = {"tests", "tools", "examples", "docs", "scripts", "bin"}

    candidates = []
    for child in repo_root.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name.startswith("_"):
            continue
        if child.name in exclude:
            continue
        if child.name.endswith(".egg-info"):
            continue
        if (child / "__init__.py").exists():
            candidates.append(child.name)

    # Prefer packages that don't have generic names
    generic = {"src", "lib", "pkg"}
    specific = [c for c in candidates if c not in generic]

    if specific:
        return specific[0]
    if candidates:
        return candidates[0]
    return None


def find_yaml_plugin_dirs(repo_root: Path, pkg_name: str) -> List[Tuple[str, str]]:
    """
    Find directories containing YAML plugin definitions.

    Returns list of (relative_path, plugin_type) tuples.
    Plugin type is inferred from the directory structure.
    """
    pkg_dir = repo_root / pkg_name
    if not pkg_dir.exists():
        return []

    results = []

    # Find all directories with .yaml files (excluding config files)
    for yaml_file in pkg_dir.rglob("*.yaml"):
        # Skip config/default files
        if yaml_file.name in ("default.yaml", "config.yaml"):
            continue

        parent = yaml_file.parent
        rel_path = str(parent.relative_to(repo_root)).replace("\\", "/")

        # Infer plugin type from path
        parts = rel_path.split("/")
        if "plugins" in parts:
            # e.g., fitz_ai/vector_db/plugins -> vector_db
            idx = parts.index("plugins")
            if idx > 0:
                plugin_type = parts[idx - 1]
            else:
                plugin_type = "plugins"
        else:
            # e.g., fitz_ai/llm/chat -> chat
            plugin_type = parts[-1] if parts else "unknown"

        if (rel_path, plugin_type) not in results:
            results.append((rel_path, plugin_type))

    return results


def find_python_plugin_namespaces(repo_root: Path, pkg_name: str) -> List[Tuple[str, str]]:
    """
    Find Python plugin namespaces by scanning for plugin_name attributes.

    Returns list of (namespace, interface_name) tuples.
    """
    pkg_dir = repo_root / pkg_name
    if not pkg_dir.exists():
        return []

    results = []
    seen_namespaces = set()

    # Look for directories named "plugins"
    for plugins_dir in pkg_dir.rglob("plugins"):
        if not plugins_dir.is_dir():
            continue

        # Check if it contains Python files with plugin_name
        has_plugins = False
        for py_file in plugins_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            try:
                content = py_file.read_text(encoding="utf-8")
                if "plugin_name" in content and "class " in content:
                    has_plugins = True
                    break
            except Exception:
                continue

        if has_plugins:
            # Convert path to module namespace
            rel = plugins_dir.relative_to(repo_root)
            namespace = ".".join(rel.parts)

            if namespace not in seen_namespaces:
                seen_namespaces.add(namespace)
                # Infer interface name from parent directory
                parts = list(rel.parts)
                if len(parts) >= 2:
                    parent = parts[-2]
                    interface = f"{parent.title().replace('_', '')}Plugin"
                else:
                    interface = "Plugin"
                results.append((namespace, interface))

    return results


def find_registry_modules(repo_root: Path, pkg_name: str) -> List[str]:
    """
    Find registry modules by looking for registry.py files.

    Returns list of module paths.
    """
    pkg_dir = repo_root / pkg_name
    if not pkg_dir.exists():
        return []

    results = []
    for registry_file in pkg_dir.rglob("registry.py"):
        rel = registry_file.relative_to(repo_root)
        # Convert to module path
        parts = list(rel.parts)
        parts[-1] = parts[-1].removesuffix(".py")
        module = ".".join(parts)
        results.append(module)

    return sorted(results)


def find_model_modules(repo_root: Path, pkg_name: str) -> List[str]:
    """
    Find modules containing Pydantic models by scanning for BaseModel imports.

    Returns list of module paths.
    """
    pkg_dir = repo_root / pkg_name
    if not pkg_dir.exists():
        return []

    results = []

    # Common locations for models
    model_patterns = [
        "**/models/*.py",
        "**/models.py",
        "**/schema.py",
        "**/schemas.py",
        "**/config/schema.py",
        "**/state/schema.py",
    ]

    seen = set()
    for pattern in model_patterns:
        for py_file in pkg_dir.glob(pattern):
            if py_file.name.startswith("_"):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                # Check for Pydantic model definitions
                if "BaseModel" in content and "class " in content:
                    rel = py_file.relative_to(repo_root)
                    parts = list(rel.parts)
                    parts[-1] = parts[-1].removesuffix(".py")
                    module = ".".join(parts)
                    if module not in seen:
                        seen.add(module)
                        results.append(module)
            except Exception:
                continue

    return sorted(results)


def find_protocol_modules(repo_root: Path, pkg_name: str) -> List[str]:
    """
    Find modules containing Protocol definitions.

    Returns list of module paths.
    """
    pkg_dir = repo_root / pkg_name
    if not pkg_dir.exists():
        return []

    results = []

    # Common locations for protocols/interfaces
    protocol_patterns = [
        "**/base.py",
        "**/protocol.py",
        "**/protocols.py",
        "**/interface.py",
        "**/interfaces.py",
        "**/contracts.py",
    ]

    seen = set()
    for pattern in protocol_patterns:
        for py_file in pkg_dir.glob(pattern):
            if py_file.name.startswith("_"):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
                # Check for Protocol definitions
                if "Protocol" in content and "class " in content:
                    # Verify it's actually using typing.Protocol
                    if "from typing" in content or "typing.Protocol" in content:
                        rel = py_file.relative_to(repo_root)
                        parts = list(rel.parts)
                        parts[-1] = parts[-1].removesuffix(".py")
                        module = ".".join(parts)
                        if module not in seen:
                            seen.add(module)
                            results.append(module)
            except Exception:
                continue

    return sorted(results)


def find_config_loaders(repo_root: Path, pkg_name: str) -> List[str]:
    """
    Find config loader modules.

    Returns list of package paths that have loader.py.
    """
    pkg_dir = repo_root / pkg_name
    if not pkg_dir.exists():
        return []

    results = []
    for loader_file in pkg_dir.rglob("loader.py"):
        rel = loader_file.relative_to(repo_root)
        parts = list(rel.parts)
        # Get the parent package (without loader.py)
        if len(parts) > 1:
            pkg_path = ".".join(parts[:-1])
            results.append(pkg_path)

    # Also check for config directories
    for config_dir in pkg_dir.rglob("config"):
        if config_dir.is_dir() and (config_dir / "loader.py").exists():
            rel = config_dir.relative_to(repo_root)
            pkg_path = ".".join(rel.parts)
            if pkg_path not in results:
                results.append(pkg_path)

    return sorted(results)


def discover_consumer_patterns(
    repo_root: Path, pkg_name: str, plugin_namespaces: List[Tuple[str, str]]
) -> Dict[str, Tuple[str, ...]]:
    """
    Discover consumer patterns for hotspot detection.

    Analyzes the codebase to find patterns that indicate plugin usage.
    Returns dict mapping interface names to search patterns.
    """
    patterns: Dict[str, Set[str]] = {}

    pkg_dir = repo_root / pkg_name
    if not pkg_dir.exists():
        return {}

    # Extract interface names from discovered namespaces
    for namespace, interface in plugin_namespaces:
        patterns[interface] = set()

        # Add the namespace as a pattern (imports)
        parent_ns = ".".join(namespace.split(".")[:-1])
        if parent_ns:
            patterns[interface].add(parent_ns)

    # Scan for common plugin access patterns
    plugin_access_patterns = [
        r"get_(\w+)_plugin\(",
        r"(\w+)Plugin",
        r"available_(\w+)_plugins",
        r"(\w+)_REGISTRY",
        r"(\w+)Engine\.from_name\(",
    ]

    for py_file in pkg_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        try:
            content = py_file.read_text(encoding="utf-8")

            for pattern in plugin_access_patterns:
                for match in re.finditer(pattern, content):
                    # Try to associate with known interfaces
                    matched_text = match.group(0)
                    for interface in patterns:
                        # Check if interface name relates to this pattern
                        base_name = interface.replace("Plugin", "").lower()
                        if base_name in matched_text.lower():
                            patterns[interface].add(matched_text.split("(")[0])

        except Exception:
            continue

    # Convert sets to tuples
    return {k: tuple(sorted(v)) for k, v in patterns.items() if v}


def discover_hotspot_namespaces(
    repo_root: Path, pkg_name: str
) -> List[Tuple[str, str]]:
    """
    Discover plugin namespaces for hotspot detection.

    Returns list of (namespace, interface) tuples for Python-based plugins.
    """
    return find_python_plugin_namespaces(repo_root, pkg_name)


class AutoDiscoveredConfig:
    """
    Auto-discovered package configuration.

    All paths are discovered dynamically from the codebase structure.
    """

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self._pkg_name: str | None = None
        self._yaml_dirs: List[Tuple[str, str]] | None = None
        self._python_plugins: List[Tuple[str, str]] | None = None
        self._registries: List[str] | None = None
        self._models: List[str] | None = None
        self._protocols: List[str] | None = None
        self._config_loaders: List[str] | None = None
        self._consumer_patterns: Dict[str, Tuple[str, ...]] | None = None

    @property
    def name(self) -> str:
        """Main package name."""
        if self._pkg_name is None:
            self._pkg_name = find_main_package(self.repo_root) or "unknown"
        return self._pkg_name

    @property
    def yaml_plugin_dirs(self) -> List[Tuple[str, str]]:
        """YAML plugin directories as (path, type) tuples."""
        if self._yaml_dirs is None:
            self._yaml_dirs = find_yaml_plugin_dirs(self.repo_root, self.name)
        return self._yaml_dirs

    @property
    def python_plugin_namespaces(self) -> List[Tuple[str, str]]:
        """Python plugin namespaces as (namespace, interface) tuples."""
        if self._python_plugins is None:
            self._python_plugins = find_python_plugin_namespaces(self.repo_root, self.name)
        return self._python_plugins

    @property
    def registry_modules(self) -> List[str]:
        """Registry module paths."""
        if self._registries is None:
            self._registries = find_registry_modules(self.repo_root, self.name)
        return self._registries

    @property
    def model_modules(self) -> List[str]:
        """Model module paths."""
        if self._models is None:
            self._models = find_model_modules(self.repo_root, self.name)
        return self._models

    @property
    def protocol_modules(self) -> List[str]:
        """Protocol module paths."""
        if self._protocols is None:
            self._protocols = find_protocol_modules(self.repo_root, self.name)
        return self._protocols

    @property
    def config_loader_packages(self) -> List[str]:
        """Config loader package paths."""
        if self._config_loaders is None:
            self._config_loaders = find_config_loaders(self.repo_root, self.name)
        return self._config_loaders

    @property
    def consumer_patterns(self) -> Dict[str, Tuple[str, ...]]:
        """Consumer patterns for hotspot detection."""
        if self._consumer_patterns is None:
            self._consumer_patterns = discover_consumer_patterns(
                self.repo_root, self.name, self.python_plugin_namespaces
            )
        return self._consumer_patterns

    def get_registry_for_type(self, plugin_type: str) -> str | None:
        """Find registry module for a given plugin type."""
        for reg in self.registry_modules:
            if plugin_type in reg:
                return reg
        return None


if __name__ == "__main__":
    # Test discovery
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    config = AutoDiscoveredConfig(repo)

    print(f"Package name: {config.name}")
    print(f"\nYAML plugin dirs:")
    for path, ptype in config.yaml_plugin_dirs:
        print(f"  {path} ({ptype})")

    print(f"\nPython plugin namespaces:")
    for ns, iface in config.python_plugin_namespaces:
        print(f"  {ns} -> {iface}")

    print(f"\nRegistry modules:")
    for reg in config.registry_modules:
        print(f"  {reg}")

    print(f"\nModel modules:")
    for mod in config.model_modules:
        print(f"  {mod}")

    print(f"\nProtocol modules:")
    for mod in config.protocol_modules:
        print(f"  {mod}")

    print(f"\nConfig loaders:")
    for pkg in config.config_loader_packages:
        print(f"  {pkg}")

    print(f"\nConsumer patterns:")
    for iface, patterns in config.consumer_patterns.items():
        print(f"  {iface}: {patterns}")
