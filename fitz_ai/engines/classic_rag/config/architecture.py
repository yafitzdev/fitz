from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml

from fitz_ai.engines.classic_rag.contracts.rules import role_by_name


class ArchitectureConfigError(RuntimeError):
    pass


def load_architecture_mapping(
    path: Path | None = None,
) -> Dict[str, str]:
    """
    Load and validate the architecture mapping.

    Returns:
        dict[str, str]: package_prefix -> role_name
    """
    if path is None:
        path = Path(__file__).resolve().parent / "architecture.yaml"

    if not path.exists():
        raise ArchitectureConfigError(f"Architecture config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ArchitectureConfigError("architecture.yaml must contain a mapping at top level")

    mapping = data.get("mapping")
    if not isinstance(mapping, dict):
        raise ArchitectureConfigError("architecture.yaml must define a 'mapping' dict")

    known_roles = role_by_name()

    resolved: Dict[str, str] = {}

    for package, role in mapping.items():
        if not isinstance(package, str):
            raise ArchitectureConfigError(f"Invalid package key: {package!r}")

        if not isinstance(role, str):
            raise ArchitectureConfigError(f"Invalid role for {package!r}: {role!r}")

        if role not in known_roles:
            raise ArchitectureConfigError(f"Unknown role {role!r} for package {package!r}")

        resolved[package] = role

    return resolved
