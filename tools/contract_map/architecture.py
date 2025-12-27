# tools/contract_map/architecture.py
from __future__ import annotations

import importlib

from .common import PKG


def _get_architecture_functions():
    """Dynamically import architecture functions based on PKG config."""
    arch_module = importlib.import_module(f"{PKG.name}.engines.classic_rag.config.architecture")
    rules_module = importlib.import_module(f"{PKG.name}.engines.classic_rag.contracts.rules")
    return (
        getattr(arch_module, "load_architecture_mapping"),
        getattr(rules_module, "allowed_importers"),
    )


class RoleResolver:
    def __init__(self) -> None:
        load_architecture_mapping, _ = _get_architecture_functions()
        self.mapping = load_architecture_mapping()

    def resolve_role(self, module: str) -> str:
        """
        Resolve a module path to a role using longest-prefix match.
        """
        best_match = None
        for prefix in self.mapping:
            if module == prefix or module.startswith(prefix + "."):
                if best_match is None or len(prefix) > len(best_match):
                    best_match = prefix

        if best_match is None:
            return "unknown"

        return self.mapping[best_match]

    def is_allowed(self, importer_role: str, imported_role: str) -> bool:
        if imported_role == "unknown" or importer_role == "unknown":
            return True
        _, allowed_importers = _get_architecture_functions()
        return importer_role in allowed_importers(imported_role)
