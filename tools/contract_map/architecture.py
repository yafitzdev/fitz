from __future__ import annotations

from fitz.engines.classic_rag.contracts.rules import allowed_importers
from fitz.engines.classic_rag.config.architecture import load_architecture_mapping


class RoleResolver:
    def __init__(self) -> None:
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
        return importer_role in allowed_importers(imported_role)
