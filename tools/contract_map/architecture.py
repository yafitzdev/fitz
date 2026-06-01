# tools/contract_map/architecture.py
from __future__ import annotations

from dataclasses import dataclass

from .common import PKG


@dataclass(frozen=True)
class RoleRule:
    """Import rule for one architectural role."""

    name: str
    may_import: frozenset[str] | None
    description: str


def _package(name: str) -> str:
    return f"{PKG.name}.{name}"


ROLE_MAPPING: dict[str, str] = {
    role: role
    for role in (
        "api",
        "cli",
        "config",
        "core",
        "encoders",
        "engines",
        "governance",
        "ingestion",
        "llm",
        "retrieval",
        "runtime",
        "sdk",
        "services",
        "storage",
        "tabular",
        "tools",
    )
}
ROLE_MAPPING.update({_package(role): role for role in tuple(ROLE_MAPPING) if role != "tools"})


ROLE_RULES: dict[str, RoleRule] = {
    "core": RoleRule(
        name="core",
        may_import=frozenset({"core"}),
        description="Paradigm-agnostic contracts and data models.",
    ),
    "retrieval": RoleRule(
        name="retrieval",
        may_import=frozenset({"core", "retrieval", "storage"}),
        description="Shared retrieval intelligence and supporting stores.",
    ),
    "llm": RoleRule(
        name="llm",
        may_import=frozenset({"core", "encoders", "llm"}),
        description="OpenAI-compatible chat and reranker providers.",
    ),
    "storage": RoleRule(
        name="storage",
        may_import=frozenset({"core", "storage"}),
        description="SQLite connection and persistence infrastructure.",
    ),
    "ingestion": RoleRule(
        name="ingestion",
        may_import=frozenset({"core", "ingestion"}),
        description="Parser and chunking plugins.",
    ),
    "encoders": RoleRule(
        name="encoders",
        may_import=frozenset({"encoders"}),
        description="Local model encoder helpers.",
    ),
    "governance": RoleRule(
        name="governance",
        may_import=frozenset({"core", "encoders", "governance"}),
        description="Epistemic classifier layer.",
    ),
    "tabular": RoleRule(
        name="tabular",
        may_import=frozenset({"core", "llm", "storage", "tabular"}),
        description="CSV and table query implementation.",
    ),
    "engines": RoleRule(
        name="engines",
        may_import=frozenset(
            {
                "core",
                "config",
                "engines",
                "governance",
                "ingestion",
                "llm",
                "retrieval",
                "storage",
                "tabular",
            }
        ),
        description="Engine implementations that compose shared layers.",
    ),
    "config": RoleRule(
        name="config",
        may_import=frozenset({"config", "core"}),
        description="Layered configuration loading.",
    ),
    "api": RoleRule(
        name="api",
        may_import=None,
        description="Application API surface; may compose runtime layers.",
    ),
    "cli": RoleRule(
        name="cli",
        may_import=None,
        description="Command-line surface; may compose runtime layers.",
    ),
    "runtime": RoleRule(
        name="runtime",
        may_import=None,
        description="Engine orchestration and registry.",
    ),
    "sdk": RoleRule(
        name="sdk",
        may_import=None,
        description="User-facing Python SDK facade.",
    ),
    "services": RoleRule(
        name="services",
        may_import=None,
        description="Application service layer.",
    ),
    "tools": RoleRule(
        name="tools",
        may_import=None,
        description="Developer tooling and repo analysis scripts.",
    ),
}


def load_architecture_mapping() -> dict[str, str]:
    """Return package-prefix to role mappings for contract-map validation."""
    return dict(ROLE_MAPPING)


class RoleResolver:
    def __init__(self) -> None:
        self.mapping = load_architecture_mapping()

    def resolve_role(self, module: str) -> str:
        """
        Resolve a module path to a role using longest-prefix match.
        """
        if module in ROLE_RULES:
            return module

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
        if importer_role == imported_role:
            return True

        rule = ROLE_RULES.get(importer_role)
        if rule is None or rule.may_import is None:
            return True

        return imported_role in rule.may_import
