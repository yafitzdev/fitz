# fitz/core/_contracts/roles.py
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet


@dataclass(frozen=True)
class Role:
    """
    Abstract architectural role.

    Roles describe *conceptual responsibilities*, not concrete modules.

    - name: canonical role identifier
    - may_be_imported_by: roles that are allowed to depend on this role
    """

    name: str
    may_be_imported_by: FrozenSet[str]


# Canonical, abstract roles (stable concepts)
ROLES: FrozenSet[Role] = frozenset(
    {
        Role(
            name="foundation",
            may_be_imported_by=frozenset(
                {"domain", "runtime", "orchestration", "tooling", "tests"}
            ),
        ),
        Role(
            name="domain",
            may_be_imported_by=frozenset({"orchestration", "tooling", "tests"}),
        ),
        Role(
            name="runtime",
            may_be_imported_by=frozenset({"domain", "orchestration", "tests"}),
        ),
        Role(
            name="orchestration",
            may_be_imported_by=frozenset({"tooling", "tests"}),
        ),
        Role(
            name="tooling",
            may_be_imported_by=frozenset({"tests"}),
        ),
        Role(
            name="tests",
            may_be_imported_by=frozenset(),
        ),
    }
)
