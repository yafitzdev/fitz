# fitz/core/_contracts/rules.py
from __future__ import annotations

from typing import Dict, FrozenSet

from fitz_ai.engines.classic_rag.contracts.roles import ROLES, Role


def role_by_name() -> Dict[str, Role]:
    """
    Return a name -> Role mapping for all known roles.

    Single source of truth derived from ROLES.
    """
    return {role.name: role for role in ROLES}


def allowed_importers(role_name: str) -> FrozenSet[str]:
    """
    Return the set of roles that are allowed to import the given role.
    """
    roles = role_by_name()
    try:
        return roles[role_name].may_be_imported_by
    except KeyError as e:
        raise KeyError(f"Unknown role: {role_name!r}") from e
