# core/vector_db/registry.py
from __future__ import annotations

from typing import TYPE_CHECKING, Type

from fitz.core.llm.registry import get_llm_plugin

if TYPE_CHECKING:
    from fitz.core.vector_db.base import VectorDBPlugin


def get_vector_db_plugin(plugin_name: str) -> Type["VectorDBPlugin"]:
    return get_llm_plugin(plugin_name=plugin_name, plugin_type="vector_db")
