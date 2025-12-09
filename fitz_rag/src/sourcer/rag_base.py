# src/fitz_rag/sourcer/rag_base.py
"""
Core plugin and retrieval system for fitz-rag.

Defines:
- ArtefactRetrievalStrategy (base)
- SourceConfig (plugin description)
- RAGContextBuilder (retrieval orchestrator)
- load_source_configs() (dynamic plugin loader)

Plugins are simple Python files living under:
    fitz_rag/sourcer/

Each plugin must define:
    SOURCE_CONFIG = SourceConfig(...)
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

from fitz_rag.src.core.types import RetrievedChunk


# ---------------------------------------------------------
# Base Strategy Interface
# ---------------------------------------------------------

class ArtefactRetrievalStrategy:
    """
    Base class for all retrieval strategies.

    A strategy accepts:
        trf (any structured input)
        query (string)

    It returns:
        list[RetrievedChunk]
    """

    def retrieve(self, trf: Dict, query: str) -> List[RetrievedChunk]:
        raise NotImplementedError


# ---------------------------------------------------------
# SourceConfig — describes a plugin
# ---------------------------------------------------------

@dataclass
class SourceConfig:
    name: str
    order: int
    strategy: ArtefactRetrievalStrategy
    label: Optional[str] = None


# ---------------------------------------------------------
# RetrievalContext — final result structure
# ---------------------------------------------------------

@dataclass
class RetrievalContext:
    query: str
    artefacts: Dict[str, List[RetrievedChunk]]


# ---------------------------------------------------------
# RAGContextBuilder — orchestrates retrieval
# ---------------------------------------------------------

class RAGContextBuilder:
    """
    Given:
        - list of SourceConfig
        - TRF-like dict
    Produces:
        RetrievalContext(query, artefacts)
    """

    def __init__(self, sources: List[SourceConfig]) -> None:
        self.sources = sources

    def retrieve_for(self, trf: Dict, query: str) -> RetrievalContext:
        artefacts: Dict[str, List[RetrievedChunk]] = {}

        for src in self.sources:
            try:
                artefacts[src.name] = src.strategy.retrieve(trf, query)
            except Exception:
                artefacts[src.name] = []

        return RetrievalContext(query=query, artefacts=artefacts)


# ---------------------------------------------------------
# Dynamic Plugin Loader
# ---------------------------------------------------------

def load_source_configs() -> List[SourceConfig]:
    """
    Load all plugins in fitz_rag.sourcer.* that define SOURCE_CONFIG.
    """

    pkg_dir = Path(__file__).parent
    base_pkg = __name__.rsplit(".", 1)[0]

    configs: List[SourceConfig] = []

    for info in pkgutil.iter_modules([str(pkg_dir)]):
        name = info.name

        # Skip internal modules
        if name in ("rag_base", "prompt_builder", "__init__"):
            continue

        full_name = f"{base_pkg}.{name}"

        try:
            module = importlib.import_module(full_name)
        except Exception:
            continue

        cfg = getattr(module, "SOURCE_CONFIG", None)

        if isinstance(cfg, SourceConfig):
            configs.append(cfg)

    # Sort according to ordering
    configs.sort(key=lambda c: c.order)
    return configs
