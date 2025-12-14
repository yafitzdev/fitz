# tools/contract_map/__init__.py
"""
Contract Map Generator - A modular tool for analyzing Python codebases.

This package provides utilities to extract and visualize:
- Models and protocols
- Plugin registries
- Import graphs and layering violations
- Configuration surfaces
- Code statistics and hotspots
"""

from .common import (
    ContractMap,
    ModelContract,
    ProtocolContract,
    RegistryContract,
    ImportGraph,
    HealthIssue,
    REPO_ROOT,
)
from .models import extract_models, extract_protocols
from .registries import extract_registries
from .imports import build_import_graph
from .discovery import scan_all_discoveries
from .analysis import (
    discover_entrypoints,
    compute_hotspots,
    compute_stats,
    compute_config_surface,
    compute_invariants,
)

__all__ = [
    "ContractMap",
    "ModelContract",
    "ProtocolContract",
    "RegistryContract",
    "ImportGraph",
    "HealthIssue",
    "REPO_ROOT",
    "extract_models",
    "extract_protocols",
    "extract_registries",
    "build_import_graph",
    "scan_all_discoveries",
    "discover_entrypoints",
    "compute_hotspots",
    "compute_stats",
    "compute_config_surface",
    "compute_invariants",
]

__version__ = "1.0.0"
