# fitz_ai/runtime/__init__.py
"""
Fitz Runtime - Universal engine orchestration.

This package provides the top-level API for running queries across any engine.
It handles engine discovery, registration, and routing.

Public API:
    - run: Universal entry point for all engines
    - create_engine: Factory for creating engine instances
    - list_engines: Get list of available engines
    - get_engine_registry: Access the global registry
    - EngineCapabilities: Declare engine capabilities

Examples:
    Simple query:
    >>> from fitz_ai.runtime import run
    >>> answer = run("What is quantum computing?")

    Specific engine:
    >>> answer = run("Explain X", engine="clara")

    List engines:
    >>> from fitz_ai.runtime import list_engines
    >>> print(list_engines())
    ['classic_rag', 'clara']

    Create reusable engine:
    >>> from fitz_ai.runtime import create_engine
    >>> engine = create_engine("classic_rag")
    >>> answer = engine.answer(query)

    Check capabilities:
    >>> from fitz_ai.runtime import get_engine_registry
    >>> caps = get_engine_registry().get_capabilities("clara")
    >>> if caps.requires_documents_at_query:
    ...     print("This engine needs documents loaded first")
"""

from .registry import (
    EngineCapabilities,
    EngineRegistration,
    EngineRegistry,
    get_engine_registry,
)
from .runner import (
    create_engine,
    list_engines,
    list_engines_with_info,
    run,
)

__all__ = [
    # Registry
    "EngineRegistry",
    "EngineRegistration",
    "EngineCapabilities",
    "get_engine_registry",
    # Runner
    "run",
    "create_engine",
    "list_engines",
    "list_engines_with_info",
]


# =============================================================================
# ENGINE AUTO-DISCOVERY
# =============================================================================
# Automatically discover and register engines from fitz_ai/engines/*/
# Each engine module that wants to be discovered should have a registration
# function that runs on import.


def _discover_engines():
    """
    Auto-discover engines from the engines directory.

    This scans fitz_ai/engines/*/ and imports any module that:
    1. Has a runtime.py or __init__.py with registration code
    2. Can be successfully imported (dependencies available)

    Engines register themselves when imported via their _register_*_engine() functions.
    """
    import importlib
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)

    # Find engines directory
    engines_dir = Path(__file__).parent.parent / "engines"
    if not engines_dir.exists():
        return

    # Scan for engine subdirectories
    for engine_dir in engines_dir.iterdir():
        if not engine_dir.is_dir():
            continue
        if engine_dir.name.startswith("_"):
            continue

        engine_name = engine_dir.name
        module_name = f"fitz_ai.engines.{engine_name}"

        # Try to import the engine module
        # First try runtime.py (preferred for registration)
        # Then fall back to __init__.py
        for submodule in ["runtime", ""]:
            full_module = f"{module_name}.{submodule}" if submodule else module_name
            try:
                importlib.import_module(full_module)
                logger.debug(f"Discovered engine: {engine_name} (via {full_module})")
                break  # Successfully imported, move to next engine
            except ImportError as e:
                # Dependencies not available - skip silently
                logger.debug(f"Could not load {full_module}: {e}")
                continue
            except Exception as e:
                # Other error - log but continue
                logger.warning(f"Error loading engine {engine_name}: {e}")
                continue


# Run discovery on import
_discover_engines()
