"""
fitz_rag package initializer.

This file explicitly exposes all public subpackages so that IDEs
(Python static analysis, PyCharm, VSCode) correctly resolve imports like:

    from fitz_rag.core import RetrievedChunk

It does NOT import any heavy modules at runtime.
It only defines the namespace for tooling and clean package structure.
"""

__all__ = [
    "core",
    "retriever",
    "llm",
    "chunker",
    "sourcer",
    "pipeline",
    "config",
]
