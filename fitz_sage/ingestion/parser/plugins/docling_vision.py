# fitz_sage/ingestion/parser/plugins/docling_vision.py
"""
Docling parser with VLM-powered figure description.

This parser extends DoclingParser to use a VLM (Vision Language Model) for
describing figures and images in documents. The KRAG ingestion pipeline owns
provider configuration and injects the vision client before parsing.

Use this parser when you want AI-generated descriptions of charts, graphs,
diagrams, and other visual content in PDFs and documents.

Usage:
    parser: docling_vision
    vision: endpoint/qwen2-vl-7b
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set

from fitz_sage.ingestion.parser.plugins.docling import DOCLING_EXTENSIONS, DoclingParser


@dataclass
class DoclingVisionParser(DoclingParser):
    """
    Docling parser with automatic VLM integration for figure description.

    Figures and images detected by Docling are sent to the injected VLM for
    description. Without an injected client, the base parser preserves its
    normal caption or placeholder behavior.
    """

    plugin_name: str = field(default="docling_vision", repr=False)
    supported_extensions: Set[str] = field(default_factory=lambda: DOCLING_EXTENSIONS)


__all__ = ["DoclingVisionParser", "DOCLING_EXTENSIONS"]
