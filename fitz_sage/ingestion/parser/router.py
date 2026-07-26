# fitz_sage/ingestion/parser/router.py
"""
ParserRouter - Routes files to appropriate parsers based on extension.

Architecture:
    ┌─────────────────────────────────────┐
    │           ParserRouter              │
    │  Routes files to appropriate parser │
    └─────────────────────────────────────┘
                    │
       ┌────────────┼────────────┐
       │            │            │
       ▼            ▼            ▼
  PlainTextParser DoclingParser  (future)
   (.txt, .md)    (.pdf, .docx)

Usage:
    router = ParserRouter()
    parsed_doc = router.parse(source_file)

VLM Integration:
    Use docling_vision parser for VLM-powered figure description:

    router = ParserRouter(parser="docling_vision")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from fitz_sage.core.document import ParsedDocument
from fitz_sage.ingestion.parser.base import ParseError, Parser
from fitz_sage.ingestion.source.base import SourceFile

logger = logging.getLogger(__name__)


@dataclass
class ParserRouter:
    """
    Routes files to appropriate parsers based on extension.

    Priority order:
    1. Extension-specific parser (if registered)
    2. Default fallback parser (PlainTextParser)

    Example:
        router = ParserRouter()
        doc = router.parse(source_file)

    With VLM for figure description (via docling_vision parser):
        router = ParserRouter(parser="docling_vision")
    """

    # Document parser mode: "cpu" (default, zero-model pypdfium2 PDF),
    # "docling", "docling_vision", or "glm_ocr".
    parser: str = field(default="cpu")

    # Map of extension -> parser instance
    _parsers: Dict[str, Parser] = field(default_factory=dict)
    _default_parser: Optional[Parser] = field(default=None)
    _warned_extensions: Set[str] = field(default_factory=set)
    _initialized: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize with default parsers."""
        if not self._initialized:
            self._initialize_default_parsers()
            self._initialized = True

    def _initialize_default_parsers(self) -> None:
        """Set up default parsers."""
        from fitz_sage.ingestion.parser.plugins.plaintext import (
            PLAINTEXT_EXTENSIONS,
            PlainTextParser,
        )

        # Default parser for text files
        plaintext = PlainTextParser()
        self._default_parser = plaintext

        # Register plaintext parser for its extensions
        for ext in PLAINTEXT_EXTENSIONS:
            self._parsers[ext] = plaintext

        # glm_ocr owns every rich-document extension when selected.
        if self.parser == "glm_ocr":
            try:
                from fitz_sage.ingestion.parser.plugins.glm_ocr import (
                    GLM_OCR_EXTENSIONS,
                    GlmOcrParser,
                )

                glm = GlmOcrParser()
                for ext in GLM_OCR_EXTENSIONS:
                    self._parsers[ext] = glm
                logger.info("Using glm_ocr parser (hybrid pypdfium2 + GLM-OCR)")
                return
            except ImportError:
                logger.warning("GLM-OCR parser import failed, falling back to defaults")

        # Docling handles rich documents (DOCX/PPTX/HTML) — and PDFs too when
        # it is the selected parser. Optional: pip install fitz-sage[docs].
        docling: Parser
        try:
            if self.parser == "docling_vision":
                from fitz_sage.ingestion.parser.plugins.docling_vision import (
                    DOCLING_EXTENSIONS,
                    DoclingVisionParser,
                )

                docling = DoclingVisionParser()
                logger.info("Using docling_vision parser (VLM-powered figure description)")
            else:
                from fitz_sage.ingestion.parser.plugins.docling import (
                    DOCLING_EXTENSIONS,
                    DoclingParser,
                )

                docling = DoclingParser()

            for ext in DOCLING_EXTENSIONS:
                self._parsers[ext] = docling
        except ImportError:
            # Docling not installed — register lightweight parsers instead
            logger.info("Docling not installed — using lightweight parsers for PDF/DOCX/PPTX")
            from fitz_sage.ingestion.parser.plugins.lightweight import (
                LightweightDOCXParser,
                LightweightPDFParser,
                LightweightPPTXParser,
            )

            self._parsers[".pdf"] = LightweightPDFParser()
            self._parsers[".docx"] = LightweightDOCXParser()
            self._parsers[".pptx"] = LightweightPPTXParser()

        # CPU-first mode (the default): the zero-model pypdfium2 parser owns
        # PDF — server-free, no torch, no OCR. docling stays registered for
        # DOCX/PPTX/HTML, where it has no memory problem (and loads lazily, so
        # a PDF-only corpus never touches it). An explicit `parser: docling*`
        # keeps docling on PDF instead.
        if self.parser not in ("docling", "docling_vision"):
            from fitz_sage.ingestion.parser.plugins.cpu_pdf import CpuPdfParser

            self._parsers[".pdf"] = CpuPdfParser()
            logger.info("Using cpu_pdf parser for PDF (server-free, zero-model)")

    def register_parser(self, parser: Parser, extensions: Optional[List[str]] = None) -> None:
        """
        Register a parser for specific extensions.

        Args:
            parser: Parser instance to register.
            extensions: List of extensions to register for.
                       If None, uses parser.supported_extensions.
        """
        exts = extensions or list(getattr(parser, "supported_extensions", []))
        for ext in exts:
            normalized = ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            self._parsers[normalized] = parser
            logger.debug(f"Registered parser '{parser.plugin_name}' for {normalized}")

    def get_parser(self, ext: str) -> Parser:
        """
        Get the appropriate parser for a file extension.

        Args:
            ext: File extension (e.g., ".md", ".pdf", "txt").

        Returns:
            Parser instance for the extension.
        """
        normalized = ext.lower() if ext.startswith(".") else f".{ext.lower()}"

        parser = self._parsers.get(normalized)
        if parser is not None:
            return parser

        if self._default_parser is None:
            raise ParseError("Parser router has no default parser", normalized)

        # Fall back to default
        if normalized not in self._warned_extensions:
            self._warned_extensions.add(normalized)
            logger.debug(
                f"No parser registered for '{normalized}', "
                f"using default '{self._default_parser.plugin_name}'"
            )

        return self._default_parser

    def can_parse(self, file: SourceFile) -> bool:
        """
        Check if any parser can handle the file.

        Args:
            file: SourceFile to check.

        Returns:
            True if a parser can handle the file.
        """
        parser = self.get_parser(file.extension)
        return parser.can_parse(file)

    def parse(self, file: SourceFile) -> ParsedDocument:
        """
        Parse a file using the appropriate parser.

        Args:
            file: SourceFile to parse.

        Returns:
            ParsedDocument with structured elements.

        Raises:
            ParseError: If parsing fails.
        """
        parser = self.get_parser(file.extension)
        logger.debug(f"Parsing '{file.name}' with {parser.plugin_name} parser")

        try:
            return parser.parse(file)
        except ParseError:
            raise
        except Exception as e:
            raise ParseError(
                f"Parser '{parser.plugin_name}' failed: {e}",
                source=file.uri,
                cause=e,
            ) from e

    def get_parser_id(self, ext: str) -> str:
        """
        Get a unique ID for the parser handling an extension.

        Args:
            ext: File extension.

        Returns:
            Parser ID string (e.g., "plaintext:v1", "docling:v1").
        """
        parser = self.get_parser(ext)
        return f"{parser.plugin_name}:v1"

    @property
    def registered_extensions(self) -> List[str]:
        """Get list of extensions with registered parsers."""
        return sorted(self._parsers.keys())

    def __repr__(self) -> str:
        ext_list = ", ".join(self.registered_extensions[:5])
        if len(self.registered_extensions) > 5:
            ext_list += f", ... ({len(self.registered_extensions)} total)"
        return f"ParserRouter(extensions=[{ext_list}])"


__all__ = ["ParserRouter"]
