# fitz_sage/ingestion/parser/plugins/docling.py
"""
Docling-based parser for PDF, DOCX, images, and more.

Uses IBM's Docling library for advanced document understanding including:
- Layout analysis and reading order
- Table structure extraction
- Figure/image detection with optional VLM description
- Code block recognition
- Formula handling

VLM Integration:
    When a vision client is provided, figures/images detected by Docling are
    sent to a Vision Language Model for description. This replaces the default
    "[Figure]" placeholder with an actual description of the image content.

    Configure vision in fitz.yaml:
        vision:
          enabled: true
          plugin_name: openai  # or "anthropic", "ollama"

Requires: pip install docling
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Set

# Fix Windows symlink issue with Hugging Face model caching
# Windows restricts symlink creation by default, causing model downloads to fail
# Setting this env var before importing docling/huggingface_hub fixes the issue
if sys.platform == "win32":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    # This actually disables symlinks (not just the warning)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

from fitz_sage.core.document import DocumentElement, ElementType, ParsedDocument
from fitz_sage.ingestion.parser.base import ParseError
from fitz_sage.ingestion.source.base import SourceFile

from .base_parser import BaseParser

logger = logging.getLogger(__name__)

# Silence verbose third-party loggers
for _logger_name in [
    "docling",
    "docling.document_converter",
    "docling.pipeline",
    "docling_core",
    "rapidocr",
    "RapidOCR",
    "httpx",
    "httpcore",
]:
    logging.getLogger(_logger_name).setLevel(logging.WARNING)

# Supported file extensions
DOCLING_EXTENSIONS: Set[str] = {
    # Documents
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
    # Images
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".bmp",
    ".webp",
}


@dataclass
class DoclingParser(BaseParser):
    """
    Parser using Docling for document understanding.

    Docling provides advanced PDF/document parsing with:
    - Vision-based layout analysis
    - Table structure detection
    - Reading order inference
    - Multi-format support
    - VLM-powered figure/image description (optional)

    Example:
        parser = DoclingParser()
        doc = parser.parse(source_file)
        for element in doc.elements:
            print(element.type, element.content[:50])

    With VLM for figure description:
        from fitz_sage.llm import get_vision
        vision_client = get_vision("openai")
        parser = DoclingParser(vision_client=vision_client)
        doc = parser.parse(source_file)  # Figures will have descriptions!
    """

    plugin_name: str = field(default="docling")
    supported_extensions: Set[str] = field(default_factory=lambda: DOCLING_EXTENSIONS)

    # Optional VLM client for describing figures/images
    vision_client: Any = field(default=None, repr=False)

    # Lazy-loaded converter
    _converter: object = field(default=None, repr=False)

    # Track VLM statistics
    _vlm_calls: int = field(default=0, repr=False)
    _vlm_errors: int = field(default=0, repr=False)

    def _get_converter(self):
        """Lazy-load the DocumentConverter."""
        if self._converter is None:
            try:
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.document_converter import DocumentConverter, PdfFormatOption

                # Enable picture image extraction if VLM is configured
                if self.vision_client is not None:
                    pipeline_options = PdfPipelineOptions()
                    pipeline_options.generate_picture_images = True
                    self._converter = DocumentConverter(
                        format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
                    )
                else:
                    self._converter = DocumentConverter()
            except ImportError as e:
                raise ImportError(
                    "Docling is required for this parser. Install it with: pip install docling"
                ) from e
        return self._converter

    def parse(self, file: SourceFile) -> ParsedDocument:
        """
        Parse a file into structured content using Docling.

        Args:
            file: SourceFile with local_path for reading.

        Returns:
            ParsedDocument with structured elements.

        Raises:
            ParseError: If parsing fails.
        """
        if not self.can_parse(file):
            raise ParseError(
                f"Unsupported file type: {file.extension}",
                source=file.uri,
            )

        try:
            return self._parse_with_docling(file)
        except ImportError:
            raise
        except Exception as e:
            raise ParseError(
                f"Failed to parse document: {e}",
                source=file.uri,
                cause=e,
            ) from e

    def _parse_with_docling(self, file: SourceFile) -> ParsedDocument:
        """Internal parsing using Docling."""
        converter = self._get_converter()

        # Convert the document
        logger.info(f"Parsing with Docling: {file.local_path}")
        result = converter.convert(str(file.local_path))
        doc = result.document

        # Extract elements
        elements: list[DocumentElement] = []

        for item, level in doc.iterate_items():
            element = self._convert_item(item, level, doc)
            if element:
                elements.append(element)

        # Build metadata
        metadata = {
            "parser": self.plugin_name,
            "source_extension": file.extension,
        }

        # Add page count if available
        if hasattr(doc, "pages") and doc.pages:
            metadata["page_count"] = len(doc.pages)

        # Add VLM statistics if vision was used
        if self.vision_client is not None:
            metadata["vlm_enabled"] = True
            metadata["vlm_calls"] = self._vlm_calls
            metadata["vlm_errors"] = self._vlm_errors
            if self._vlm_calls > 0:
                logger.info(
                    f"VLM processed {self._vlm_calls} figures "
                    f"({self._vlm_errors} errors) in {file.uri}"
                )

        return ParsedDocument(
            source=file.uri,
            elements=elements,
            metadata=metadata,
        )

    def _convert_item(
        self,
        item,
        level: int,
        doc,
    ) -> DocumentElement | None:
        """Convert a Docling item to DocumentElement."""
        from docling_core.types.doc.labels import DocItemLabel

        label = getattr(item, "label", None)
        if label is None:
            return None

        # Map Docling labels to our ElementType
        if label == DocItemLabel.SECTION_HEADER:
            text = self._get_text(item)
            if not text:
                return None
            heading_level = getattr(item, "level", level) or 1
            return DocumentElement(
                type=ElementType.HEADING,
                content=text,
                level=min(heading_level, 6),  # Cap at h6
            )

        elif label == DocItemLabel.TITLE:
            text = self._get_text(item)
            if not text:
                return None
            return DocumentElement(
                type=ElementType.HEADING,
                content=text,
                level=1,  # Title is always h1
            )

        elif label in (DocItemLabel.TEXT, DocItemLabel.PARAGRAPH):
            text = self._get_text(item)
            if not text:
                return None
            return DocumentElement(
                type=ElementType.TEXT,
                content=text,
            )

        elif label == DocItemLabel.LIST_ITEM:
            text = self._get_text(item)
            if not text:
                return None
            return DocumentElement(
                type=ElementType.LIST_ITEM,
                content=text,
                level=level,
            )

        elif label == DocItemLabel.TABLE:
            # Build clean table from structured grid data (avoids markdown formatting)
            try:
                table_md = self._build_table_from_grid(item)
                if table_md:
                    return DocumentElement(
                        type=ElementType.TABLE,
                        content=table_md,
                        metadata={"rows": len(item.data.grid) if item.data else 0},
                    )
                # Fallback to export_to_markdown if grid extraction fails
                table_md = item.export_to_markdown(doc=doc)
                return DocumentElement(
                    type=ElementType.TABLE,
                    content=table_md,
                    metadata={"rows": len(item.data.grid) if item.data else 0},
                )
            except Exception as e:
                logger.warning(f"Failed to export table: {e}")
                return None

        elif label == DocItemLabel.PICTURE:
            # For pictures, try VLM description first, then caption, then placeholder
            caption = ""
            if hasattr(item, "captions") and item.captions:
                # Try to get caption text
                for cap_ref in item.captions:
                    if hasattr(cap_ref, "text"):
                        caption = cap_ref.text
                        break

            # Try VLM description if available (pass doc for image extraction)
            vlm_description = self._describe_image_with_vlm(item, doc)

            # Build content: VLM description > caption > placeholder
            if vlm_description:
                # Include caption as context if available
                if caption:
                    content = f"{vlm_description}\n\nCaption: {caption}"
                else:
                    content = vlm_description
            else:
                content = caption or "[Figure]"

            return DocumentElement(
                type=ElementType.FIGURE,
                content=content,
                metadata={
                    "has_image": item.image is not None if hasattr(item, "image") else False,
                    "vlm_described": vlm_description is not None,
                },
            )

        elif label == DocItemLabel.CODE:
            text = self._get_text(item)
            if not text:
                return None
            return DocumentElement(
                type=ElementType.CODE_BLOCK,
                content=text,
                language=None,  # Docling doesn't detect language
            )

        elif label == DocItemLabel.FORMULA:
            text = self._get_text(item)
            if not text:
                return None
            return DocumentElement(
                type=ElementType.TEXT,
                content=text,
                metadata={"is_formula": True},
            )

        elif label in (DocItemLabel.FOOTNOTE, DocItemLabel.REFERENCE):
            text = self._get_text(item)
            if not text:
                return None
            return DocumentElement(
                type=ElementType.TEXT,
                content=text,
                metadata={"is_footnote": label == DocItemLabel.FOOTNOTE},
            )

        elif label == DocItemLabel.CAPTION:
            text = self._get_text(item)
            if not text:
                return None
            return DocumentElement(
                type=ElementType.TEXT,
                content=text,
                metadata={"is_caption": True},
            )

        elif label in (DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER):
            # Skip headers/footers by default (furniture)
            return None

        else:
            # Unknown label - try to extract text anyway
            text = self._get_text(item)
            if text:
                return DocumentElement(
                    type=ElementType.TEXT,
                    content=text,
                    metadata={"original_label": str(label)},
                )
            return None

    def _get_text(self, item) -> str:
        """Extract text from a Docling item."""
        # TextItem has 'text' attribute
        if hasattr(item, "text") and item.text:
            return item.text.strip()
        # Some items have 'orig' for original text
        if hasattr(item, "orig") and item.orig:
            return item.orig.strip()
        return ""

    def _build_table_from_grid(self, item) -> str | None:
        """
        Build clean markdown table from Docling's structured grid data.

        This bypasses export_to_markdown() which adds unwanted markdown formatting
        (bold headers, etc.) that breaks downstream SQL generation.

        Args:
            item: Docling table item with item.data.grid structured data.

        Returns:
            Clean markdown table string, or None if grid is empty/invalid.
        """
        if not item.data or not item.data.grid:
            return None

        grid = item.data.grid
        if not grid or len(grid) < 1:
            return None

        rows: list[str] = []
        num_cols = 0

        for row in grid:
            # Extract text from each cell
            cells: list[str] = []
            for cell in row:
                # Docling cells have 'text' attribute
                if hasattr(cell, "text"):
                    cells.append(cell.text.strip() if cell.text else "")
                else:
                    # Fallback for unexpected cell types
                    cells.append(str(cell).strip() if cell else "")

            if cells:
                num_cols = max(num_cols, len(cells))
                rows.append("| " + " | ".join(cells) + " |")

        if not rows:
            return None

        # Ensure all rows have same number of columns (pad if needed)
        normalized_rows: list[str] = []
        for row in rows:
            # Count columns in this row
            cols_in_row = row.count("|") - 1
            if cols_in_row < num_cols:
                # Pad with empty cells
                padding = " |" * (num_cols - cols_in_row)
                row = row[:-1] + padding + "|"
            normalized_rows.append(row)

        # Build separator row after header
        separator = "|" + "|".join(["---"] * num_cols) + "|"

        # Combine: header, separator, data rows
        return "\n".join([normalized_rows[0], separator] + normalized_rows[1:])

    def _describe_image_with_vlm(self, item, doc) -> str | None:
        """
        Use VLM to describe an image from a Docling picture item.

        Args:
            item: Docling picture item with potential image data.
            doc: Docling document (needed to extract image via get_image).

        Returns:
            Description string if successful, None otherwise.
        """
        if self.vision_client is None:
            return None

        try:
            # Primary method: use item.get_image(doc) which returns PIL Image
            pil_image = None
            if hasattr(item, "get_image"):
                pil_image = item.get_image(doc)

            # Fallback: try item.image attribute
            if pil_image is None:
                image_data = getattr(item, "image", None)
                if image_data is not None:
                    if hasattr(image_data, "pil_image"):
                        pil_image = image_data.pil_image
                    elif hasattr(image_data, "save"):
                        pil_image = image_data

            if pil_image is None:
                logger.debug("No image data available for VLM description")
                return None

            # Convert PIL image to base64 PNG
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

            # Call vision client
            self._vlm_calls += 1
            description = self.vision_client.describe_image(image_base64)

            if description:
                logger.debug(f"VLM described image: {description[:100]}...")
                return description

        except Exception as e:
            self._vlm_errors += 1
            logger.warning(f"VLM failed to describe image: {e}")

        return None


__all__ = ["DoclingParser", "DOCLING_EXTENSIONS"]
