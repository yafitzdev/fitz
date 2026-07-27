# fitz_sage/engines/fitz_krag/progressive/builder.py
"""
ManifestBuilder — fast directory scan with AST symbol + heading extraction.

No LLM calls. Runs in <500ms for 100 files.
"""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fitz_sage.engines.fitz_krag.ingestion.formats import (
    CODE_EXTENSION_MAP,
    DOCUMENT_EXTENSIONS,
    RICH_DOCUMENT_EXTENSIONS,
    enabled_extensions,
)
from fitz_sage.engines.fitz_krag.progressive.manifest import (
    FileManifest,
    FileState,
    ManifestEntry,
    ManifestHeading,
    ManifestSymbol,
)
from fitz_sage.ingestion.hashing import compute_bytes_hash

if TYPE_CHECKING:
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig

logger = logging.getLogger(__name__)

# Directories to skip
_SKIP_DIRS = {
    ".fitz",
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".tox",
    ".eggs",
}

# Heading regex for markdown
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class ManifestBuilder:
    """Builds a FileManifest from a source directory using fast extraction."""

    def __init__(self, config: "FitzKragConfig") -> None:
        self._config = config

    def build(
        self,
        source: Path,
        manifest_path: Path,
        *,
        progress: Any = None,
    ) -> FileManifest:
        """Scan directory, extract symbols/headings, create manifest.

        Reuses:
        - PythonCodeIngestStrategy().extract() for .py (stdlib ast, ~50ms/file)
        - _extract_headings() for .md/.rst/.txt (regex, instant)
        - parsed_cache for rich docs (PDF, DOCX) — parse once, cache forever

        No LLM calls.
        """
        from fitz_sage.engines.fitz_krag.progressive.parsed_cache import (
            get_parsed_text,
        )

        manifest = FileManifest(manifest_path)
        existing = manifest.entries()

        # Clear old entries — each point() replaces the manifest completely
        manifest.clear()

        # When source is a single file, use its parent as the base directory
        base_dir = source.parent if source.is_file() else source

        # Parsed text cache lives next to the manifest
        cache_dir = manifest_path.parent / "parsed"

        _progress = progress or (lambda _: None)

        file_paths, unsupported_paths = self._scan_files(source)
        for abs_path in unsupported_paths:
            rel_path = self._relative_path(abs_path, base_dir)
            existing_entry = existing.get(rel_path)
            manifest.add(
                ManifestEntry(
                    file_id=(
                        existing_entry.file_id if existing_entry is not None else str(uuid.uuid4())
                    ),
                    rel_path=rel_path,
                    abs_path=str(abs_path),
                    content_hash="",
                    file_type=abs_path.suffix.lower(),
                    size_bytes=_file_size(abs_path),
                    state=FileState.UNSUPPORTED,
                )
            )

        for abs_path in file_paths:
            rel_path = self._relative_path(abs_path, base_dir)
            ext = abs_path.suffix.lower()

            # Rich docs (PDF, DOCX, etc.) — hash raw bytes, parse + cache text
            if ext in RICH_DOCUMENT_EXTENSIONS:
                try:
                    raw_bytes = abs_path.read_bytes()
                except Exception as e:
                    logger.warning(f"Cannot read {abs_path}: {e}")
                    existing_entry = existing.get(rel_path)
                    manifest.add(
                        _failed_entry(
                            abs_path,
                            rel_path,
                            existing_entry,
                            stage="read",
                            message=str(e),
                        )
                    )
                    continue

                content_hash = compute_bytes_hash(raw_bytes)

                # Ensure parsed text is cached (cheap if already cached)
                cache_file = cache_dir / f"{content_hash}.txt"
                if not cache_file.exists():
                    _progress(f"Parsing {abs_path.name}...")
                    get_parsed_text(abs_path, content_hash, cache_dir)

                existing_entry = existing.get(rel_path)
                if (
                    existing_entry
                    and existing_entry.content_hash == content_hash
                    and existing_entry.state != FileState.FAILED
                ):
                    manifest.add(existing_entry)
                    continue

                # Read cached headings extracted during Docling parse
                from fitz_sage.engines.fitz_krag.progressive.parsed_cache import (
                    get_parsed_headings,
                )

                heading_dicts = get_parsed_headings(content_hash, cache_dir)
                rich_headings = [
                    ManifestHeading(title=str(h["title"]), level=int(str(h["level"])))
                    for h in heading_dicts
                ]

                file_id = existing_entry.file_id if existing_entry else str(uuid.uuid4())
                entry = ManifestEntry(
                    file_id=file_id,
                    rel_path=rel_path,
                    abs_path=str(abs_path),
                    content_hash=content_hash,
                    file_type=ext,
                    size_bytes=len(raw_bytes),
                    state=FileState.REGISTERED,
                    symbols=[],
                    headings=rich_headings,
                )
                manifest.add(entry)
                continue

            # Text-based files — read as text, extract symbols/headings
            try:
                content = abs_path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                logger.warning(f"Cannot read {abs_path}: {e}")
                existing_entry = existing.get(rel_path)
                manifest.add(
                    _failed_entry(
                        abs_path,
                        rel_path,
                        existing_entry,
                        stage="read",
                        message=str(e),
                    )
                )
                continue

            content_hash = compute_bytes_hash(content.encode())

            # Re-add unchanged files (manifest was cleared, so we must re-add)
            existing_entry = existing.get(rel_path)
            if (
                existing_entry
                and existing_entry.content_hash == content_hash
                and existing_entry.state != FileState.FAILED
            ):
                manifest.add(existing_entry)
                continue

            file_id = existing_entry.file_id if existing_entry else str(uuid.uuid4())

            # Extract symbols or headings
            symbols: list[ManifestSymbol] = []
            headings: list[ManifestHeading] = []

            if ext == ".py":
                symbols = self._extract_python_symbols(content, rel_path)
            elif ext in {".ts", ".tsx", ".js", ".jsx"}:
                symbols = self._extract_ts_symbols(content, rel_path)
            elif ext == ".java":
                symbols = self._extract_java_symbols(content, rel_path)
            elif ext == ".go":
                symbols = self._extract_go_symbols(content, rel_path)
            elif ext in DOCUMENT_EXTENSIONS:
                headings = self._extract_headings(content)

            entry = ManifestEntry(
                file_id=file_id,
                rel_path=rel_path,
                abs_path=str(abs_path),
                content_hash=content_hash,
                file_type=ext,
                size_bytes=len(content.encode()),
                state=FileState.REGISTERED,
                symbols=symbols,
                headings=headings,
            )
            manifest.add(entry)

        manifest.save()
        return manifest

    def _scan_files(self, source: Path) -> tuple[list[Path], list[Path]]:
        """Return enabled and unsupported files from the pointed source."""
        extensions = self._enabled_extensions()
        if source.is_file():
            return ([source], []) if source.suffix.lower() in extensions else ([], [source])

        discovered = sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and not any(part in _SKIP_DIRS for part in path.parts)
        )
        supported = [path for path in discovered if path.suffix.lower() in extensions]
        unsupported = [path for path in discovered if path.suffix.lower() not in extensions]
        return supported, unsupported

    def _enabled_extensions(self) -> frozenset[str]:
        code_languages = getattr(self._config, "code_languages", None)
        if not isinstance(code_languages, (list, tuple, set, frozenset)):
            code_languages = set(CODE_EXTENSION_MAP.values())
        table_extensions = getattr(self._config, "table_extensions", None)
        if not isinstance(table_extensions, (list, tuple, set, frozenset)):
            table_extensions = {".csv", ".tsv"}
        return enabled_extensions(
            code_languages=code_languages,
            table_extensions=table_extensions,
        )

    def _relative_path(self, abs_path: Path, source: Path) -> str:
        """Get relative path string with forward slashes."""
        try:
            return str(abs_path.relative_to(source)).replace("\\", "/")
        except ValueError:
            return str(abs_path).replace("\\", "/")

    def _extract_python_symbols(self, content: str, file_path: str) -> list[ManifestSymbol]:
        """Extract symbols from Python source using PythonCodeIngestStrategy."""
        try:
            from fitz_sage.engines.fitz_krag.ingestion.strategies.python_code import (
                PythonCodeIngestStrategy,
            )

            strategy = PythonCodeIngestStrategy()
            result = strategy.extract(content, file_path)

            return [
                ManifestSymbol(
                    name=sym.name,
                    qualified_name=sym.qualified_name,
                    kind=sym.kind,
                    signature=sym.signature,
                    start_line=sym.start_line,
                    end_line=sym.end_line,
                )
                for sym in result.symbols
            ]
        except Exception as e:
            logger.debug(f"Python symbol extraction failed for {file_path}: {e}")
            return []

    def _extract_ts_symbols(self, content: str, file_path: str) -> list[ManifestSymbol]:
        """Extract symbols from TypeScript/JavaScript using TypeScriptIngestStrategy."""
        try:
            from fitz_sage.engines.fitz_krag.ingestion.strategies.typescript import (
                TypeScriptIngestStrategy,
            )

            strategy = TypeScriptIngestStrategy()
            result = strategy.extract(content, file_path)

            return [
                ManifestSymbol(
                    name=sym.name,
                    qualified_name=sym.qualified_name,
                    kind=sym.kind,
                    signature=sym.signature,
                    start_line=sym.start_line,
                    end_line=sym.end_line,
                )
                for sym in result.symbols
            ]
        except Exception as e:
            logger.warning(f"TypeScript symbol extraction failed for {file_path}: {e}")
            return []

    def _extract_java_symbols(self, content: str, file_path: str) -> list[ManifestSymbol]:
        """Extract symbols from Java source using JavaIngestStrategy."""
        try:
            from fitz_sage.engines.fitz_krag.ingestion.strategies.java import (
                JavaIngestStrategy,
            )

            strategy = JavaIngestStrategy()
            result = strategy.extract(content, file_path)

            return [
                ManifestSymbol(
                    name=sym.name,
                    qualified_name=sym.qualified_name,
                    kind=sym.kind,
                    signature=sym.signature,
                    start_line=sym.start_line,
                    end_line=sym.end_line,
                )
                for sym in result.symbols
            ]
        except Exception as e:
            logger.warning(f"Java symbol extraction failed for {file_path}: {e}")
            return []

    def _extract_go_symbols(self, content: str, file_path: str) -> list[ManifestSymbol]:
        """Extract symbols from Go source using GoIngestStrategy."""
        try:
            from fitz_sage.engines.fitz_krag.ingestion.strategies.go import (
                GoIngestStrategy,
            )

            strategy = GoIngestStrategy()
            result = strategy.extract(content, file_path)

            return [
                ManifestSymbol(
                    name=sym.name,
                    qualified_name=sym.qualified_name,
                    kind=sym.kind,
                    signature=sym.signature,
                    start_line=sym.start_line,
                    end_line=sym.end_line,
                )
                for sym in result.symbols
            ]
        except Exception as e:
            logger.warning(f"Go symbol extraction failed for {file_path}: {e}")
            return []

    def _extract_headings(self, content: str) -> list[ManifestHeading]:
        """Extract headings from markdown/rst/text files using regex."""
        headings: list[ManifestHeading] = []
        for match in _MD_HEADING_RE.finditer(content):
            level = len(match.group(1))
            title = match.group(2).strip()
            headings.append(ManifestHeading(title=title, level=level))
        return headings


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _failed_entry(
    path: Path,
    rel_path: str,
    existing: ManifestEntry | None,
    *,
    stage: str,
    message: str,
) -> ManifestEntry:
    return ManifestEntry(
        file_id=existing.file_id if existing is not None else str(uuid.uuid4()),
        rel_path=rel_path,
        abs_path=str(path),
        content_hash="",
        file_type=path.suffix.lower(),
        size_bytes=_file_size(path),
        state=FileState.FAILED,
        failure_stage=stage,
        failure_message=message,
    )
