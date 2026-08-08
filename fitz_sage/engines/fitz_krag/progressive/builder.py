# fitz_sage/engines/fitz_krag/progressive/builder.py
"""Fast source discovery and content hashing for the searchable index."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fitz_sage.engines.fitz_krag.ingestion.formats import (
    CODE_EXTENSION_MAP,
    enabled_extensions,
)
from fitz_sage.engines.fitz_krag.progressive.manifest import (
    EnrichmentState,
    FileManifest,
    FileState,
    ManifestEntry,
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


class ManifestBuilder:
    """Build a manifest without parsing or invoking a model."""

    def __init__(self, config: "FitzKragConfig") -> None:
        self._config = config

    def build(
        self,
        source: Path,
        manifest_path: Path,
        *,
        progress: Any = None,
    ) -> FileManifest:
        """Scan supported files, hash their bytes, and preserve unchanged state."""
        manifest = FileManifest(manifest_path)
        existing = manifest.entries()

        # Clear old entries — each point() replaces the manifest completely
        manifest.clear()

        # When source is a single file, use its parent as the base directory
        base_dir = source.parent if source.is_file() else source

        _progress = progress or (lambda _: None)

        file_paths, unsupported_paths = self._scan_files(source)
        discovered_supported: set[str] = set()
        existing_supported = {
            path for path, entry in existing.items() if entry.state != FileState.UNSUPPORTED
        }
        corpus_changed = False

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
                    enrichment_state=EnrichmentState.NOT_APPLICABLE,
                )
            )

        for abs_path in file_paths:
            rel_path = self._relative_path(abs_path, base_dir)
            discovered_supported.add(rel_path)
            ext = abs_path.suffix.lower()

            try:
                raw_bytes = abs_path.read_bytes()
            except Exception as e:
                logger.warning(f"Cannot read {abs_path}: {e}")
                corpus_changed = True
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

            existing_entry = existing.get(rel_path)
            if (
                existing_entry
                and existing_entry.content_hash == content_hash
                and existing_entry.state != FileState.FAILED
            ):
                manifest.add(existing_entry)
                continue

            corpus_changed = True
            file_id = existing_entry.file_id if existing_entry else str(uuid.uuid4())
            entry = ManifestEntry(
                file_id=file_id,
                rel_path=rel_path,
                abs_path=str(abs_path),
                content_hash=content_hash,
                file_type=ext,
                size_bytes=len(raw_bytes),
                state=FileState.REGISTERED,
                enrichment_state=EnrichmentState.PENDING,
            )
            manifest.add(entry)

        if existing_supported != discovered_supported:
            corpus_changed = True
        if corpus_changed:
            manifest.reset_finalization()

        _progress(f"Discovered {len(file_paths)} supported file(s).")
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
            parser=str(getattr(self._config, "parser", "cpu")),
        )

    def _relative_path(self, abs_path: Path, source: Path) -> str:
        """Get relative path string with forward slashes."""
        try:
            return str(abs_path.relative_to(source)).replace("\\", "/")
        except ValueError:
            return str(abs_path).replace("\\", "/")


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
        enrichment_state=EnrichmentState.NOT_APPLICABLE,
        failure_stage=stage,
        failure_message=message,
    )
