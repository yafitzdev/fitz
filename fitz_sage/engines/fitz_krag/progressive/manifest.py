# fitz_sage/engines/fitz_krag/progressive/manifest.py
"""
FileManifest — thread-safe manifest with JSON persistence.

Tracks every file in a pointed source directory, its indexing state,
extracted symbols/headings, and priority for background ingestion.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FileState(str, Enum):
    """Progressive indexing state for a file."""

    REGISTERED = "registered"  # In manifest only, no DB data
    PARSED = "parsed"  # Raw content + symbols/sections stored
    KEYWORDED = "keyworded"  # Qwen keywords/aliases extracted
    QUERY_READY = "query_ready"  # Minimum retrieval index is usable
    ENTITY_LINKED = "entity_linked"  # Entity graph was populated
    HIERARCHY_READY = "hierarchy_ready"  # L1 hierarchy summary exists
    ENRICHED = "enriched"  # Required deep enrichment completed
    SUMMARIZED = "summarized"  # Demand summaries exist for queried files


_QUERY_READY_STATES = frozenset(
    {
        FileState.QUERY_READY,
        FileState.ENTITY_LINKED,
        FileState.HIERARCHY_READY,
        FileState.ENRICHED,
        FileState.SUMMARIZED,
    }
)
_FULLY_ENRICHED_STATES = frozenset({FileState.ENRICHED, FileState.SUMMARIZED})


def is_query_ready_state(state: FileState) -> bool:
    """Return whether a manifest state can answer retrieval queries."""
    return state in _QUERY_READY_STATES


def is_fully_enriched_state(state: FileState) -> bool:
    """Return whether required deep enrichment has completed for a file."""
    return state in _FULLY_ENRICHED_STATES


@dataclass
class ManifestSymbol:
    """A code symbol extracted from AST (no LLM needed)."""

    name: str
    qualified_name: str
    kind: str  # function, class, method, constant
    signature: str | None
    start_line: int
    end_line: int


@dataclass
class ManifestHeading:
    """A heading extracted from a document file."""

    title: str
    level: int


@dataclass
class ManifestEntry:
    """A single file tracked in the manifest."""

    file_id: str
    rel_path: str
    abs_path: str
    content_hash: str
    file_type: str  # .py, .md, etc.
    size_bytes: int
    state: FileState
    symbols: list[ManifestSymbol] = field(default_factory=list)
    headings: list[ManifestHeading] = field(default_factory=list)
    priority: int = 4  # 1=highest (queried), 4=default
    last_queried_at: float | None = None


class FileManifest:
    """Thread-safe manifest with JSON persistence.

    Persisted at .fitz/collections/{collection}/manifest.json in the workspace.
    All mutations are guarded by a threading.Lock.
    """

    def __init__(self, manifest_path: Path) -> None:
        self._path = manifest_path
        self._lock = threading.Lock()
        self._entries: dict[str, ManifestEntry] = {}
        if self._path.exists():
            self.load()

    def entries(self) -> dict[str, ManifestEntry]:
        """Return a snapshot of all entries keyed by rel_path."""
        with self._lock:
            return dict(self._entries)

    def get(self, rel_path: str) -> ManifestEntry | None:
        """Get a single entry by relative path."""
        with self._lock:
            return self._entries.get(rel_path)

    def add(self, entry: ManifestEntry) -> None:
        """Add or replace an entry."""
        with self._lock:
            self._entries[entry.rel_path] = entry

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._entries.clear()

    def update_state(self, rel_path: str, state: FileState) -> None:
        """Transition a file to a new state."""
        with self._lock:
            entry = self._entries.get(rel_path)
            if entry:
                self._entries[rel_path] = ManifestEntry(
                    file_id=entry.file_id,
                    rel_path=entry.rel_path,
                    abs_path=entry.abs_path,
                    content_hash=entry.content_hash,
                    file_type=entry.file_type,
                    size_bytes=entry.size_bytes,
                    state=state,
                    symbols=entry.symbols,
                    headings=entry.headings,
                    priority=entry.priority,
                    last_queried_at=entry.last_queried_at,
                )

    def bump_priority(self, rel_paths: list[str]) -> None:
        """Set queried files to P1, record query time."""
        now = time.time()
        with self._lock:
            for rp in rel_paths:
                entry = self._entries.get(rp)
                if entry:
                    self._entries[rp] = ManifestEntry(
                        file_id=entry.file_id,
                        rel_path=entry.rel_path,
                        abs_path=entry.abs_path,
                        content_hash=entry.content_hash,
                        file_type=entry.file_type,
                        size_bytes=entry.size_bytes,
                        state=entry.state,
                        symbols=entry.symbols,
                        headings=entry.headings,
                        priority=1,
                        last_queried_at=now,
                    )

    def bump_priority_level(self, rel_paths: list[str], level: int) -> None:
        """Set files to a specific priority level (only if it improves priority)."""
        with self._lock:
            for rp in rel_paths:
                entry = self._entries.get(rp)
                if entry and entry.priority > level:
                    self._entries[rp] = ManifestEntry(
                        file_id=entry.file_id,
                        rel_path=entry.rel_path,
                        abs_path=entry.abs_path,
                        content_hash=entry.content_hash,
                        file_type=entry.file_type,
                        size_bytes=entry.size_bytes,
                        state=entry.state,
                        symbols=entry.symbols,
                        headings=entry.headings,
                        priority=level,
                        last_queried_at=entry.last_queried_at,
                    )

    def files_in_state(self, state: FileState) -> list[ManifestEntry]:
        """Return entries at a specific state."""
        with self._lock:
            return [e for e in self._entries.values() if e.state == state]

    def files_not_in_state(self, state: FileState) -> list[ManifestEntry]:
        """Return entries NOT at a specific state."""
        with self._lock:
            return [e for e in self._entries.values() if e.state != state]

    def files_not_query_ready(self) -> list[ManifestEntry]:
        """Return entries that still need the minimum retrieval index."""
        with self._lock:
            return [e for e in self._entries.values() if not is_query_ready_state(e.state)]

    def to_manifest_text(self, entries: list[ManifestEntry] | None = None) -> str:
        """Build compact manifest text for LLM consumption.

        Format per file (~50-100 tokens):
            path/to/file.py [4.2KB, python]
              fn: main(args) L10-25
              cls: MyClass L30-80
            path/to/doc.md [1.1KB, markdown]
              # Introduction
              ## Getting Started
        """
        if entries is None:
            with self._lock:
                items = list(self._entries.values())
        else:
            items = entries
        lines: list[str] = []
        for entry in items:
            size_str = _format_size(entry.size_bytes)
            lang = _ext_to_lang(entry.file_type)
            lines.append(f"{entry.rel_path} [{size_str}, {lang}]")
            for sym in entry.symbols:
                sig = f"({sym.signature})" if sym.signature else ""
                lines.append(f"  {sym.kind[:3]}: {sym.name}{sig} L{sym.start_line}-{sym.end_line}")
            for heading in entry.headings:
                prefix = "#" * heading.level
                lines.append(f"  {prefix} {heading.title}")
        return "\n".join(lines)

    def save(self) -> None:
        """Persist manifest to JSON."""
        with self._lock:
            data = {rp: _entry_to_dict(entry) for rp, entry in self._entries.items()}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self) -> None:
        """Load manifest from JSON."""
        try:
            text = self._path.read_text(encoding="utf-8")
            raw = json.loads(text)
            with self._lock:
                self._entries = {rp: _dict_to_entry(d) for rp, d in raw.items()}
        except Exception as e:
            logger.warning(f"Failed to load manifest from {self._path}: {e}")
            with self._lock:
                self._entries = {}


def indexing_status(manifest: "FileManifest | None") -> dict[str, Any]:
    """Summarize a manifest's background-indexing progress.

    Returns ``total`` / ``indexed`` / ``pending`` counts for the query-ready
    index, plus deep-enrichment progress. ``complete`` intentionally means the
    collection can serve retrieval queries while the daemon may still be doing
    mandatory entity/hierarchy work in the background.
    """
    if manifest is None:
        return {
            "total": 0,
            "indexed": 0,
            "pending": 0,
            "complete": True,
            "query_ready": True,
            "deep_pending": 0,
            "deep_pending_files": [],
            "fully_enriched": True,
            "by_state": {},
        }

    by_state: dict[str, int] = {}
    entries = list(manifest.entries().values())
    for entry in entries:
        by_state[entry.state.value] = by_state.get(entry.state.value, 0) + 1

    total = sum(by_state.values())
    indexed = sum(1 for entry in entries if is_query_ready_state(entry.state))
    fully_enriched = sum(1 for entry in entries if is_fully_enriched_state(entry.state))
    pending = total - indexed
    deep_pending = total - fully_enriched
    deep_pending_files = [
        {
            "path": entry.rel_path,
            "state": entry.state.value,
            "priority": entry.priority,
        }
        for entry in sorted(
            entries,
            key=lambda item: (item.priority, item.size_bytes, item.rel_path),
        )
        if not is_fully_enriched_state(entry.state)
    ][:5]
    return {
        "total": total,
        "indexed": indexed,
        "pending": pending,
        "complete": pending == 0,
        "query_ready": pending == 0,
        "deep_pending": deep_pending,
        "deep_pending_files": deep_pending_files,
        "fully_enriched": deep_pending == 0,
        "by_state": by_state,
    }


def _entry_to_dict(entry: ManifestEntry) -> dict[str, Any]:
    """Serialize ManifestEntry to JSON-compatible dict."""
    d = asdict(entry)
    d["state"] = entry.state.value
    return d


def _dict_to_entry(d: dict[str, Any]) -> ManifestEntry:
    """Deserialize ManifestEntry from dict."""
    return ManifestEntry(
        file_id=d["file_id"],
        rel_path=d["rel_path"],
        abs_path=d["abs_path"],
        content_hash=d["content_hash"],
        file_type=d["file_type"],
        size_bytes=d["size_bytes"],
        state=FileState(d["state"]),
        symbols=[ManifestSymbol(**s) for s in d.get("symbols", [])],
        headings=[ManifestHeading(**h) for h in d.get("headings", [])],
        priority=d.get("priority", 4),
        last_queried_at=d.get("last_queried_at"),
    )


def _format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f}MB"


def _ext_to_lang(ext: str) -> str:
    """Map file extension to language label."""
    mapping = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".java": "java",
        ".go": "go",
        ".md": "markdown",
        ".rst": "rst",
        ".txt": "text",
        ".csv": "csv",
        ".xlsx": "excel",
    }
    return mapping.get(ext, ext.lstrip("."))
