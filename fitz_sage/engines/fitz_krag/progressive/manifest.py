# fitz_sage/engines/fitz_krag/progressive/manifest.py
"""Thread-safe source manifest with separate index and enrichment state."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FileState(str, Enum):
    """Source-index state for one file."""

    REGISTERED = "registered"
    INDEXED = "indexed"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


class EnrichmentState(str, Enum):
    """Background enrichment state, independent from source indexing."""

    PENDING = "pending"
    ENTITY_LINKED = "entity_linked"
    COMPLETE = "complete"
    SUMMARIZED = "summarized"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


class FinalizationState(str, Enum):
    """Collection-level hierarchy finalization state."""

    PENDING = "pending"
    COMPLETE = "complete"
    FAILED = "failed"


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
    enrichment_state: EnrichmentState = EnrichmentState.PENDING
    priority: int = 4  # 1=highest (queried), 4=default
    last_queried_at: float | None = None
    failure_stage: str | None = None
    failure_message: str | None = None
    enrichment_failure_stage: str | None = None
    enrichment_failure_message: str | None = None


class FileManifest:
    """Thread-safe manifest with JSON persistence.

    Persisted at .fitz/collections/{collection}/manifest.json in the workspace.
    All in-process mutations are guarded by a threading.Lock. Complete writer
    operations are serialized across manifest instances and processes by the
    collection write lock owned by the engine.
    """

    def __init__(self, manifest_path: Path) -> None:
        self._path = manifest_path
        self._lock = threading.Lock()
        self._entries: dict[str, ManifestEntry] = {}
        self._finalization_state = FinalizationState.PENDING
        self._finalization_failure: str | None = None
        if self._path.exists():
            self.load()

    @property
    def path(self) -> Path:
        """Return the persistence path used by this manifest."""
        return self._path

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
                    enrichment_state=entry.enrichment_state,
                    priority=entry.priority,
                    last_queried_at=entry.last_queried_at,
                    failure_stage=None,
                    failure_message=None,
                    enrichment_failure_stage=entry.enrichment_failure_stage,
                    enrichment_failure_message=entry.enrichment_failure_message,
                )

    def update_enrichment_state(self, rel_path: str, state: EnrichmentState) -> None:
        """Transition background enrichment without changing index availability."""
        with self._lock:
            entry = self._entries.get(rel_path)
            if entry:
                self._entries[rel_path] = _replace_entry(
                    entry,
                    enrichment_state=state,
                    enrichment_failure_stage=None,
                    enrichment_failure_message=None,
                )

    def mark_failed(self, rel_path: str, *, stage: str, message: str) -> None:
        """Record a terminal per-file failure."""
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
                    state=FileState.FAILED,
                    enrichment_state=EnrichmentState.NOT_APPLICABLE,
                    priority=entry.priority,
                    last_queried_at=entry.last_queried_at,
                    failure_stage=str(stage),
                    failure_message=str(message),
                )

    def mark_enrichment_failed(self, rel_path: str, *, stage: str, message: str) -> None:
        """Record an enrichment failure while preserving the searchable source index."""
        with self._lock:
            entry = self._entries.get(rel_path)
            if entry:
                self._entries[rel_path] = _replace_entry(
                    entry,
                    enrichment_state=EnrichmentState.FAILED,
                    enrichment_failure_stage=str(stage),
                    enrichment_failure_message=str(message),
                )

    def prepare_enrichment_retry(self) -> None:
        """Retry failed enrichment from the last durable per-file stage."""
        with self._lock:
            for rel_path, entry in self._entries.items():
                if entry.enrichment_state != EnrichmentState.FAILED:
                    continue
                retry_state = (
                    {
                        "hierarchy": EnrichmentState.ENTITY_LINKED,
                        "summary": EnrichmentState.COMPLETE,
                    }.get(entry.enrichment_failure_stage, EnrichmentState.PENDING)
                    if entry.enrichment_failure_stage is not None
                    else EnrichmentState.PENDING
                )
                self._entries[rel_path] = _replace_entry(
                    entry,
                    enrichment_state=retry_state,
                    enrichment_failure_stage=None,
                    enrichment_failure_message=None,
                )
            if self._finalization_state == FinalizationState.FAILED:
                self._finalization_state = FinalizationState.PENDING
                self._finalization_failure = None

    def reset_finalization(self) -> None:
        with self._lock:
            self._finalization_state = FinalizationState.PENDING
            self._finalization_failure = None

    def mark_finalized(self) -> None:
        with self._lock:
            self._finalization_state = FinalizationState.COMPLETE
            self._finalization_failure = None

    def mark_finalization_failed(self, message: str) -> None:
        with self._lock:
            self._finalization_state = FinalizationState.FAILED
            self._finalization_failure = str(message)

    def finalization_status(self) -> tuple[FinalizationState, str | None]:
        with self._lock:
            return self._finalization_state, self._finalization_failure

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
                        enrichment_state=entry.enrichment_state,
                        priority=1,
                        last_queried_at=now,
                        failure_stage=entry.failure_stage,
                        failure_message=entry.failure_message,
                        enrichment_failure_stage=entry.enrichment_failure_stage,
                        enrichment_failure_message=entry.enrichment_failure_message,
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
                        enrichment_state=entry.enrichment_state,
                        priority=level,
                        last_queried_at=entry.last_queried_at,
                        failure_stage=entry.failure_stage,
                        failure_message=entry.failure_message,
                        enrichment_failure_stage=entry.enrichment_failure_stage,
                        enrichment_failure_message=entry.enrichment_failure_message,
                    )

    def files_in_state(self, state: FileState) -> list[ManifestEntry]:
        """Return entries at a specific state."""
        with self._lock:
            return [e for e in self._entries.values() if e.state == state]

    def files_in_enrichment_state(self, state: EnrichmentState) -> list[ManifestEntry]:
        """Return indexed files at a specific enrichment state."""
        with self._lock:
            return [
                entry
                for entry in self._entries.values()
                if entry.state == FileState.INDEXED and entry.enrichment_state == state
            ]

    def save(self) -> None:
        """Persist manifest atomically so readers never observe partial JSON."""
        with self._lock:
            data = {
                "version": 1,
                "files": {rp: _entry_to_dict(entry) for rp, entry in self._entries.items()},
                "finalization": {
                    "state": self._finalization_state.value,
                    "failure": self._finalization_failure,
                },
            }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._path.with_name(f".{self._path.name}.tmp")
        temporary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary_path.replace(self._path)

    def load(self) -> None:
        """Load manifest from JSON."""
        try:
            text = self._path.read_text(encoding="utf-8")
            raw = json.loads(text)
            if raw["version"] != 1:
                raise ValueError(f"unsupported manifest version: {raw['version']}")
            files = raw["files"]
            finalization = raw["finalization"]
            with self._lock:
                self._entries = {rp: _dict_to_entry(d) for rp, d in files.items()}
                self._finalization_state = FinalizationState(finalization["state"])
                self._finalization_failure = finalization["failure"]
        except Exception as e:
            logger.warning(f"Failed to load manifest from {self._path}: {e}")
            with self._lock:
                self._entries = {}
                self._finalization_state = FinalizationState.PENDING
                self._finalization_failure = None


def indexing_status(manifest: "FileManifest | None") -> dict[str, Any]:
    """Report source-index health and independent background enrichment."""
    if manifest is None:
        return {
            "discovered": 0,
            "total": 0,
            "indexed": 0,
            "pending": 0,
            "failed": 0,
            "failed_files": [],
            "unsupported": 0,
            "unsupported_files": [],
            "healthy": True,
            "complete": True,
            "query_ready": True,
            "by_index_state": {},
            "enrichment": {
                "total": 0,
                "completed": 0,
                "pending": 0,
                "failed": 0,
                "failed_files": [],
                "pending_files": [],
                "finalization": "complete",
                "finalization_error": None,
                "complete": True,
            },
            "by_enrichment_state": {},
        }

    by_index_state: dict[str, int] = {}
    by_enrichment_state: dict[str, int] = {}
    entries = list(manifest.entries().values())
    for entry in entries:
        by_index_state[entry.state.value] = by_index_state.get(entry.state.value, 0) + 1
        if entry.state == FileState.INDEXED:
            key = entry.enrichment_state.value
            by_enrichment_state[key] = by_enrichment_state.get(key, 0) + 1

    unsupported_entries = [entry for entry in entries if entry.state == FileState.UNSUPPORTED]
    failed_entries = [entry for entry in entries if entry.state == FileState.FAILED]
    indexed_entries = [entry for entry in entries if entry.state == FileState.INDEXED]
    registered_entries = [entry for entry in entries if entry.state == FileState.REGISTERED]
    total = len(entries) - len(unsupported_entries)
    pending = len(registered_entries)

    enrichment_complete_states = {EnrichmentState.COMPLETE, EnrichmentState.SUMMARIZED}
    enrichment_pending_entries = [
        entry
        for entry in indexed_entries
        if entry.enrichment_state in {EnrichmentState.PENDING, EnrichmentState.ENTITY_LINKED}
    ]
    enrichment_failed_entries = [
        entry for entry in indexed_entries if entry.enrichment_state == EnrichmentState.FAILED
    ]
    enrichment_completed = sum(
        1 for entry in indexed_entries if entry.enrichment_state in enrichment_complete_states
    )
    enrichment_pending_files = [
        {
            "path": entry.rel_path,
            "state": entry.enrichment_state.value,
            "priority": entry.priority,
        }
        for entry in sorted(
            enrichment_pending_entries,
            key=lambda item: (item.priority, item.size_bytes, item.rel_path),
        )
    ][:5]
    enrichment_failed_files = [
        {
            "path": entry.rel_path,
            "stage": entry.enrichment_failure_stage,
            "error": entry.enrichment_failure_message,
        }
        for entry in sorted(enrichment_failed_entries, key=lambda item: item.rel_path)
    ]
    failed_files = [
        {
            "path": entry.rel_path,
            "stage": entry.failure_stage,
            "error": entry.failure_message,
        }
        for entry in sorted(failed_entries, key=lambda item: item.rel_path)
    ]
    unsupported_files = [
        {"path": entry.rel_path, "extension": entry.file_type}
        for entry in sorted(unsupported_entries, key=lambda item: item.rel_path)
    ]
    finalization_state, finalization_error = manifest.finalization_status()
    if not indexed_entries:
        finalization_state = FinalizationState.COMPLETE
        finalization_error = None
    enrichment_complete = (
        not enrichment_pending_entries
        and not enrichment_failed_entries
        and finalization_state == FinalizationState.COMPLETE
    )
    return {
        "discovered": len(entries),
        "total": total,
        "indexed": len(indexed_entries),
        "pending": pending,
        "failed": len(failed_entries),
        "failed_files": failed_files,
        "unsupported": len(unsupported_entries),
        "unsupported_files": unsupported_files,
        "healthy": not failed_entries,
        "complete": pending == 0 and not failed_entries,
        "query_ready": pending == 0,
        "by_index_state": by_index_state,
        "enrichment": {
            "total": len(indexed_entries),
            "completed": enrichment_completed,
            "pending": len(enrichment_pending_entries),
            "failed": len(enrichment_failed_entries),
            "failed_files": enrichment_failed_files,
            "pending_files": enrichment_pending_files,
            "finalization": finalization_state.value,
            "finalization_error": finalization_error,
            "complete": enrichment_complete,
        },
        "by_enrichment_state": by_enrichment_state,
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
        enrichment_state=EnrichmentState(d["enrichment_state"]),
        priority=d.get("priority", 4),
        last_queried_at=d.get("last_queried_at"),
        failure_stage=d.get("failure_stage"),
        failure_message=d.get("failure_message"),
        enrichment_failure_stage=d.get("enrichment_failure_stage"),
        enrichment_failure_message=d.get("enrichment_failure_message"),
    )


def _replace_entry(entry: ManifestEntry, **changes: Any) -> ManifestEntry:
    values = asdict(entry)
    values.update(changes)
    return ManifestEntry(**values)
