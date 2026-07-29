"""Background scheduling for model-backed collection enrichment."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from fitz_sage.engines.fitz_krag.progressive.manifest import (
    EnrichmentState,
    FileState,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from fitz_sage.engines.fitz_krag.ingestion.pipeline import KragIngestPipeline
    from fitz_sage.engines.fitz_krag.progressive.manifest import FileManifest, ManifestEntry

logger = logging.getLogger(__name__)


class BackgroundEnrichmentWorker:
    """Enrich an already-searchable source index without blocking queries."""

    def __init__(
        self,
        manifest: "FileManifest",
        core: "KragIngestPipeline",
    ) -> None:
        self._manifest = manifest
        self._core = core
        self._stop_event = threading.Event()
        self._query_active = threading.Event()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._failure: Exception | None = None

    def start(self) -> None:
        """Start enrichment on a daemon thread."""
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="fitz-enrichment-worker",
        )
        self._thread.start()
        logger.info("Background enrichment worker started")

    def run_until_complete(self) -> None:
        """Run pending file and collection enrichment synchronously."""
        self._run_enrichment()

    def stop(self) -> None:
        """Signal the worker and wait briefly for its current operation."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("Background enrichment worker stopped")

    def wait(self, progress: "Callable[[str], None] | None" = None) -> None:
        """Block until background enrichment settles or fails."""
        if self._thread is None:
            return
        last_emit = 0.0
        while not self._done.is_set() and self._thread.is_alive():
            if progress and time.monotonic() - last_emit >= 10.0:
                line = self._status_line()
                if line:
                    progress(line)
                last_emit = time.monotonic()
            self._done.wait(timeout=1.0)
        if self._failure is not None:
            raise RuntimeError(f"Collection enrichment failed: {self._failure}") from self._failure
        if progress:
            progress("Background enrichment complete.")

    def signal_query_start(self) -> None:
        """Pause before the next model call while a query is active."""
        self._query_active.set()

    def signal_query_end(self) -> None:
        """Allow background model work to continue."""
        self._query_active.clear()

    def boost_files(self, rel_paths: list[str]) -> None:
        """Prioritize files surfaced by a query and their directory siblings."""
        self._manifest.bump_priority(rel_paths)
        dirs = {str(Path(path).parent) for path in rel_paths}
        siblings = [
            path
            for path, entry in self._manifest.entries().items()
            if entry.state == FileState.INDEXED
            and entry.enrichment_state
            not in {
                EnrichmentState.COMPLETE,
                EnrichmentState.SUMMARIZED,
                EnrichmentState.NOT_APPLICABLE,
            }
            and str(Path(path).parent) in dirs
            and path not in rel_paths
        ]
        if siblings:
            self._manifest.bump_priority_level(siblings, level=2)

    def _run(self) -> None:
        self._run_enrichment()
        if self._failure is None:
            self._warm_loop()

    def _run_enrichment(self) -> None:
        try:
            started = time.perf_counter()
            self._enrich_files()
            if not self._stop_event.is_set():
                self._finalize_collection()
            logger.info(
                "Background enrichment settled in %.1fs",
                time.perf_counter() - started,
            )
        except Exception as exc:
            self._failure = exc
            logger.error("Background enrichment failed: %s", exc)
        finally:
            self._manifest.save()
            self._done.set()

    def _ordered_files(self, state: EnrichmentState) -> list["ManifestEntry"]:
        files = self._manifest.files_in_enrichment_state(state)
        files.sort(key=lambda entry: (entry.priority, entry.size_bytes, entry.rel_path))
        return files

    def _wait_if_query_active(self) -> None:
        while self._query_active.is_set() and not self._stop_event.is_set():
            self._stop_event.wait(timeout=0.5)

    def _enrich_files(self) -> None:
        for state in (EnrichmentState.PENDING, EnrichmentState.ENTITY_LINKED):
            for entry in self._ordered_files(state):
                if self._stop_event.is_set():
                    return
                self._wait_if_query_active()
                if self._stop_event.is_set():
                    return
                self._enrich_entry(entry)
                self._manifest.save()

    def _enrich_entry(self, entry: "ManifestEntry") -> None:
        current = self._manifest.get(entry.rel_path)
        if current is None or current.state != FileState.INDEXED:
            return

        if current.enrichment_state == EnrichmentState.PENDING:
            try:
                self._core.link_entities_file(current.file_id, current.file_type)
            except Exception as exc:
                self._manifest.mark_enrichment_failed(
                    current.rel_path,
                    stage="entities",
                    message=str(exc),
                )
                logger.warning("Entity enrichment failed for %s: %s", current.rel_path, exc)
                return
            self._manifest.update_enrichment_state(
                current.rel_path,
                EnrichmentState.ENTITY_LINKED,
            )
            current = self._manifest.get(current.rel_path)
            if current is None:
                return

        if current.enrichment_state == EnrichmentState.ENTITY_LINKED:
            try:
                self._core.build_hierarchy_file(current.file_id, current.file_type)
            except Exception as exc:
                self._manifest.mark_enrichment_failed(
                    current.rel_path,
                    stage="hierarchy",
                    message=str(exc),
                )
                logger.warning("Hierarchy enrichment failed for %s: %s", current.rel_path, exc)
                return
            self._manifest.update_enrichment_state(
                current.rel_path,
                EnrichmentState.COMPLETE,
            )

    def _finalize_collection(self) -> None:
        self._wait_if_query_active()
        if self._stop_event.is_set():
            return
        try:
            self._core.build_corpus_hierarchy()
        except Exception as exc:
            self._manifest.mark_finalization_failed(str(exc))
            logger.warning("Collection hierarchy finalization failed: %s", exc)
            return
        self._manifest.mark_finalized()

    def _status_line(self) -> str:
        entries = [
            entry
            for entry in self._manifest.entries().values()
            if entry.state == FileState.INDEXED
        ]
        if not entries:
            return ""
        completed = sum(
            entry.enrichment_state in {EnrichmentState.COMPLETE, EnrichmentState.SUMMARIZED}
            for entry in entries
        )
        return f"Enriching documents... {completed}/{len(entries)}"

    def _warm_loop(self) -> None:
        """Generate optional summaries only for files surfaced by a query."""
        while not self._stop_event.is_set():
            entry = self._next_warm_target()
            if entry is None:
                self._stop_event.wait(timeout=2.0)
                continue
            self._wait_if_query_active()
            if self._stop_event.is_set():
                return
            self._summarize_entry(entry)

    def _summarize_entry(self, entry: "ManifestEntry") -> None:
        """Persist demand-summary success or a retryable failure."""
        try:
            self._core.summarize_file(entry.file_id, entry.file_type)
        except Exception as exc:
            self._manifest.mark_enrichment_failed(
                entry.rel_path,
                stage="summary",
                message=str(exc),
            )
            logger.warning("Demand summary failed for %s: %s", entry.rel_path, exc)
        else:
            self._manifest.update_enrichment_state(
                entry.rel_path,
                EnrichmentState.SUMMARIZED,
            )
        self._manifest.save()

    def _next_warm_target(self) -> "ManifestEntry | None":
        candidates = [
            entry
            for entry in self._manifest.entries().values()
            if entry.state == FileState.INDEXED
            and entry.enrichment_state == EnrichmentState.COMPLETE
            and entry.last_queried_at is not None
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda entry: (entry.priority, -(entry.last_queried_at or 0.0)))
        return candidates[0]
