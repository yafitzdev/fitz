# fitz_sage/engines/fitz_krag/progressive/worker.py
"""
BackgroundIngestWorker — daemon thread that schedules ingestion progressively.

The worker is a *scheduler*: it owns the manifest, the priority queue, the
state machine, the daemon thread, and query-pausing. The actual ingestion
work is delegated to the shared ``KragIngestPipeline`` core — the worker
never reimplements parse/summarize/enrich.

State machine per file:
    REGISTERED → PARSED    (core.parse_file — store raw, extract symbols/sections)
    PARSED     → ENRICHED  (core.enrich_file — optional keywords/entities,
                            entity graph, L1 hierarchy summary)
Once every file is ENRICHED the worker runs ``core.finalize`` (import graph +
L2 hierarchy summary): eager indexing is then complete.

ENRICHED → SUMMARIZED (core.summarize_file) is demand-driven. Summaries are
generated only when a summarizer provider is configured and only for files a
query has surfaced. Un-queried files are never summarized.

Priority queue:
    P1: Files the user just queried about
    P2: Files in the same directory as queried files
    P4: Remaining files by size ascending
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from fitz_sage.engines.fitz_krag.progressive.manifest import FileState

if TYPE_CHECKING:
    from collections.abc import Callable

    from fitz_sage.engines.fitz_krag.ingestion.pipeline import KragIngestPipeline
    from fitz_sage.engines.fitz_krag.progressive.manifest import FileManifest, ManifestEntry

logger = logging.getLogger(__name__)


class BackgroundIngestWorker:
    """Daemon thread that schedules the ingestion core: REGISTERED → ENRICHED."""

    def __init__(
        self,
        manifest: "FileManifest",
        source_dir: Path,
        core: "KragIngestPipeline",
    ) -> None:
        self._manifest = manifest
        self._source_dir = source_dir
        self._core = core

        self._stop_event = threading.Event()
        self._query_active = threading.Event()  # Set = query is running
        self._eager_done = threading.Event()  # Set = parse/enrich/finalize done
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start daemon thread (daemon=True, won't block process exit)."""
        self._thread = threading.Thread(target=self._run, daemon=True, name="fitz-bg-worker")
        self._thread.start()
        logger.info("Background ingestion worker started")

    def stop(self) -> None:
        """Signal stop, join with timeout."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("Background ingestion worker stopped")

    def wait(self, progress: "Callable[[str], None] | None" = None) -> None:
        """Block until *eager* indexing (parse/enrich/finalize) has finished.

        Reports coarse progress (files enriched / total) at ~10s intervals.
        Returns once the corpus is queryable — the worker thread keeps
        running afterwards to summarize queried files on demand. A no-op
        when the worker was never started.
        """
        if self._thread is None:
            return
        last_emit = 0.0
        while not self._eager_done.is_set() and self._thread.is_alive():
            if progress and time.monotonic() - last_emit >= 10.0:
                line = self._status_line()
                if line:
                    progress(line)
                last_emit = time.monotonic()
            self._eager_done.wait(timeout=1.0)
        if progress:
            progress("Indexing complete.")

    def _status_line(self) -> str:
        """Coarse indexing-progress line from manifest file states."""
        entries = self._manifest.entries()
        total = len(entries)
        if total == 0:
            return ""
        done = sum(
            1 for e in entries.values() if e.state in (FileState.ENRICHED, FileState.SUMMARIZED)
        )
        return f"Indexing documents... {done}/{total}"

    def signal_query_start(self) -> None:
        """Pause LLM calls (let query have priority)."""
        self._query_active.set()

    def signal_query_end(self) -> None:
        """Resume LLM calls."""
        self._query_active.clear()

    def boost_files(self, rel_paths: list[str]) -> None:
        """Bump queried files to P1, same-directory files to P2."""
        self._manifest.bump_priority(rel_paths)

        # Find directory siblings and bump to P2
        dirs = {str(Path(rp).parent) for rp in rel_paths}
        siblings: list[str] = []
        for rp, entry in self._manifest.entries().items():
            if entry.state == FileState.ENRICHED:
                continue
            parent = str(Path(rp).parent)
            if parent in dirs and rp not in rel_paths:
                siblings.append(rp)
        if siblings:
            self._manifest.bump_priority_level(siblings, level=2)

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Eager phases (parse → enrich → finalize), then the demand-driven warm loop."""
        try:
            t0 = time.perf_counter()
            self._parse_phase()  # REGISTERED → PARSED (no LLM)
            t1 = time.perf_counter()
            self._enrich_phase()  # PARSED → ENRICHED (LLM)
            t2 = time.perf_counter()
            self._finalize_phase()  # corpus finalize
            t3 = time.perf_counter()
            logger.info(
                "Eager indexing complete in %.1fs (parse=%.1fs, enrich=%.1fs, finalize=%.1fs)",
                t3 - t0,
                t1 - t0,
                t2 - t1,
                t3 - t2,
            )
        except Exception as e:
            logger.error(f"Background worker failed: {e}")
        finally:
            self._eager_done.set()
        # Demand-driven summarization runs until the worker is stopped.
        self._warm_loop()

    def _get_ordered_files(self, state: FileState) -> list["ManifestEntry"]:
        """Get files in priority order for a given state.

        Uses manifest priority (set by bump_priority) rather than draining
        the queue, so boosts persist across processing phases.
        """
        files = self._manifest.files_in_state(state)
        files.sort(key=lambda entry: (entry.priority, entry.size_bytes))
        return files

    def _wait_if_query_active(self) -> None:
        """Block while a query is running so the LLM stays free for it."""
        while self._query_active.is_set() and not self._stop_event.is_set():
            self._stop_event.wait(timeout=0.5)

    def _parse_phase(self) -> None:
        """REGISTERED → PARSED: store raw content + extract symbols/sections."""
        for entry in self._get_ordered_files(FileState.REGISTERED):
            if self._stop_event.is_set():
                return
            try:
                abs_path = Path(entry.abs_path)
                if not abs_path.exists():
                    abs_path = self._source_dir / entry.rel_path
                self._core.parse_file(entry.rel_path, abs_path, entry.file_id)
                self._manifest.update_state(entry.rel_path, FileState.PARSED)
            except Exception as e:
                logger.warning(f"Background parse failed for {entry.rel_path}: {e}")
        self._manifest.save()

    def _enrich_phase(self) -> None:
        """PARSED → ENRICHED: extract keywords/entities + L1 (pauses during queries)."""
        for entry in self._get_ordered_files(FileState.PARSED):
            if self._stop_event.is_set():
                return
            self._wait_if_query_active()
            if self._stop_event.is_set():
                return
            try:
                self._core.enrich_file(entry.file_id, entry.file_type)
                self._manifest.update_state(entry.rel_path, FileState.ENRICHED)
            except Exception as e:
                logger.warning(f"Background enrichment failed for {entry.rel_path}: {e}")
        self._manifest.save()

    def _finalize_phase(self) -> None:
        """Corpus finalize — import graph + L2 hierarchy summary."""
        if self._stop_event.is_set():
            return
        self._wait_if_query_active()
        if self._stop_event.is_set():
            return
        try:
            self._core.finalize()
        except Exception as e:
            logger.warning(f"Background finalize failed: {e}")

    # ------------------------------------------------------------------
    # Demand-driven summarization — runs after the eager phases
    # ------------------------------------------------------------------

    def _warm_loop(self) -> None:
        """Summarize files that queries have surfaced, until the worker stops.

        Eager indexing leaves summaries ungenerated. ``boost_files`` records
        which files a query touched (``last_queried_at``); this loop picks
        those up and runs ``summarize_file`` so the reranker has real
        summaries on the next query. Files no query ever touches are never
        summarized — that is the point of demand-driven.
        """
        while not self._stop_event.is_set():
            entry = self._next_warm_target()
            if entry is None:
                self._stop_event.wait(timeout=2.0)
                continue
            self._wait_if_query_active()
            if self._stop_event.is_set():
                return
            try:
                self._core.summarize_file(entry.file_id, entry.file_type)
            except Exception as e:
                logger.warning(f"Demand summarize failed for {entry.rel_path}: {e}")
            # → SUMMARIZED regardless: a failed file must not loop forever.
            self._manifest.update_state(entry.rel_path, FileState.SUMMARIZED)
            self._manifest.save()

    def _next_warm_target(self) -> "ManifestEntry | None":
        """The highest-priority ENRICHED file a query has surfaced, or None."""
        candidates = [
            e
            for e in self._manifest.entries().values()
            if e.state == FileState.ENRICHED and e.last_queried_at is not None
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda e: (e.priority, -(e.last_queried_at or 0.0)))
        return candidates[0]
