# fitz_sage/engines/fitz_krag/progressive/worker.py
"""
BackgroundIngestWorker — daemon thread that schedules ingestion progressively.

The worker is a *scheduler*: it owns the manifest, the priority queue, the
state machine, the daemon thread, and query-pausing. The actual ingestion
work is delegated to the shared ``KragIngestPipeline`` core — the worker
never reimplements parse/summarize/enrich.

State machine per file:
    REGISTERED → PARSED      (core.parse_file — store raw, extract symbols/sections)
    PARSED     → SUMMARIZED  (core.summarize_file — LLM summaries)
    SUMMARIZED → ENRICHED    (core.enrich_file — keywords/entities, vocabulary,
                              entity graph, L1 hierarchy)
    ENRICHED is terminal.
Once every file is ENRICHED, the worker runs ``core.finalize`` (import graph
+ L2 hierarchy summary).

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
        """Block until the worker has finished indexing the whole corpus.

        Reports coarse progress (files enriched / total) at ~10s intervals.
        A no-op when the worker was never started.
        """
        if self._thread is None:
            return
        last_emit = 0.0
        while self._thread.is_alive():
            if progress and time.monotonic() - last_emit >= 10.0:
                line = self._status_line()
                if line:
                    progress(line)
                last_emit = time.monotonic()
            self._thread.join(timeout=1.0)
        if progress:
            progress("Indexing complete.")

    def _status_line(self) -> str:
        """Coarse indexing-progress line from manifest file states."""
        entries = self._manifest.entries()
        total = len(entries)
        if total == 0:
            return ""
        done = sum(1 for e in entries.values() if e.state == FileState.ENRICHED)
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
        """Main worker loop — schedule the core's ops phase by phase."""
        try:
            t0 = time.perf_counter()
            self._parse_phase()  # REGISTERED → PARSED (no LLM)
            t1 = time.perf_counter()
            self._summarize_phase()  # PARSED → SUMMARIZED (LLM)
            t2 = time.perf_counter()
            self._enrich_phase()  # SUMMARIZED → ENRICHED (LLM)
            t3 = time.perf_counter()
            self._finalize_phase()  # corpus finalize
            t4 = time.perf_counter()
            logger.info(
                "Background indexing complete in %.1fs "
                "(parse=%.1fs, summarize=%.1fs, enrich=%.1fs, finalize=%.1fs)",
                t4 - t0,
                t1 - t0,
                t2 - t1,
                t3 - t2,
                t4 - t3,
            )
        except Exception as e:
            logger.error(f"Background worker failed: {e}")

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

    def _summarize_phase(self) -> None:
        """PARSED → SUMMARIZED: generate LLM summaries (pauses during queries)."""
        for entry in self._get_ordered_files(FileState.PARSED):
            if self._stop_event.is_set():
                return
            self._wait_if_query_active()
            if self._stop_event.is_set():
                return
            try:
                self._core.summarize_file(entry.file_id, entry.file_type)
                self._manifest.update_state(entry.rel_path, FileState.SUMMARIZED)
            except Exception as e:
                logger.warning(f"Background summarize failed for {entry.rel_path}: {e}")
        self._manifest.save()

    def _enrich_phase(self) -> None:
        """SUMMARIZED → ENRICHED: extract keywords/entities (pauses during queries)."""
        for entry in self._get_ordered_files(FileState.SUMMARIZED):
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
