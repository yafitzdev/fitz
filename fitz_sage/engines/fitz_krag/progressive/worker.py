# fitz_sage/engines/fitz_krag/progressive/worker.py
"""
BackgroundIngestWorker — daemon thread that schedules ingestion progressively.

The worker is a *scheduler*: it owns the manifest, the priority queue, the
state machine, the daemon thread, and query-pausing. The actual ingestion
work is delegated to the shared ``KragIngestPipeline`` core — the worker
never reimplements parse/summarize/enrich.

State machine per file:
    REGISTERED → PARSED    (core.parse_file — store raw, extract symbols/sections)
    PARSED     → KEYWORDED → QUERY_READY
                           (core.keyword_file — minimum Qwen retrieval index)
    QUERY_READY → ENTITY_LINKED → HIERARCHY_READY → ENRICHED
                           (entity graph + L1 hierarchy summary)
Once every file reaches ENRICHED the worker runs ``core.finalize`` (import
graph + L2 hierarchy summary). ``wait()`` only blocks through QUERY_READY so
queries can run while mandatory deep enrichment continues.

ENRICHED → SUMMARIZED (core.summarize_file) is demand-driven. Summaries are
generated only for files a query has surfaced. Un-queried files are never
summarized.

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

from fitz_sage.engines.fitz_krag.progressive.manifest import (
    FileState,
    is_fully_enriched_state,
    is_query_ready_state,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from fitz_sage.engines.fitz_krag.ingestion.pipeline import KragIngestPipeline
    from fitz_sage.engines.fitz_krag.progressive.manifest import FileManifest, ManifestEntry

logger = logging.getLogger(__name__)


class BackgroundIngestWorker:
    """Daemon thread that schedules the ingestion core through staged enrichment."""

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
        self._parse_done = threading.Event()  # Set = typed units are searchable
        self._eager_done = threading.Event()  # Set = minimum query-ready index done
        self._thread: threading.Thread | None = None
        self._failure: Exception | None = None
        self._deep_failure: Exception | None = None

    def start(self) -> None:
        """Start daemon thread (daemon=True, won't block process exit)."""
        self._thread = threading.Thread(target=self._run, daemon=True, name="fitz-bg-worker")
        self._thread.start()
        logger.info("Background ingestion worker started")

    def run_until_deep_complete(self) -> None:
        """Run query-ready and deep enrichment synchronously, then exit."""
        query_ready = False
        try:
            self._parse_phase()
            self._parse_done.set()
            self._keyword_phase()
            query_ready = True
            self._eager_done.set()
        except Exception as e:
            self._failure = e
            logger.error(f"Background worker failed: {e}")
        finally:
            self._parse_done.set()
            self._eager_done.set()

        if not query_ready or self._failure is not None or self._stop_event.is_set():
            return

        try:
            self._deep_enrich_phase()
            self._finalize_phase()
        except Exception as e:
            self._deep_failure = e
            logger.error(f"Background deep enrichment failed: {e}")

    def stop(self) -> None:
        """Signal stop, join with timeout."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("Background ingestion worker stopped")

    def wait(self, progress: "Callable[[str], None] | None" = None) -> None:
        """Block until the minimum query-ready index has finished.

        Reports coarse progress (query-ready files / total) at ~10s intervals.
        Returns once the corpus can serve best-effort retrieval. The worker
        thread keeps running afterwards to complete mandatory entity/hierarchy
        enrichment and demand summaries. A no-op when the worker was never
        started.
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
        if self._failure is not None:
            if progress:
                progress(f"Indexing failed: {self._failure}")
            raise RuntimeError(f"Indexing failed: {self._failure}") from self._failure
        if progress:
            progress("Query-ready indexing complete; deep enrichment continues.")

    def wait_for_query_surface(self, progress: "Callable[[str], None] | None" = None) -> None:
        """Block until parsed symbols/sections/tables are searchable.

        This is the CLI fast path: BM25 can run after parsing, while Qwen
        keyword/entity/hierarchy enrichment continues behind the query.
        """
        if self._thread is None:
            return
        last_emit = 0.0
        while not self._parse_done.is_set() and self._thread.is_alive():
            if progress and time.monotonic() - last_emit >= 10.0:
                line = self._parse_status_line()
                if line:
                    progress(line)
                last_emit = time.monotonic()
            self._parse_done.wait(timeout=1.0)
        if self._failure is not None and not self._parse_done.is_set():
            if progress:
                progress(f"Indexing failed: {self._failure}")
            raise RuntimeError(f"Indexing failed: {self._failure}") from self._failure
        if progress:
            progress("Search surface ready; enrichment continues.")

    def _status_line(self) -> str:
        """Coarse indexing-progress line from manifest file states."""
        entries = self._manifest.entries()
        total = len(entries)
        if total == 0:
            return ""
        done = sum(1 for e in entries.values() if is_query_ready_state(e.state))
        return f"Indexing documents... {done}/{total}"

    def _parse_status_line(self) -> str:
        """Coarse parse-progress line from manifest file states."""
        entries = self._manifest.entries()
        total = len(entries)
        if total == 0:
            return ""
        parsed = sum(1 for e in entries.values() if e.state != FileState.REGISTERED)
        return f"Parsing documents... {parsed}/{total}"

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
            if is_fully_enriched_state(entry.state):
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
        """Query-ready phases first, then deep enrichment and the warm loop."""
        query_ready = False
        try:
            t0 = time.perf_counter()
            self._parse_phase()  # REGISTERED → PARSED (no LLM)
            t1 = time.perf_counter()
            self._parse_done.set()
            self._keyword_phase()  # PARSED → QUERY_READY (minimum LLM)
            t2 = time.perf_counter()
            query_ready = True
            self._eager_done.set()
            logger.info(
                "Query-ready indexing complete in %.1fs (parse=%.1fs, keyword=%.1fs)",
                t2 - t0,
                t1 - t0,
                t2 - t1,
            )
        except Exception as e:
            self._failure = e
            logger.error(f"Background worker failed: {e}")
        finally:
            self._parse_done.set()
            self._eager_done.set()

        if query_ready and self._failure is None and not self._stop_event.is_set():
            try:
                t3 = time.perf_counter()
                self._deep_enrich_phase()  # QUERY_READY → ENRICHED
                t4 = time.perf_counter()
                self._finalize_phase()  # corpus finalize
                t5 = time.perf_counter()
                logger.info(
                    "Deep enrichment complete in %.1fs (enrich=%.1fs, finalize=%.1fs)",
                    t5 - t3,
                    t4 - t3,
                    t5 - t4,
                )
            except Exception as e:
                self._deep_failure = e
                logger.error(f"Background deep enrichment failed: {e}")

        # Demand-driven summarization runs after the query-ready phase succeeds.
        if self._failure is None:
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

    def _keyword_phase(self) -> None:
        """PARSED → QUERY_READY: extract minimum keywords (pauses during queries)."""
        failures: list[str] = []
        for entry in self._get_ordered_files(FileState.PARSED):
            if self._stop_event.is_set():
                return
            self._wait_if_query_active()
            if self._stop_event.is_set():
                return
            try:
                self._core.keyword_file(entry.file_id, entry.file_type)
                self._manifest.update_state(entry.rel_path, FileState.KEYWORDED)
                self._manifest.update_state(entry.rel_path, FileState.QUERY_READY)
            except Exception as e:
                failures.append(f"{entry.rel_path}: {e}")
                logger.error(f"Background keyword enrichment failed for {entry.rel_path}: {e}")
                break
        self._manifest.save()
        if failures:
            raise RuntimeError(
                "Required keyword enrichment failed; indexing stopped before query-ready. "
                + failures[0]
            )

    def _deep_enrich_phase(self) -> None:
        """QUERY_READY → ENRICHED: entity graph + L1 hierarchy summaries."""
        failures: list[str] = []
        for state in (FileState.QUERY_READY, FileState.ENTITY_LINKED, FileState.HIERARCHY_READY):
            for entry in self._get_ordered_files(state):
                if self._stop_event.is_set():
                    return
                self._wait_if_query_active()
                if self._stop_event.is_set():
                    return
                try:
                    self._deep_enrich_entry(entry)
                except Exception as e:
                    failures.append(f"{entry.rel_path}: {e}")
                    logger.error(f"Background deep enrichment failed for {entry.rel_path}: {e}")
                    break
            if failures:
                break
        self._manifest.save()
        if failures:
            raise RuntimeError(
                "Required deep enrichment failed; finalize was skipped. " + failures[0]
            )

    def _deep_enrich_entry(self, entry: "ManifestEntry") -> None:
        """Run remaining enrichment phases for one query-ready entry."""
        current = self._manifest.get(entry.rel_path)
        if current is None:
            return

        if current.state == FileState.QUERY_READY:
            logger.info("Deep enrichment entity linking started for %s", current.rel_path)
            self._core.link_entities_file(current.file_id, current.file_type)
            self._manifest.update_state(current.rel_path, FileState.ENTITY_LINKED)
            logger.info("Deep enrichment entity linking finished for %s", current.rel_path)
            current = self._manifest.get(entry.rel_path)
            if current is None:
                return

        if current.state == FileState.ENTITY_LINKED:
            logger.info("Deep enrichment hierarchy started for %s", current.rel_path)
            self._core.build_hierarchy_file(current.file_id, current.file_type)
            self._manifest.update_state(current.rel_path, FileState.HIERARCHY_READY)
            logger.info("Deep enrichment hierarchy finished for %s", current.rel_path)
            current = self._manifest.get(entry.rel_path)
            if current is None:
                return

        if current.state == FileState.HIERARCHY_READY:
            self._manifest.update_state(current.rel_path, FileState.ENRICHED)
            logger.info("Deep enrichment completed for %s", current.rel_path)

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
