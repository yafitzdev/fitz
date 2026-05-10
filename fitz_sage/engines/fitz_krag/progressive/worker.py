# fitz_sage/engines/fitz_krag/progressive/worker.py
"""
BackgroundIngestWorker — daemon thread that indexes files progressively.

State machine per file:
    REGISTERED → PARSED     (store raw content, extract symbols/sections — no LLM)
    PARSED     → SUMMARIZED (generate LLM summaries, pause during active queries)
    SUMMARIZED → EMBEDDED   (compute embeddings, runs concurrently with queries)

Priority queue (stdlib PriorityQueue):
    P1: Files user just queried about
    P2: Files in same directory as queried files
    P3: Small files (<10KB, quick wins)
    P4: Remaining files by size ascending
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fitz_sage.engines.fitz_krag.progressive.manifest import FileState

if TYPE_CHECKING:
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.engines.fitz_krag.progressive.manifest import FileManifest, ManifestEntry
    from fitz_sage.llm.providers.base import ChatProvider, EmbeddingProvider
    from fitz_sage.storage.postgres import PostgresConnectionManager

logger = logging.getLogger(__name__)

_CODE_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".java", ".go"}
_TABLE_EXTENSIONS = {".csv", ".tsv"}


class BackgroundIngestWorker:
    """Daemon thread that indexes files: REGISTERED → PARSED → SUMMARIZED → EMBEDDED."""

    def __init__(
        self,
        manifest: "FileManifest",
        source_dir: Path,
        config: "FitzKragConfig",
        chat: "ChatProvider",
        embedder: "EmbeddingProvider | None",
        connection_manager: "PostgresConnectionManager",
        collection: str,
        stores: dict[str, Any],
        vocabulary_store: Any = None,
        entity_graph_store: Any = None,
        pg_table_store: Any = None,
        enricher: Any = None,
    ) -> None:
        self._manifest = manifest
        self._source_dir = source_dir
        self._config = config
        self._chat = chat
        self._embedder = embedder
        self._cm = connection_manager
        self._collection = collection
        self._raw_store = stores["raw"]
        self._symbol_store = stores["symbol"]
        self._import_store = stores["import"]
        self._section_store = stores["section"]
        self._table_store = stores["table"]
        self._vocabulary_store = vocabulary_store
        self._entity_graph_store = entity_graph_store
        self._pg_table_store = pg_table_store
        self._enricher = enricher

        # Parsed text cache dir — same location builder uses
        # Access manifest's private _path (same package, internal use)
        self._cache_dir = manifest._path.parent / "parsed" if hasattr(manifest, "_path") else None

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
        entries = self._manifest.entries()
        for rp, entry in entries.items():
            if entry.state == FileState.EMBEDDED:
                continue
            parent = str(Path(rp).parent)
            if parent in dirs and rp not in rel_paths:
                siblings.append(rp)
        if siblings:
            self._manifest.bump_priority_level(siblings, level=2)

    def _run(self) -> None:
        """Main worker loop."""
        try:
            # Phase 1: REGISTERED → PARSED (fast, no LLM)
            self._process_registered_files()

            # Phase 2: PARSED → SUMMARIZED (LLM, pauses during queries)
            self._process_parsed_files()

            # Phase 2.5: Enrich SUMMARIZED files with keywords + entities (LLM)
            self._enrich_summarized_files()

            # Phase 3: SUMMARIZED → EMBEDDED (embedding API, concurrent)
            self._process_summarized_files()

            logger.info("Background indexing complete")
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

    def _process_registered_files(self) -> None:
        """REGISTERED → PARSED: Store raw content + extract symbols/sections."""
        from fitz_sage.engines.fitz_krag.ingestion.strategies.python_code import (
            PythonCodeIngestStrategy,
        )
        from fitz_sage.engines.fitz_krag.ingestion.strategies.technical_doc import (
            DOC_EXTENSIONS,
            TechnicalDocIngestStrategy,
        )

        py_strategy = PythonCodeIngestStrategy()
        doc_strategy = TechnicalDocIngestStrategy()

        # Lazy-init non-Python strategies (tree-sitter may not be installed)
        ts_strategy = None
        java_strategy = None
        go_strategy = None

        for entry in self._get_ordered_files(FileState.REGISTERED):
            if self._stop_event.is_set():
                return

            try:
                content = self._read_file(entry)
                if content is None:
                    continue

                content_hash = hashlib.sha256(content.encode()).hexdigest()
                ext = entry.file_type

                # Store raw file
                self._raw_store.upsert(
                    file_id=entry.file_id,
                    path=entry.rel_path,
                    content=content,
                    content_hash=content_hash,
                    file_type=ext,
                    size_bytes=entry.size_bytes,
                )

                # Extract symbols for code files
                result = None

                if ext == ".py":
                    result = py_strategy.extract(content, entry.rel_path)

                elif ext in {".ts", ".tsx", ".js", ".jsx"}:
                    try:
                        if ts_strategy is None:
                            from fitz_sage.engines.fitz_krag.ingestion.strategies.typescript import (
                                TypeScriptIngestStrategy,
                            )

                            ts_strategy = TypeScriptIngestStrategy()
                        result = ts_strategy.extract(content, entry.rel_path)
                    except Exception as e:
                        logger.warning(f"TypeScript strategy unavailable: {e}")

                elif ext == ".java":
                    try:
                        if java_strategy is None:
                            from fitz_sage.engines.fitz_krag.ingestion.strategies.java import (
                                JavaIngestStrategy,
                            )

                            java_strategy = JavaIngestStrategy()
                        result = java_strategy.extract(content, entry.rel_path)
                    except Exception as e:
                        logger.warning(f"Java strategy unavailable: {e}")

                elif ext == ".go":
                    try:
                        if go_strategy is None:
                            from fitz_sage.engines.fitz_krag.ingestion.strategies.go import (
                                GoIngestStrategy,
                            )

                            go_strategy = GoIngestStrategy()
                        result = go_strategy.extract(content, entry.rel_path)
                    except Exception as e:
                        logger.warning(f"Go strategy unavailable: {e}")

                elif ext in _TABLE_EXTENSIONS:
                    self._process_table_file(entry, content, content_hash)

                elif ext in DOC_EXTENSIONS or ext == "":
                    self._process_doc_sections(entry, content, doc_strategy)

                # Store symbols and imports from code extraction
                if result is not None:
                    if result.symbols:
                        symbol_dicts = []
                        for sym in result.symbols:
                            symbol_dicts.append(
                                {
                                    "id": str(uuid.uuid4()),
                                    "name": sym.name,
                                    "qualified_name": sym.qualified_name,
                                    "kind": sym.kind,
                                    "raw_file_id": entry.file_id,
                                    "start_line": sym.start_line,
                                    "end_line": sym.end_line,
                                    "signature": sym.signature,
                                    "summary": None,
                                    "summary_vector": None,
                                    "imports": sym.imports,
                                    "references": sym.references,
                                    "keywords": [],
                                    "entities": [],
                                    "metadata": {},
                                }
                            )
                        self._symbol_store.upsert_batch(symbol_dicts)

                        # Embed symbol signatures immediately (no LLM needed)
                        self._embed_symbols_from_signatures(entry.file_id, symbol_dicts)

                    if result.imports:
                        import_edges = [
                            {
                                "source_file_id": entry.file_id,
                                "target_module": imp.target_module,
                                "target_file_id": None,
                                "import_names": imp.import_names,
                            }
                            for imp in result.imports
                        ]
                        self._import_store.upsert_batch(import_edges)

                self._manifest.update_state(entry.rel_path, FileState.PARSED)

            except Exception as e:
                logger.warning(f"Background parse failed for {entry.rel_path}: {e}")

        self._manifest.save()

    def _process_doc_sections(
        self, entry: "ManifestEntry", content: str, doc_strategy: Any
    ) -> None:
        """Extract and store document sections for a parsed doc file."""
        try:
            from fitz_sage.ingestion.parser.router import ParserRouter
            from fitz_sage.ingestion.source.base import SourceFile

            abs_path = Path(entry.abs_path)
            router = ParserRouter(docling_parser=self._config.parser)
            source_file = SourceFile(uri=f"file://{abs_path}", local_path=abs_path)
            parsed_doc = router.parse(source_file)
            if not parsed_doc:
                return

            result = doc_strategy.extract(parsed_doc, entry.rel_path)
            if not result.sections:
                return

            section_dicts = []
            for sec in result.sections:
                section_dicts.append(
                    {
                        "id": str(uuid.uuid4()),
                        "raw_file_id": entry.file_id,
                        "title": sec.title,
                        "level": sec.level,
                        "page_start": sec.page_start,
                        "page_end": sec.page_end,
                        "content": sec.content,
                        "summary": None,
                        "summary_vector": None,
                        "parent_section_id": None,
                        "position": sec.position,
                        "keywords": [],
                        "entities": [],
                        "metadata": sec.metadata,
                    }
                )
            self._section_store.upsert_batch(section_dicts)

            # Embed content immediately so vector search works before LLM
            # summaries are generated. Phase 3 will upgrade these with
            # summary-based embeddings if summaries become available.
            self._embed_sections_from_content(entry.file_id, section_dicts)

        except Exception as e:
            logger.debug(f"Doc section extraction failed for {entry.rel_path}: {e}")

    def _process_table_file(self, entry: "ManifestEntry", content: str, content_hash: str) -> None:
        """Parse CSV, store rows in PostgresTableStore, store metadata in TableStore."""
        from fitz_sage.tabular.parser.csv_parser import get_sample_rows, parse_csv

        abs_path = Path(entry.abs_path)
        try:
            parsed = parse_csv(abs_path)
        except Exception as e:
            logger.warning(f"CSV parsing failed for {entry.rel_path}: {e}")
            return

        # Store raw preview (first 50 lines) for ContentReader
        preview_lines = content.splitlines()[:50]
        self._raw_store.upsert(
            file_id=entry.file_id,
            path=entry.rel_path,
            content="\n".join(preview_lines),
            content_hash=content_hash,
            file_type=entry.file_type,
            size_bytes=entry.size_bytes,
        )

        # Store all rows in PostgresTableStore for SQL queries
        if self._pg_table_store:
            try:
                self._pg_table_store.store(
                    table_id=parsed.table_id,
                    columns=parsed.columns,
                    rows=parsed.rows,
                    source_file=entry.rel_path,
                    file_hash=content_hash,
                )
            except Exception as e:
                logger.warning(f"PostgresTableStore.store failed for {entry.rel_path}: {e}")
                return

        # Delete old table metadata for this file
        self._table_store.delete_by_file(entry.file_id)

        # Build human-readable name from filename
        name = Path(entry.rel_path).stem.replace("_", " ").replace("-", " ").title()

        # Get sample rows for summarization prompt
        try:
            samples = get_sample_rows(parsed, n=3)
        except Exception:
            samples = []

        # Store metadata in krag_table_index
        self._table_store.upsert_batch(
            [
                {
                    "id": str(uuid.uuid4()),
                    "raw_file_id": entry.file_id,
                    "table_id": parsed.table_id,
                    "name": name,
                    "columns": parsed.columns,
                    "row_count": parsed.row_count,
                    "summary": None,
                    "summary_vector": None,
                    "metadata": {"source_file": entry.rel_path, "sample_rows": samples},
                }
            ]
        )

    def _summarize_table(self, entry: "ManifestEntry") -> None:
        """Generate LLM summary for table schema."""
        records = self._table_store.get_by_file(entry.file_id)
        if not records:
            return

        for record in records:
            cols = ", ".join(record["columns"][:20])
            samples = record.get("metadata", {}).get("sample_rows", [])
            sample_str = ""
            if samples:
                sample_lines = []
                for row in samples[:2]:
                    pairs = [f"{col}={val}" for col, val in zip(record["columns"], row) if val]
                    sample_lines.append(" | ".join(pairs[:8]))
                sample_str = "\nSample rows:\n" + "\n".join(sample_lines)

            prompt = (
                f"Table: '{record['name']}'\n"
                f"Columns: {cols}\n"
                f"Row count: {record['row_count']}"
                f"{sample_str}"
            )

            try:
                response = self._chat.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You describe table schemas. Write a concise 1-2 sentence "
                                "description of what data this table contains and what "
                                "questions it could answer. Return ONLY the description text."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ]
                )
                summary = response.strip()
            except Exception as e:
                logger.warning(f"Table summary generation failed for {entry.rel_path}: {e}")
                summary = f"Table {record['name']} with columns: {cols}"

            self._table_store.update_summary(record["id"], summary)

    def _embed_table(self, entry: "ManifestEntry") -> None:
        """Compute and store embeddings for table summaries."""
        records = self._table_store.get_by_file(entry.file_id)
        if not records:
            return

        if self._embedder is None:
            # chat-only retrieval mode: skip vector generation entirely.
            return

        texts = []
        record_ids = []
        for record in records:
            text = record.get("summary") or (
                f"Table {record['name']} with columns: {', '.join(record['columns'][:20])}"
            )
            texts.append(text)
            record_ids.append(record["id"])

        try:
            vectors = self._embedder.embed_batch(texts, task_type="document")
            for record_id, vector in zip(record_ids, vectors):
                self._table_store.update_vector(record_id, vector)
        except Exception as e:
            logger.warning(f"Table embedding failed for {entry.rel_path}: {e}")

    def _process_parsed_files(self) -> None:
        """PARSED → SUMMARIZED: Generate LLM summaries."""
        # Collect symbols needing summaries
        for entry in self._get_ordered_files(FileState.PARSED):
            if self._stop_event.is_set():
                return

            # Wait while query is active (LLM priority)
            while self._query_active.is_set() and not self._stop_event.is_set():
                self._stop_event.wait(timeout=0.5)
            if self._stop_event.is_set():
                return

            try:
                ext = entry.file_type

                if ext in _CODE_EXTENSIONS:
                    self._summarize_file_symbols(entry)
                elif ext in _TABLE_EXTENSIONS:
                    self._summarize_table(entry)
                else:
                    self._summarize_file_sections(entry)

                self._manifest.update_state(entry.rel_path, FileState.SUMMARIZED)

            except Exception as e:
                logger.warning(f"Background summarize failed for {entry.rel_path}: {e}")

        self._manifest.save()

    def _summarize_file_symbols(self, entry: "ManifestEntry") -> None:
        """Generate summaries for all symbols in a file."""
        from fitz_sage.engines.fitz_krag.ingestion.strategies.base import SymbolEntry

        # Read file content to get symbol source code
        content = self._read_file(entry)
        if not content:
            return

        lines = content.splitlines()

        # Build SymbolEntry objects with source code
        symbols = []
        for sym in entry.symbols:
            source = "\n".join(lines[max(0, sym.start_line - 1) : sym.end_line])
            symbols.append(
                SymbolEntry(
                    name=sym.name,
                    qualified_name=sym.qualified_name,
                    kind=sym.kind,
                    start_line=sym.start_line,
                    end_line=sym.end_line,
                    signature=sym.signature,
                    source=source[:500],
                )
            )

        if not symbols:
            return

        # Generate summaries
        summaries = self._batch_summarize_symbols(symbols)

        # Update symbol store with summaries
        self._symbol_store.update_summaries_by_file(entry.file_id, summaries)

    def _summarize_file_sections(self, entry: "ManifestEntry") -> None:
        """Generate summaries for all sections in a file."""
        sections = self._section_store.get_by_file(entry.file_id)
        if not sections:
            return

        # Build section summaries
        batch: list[dict[str, str]] = []
        for sec in sections:
            content_preview = (sec.get("content") or "")[:800]
            batch.append(
                {
                    "title": sec.get("title", ""),
                    "level": sec.get("level", 1),
                    "content": content_preview,
                }
            )

        summaries = self._batch_summarize_sections(batch)

        # Update section store with summaries
        self._section_store.update_summaries_by_file(entry.file_id, summaries)

    def _batch_summarize_symbols(self, symbols: list) -> list[str]:
        """Generate summaries for a batch of symbols."""
        parts = []
        for i, sym in enumerate(symbols):
            source = sym.source[:500] if sym.source else "(no source)"
            parts.append(
                f"Symbol {i + 1}: {sym.kind} '{sym.qualified_name}'\n"
                f"Signature: {sym.signature or 'N/A'}\n"
                f"Source:\n```\n{source}\n```"
            )
        prompt = "\n\n".join(parts)

        try:
            response = self._chat.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You summarize code symbols. For each symbol, write a concise "
                            "1-2 sentence description of what it does. Return a JSON array "
                            "of strings, one per symbol, in the same order."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            return self._parse_summary_response(response, len(symbols))
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            return [f"{sym.kind} {sym.name}" for sym in symbols]

    def _batch_summarize_sections(self, sections: list[dict]) -> list[str]:
        """Generate summaries for a batch of sections."""
        parts = []
        for i, sec in enumerate(sections):
            content = sec.get("content", "")[:800]
            parts.append(
                f"Section {i + 1}: '{sec['title']}' (level {sec['level']})\n" f"Content:\n{content}"
            )
        prompt = "\n\n".join(parts)

        try:
            response = self._chat.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You summarize document sections. For each section, write a "
                            "concise 1-2 sentence description of its content. Return a "
                            "JSON array of strings, one per section, in the same order."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            return self._parse_summary_response(response, len(sections))
        except Exception as e:
            logger.warning(f"Section summary generation failed: {e}")
            return [sec.get("title", "(untitled)") for sec in sections]

    def _parse_summary_response(self, response: str, expected_count: int) -> list[str]:
        """Parse LLM response into list of summary strings."""
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list) and len(parsed) >= expected_count:
                return [str(s) for s in parsed[:expected_count]]
        except (json.JSONDecodeError, IndexError):
            pass

        # Fallback: split by lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return (lines + ["(no summary)"] * expected_count)[:expected_count]

    def _process_summarized_files(self) -> None:
        """SUMMARIZED → EMBEDDED: Compute embeddings and update vector fields."""
        for entry in self._get_ordered_files(FileState.SUMMARIZED):
            if self._stop_event.is_set():
                return

            try:
                ext = entry.file_type

                if ext in _CODE_EXTENSIONS:
                    self._embed_file_symbols(entry)
                elif ext in _TABLE_EXTENSIONS:
                    self._embed_table(entry)
                else:
                    self._embed_file_sections(entry)

                self._manifest.update_state(entry.rel_path, FileState.EMBEDDED)

            except Exception as e:
                logger.warning(f"Background embed failed for {entry.rel_path}: {e}")

        self._manifest.save()

    def _embed_file_symbols(self, entry: "ManifestEntry") -> None:
        """Compute and store embeddings for symbols in a file."""
        if self._embedder is None:
            # chat-only retrieval mode: vector columns stay NULL.
            return

        summaries = self._symbol_store.get_summaries_by_file(entry.file_id)
        if not summaries:
            return

        texts = [s.get("summary") or f"{s.get('kind', '')} {s.get('name', '')}" for s in summaries]
        try:
            vectors = self._embedder.embed_batch(texts, task_type="document")
            self._symbol_store.update_vectors_by_file(entry.file_id, vectors)
        except Exception as e:
            logger.warning(f"Embedding failed for {entry.rel_path}: {e}")

    def _embed_symbols_from_signatures(self, file_id: str, symbol_dicts: list[dict]) -> None:
        """Embed symbol signatures immediately during Phase 1 (no LLM needed).

        Uses kind + qualified_name + signature to produce a vector that
        enables vector search before LLM summaries are generated.
        """
        if self._embedder is None:
            return
        if not symbol_dicts:
            return
        texts = [
            f"{s['kind']} {s['qualified_name']} {s.get('signature') or ''}" for s in symbol_dicts
        ]
        try:
            vectors = self._embedder.embed_batch(texts, task_type="document")
            self._symbol_store.update_vectors_by_file(file_id, vectors)
            logger.debug(f"Embedded {len(vectors)} symbols from signatures for file {file_id[:8]}")
        except Exception as e:
            logger.debug(f"Signature embedding skipped for {file_id[:8]}: {e}")

    def _embed_sections_from_content(self, file_id: str, section_dicts: list[dict]) -> None:
        """Embed section content immediately during Phase 1 (no LLM needed).

        Uses title + first 2000 chars of content to produce a vector that
        enables vector search before LLM summaries are generated. Phase 3
        will upgrade these with summary-based embeddings when available.
        """
        if self._embedder is None:
            return
        if not section_dicts:
            return
        texts = [f"{s['title']}. {(s.get('content') or '')[:2000]}" for s in section_dicts]
        try:
            vectors = self._embedder.embed_batch(texts, task_type="document")
            self._section_store.update_vectors_by_file(file_id, vectors)
            logger.debug(f"Embedded {len(vectors)} sections from content for file {file_id[:8]}")
        except Exception as e:
            logger.debug(f"Content embedding skipped for {file_id[:8]}: {e}")

    def _embed_file_sections(self, entry: "ManifestEntry") -> None:
        """Compute and store embeddings for sections in a file."""
        if self._embedder is None:
            return

        sections = self._section_store.get_by_file(entry.file_id)
        if not sections:
            return

        texts = [s.get("summary") or s.get("title", "(untitled)") for s in sections]
        try:
            vectors = self._embedder.embed_batch(texts, task_type="document")
            self._section_store.update_vectors_by_file(entry.file_id, vectors)
        except Exception as e:
            logger.warning(f"Section embedding failed for {entry.rel_path}: {e}")

    # ------------------------------------------------------------------
    # Enrichment (between summarize and embed)
    # ------------------------------------------------------------------

    def _enrich_summarized_files(self) -> None:
        """Enrich SUMMARIZED files with keywords + entities via LLM."""
        if not self._enricher:
            return

        for entry in self._get_ordered_files(FileState.SUMMARIZED):
            if self._stop_event.is_set():
                return

            # Pause during active queries (LLM priority)
            while self._query_active.is_set() and not self._stop_event.is_set():
                self._stop_event.wait(timeout=0.5)
            if self._stop_event.is_set():
                return

            try:
                ext = entry.file_type
                if ext in _CODE_EXTENSIONS:
                    self._enrich_file_symbols(entry)
                elif ext not in _TABLE_EXTENSIONS:
                    self._enrich_file_sections(entry)
            except Exception as e:
                logger.warning(f"Background enrichment failed for {entry.rel_path}: {e}")

    def _enrich_file_symbols(self, entry: "ManifestEntry") -> None:
        """Enrich symbols with keywords + entities."""
        symbols = self._symbol_store.get_by_file(entry.file_id)
        if not symbols:
            return
        self._enricher.enrich_symbols(symbols)
        self._symbol_store.update_enrichment_by_file(entry.file_id, symbols)

    def _enrich_file_sections(self, entry: "ManifestEntry") -> None:
        """Enrich sections with keywords + entities."""
        sections = self._section_store.get_by_file(entry.file_id)
        if not sections:
            return
        self._enricher.enrich_sections(sections)
        self._section_store.update_enrichment_by_file(entry.file_id, sections)

    # ------------------------------------------------------------------
    # File reading
    # ------------------------------------------------------------------

    def _read_file(self, entry: "ManifestEntry") -> str | None:
        """Read file content from disk, using parsed cache for rich docs."""
        try:
            path = Path(entry.abs_path)
            if not path.exists():
                path = self._source_dir / entry.rel_path
            if not path.exists():
                return None

            ext = path.suffix.lower()
            if ext in {".pdf", ".docx", ".pptx", ".html", ".htm"}:
                # Use parsed cache — builder already populated it during point()
                if self._cache_dir and entry.content_hash:
                    from fitz_sage.engines.fitz_krag.progressive.parsed_cache import (
                        get_parsed_text,
                    )

                    text = get_parsed_text(path, entry.content_hash, self._cache_dir)
                    if text:
                        return text.replace("\x00", "")
                    return None
                # No cache available — fall through to direct parse
                from fitz_sage.ingestion.parser import ParserRouter
                from fitz_sage.ingestion.source.base import SourceFile

                source_file = SourceFile(uri=path.as_uri(), local_path=path)
                parsed = ParserRouter().parse(source_file)
                text = parsed.full_text
                if text:
                    return text.replace("\x00", "")
                return None

            content = path.read_text(encoding="utf-8", errors="replace")
            # Strip NUL bytes for PostgreSQL compatibility
            return content.replace("\x00", "")
        except Exception as e:
            logger.debug(f"Cannot read {entry.rel_path}: {e}")
            return None
