# fitz_sage/engines/fitz_krag/ingestion/pipeline.py
# fitz_sage/engines/fitz_krag/ingestion/pipeline.py
"""
KRAG ingestion core.

The single ingestion implementation for the KRAG engine, structured as
composable operations:

- per file — ``parse_file`` (extract symbols / sections / tables, store
  raw content; no LLM), ``summarize_file`` (provider summaries on demand),
  ``keyword_file`` (minimum retrieval keywords), ``link_entities_file``
  (entity graph), ``build_hierarchy_file`` (L1 hierarchy summary), and
  ``enrich_file`` (blocking whole-file enrichment)
- corpus — ``finalize`` (resolve the import graph, build the L2 hierarchy
  summary)

``ingest()`` is a thin synchronous loop over these ops for blocking
whole-corpus ingestion. The progressive ``BackgroundIngestWorker``
schedules the same ops file-by-file on a background thread.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fitz_sage.core import ConfigurationError
from fitz_sage.core.json_utils import parse_llm_json
from fitz_sage.engines.fitz_krag.ingestion.formats import (
    BINARY_DOCUMENT_EXTENSIONS,
    CODE_EXTENSION_MAP,
    enabled_extensions,
)
from fitz_sage.engines.fitz_krag.ingestion.import_graph_store import ImportGraphStore
from fitz_sage.engines.fitz_krag.ingestion.raw_file_store import RawFileStore
from fitz_sage.engines.fitz_krag.ingestion.schema import ensure_schema
from fitz_sage.engines.fitz_krag.ingestion.section_store import (
    CORPUS_SUMMARY_SCHEMA_VERSION,
    SectionStore,
)
from fitz_sage.engines.fitz_krag.ingestion.strategies.base import IngestResult
from fitz_sage.engines.fitz_krag.ingestion.strategies.python_code import (
    PythonCodeIngestStrategy,
)
from fitz_sage.engines.fitz_krag.ingestion.strategies.technical_doc import (
    DocIngestResult,
    TechnicalDocIngestStrategy,
)
from fitz_sage.engines.fitz_krag.ingestion.symbol_store import SymbolStore, symbol_entry_to_dict
from fitz_sage.engines.fitz_krag.ingestion.table_store import TableStore
from fitz_sage.ingestion.hashing import compute_bytes_hash, compute_content_hash

if TYPE_CHECKING:
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.llm.providers.base import ChatProvider
    from fitz_sage.storage.sqlite import SqliteConnectionManager
    from fitz_sage.tabular.store.sqlite import SqliteTableStore

logger = logging.getLogger(__name__)

# Backwards-compatible alias for internal callers.
EXTENSION_MAP = CODE_EXTENSION_MAP

# Synthetic raw-file / section that carries the L2 corpus summary. Fixed IDs
# so re-ingest upserts in place rather than accumulating duplicates.
_CORPUS_FILE_ID = "__krag_corpus__"
_CORPUS_FILE_PATH = "__corpus_summary__"
_CORPUS_SECTION_ID = "__krag_corpus_summary__"


class KragIngestPipeline:
    """
    Ingestion core for the KRAG engine.

    Exposes per-file operations (``parse_file`` → ``summarize_file`` →
    ``enrich_file``) and a corpus ``finalize`` step. ``ingest()`` drives them
    synchronously over a whole directory; the background worker schedules
    them incrementally.
    """

    def __init__(
        self,
        config: "FitzKragConfig",
        chat: "ChatProvider | None",
        connection_manager: "SqliteConnectionManager",
        collection: str,
        table_store: "TableStore | None" = None,
        sqlite_table_store: "SqliteTableStore | None" = None,
        entity_graph_store: Any = None,
        enricher_chat: "ChatProvider | None" = None,
        summarizer_chat: "ChatProvider | None" = None,
    ):
        self._config = config
        self._chat = chat
        standard_chat = enricher_chat or summarizer_chat or chat
        if standard_chat is None:
            from fitz_sage.llm.providers.onnx_chat import OnnxChat

            standard_chat = OnnxChat()
        self._enricher_chat = enricher_chat or standard_chat
        self._summarizer_chat = summarizer_chat or standard_chat
        self._cm = connection_manager
        self._collection = collection
        self._table_extensions = set(config.table_extensions)

        # Stores
        self._raw_store = RawFileStore(connection_manager, collection)
        self._symbol_store = SymbolStore(connection_manager, collection)
        self._import_store = ImportGraphStore(connection_manager, collection)
        self._section_store = SectionStore(connection_manager, collection)
        self._table_store = table_store or TableStore(connection_manager, collection)
        self._sqlite_table_store = sqlite_table_store
        self._entity_graph_store = entity_graph_store

        # Enricher
        from fitz_sage.engines.fitz_krag.ingestion.enricher import KragEnricher

        # The managed Qwen path is most reliable when each enrichment
        # response contains one compact JSON object. Summary batching remains
        # controlled by config.summary_batch_size.
        self._enricher: Any = KragEnricher(
            self._enricher_chat,
            batch_size=1,
        )

        # Code strategies
        self._strategies: dict[str, Any] = {}
        if "python" in config.code_languages:
            self._strategies["python"] = PythonCodeIngestStrategy()

        if "typescript" in config.code_languages:
            try:
                from fitz_sage.engines.fitz_krag.ingestion.strategies.typescript import (
                    TypeScriptIngestStrategy,
                )

                self._strategies["typescript"] = TypeScriptIngestStrategy()
            except ImportError:
                logger.debug("tree-sitter-typescript not installed, skipping TypeScript support")

        if "java" in config.code_languages:
            try:
                from fitz_sage.engines.fitz_krag.ingestion.strategies.java import (
                    JavaIngestStrategy,
                )

                self._strategies["java"] = JavaIngestStrategy()
            except ImportError:
                logger.debug("tree-sitter-java not installed, skipping Java support")

        if "go" in config.code_languages:
            try:
                from fitz_sage.engines.fitz_krag.ingestion.strategies.go import (
                    GoIngestStrategy,
                )

                self._strategies["go"] = GoIngestStrategy()
            except ImportError:
                logger.debug("tree-sitter-go not installed, skipping Go support")

        # Document strategy
        self._doc_strategy = TechnicalDocIngestStrategy()

        ensure_schema(connection_manager, collection)

    # ------------------------------------------------------------------
    # Per-file operations — parse / summarize / enrich
    # ------------------------------------------------------------------

    def parse_file(self, rel_path: str, abs_path: Path, file_id: str) -> dict[str, int]:
        """Parse one file: store raw content + extract symbols/sections/tables.

        No LLM calls. Routes by extension. This is the single parse
        implementation shared by the synchronous ``ingest()`` loop and the
        progressive background worker.

        Returns:
            Counts dict: ``symbols``, ``sections``, ``tables``.
        """
        abs_path = Path(abs_path)
        ext = abs_path.suffix.lower()
        counts = {"symbols": 0, "sections": 0, "tables": 0}

        if ext in EXTENSION_MAP:
            counts["symbols"] = self._parse_code_file(rel_path, abs_path, file_id)
        elif ext in self._table_extensions:
            counts["tables"] = self._parse_table_file(rel_path, abs_path, file_id)
        else:
            counts["sections"] = self._parse_doc_file(rel_path, abs_path, file_id)

        if not any(counts.values()):
            raise ValueError(
                f"No searchable content could be extracted from supported file '{rel_path}'."
            )
        return counts

    def summarize_file(self, file_id: str, file_type: str) -> None:
        """Generate LLM summaries for one file's sections or table schema.

        Code symbols carry no summary — code files are a no-op here.
        """
        if file_type in self._table_extensions:
            self._require_summarizer()
            self._summarize_table_file(file_id)
        elif file_type not in EXTENSION_MAP:
            self._require_summarizer()
            self._summarize_doc_file(file_id)

    def keyword_file(self, file_id: str, file_type: str) -> None:
        """Extract the minimum keyword index needed for query-ready retrieval."""
        if file_type in EXTENSION_MAP:
            self._keyword_code_file(file_id)
        elif file_type not in self._table_extensions:
            self._keyword_doc_file(file_id)

    def link_entities_file(self, file_id: str, file_type: str) -> None:
        """Extract entities and populate the entity graph for one file."""
        if file_type in EXTENSION_MAP:
            self._link_code_entities_file(file_id)
        elif file_type not in self._table_extensions:
            self._link_doc_entities_file(file_id)

    def build_hierarchy_file(self, file_id: str, file_type: str) -> None:
        """Generate file-level hierarchy summaries for one document file."""
        if file_type not in EXTENSION_MAP and file_type not in self._table_extensions:
            self._build_doc_hierarchy_file(file_id)

    def enrich_file(self, file_id: str, file_type: str) -> None:
        """Extract keywords/entities for one file and feed downstream stores.

        Populates the vocabulary store and entity graph (incremental, per
        file) and — for document files — the L1 hierarchy summary stored on
        each section's metadata. Enrichment is part of the ingestion contract;
        model runtime failures are surfaced instead of silently weakening the
        retrieval index. Table files are summarized separately and are not
        entity-enriched.
        """
        if file_type in EXTENSION_MAP:
            self._enrich_code_file(file_id)
        elif file_type not in self._table_extensions:
            self._enrich_doc_file(file_id)

    # ------------------------------------------------------------------
    # Corpus operations — finalize
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """Corpus-level steps, run once after every file has been processed.

        Resolves the import graph and builds the L2 hierarchy summary.
        Re-runs wholesale on re-ingest (incremental hierarchy is a v2 concern).
        """
        self.resolve_imports()
        self._require_summarizer()
        self._build_corpus_summary()

    def resolve_imports(self) -> int:
        """Resolve import-graph ``target_file_id``s now that all files exist."""
        try:
            return self._import_store.resolve_targets(self._raw_store.list_ids_by_path())
        except Exception as e:
            logger.debug(f"Import target resolution failed: {e}")
            return 0

    def delete_files_not_in_paths(self, current_paths: set[str]) -> int:
        """Delete stored files that are no longer part of the current source manifest."""
        existing_ids = self._raw_store.list_ids_by_path()
        stale_paths = set(existing_ids) - set(current_paths) - {_CORPUS_FILE_PATH}
        for stale_path in stale_paths:
            self._delete_file(existing_ids[stale_path])
        return len(stale_paths)

    def discard_file(self, file_id: str) -> None:
        """Remove stored retrieval units after a file fails to re-index."""
        self._delete_file(file_id)

    # ------------------------------------------------------------------
    # Synchronous whole-corpus ingest — a thin loop over the core ops
    # ------------------------------------------------------------------

    def ingest(
        self,
        source: Path,
        force: bool = False,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, Any]:
        """
        Run a blocking whole-corpus ingest over ``source``.

        Scans + diffs against stored hashes, then drives the core ops:
        parse → summarize → enrich per file, delete removed files, finalize.

        Args:
            source: Path to source directory or single file
            force: If True, re-ingest all files regardless of hash state
            on_progress: Optional callback(current, total, file_path)

        Returns:
            Stats dict: files_scanned, files_new, files_changed, files_deleted,
                        symbols_extracted, sections_extracted, tables_ingested
        """
        source = Path(source)
        stats: dict[str, Any] = {
            "files_scanned": 0,
            "files_new": 0,
            "files_changed": 0,
            "files_deleted": 0,
            "symbols_extracted": 0,
            "sections_extracted": 0,
            "tables_ingested": 0,
            "files_failed": 0,
            "failures": [],
            "collection": self._collection,
        }

        # 1. Scan + diff against stored hashes
        file_paths = self._scan_files(source)
        stats["files_scanned"] = len(file_paths)

        existing_hashes = self._raw_store.list_hashes()
        existing_ids = self._raw_store.list_ids_by_path()
        current_paths: set[str] = set()
        to_process: list[tuple[str, Path, str]] = []

        for abs_path in file_paths:
            rel_path = self._relative_path(abs_path, source)
            current_paths.add(rel_path)
            file_id = existing_ids.get(rel_path, str(uuid.uuid4()))

            if force:
                to_process.append((rel_path, abs_path, file_id))
                stats["files_new"] += 1
                continue

            content_hash = _hash_file(abs_path)
            if rel_path not in existing_hashes:
                to_process.append((rel_path, abs_path, file_id))
                stats["files_new"] += 1
            elif existing_hashes[rel_path] != content_hash:
                to_process.append((rel_path, abs_path, file_id))
                stats["files_changed"] += 1

        total = len(to_process)

        # 2. Parse every file (no LLM)
        parsed_files: list[tuple[str, Path, str]] = []
        for i, (rel_path, abs_path, file_id) in enumerate(to_process):
            if on_progress:
                on_progress(i + 1, total, rel_path)
            try:
                counts = self.parse_file(rel_path, abs_path, file_id)
            except Exception as exc:
                self.discard_file(file_id)
                stats["files_failed"] += 1
                stats["failures"].append({"path": rel_path, "error": str(exc)})
                logger.warning("Failed to index %s: %s", rel_path, exc)
                continue
            stats["symbols_extracted"] += counts["symbols"]
            stats["sections_extracted"] += counts["sections"]
            stats["tables_ingested"] += counts["tables"]
            parsed_files.append((rel_path, abs_path, file_id))

        # 3. Summarize, then 4. enrich (LLM)
        for rel_path, abs_path, file_id in parsed_files:
            self.summarize_file(file_id, abs_path.suffix.lower())
        for rel_path, abs_path, file_id in parsed_files:
            self.enrich_file(file_id, abs_path.suffix.lower())

        # 5. Delete removed files (the synthetic corpus file is not scanned)
        stats["files_deleted"] = self.delete_files_not_in_paths(current_paths)

        # 6. Corpus finalize — import graph + L2 hierarchy summary
        self.finalize()

        logger.info(
            f"KRAG ingest complete: {stats['files_scanned']} scanned, "
            f"{stats['files_new']} new, {stats['files_changed']} changed, "
            f"{stats['symbols_extracted']} symbols, "
            f"{stats['sections_extracted']} sections"
        )
        return stats

    # ------------------------------------------------------------------
    # Parse helpers
    # ------------------------------------------------------------------

    def _parse_code_file(self, rel_path: str, abs_path: Path, file_id: str) -> int:
        """Store raw content + extract/store symbols and imports. Returns symbol count."""
        ext = abs_path.suffix.lower()
        lang = EXTENSION_MAP.get(ext)
        if not lang or lang not in self._strategies:
            return 0

        try:
            content = abs_path.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
        except Exception as e:
            logger.warning(f"Cannot read {abs_path}: {e}")
            return 0

        content_hash = compute_bytes_hash(content.encode())
        self._raw_store.upsert(
            file_id=file_id,
            path=rel_path,
            content=content,
            content_hash=content_hash,
            file_type=ext,
            size_bytes=len(content.encode()),
        )

        result: IngestResult = self._strategies[lang].extract(content, rel_path)

        # Replace any prior symbols/imports for this file
        self._symbol_store.delete_by_file(file_id)
        self._import_store.delete_by_file(file_id)

        if result.symbols:
            symbol_dicts = [symbol_entry_to_dict(sym, file_id) for sym in result.symbols]
            self._symbol_store.upsert_batch(symbol_dicts)

        if result.imports:
            self._import_store.upsert_batch(
                [
                    {
                        "source_file_id": file_id,
                        "target_module": imp.target_module,
                        "target_file_id": None,
                        "import_names": imp.import_names,
                    }
                    for imp in result.imports
                ]
            )

        return len(result.symbols)

    def _parse_doc_file(self, rel_path: str, abs_path: Path, file_id: str) -> int:
        """Parse a document: store raw content + extract/store sections. Returns section count."""
        ext = abs_path.suffix.lower()
        is_binary = ext in BINARY_DOCUMENT_EXTENSIONS

        content_hash = _hash_file(abs_path)

        parsed_doc = self._parse_document(abs_path)
        if not parsed_doc:
            return 0

        if is_binary:
            content = "\n\n".join(el.content for el in parsed_doc.elements if el.content)
        else:
            try:
                content = abs_path.read_text(encoding="utf-8", errors="replace")
                content = content.replace("\x00", "")
            except Exception as e:
                logger.warning(f"Cannot read {abs_path}: {e}")
                return 0

        self._raw_store.upsert(
            file_id=file_id,
            path=rel_path,
            content=content,
            content_hash=content_hash,
            file_type=ext,
            size_bytes=abs_path.stat().st_size,
        )

        result: DocIngestResult = self._doc_strategy.extract(parsed_doc, rel_path)
        if not result.sections:
            return 0

        # Replace any prior sections for this file
        self._section_store.delete_by_file(file_id)

        section_dicts: list[dict[str, Any]] = []
        for sec in result.sections:
            section_dicts.append(
                {
                    "id": str(uuid.uuid4()),
                    "raw_file_id": file_id,
                    "title": sec.title,
                    "level": sec.level,
                    "page_start": sec.page_start,
                    "page_end": sec.page_end,
                    "content": sec.content,
                    "summary": None,
                    "parent_section_id": sec.parent_id,
                    "position": sec.position,
                    "keywords": [],
                    "entities": [],
                    "metadata": sec.metadata,
                }
            )
        _resolve_section_parents(section_dicts, [file_id] * len(section_dicts))
        self._section_store.upsert_batch(section_dicts)

        return len(section_dicts)

    def _parse_table_file(self, rel_path: str, abs_path: Path, file_id: str) -> int:
        """Parse a table file: store raw preview + table rows + metadata. Returns table count."""
        try:
            from fitz_sage.tabular.parser.csv_parser import get_sample_rows, parse_csv

            parsed = parse_csv(abs_path)
        except Exception as e:
            logger.warning(f"CSV parsing failed for {abs_path}: {e}")
            return 0

        try:
            content = abs_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Cannot read {abs_path}: {e}")
            return 0

        preview = "\n".join(content.splitlines()[:50])
        ext = abs_path.suffix.lower()
        content_hash = compute_bytes_hash(content.encode())

        self._raw_store.upsert(
            file_id=file_id,
            path=rel_path,
            content=preview,
            content_hash=content_hash,
            file_type=ext,
            size_bytes=len(content.encode()),
        )

        if self._sqlite_table_store:
            try:
                self._sqlite_table_store.store(
                    table_id=parsed.table_id,
                    columns=parsed.columns,
                    rows=parsed.rows,
                    source_file=rel_path,
                    file_hash=content_hash,
                )
            except Exception as e:
                logger.warning(f"SqliteTableStore.store failed for {rel_path}: {e}")
                return 0

        # Replace any prior table metadata for this file
        self._table_store.delete_by_file(file_id)

        name = abs_path.stem.replace("_", " ").replace("-", " ").title()
        try:
            samples = get_sample_rows(parsed, n=3)
        except Exception:
            samples = []

        self._table_store.upsert_batch(
            [
                {
                    "id": str(uuid.uuid4()),
                    "raw_file_id": file_id,
                    "table_id": parsed.table_id,
                    "name": name,
                    "columns": parsed.columns,
                    "row_count": parsed.row_count,
                    "summary": None,
                    "metadata": {"source_file": rel_path, "sample_rows": samples},
                }
            ]
        )
        return 1

    def _delete_file(self, file_id: str) -> None:
        """Cascade-delete a removed file's stored data."""
        for rec in self._table_store.get_by_file(file_id):
            if self._sqlite_table_store:
                self._sqlite_table_store.delete(rec["table_id"])
        self._table_store.delete_by_file(file_id)
        # raw_files delete cascades to symbols / imports / sections via FK
        self._raw_store.delete(file_id)

    # ------------------------------------------------------------------
    # Summarize helpers
    # ------------------------------------------------------------------

    def _summarize_doc_file(self, file_id: str) -> None:
        """Generate 1-2 sentence summaries for all sections in a document file."""
        sections = self._section_store.get_by_file(file_id)
        if not sections:
            return
        summaries = self._summarize_section_dicts(sections)
        self._section_store.update_summaries_by_file(file_id, summaries)

    def _summarize_section_dicts(self, sections: list[dict[str, Any]]) -> list[str]:
        """Generate section summaries, batched, preserving input order."""
        summaries: list[str] = []
        batch_size = self._config.summary_batch_size

        for i in range(0, len(sections), batch_size):
            batch = sections[i : i + batch_size]
            prompt = self._build_section_summary_prompt(batch)

            try:
                response = self._summarizer_chat.chat(
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
                batch_summaries = self._parse_summary_response(response, len(batch))
            except Exception as e:
                logger.warning(f"Section summary generation failed for batch: {e}")
                batch_summaries = [sec.get("title") or "(untitled)" for sec in batch]
            summaries.extend(batch_summaries)

        return summaries

    @staticmethod
    def _build_section_summary_prompt(batch: list[dict[str, Any]]) -> str:
        """Build the prompt for a batch of section dicts."""
        parts = []
        for i, sec in enumerate(batch):
            content = (sec.get("content") or "")[:800] or "(no content)"
            parts.append(
                f"Section {i + 1}: '{sec.get('title', '')}' (level {sec.get('level', 1)})\n"
                f"Content:\n{content}"
            )
        return "\n\n".join(parts)

    def _summarize_table_file(self, file_id: str) -> None:
        """Generate a schema description for each table in a table file."""
        for record in self._table_store.get_by_file(file_id):
            summary = self._summarize_table_record(record)
            self._table_store.update_summary(record["id"], summary)

    def _summarize_table_record(self, record: dict[str, Any]) -> str:
        """Generate a 1-2 sentence schema description for a single table."""
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
            response = self._summarizer_chat.chat(
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
            return response.strip()
        except Exception as e:
            logger.warning(f"Table summary generation failed: {e}")
            return f"Table {record['name']} with columns: {cols}"

    # ------------------------------------------------------------------
    # Enrich helpers
    # ------------------------------------------------------------------

    def _require_enricher(self) -> None:
        """Fail closed when keyword/entity enrichment cannot run."""
        if self._enricher and self._enricher_chat:
            return
        raise ConfigurationError(
            "Ingestion requires keyword/entity enrichment, but the managed "
            "local Qwen ONNX runtime was not initialized."
        )

    def _require_summarizer(self) -> None:
        """Fail closed when hierarchy/table summarization cannot run."""
        if self._summarizer_chat:
            return
        raise ConfigurationError(
            "Ingestion requires hierarchy summarization, but the managed "
            "local Qwen ONNX runtime was not initialized."
        )

    def _keyword_code_file(self, file_id: str) -> None:
        """Extract query-ready keywords for a code file's symbols."""
        self._require_enricher()
        symbols = self._symbol_store.get_by_file(file_id)
        if not symbols:
            return
        self._enricher.enrich_symbol_keywords(symbols)
        self._symbol_store.update_enrichment_by_file(file_id, symbols)

    def _keyword_doc_file(self, file_id: str) -> None:
        """Extract query-ready keywords for a document file's sections."""
        self._require_enricher()
        sections = self._section_store.get_by_file(file_id)
        if not sections:
            return
        self._enricher.enrich_section_keywords(sections)
        self._section_store.update_enrichment_by_file(file_id, sections)

    def _link_code_entities_file(self, file_id: str) -> None:
        """Extract entities for code symbols and update the entity graph."""
        self._require_enricher()
        symbols = self._symbol_store.get_by_file(file_id)
        if not symbols:
            return
        self._enricher.enrich_symbol_entities(symbols)
        self._symbol_store.update_enrichment_by_file(file_id, symbols)
        if self._entity_graph_store:
            self._populate_entity_graph(symbols)

    def _link_doc_entities_file(self, file_id: str) -> None:
        """Extract entities for document sections and update the entity graph."""
        self._require_enricher()
        sections = self._section_store.get_by_file(file_id)
        if not sections:
            return
        self._enricher.derive_section_entities(sections)
        self._section_store.update_enrichment_by_file(file_id, sections)
        if self._entity_graph_store:
            self._populate_entity_graph(sections)

    def _build_doc_hierarchy_file(self, file_id: str) -> None:
        """Generate and persist an L1 hierarchy summary for a document file."""
        self._require_summarizer()
        sections = self._section_store.get_by_file(file_id)
        if not sections:
            return
        self._generate_l1_summary(sections)
        self._section_store.update_enrichment_by_file(file_id, sections)

    def _enrich_code_file(self, file_id: str) -> None:
        """Enrich a code file's symbols with keywords + entities.

        Code symbols have no hierarchy stage, but keyword/entity enrichment is
        still required for the retrieval index.
        """
        self._require_enricher()
        symbols = self._symbol_store.get_by_file(file_id)
        if not symbols:
            return
        self._enricher.enrich_symbols(symbols)
        self._symbol_store.update_enrichment_by_file(file_id, symbols)
        if self._entity_graph_store:
            self._populate_entity_graph(symbols)

    def _enrich_doc_file(self, file_id: str) -> None:
        """Enrich a document file's sections with keywords + entities + L1 summary.

        Keyword/entity extraction and the L1 hierarchy summary are both required
        parts of the document retrieval index.
        """
        self._require_enricher()
        self._require_summarizer()
        sections = self._section_store.get_by_file(file_id)
        if not sections:
            return
        self._enricher.enrich_sections(sections)
        self._generate_l1_summary(sections)
        # One write persists keywords, entities, and the L1 hierarchy summary
        self._section_store.update_enrichment_by_file(file_id, sections)
        if self._entity_graph_store:
            self._populate_entity_graph(sections)

    # ------------------------------------------------------------------
    # Entity graph integration
    # ------------------------------------------------------------------

    def _populate_entity_graph(self, item_dicts: list[dict[str, Any]]) -> None:
        """Add entities from enriched items to the entity graph store."""
        try:
            for item in item_dicts:
                entities = item.get("entities", [])
                if not entities:
                    continue
                entity_tuples = []
                for e in entities:
                    if isinstance(e, dict):
                        entity_tuples.append((e.get("name", ""), e.get("type", "unknown")))
                if entity_tuples:
                    self._entity_graph_store.add_chunk_entities(item["id"], entity_tuples)
        except Exception as e:
            logger.warning(f"Failed to populate entity graph: {e}")

    # ------------------------------------------------------------------
    # Hierarchical summaries
    # ------------------------------------------------------------------

    def _generate_l1_summary(self, sections: list[dict[str, Any]]) -> None:
        """Generate one L1 group summary for a file's sections (stored in metadata).

        Document sections only. Code symbols carry their own machine-readable
        structure (imports, AST), so symbol-level hierarchy summaries are
        redundant for code.
        """
        # L1 reads raw section content, not summaries: summarization is demand-driven
        content = "\n".join(
            f"- {s.get('title', '')}: {(s.get('content') or '')[:300]}"
            for s in sections[:10]
            if s.get("content")
        )
        if not content:
            return

        try:
            group_summary = self._summarizer_chat.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "Summarize this group of document sections in 2-3 sentences. "
                            "Focus on what this document covers overall."
                        ),
                    },
                    {"role": "user", "content": content},
                ]
            )
        except Exception as e:
            logger.debug(f"L1 summary failed for section group: {e}")
            return

        for sec in sections:
            meta = sec.get("metadata") or {}
            meta["hierarchy_summary"] = group_summary
            sec["metadata"] = meta

    def _build_corpus_summary(self) -> None:
        """Roll L1 file summaries up into the L2 corpus summary and store it."""
        l1_summaries = self._section_store.get_hierarchy_summaries()
        self._delete_corpus_summary()
        if not l1_summaries:
            return
        source_signature = self._corpus_summary_source_signature(l1_summaries)
        corpus_summary = self._generate_corpus_summary(l1_summaries)
        if corpus_summary:
            self._store_corpus_summary(corpus_summary, source_signature)

    def _delete_corpus_summary(self) -> None:
        """Remove the synthetic L2 summary before regeneration."""
        self._section_store.delete_by_file(_CORPUS_FILE_ID)
        self._raw_store.delete(_CORPUS_FILE_ID)

    def _corpus_summary_source_signature(self, l1_summaries: list[str]) -> str:
        """Hash the L1 rollup inputs so stored L2 summaries are traceable."""
        normalized = "\n".join(sorted(s.strip() for s in l1_summaries if s.strip()))
        return compute_bytes_hash(normalized.encode())

    def _generate_corpus_summary(self, l1_summaries: list[str]) -> str | None:
        """Generate the L2 corpus-level summary from L1 summaries."""
        content = "\n".join(f"- {s}" for s in l1_summaries[:20])
        try:
            return self._summarizer_chat.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "Summarize this collection of document modules in 3-5 sentences. "
                            "Describe the overall system architecture and purpose."
                        ),
                    },
                    {"role": "user", "content": content},
                ]
            ).strip()
        except Exception as e:
            logger.warning(f"L2 corpus summary failed: {e}")
            return None

    def _store_corpus_summary(self, summary: str, source_signature: str) -> None:
        """Persist the L2 summary as a retrievable section under a synthetic raw file."""
        content_hash = compute_bytes_hash(summary.encode())
        self._raw_store.upsert(
            file_id=_CORPUS_FILE_ID,
            path=_CORPUS_FILE_PATH,
            content=summary,
            content_hash=content_hash,
            file_type=".md",
            size_bytes=len(summary.encode()),
        )
        self._section_store.upsert_batch(
            [
                {
                    "id": _CORPUS_SECTION_ID,
                    "raw_file_id": _CORPUS_FILE_ID,
                    "title": "Corpus Overview",
                    "level": 0,
                    "page_start": None,
                    "page_end": None,
                    "content": summary,
                    "summary": summary,
                    "parent_section_id": None,
                    "position": 0,
                    "keywords": [],
                    "entities": [],
                    "metadata": {
                        "is_corpus_summary": True,
                        "is_hierarchy_summary": True,
                        "corpus_summary_schema": CORPUS_SUMMARY_SCHEMA_VERSION,
                        "source_signature": source_signature,
                    },
                }
            ]
        )

    # ------------------------------------------------------------------
    # Document parsing
    # ------------------------------------------------------------------

    def _parse_document(self, abs_path: Path) -> Any:
        """Parse a document file using the ingestion parser router."""
        try:
            from fitz_sage.ingestion.parser.router import ParserRouter
            from fitz_sage.ingestion.source.base import SourceFile

            router = ParserRouter(parser=self._config.parser)

            # Inject vision client when using docling_vision parser
            if self._config.parser == "docling_vision" and self._config.vision:
                self._inject_vision_client(router)

            source_file = SourceFile(
                uri=f"file://{abs_path}",
                local_path=abs_path,
            )
            return router.parse(source_file)
        except Exception as e:
            logger.warning(f"Document parsing failed for {abs_path}: {e}")
            return None

    def _inject_vision_client(self, router: Any) -> None:
        """Inject vision provider into the docling_vision parser."""
        try:
            from fitz_sage.llm.client import get_vision

            # Vision falls back to chat_base_url when vision_base_url is unset.
            vision_base_url = self._config.vision_base_url or self._config.chat_base_url
            vision_config: dict[str, Any] | None = None
            if self._config.vision and self._config.vision.startswith("endpoint/"):
                vision_config = {}
                if vision_base_url:
                    vision_config["base_url"] = vision_base_url
                if self._config.vision_api_key_env:
                    vision_config["auth"] = {"api_key_env": self._config.vision_api_key_env}
                if not vision_config:
                    vision_config = None

            vision_client = get_vision(self._config.vision, vision_config)
            if vision_client:
                for parser in router._parsers.values():
                    if hasattr(parser, "vision_client"):
                        parser.vision_client = vision_client
                        break
        except Exception as e:
            logger.warning(f"Failed to inject vision client: {e}")

    def _parse_summary_response(self, response: str, expected_count: int) -> list[str]:
        """Parse an LLM response into a list of summary strings."""
        # Try JSON array first
        parsed = parse_llm_json(response, as_array=True)
        if isinstance(parsed, list) and len(parsed) >= expected_count:
            return [str(s) for s in parsed[:expected_count]]

        # Fallback: split by numbered lines
        lines = [
            line.strip()
            for line in response.strip().splitlines()
            if line.strip() and not line.strip().startswith("```")
        ]
        # Strip leading numbers like "1. " or "1: "
        cleaned = []
        for line in lines:
            for prefix_len in range(1, 4):
                if len(line) > prefix_len + 2 and line[prefix_len] in ".):":
                    line = line[prefix_len + 1 :].strip()
                    break
            cleaned.append(line)

        if len(cleaned) >= expected_count:
            return cleaned[:expected_count]

        # Pad if needed
        return cleaned + ["(no summary)"] * (expected_count - len(cleaned))

    # ------------------------------------------------------------------
    # File scanning
    # ------------------------------------------------------------------

    def _scan_files(self, source: Path) -> list[Path]:
        """Scan source for files matching enabled code + document + table strategies."""
        extensions = enabled_extensions(
            code_languages=self._config.code_languages,
            table_extensions=self._config.table_extensions,
        )

        if source.is_file():
            if source.suffix.lower() in extensions:
                return [source]
            return []

        files: list[Path] = []
        for ext in extensions:
            files.extend(source.rglob(f"*{ext}"))

        # Filter out common non-source directories
        skip_dirs = {
            ".fitz",
            ".git",
            ".venv",
            "venv",
            "__pycache__",
            "node_modules",
            ".tox",
            ".eggs",
        }
        return [f for f in sorted(files) if not any(part in skip_dirs for part in f.parts)]

    def _relative_path(self, abs_path: Path, source: Path) -> str:
        """Get relative path string."""
        try:
            return str(abs_path.relative_to(source)).replace("\\", "/")
        except ValueError:
            return str(abs_path).replace("\\", "/")


def _resolve_section_parents(section_dicts: list[dict[str, Any]], file_ids: list[str]) -> None:
    """Resolve placeholder parent IDs (_parent_N) to actual UUIDs.

    Parent indices are local to each file's section batch, so we group
    sections by file_id and resolve within each group.
    """
    # Group indices by file_id (preserving order)
    file_groups: dict[str, list[int]] = {}
    for i, fid in enumerate(file_ids):
        file_groups.setdefault(fid, []).append(i)

    for indices in file_groups.values():
        for global_idx in indices:
            parent_id = section_dicts[global_idx].get("parent_section_id")
            if not parent_id or not parent_id.startswith("_parent_"):
                continue
            try:
                local_idx = int(parent_id.removeprefix("_parent_"))
            except (ValueError, TypeError):
                section_dicts[global_idx]["parent_section_id"] = None
                continue
            # Map local index to the global index within this file group
            if 0 <= local_idx < len(indices):
                section_dicts[global_idx]["parent_section_id"] = section_dicts[indices[local_idx]][
                    "id"
                ]
            else:
                section_dicts[global_idx]["parent_section_id"] = None


def _hash_file(path: Path) -> str:
    """Compute SHA-256 hash of file content."""
    return compute_content_hash(path)
