# fitz_sage/engines/fitz_krag/ingestion/pipeline.py
"""
KRAG Ingestion Pipeline.

Scans source files, extracts symbols via language strategies, generates
LLM summaries in batches, embeds summaries, and stores everything in
PostgreSQL.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fitz_sage.engines.fitz_krag.ingestion.import_graph_store import ImportGraphStore
from fitz_sage.engines.fitz_krag.ingestion.raw_file_store import RawFileStore
from fitz_sage.engines.fitz_krag.ingestion.schema import ensure_schema
from fitz_sage.engines.fitz_krag.ingestion.section_store import SectionStore
from fitz_sage.engines.fitz_krag.ingestion.strategies.base import IngestResult, SymbolEntry
from fitz_sage.engines.fitz_krag.ingestion.strategies.python_code import (
    PythonCodeIngestStrategy,
)
from fitz_sage.engines.fitz_krag.ingestion.strategies.technical_doc import (
    DOC_EXTENSIONS,
    DocIngestResult,
    SectionEntry,
    TechnicalDocIngestStrategy,
)
from fitz_sage.engines.fitz_krag.ingestion.symbol_store import SymbolStore
from fitz_sage.engines.fitz_krag.ingestion.table_store import TableStore

if TYPE_CHECKING:
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.llm.providers.base import ChatProvider
    from fitz_sage.storage.postgres import PostgresConnectionManager
    from fitz_sage.tabular.store.postgres import PostgresTableStore

logger = logging.getLogger(__name__)

# Extensions handled by code strategies
EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "typescript",
    ".jsx": "typescript",
    ".java": "java",
    ".go": "go",
}


class KragIngestPipeline:
    """
    Full ingestion pipeline for KRAG engine.

    Steps:
    1. Scan source for files
    2. Compute content hashes, diff against stored hashes
    3. For new/changed files: extract -> summarize -> enrich -> embed -> store
    4. For deleted files: cascade delete
    """

    def __init__(
        self,
        config: "FitzKragConfig",
        chat: "ChatProvider",
        connection_manager: "PostgresConnectionManager",
        collection: str,
        table_store: "TableStore | None" = None,
        pg_table_store: "PostgresTableStore | None" = None,
        vocabulary_store: Any = None,
        entity_graph_store: Any = None,
    ):
        self._config = config
        self._chat = chat
        self._cm = connection_manager
        self._collection = collection

        # Stores
        self._raw_store = RawFileStore(connection_manager, collection)
        self._symbol_store = SymbolStore(connection_manager, collection)
        self._import_store = ImportGraphStore(connection_manager, collection)
        self._section_store = SectionStore(connection_manager, collection)
        self._table_store = table_store or TableStore(connection_manager, collection)
        self._pg_table_store = pg_table_store
        self._vocabulary_store = vocabulary_store
        self._entity_graph_store = entity_graph_store

        # Enricher
        self._enricher: Any = None
        if config.enable_enrichment:
            from fitz_sage.engines.fitz_krag.ingestion.enricher import KragEnricher

            self._enricher = KragEnricher(chat, batch_size=config.summary_batch_size)

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

        # Schema setup. fitz-sage uses no dense embeddings; the legacy
        # vector columns are created idle and stay NULL.
        from fitz_sage.engines.fitz_krag.engine import _LEGACY_VECTOR_DIM

        ensure_schema(connection_manager, collection, _LEGACY_VECTOR_DIM)

    def ingest(
        self,
        source: Path,
        force: bool = False,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, Any]:
        """
        Run the full ingestion pipeline.

        Args:
            source: Path to source directory or single file
            force: If True, re-ingest all files regardless of hash state
            on_progress: Optional callback(current, total, file_path) for progress

        Returns:
            Stats dict: files_scanned, files_new, files_changed, files_deleted,
                        symbols_extracted, symbols_embedded
        """
        source = Path(source)
        stats = {
            "files_scanned": 0,
            "files_new": 0,
            "files_changed": 0,
            "files_deleted": 0,
            "symbols_extracted": 0,
            "symbols_embedded": 0,
            "sections_extracted": 0,
            "sections_embedded": 0,
            "collection": self._collection,
        }

        # 1. Scan files
        file_paths = self._scan_files(source)
        stats["files_scanned"] = len(file_paths)

        # 2. Compute hashes and diff
        existing_hashes = self._raw_store.list_hashes()
        existing_ids = self._raw_store.list_ids_by_path()
        current_paths: set[str] = set()

        new_files: list[tuple[str, Path]] = []  # (relative_path, absolute_path)
        changed_files: list[tuple[str, Path]] = []

        for abs_path in file_paths:
            rel_path = self._relative_path(abs_path, source)
            current_paths.add(rel_path)

            if force:
                new_files.append((rel_path, abs_path))
                continue

            content_hash = _hash_file(abs_path)

            if rel_path not in existing_hashes:
                new_files.append((rel_path, abs_path))
            elif existing_hashes[rel_path] != content_hash:
                changed_files.append((rel_path, abs_path))

        table_extensions = set(self._config.table_extensions)

        # 3. Process new/changed files
        all_symbols: list[SymbolEntry] = []
        all_symbol_file_ids: list[str] = []
        all_import_edges: list[dict[str, Any]] = []
        all_sections: list[SectionEntry] = []
        all_section_file_ids: list[str] = []
        all_table_metas: list[dict[str, Any]] = []

        files_to_process = new_files + changed_files
        total_files = len(files_to_process)

        for i, (rel_path, abs_path) in enumerate(files_to_process):
            if on_progress:
                on_progress(i + 1, total_files, rel_path)

            file_id = existing_ids.get(rel_path, str(uuid.uuid4()))
            ext = abs_path.suffix.lower()

            if ext in EXTENSION_MAP:
                result = self._process_code_file(rel_path, abs_path, file_id)
                if result:
                    symbols, import_edges = result
                    all_symbols.extend(symbols)
                    all_symbol_file_ids.extend([file_id] * len(symbols))
                    all_import_edges.extend(import_edges)
            elif ext in table_extensions:
                table_meta = self._process_table_file(rel_path, abs_path, file_id)
                if table_meta:
                    all_table_metas.append(table_meta)
            elif ext in DOC_EXTENSIONS:
                sections = self._process_doc_file(rel_path, abs_path, file_id)
                if sections:
                    all_sections.extend(sections)
                    all_section_file_ids.extend([file_id] * len(sections))

        stats["files_new"] = len(new_files)
        stats["files_changed"] = len(changed_files)
        stats["symbols_extracted"] = len(all_symbols)
        stats["sections_extracted"] = len(all_sections)

        # 4a. Batch summarize + embed symbols
        if all_symbols:
            summaries = self._summarize_symbols(all_symbols)
            vectors = self._embed_summaries(summaries)

            # Store symbols with summaries and vectors
            symbol_dicts = []
            for i, sym in enumerate(all_symbols):
                symbol_dicts.append(
                    {
                        "id": str(uuid.uuid4()),
                        "name": sym.name,
                        "qualified_name": sym.qualified_name,
                        "kind": sym.kind,
                        "raw_file_id": all_symbol_file_ids[i],
                        "start_line": sym.start_line,
                        "end_line": sym.end_line,
                        "signature": sym.signature,
                        "summary": summaries[i] if i < len(summaries) else None,
                        "summary_vector": vectors[i] if i < len(vectors) else None,
                        "imports": sym.imports,
                        "references": sym.references,
                        "keywords": [],
                        "entities": [],
                        "metadata": {},
                    }
                )

            # Enrich symbols with keywords + entities
            if self._enricher:
                self._enricher.enrich_symbols(symbol_dicts)

            self._symbol_store.upsert_batch(symbol_dicts)
            stats["symbols_embedded"] = len(vectors)

            # Save keywords to VocabularyStore
            if self._vocabulary_store:
                self._save_keywords_to_vocabulary(symbol_dicts, [])

            # Populate entity graph
            if self._entity_graph_store:
                self._populate_entity_graph(symbol_dicts, "symbol_id")

        # Store import edges
        if all_import_edges:
            self._import_store.upsert_batch(all_import_edges)

        # 4b. Batch summarize + embed sections
        if all_sections:
            section_summaries = self._summarize_sections(all_sections)
            section_vectors = self._embed_summaries(section_summaries)

            section_dicts = []
            for i, sec in enumerate(all_sections):
                section_dicts.append(
                    {
                        "id": str(uuid.uuid4()),
                        "raw_file_id": all_section_file_ids[i],
                        "title": sec.title,
                        "level": sec.level,
                        "page_start": sec.page_start,
                        "page_end": sec.page_end,
                        "content": sec.content,
                        "summary": (section_summaries[i] if i < len(section_summaries) else None),
                        "summary_vector": (
                            section_vectors[i] if i < len(section_vectors) else None
                        ),
                        "parent_section_id": sec.parent_id,
                        "position": sec.position,
                        "keywords": [],
                        "entities": [],
                        "metadata": sec.metadata,
                    }
                )
            _resolve_section_parents(section_dicts, all_section_file_ids)

            # Enrich sections with keywords + entities
            if self._enricher:
                self._enricher.enrich_sections(section_dicts)

            self._section_store.upsert_batch(section_dicts)
            stats["sections_embedded"] = len(section_vectors)

            # Save section keywords to VocabularyStore
            if self._vocabulary_store:
                self._save_keywords_to_vocabulary([], section_dicts)

            # Populate entity graph for sections
            if self._entity_graph_store:
                self._populate_entity_graph(section_dicts, "section_id")

        # 4b-2. Hierarchical summaries (L1 groups + L2 corpus)
        if self._config.enable_hierarchy:
            if all_symbols:
                self._generate_hierarchy_symbols(symbol_dicts, all_symbol_file_ids)
            if all_sections:
                self._generate_hierarchy_sections(section_dicts, all_section_file_ids)

        # 4c. Batch summarize + embed tables
        if all_table_metas:
            table_summaries = self._summarize_tables(all_table_metas)
            table_vectors = self._embed_summaries(table_summaries)

            for i, meta in enumerate(all_table_metas):
                meta["summary"] = table_summaries[i] if i < len(table_summaries) else None
                meta["summary_vector"] = table_vectors[i] if i < len(table_vectors) else None
            self._table_store.upsert_batch(all_table_metas)
            stats["tables_ingested"] = len(all_table_metas)

        # 4d. Resolve import target_file_ids now that all files are stored
        all_path_to_id = self._raw_store.list_ids_by_path()
        resolved = self._import_store.resolve_targets(all_path_to_id)
        stats["imports_resolved"] = resolved

        # 5. Delete removed files
        deleted_paths = set(existing_hashes.keys()) - current_paths
        for del_path in deleted_paths:
            if del_path in existing_ids:
                file_id = existing_ids[del_path]
                # Clean up table data for deleted files
                table_records = self._table_store.get_by_file(file_id)
                for rec in table_records:
                    if self._pg_table_store:
                        self._pg_table_store.delete(rec["table_id"])
                self._table_store.delete_by_file(file_id)
                self._raw_store.delete(file_id)
        stats["files_deleted"] = len(deleted_paths)

        logger.info(
            f"KRAG ingest complete: {stats['files_scanned']} scanned, "
            f"{stats['files_new']} new, {stats['files_changed']} changed, "
            f"{stats['symbols_extracted']} symbols, "
            f"{stats['sections_extracted']} sections"
        )
        return stats

    def _scan_files(self, source: Path) -> list[Path]:
        """Scan source for files matching enabled code + document + table strategies."""
        extensions = set()
        for lang in self._config.code_languages:
            for ext, lang_name in EXTENSION_MAP.items():
                if lang_name == lang:
                    extensions.add(ext)

        # Include document extensions
        extensions.update(DOC_EXTENSIONS)

        # Include table extensions
        extensions.update(self._config.table_extensions)

        if source.is_file():
            if source.suffix.lower() in extensions:
                return [source]
            return []

        files = []
        for ext in extensions:
            files.extend(source.rglob(f"*{ext}"))

        # Filter out common non-source directories
        skip_dirs = {".git", ".venv", "venv", "__pycache__", "node_modules", ".tox", ".eggs"}
        return [f for f in sorted(files) if not any(part in skip_dirs for part in f.parts)]

    def _relative_path(self, abs_path: Path, source: Path) -> str:
        """Get relative path string."""
        try:
            return str(abs_path.relative_to(source)).replace("\\", "/")
        except ValueError:
            return str(abs_path).replace("\\", "/")

    def _process_code_file(
        self, rel_path: str, abs_path: Path, file_id: str
    ) -> tuple[list[SymbolEntry], list[dict[str, Any]]] | None:
        """Process a code file: store raw + extract symbols/imports."""
        ext = abs_path.suffix.lower()
        lang = EXTENSION_MAP.get(ext)
        if not lang or lang not in self._strategies:
            return None

        try:
            content = abs_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Cannot read {abs_path}: {e}")
            return None

        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Store raw file
        self._raw_store.upsert(
            file_id=file_id,
            path=rel_path,
            content=content,
            content_hash=content_hash,
            file_type=ext,
            size_bytes=len(content.encode()),
        )

        # Extract symbols
        strategy = self._strategies[lang]
        result: IngestResult = strategy.extract(content, rel_path)

        # Delete old symbols for this file (will be replaced)
        self._symbol_store.delete_by_file(file_id)
        self._import_store.delete_by_file(file_id)

        # Build import edges
        import_edges = [
            {
                "source_file_id": file_id,
                "target_module": imp.target_module,
                "target_file_id": None,  # Resolved later if needed
                "import_names": imp.import_names,
            }
            for imp in result.imports
        ]

        return result.symbols, import_edges

    # Binary document formats where read_text() produces garbled output.
    # For these, we hash the raw bytes and store Docling-extracted text instead.
    _BINARY_DOC_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx"}

    def _process_doc_file(
        self, rel_path: str, abs_path: Path, file_id: str
    ) -> list[SectionEntry] | None:
        """Process a document file: parse + store raw + extract sections."""
        ext = abs_path.suffix.lower()
        is_binary = ext in self._BINARY_DOC_EXTENSIONS

        # Hash raw bytes for stable change detection
        content_hash = _hash_file(abs_path)

        # Parse document to get structured elements
        parsed_doc = self._parse_document(abs_path)
        if not parsed_doc:
            return None

        if is_binary:
            # For binary formats, store parsed text instead of garbled read_text()
            content = "\n\n".join(el.content for el in parsed_doc.elements if el.content)
        else:
            try:
                content = abs_path.read_text(encoding="utf-8", errors="replace")
                content = content.replace("\x00", "")
            except Exception as e:
                logger.warning(f"Cannot read {abs_path}: {e}")
                return None

        # Store raw file
        self._raw_store.upsert(
            file_id=file_id,
            path=rel_path,
            content=content,
            content_hash=content_hash,
            file_type=ext,
            size_bytes=abs_path.stat().st_size,
        )

        # Extract sections from parsed document
        result: DocIngestResult = self._doc_strategy.extract(parsed_doc, rel_path)
        if not result.sections:
            return None

        # Delete old sections for this file (will be replaced)
        self._section_store.delete_by_file(file_id)

        return result.sections

    def _parse_document(self, abs_path: Path) -> Any:
        """Parse a document file using the ingestion parser router."""
        try:
            from fitz_sage.ingestion.parser.router import ParserRouter
            from fitz_sage.ingestion.source.base import SourceFile

            router = ParserRouter(docling_parser=self._config.parser)

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
            vision_base_url = (
                self._config.vision_base_url or self._config.chat_base_url
            )
            vision_config: dict[str, Any] | None = None
            if self._config.vision and self._config.vision.startswith("endpoint/"):
                vision_config = {}
                if vision_base_url:
                    vision_config["base_url"] = vision_base_url
                if self._config.vision_api_key_env:
                    vision_config["auth"] = {
                        "api_key_env": self._config.vision_api_key_env
                    }
                if not vision_config:
                    vision_config = None

            vision_client = get_vision(self._config.vision, vision_config)
            if vision_client:
                for parser in router._parsers.values():
                    if hasattr(parser, "vision_client"):
                        parser.vision_client = vision_client
                        if hasattr(parser, "_vision_client_loaded"):
                            parser._vision_client_loaded = True
                        break
        except Exception as e:
            logger.warning(f"Failed to inject vision client: {e}")

    def _summarize_symbols(self, symbols: list[SymbolEntry]) -> list[str]:
        """Generate 1-2 sentence summaries for symbols, batched."""
        summaries: list[str] = []
        batch_size = self._config.summary_batch_size

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            prompt = self._build_summary_prompt(batch)

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
                batch_summaries = self._parse_summary_response(response, len(batch))
            except Exception as e:
                logger.warning(f"Summary generation failed for batch: {e}")
                batch_summaries = [f"{sym.kind} {sym.name}" for sym in batch]
            summaries.extend(batch_summaries)

        return summaries

    def _build_summary_prompt(self, batch: list[SymbolEntry]) -> str:
        """Build prompt for batch summarization."""
        parts = []
        for i, sym in enumerate(batch):
            # Truncate source to avoid token overflow
            source = sym.source[:500] if sym.source else "(no source)"
            parts.append(
                f"Symbol {i + 1}: {sym.kind} '{sym.qualified_name}'\n"
                f"Signature: {sym.signature or 'N/A'}\n"
                f"Source:\n```\n{source}\n```"
            )
        return "\n\n".join(parts)

    def _parse_summary_response(self, response: str, expected_count: int) -> list[str]:
        """Parse LLM response into list of summary strings."""
        # Try JSON array first
        try:
            # Extract JSON from possible markdown code block
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                text = text.rsplit("```", 1)[0]
            parsed = json.loads(text)
            if isinstance(parsed, list) and len(parsed) >= expected_count:
                return [str(s) for s in parsed[:expected_count]]
        except (json.JSONDecodeError, IndexError):
            pass

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

    def _summarize_sections(self, sections: list[SectionEntry]) -> list[str]:
        """Generate 1-2 sentence summaries for document sections, batched."""
        summaries: list[str] = []
        batch_size = self._config.summary_batch_size

        for i in range(0, len(sections), batch_size):
            batch = sections[i : i + batch_size]
            prompt = self._build_section_summary_prompt(batch)

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
                batch_summaries = self._parse_summary_response(response, len(batch))
            except Exception as e:
                logger.warning(f"Section summary generation failed for batch: {e}")
                batch_summaries = [sec.title for sec in batch]
            summaries.extend(batch_summaries)

        return summaries

    def _build_section_summary_prompt(self, batch: list[SectionEntry]) -> str:
        """Build prompt for section batch summarization."""
        parts = []
        for i, sec in enumerate(batch):
            # Truncate content to avoid token overflow
            content = sec.content[:800] if sec.content else "(no content)"
            parts.append(
                f"Section {i + 1}: '{sec.title}' (level {sec.level})\n" f"Content:\n{content}"
            )
        return "\n\n".join(parts)

    def _process_table_file(
        self, rel_path: str, abs_path: Path, file_id: str
    ) -> dict[str, Any] | None:
        """Process a table file: store raw preview + store in PostgresTableStore."""
        try:
            from fitz_sage.tabular.parser.csv_parser import get_sample_rows, parse_csv

            parsed = parse_csv(abs_path)
        except Exception as e:
            logger.warning(f"CSV parsing failed for {abs_path}: {e}")
            return None

        # Store raw file with first 50 lines as preview
        try:
            content = abs_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Cannot read {abs_path}: {e}")
            return None

        preview_lines = content.splitlines()[:50]
        preview = "\n".join(preview_lines)
        ext = abs_path.suffix.lower()
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        self._raw_store.upsert(
            file_id=file_id,
            path=rel_path,
            content=preview,
            content_hash=content_hash,
            file_type=ext,
            size_bytes=len(content.encode()),
        )

        # Store in shared PostgresTableStore
        if self._pg_table_store:
            try:
                self._pg_table_store.store(
                    table_id=parsed.table_id,
                    columns=parsed.columns,
                    rows=parsed.rows,
                    source_file=rel_path,
                    file_hash=content_hash,
                )
            except Exception as e:
                logger.warning(f"PostgresTableStore.store failed for {rel_path}: {e}")
                return None

        # Delete old table metadata for this file
        self._table_store.delete_by_file(file_id)

        # Build human-readable name from filename
        name = abs_path.stem.replace("_", " ").replace("-", " ").title()

        # Get sample rows for summary prompt
        try:
            samples = get_sample_rows(parsed, n=3)
        except Exception:
            samples = []

        return {
            "id": str(uuid.uuid4()),
            "raw_file_id": file_id,
            "table_id": parsed.table_id,
            "name": name,
            "columns": parsed.columns,
            "row_count": parsed.row_count,
            "metadata": {"source_file": rel_path, "sample_rows": samples},
        }

    def _summarize_tables(self, table_metas: list[dict[str, Any]]) -> list[str]:
        """Generate schema descriptions for tables, batched."""
        summaries: list[str] = []
        batch_size = self._config.summary_batch_size

        for i in range(0, len(table_metas), batch_size):
            batch = table_metas[i : i + batch_size]
            prompt = self._build_table_summary_prompt(batch)

            try:
                response = self._chat.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You describe table schemas. For each table, write a concise "
                                "1-2 sentence description of what data it contains and what "
                                "questions it could answer. Return a JSON array of strings, "
                                "one per table, in the same order."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ]
                )
                batch_summaries = self._parse_summary_response(response, len(batch))
            except Exception as e:
                logger.warning(f"Table summary generation failed for batch: {e}")
                batch_summaries = [
                    f"Table {m['name']} with columns: {', '.join(m['columns'][:10])}" for m in batch
                ]
            summaries.extend(batch_summaries)

        return summaries

    def _build_table_summary_prompt(self, batch: list[dict[str, Any]]) -> str:
        """Build prompt for table batch summarization."""
        parts = []
        for i, meta in enumerate(batch):
            cols = ", ".join(meta["columns"][:20])
            samples = meta.get("metadata", {}).get("sample_rows", [])
            sample_str = ""
            if samples:
                sample_lines = []
                for row in samples[:2]:
                    pairs = [f"{col}={val}" for col, val in zip(meta["columns"], row) if val]
                    sample_lines.append(" | ".join(pairs[:8]))
                sample_str = "\nSample rows:\n" + "\n".join(sample_lines)
            parts.append(
                f"Table {i + 1}: '{meta['name']}'\n"
                f"Columns: {cols}\n"
                f"Row count: {meta['row_count']}"
                f"{sample_str}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _embed_summaries(summaries: list[str]) -> list[list[float]]:
        """fitz-sage uses no dense embeddings; this returns []."""
        return []

    # ------------------------------------------------------------------
    # Vocabulary integration
    # ------------------------------------------------------------------

    def _save_keywords_to_vocabulary(
        self,
        symbol_dicts: list[dict[str, Any]],
        section_dicts: list[dict[str, Any]],
    ) -> None:
        """Collect keywords from enriched dicts and save to VocabularyStore."""
        try:
            from fitz_sage.retrieval.vocabulary.models import Keyword

            keywords: list[Keyword] = []
            seen: set[str] = set()

            for item_list in [symbol_dicts, section_dicts]:
                for item in item_list:
                    for kw_str in item.get("keywords", []):
                        kw_lower = kw_str.lower()
                        if kw_lower not in seen:
                            seen.add(kw_lower)
                            keywords.append(
                                Keyword(
                                    id=kw_str,
                                    category="auto",
                                    match=[kw_str],
                                    occurrences=1,
                                    auto_generated=[kw_str],
                                )
                            )

            if keywords:
                self._vocabulary_store.merge_and_save(keywords, source_docs=len(symbol_dicts))
                logger.debug(f"Saved {len(keywords)} keywords to vocabulary store")
        except Exception as e:
            logger.warning(f"Failed to save keywords to vocabulary: {e}")

    # ------------------------------------------------------------------
    # Entity graph integration
    # ------------------------------------------------------------------

    def _populate_entity_graph(self, item_dicts: list[dict[str, Any]], id_field: str) -> None:
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

    def _generate_hierarchy_symbols(
        self, symbol_dicts: list[dict[str, Any]], file_ids: list[str]
    ) -> None:
        """Generate L1 file-level summaries for symbols."""
        try:
            groups: dict[str, list[dict]] = {}
            for i, sym in enumerate(symbol_dicts):
                fid = file_ids[i] if i < len(file_ids) else sym.get("raw_file_id", "")
                groups.setdefault(fid, []).append(sym)

            l1_summaries: list[str] = []
            for file_id, symbols in groups.items():
                names = [s.get("name", "") for s in symbols[:10]]
                summaries = [s.get("summary", "") for s in symbols[:10] if s.get("summary")]
                content = "\n".join(f"- {n}: {s}" for n, s in zip(names, summaries) if s)
                if not content:
                    continue

                try:
                    group_summary = self._chat.chat(
                        [
                            {
                                "role": "system",
                                "content": (
                                    "Summarize this group of code symbols in 2-3 sentences. "
                                    "Focus on what this file/module does overall."
                                ),
                            },
                            {"role": "user", "content": content},
                        ]
                    )
                    l1_summaries.append(group_summary)
                    for sym in symbols:
                        meta = sym.get("metadata", {})
                        meta["hierarchy_summary"] = group_summary
                        sym["metadata"] = meta
                except Exception as e:
                    logger.debug(f"L1 summary failed for file group: {e}")

            # L2 corpus summary
            if l1_summaries:
                self._generate_corpus_summary(l1_summaries, "code")
        except Exception as e:
            logger.warning(f"Hierarchy generation for symbols failed: {e}")

    def _generate_hierarchy_sections(
        self, section_dicts: list[dict[str, Any]], file_ids: list[str]
    ) -> None:
        """Generate L1 file-level summaries for sections."""
        try:
            groups: dict[str, list[dict]] = {}
            for i, sec in enumerate(section_dicts):
                fid = file_ids[i] if i < len(file_ids) else sec.get("raw_file_id", "")
                groups.setdefault(fid, []).append(sec)

            l1_summaries: list[str] = []
            for file_id, sections in groups.items():
                titles = [s.get("title", "") for s in sections[:10]]
                summaries = [s.get("summary", "") for s in sections[:10] if s.get("summary")]
                content = "\n".join(f"- {t}: {s}" for t, s in zip(titles, summaries) if s)
                if not content:
                    continue

                try:
                    group_summary = self._chat.chat(
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
                    l1_summaries.append(group_summary)
                    for sec in sections:
                        meta = sec.get("metadata", {})
                        meta["hierarchy_summary"] = group_summary
                        sec["metadata"] = meta
                except Exception as e:
                    logger.debug(f"L1 summary failed for section group: {e}")

            if l1_summaries:
                self._generate_corpus_summary(l1_summaries, "document")
        except Exception as e:
            logger.warning(f"Hierarchy generation for sections failed: {e}")

    def _generate_corpus_summary(self, l1_summaries: list[str], kind: str) -> None:
        """Generate L2 corpus-level summary from L1 summaries."""
        try:
            content = "\n".join(f"- {s}" for s in l1_summaries[:20])
            corpus_summary = self._chat.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            f"Summarize this collection of {kind} modules in 3-5 sentences. "
                            "Describe the overall system architecture and purpose."
                        ),
                    },
                    {"role": "user", "content": content},
                ]
            )
            logger.debug(f"L2 corpus summary ({kind}): {corpus_summary[:100]}...")
        except Exception as e:
            logger.warning(f"L2 corpus summary failed: {e}")


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
    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest()
