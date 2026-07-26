# fitz_sage/engines/fitz_krag/retrieval/strategies/agentic_search.py
"""
Agentic search strategy — file selection from manifest before query-ready indexing.

For files not yet query-ready, this strategy:
1. Builds compact manifest text
2. Uses BM25 pre-filter when >50 unindexed files
3. Asks LLM to pick relevant files
4. Reads content from disk
5. Returns standard Address objects
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fitz_sage.core.json_utils import parse_llm_json
from fitz_sage.engines.fitz_krag.progressive.manifest import FileState, is_query_ready_state
from fitz_sage.engines.fitz_krag.progressive.parsed_cache import RICH_DOC_EXTENSIONS
from fitz_sage.engines.fitz_krag.types import Address, AddressKind

if TYPE_CHECKING:
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.engines.fitz_krag.progressive.manifest import (
        FileManifest,
        ManifestEntry,
    )

logger = logging.getLogger(__name__)

# Max unindexed files to send to LLM
_SKIP_LLM_THRESHOLD = 15
_BM25_PREFILTER_THRESHOLD = 50
_LLM_MAX_FILES = 10


class AgenticSearchStrategy:
    """File selection from manifest for files that are not query-ready yet."""

    def __init__(
        self,
        manifest: "FileManifest",
        source_dir: Path,
        chat_factory: Any,
        config: "FitzKragConfig",
        cache_dir: Path | None = None,
    ) -> None:
        self._manifest = manifest
        self._source_dir = source_dir
        self._chat_factory = chat_factory
        self._config = config
        self._cache_dir = cache_dir

    def has_pending_files(self) -> bool:
        """Return whether supplemental disk search has any files to inspect."""
        return any(
            entry.state not in {FileState.FAILED, FileState.UNSUPPORTED}
            and not is_query_ready_state(entry.state)
            for entry in self._manifest.entries().values()
        )

    def retrieve(self, query: str, limit: int, *, allow_llm: bool = True) -> list[Address]:
        """Retrieve addresses for not-query-ready files via file selection.

        1. Get files that are not query-ready from manifest
        2. Build compact manifest text (~50-100 tokens/file)
        3. If >50 unindexed files: BM25 pre-filter to top 50
        4. LLM picks ~5-10 candidate files
        5. Read file content from disk
        6. Create Address objects with AST line ranges from manifest
        """
        unindexed = [
            entry
            for entry in self._manifest.entries().values()
            if entry.state not in {FileState.FAILED, FileState.UNSUPPORTED}
            and not is_query_ready_state(entry.state)
        ]
        if not unindexed:
            return []

        if self._chat_factory is None:
            allow_llm = False

        # Single file: always use it — user pointed at exactly this file
        if len(unindexed) == 1:
            selected_entries = unindexed
        # Fast path: few files — skip LLM, use BM25 + path matching
        elif len(unindexed) <= _SKIP_LLM_THRESHOLD:
            path_matches = set(self._path_match_files(unindexed, query))
            scored = self._bm25_prefilter(unindexed, query)
            top: list["ManifestEntry"] = []
            seen: set[str] = set()
            # Prioritize path matches first
            for entry in scored:
                if entry.rel_path in path_matches and entry.rel_path not in seen:
                    top.append(entry)
                    seen.add(entry.rel_path)
            # Then BM25 order
            for entry in scored:
                if entry.rel_path not in seen:
                    top.append(entry)
                    seen.add(entry.rel_path)
            selected_entries = top[:_LLM_MAX_FILES]
        else:
            # BM25 pre-filter if too many files
            if not allow_llm:
                selected_entries = self._bm25_prefilter(unindexed, query)[:_LLM_MAX_FILES]
            elif len(unindexed) > _BM25_PREFILTER_THRESHOLD:
                unindexed = self._bm25_prefilter(unindexed, query)

                # Build manifest text for LLM
                manifest_text = self._manifest.to_manifest_text(unindexed)

                # LLM selects files
                selected_paths = self._llm_select_files(manifest_text, query)

                # Path-match fallback: always include files whose path contains query terms
                entries_map = {e.rel_path: e for e in unindexed}
                path_matched = self._path_match_files(unindexed, query)
                for pm in path_matched:
                    if pm not in selected_paths:
                        selected_paths.append(pm)

                if not selected_paths:
                    return []

                selected_entries = [entries_map[p] for p in selected_paths if p in entries_map]
            else:
                # Build manifest text for LLM
                manifest_text = self._manifest.to_manifest_text(unindexed)

                # LLM selects files
                selected_paths = self._llm_select_files(manifest_text, query)

                # Path-match fallback: always include files whose path contains query terms
                entries_map = {e.rel_path: e for e in unindexed}
                path_matched = self._path_match_files(unindexed, query)
                for pm in path_matched:
                    if pm not in selected_paths:
                        selected_paths.append(pm)

                if not selected_paths:
                    return []

                selected_entries = [entries_map[p] for p in selected_paths if p in entries_map]

        # Create addresses from selected files
        addresses: list[Address] = []
        for entry in selected_entries[:limit]:
            content = self._read_from_disk(entry)
            if content is None:
                continue

            entry_addresses = self._create_addresses(entry, content)
            addresses.extend(entry_addresses)

        return addresses

    def _bm25_prefilter(self, entries: list["ManifestEntry"], query: str) -> list["ManifestEntry"]:
        """In-memory token overlap scoring (pure Python, no deps).

        Tokenize query + manifest text, score by term frequency, return top 50.
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return entries[:_BM25_PREFILTER_THRESHOLD]

        query_tf = Counter(query_tokens)
        scored: list[tuple[float, "ManifestEntry"]] = []

        for entry in entries:
            doc_text = _entry_to_text(entry)
            doc_tokens = _tokenize(doc_text)
            if not doc_tokens:
                continue

            doc_tf = Counter(doc_tokens)
            doc_len = len(doc_tokens)

            # Simple BM25-like scoring
            score = 0.0
            for term, qf in query_tf.items():
                tf = doc_tf.get(term, 0)
                if tf > 0:
                    # TF saturation
                    score += qf * (tf / (tf + 1.0 + 0.5 * doc_len / 100.0))

            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:_BM25_PREFILTER_THRESHOLD]]

    def _llm_select_files(self, manifest_text: str, query: str) -> list[str]:
        """Single LLM call to select relevant files from manifest."""
        try:
            chat = self._chat_factory("fast")
            prompt = (
                "You are a file selection assistant. Given a user query and a manifest "
                "of available files, select the most relevant files.\n\n"
                f"Query: {query}\n\n"
                f"Available files:\n{manifest_text}\n\n"
                "Instructions:\n"
                f"- Select up to {_LLM_MAX_FILES} files most likely to contain the answer\n"
                "- Prefer files with relevant symbols, headings, or paths\n"
                "- Return ONLY a JSON array of file paths: "
                '["path/to/file1.py", "path/to/file2.md"]\n'
                "- If no files seem relevant, return an empty array: []"
            )
            response = chat.chat([{"role": "user", "content": prompt}])
            text = response.strip()
            parsed = parse_llm_json(text, as_array=True)
            return [str(p) for p in parsed[:_LLM_MAX_FILES] if isinstance(p, str) and p.strip()]
        except Exception as e:
            logger.warning(f"Agentic file selection failed: {e}")

        return []

    def _path_match_files(self, entries: list["ManifestEntry"], query: str) -> list[str]:
        """Return rel_paths of files whose path contains meaningful query terms."""
        # Extract non-stopword tokens from query
        tokens = _tokenize(query)
        stopwords = {
            "what",
            "is",
            "the",
            "how",
            "does",
            "do",
            "are",
            "was",
            "were",
            "can",
            "will",
            "about",
            "where",
            "when",
            "which",
            "who",
            "why",
            "show",
            "me",
            "tell",
            "find",
            "get",
            "all",
            "this",
            "that",
            "it",
        }
        terms = [t for t in tokens if t not in stopwords and len(t) > 2]
        if not terms:
            return []

        matched: list[str] = []
        for entry in entries:
            path_lower = entry.rel_path.lower()
            for term in terms:
                if term in path_lower:
                    matched.append(entry.rel_path)
                    break
        return matched[:_LLM_MAX_FILES]

    def _read_from_disk(self, entry: "ManifestEntry") -> str | None:
        """Read file content from disk, using parsed cache for rich docs."""
        try:
            path = Path(entry.abs_path)
            if not path.exists():
                path = self._source_dir / entry.rel_path
            if not path.exists():
                return None

            ext = path.suffix.lower()
            if ext in RICH_DOC_EXTENSIONS:
                from fitz_sage.engines.fitz_krag.progressive.parsed_cache import (
                    get_parsed_text,
                    parse_document_text,
                )

                if self._cache_dir:
                    return get_parsed_text(path, entry.content_hash, self._cache_dir)
                return parse_document_text(path)

            return path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.debug(f"Cannot read {entry.rel_path} from disk: {e}")
            return None

    def _create_addresses(self, entry: "ManifestEntry", content: str) -> list[Address]:
        """Create Address objects from a manifest entry.

        For code files: one SYMBOL address per symbol from manifest.
        For doc files: one SECTION address per heading, plus one FILE address.
        For other files: one FILE address.
        """
        addresses: list[Address] = []

        if entry.symbols:
            # Code file — create symbol addresses with AST line ranges
            for sym in entry.symbols:
                addresses.append(
                    Address(
                        kind=AddressKind.SYMBOL,
                        source_id=entry.file_id,
                        location=f"{entry.rel_path}:{sym.start_line}",
                        summary=f"{sym.kind} {sym.qualified_name}",
                        score=0.7,  # LLM-validated relevance
                        metadata={
                            "start_line": sym.start_line,
                            "end_line": sym.end_line,
                            "name": sym.name,
                            "kind": sym.kind,
                            "signature": sym.signature,
                            "disk_path": entry.rel_path,
                            "agentic": True,
                        },
                    )
                )
        elif entry.headings:
            # Doc file — one file-level address with heading metadata
            heading_list = ", ".join(h.title for h in entry.headings[:5])
            addresses.append(
                Address(
                    kind=AddressKind.FILE,
                    source_id=entry.file_id,
                    location=entry.rel_path,
                    summary=f"Document: {heading_list}",
                    score=0.7,
                    metadata={
                        "disk_path": entry.rel_path,
                        "agentic": True,
                        "text": content,
                    },
                )
            )
        else:
            # Generic file (includes rich docs like PDF)
            addresses.append(
                Address(
                    kind=AddressKind.FILE,
                    source_id=entry.file_id,
                    location=entry.rel_path,
                    summary=f"File: {entry.rel_path}",
                    score=0.7,
                    metadata={
                        "disk_path": entry.rel_path,
                        "agentic": True,
                        "text": content,
                    },
                )
            )

        return addresses


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    import re

    tokens = re.split(r"[\s/._\-:;,(){}[\]]+", text)
    return [re.sub(r"[?!\"']+$", "", t).lower() for t in tokens if len(t) > 1]


def _entry_to_text(entry: "ManifestEntry") -> str:
    """Build searchable text from a manifest entry for BM25."""
    parts = [entry.rel_path]
    for sym in entry.symbols:
        parts.append(sym.name)
        parts.append(sym.qualified_name)
        if sym.signature:
            parts.append(sym.signature)
    for heading in entry.headings:
        parts.append(heading.title)
    return " ".join(parts)
