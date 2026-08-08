# fitz_sage/engines/fitz_krag/retrieval/strategies/llm_code_search.py
"""
LLM-based structural code search strategy.

Sends a compact AST manifest to the LLM and lets it reason about which files
are relevant. Falls back to hybrid CodeSearchStrategy on any failure.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from fitz_sage.engines.fitz_krag.types import Address, AddressKind

if TYPE_CHECKING:
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.engines.fitz_krag.ingestion.import_graph_store import ImportGraphStore
    from fitz_sage.engines.fitz_krag.ingestion.symbol_store import SymbolStore
    from fitz_sage.engines.fitz_krag.retrieval.strategies.code_search import CodeSearchStrategy
    from fitz_sage.llm.factory import ChatFactory

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are selecting files from a codebase to answer a question.\n\n"
    "Question: {query}\n\n"
    "Below is the structural index of every indexed file. Each entry shows "
    "file path, classes, functions, and imports.\n\n"
    "{structural_index}\n\n"
    "First, generate 3-5 search terms, synonyms, and related concepts for the "
    "question. Then, using those terms, select 5-15 relevant files.\n\n"
    "Include files that:\n"
    "- Contain the code being asked about\n"
    "- Define protocols, base classes, or types used by relevant code\n"
    "- Contain configuration or factory patterns that affect the relevant code\n\n"
    "Err on the side of including MORE files — missing a relevant file is worse "
    "than including an extra one.\n\n"
    "Return JSON:\n"
    '```json\n{{"search_terms": ["term1", "term2"], '
    '"files": ["path/to/file1.py", "path/to/file2.py"]}}\n```'
)


class ManifestBuilder:
    """Builds a compact structural manifest from the symbol index."""

    def __init__(
        self,
        symbol_store: "SymbolStore",
        import_store: "ImportGraphStore",
        max_chars: int = 120_000,
    ):
        self._symbol_store = symbol_store
        self._import_store = import_store
        self._max_chars = max_chars
        self._cached_manifest: str | None = None
        self._cached_file_data: dict[str, dict] | None = None
        self._cached_symbol_count: int | None = None

    def build(self) -> tuple[str, dict[str, dict]]:
        """Build manifest text and file data index.

        Returns:
            (manifest_text, file_data_by_path) where file_data_by_path maps
            path -> {"raw_file_id": str, "symbols": list[dict]}
        """
        manifest_data = self._symbol_store.get_structural_manifest()
        symbol_count = sum(len(f["symbols"]) for f in manifest_data)

        if self._cached_manifest is not None and self._cached_symbol_count == symbol_count:
            return self._cached_manifest, self._cached_file_data  # type: ignore[return-value]

        file_data: dict[str, dict] = {}
        for f in manifest_data:
            file_data[f["path"]] = {
                "raw_file_id": f["raw_file_id"],
                "symbols": f["symbols"],
            }

        # Build full manifest lines per file
        lines_by_path: dict[str, str] = {}
        for f in manifest_data:
            lines_by_path[f["path"]] = self._format_file(f)

        manifest_text = "\n".join(lines_by_path[p] for p in sorted(lines_by_path))

        # Smart truncation if over budget
        if len(manifest_text) > self._max_chars:
            manifest_text = self._truncate(manifest_data, lines_by_path)

        self._cached_manifest = manifest_text
        self._cached_file_data = file_data
        self._cached_symbol_count = symbol_count
        return manifest_text, file_data

    def _format_file(self, file_entry: dict) -> str:
        """Format a single file's symbols into compact manifest text."""
        path = file_entry["path"]
        symbols = file_entry["symbols"]
        parts = [f"## {path}"]

        classes = []
        functions = []
        imports_set: set[str] = set()

        for sym in symbols:
            kind = sym["kind"]
            name = sym["name"]
            sig = sym.get("signature") or ""

            if kind == "class":
                # Collect methods for this class
                methods = [
                    s
                    for s in symbols
                    if s["kind"] == "method"
                    and s["qualified_name"].startswith(sym["qualified_name"] + ".")
                ]
                if methods:
                    method_strs = []
                    for m in methods:
                        m_sig = m.get("signature") or ""
                        ret = self._extract_return_type(m_sig)
                        method_strs.append(f"{m['name']} -> {ret}")
                    classes.append(f"{name}[{', '.join(method_strs)}]")
                else:
                    classes.append(name)

            elif kind == "function":
                ret = self._extract_return_type(sig)
                params = self._extract_params(sig)
                functions.append(f"{name}({params}) -> {ret}")

            # Collect imports
            for imp in sym.get("imports", []):
                imports_set.add(imp)

        if classes:
            parts.append(f"classes: {', '.join(classes)}")
        if functions:
            parts.append(f"functions: {', '.join(functions)}")
        if imports_set:
            parts.append(f"imports: {', '.join(sorted(imports_set))}")

        return "\n".join(parts)

    def _truncate(self, manifest_data: list[dict], lines_by_path: dict[str, str]) -> str:
        """Smart truncation: 3-pass, least-connected files lose detail first."""
        reverse_counts = self._import_store.get_reverse_counts()

        # Sort files by connectivity (least connected first)
        sorted_files = sorted(
            manifest_data,
            key=lambda f: reverse_counts.get(f["raw_file_id"], 0),
        )

        # Pass 1: strip imports from least-connected files
        for f in sorted_files:
            path = f["path"]
            lines = lines_by_path[path].split("\n")
            lines_by_path[path] = "\n".join(ln for ln in lines if not ln.startswith("imports:"))
            text = "\n".join(lines_by_path[p] for p in sorted(lines_by_path))
            if len(text) <= self._max_chars:
                return text

        # Pass 2: strip functions from least-connected files
        for f in sorted_files:
            path = f["path"]
            lines = lines_by_path[path].split("\n")
            lines_by_path[path] = "\n".join(
                ln
                for ln in lines
                if not ln.startswith("functions:") and not ln.startswith("imports:")
            )
            text = "\n".join(lines_by_path[p] for p in sorted(lines_by_path))
            if len(text) <= self._max_chars:
                return text

        # Pass 3: path-only for least-connected files
        for f in sorted_files:
            path = f["path"]
            lines_by_path[path] = f"## {path}"
            text = "\n".join(lines_by_path[p] for p in sorted(lines_by_path))
            if len(text) <= self._max_chars:
                return text

        return "\n".join(lines_by_path[p] for p in sorted(lines_by_path))

    @staticmethod
    def _extract_return_type(sig: str) -> str:
        if " -> " in sig:
            return sig.split(" -> ", 1)[1].strip()
        return "None"

    @staticmethod
    def _extract_params(sig: str) -> str:
        if "(" in sig and ")" in sig:
            inner = sig[sig.index("(") + 1 : sig.rindex(")")]
            # Strip 'self, ' prefix for methods
            if inner.startswith("self, "):
                inner = inner[6:]
            elif inner == "self":
                inner = ""
            return inner.strip()
        return ""


class LlmCodeSearchStrategy:
    """LLM-based structural code search with hybrid fallback.

    Sends a compact AST manifest to the LLM, which selects relevant files.
    On any failure, transparently falls back to hybrid CodeSearchStrategy.
    """

    def __init__(
        self,
        symbol_store: "SymbolStore",
        import_store: "ImportGraphStore",
        chat_factory: "ChatFactory",
        config: "FitzKragConfig",
        fallback_strategy: "CodeSearchStrategy",
    ):
        self._symbol_store = symbol_store
        self._import_store = import_store
        self._chat_factory = chat_factory
        self._config = config
        self._fallback = fallback_strategy
        self._manifest_builder = ManifestBuilder(symbol_store, import_store)

        # Forwarded attributes — engine wires these after construction
        self._raw_store: Any = None

    def retrieve(
        self,
        query: str,
        limit: int,
        detection: Any = None,
    ) -> list[Address]:
        """Retrieve code addresses via LLM structural search, with fallback."""
        try:
            return self._llm_retrieve(query, limit)
        except Exception as e:
            logger.info(f"LLM code search failed, falling back to hybrid: {e}")
            return self._fallback.retrieve(query, limit, detection=detection)

    def _llm_retrieve(self, query: str, limit: int) -> list[Address]:
        """Core LLM-based retrieval path."""
        manifest_text, file_data = self._manifest_builder.build()
        if not manifest_text.strip():
            raise ValueError("Empty manifest — no indexed symbols")

        search_terms, selected_paths = self._llm_expand_and_select(manifest_text, query)
        if search_terms:
            logger.debug(f"LLM search terms: {search_terms}")
        if not selected_paths:
            raise ValueError("LLM returned no file selections")

        # Map paths to file IDs
        path_to_id = {path: data["raw_file_id"] for path, data in file_data.items()}
        selected_ids = [path_to_id[p] for p in selected_paths if p in path_to_id]

        if not selected_ids:
            raise ValueError("No selected paths matched indexed files")

        # Track origin for scoring
        origin: dict[str, str] = {fid: "selected" for fid in selected_ids}

        # Import expansion (depth 1, forward only)
        import_ids = self._expand_imports(selected_ids)
        for fid in import_ids:
            origin[fid] = "import"

        # Neighbor expansion (same-directory siblings)
        all_so_far = list(dict.fromkeys(selected_ids + import_ids))
        neighbor_ids = self._neighbor_expand(all_so_far, file_data)
        for fid in neighbor_ids:
            origin[fid] = "neighbor"

        all_ids = list(dict.fromkeys(selected_ids + import_ids + neighbor_ids))
        return self._files_to_addresses(all_ids, file_data, limit, origin)

    def _llm_expand_and_select(self, manifest: str, query: str) -> tuple[list[str], list[str]]:
        """Combined query expansion + file selection in one LLM call."""
        prompt = _SYSTEM_PROMPT.format(query=query, structural_index=manifest)
        chat = self._chat_factory("fast")
        response = chat.chat([{"role": "user", "content": prompt}])

        text = response.strip()

        # Try parsing as combined JSON object first
        brace_start = text.find("{")
        brace_end = text.rfind("}") + 1
        if brace_start >= 0 and brace_end > brace_start:
            try:
                parsed = json.loads(text[brace_start:brace_end])
                if isinstance(parsed, dict):
                    terms = parsed.get("search_terms", [])
                    files = parsed.get("files", [])
                    return (
                        [str(t) for t in terms if isinstance(t, str)],
                        [str(f) for f in files if isinstance(f, str) and f.strip()],
                    )
            except json.JSONDecodeError:
                pass

        # Also accept a direct JSON array of file paths.
        start = text.find("[")
        end = text.rfind("]") + 1
        if start < 0 or end <= start:
            raise ValueError(f"No JSON in LLM response: {text[:200]}")

        parsed_list = json.loads(text[start:end])
        if not isinstance(parsed_list, list):
            raise ValueError(f"LLM returned non-list: {type(parsed_list)}")

        return [], [str(p) for p in parsed_list if isinstance(p, str) and p.strip()]

    def _expand_imports(self, file_ids: list[str]) -> list[str]:
        """Forward-only depth-1 import expansion."""
        expanded: list[str] = []
        seen = set(file_ids)
        for fid in file_ids:
            imports = self._import_store.get_imports(fid)
            for edge in imports:
                target = edge.get("target_file_id")
                if target and target not in seen:
                    expanded.append(target)
                    seen.add(target)
        return expanded

    def _neighbor_expand(self, file_ids: list[str], file_data: dict[str, dict]) -> list[str]:
        """Add sibling files from the same directories as selected files."""
        # Build id -> path and path -> id mappings
        id_to_path = {data["raw_file_id"]: path for path, data in file_data.items()}
        path_to_id = {path: data["raw_file_id"] for path, data in file_data.items()}
        seen = set(file_ids)

        # Find directories of selected files
        trigger_dirs: set[str] = set()
        for fid in file_ids:
            path = id_to_path.get(fid, "")
            if "/" in path:
                trigger_dirs.add(path.rsplit("/", 1)[0])

        # Group all indexed files by directory
        dir_files: dict[str, list[str]] = {}
        for path in file_data:
            if "/" in path:
                parent = path.rsplit("/", 1)[0]
                dir_files.setdefault(parent, []).append(path)

        # Add siblings from triggered directories
        neighbors: list[str] = []
        for d in trigger_dirs:
            siblings = dir_files.get(d, [])
            new_siblings = [path_to_id[p] for p in siblings if path_to_id[p] not in seen]
            # Skip directories with too many new siblings (avoid noise)
            if len(new_siblings) > 10:
                continue
            for fid in new_siblings:
                if fid not in seen:
                    neighbors.append(fid)
                    seen.add(fid)

        return neighbors

    def _files_to_addresses(
        self,
        file_ids: list[str],
        file_data: dict[str, dict],
        limit: int,
        origin: dict[str, str],
    ) -> list[Address]:
        """Convert file IDs to FILE-level Address objects.

        One Address per file. Score is flat by origin tier:
        selected=1.0, import=0.9, neighbor=0.8.
        """
        id_to_path = {data["raw_file_id"]: path for path, data in file_data.items()}

        _ORIGIN_SCORES = {"selected": 1.0, "import": 0.9, "neighbor": 0.8}

        addresses: list[Address] = []
        for fid in file_ids:
            path = id_to_path.get(fid)
            if not path:
                continue
            score = _ORIGIN_SCORES.get(origin.get(fid, "neighbor"), 0.8)
            symbols = file_data[path].get("symbols", [])
            summary_parts = []
            for sym in symbols[:5]:
                summary_parts.append(f"{sym['kind']} {sym['name']}")
            summary = ", ".join(summary_parts) if summary_parts else path

            addresses.append(
                Address(
                    kind=AddressKind.FILE,
                    source_id=fid,
                    location=path,
                    summary=summary,
                    score=score,
                    metadata={"origin": origin.get(fid, "neighbor")},
                )
            )
        return addresses[:limit]
