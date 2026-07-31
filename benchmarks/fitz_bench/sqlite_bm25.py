"""Disk-backed plain BM25 baseline for corpora too large for Python memory."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from benchmarks.fitz_bench.retrieval_eval import tokenize

SCHEMA_VERSION = 1


class SqliteBm25:
    """Persistent SQLite FTS5 BM25 index with exact corpus identity checks."""

    def __init__(self, connection: sqlite3.Connection, *, action: str) -> None:
        self._connection = connection
        self.action = action

    @classmethod
    def open_or_build(
        cls,
        path: Path,
        *,
        fingerprint: dict[str, Any],
        expected_documents: int,
        documents: Callable[[], Iterable[tuple[str, str, str]]],
        progress: Callable[[str], None] | None = None,
    ) -> "SqliteBm25":
        """Open an exact cached index or atomically build it from source documents."""
        target = Path(path).resolve()
        expected = {
            "schema_version": SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "documents": expected_documents,
        }
        if target.is_file() and _metadata_matches(target, expected):
            return cls(_open_read_only(target), action="reused_verified")

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.building")
        temporary.unlink(missing_ok=True)
        emit = progress or (lambda _message: None)
        connection = sqlite3.connect(temporary)
        try:
            _configure_build(connection)
            _create_schema(connection)
            count = _insert_documents(connection, documents(), emit=emit)
            if count != expected_documents:
                raise ValueError(
                    "BM25 source document count mismatch: "
                    f"expected {expected_documents}, found {count}"
                )
            connection.execute("INSERT INTO corpus(corpus) VALUES('optimize')")
            connection.execute(
                "INSERT INTO benchmark_metadata(key, value) VALUES(?, ?)",
                ("identity", json.dumps(expected, sort_keys=True, separators=(",", ":"))),
            )
            connection.commit()
        except Exception:
            connection.close()
            temporary.unlink(missing_ok=True)
            raise
        connection.close()
        target.unlink(missing_ok=True)
        temporary.replace(target)
        return cls(_open_read_only(target), action="built")

    def search(self, query: str, *, top_k: int) -> list[str]:
        """Return unique external document IDs in ascending FTS5 BM25 order."""
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in terms)
        ranking: list[str] = []
        seen: set[str] = set()
        offset = 0
        while len(ranking) < top_k:
            rows = self._connection.execute(
                """
                SELECT document_id
                FROM corpus
                WHERE corpus MATCH ?
                ORDER BY bm25(corpus), rowid
                LIMIT ? OFFSET ?
                """,
                (expression, top_k, offset),
            ).fetchall()
            for (document_id,) in rows:
                value = str(document_id)
                if value in seen:
                    continue
                seen.add(value)
                ranking.append(value)
                if len(ranking) == top_k:
                    break
            if len(rows) < top_k:
                break
            offset += len(rows)
        return ranking

    def close(self) -> None:
        self._connection.close()


def _metadata_matches(path: Path, expected: dict[str, Any]) -> bool:
    try:
        with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as connection:
            row = connection.execute(
                "SELECT value FROM benchmark_metadata WHERE key = 'identity'"
            ).fetchone()
            document_count = int(connection.execute("SELECT count(*) FROM corpus").fetchone()[0])
    except sqlite3.DatabaseError:
        return False
    if row is None:
        return False
    try:
        return bool(
            json.loads(str(row[0])) == expected and document_count == int(expected["documents"])
        )
    except json.JSONDecodeError:
        return False


def _open_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def _configure_build(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA page_size = 32768")
    connection.execute("PRAGMA journal_mode = OFF")
    connection.execute("PRAGMA synchronous = OFF")
    connection.execute("PRAGMA temp_store = MEMORY")
    connection.execute("PRAGMA cache_size = -262144")


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.execute("CREATE TABLE benchmark_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        """
        CREATE VIRTUAL TABLE corpus USING fts5(
            document_id UNINDEXED,
            physical_id UNINDEXED,
            content,
            tokenize = 'unicode61'
        )
        """
    )


def _insert_documents(
    connection: sqlite3.Connection,
    documents: Iterable[tuple[str, str, str]],
    *,
    emit: Callable[[str], None],
) -> int:
    batch: list[tuple[str, str, str]] = []
    count = 0
    for physical_id, document_id, content in documents:
        batch.append((document_id, physical_id, content))
        if len(batch) < 1000:
            continue
        connection.executemany(
            "INSERT INTO corpus(document_id, physical_id, content) VALUES(?, ?, ?)",
            batch,
        )
        count += len(batch)
        batch.clear()
        if count % 25_000 == 0:
            emit(f"  BM25 indexed {count} documents")
    if batch:
        connection.executemany(
            "INSERT INTO corpus(document_id, physical_id, content) VALUES(?, ?, ?)",
            batch,
        )
        count += len(batch)
    emit(f"  BM25 indexed {count} documents")
    return count
