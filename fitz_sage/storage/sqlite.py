# fitz_sage/storage/sqlite.py
"""
SQLite connection management for unified storage.

Each collection is a single ``.db`` file under the workspace's storage
directory. SQLite is opened in WAL journal mode so multiple readers
can coexist with a single writer, and ``foreign_keys`` is enabled.

There is no server lifecycle to manage — the file *is* the database.
"""

from __future__ import annotations

import atexit
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

from fitz_sage.core.collections import validate_collection_name
from fitz_sage.core.paths import FitzPaths
from fitz_sage.logging.logger import get_logger
from fitz_sage.logging.tags import STORAGE
from fitz_sage.storage.config import StorageConfig

logger = get_logger(__name__)


_DB_PREFIX = "fitz_"


def _configure_connection(conn: sqlite3.Connection) -> None:
    """Apply standard pragmas to a freshly opened connection."""
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA busy_timeout = 30000")


class SqliteConnectionManager:
    """Manages SQLite connections for fitz-sage.

    Singleton that handles:
    - Storage directory lifecycle (creation on start, nothing else)
    - One ``.db`` file per collection under the storage directory
    - Per-call connection opening with WAL + foreign_keys pragmas

    SQLite has no server, no pool, no admin database — every operation
    is just a file open + close.
    """

    _instance: Optional["SqliteConnectionManager"] = None
    _lock = threading.RLock()

    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or StorageConfig()
        self._storage_dir: Optional[Path] = None
        self._started = False

    @classmethod
    def get_instance(cls, config: Optional[StorageConfig] = None) -> "SqliteConnectionManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config)
                atexit.register(cls._instance.stop)
            elif config is not None and not cls._instance._started:
                cls._instance.config = config
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.stop()
                cls._instance = None

    def start(self) -> None:
        """Ensure the storage directory exists."""
        if self._started:
            return
        with self._lock:
            if self._started:
                return
            self._storage_dir = (
                Path(self.config.storage_path)
                if self.config.storage_path
                else FitzPaths.workspace() / "sqlite"
            )
            self._storage_dir.mkdir(parents=True, exist_ok=True)
            self._started = True
            logger.info(f"{STORAGE} SQLite storage at {self._storage_dir}")

    def database_path(self, collection: str) -> Path:
        """Return the file path for a collection's database (does not create)."""
        if not self._started:
            self.start()
        assert self._storage_dir is not None
        return self._storage_dir / f"{_DB_PREFIX}{validate_collection_name(collection)}.db"

    def list_collections(self) -> list[str]:
        """List existing collection names by scanning the storage directory."""
        if not self._started:
            self.start()
        assert self._storage_dir is not None
        names = []
        for path in self._storage_dir.glob(f"{_DB_PREFIX}*.db"):
            stem = path.stem
            if stem.startswith(_DB_PREFIX):
                names.append(stem[len(_DB_PREFIX) :])
        return sorted(names)

    def delete_collection(self, collection: str) -> bool:
        """Delete a collection's database file. Returns True on success."""
        path = self.database_path(collection)
        if not path.exists():
            return False
        # Remove WAL/SHM sidecar files too
        for sidecar in (path.with_suffix(".db-wal"), path.with_suffix(".db-shm")):
            if sidecar.exists():
                try:
                    sidecar.unlink()
                except OSError as e:
                    logger.warning(f"{STORAGE} Could not unlink {sidecar}: {e}")
        try:
            path.unlink()
            logger.info(f"{STORAGE} Deleted collection database: {path.name}")
            return True
        except OSError as e:
            logger.error(f"{STORAGE} Failed to delete {path}: {e}")
            return False

    @contextmanager
    def connection(self, collection: str) -> Generator[sqlite3.Connection, None, None]:
        """Open a connection to a collection's database.

        A new connection is opened on each call and closed when the context
        exits. SQLite open is microseconds so this is cheaper than maintaining
        a pool.
        """
        if not self._started:
            self.start()
        path = self.database_path(collection)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        try:
            _configure_connection(conn)
            yield conn
        finally:
            try:
                conn.close()
            except Exception as e:
                logger.debug(f"{STORAGE} Error closing connection: {e}")

    def execute(self, collection: str, sql: str, params: tuple = ()) -> Any:
        """Execute SQL on a collection's database and commit immediately."""
        with self.connection(collection) as conn:
            cur = conn.execute(sql, params)
            conn.commit()
            return cur

    def stop(self) -> None:
        """Mark stopped. No persistent state to release (no pool, no server)."""
        with self._lock:
            self._started = False
            self._storage_dir = None


# Module-level convenience functions


def get_connection_manager(
    config: Optional[StorageConfig] = None,
) -> SqliteConnectionManager:
    """Get the singleton connection manager."""
    return SqliteConnectionManager.get_instance(config)


@contextmanager
def get_connection(collection: str) -> Generator[sqlite3.Connection, None, None]:
    """Get a database connection for a collection (singleton-manager shortcut)."""
    manager = get_connection_manager()
    with manager.connection(collection) as conn:
        yield conn
