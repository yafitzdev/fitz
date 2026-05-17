# fitz_sage/storage/__init__.py
"""
Unified SQLite storage for fitz-sage.

Each collection is a single ``.db`` file under the workspace storage
directory. Full-text search uses SQLite FTS5 (``bm25()``). No server,
no pool, no admin database.

Usage:
    from fitz_sage.storage import get_connection_manager, get_connection

    manager = get_connection_manager()
    manager.start()

    with get_connection("my_collection") as conn:
        conn.execute("SELECT * FROM krag_section_index LIMIT 10")
"""

from fitz_sage.storage.config import StorageConfig
from fitz_sage.storage.sqlite import (
    SqliteConnectionManager,
    get_connection,
    get_connection_manager,
)

__all__ = [
    "StorageConfig",
    "SqliteConnectionManager",
    "get_connection_manager",
    "get_connection",
]
