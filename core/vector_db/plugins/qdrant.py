from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from core.logging.logger import get_logger
from core.logging.tags import VECTOR_DB
from core.vector_db.base import VectorDBPlugin, VectorRecord
from core.vector_db.registry import register_vector_db_plugin

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest
except ImportError:  # pragma: no cover - optional dependency
    QdrantClient = None  # type: ignore[assignment]
    rest = None  # type: ignore[assignment]

logger = get_logger(__name__)


@dataclass
class QdrantVectorDB(VectorDBPlugin):
    """
    Qdrant-based vector DB plugin.

    Thin wrapper around `qdrant_client.QdrantClient` that provides a
    uniform interface that matches what the retriever expects:

    - `.search(collection_name=..., query_vector=..., limit=..., with_payload=True)`
    - `.upsert(collection, records)`
    - `.delete_collection(collection)`
    """

    plugin_name: str = "qdrant"

    host: str = "localhost"
    port: int = 6333
    https: bool = False
    api_key: Optional[str] = None

    def __post_init__(self) -> None:
        if QdrantClient is None:
            raise RuntimeError(
                "qdrant-client is not installed. "
                "Install with: `pip install qdrant-client`."
            )

        url = f"{'https' if self.https else 'http'}://{self.host}:{self.port}"

        logger.info(
            f"{VECTOR_DB} Initializing QdrantClient: url={url}, api_key={'***' if self.api_key else '<none>'}"
        )

        # For local, usually url/host+port is enough. For cloud, api_key/url.
        self._client = QdrantClient(
            url=url,
            api_key=self.api_key,
        )

    # -------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------
    def upsert(self, collection: str, records: List[VectorRecord]) -> None:
        if rest is None:
            raise RuntimeError(
                "qdrant-client HTTP models not available. "
                "Ensure `qdrant-client` is properly installed."
            )

        points = [
            rest.PointStruct(
                id=rec.id,
                vector=rec.vector,
                payload=rec.payload or {},
            )
            for rec in records
        ]

        logger.info(
            f"{VECTOR_DB} Upserting {len(points)} points into collection='{collection}'"
        )

        self._client.upsert(collection_name=collection, points=points)

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int,
        with_payload: bool = True,
    ) -> List[Any]:
        """
        Signature intentionally mirrors the Qdrant client's `search` so that
        existing retriever code (and tests using dummy client.search) stays
        compatible.
        """
        logger.info(
            f"{VECTOR_DB} Searching collection='{collection_name}', "
            f"limit={limit}, with_payload={with_payload}"
        )

        return self._client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=with_payload,
        )

    def delete_collection(self, collection: str) -> None:
        logger.info(f"{VECTOR_DB} Deleting collection='{collection}'")
        self._client.delete_collection(collection_name=collection)


# Register on import
register_vector_db_plugin(QdrantVectorDB, plugin_name="qdrant")
