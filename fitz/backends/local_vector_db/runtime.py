from fitz.backends.local_vector_db.config import LocalVectorDBConfig
from fitz.backends.local_vector_db.faiss import FaissLocalVectorDB
from fitz.vector_db.base import VectorDBPlugin


class LocalVectorDBRuntime:
    """
    Thin runtime wrapper to build a local VectorDB
    from an embedding engine.
    """

    def __init__(self, config: LocalVectorDBConfig | None = None):
        self._config = config or LocalVectorDBConfig()

    def build(self, *, embedding_dim: int) -> VectorDBPlugin:
        return FaissLocalVectorDB(
            dim=embedding_dim,
            config=self._config,
        )
