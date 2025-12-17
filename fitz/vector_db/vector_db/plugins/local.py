# fitz/core/vector_db/plugins/local.py

from fitz.backends.local_vector_db.faiss import FaissLocalVectorDB


class LocalVectorDB(FaissLocalVectorDB):
    """
    Local FAISS-backed VectorDB plugin.

    Registered under the name: `local-faiss`
    """

    name = "local-faiss"
