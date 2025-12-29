# fitz_ai/engines/classic_rag/engine.py
"""
ClassicRagEngine - Knowledge engine implementation for classic RAG paradigm.

This engine wraps the existing RAG pipeline (retrieval + generation) behind
the paradigm-agnostic KnowledgeEngine interface.
"""

from fitz_ai.core import (
    Answer,
    ConfigurationError,
    GenerationError,
    KnowledgeError,
    Provenance,
    Query,
    QueryError,
)
from fitz_ai.engines.classic_rag.config.schema import ClassicRagConfig
from fitz_ai.engines.classic_rag.generation.retrieval_guided.synthesis import RGSAnswer
from fitz_ai.engines.classic_rag.pipeline.engine import RAGPipeline


class ClassicRagEngine:
    """
    Classic RAG engine implementation.

    This engine implements the retrieval-augmented generation paradigm:
    1. Embed the query
    2. Retrieve relevant chunks from vector DB
    3. Optionally rerank chunks
    4. Generate answer using LLM + retrieved context

    The engine wraps the existing RAGPipeline and adapts it to the
    KnowledgeEngine protocol.

    Examples:
        >>> from fitz_ai.engines.classic_rag.config import load_config
        >>>
        >>> config = load_config("fitz.yaml")
        >>> engine = ClassicRagEngine(config)
        >>>
        >>> query = Query(text="What is quantum computing?")
        >>> answer = engine.answer(query)
        >>> print(answer.text)
        >>> for source in answer.provenance:
        ...     print(f"Source: {source.source_id}")
    """

    def __init__(self, config: ClassicRagConfig):
        """
        Initialize the Classic RAG engine.

        Args:
            config: ClassicRagConfig object with all RAG settings

        Raises:
            ConfigurationError: If configuration is invalid or required
                              components cannot be initialized
        """
        try:
            # Use the factory method to create RAGPipeline from config
            # This properly initializes all components (retrieval, llm, rgs, context)
            self._pipeline = RAGPipeline.from_config(config)
            self._config = config
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Classic RAG engine: {e}") from e

    def answer(self, query: Query) -> Answer:
        """
        Execute a query using classic RAG approach.

        This method:
        1. Validates the query
        2. Applies constraints (if any)
        3. Runs the RAG pipeline (retrieve → generate)
        4. Converts the result to the standard Answer format

        Args:
            query: Query object with the question and optional constraints

        Returns:
            Answer object with generated text and source provenance

        Raises:
            QueryError: If query is invalid
            KnowledgeError: If retrieval fails
            GenerationError: If answer generation fails
        """
        # Validate query
        if not query.text or not query.text.strip():
            raise QueryError("Query text cannot be empty")

        try:
            # Run the RAG pipeline
            rgs_answer: RGSAnswer = self._pipeline.run(query.text)

            # Convert to standard Answer format
            provenance = [
                Provenance(
                    source_id=src.doc_id or src.source_id,
                    excerpt=src.content or "",
                    metadata=src.metadata,
                )
                for src in rgs_answer.sources
            ]

            return Answer(
                text=rgs_answer.answer,
                provenance=provenance,
                metadata={
                    "engine": "classic_rag",
                    "query": query.text,
                },
            )

        except Exception as e:
            # Determine error type and re-raise appropriately
            error_msg = str(e).lower()
            if "retriev" in error_msg or "search" in error_msg:
                raise KnowledgeError(f"Retrieval failed: {e}") from e
            elif "generat" in error_msg or "llm" in error_msg:
                raise GenerationError(f"Generation failed: {e}") from e
            else:
                raise KnowledgeError(f"RAG pipeline error: {e}") from e

    @property
    def config(self) -> ClassicRagConfig:
        """Get the engine's configuration."""
        return self._config

    @classmethod
    def from_yaml(cls, config_path: str) -> "ClassicRagEngine":
        """
        Create engine from a YAML configuration file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Configured ClassicRagEngine instance
        """
        from fitz_ai.engines.classic_rag.config import load_config

        config = load_config(config_path)
        return cls(config)
