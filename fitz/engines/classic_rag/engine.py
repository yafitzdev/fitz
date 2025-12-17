"""
ClassicRagEngine - Knowledge engine implementation for classic RAG paradigm.

This engine wraps the existing RAG pipeline (retrieval + generation) behind
the paradigm-agnostic KnowledgeEngine interface.
"""

from typing import Optional, Dict, Any

from fitz.core import (
    KnowledgeEngine,
    Query,
    Answer,
    Provenance,
    QueryError,
    KnowledgeError,
    GenerationError,
    ConfigurationError,
)

# RAG-specific imports (from the moved modules)
from fitz.engines.classic_rag.pipeline.pipeline.engine import RAGPipeline
from fitz.engines.classic_rag.generation.retrieval_guided.synthesis import RGSAnswer
from fitz.engines.classic_rag.config.schema import FitzConfig


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
        >>> from fitz.engines.classic_rag.config.loader import load_config
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

    def __init__(self, config: FitzConfig):
        """
        Initialize the Classic RAG engine.

        Args:
            config: Fitz configuration object with all RAG settings

        Raises:
            ConfigurationError: If configuration is invalid or required
                              components cannot be initialized
        """
        try:
            # Initialize the underlying RAG pipeline
            self._pipeline = RAGPipeline(config)
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
        4. Converts RAG-specific output to paradigm-agnostic Answer

        Args:
            query: Query object with text and optional constraints

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
            # Extract RAG-specific parameters from query
            rag_kwargs = self._extract_rag_params(query)

            # Run the RAG pipeline
            # Note: RAGPipeline.run() returns RGSAnswer
            rag_answer: RGSAnswer = self._pipeline.run(
                query=query.text,
                **rag_kwargs
            )

            # Convert RAG-specific answer to paradigm-agnostic Answer
            return self._convert_to_core_answer(rag_answer, query)

        except QueryError:
            # Re-raise query errors as-is
            raise
        except Exception as e:
            # Classify other exceptions appropriately
            if "retriev" in str(e).lower() or "vector" in str(e).lower():
                raise KnowledgeError(f"Failed to retrieve knowledge: {e}") from e
            elif "generat" in str(e).lower() or "llm" in str(e).lower():
                raise GenerationError(f"Failed to generate answer: {e}") from e
            else:
                raise GenerationError(f"Failed to answer query: {e}") from e

    def _extract_rag_params(self, query: Query) -> Dict[str, Any]:
        """
        Extract RAG-specific parameters from query metadata and constraints.

        Maps core Query/Constraints to RAG pipeline parameters.

        Args:
            query: The query object

        Returns:
            Dictionary of RAG-specific parameters for RAGPipeline.run()
        """
        kwargs = {}

        # Apply constraints if present
        if query.constraints:
            # Map max_sources to top_k
            if query.constraints.max_sources is not None:
                kwargs["top_k"] = query.constraints.max_sources

            # Pass filters as metadata filters
            if query.constraints.filters:
                kwargs["metadata_filters"] = query.constraints.filters

            # Pass through any RAG-specific constraint metadata
            if query.constraints.metadata:
                # Allow overriding specific RAG parameters via constraints
                # e.g., constraints.metadata = {"rerank": False, "temperature": 0.3}
                for key, value in query.constraints.metadata.items():
                    if key in ["rerank", "temperature", "top_k", "model"]:
                        kwargs[key] = value

        # Pass through query metadata
        # This allows engine-specific hints like {"temperature": 0.5}
        if query.metadata:
            kwargs.update(query.metadata)

        return kwargs

    def _convert_to_core_answer(self, rag_answer: RGSAnswer, original_query: Query) -> Answer:
        """
        Convert RAG-specific RGSAnswer to paradigm-agnostic Answer.

        Args:
            rag_answer: The RGSAnswer from the RAG pipeline
            original_query: The original query (for metadata)

        Returns:
            Core Answer object
        """
        # Convert source references to Provenance objects
        provenance = []
        if hasattr(rag_answer, "sources") and rag_answer.sources:
            for source_ref in rag_answer.sources:
                prov = Provenance(
                    source_id=source_ref.chunk_id if hasattr(source_ref, "chunk_id") else str(source_ref),
                    excerpt=source_ref.text if hasattr(source_ref, "text") else None,
                    metadata=source_ref.metadata if hasattr(source_ref, "metadata") else {}
                )
                provenance.append(prov)

        # Build metadata about how answer was generated
        answer_metadata = {
            "engine": "classic_rag",
            "query_text": original_query.text,
        }

        # Include any RAG-specific metadata from the answer
        if hasattr(rag_answer, "metadata") and rag_answer.metadata:
            answer_metadata["rag_metadata"] = rag_answer.metadata

        # Return core Answer
        return Answer(
            text=rag_answer.answer if hasattr(rag_answer, "answer") else str(rag_answer),
            provenance=provenance,
            metadata=answer_metadata
        )

    @property
    def config(self) -> FitzConfig:
        """Get the engine's configuration."""
        return self._config