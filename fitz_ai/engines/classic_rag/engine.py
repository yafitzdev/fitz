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
from fitz_ai.engines.classic_rag.config.schema import (
    FitzConfig,
    LoggingConfig,
    PipelinePluginConfig,
    RAGConfig,
    RerankConfig,
    RetrieverConfig,
    RGSConfig,
)
from fitz_ai.engines.classic_rag.generation.retrieval_guided.synthesis import RGSAnswer

# RAG-specific imports (from the moved modules)
from fitz_ai.engines.classic_rag.pipeline.pipeline.engine import RAGPipeline


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
        >>> from fitz_ai.engines.classic_rag.config.loader import load_config
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
            # Convert FitzConfig to RAGConfig
            # FitzConfig is the top-level config, RAGConfig is the pipeline-specific config
            rag_config = self._convert_to_rag_config(config)

            # Use the factory method to create RAGPipeline from config
            # This properly initializes all components (retriever, llm, rgs, context)
            self._pipeline = RAGPipeline.from_config(rag_config)
            self._config = config
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Classic RAG engine: {e}") from e

    def _convert_to_rag_config(self, fitz_config: FitzConfig) -> RAGConfig:
        """
        Convert FitzConfig to RAGConfig.

        FitzConfig is the engine-level config, RAGConfig is the pipeline-specific config.
        This method bridges the two.

        Args:
            fitz_config: The engine configuration

        Returns:
            RAGConfig suitable for RAGPipeline.from_config()
        """
        # Map FitzConfig fields to RAGConfig fields
        rag_config = RAGConfig(
            llm=PipelinePluginConfig(
                plugin_name=fitz_config.chat.plugin_name, kwargs=fitz_config.chat.kwargs
            ),
            embedding=PipelinePluginConfig(
                plugin_name=fitz_config.embedding.plugin_name, kwargs=fitz_config.embedding.kwargs
            ),
            vector_db=PipelinePluginConfig(
                plugin_name=fitz_config.vector_db.plugin_name, kwargs=fitz_config.vector_db.kwargs
            ),
            rerank=RerankConfig(
                enabled=fitz_config.rerank is not None,
                plugin_name=fitz_config.rerank.plugin_name if fitz_config.rerank else None,
                kwargs=fitz_config.rerank.kwargs if fitz_config.rerank else {},
            ),
            # Pipeline plugin config - this determines which pipeline plugin to use
            # For now, we'll use the standard pipeline
            retriever=RetrieverConfig(
                plugin_name="dense",  # Default to dense retrieval
                collection="default",  # This should come from config or be parameterized
                top_k=5,  # Default
            ),
            rgs=RGSConfig(),  # Use defaults
            logging=LoggingConfig(),  # Use defaults
        )

        return rag_config

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
            # Extract RAG-specific parameters from query (if any)
            # For now, RAGPipeline.run() just takes the query text
            # Future: pass constraints via pipeline kwargs if needed

            # Run the RAG pipeline
            # Note: RAGPipeline.run() returns RGSAnswer
            rag_answer: RGSAnswer = self._pipeline.run(query.text)

            # Convert RAG-specific answer to paradigm-agnostic Answer
            return self._convert_to_core_answer(rag_answer, query)

        except QueryError:
            # Re-raise query errors as-is
            raise
        except Exception as e:
            # Classify other exceptions appropriately
            error_msg = str(e).lower()
            if "retriev" in error_msg or "vector" in error_msg:
                raise KnowledgeError(f"Failed to retrieve knowledge: {e}") from e
            elif "generat" in error_msg or "llm" in error_msg:
                raise GenerationError(f"Failed to generate answer: {e}") from e
            else:
                raise GenerationError(f"Failed to answer query: {e}") from e

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
                # RGSSourceRef has: source_id, text, metadata
                prov = Provenance(
                    source_id=getattr(source_ref, "source_id", str(source_ref)),
                    excerpt=getattr(source_ref, "text", None),
                    metadata=getattr(source_ref, "metadata", {}),
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
            metadata=answer_metadata,
        )

    @property
    def config(self) -> FitzConfig:
        """Get the engine's configuration."""
        return self._config
