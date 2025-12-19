"""
KnowledgeEngine Protocol - Core abstraction for all knowledge engines.

This is the single stable contract that all engines (ClassicRAG, CLaRa, etc.) must implement.
The platform architecture is: Knowledge → Engine → Answer.
"""

from typing import Protocol, runtime_checkable

from .answer import Answer
from .query import Query


@runtime_checkable
class KnowledgeEngine(Protocol):
    """
    Paradigm-agnostic protocol for knowledge engines.

    All engines (RAG, CLaRa, future paradigms) must implement this interface.
    This is the only stable abstraction in the Fitz platform.

    Philosophy:
        - Engines are black boxes that transform queries into answers
        - Implementation details (retrieval, LLMs, reasoning) are engine-specific
        - The platform only cares about: Query in → Answer out

    Examples:
        >>> engine = ClassicRagEngine(config)
        >>> query = Query(text="What is quantum computing?")
        >>> answer = engine.answer(query)
        >>> print(answer.text)
    """

    def answer(self, query: Query) -> Answer:
        """
        Execute a query against knowledge and return an answer.

        This is the only required method. How the engine generates the answer
        is entirely up to the implementation. Classic RAG uses retrieval + generation,
        CLaRa might use uncertainty-guided reasoning, future engines might use
        completely different approaches.

        Args:
            query: Query object containing the question text, optional constraints,
                   and engine-specific metadata hints

        Returns:
            Answer object with the answer text, source provenance, and metadata

        Raises:
            QueryError: If the query is invalid or cannot be processed
            KnowledgeError: If knowledge retrieval/processing fails
            EngineError: For any other engine-specific errors

        Note:
            Implementations should be idempotent when possible. The same query
            should produce consistent answers (though not necessarily identical,
            since LLMs may vary).
        """
        ...
