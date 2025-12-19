"""
Core exceptions for all knowledge engines.

These exceptions provide a standard error hierarchy for the Fitz platform.
"""


class EngineError(Exception):
    """
    Base exception for all engine errors.

    All engines should raise exceptions that inherit from this base class.
    This allows consumers to catch all engine-related errors with a single handler.

    Examples:
        >>> try:
        ...     answer = engine.answer(query)
        ... except EngineError as e:
        ...     print(f"Engine failed: {e}")
    """

    pass


class QueryError(EngineError):
    """
    Invalid or malformed query.

    Raised when the query cannot be processed, for example:
    - Empty query text
    - Invalid constraints
    - Malformed metadata

    Examples:
        >>> try:
        ...     query = Query(text="")  # Empty text
        ... except QueryError as e:
        ...     print(f"Invalid query: {e}")
    """

    pass


class KnowledgeError(EngineError):
    """
    Knowledge retrieval or processing error.

    Raised when the engine cannot access or process knowledge, for example:
    - Vector DB connection failure
    - Document not found
    - Corrupted knowledge base
    - Index unavailable

    This is distinct from generation errors (see below) - this is about
    accessing the knowledge itself.

    Examples:
        >>> try:
        ...     answer = engine.answer(query)
        ... except KnowledgeError as e:
        ...     print(f"Knowledge system unavailable: {e}")
    """

    pass


class GenerationError(EngineError):
    """
    Answer generation error.

    Raised when the engine retrieved knowledge successfully but failed to
    generate an answer, for example:
    - LLM API failure
    - Generation timeout
    - Invalid LLM response
    - Reasoning failure

    This is distinct from knowledge errors - the knowledge was accessible,
    but answer generation failed.

    Examples:
        >>> try:
        ...     answer = engine.answer(query)
        ... except GenerationError as e:
        ...     print(f"Failed to generate answer: {e}")
    """

    pass


class ConfigurationError(EngineError):
    """
    Engine configuration error.

    Raised when the engine is misconfigured, for example:
    - Missing required credentials
    - Invalid engine parameters
    - Incompatible configuration

    These errors typically occur during engine initialization, not during
    query execution.

    Examples:
        >>> try:
        ...     engine = ClassicRagEngine(invalid_config)
        ... except ConfigurationError as e:
        ...     print(f"Engine misconfigured: {e}")
    """

    pass


class TimeoutError(EngineError):
    """
    Query execution timeout.

    Raised when query execution exceeds time constraints, for example:
    - Query took too long to execute
    - Constraint timeout_seconds exceeded

    Examples:
        >>> constraints = Constraints(metadata={"timeout_seconds": 5})
        >>> try:
        ...     answer = engine.answer(Query("complex query", constraints=constraints))
        ... except TimeoutError as e:
        ...     print(f"Query timed out: {e}")
    """

    pass


class UnsupportedOperationError(EngineError):
    """
    Operation not supported by this engine.

    Raised when a requested operation is not available for this engine type,
    for example:
    - Engine doesn't support certain constraint types
    - Feature not implemented for this paradigm

    Examples:
        >>> try:
        ...     # Some hypothetical feature not all engines support
        ...     answer = engine.answer_with_uncertainty(query)
        ... except UnsupportedOperationError as e:
        ...     print(f"This engine doesn't support uncertainty: {e}")
    """

    pass
