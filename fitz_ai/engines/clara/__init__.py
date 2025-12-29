# fitz_ai/engines/clara/__init__.py
"""
CLaRa Engine - Continuous Latent Reasoning for RAG.

This engine implements Apple's CLaRa paradigm which bridges retrieval and
generation using continuous latent reasoning. It compresses documents into
memory tokens and performs retrieval in latent space.

Key features:
- 16x-128x document compression while preserving semantics
- Unified retrieval-generation optimization
- No separate embedding model needed
- Superior multi-hop reasoning performance

Public API:
    - ClaraEngine: Main engine class implementing KnowledgeEngine protocol
    - run_clara: Convenience function for quick queries
    - create_clara_engine: Factory for creating engine instances
    - ClaraConfig: Configuration dataclass

Models (from HuggingFace):
    - apple/CLaRa-7B-Base: Base compression model
    - apple/CLaRa-7B-Instruct: Instruction-tuned model
    - apple/CLaRa-7B-E2E: End-to-end retrieval + generation

Examples:
    Quick query:
    >>> from fitz_ai.engines.clara import run_clara
    >>>
    >>> documents = [
    ...     "Quantum computing uses qubits instead of classical bits...",
    ...     "Machine learning models learn patterns from data...",
    ... ]
    >>> answer = run_clara("What is quantum computing?", documents=documents)
    >>> print(answer.text)

    Reusable engine:
    >>> from fitz_ai.engines.clara import create_clara_engine, ClaraConfig
    >>>
    >>> config = ClaraConfig()  # Use defaults
    >>> engine = create_clara_engine(config=config)
    >>>
    >>> # Add documents once
    >>> engine.add_documents(my_documents)
    >>>
    >>> # Query multiple times
    >>> answer1 = engine.answer(Query(text="Question 1?"))
    >>> answer2 = engine.answer(Query(text="Question 2?"))

    Via universal runtime:
    >>> from fitz import run
    >>> answer = run("What is X?", engine="clara")

References:
    - Paper: https://arxiv.org/abs/2511.18659
    - GitHub: https://github.com/apple/ml-clara
    - Models: https://huggingface.co/apple/CLaRa-7B-E2E
"""

# =============================================================================
# CONFIGURATION (import first, no dependencies on engine)
# =============================================================================
from fitz_ai.engines.clara.config.schema import (
    ClaraCompressionConfig,
    ClaraConfig,
    ClaraGenerationConfig,
    ClaraModelConfig,
    ClaraRetrievalConfig,
    load_clara_config,
)

# =============================================================================
# ENGINE (import after config to avoid circular imports)
# =============================================================================
from fitz_ai.engines.clara.engine import ClaraEngine

# =============================================================================
# RUNTIME (import after engine)
# =============================================================================
from fitz_ai.engines.clara.runtime import (
    clear_engine_cache,
    create_clara_engine,
    run_clara,
)

__all__ = [
    # Engine
    "ClaraEngine",
    # Runtime
    "run_clara",
    "create_clara_engine",
    "clear_engine_cache",
    # Config
    "ClaraConfig",
    "ClaraModelConfig",
    "ClaraCompressionConfig",
    "ClaraRetrievalConfig",
    "ClaraGenerationConfig",
    "load_clara_config",
]


# =============================================================================
# ENGINE REGISTRATION
# =============================================================================
# Register CLaRa engine with the global registry so it can be accessed via
# the universal run() function.
#
# This is done at the END of __init__.py to ensure all imports are complete.


def _register_clara_engine():
    """Register CLaRa with the engine registry."""
    from fitz_ai.runtime import EngineCapabilities, EngineRegistry

    def _create_clara_engine_factory(config) -> ClaraEngine:
        """
        Factory function for creating CLaRa engine instances.

        This is registered with the global engine registry and called
        by the universal run() function when engine="clara".
        """
        if config is None:
            config = ClaraConfig(model=ClaraModelConfig(load_in_4bit=True))
        elif isinstance(config, dict):
            # Convert dict to ClaraConfig
            config = ClaraConfig(
                model=ClaraModelConfig(**config.get("model", {})),
                compression=ClaraCompressionConfig(**config.get("compression", {})),
                retrieval=ClaraRetrievalConfig(**config.get("retrieval", {})),
                generation=ClaraGenerationConfig(**config.get("generation", {})),
            )

        return ClaraEngine(config)

    def _clara_config_loader(config_path):
        """Load config for clara engine."""
        if config_path:
            return load_clara_config(config_path)
        # Default config with 4-bit quantization for consumer GPUs
        return ClaraConfig(model=ClaraModelConfig(load_in_4bit=True))

    # Define capabilities - CLaRa is fundamentally different from classic RAG
    capabilities = EngineCapabilities(
        supports_collections=False,  # No persistent storage
        requires_documents_at_query=True,  # Must add docs before query
        supports_chat=False,  # No multi-turn yet
        supports_streaming=False,
        requires_config=False,  # Works with defaults
        requires_api_key=False,  # Local GPU model
        cli_query_message=(
            "CLaRa requires documents to be loaded first.\n"
            "Use 'fitz quickstart <folder> \"question\" --engine clara' for one-off queries."
        ),
    )

    # Register with global registry
    registry = EngineRegistry.get_global()

    # Only register if not already registered
    if "clara" not in registry.list():
        registry.register(
            name="clara",
            factory=_create_clara_engine_factory,
            description="CLaRa: Continuous Latent Reasoning - Compression-native RAG with 16x-128x document compression",
            config_type=ClaraConfig,
            config_loader=_clara_config_loader,
            capabilities=capabilities,
        )


# Perform registration when module is imported
_register_clara_engine()
