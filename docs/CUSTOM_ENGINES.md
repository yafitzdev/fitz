# Creating Custom Engines

This guide walks you through creating your own knowledge engine for Fitz.

---

## Overview

Fitz's pluggable architecture makes it easy to add custom engines. You just need to:

1. Implement the `answer(Query) -> Answer` method
2. Register with the engine registry
3. (Optional) Add configuration support

---

## Quick Start

### Minimal Engine (5 Minutes)

```python
# my_engine.py

from fitz.core import Query, Answer, Provenance
from fitz.runtime import EngineRegistry

class MySimpleEngine:
    """A minimal custom engine."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.knowledge = []
    
    def add_documents(self, documents: list[str]):
        """Add documents to knowledge base."""
        self.knowledge.extend(documents)
    
    def answer(self, query: Query) -> Answer:
        """Answer a query using simple keyword matching."""
        # Simple keyword search
        relevant = []
        for i, doc in enumerate(self.knowledge):
            if any(word.lower() in doc.lower() 
                   for word in query.text.split()):
                relevant.append((i, doc))
        
        # Build answer
        if relevant:
            answer_text = f"Based on {len(relevant)} sources: " + relevant[0][1][:200]
        else:
            answer_text = "No relevant information found."
        
        # Build provenance
        provenance = [
            Provenance(
                source_id=f"doc_{i}",
                excerpt=doc[:100],
                metadata={"index": i}
            )
            for i, doc in relevant[:3]
        ]
        
        return Answer(
            text=answer_text,
            provenance=provenance,
            metadata={"engine": "my_simple", "sources_found": len(relevant)}
        )

# Register the engine
registry = EngineRegistry.get_global()
registry.register(
    name="my_simple",
    factory=lambda config: MySimpleEngine(config),
    description="Simple keyword-based search engine"
)
```

### Using Your Engine

```python
from fitz import run

# Use via universal runtime
answer = run("What is Python?", engine="my_simple")
print(answer.text)

# Or use directly
from my_engine import MySimpleEngine

engine = MySimpleEngine()
engine.add_documents(["Python is a programming language..."])
answer = engine.answer(Query(text="What is Python?"))
```

---

## Complete Engine Implementation

Let's build a more complete engine with proper structure.

### Directory Structure

```
fitz/engines/my_engine/
├── __init__.py          # Package exports + registration
├── engine.py            # Main engine class
├── runtime.py           # Convenience functions
└── config/
    ├── __init__.py
    └── schema.py        # Configuration dataclasses
```

### Step 1: Configuration (`config/schema.py`)

```python
# fitz/engines/my_engine/config/schema.py

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
import yaml

@dataclass
class MyEngineModelConfig:
    """Model configuration."""
    model_name: str = "default-model"
    device: str = "cpu"
    max_length: int = 512

@dataclass
class MyEngineRetrievalConfig:
    """Retrieval configuration."""
    top_k: int = 5
    similarity_threshold: float = 0.5

@dataclass
class MyEngineConfig:
    """Main configuration for MyEngine."""
    model: MyEngineModelConfig = field(default_factory=MyEngineModelConfig)
    retrieval: MyEngineRetrievalConfig = field(default_factory=MyEngineRetrievalConfig)
    cache_results: bool = True

def load_my_engine_config(config_path: Optional[str] = None) -> MyEngineConfig:
    """Load configuration from YAML file."""
    if config_path is None:
        return MyEngineConfig()
    
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(path) as f:
        data = yaml.safe_load(f)
    
    engine_data = data.get("my_engine", {})
    
    return MyEngineConfig(
        model=MyEngineModelConfig(**engine_data.get("model", {})),
        retrieval=MyEngineRetrievalConfig(**engine_data.get("retrieval", {})),
        cache_results=engine_data.get("cache_results", True),
    )
```

### Step 2: Engine Class (`engine.py`)

```python
# fitz/engines/my_engine/engine.py

from typing import Optional, Dict, Any, List
import logging

from fitz.core import (
    Query,
    Answer,
    Provenance,
    QueryError,
    KnowledgeError,
    GenerationError,
    ConfigurationError,
)
from fitz.engines.my_engine.config.schema import MyEngineConfig

logger = logging.getLogger(__name__)


class MyEngine:
    """
    Custom knowledge engine implementation.
    
    This engine demonstrates a complete implementation with:
    - Configuration support
    - Document management
    - Query processing
    - Error handling
    """
    
    def __init__(self, config: MyEngineConfig):
        """
        Initialize the engine.
        
        Args:
            config: Engine configuration
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        self._config = config
        self._documents: Dict[str, str] = {}  # id -> content
        self._embeddings: Dict[str, Any] = {}  # id -> embedding
        
        try:
            self._initialize()
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize: {e}") from e
    
    def _initialize(self) -> None:
        """Initialize internal components."""
        logger.info(f"Initializing MyEngine with model: {self._config.model.model_name}")
        # Load your model, set up connections, etc.
    
    def add_documents(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of document texts
            doc_ids: Optional custom IDs
            
        Returns:
            List of document IDs
            
        Raises:
            KnowledgeError: If documents cannot be added
        """
        if doc_ids is None:
            import uuid
            doc_ids = [f"doc_{uuid.uuid4().hex[:8]}" for _ in documents]
        
        if len(doc_ids) != len(documents):
            raise ValueError("doc_ids length must match documents")
        
        try:
            for doc_id, content in zip(doc_ids, documents):
                self._documents[doc_id] = content
                self._embeddings[doc_id] = self._embed(content)
            
            logger.info(f"Added {len(documents)} documents")
            return doc_ids
            
        except Exception as e:
            raise KnowledgeError(f"Failed to add documents: {e}") from e
    
    def _embed(self, text: str) -> Any:
        """Create embedding for text."""
        # Your embedding logic here
        # For demo, just return word count as "embedding"
        return len(text.split())
    
    def answer(self, query: Query) -> Answer:
        """
        Execute a query and return an answer.
        
        This is the main method required by the KnowledgeEngine protocol.
        
        Args:
            query: The query to answer
            
        Returns:
            Answer with text and provenance
            
        Raises:
            QueryError: If query is invalid
            KnowledgeError: If retrieval fails
            GenerationError: If generation fails
        """
        # Validate query
        if not query.text.strip():
            raise QueryError("Query text cannot be empty")
        
        if not self._documents:
            raise KnowledgeError("No documents in knowledge base")
        
        try:
            # Retrieve relevant documents
            top_k = self._config.retrieval.top_k
            if query.constraints and query.constraints.max_sources:
                top_k = min(top_k, query.constraints.max_sources)
            
            retrieved = self._retrieve(query.text, top_k)
            
            # Generate answer
            answer_text = self._generate(query.text, retrieved)
            
            # Build provenance
            provenance = [
                Provenance(
                    source_id=doc_id,
                    excerpt=self._documents[doc_id][:200],
                    metadata={"relevance_score": score}
                )
                for doc_id, score in retrieved
            ]
            
            return Answer(
                text=answer_text,
                provenance=provenance,
                metadata={
                    "engine": "my_engine",
                    "model": self._config.model.model_name,
                    "docs_retrieved": len(retrieved),
                }
            )
            
        except (QueryError, KnowledgeError):
            raise
        except Exception as e:
            raise GenerationError(f"Generation failed: {e}") from e
    
    def _retrieve(self, query: str, top_k: int) -> List[tuple[str, float]]:
        """Retrieve top-k relevant documents."""
        # Your retrieval logic here
        # For demo, simple keyword matching
        scores = []
        query_words = set(query.lower().split())
        
        for doc_id, content in self._documents.items():
            doc_words = set(content.lower().split())
            overlap = len(query_words & doc_words)
            score = overlap / max(len(query_words), 1)
            scores.append((doc_id, score))
        
        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def _generate(self, query: str, context: List[tuple[str, float]]) -> str:
        """Generate answer from query and context."""
        # Your generation logic here
        # For demo, concatenate excerpts
        if not context:
            return "I don't have enough information to answer that."
        
        excerpts = [self._documents[doc_id][:100] for doc_id, _ in context]
        return f"Based on the available information: {' '.join(excerpts)}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            "num_documents": len(self._documents),
            "model": self._config.model.model_name,
        }
    
    def clear(self) -> None:
        """Clear the knowledge base."""
        self._documents.clear()
        self._embeddings.clear()
        logger.info("Knowledge base cleared")
```

### Step 3: Runtime Functions (`runtime.py`)

```python
# fitz/engines/my_engine/runtime.py

from typing import Optional, Union, List
from pathlib import Path

from fitz.core import Query, Answer, Constraints
from fitz.engines.my_engine.config.schema import (
    MyEngineConfig,
    load_my_engine_config,
)

# Module-level cache
_cached_engine = None
_cached_config_path = None


def run_my_engine(
    query: Union[str, Query],
    documents: Optional[List[str]] = None,
    config: Optional[MyEngineConfig] = None,
    config_path: Optional[Union[str, Path]] = None,
    constraints: Optional[Constraints] = None,
) -> Answer:
    """
    Run a query through MyEngine.
    
    Args:
        query: Question text or Query object
        documents: Documents to add (optional)
        config: Configuration object
        config_path: Path to config file
        constraints: Query constraints
        
    Returns:
        Answer object
    """
    global _cached_engine, _cached_config_path
    
    # Import here to avoid circular imports
    from fitz.engines.my_engine.engine import MyEngine
    
    # Reuse cached engine if config unchanged
    current_path = str(config_path) if config_path else None
    if _cached_engine and _cached_config_path == current_path and config is None:
        engine = _cached_engine
    else:
        if config is None:
            config = load_my_engine_config(current_path)
        engine = MyEngine(config)
        _cached_engine = engine
        _cached_config_path = current_path
    
    # Add documents if provided
    if documents:
        engine.add_documents(documents)
    
    # Build query
    if isinstance(query, str):
        query_obj = Query(text=query, constraints=constraints)
    else:
        query_obj = query
    
    return engine.answer(query_obj)


def create_my_engine(
    config: Optional[MyEngineConfig] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> "MyEngine":
    """Create a reusable MyEngine instance."""
    from fitz.engines.my_engine.engine import MyEngine
    
    if config is None:
        config = load_my_engine_config(str(config_path) if config_path else None)
    
    return MyEngine(config)


def clear_engine_cache() -> None:
    """Clear the cached engine."""
    global _cached_engine, _cached_config_path
    _cached_engine = None
    _cached_config_path = None
```

### Step 4: Package Init (`__init__.py`)

```python
# fitz/engines/my_engine/__init__.py

"""
MyEngine - Custom knowledge engine for Fitz.

Usage:
    >>> from fitz.engines.my_engine import run_my_engine
    >>> answer = run_my_engine("What is X?", documents=["..."])
"""

# Configuration
from fitz.engines.my_engine.config.schema import (
    MyEngineConfig,
    MyEngineModelConfig,
    MyEngineRetrievalConfig,
    load_my_engine_config,
)

# Engine
from fitz.engines.my_engine.engine import MyEngine

# Runtime
from fitz.engines.my_engine.runtime import (
    run_my_engine,
    create_my_engine,
    clear_engine_cache,
)

__all__ = [
    "MyEngine",
    "run_my_engine",
    "create_my_engine",
    "clear_engine_cache",
    "MyEngineConfig",
    "MyEngineModelConfig",
    "MyEngineRetrievalConfig",
    "load_my_engine_config",
]


# Register with global registry
def _register_engine():
    from fitz.runtime import EngineRegistry
    
    registry = EngineRegistry.get_global()
    
    if "my_engine" not in registry.list():
        registry.register(
            name="my_engine",
            factory=lambda config: MyEngine(config or MyEngineConfig()),
            description="Custom knowledge engine",
        )

_register_engine()
```

---

## Testing Your Engine

### Unit Tests

```python
# tests/test_my_engine.py

import pytest
from fitz.core import Query, Answer, Provenance, QueryError, KnowledgeError

class TestMyEngine:
    
    @pytest.fixture
    def engine(self):
        from fitz.engines.my_engine import MyEngine, MyEngineConfig
        return MyEngine(MyEngineConfig())
    
    def test_implements_protocol(self, engine):
        assert hasattr(engine, 'answer')
        assert callable(engine.answer)
    
    def test_add_documents(self, engine):
        ids = engine.add_documents(["Doc 1", "Doc 2"])
        assert len(ids) == 2
    
    def test_answer_success(self, engine):
        engine.add_documents(["Python is a programming language"])
        answer = engine.answer(Query(text="What is Python?"))
        
        assert isinstance(answer, Answer)
        assert answer.text
        assert answer.metadata["engine"] == "my_engine"
    
    def test_answer_empty_query_error(self, engine):
        engine.add_documents(["Doc"])
        with pytest.raises(QueryError):
            engine.answer(Query(text=""))  # Note: Query validates this
    
    def test_answer_no_docs_error(self, engine):
        with pytest.raises(KnowledgeError):
            engine.answer(Query(text="What is X?"))
```

### Integration Test

```python
def test_my_engine_via_universal_runtime():
    from fitz import run
    from fitz.runtime import list_engines
    
    # Verify registration
    assert "my_engine" in list_engines()
    
    # Use via runtime (requires documents to be pre-loaded)
    # answer = run("What is X?", engine="my_engine")
```

---

## Best Practices

### 1. Error Handling

Use Fitz's exception hierarchy:

```python
from fitz.core import QueryError, KnowledgeError, GenerationError, ConfigurationError

# Configuration issues
raise ConfigurationError("Invalid model path")

# Query validation
raise QueryError("Query text cannot be empty")

# Knowledge/retrieval issues
raise KnowledgeError("No documents in knowledge base")

# Generation issues
raise GenerationError("LLM failed to generate response")
```

### 2. Logging

Use standard Python logging:

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Engine initialized")
logger.debug(f"Retrieved {len(docs)} documents")
logger.warning("No relevant documents found")
logger.error(f"Generation failed: {e}")
```

### 3. Configuration

Make everything configurable:

```python
@dataclass
class MyConfig:
    # Good: configurable with sensible defaults
    model_name: str = "default-model"
    top_k: int = 5
    temperature: float = 0.7
    
    # Bad: hardcoded values in engine code
```

### 4. Provenance

Always provide meaningful provenance:

```python
Provenance(
    source_id="unique_id",          # Stable, retrievable ID
    excerpt="relevant text...",      # What was actually used
    metadata={
        "relevance_score": 0.95,     # How relevant
        "source_type": "document",   # What kind of source
        "timestamp": "2024-01-01",   # When added
    }
)
```

### 5. Lazy Loading

Don't load heavy resources until needed:

```python
class MyEngine:
    def __init__(self, config):
        self._config = config
        self._model = None  # Don't load yet
    
    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model
```

---

## Publishing Your Engine

### As a Fitz Plugin

1. Create a separate package: `fitz-my-engine`
2. Add entry point in `pyproject.toml`:

```toml
[project.entry-points."fitz.engines"]
my_engine = "fitz_my_engine:register"
```

3. Users install and use:

```bash
pip install fitz-my-engine
```

```python
from fitz import run
answer = run("Question?", engine="my_engine")
```

### Contributing to Fitz

1. Add engine to `fitz/engines/`
2. Add tests to `tests/`
3. Update documentation
4. Submit PR

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
