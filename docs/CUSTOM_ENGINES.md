# Custom Engines

Fitz-Sage exposes a small in-process engine registry. It is useful when an
application needs a different knowledge engine while keeping the shared
`Query`, `Answer`, and optional `EvidencePack` contracts.

This is a Python integration point, not a third-party plugin ABI. The installed
package does not load `fitz_sage.engines` entry points from other distributions.

## Discovery Boundary

At import time, `fitz_sage.runtime` scans only directories bundled inside:

```text
fitz_sage/engines/<engine_name>/
```

For application-owned engines, import the module that performs registration
before calling `run()`, `create_engine()`, or `list_engines()`. A separate CLI
process cannot see registrations made only in another Python process.

## Minimal Knowledge Engine

The minimum protocol is `answer(Query) -> Answer`:

```python
from fitz_sage.core import Answer, Query
from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.runtime import EngineCapabilities, EngineRegistry


class MyEngine:
    def __init__(self, config: object | None = None) -> None:
        self.config = config

    def answer(self, query: Query) -> Answer:
        return Answer(
            text=f"Received: {query.text}",
            mode=AnswerMode.SUFFICIENT,
            provenance=[],
            metadata={"engine": "my_engine"},
        )


@EngineRegistry.register_engine(
    name="my_engine",
    description="Application-owned example engine",
    capabilities=EngineCapabilities(
        supports_collections=False,
        supports_persistent_ingest=False,
        requires_config=False,
    ),
)
def create_my_engine(config: object | None) -> MyEngine:
    return MyEngine(config)
```

Use it after importing that module:

```python
import my_application.fitz_engine  # registers my_engine

from fitz_sage import create_engine, run

answer = run("What is indexed?", engine="my_engine")
engine = create_engine("my_engine")
```

Protocol conformance uses structural typing. No Fitz-Sage base class is
required.

## Optional Configuration Loader

The registry can load engine-specific configuration when callers do not pass a
config object:

```python
from pathlib import Path

import yaml

from fitz_sage.runtime import EngineRegistry


def load_my_config(config_path: str | None) -> dict:
    if config_path is None:
        return {"prefix": "default"}
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@EngineRegistry.register_engine(
    name="configured_engine",
    description="Engine with application-owned YAML",
    config_loader=load_my_config,
)
def create_configured_engine(config: dict):
    return ConfiguredEngine(config)
```

`config_type`, `default_config_path`, and `list_collections` can also be
attached to a registration. They are registry metadata; the engine remains
responsible for validating and honoring its own configuration.

## Retrieval Engines

An engine that exposes persistent ingest and evidence retrieval can implement
the richer `RetrievalEngine` protocol:

```python
from pathlib import Path

from fitz_sage.core import EvidencePack, Query


class MyRetrievalEngine:
    def load(self, collection: str) -> None: ...

    def point(
        self,
        source: Path,
        collection: str | None = None,
        *,
        start_worker: bool = True,
    ) -> object: ...

    def wait_for_enrichment(self) -> None: ...

    def indexing_status(self) -> dict: ...

    def retrieve(self, query: Query) -> list: ...

    def evidence(self, query: Query) -> EvidencePack: ...

    def trace(self, query: Query): ...

    def answer(self, query: Query): ...
```

Return types and lifecycle semantics are documented in
[API Reference](API_REFERENCE.md). A custom retrieval engine does not need to
reuse KRAG's SQLite schema, parsers, reranker, or Pyrrho integration unless that
is part of the engine's own contract.

Declare persistent-ingest support in registration metadata:

```python
from fitz_sage.runtime import EngineCapabilities

capabilities = EngineCapabilities(
    supports_collections=True,
    supports_persistent_ingest=True,
    requires_config=True,
)
```

## Registration Without A Decorator

Direct registration is equivalent:

```python
from fitz_sage.runtime import EngineRegistry

EngineRegistry.get_global().register(
    name="my_engine",
    factory=lambda config: MyEngine(config),
    description="Application-owned engine",
)
```

Duplicate names raise `ValueError`. Unknown names raise
`ConfigurationError` when resolved.

## Testing

Test the same lifecycle your application uses:

```python
import my_application.fitz_engine

from fitz_sage import create_engine, list_engines
from fitz_sage.core import KnowledgeEngine, Query


def test_registration_and_contract() -> None:
    assert "my_engine" in list_engines()
    engine = create_engine("my_engine")
    assert isinstance(engine, KnowledgeEngine)
    assert engine.answer(Query(text="test")).text
```

For persistent engines, also test collection isolation, repeated `point()`
behavior, visible ingest failures, and provenance stability.

## Deliberate Non-Features

- No external package entry-point discovery.
- No generic chunker, source-cleanup, or identifier-normalization plugin.
- No guarantee that KRAG internals form a stable extension ABI.
- No automatic import of application registration modules in the standalone
  Fitz CLI or API server.

Contributing a generally useful engine directly under `fitz_sage/engines/` is a
package change and should include registration, config, contract tests, and
documentation.

## Related

- [Engines](ENGINES.md)
- [API Reference](API_REFERENCE.md)
- [Extension Points](PLUGINS.md)
- [Architecture](ARCHITECTURE.md)
