# fitz_ai/engines/classic_rag/retrieval/runtime/loader.py
"""
YAML-based Retrieval Plugin Loader.

Loads retrieval plugin definitions from YAML files and builds
executable pipelines using the standard step classes.

Plugin YAML files live in: fitz_ai/engines/classic_rag/retrieval/runtime/plugins/*.yaml
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from fitz_ai.engines.classic_rag.retrieval.steps import (
    DedupeStep,
    Embedder,
    LimitStep,
    Reranker,
    RerankStep,
    RetrievalStep,
    ThresholdStep,
    VectorClient,
    VectorSearchStep,
    get_step_class,
)
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import RETRIEVER

logger = get_logger(__name__)

# Path to plugin YAML files
PLUGINS_DIR = Path(__file__).parent / "plugins"


# =============================================================================
# Plugin Spec (parsed from YAML)
# =============================================================================


@dataclass
class StepSpec:
    """Specification for a single retrieval step."""

    type: str
    params: dict[str, Any] = field(default_factory=dict)
    enabled_if: str | None = None  # Dependency name that must be present
    use_config: str | None = None  # Config key to override a param

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepSpec":
        step_type = data.get("type")
        if not step_type:
            raise ValueError("Step must have 'type' field")

        return cls(
            type=step_type,
            params={k: v for k, v in data.items() if k not in ("type", "enabled_if", "use_config")},
            enabled_if=data.get("enabled_if"),
            use_config=data.get("use_config"),
        )


@dataclass
class RetrievalPluginSpec:
    """Specification for a retrieval plugin (loaded from YAML)."""

    plugin_name: str
    description: str
    steps: list[StepSpec]

    @classmethod
    def from_yaml(cls, path: Path) -> "RetrievalPluginSpec":
        """Load plugin spec from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        plugin_name = data.get("plugin_name")
        if not plugin_name:
            raise ValueError(f"Plugin YAML must have 'plugin_name': {path}")

        steps_data = data.get("steps", [])
        steps = [StepSpec.from_dict(s) for s in steps_data]

        return cls(
            plugin_name=plugin_name,
            description=data.get("description", ""),
            steps=steps,
        )


# =============================================================================
# Dependencies Container
# =============================================================================


@dataclass
class RetrievalDependencies:
    """
    Container for retrieval dependencies.

    These are injected when building the pipeline.
    """

    vector_client: VectorClient
    embedder: Embedder
    collection: str
    reranker: Reranker | None = None

    # Config overrides (from retrieval config)
    top_k: int = 5


# =============================================================================
# Plugin Loader
# =============================================================================


def load_plugin_spec(plugin_name: str) -> RetrievalPluginSpec:
    """
    Load a retrieval plugin specification by name.

    Searches for {plugin_name}.yaml in the plugins directory.
    """
    yaml_path = PLUGINS_DIR / f"{plugin_name}.yaml"

    if not yaml_path.exists():
        available = list_available_plugins()
        raise FileNotFoundError(
            f"Retrieval plugin not found: {plugin_name!r}. "
            f"Available: {available}. "
            f"Searched: {yaml_path}"
        )

    return RetrievalPluginSpec.from_yaml(yaml_path)


def list_available_plugins() -> list[str]:
    """List all available retrieval plugins."""
    if not PLUGINS_DIR.exists():
        return []

    return [
        p.stem for p in PLUGINS_DIR.glob("*.yaml")
    ]


def build_pipeline_from_spec(
    spec: RetrievalPluginSpec,
    deps: RetrievalDependencies,
) -> list[RetrievalStep]:
    """
    Build executable step instances from a plugin spec.

    Args:
        spec: Plugin specification (from YAML)
        deps: Injected dependencies

    Returns:
        List of configured RetrievalStep instances
    """
    steps: list[RetrievalStep] = []

    for step_spec in spec.steps:
        # Check if step should be enabled
        if step_spec.enabled_if:
            dep_value = getattr(deps, step_spec.enabled_if, None)
            if dep_value is None:
                logger.debug(f"{RETRIEVER} Skipping step {step_spec.type}: {step_spec.enabled_if} not provided")
                continue

        # Build params
        params = dict(step_spec.params)

        # Apply config overrides
        if step_spec.use_config:
            config_value = getattr(deps, step_spec.use_config, None)
            if config_value is not None:
                # Override 'k' with config value
                params["k"] = config_value

        # Inject dependencies based on step type
        step = _build_step(step_spec.type, params, deps)
        steps.append(step)

    return steps


def _build_step(
    step_type: str,
    params: dict[str, Any],
    deps: RetrievalDependencies,
) -> RetrievalStep:
    """Build a single step instance."""
    step_cls = get_step_class(step_type)

    # Inject dependencies based on step type
    if step_type == "vector_search":
        params.setdefault("client", deps.vector_client)
        params.setdefault("embedder", deps.embedder)
        params.setdefault("collection", deps.collection)

    elif step_type == "rerank":
        if deps.reranker is None:
            raise ValueError("Rerank step requires reranker dependency")
        params.setdefault("reranker", deps.reranker)

    return step_cls(**params)


# =============================================================================
# Main Entry Point
# =============================================================================


def create_retrieval_pipeline(
    plugin_name: str,
    vector_client: VectorClient,
    embedder: Embedder,
    collection: str,
    reranker: Reranker | None = None,
    top_k: int = 5,
) -> "RetrievalPipelineFromYaml":
    """
    Create a retrieval pipeline from a YAML plugin definition.

    Args:
        plugin_name: Name of the plugin (e.g., "dense", "dense_rerank")
        vector_client: Vector database client
        embedder: Embedding service
        collection: Collection name
        reranker: Optional reranking service
        top_k: Final number of chunks to return

    Returns:
        Configured retrieval pipeline
    """
    spec = load_plugin_spec(plugin_name)

    deps = RetrievalDependencies(
        vector_client=vector_client,
        embedder=embedder,
        collection=collection,
        reranker=reranker,
        top_k=top_k,
    )

    steps = build_pipeline_from_spec(spec, deps)

    return RetrievalPipelineFromYaml(
        plugin_name=spec.plugin_name,
        description=spec.description,
        steps=steps,
    )


# =============================================================================
# Pipeline Class
# =============================================================================


@dataclass
class RetrievalPipelineFromYaml:
    """
    Retrieval pipeline built from YAML plugin definition.

    This is the runtime object that executes retrieval.
    """

    plugin_name: str
    description: str
    steps: list[RetrievalStep]

    def retrieve(self, query: str) -> list:
        """Execute the retrieval pipeline."""
        from fitz_ai.engines.classic_rag.models.chunk import Chunk

        logger.info(f"{RETRIEVER} Running {self.plugin_name} pipeline ({len(self.steps)} steps)")

        chunks: list[Chunk] = []

        for i, step in enumerate(self.steps):
            logger.debug(f"{RETRIEVER} Step {i + 1}/{len(self.steps)}: {step.name}")
            chunks = step.execute(query, chunks)

        logger.info(f"{RETRIEVER} Pipeline complete: {len(chunks)} chunks")
        return chunks