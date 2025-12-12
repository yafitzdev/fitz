from __future__ import annotations

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator


# -------------------------------------------------------------------
# LLM Config
# -------------------------------------------------------------------

class LLMConfig(BaseModel):
    provider: str = Field(..., description="LLM provider name, e.g., cohere, openai")
    model: str = Field(..., description="Model identifier for chat completion.")
    api_key: Optional[str] = Field(None, description="API key for the LLM provider.")
    temperature: float = Field(0.2, description="Sampling temperature.")


# -------------------------------------------------------------------
# Embedding Config
# -------------------------------------------------------------------

class EmbeddingConfig(BaseModel):
    provider: str = Field(..., description="Embedding provider name.")
    model: str = Field(..., description="Embedding model ID.")
    api_key: Optional[str] = None
    input_type: Optional[str] = None
    output_dimension: Optional[int] = None


# -------------------------------------------------------------------
# Reranker Config
# -------------------------------------------------------------------

class RerankConfig(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    enabled: bool = True


# -------------------------------------------------------------------
# Retriever Config
# -------------------------------------------------------------------

class RetrieverConfig(BaseModel):
    collection: str
    top_k: int = 20
    final_top_k: Optional[int] = None  # kept for backward compatibility
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    distance: str = "cosine"

    @validator("top_k")
    def ensure_top_k(cls, v):
        if v < 1:
            raise ValueError("top_k must be >= 1")
        return v


# -------------------------------------------------------------------
# RGS Config
# -------------------------------------------------------------------

class RGSSettings(BaseModel):
    enable_citations: bool = True
    strict_grounding: bool = True
    answer_style: Optional[str] = None
    max_chunks: Optional[int] = 8
    max_answer_chars: Optional[int] = None


# -------------------------------------------------------------------
# Logging Config
# -------------------------------------------------------------------

class LoggingConfig(BaseModel):
    level: str = "INFO"
    pretty: bool = True


# -------------------------------------------------------------------
# Ingestion Config
# -------------------------------------------------------------------

class IngestionConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 100
    allowed_extensions: List[str] = Field(default_factory=lambda: ["txt", "md", "pdf"])

    @validator("chunk_size")
    def check_chunk_size(cls, v):
        if v < 100:
            raise ValueError("chunk_size must be >= 100")
        return v


# -------------------------------------------------------------------
# Unified RAG Config
# -------------------------------------------------------------------

class RAGConfig(BaseModel):
    """
    The full unified Pydantic-based configuration for fitz-rag.
    This model mirrors your default.yaml and loader structure.
    """

    llm: LLMConfig
    embedding: EmbeddingConfig
    rerank: RerankConfig
    retriever: RetrieverConfig
    rgs: RGSSettings
    logging: LoggingConfig
    ingestion: IngestionConfig

    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGConfig":
        """
        Safely convert raw dict (from loader.py) into typed RAGConfig.
        """
        return cls(**data)
