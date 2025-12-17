# fitz/engines/clara/config/schema.py
"""
CLaRa Configuration Schema.

Configuration dataclasses for the CLaRa engine.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class ClaraModelConfig:
    """
    Configuration for CLaRa model loading.

    Note: CLaRa models on HuggingFace have subdirectories for compression rates.
    Use the full path like "apple/CLaRa-7B-E2E/compression-16".
    """

    # Model path - NOTE: includes compression subfolder!
    model_name_or_path: str = "apple/CLaRa-7B-Instruct/compression-16"

    # Model variant: base, instruct, or e2e
    variant: Literal["base", "instruct", "e2e"] = "instruct"

    # Device to load model on
    device: str = "cuda"

    # Torch dtype for model weights
    torch_dtype: str = "bfloat16"

    # Whether to trust remote code (required for CLaRa)
    trust_remote_code: bool = True

    # Quantization options
    load_in_8bit: bool = False
    load_in_4bit: bool = False


@dataclass
class ClaraCompressionConfig:
    """Configuration for document compression."""

    # Compression rate: 16 or 128
    compression_rate: int = 16

    # Maximum document length before compression
    doc_max_length: int = 2048

    # Batch size for compression
    compression_batch_size: int = 4


@dataclass
class ClaraRetrievalConfig:
    """Configuration for latent space retrieval."""

    # Number of documents to retrieve
    top_k: int = 5

    # Size of candidate pool for retrieval
    candidate_pool_size: int = 20

    # Whether to use differentiable top-k (for E2E model)
    differentiable_topk: bool = True


@dataclass
class ClaraGenerationConfig:
    """Configuration for answer generation."""

    # Maximum new tokens to generate
    max_new_tokens: int = 256

    # Temperature for sampling
    temperature: float = 0.7

    # Top-p (nucleus) sampling
    top_p: float = 0.9

    # Whether to use sampling or greedy decoding
    do_sample: bool = True


@dataclass
class ClaraConfig:
    """
    Main configuration for CLaRa engine.

    Examples:
        Default config:
        >>> config = ClaraConfig()

        Custom config:
        >>> config = ClaraConfig(
        ...     model=ClaraModelConfig(
        ...         model_name_or_path="apple/CLaRa-7B-E2E/compression-16",
        ...         variant="e2e",
        ...         device="cuda",
        ...     ),
        ...     compression=ClaraCompressionConfig(compression_rate=16),
        ... )

        Load from YAML:
        >>> config = load_clara_config("clara_config.yaml")
    """

    model: ClaraModelConfig = field(default_factory=ClaraModelConfig)
    compression: ClaraCompressionConfig = field(default_factory=ClaraCompressionConfig)
    retrieval: ClaraRetrievalConfig = field(default_factory=ClaraRetrievalConfig)
    generation: ClaraGenerationConfig = field(default_factory=ClaraGenerationConfig)

    # Whether to cache compressed documents
    cache_compressed_docs: bool = True


def load_clara_config(config_path: Optional[str] = None) -> ClaraConfig:
    """
    Load CLaRa configuration from a YAML file.

    Args:
        config_path: Path to YAML config file. If None, returns defaults.

    Returns:
        ClaraConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if config_path is None:
        return ClaraConfig()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)

    # Extract clara section if present
    clara_data = data.get("clara", data)

    # Build config from nested dicts
    model_data = clara_data.get("model", {})
    compression_data = clara_data.get("compression", {})
    retrieval_data = clara_data.get("retrieval", {})
    generation_data = clara_data.get("generation", {})

    return ClaraConfig(
        model=ClaraModelConfig(**model_data),
        compression=ClaraCompressionConfig(**compression_data),
        retrieval=ClaraRetrievalConfig(**retrieval_data),
        generation=ClaraGenerationConfig(**generation_data),
        cache_compressed_docs=clara_data.get("cache_compressed_docs", True),
    )
