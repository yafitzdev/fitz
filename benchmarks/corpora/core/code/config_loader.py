# benchmarks/corpora/core/code/config_loader.py
"""Configuration loader used by benchmark cases."""

import os

REQUIRED_ENV_VARS = ("FITZ_API_TOKEN", "FITZ_WORKSPACE")


def load_required_env() -> dict[str, str]:
    """Load required environment variables or raise a clear error."""
    missing = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    return {name: os.environ[name] for name in REQUIRED_ENV_VARS}
