from .base import FitzRAGError

class LLMError(FitzRAGError):
    """General LLM call failure."""
    pass


class LLMResponseError(LLMError):
    """LLM returned malformed / unusable output."""
    pass
