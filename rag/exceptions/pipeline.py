from .base import FitzRAGError

class PipelineError(FitzRAGError):
    """Raised when a pipeline-level operation fails."""
    pass


class RGSGenerationError(PipelineError):
    """Raised when RGS prompt generation or answer synthesis fails."""
    pass
