# fitz/generation/exceptions.py


class GenerationError(Exception):
    """Base error for generation failures."""


class RGSGenerationError(GenerationError):
    """Raised when RGS prompt or answer construction fails."""
