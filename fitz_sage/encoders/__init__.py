# fitz_sage/encoders/__init__.py
"""Shared runtime machinery for Fitz-Sage's local CPU encoders.

``OnnxEncoderBackend`` owns the lock-guarded lazy load of a pre-built ONNX
graph from the Hugging Face Hub and the forward pass. Fitz-Sage's ONNX
cross-encoder reranker subclasses it.

Pyrrho has a model-specific managed adapter because its multi-head output and
artifact validation differ from the shared retrieval-encoder mechanics.
"""

from fitz_sage.encoders.onnx import OnnxEncoderBackend

__all__ = ["OnnxEncoderBackend"]
