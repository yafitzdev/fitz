# fitz_sage/encoders/__init__.py
"""Shared runtime machinery for Fitz-Sage's local CPU encoders.

``OnnxEncoderBackend`` owns the lock-guarded lazy load of a pre-built ONNX
graph from the Hugging Face Hub and the forward pass. Fitz-Sage's ONNX
cross-encoder reranker subclasses it.

Pyrrho is an independent package with its own runtime and does not subclass
Fitz-Sage internals.
"""

from fitz_sage.encoders.onnx import OnnxEncoderBackend

__all__ = ["OnnxEncoderBackend"]
