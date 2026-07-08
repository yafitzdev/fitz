# fitz_sage/encoders/__init__.py
"""
Local CPU encoders.

Small, fine-tuned transformer encoders served as pre-quantized INT8 ONNX
on `onnxruntime` — no `torch`, no chat call. Each one replaces a chat
call (or an sklearn cascade) that previously produced a categorical or
scalar output.

This package holds the shared machinery — `OnnxEncoderBackend`, which
owns the lock-guarded lazy load of a pre-built ONNX from the HuggingFace
hub and the forward pass. The concrete encoders live next to the
features they serve:

- `fitz_sage.governance.pyrrho` — SUFFICIENT / DISPUTED / INSUFFICIENT classifier
- `fitz_sage.llm.providers.onnx_reranker` — cross-encoder reranker

Both subclass `OnnxEncoderBackend`.
"""

from fitz_sage.encoders.onnx import OnnxEncoderBackend

__all__ = ["OnnxEncoderBackend"]
