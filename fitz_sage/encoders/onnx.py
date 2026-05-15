# fitz_sage/encoders/onnx.py
"""
Shared backend for INT8 ONNX encoders.

`OnnxEncoderBackend` owns the parts that the pyrrho governance classifier
and the ONNX cross-encoder reranker would otherwise duplicate line for
line:

- a `threading.Lock`-guarded lazy load that runs once per process,
- pulling a pre-built ONNX straight from a HuggingFace repo with
  `huggingface_hub.hf_hub_download` (no on-the-fly export, no `optimum`,
  no `torch`),
- building an `onnxruntime.InferenceSession` on the CPU provider,
- the `transformers` tokenizer load,
- feeding only the inputs the ONNX graph actually declares.

Subclasses own what genuinely differs: the public method, the tokenizer
call shape (single text vs. query/document pairs), batching, and the
numpy post-processing of logits. A new encoder is then a ~30-line
subclass, not another ~150-line module.
"""

from __future__ import annotations

import threading
from typing import Any

from fitz_sage.logging.logger import get_logger

logger = get_logger(__name__)


class OnnxEncoderBackend:
    """Lazy-loaded INT8 ONNX encoder served on CPU.

    Args:
        model_id: HuggingFace repo id of the encoder.
        onnx_file: Pre-built ONNX filename to load from the repo.
        onnx_subfolder: Repo subfolder holding the ONNX file. Empty
            string when the file sits at the repo root.
    """

    def __init__(
        self,
        model_id: str,
        onnx_file: str,
        onnx_subfolder: str = "",
    ) -> None:
        self._model_id = model_id
        self._onnx_file = onnx_file
        self._onnx_subfolder = onnx_subfolder
        self._lock = threading.Lock()
        self._tokenizer: Any = None
        self._session: Any = None

    def _load(self) -> None:
        """Load the tokenizer + ONNX session once per process (idempotent)."""
        if self._tokenizer is not None and self._session is not None:
            return
        with self._lock:
            if self._tokenizer is not None and self._session is not None:
                return
            import os

            # transformers is used purely as a tokenizer here — silence its
            # advisory that no DL framework (torch/TF/Flax) is installed.
            os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from transformers import AutoTokenizer

            logger.info(
                f"Loading ONNX encoder {self._model_id} "
                f"({self._onnx_subfolder or '.'}/{self._onnx_file})"
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
            try:
                onnx_path = hf_hub_download(
                    repo_id=self._model_id,
                    filename=self._onnx_file,
                    subfolder=self._onnx_subfolder or None,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Could not fetch the pre-built ONNX "
                    f"'{self._onnx_subfolder or '.'}/{self._onnx_file}' from "
                    f"{self._model_id}: {e}. Point the encoder at an ONNX "
                    f"file the repo actually ships."
                ) from e
            self._session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def _encode(self, *args: Any, **tokenizer_kwargs: Any) -> Any:
        """Tokenize into numpy tensors, loading the model on first call.

        Positional args and `tokenizer_kwargs` are passed straight to the
        `transformers` tokenizer; `return_tensors="np"` is always set.
        """
        self._load()
        return self._tokenizer(*args, return_tensors="np", **tokenizer_kwargs)

    def _run(self, encoded: Any) -> Any:
        """Run one forward pass, feeding only the graph's declared inputs.

        Returns the first output tensor of the ONNX graph (the logits).
        """
        feed = {i.name: encoded[i.name] for i in self._session.get_inputs()}
        return self._session.run(None, feed)[0]


__all__ = ["OnnxEncoderBackend"]
