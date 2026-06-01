# fitz_sage/llm/providers/onnx_chat.py
"""
In-process ONNX chat provider for the required local Qwen enrichment model.

This is deliberately narrow: Fitz owns one tiny local generation backend for
the enrichment/summarization spine, using a pre-built Qwen3.5 0.8B ONNX graph
from Hugging Face and raw ``onnxruntime`` on CPU. No external server, no GGUF,
no llama.cpp, no torch.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_QWEN_MODEL_ALIAS = "qwen3.5-0.8b"
DEFAULT_QWEN_MODEL_ID = "onnx-community/Qwen3.5-0.8B-Text-ONNX"
DEFAULT_QWEN_ONNX_SUBFOLDER = "onnx"
DEFAULT_QWEN_ONNX_FILE = "model_q4.onnx"
DEFAULT_MAX_NEW_TOKENS = 512

_MODEL_ALIASES: dict[str, tuple[str, str, str]] = {
    DEFAULT_QWEN_MODEL_ALIAS: (
        DEFAULT_QWEN_MODEL_ID,
        DEFAULT_QWEN_ONNX_SUBFOLDER,
        DEFAULT_QWEN_ONNX_FILE,
    )
}

_SPECIAL_INPUTS = {"input_ids", "attention_mask", "num_logits_to_keep"}
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

logger = logging.getLogger(__name__)


def _strip_thinking(text: str) -> str:
    """Remove Qwen reasoning blocks from completed responses."""
    text = _THINK_RE.sub("", text)
    if "<think>" in text:
        text = (
            text.split("</think>")[-1].lstrip()
            if "</think>" in text
            else text.split("<think>")[0].rstrip()
        )
    return text


def _token_text(value: Any) -> str | None:
    """Extract a token string from tokenizer_config values."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        return str(content) if content is not None else None
    return None


def _node_arg_dtype(node_arg: Any) -> Any:
    """Map an ONNX Runtime node type string to a numpy dtype."""
    if node_arg.type == "tensor(float16)":
        return np.float16
    if node_arg.type == "tensor(float)":
        return np.float32
    if node_arg.type == "tensor(int64)":
        return np.int64
    raise RuntimeError(f"Unsupported ONNX input type for {node_arg.name}: {node_arg.type}")


def _present_to_past_name(name: str) -> str | None:
    """Map Qwen ONNX present-state output names back to past-state input names."""
    if name.startswith("present_conv."):
        return "past_conv." + name.split(".", 1)[1]
    if name.startswith("present_recurrent."):
        return "past_recurrent." + name.split(".", 1)[1]
    if name.startswith("present."):
        return "past_key_values." + name.split(".", 1)[1]
    return None


class OnnxChat:
    """Greedy CPU generation over the Fitz-managed Qwen3.5 0.8B ONNX graph."""

    def __init__(
        self,
        model_id: str = DEFAULT_QWEN_MODEL_ALIAS,
        *,
        onnx_subfolder: str | None = None,
        onnx_file: str | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        resolved = _MODEL_ALIASES.get(model_id)
        if resolved:
            self._model_id, default_subfolder, default_file = resolved
            self._model_name = model_id
        else:
            self._model_id = model_id
            default_subfolder = DEFAULT_QWEN_ONNX_SUBFOLDER
            default_file = DEFAULT_QWEN_ONNX_FILE
            self._model_name = model_id

        self._onnx_subfolder = onnx_subfolder if onnx_subfolder is not None else default_subfolder
        self._onnx_file = onnx_file if onnx_file is not None else default_file
        self._max_new_tokens = max_new_tokens
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._tokenizer: Any = None
        self._session: Any = None
        self._input_args: list[Any] = []
        self._output_names: list[str] = []
        self._eos_token_id: int | None = None

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Generate a chat completion with deterministic local ONNX inference."""
        self._load()
        max_new_tokens = int(
            kwargs.pop("max_tokens", kwargs.pop("max_new_tokens", self._max_new_tokens))
        )
        temperature = float(kwargs.pop("temperature", 0.0) or 0.0)
        top_p = float(kwargs.pop("top_p", 1.0) or 1.0)
        stop = kwargs.pop("stop", None)

        prompt = self._format_messages(messages)
        encoded = self._tokenizer(prompt, return_tensors="np")
        input_ids = encoded["input_ids"].astype(np.int64)
        if input_ids.shape[0] != 1:
            raise RuntimeError("ONNX Qwen chat only supports batch size 1")

        with self._run_lock:
            generated = self._generate_ids(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        text = _strip_thinking(text).strip()
        return self._apply_stop(text, stop)

    def _load(self) -> None:
        """Load tokenizer and ONNX session once per process."""
        if self._tokenizer is not None and self._session is not None:
            return
        with self._load_lock:
            if self._tokenizer is not None and self._session is not None:
                return

            import onnxruntime as ort
            from huggingface_hub import snapshot_download

            logger.info(
                "Loading ONNX chat model %s (%s/%s)",
                self._model_id,
                self._onnx_subfolder,
                self._onnx_file,
            )
            allow_patterns = [
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "config.json",
                "generation_config.json",
                f"{self._onnx_subfolder}/{self._onnx_file}",
                f"{self._onnx_subfolder}/{self._onnx_file}_data*",
            ]
            snapshot_dir = Path(
                snapshot_download(repo_id=self._model_id, allow_patterns=allow_patterns)
            )
            self._tokenizer = self._load_tokenizer(snapshot_dir)
            self._eos_token_id = self._tokenizer.eos_token_id
            onnx_path = snapshot_dir / self._onnx_subfolder / self._onnx_file
            self._session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            self._input_args = list(self._session.get_inputs())
            self._output_names = [output.name for output in self._session.get_outputs()]

    def _load_tokenizer(self, snapshot_dir: Path) -> Any:
        """Load Qwen tokenizer without requiring a model-specific Python class."""
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(str(snapshot_dir))
        except ValueError:
            from transformers import PreTrainedTokenizerFast

            config_path = snapshot_dir / "tokenizer_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(snapshot_dir / "tokenizer.json"),
                bos_token=_token_text(config.get("bos_token")),
                eos_token=_token_text(config.get("eos_token")),
                pad_token=_token_text(config.get("pad_token")),
                unk_token=_token_text(config.get("unk_token")),
            )
            tokenizer.chat_template = config.get("chat_template")
            return tokenizer

    def _format_messages(self, messages: list[dict[str, Any]]) -> str:
        """Apply the model chat template, disabling Qwen thinking by default."""
        if getattr(self._tokenizer, "chat_template", None):
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        rendered: list[str] = []
        for message in messages:
            rendered.append(f"{message.get('role', 'user')}: {message.get('content', '')}")
        rendered.append("assistant:")
        return "\n".join(rendered)

    def _generate_ids(
        self,
        *,
        input_ids: np.ndarray,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[int]:
        """Run an autoregressive ONNX loop and return generated token ids."""
        states = self._initial_states()
        total_length = int(input_ids.shape[1])
        attention_mask = np.ones((1, total_length), dtype=np.int64)
        generated: list[int] = []

        for _ in range(max_new_tokens):
            feed: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if self._has_input("num_logits_to_keep"):
                feed["num_logits_to_keep"] = np.array(1, dtype=np.int64)
            feed.update(states)

            outputs = self._session.run(None, feed)
            by_name = dict(zip(self._output_names, outputs))
            token_id = self._select_next_token(
                by_name["logits"][:, -1, :],
                temperature=temperature,
                top_p=top_p,
            )
            if self._eos_token_id is not None and token_id == self._eos_token_id:
                break
            generated.append(token_id)

            states = self._next_states(by_name)
            input_ids = np.array([[token_id]], dtype=np.int64)
            total_length += 1
            attention_mask = np.ones((1, total_length), dtype=np.int64)

        return generated

    def _initial_states(self) -> dict[str, np.ndarray]:
        """Create zero-filled recurrent/KV state tensors for the first forward pass."""
        states: dict[str, np.ndarray] = {}
        for node_arg in self._input_args:
            if node_arg.name in _SPECIAL_INPUTS:
                continue
            states[node_arg.name] = np.zeros(
                self._input_shape(node_arg, past_sequence_length=0),
                dtype=_node_arg_dtype(node_arg),
            )
        return states

    def _input_shape(self, node_arg: Any, *, past_sequence_length: int) -> tuple[int, ...]:
        """Resolve symbolic ONNX dimensions for batch-1 generation."""
        dims: list[int] = []
        for dim in node_arg.shape:
            if dim in (None, "batch_size"):
                dims.append(1)
            elif dim == "past_sequence_length":
                dims.append(past_sequence_length)
            elif dim == "total_sequence_length":
                dims.append(max(1, past_sequence_length))
            elif isinstance(dim, str):
                raise RuntimeError(f"Unsupported symbolic dim {dim!r} for {node_arg.name}")
            else:
                dims.append(int(dim))
        return tuple(dims)

    def _next_states(self, outputs_by_name: dict[str, Any]) -> dict[str, np.ndarray]:
        """Convert present-state outputs into past-state inputs for the next token."""
        states: dict[str, np.ndarray] = {}
        for name, value in outputs_by_name.items():
            past_name = _present_to_past_name(name)
            if past_name:
                states[past_name] = value
        return states

    def _select_next_token(self, logits: np.ndarray, *, temperature: float, top_p: float) -> int:
        """Pick the next token using greedy decoding unless sampling is requested."""
        row = logits[0].astype(np.float64)
        if temperature <= 0:
            return int(np.argmax(row))

        row = row / max(temperature, 1e-6)
        row = row - np.max(row)
        probabilities = np.exp(row)
        probabilities = probabilities / probabilities.sum()

        if 0.0 < top_p < 1.0:
            order = np.argsort(probabilities)[::-1]
            cumulative = np.cumsum(probabilities[order])
            keep = cumulative <= top_p
            keep[0] = True
            mask = np.zeros_like(probabilities, dtype=bool)
            mask[order[keep]] = True
            probabilities = np.where(mask, probabilities, 0.0)
            probabilities = probabilities / probabilities.sum()

        return int(np.random.choice(len(probabilities), p=probabilities))

    def _has_input(self, name: str) -> bool:
        """Return whether the ONNX graph declares the named input."""
        return any(node_arg.name == name for node_arg in self._input_args)

    @staticmethod
    def _apply_stop(text: str, stop: Any) -> str:
        """Apply OpenAI-style stop string truncation."""
        if stop is None:
            return text
        stops = [stop] if isinstance(stop, str) else list(stop)
        cut = len(text)
        for marker in stops:
            if not marker:
                continue
            index = text.find(str(marker))
            if index >= 0:
                cut = min(cut, index)
        return text[:cut].rstrip()


__all__ = [
    "DEFAULT_QWEN_MODEL_ALIAS",
    "DEFAULT_QWEN_MODEL_ID",
    "DEFAULT_QWEN_ONNX_FILE",
    "OnnxChat",
]
