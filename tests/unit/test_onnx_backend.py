# tests/unit/test_onnx_backend.py
"""Tests for the shared ONNX encoder backend."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
from huggingface_hub.errors import EntryNotFoundError

from fitz_sage.encoders.onnx import OnnxEncoderBackend


@pytest.fixture(autouse=True)
def _clear_runtime_cache():
    with OnnxEncoderBackend._runtime_cache_lock:
        OnnxEncoderBackend._runtime_cache.clear()
    yield
    with OnnxEncoderBackend._runtime_cache_lock:
        OnnxEncoderBackend._runtime_cache.clear()


def _install_transformers_stub(
    monkeypatch,
    *,
    from_pretrained,
    tokenizer_fast=None,
) -> None:
    module = ModuleType("transformers")
    module.AutoTokenizer = SimpleNamespace(from_pretrained=from_pretrained)
    module.PreTrainedTokenizerFast = tokenizer_fast or (lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", module)


def _install_onnxruntime_stub(monkeypatch, inference_session) -> None:
    module = ModuleType("onnxruntime")
    module.InferenceSession = inference_session
    monkeypatch.setitem(sys.modules, "onnxruntime", module)


def test_load_downloads_external_data_sidecar(monkeypatch):
    """A split ONNX model pulls `<model>.onnx.data` before session creation."""
    calls: list[str] = []

    def fake_hf_hub_download(*, repo_id, filename, subfolder):
        calls.append(filename)
        return f"C:/cache/{filename}"

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)
    _install_transformers_stub(
        monkeypatch,
        from_pretrained=lambda model_id: SimpleNamespace(),
    )
    _install_onnxruntime_stub(
        monkeypatch,
        lambda path, providers: SimpleNamespace(path=path, providers=providers),
    )

    backend = OnnxEncoderBackend("acme/model", "model_quantized.onnx")
    backend._load()

    assert calls == ["model_quantized.onnx", "model_quantized.onnx.data"]
    assert backend._session.path == "C:/cache/model_quantized.onnx"


def test_load_ignores_missing_external_data_sidecar(monkeypatch):
    """Single-file ONNX repos still load when no sidecar exists."""
    calls: list[str] = []

    def fake_hf_hub_download(*, repo_id, filename, subfolder):
        calls.append(filename)
        if filename.endswith(".data"):
            raise EntryNotFoundError("missing")
        return f"C:/cache/{filename}"

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)
    _install_transformers_stub(
        monkeypatch,
        from_pretrained=lambda model_id: SimpleNamespace(),
    )
    _install_onnxruntime_stub(
        monkeypatch,
        lambda path, providers: SimpleNamespace(path=path, providers=providers),
    )

    backend = OnnxEncoderBackend("acme/model", "model_int8.onnx", onnx_subfolder="onnx")
    backend._load()

    assert calls == ["model_int8.onnx", "model_int8.onnx.data"]
    assert backend._session.path == "C:/cache/model_int8.onnx"


def test_identical_model_specs_share_one_native_runtime(monkeypatch):
    """Collections using the same encoder do not duplicate its native session."""
    session_calls: list[str] = []
    tokenizer = SimpleNamespace()

    def fake_hf_hub_download(*, repo_id, filename, subfolder):
        if filename.endswith(".data"):
            raise EntryNotFoundError("missing")
        return f"C:/cache/{filename}"

    def fake_inference_session(path, providers):
        session_calls.append(path)
        return SimpleNamespace(path=path, providers=providers)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)
    _install_transformers_stub(
        monkeypatch,
        from_pretrained=lambda model_id: tokenizer,
    )
    _install_onnxruntime_stub(monkeypatch, fake_inference_session)

    first = OnnxEncoderBackend("acme/model", "model_int8.onnx", onnx_subfolder="onnx")
    second = OnnxEncoderBackend("acme/model", "model_int8.onnx", onnx_subfolder="onnx")
    first._load()
    second._load()

    assert session_calls == ["C:/cache/model_int8.onnx"]
    assert first._session is second._session
    assert first._tokenizer is second._tokenizer is tokenizer


def test_load_falls_back_to_tokenizer_json(monkeypatch, tmp_path):
    """Repos with unavailable tokenizer wrappers can still load tokenizer.json."""
    tokenizer_json = tmp_path / "tokenizer.json"
    tokenizer_json.write_text("{}", encoding="utf-8")
    tokenizer_config = tmp_path / "tokenizer_config.json"
    tokenizer_config.write_text(
        """
        {
          "unk_token": "[UNK]",
          "sep_token": "[SEP]",
          "pad_token": "[PAD]",
          "cls_token": "[CLS]",
          "mask_token": "[MASK]"
        }
        """,
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_hf_hub_download(*, repo_id, filename, subfolder):
        calls.append(filename)
        if filename.endswith(".data"):
            raise EntryNotFoundError("missing")
        if filename == "tokenizer.json":
            return str(tokenizer_json)
        if filename == "tokenizer_config.json":
            return str(tokenizer_config)
        return f"C:/cache/{filename}"

    def fake_from_pretrained(model_id):
        raise ValueError("Tokenizer class TokenizersBackend does not exist")

    def fake_tokenizer_fast(*, tokenizer_file, **special_tokens):
        return SimpleNamespace(tokenizer_file=tokenizer_file, special_tokens=special_tokens)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)
    _install_transformers_stub(
        monkeypatch,
        from_pretrained=fake_from_pretrained,
        tokenizer_fast=fake_tokenizer_fast,
    )
    _install_onnxruntime_stub(
        monkeypatch,
        lambda path, providers: SimpleNamespace(path=path, providers=providers),
    )

    backend = OnnxEncoderBackend("acme/model", "model_quantized.onnx")
    backend._load()

    assert calls == [
        "model_quantized.onnx",
        "model_quantized.onnx.data",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    assert backend._tokenizer.tokenizer_file == str(tokenizer_json)
    assert backend._tokenizer.special_tokens["cls_token"] == "[CLS]"
