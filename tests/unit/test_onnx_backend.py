# tests/unit/test_onnx_backend.py
"""Tests for the shared ONNX encoder backend."""

from __future__ import annotations

from types import SimpleNamespace

from huggingface_hub.errors import EntryNotFoundError

from fitz_sage.encoders.onnx import OnnxEncoderBackend


def test_load_downloads_external_data_sidecar(monkeypatch):
    """A split ONNX model pulls `<model>.onnx.data` before session creation."""
    calls: list[str] = []

    def fake_hf_hub_download(*, repo_id, filename, subfolder):
        calls.append(filename)
        return f"C:/cache/{filename}"

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda model_id: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "onnxruntime.InferenceSession",
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
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda model_id: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "onnxruntime.InferenceSession",
        lambda path, providers: SimpleNamespace(path=path, providers=providers),
    )

    backend = OnnxEncoderBackend("acme/model", "model_int8.onnx", onnx_subfolder="onnx")
    backend._load()

    assert calls == ["model_int8.onnx", "model_int8.onnx.data"]
    assert backend._session.path == "C:/cache/model_int8.onnx"


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
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", fake_from_pretrained)
    monkeypatch.setattr("transformers.PreTrainedTokenizerFast", fake_tokenizer_fast)
    monkeypatch.setattr(
        "onnxruntime.InferenceSession",
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
