# tests/unit/llm/test_onnx_chat.py
"""Tests for the managed Qwen ONNX chat provider lifecycle."""

from __future__ import annotations

import json
from hashlib import sha256
from types import SimpleNamespace

import pytest

from fitz_sage.llm.providers import onnx_chat as onnx_chat_module
from fitz_sage.llm.providers.onnx_chat import (
    DEFAULT_QWEN_MODEL_ID,
    DEFAULT_QWEN_ONNX_FILE,
    DEFAULT_QWEN_ONNX_SUBFOLDER,
    GenAiRuntimeBundle,
    OnnxChat,
    OnnxChatModelError,
)


@pytest.fixture(autouse=True)
def _clear_runtime_cache():
    with onnx_chat_module._ONNX_CHAT_RUNTIME_CACHE_LOCK:
        onnx_chat_module._ONNX_CHAT_RUNTIME_CACHE.clear()
    yield
    with onnx_chat_module._ONNX_CHAT_RUNTIME_CACHE_LOCK:
        onnx_chat_module._ONNX_CHAT_RUNTIME_CACHE.clear()


def _snapshot(tmp_path, *, with_model: bool = True):
    snapshot = tmp_path / "models--qwen" / "snapshots" / "abc123"
    (snapshot / DEFAULT_QWEN_ONNX_SUBFOLDER).mkdir(parents=True)
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    (snapshot / "genai_config.json").write_text(
        json.dumps(
            {
                "model": {
                    "decoder": {
                        "filename": "model.onnx",
                        "session_options": {
                            "provider_options": [{"webgpu": {"forceCpuNodeNames": "x"}}]
                        },
                    }
                },
                "search": {"do_sample": True, "temperature": 1, "max_length": 40960},
            }
        ),
        encoding="utf-8",
    )
    model_bytes = b"fake-onnx"
    data_bytes = b"fake-weights"
    if with_model:
        (snapshot / DEFAULT_QWEN_ONNX_SUBFOLDER / DEFAULT_QWEN_ONNX_FILE).write_bytes(model_bytes)
        (snapshot / DEFAULT_QWEN_ONNX_SUBFOLDER / f"{DEFAULT_QWEN_ONNX_FILE}_data").write_bytes(
            data_bytes
        )
    return snapshot, model_bytes, data_bytes


@pytest.fixture(autouse=True)
def _stub_genai_runtime(monkeypatch):
    """Snapshot lifecycle tests should not require the native GenAI wheel."""
    monkeypatch.setattr(GenAiRuntimeBundle, "require_genai", staticmethod(lambda: object()))


def test_ensure_available_returns_inspectable_snapshot_metadata(monkeypatch, tmp_path):
    """The provider exposes the resolved repo, revision, path, size, and checksum."""
    snapshot, model_bytes, data_bytes = _snapshot(tmp_path)
    calls = {}

    def fake_snapshot_download(*, repo_id, allow_patterns):
        calls["repo_id"] = repo_id
        calls["allow_patterns"] = allow_patterns
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    chat = OnnxChat()
    info = chat.ensure_available(include_checksum=True)

    assert calls["repo_id"] == DEFAULT_QWEN_MODEL_ID
    assert f"{DEFAULT_QWEN_ONNX_SUBFOLDER}/{DEFAULT_QWEN_ONNX_FILE}" in calls["allow_patterns"]
    assert (
        f"{DEFAULT_QWEN_ONNX_SUBFOLDER}/{DEFAULT_QWEN_ONNX_FILE}_data*" in calls["allow_patterns"]
    )
    assert "genai_config.json" in calls["allow_patterns"]
    assert info.name == "qwen3-0.6b"
    assert info.repo_id == DEFAULT_QWEN_MODEL_ID
    assert info.revision == "abc123"
    model_path = snapshot / DEFAULT_QWEN_ONNX_SUBFOLDER / DEFAULT_QWEN_ONNX_FILE
    data_path = snapshot / DEFAULT_QWEN_ONNX_SUBFOLDER / f"{DEFAULT_QWEN_ONNX_FILE}_data"
    assert info.onnx_path == str(model_path)
    assert info.external_data_paths == [str(data_path)]
    assert info.onnx_bytes == len(model_bytes)
    assert info.total_bytes == len(model_bytes) + len(data_bytes)
    digest = sha256()
    for path, content in ((model_path, model_bytes), (data_path, data_bytes)):
        digest.update(path.name.encode("utf-8"))
        digest.update(content)
    assert info.bundle_sha256 == digest.hexdigest()
    assert info.as_dict()["repo_id"] == DEFAULT_QWEN_MODEL_ID


def test_prepare_genai_runtime_writes_cpu_config(monkeypatch, tmp_path):
    """The managed runtime uses a CPU GenAI config without mutating the HF snapshot."""
    snapshot, _, _ = _snapshot(tmp_path)

    def fake_snapshot_download(*, repo_id, allow_patterns):
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(
        "fitz_sage.core.paths.FitzPaths.user_home",
        classmethod(lambda cls: tmp_path / ".fitz"),
    )

    chat = OnnxChat()
    info = chat.ensure_available()
    runtime_dir = chat._prepare_genai_runtime(info)

    runtime_config = json.loads((runtime_dir / "genai_config.json").read_text(encoding="utf-8"))
    assert runtime_config["model"]["decoder"]["filename"] == (
        f"{DEFAULT_QWEN_ONNX_SUBFOLDER}/{DEFAULT_QWEN_ONNX_FILE}"
    )
    assert runtime_config["model"]["decoder"]["session_options"]["provider_options"] == []
    assert runtime_config["search"]["do_sample"] is False
    assert runtime_config["search"]["temperature"] == 0
    assert runtime_config["search"]["max_length"] == 8192
    assert (runtime_dir / DEFAULT_QWEN_ONNX_SUBFOLDER / DEFAULT_QWEN_ONNX_FILE).exists()
    source_config = json.loads((snapshot / "genai_config.json").read_text(encoding="utf-8"))
    assert source_config["model"]["decoder"]["filename"] == "model.onnx"


def test_ensure_available_rejects_incomplete_snapshot(monkeypatch, tmp_path):
    """A partial HF cache produces an actionable managed-model error."""
    snapshot, _, _ = _snapshot(tmp_path, with_model=False)

    def fake_snapshot_download(*, repo_id, allow_patterns):
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    chat = OnnxChat()
    with pytest.raises(OnnxChatModelError, match="snapshot is incomplete"):
        chat.ensure_available()


def test_ensure_available_wraps_download_failures(monkeypatch):
    """Network/cache failures are surfaced as managed Qwen lifecycle errors."""

    def fake_snapshot_download(*, repo_id, allow_patterns):
        raise RuntimeError("offline")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    chat = OnnxChat()
    with pytest.raises(OnnxChatModelError, match="Could not download"):
        chat.ensure_available()


def test_identical_model_specs_share_one_native_runtime(monkeypatch, tmp_path):
    """Collection-local providers share one heavyweight GenAI model."""
    model_calls: list[str] = []
    model = object()
    genai_tokenizer = object()
    info = SimpleNamespace(snapshot_dir=str(tmp_path))

    def fake_model(path):
        model_calls.append(path)
        return model

    runtime = SimpleNamespace(
        Model=fake_model,
        Tokenizer=lambda loaded_model: genai_tokenizer,
    )
    monkeypatch.setattr(GenAiRuntimeBundle, "require_genai", staticmethod(lambda: runtime))
    monkeypatch.setattr(
        GenAiRuntimeBundle,
        "prepare",
        lambda self, model_info: tmp_path,
    )
    monkeypatch.setattr(
        OnnxChat,
        "ensure_available",
        lambda self, include_checksum=False: info,
    )
    monkeypatch.setattr(
        OnnxChat,
        "_load_tokenizer",
        lambda self, snapshot_dir: object(),
    )

    first = OnnxChat()
    second = OnnxChat()
    first._load()
    second._load()

    assert model_calls == [str(tmp_path)]
    assert first._loaded_runtime is second._loaded_runtime
