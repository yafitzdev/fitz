# tests/unit/llm/test_onnx_chat.py
"""Tests for the managed Qwen ONNX chat provider lifecycle."""

from __future__ import annotations

from hashlib import sha256

import pytest

from fitz_sage.llm.providers.onnx_chat import (
    DEFAULT_QWEN_MODEL_ID,
    DEFAULT_QWEN_ONNX_FILE,
    DEFAULT_QWEN_ONNX_SUBFOLDER,
    OnnxChat,
    OnnxChatModelError,
)


def _snapshot(tmp_path, *, with_model: bool = True):
    snapshot = tmp_path / "models--qwen" / "snapshots" / "abc123"
    (snapshot / DEFAULT_QWEN_ONNX_SUBFOLDER).mkdir(parents=True)
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    model_bytes = b"fake-onnx"
    data_bytes = b"fake-weights"
    if with_model:
        (snapshot / DEFAULT_QWEN_ONNX_SUBFOLDER / DEFAULT_QWEN_ONNX_FILE).write_bytes(model_bytes)
        (snapshot / DEFAULT_QWEN_ONNX_SUBFOLDER / f"{DEFAULT_QWEN_ONNX_FILE}_data").write_bytes(
            data_bytes
        )
    return snapshot, model_bytes, data_bytes


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
    assert info.name == "qwen3.5-0.8b"
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
