"""REST API trust-boundary and request-lifecycle tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from fitz_sage.api.app import create_app
from fitz_sage.api.security import (
    ALLOWED_ORIGINS_ENV,
    API_KEY_ENV,
    SOURCE_ROOTS_ENV,
    _is_loopback,
    allowed_api_origins,
    resolve_api_source,
)
from fitz_sage.core import Answer
from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.services.fitz_service import CollectionNotFoundError


def test_remote_request_requires_configured_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)

    response = TestClient(create_app()).get("/openapi.json")

    assert response.status_code == 403
    assert API_KEY_ENV in response.json()["detail"]


def test_remote_request_accepts_matching_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "secret")
    client = TestClient(create_app())

    assert client.get("/openapi.json").status_code == 401
    assert client.get("/openapi.json", headers={"X-Fitz-API-Key": "secret"}).status_code == 200


def test_loopback_detection() -> None:
    assert _is_loopback(("127.0.0.1", 1234))
    assert _is_loopback(("::1", 1234))
    assert not _is_loopback(("192.0.2.10", 1234))


def test_cors_origins_are_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ALLOWED_ORIGINS_ENV, raising=False)
    assert allowed_api_origins() == []


def test_cors_origins_are_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        ALLOWED_ORIGINS_ENV,
        "https://app.example, http://localhost:3000",
    )
    assert allowed_api_origins() == ["https://app.example", "http://localhost:3000"]


def test_api_source_must_stay_within_allowed_roots(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    inside = root / "inside.txt"
    inside.write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    monkeypatch.setenv(SOURCE_ROOTS_ENV, str(root))

    assert resolve_api_source(inside) == inside.resolve()
    with pytest.raises(ValueError, match="outside the allowed API roots"):
        resolve_api_source(outside)


def test_answer_source_is_searchable_when_point_returns(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    monkeypatch.setenv(API_KEY_ENV, "secret")
    monkeypatch.setenv(SOURCE_ROOTS_ENV, str(tmp_path))
    service = MagicMock()
    service.answer.return_value = Answer(
        text="answer",
        mode=AnswerMode.SUFFICIENT,
        provenance=[],
    )

    with patch("fitz_sage.api.routes.query.get_service", return_value=service):
        response = TestClient(create_app()).post(
            "/answer",
            headers={"X-Fitz-API-Key": "secret"},
            json={"question": "question", "source": str(source), "collection": "docs"},
        )

    assert response.status_code == 200
    service.point.assert_called_once_with(source=source.resolve(), collection="docs")
    service.answer.assert_called_once()


def test_query_synthesis_route_is_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "secret")

    response = TestClient(create_app()).post(
        "/query",
        headers={"X-Fitz-API-Key": "secret"},
        json={"question": "question", "collection": "docs"},
    )

    assert response.status_code == 404


def test_missing_collection_maps_to_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "secret")
    service = MagicMock()
    service.get_collection.side_effect = CollectionNotFoundError("missing")

    with patch("fitz_sage.api.routes.collections.get_service", return_value=service):
        response = TestClient(create_app()).get(
            "/collections/missing",
            headers={"X-Fitz-API-Key": "secret"},
        )

    assert response.status_code == 404
