"""Malformed-query tests for the retrieval-first surface."""

from __future__ import annotations

import pytest

from fitz_sage.core import Query

pytestmark = pytest.mark.security


@pytest.fixture
def engine(krag_e2e_runner):
    return krag_e2e_runner.engine


def test_empty_query_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        Query(text="")


@pytest.mark.parametrize("text", ["   ", "\t\n"])
def test_whitespace_only_query_is_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        Query(text=text)


def test_unicode_query_does_not_crash(engine) -> None:
    pack = engine.evidence(Query(text="What is TechCorp? 你好 🚗"))

    assert pack is not None


def test_special_characters_are_data_not_commands(engine) -> None:
    pack = engine.evidence(
        Query(
            text="TechCorp; DROP TABLE users;-- <script>alert('xss')</script>"
            "???????? What\n\n  is    TechCorp?"
        )
    )

    assert pack is not None
    assert all("<script>" not in (item.content or "") for item in pack.items)


def test_very_long_query_is_bounded(engine) -> None:
    pack = engine.evidence(Query(text="What is TechCorp? " * 100))

    assert len(pack.items) <= engine.config.top_read


def test_single_very_long_token_does_not_crash(engine) -> None:
    pack = engine.evidence(Query(text=f"What is {'a' * 2000}?"))

    assert pack is not None
