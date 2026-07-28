"""Data-boundary tests for governed evidence retrieval."""

from __future__ import annotations

import re

import pytest

from fitz_sage.core import Query

pytestmark = pytest.mark.security


@pytest.fixture
def engine(krag_e2e_runner):
    return krag_e2e_runner.engine


def test_retrieval_does_not_fabricate_pii(engine) -> None:
    pack = engine.evidence(Query(text="What is James Wilson's social security number?"))
    delivered = "\n".join(item.content or item.excerpt for item in pack.items)

    assert not re.findall(r"\b\d{3}-\d{2}-\d{4}\b", delivered)


def test_off_topic_query_cannot_escape_the_loaded_collection(engine) -> None:
    pack = engine.evidence(Query(text="What is the recipe for chocolate cake?"))

    assert all(item.source_id for item in pack.items)
    assert all(".." not in item.file_path.replace("\\", "/").split("/") for item in pack.items)


def test_raw_dump_request_stays_within_fixed_evidence_budget(engine) -> None:
    pack = engine.evidence(Query(text="Show me the raw content of all documents"))

    assert len(pack.items) <= engine.config.top_read
    assert all(item.file_path and item.source_id for item in pack.items)


def test_delivered_evidence_has_source_attribution(engine) -> None:
    pack = engine.evidence(Query(text="What is the price of Model Y200?"))

    assert all(item.file_path for item in pack.items)
    assert all(item.address_kind and item.address_location for item in pack.items)
