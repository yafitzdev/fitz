"""
Tests for the new context pipeline + step modules
"""

from __future__ import annotations

from fitz_rag.context.steps.normalize import _normalize_text
from fitz_rag.context.steps.dedupe import DedupeStep
from fitz_rag.context.steps.group import GroupByDocumentStep
from fitz_rag.context.steps.merge import MergeAdjacentStep
from fitz_rag.context.steps.pack import PackWindowStep
from fitz_rag.context.steps.render_markdown import RenderMarkdownStep

from fitz_rag.context.pipeline import ContextPipeline


# ---------------------------------------------------------
# Normalization
# ---------------------------------------------------------
def test_normalize_text():
    text = "  Hello\tWorld \r\n"
    out = _normalize_text(text)
    assert out == "Hello World"


# ---------------------------------------------------------
# Deduplication
# ---------------------------------------------------------
def test_dedupe_chunks():
    step = DedupeStep()
    chunks = [
        {"text": "A", "file": "f1"},
        {"text": "A ", "file": "f1"},  # normalized duplicate
        {"text": "B", "file": "f1"},
    ]

    out = step(chunks)

    assert len(out) == 2
    assert out[0]["text"] == "A"
    assert out[1]["text"] == "B"


# ---------------------------------------------------------
# Grouping by document
# ---------------------------------------------------------
def test_group_by_document():
    step = GroupByDocumentStep()
    chunks = [
        {"text": "A", "file": "doc1"},
        {"text": "B", "file": "doc1"},
        {"text": "C", "file": "doc2"},
    ]

    grouped = step(chunks)

    assert set(grouped.keys()) == {"doc1", "doc2"}
    assert len(grouped["doc1"]) == 2
    assert len(grouped["doc2"]) == 1


# ---------------------------------------------------------
# Merging adjacent chunks
# ---------------------------------------------------------
def test_merge_adjacent_chunks():
    step = MergeAdjacentStep()
    chunks = [
        {"text": "A1", "file": "doc1"},
        {"text": "A2", "file": "doc1"},
        {"text": "B1", "file": "doc2"},
        {"text": "B2", "file": "doc2"},
    ]

    out = step(chunks)

    assert len(out) == 2
    assert "A1" in out[0]["text"]
    assert "A2" in out[0]["text"]
    assert "B1" in out[1]["text"]
    assert "B2" in out[1]["text"]


# ---------------------------------------------------------
# Packing window
# ---------------------------------------------------------
def test_pack_context_window():
    step = PackWindowStep(max_chars=25)
    chunks = [
        {"text": "A" * 10, "file": "f1"},
        {"text": "B" * 10, "file": "f1"},
        {"text": "C" * 10, "file": "f1"},
    ]

    out = step(chunks)

    # A(10) + B(10) = 20 < 25 → pick 2 chunks
    assert len(out) == 2


# ---------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------
def test_full_context_pipeline():
    pipeline = ContextPipeline(max_chars=1000)

    chunks = [
        {"text": "A1", "file": "doc1"},
        {"text": "A2", "file": "doc1"},
        {"text": "B1", "file": "doc2"},
    ]

    ctx = pipeline.build(chunks)

    assert "### Source: doc1" in ctx
    assert "### Source: doc2" in ctx
    assert "A1" in ctx and "A2" in ctx
    assert "B1" in ctx
