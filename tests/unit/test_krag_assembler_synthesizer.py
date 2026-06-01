# tests/unit/test_krag_assembler_synthesizer.py
"""
Unit tests for ContextAssembler and CodeSynthesizer in the Fitz KRAG engine.

Tests context formatting, token budget enforcement, language detection,
mixed-type grouping, answer generation, provenance building, and error handling.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fitz_sage.core import Answer, GenerationError
from fitz_sage.engines.fitz_krag.context.assembler import (
    CHARS_PER_TOKEN,
    ContextAssembler,
)
from fitz_sage.engines.fitz_krag.generation.synthesizer import (
    SYSTEM_PROMPT_GROUNDED,
    SYSTEM_PROMPT_OPEN,
    CodeSynthesizer,
)
from fitz_sage.engines.fitz_krag.types import Address, AddressKind, ReadResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config(
    max_context_tokens: int = 8000,
    include_file_header: bool = True,
    strict_grounding: bool = True,
    enable_citations: bool = True,
) -> MagicMock:
    """Build a mock FitzKragConfig with the given fields."""
    cfg = MagicMock()
    cfg.max_context_tokens = max_context_tokens
    cfg.include_file_header = include_file_header
    cfg.strict_grounding = strict_grounding
    cfg.enable_citations = enable_citations
    return cfg


def _make_result(
    kind: AddressKind = AddressKind.SYMBOL,
    source_id: str = "src_1",
    location: str = "mod.func",
    summary: str = "A function",
    file_path: str = "src/app.py",
    content: str = "def foo(): pass",
    line_range: tuple[int, int] | None = (10, 15),
    metadata: dict | None = None,
    address_metadata: dict | None = None,
) -> ReadResult:
    """Build a ReadResult with sensible defaults."""
    address = Address(
        kind=kind,
        source_id=source_id,
        location=location,
        summary=summary,
        metadata=address_metadata or {},
    )
    return ReadResult(
        address=address,
        content=content,
        file_path=file_path,
        line_range=line_range,
        metadata=metadata or {},
    )


# ===========================================================================
# TestContextAssembler
# ===========================================================================


class TestContextAssembler:
    """Tests for ContextAssembler."""

    # -- test_assemble_empty_results ----------------------------------------

    def test_assemble_empty_results(self) -> None:
        """Empty results list produces an empty string."""
        asm = ContextAssembler(_make_config())
        assert asm.assemble("any query", []) == ""

    # -- test_assemble_single_result ----------------------------------------

    def test_assemble_single_result(self) -> None:
        """Single result is formatted as [S1] with a fenced code block."""
        asm = ContextAssembler(_make_config())
        result = _make_result(
            content="def hello(): pass",
            file_path="src/hello.py",
            line_range=(1, 5),
        )
        output = asm.assemble("what is hello?", [result])

        assert "[S1]" in output
        assert "src/hello.py" in output
        assert "```python" in output
        assert "def hello(): pass" in output

    # -- test_assemble_multiple_results -------------------------------------

    def test_assemble_multiple_results(self) -> None:
        """Multiple results produce [S1] and [S2] blocks separated by blank lines."""
        asm = ContextAssembler(_make_config())
        r1 = _make_result(source_id="a", content="alpha", file_path="a.py")
        r2 = _make_result(source_id="b", content="beta", file_path="b.py")
        output = asm.assemble("query", [r1, r2])

        assert "[S1]" in output
        assert "[S2]" in output
        assert "alpha" in output
        assert "beta" in output

    # -- test_assemble_includes_line_range ----------------------------------

    def test_assemble_includes_line_range(self) -> None:
        """Line range is shown as '(lines 10-15)' in the header."""
        asm = ContextAssembler(_make_config())
        result = _make_result(line_range=(10, 15))
        output = asm.assemble("query", [result])

        assert "(lines 10-15)" in output

    # -- test_assemble_section_header ---------------------------------------

    def test_assemble_section_header(self) -> None:
        """SECTION results display section_title and page range."""
        asm = ContextAssembler(_make_config())
        result = _make_result(
            kind=AddressKind.SECTION,
            file_path="docs/guide.pdf",
            content="Installation instructions...",
            line_range=None,
            metadata={
                "section_title": "Getting Started",
                "page_start": 1,
                "page_end": 3,
            },
        )
        output = asm.assemble("how to install?", [result])

        assert "Getting Started" in output
        assert "(pages 1-3)" in output
        # SECTION kind uses empty language tag
        assert "```\n" in output

    # -- test_assemble_mixed_type_groups ------------------------------------

    def test_assemble_mixed_type_groups(self) -> None:
        """Mixed code + section results add '--- Code ---' and '--- Documentation ---' headers."""
        asm = ContextAssembler(_make_config())
        code_result = _make_result(
            kind=AddressKind.SYMBOL,
            content="class Engine: ...",
            file_path="engine.py",
        )
        doc_result = _make_result(
            kind=AddressKind.SECTION,
            content="The engine handles ...",
            file_path="docs/engine.md",
            line_range=None,
            metadata={"section_title": "Overview"},
        )
        output = asm.assemble("what is the engine?", [code_result, doc_result])

        assert "--- Code ---" in output
        assert "--- Documentation ---" in output

    # -- test_assemble_homogeneous_no_groups --------------------------------

    def test_assemble_homogeneous_no_groups(self) -> None:
        """All-code results do NOT include group headers."""
        asm = ContextAssembler(_make_config())
        r1 = _make_result(kind=AddressKind.SYMBOL, content="a", file_path="a.py")
        r2 = _make_result(kind=AddressKind.FILE, content="b", file_path="b.py")
        output = asm.assemble("query", [r1, r2])

        assert "---" not in output

    # -- test_assemble_respects_token_budget --------------------------------

    def test_assemble_respects_token_budget(self) -> None:
        """Content exceeding max_context_tokens budget is truncated with '(truncated)'."""
        tiny_budget = 50  # 50 tokens = 200 chars
        asm = ContextAssembler(_make_config(max_context_tokens=tiny_budget))

        # Create a result whose formatted block is much larger than 200 chars
        big_content = "x" * 500
        result = _make_result(content=big_content, file_path="big.py")
        output = asm.assemble("query", [result])

        assert "(truncated)" in output
        # Total output must be within budget + the truncation suffix
        # The assembler truncates at budget boundary, so check it is bounded
        max_chars = tiny_budget * CHARS_PER_TOKEN
        # The block before truncation suffix should be at most budget chars
        # (plus the suffix text itself)
        assert len(output) < max_chars + 100

    # -- test_assemble_language_detection ------------------------------------

    def test_assemble_language_detection(self) -> None:
        """File extensions map to correct language tags: .py -> python, .ts -> typescript."""
        asm = ContextAssembler(_make_config())

        py_result = _make_result(file_path="mod.py", content="pass")
        ts_result = _make_result(file_path="mod.ts", content="const x = 1")
        tsx_result = _make_result(file_path="comp.tsx", content="<div/>")
        js_result = _make_result(file_path="util.js", content="var a")
        go_result = _make_result(file_path="main.go", content="func main()")
        java_result = _make_result(file_path="App.java", content="class App")
        md_result = _make_result(file_path="README.md", content="# Title")
        unknown_result = _make_result(file_path="data.xyz", content="stuff")

        assert "```python" in asm.assemble("q", [py_result])
        assert "```typescript" in asm.assemble("q", [ts_result])
        assert "```typescript" in asm.assemble("q", [tsx_result])
        assert "```javascript" in asm.assemble("q", [js_result])
        assert "```go" in asm.assemble("q", [go_result])
        assert "```java" in asm.assemble("q", [java_result])
        assert "```markdown" in asm.assemble("q", [md_result])
        # Unknown extension => empty language tag -> just "```\n"
        unknown_output = asm.assemble("q", [unknown_result])
        assert "```\n" in unknown_output

    # -- test_assemble_no_file_header ---------------------------------------

    def test_assemble_no_file_header(self) -> None:
        """include_file_header=False omits file path from header."""
        asm = ContextAssembler(_make_config(include_file_header=False))
        result = _make_result(file_path="src/mod.py")
        output = asm.assemble("test", [result])

        assert "[S1]" in output
        assert "# src/mod.py" not in output

    # -- test_assemble_chunk_no_language ------------------------------------

    def test_assemble_chunk_no_language(self) -> None:
        """CHUNK addresses use a bare code fence without language hint."""
        asm = ContextAssembler(_make_config())
        addr = Address(kind=AddressKind.CHUNK, source_id="c1", location="doc", summary="")
        result = ReadResult(address=addr, content="some text", file_path="doc.pdf")
        output = asm.assemble("test", [result])

        assert "```\n" in output

    # -- test_assemble_kind_label ------------------------------------------

    def test_assemble_kind_label(self) -> None:
        """Address metadata 'kind' appears as a label in the header."""
        asm = ContextAssembler(_make_config())
        result = _make_result(address_metadata={"kind": "function"})
        output = asm.assemble("test", [result])

        assert "[function]" in output

    # -- test_assemble_section_single_page ---------------------------------

    def test_assemble_section_single_page(self) -> None:
        """Single-page section shows 'page 1' not 'pages 1-1'."""
        asm = ContextAssembler(_make_config())
        result = _make_result(
            kind=AddressKind.SECTION,
            file_path="doc.pdf",
            content="Introduction text.",
            line_range=None,
            metadata={
                "page_start": 1,
                "page_end": 1,
                "section_title": "Introduction",
                "section_level": 1,
            },
        )
        output = asm.assemble("test", [result])

        assert "page 1" in output
        assert "pages" not in output

    # -- test_assemble_context_type_shown -----------------------------------

    def test_assemble_context_type_shown(self) -> None:
        """context_type metadata is shown in parentheses in the header."""
        asm = ContextAssembler(_make_config())
        result = _make_result(
            metadata={"context_type": "import_context"},
        )
        output = asm.assemble("query", [result])

        assert "(import_context)" in output


# ===========================================================================
# TestCodeSynthesizer
# ===========================================================================


class TestCodeSynthesizer:
    """Tests for CodeSynthesizer."""

    # -- test_generate_happy_path -------------------------------------------

    def test_generate_happy_path(self) -> None:
        """generate() returns an Answer with text and provenance."""
        chat = MagicMock()
        chat.chat.return_value = "The function does X [S1]."
        config = _make_config(strict_grounding=True)
        synth = CodeSynthesizer(chat=chat, config=config)

        result = _make_result(source_id="src_1")
        answer = synth.generate("what does foo do?", "context...", [result])

        assert isinstance(answer, Answer)
        assert answer.text == "The function does X [S1]."
        assert len(answer.provenance) == 1
        chat.chat.assert_called_once()
        assert chat.chat.call_args.kwargs["max_tokens"] == config.max_answer_tokens

    # -- test_generate_strict_grounding -------------------------------------

    def test_generate_strict_grounding(self) -> None:
        """strict_grounding=True sends SYSTEM_PROMPT_GROUNDED as system message."""
        chat = MagicMock()
        chat.chat.return_value = "answer"
        config = _make_config(strict_grounding=True)
        synth = CodeSynthesizer(chat=chat, config=config)

        synth.generate("q", "ctx", [])

        messages = chat.chat.call_args[0][0]
        system_msg = messages[0]
        assert system_msg["role"] == "system"
        assert SYSTEM_PROMPT_GROUNDED in system_msg["content"]

    # -- test_generate_open_grounding ---------------------------------------

    def test_generate_open_grounding(self) -> None:
        """strict_grounding=False sends SYSTEM_PROMPT_OPEN as system message."""
        chat = MagicMock()
        chat.chat.return_value = "answer"
        config = _make_config(strict_grounding=False)
        synth = CodeSynthesizer(chat=chat, config=config)

        synth.generate("q", "ctx", [])

        messages = chat.chat.call_args[0][0]
        system_msg = messages[0]
        assert system_msg["role"] == "system"
        assert SYSTEM_PROMPT_OPEN in system_msg["content"]

    # -- test_generate_llm_error_raises_generation_error --------------------

    def test_generate_llm_error_raises_generation_error(self) -> None:
        """When chat.chat() raises, CodeSynthesizer wraps it in GenerationError."""
        chat = MagicMock()
        chat.chat.side_effect = RuntimeError("LLM unavailable")
        config = _make_config()
        synth = CodeSynthesizer(chat=chat, config=config)

        with pytest.raises(GenerationError, match="LLM generation failed"):
            synth.generate("q", "ctx", [])

    # -- test_provenance_from_results ---------------------------------------

    def test_provenance_from_results(self) -> None:
        """Provenance entries are built with correct source_id and excerpt."""
        chat = MagicMock()
        chat.chat.return_value = "answer"
        config = _make_config()
        synth = CodeSynthesizer(chat=chat, config=config)

        r = _make_result(
            source_id="file_42",
            content="def longfunc():\n    " + "x" * 300,
            line_range=(5, 20),
            address_metadata={"kind": "function"},
        )
        answer = synth.generate("q", "ctx", [r])

        assert len(answer.provenance) == 1
        prov = answer.provenance[0]
        assert prov.source_id == "file_42"
        # excerpt is first 200 chars
        assert len(prov.excerpt) == 200
        assert prov.excerpt == r.content[:200]

    # -- test_provenance_skips_context_type ---------------------------------

    def test_provenance_skips_context_type(self) -> None:
        """Results whose metadata has 'context_type' are excluded from provenance."""
        chat = MagicMock()
        chat.chat.return_value = "answer"
        config = _make_config()
        synth = CodeSynthesizer(chat=chat, config=config)

        primary = _make_result(source_id="primary", metadata={})
        expansion = _make_result(
            source_id="expansion",
            metadata={"context_type": "import_context"},
        )
        answer = synth.generate("q", "ctx", [primary, expansion])

        source_ids = [p.source_id for p in answer.provenance]
        assert "primary" in source_ids
        assert "expansion" not in source_ids

    # -- test_provenance_includes_line_range --------------------------------

    def test_provenance_includes_line_range(self) -> None:
        """Provenance metadata includes line_range as a list when present."""
        chat = MagicMock()
        chat.chat.return_value = "answer"
        config = _make_config()
        synth = CodeSynthesizer(chat=chat, config=config)

        r = _make_result(line_range=(10, 25))
        answer = synth.generate("q", "ctx", [r])

        prov = answer.provenance[0]
        assert prov.metadata["line_range"] == [10, 25]

    # -- test_generate_metadata ---------------------------------------------

    def test_generate_metadata(self) -> None:
        """Answer metadata contains engine, query, and sources_count."""
        chat = MagicMock()
        chat.chat.return_value = "answer"
        config = _make_config()
        synth = CodeSynthesizer(chat=chat, config=config)

        r1 = _make_result(source_id="a")
        r2 = _make_result(source_id="b")
        answer = synth.generate("my question", "ctx", [r1, r2])

        assert answer.metadata["engine"] == "fitz_krag"
        assert answer.metadata["query"] == "my question"
        assert answer.metadata["sources_count"] == 2
