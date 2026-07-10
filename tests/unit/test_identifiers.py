"""Tests for literal structured-identifier handling."""

from fitz_sage.core.identifiers import contains_exact_identifier, exact_identifiers


def test_exact_identifiers_preserve_spelling_and_separator() -> None:
    assert exact_identifiers("Compare AX-156 with MOD_88X") == ["AX-156", "MOD_88X"]


def test_exact_identifier_match_rejects_neighbors_and_variants() -> None:
    assert contains_exact_identifier("Ticket AX-156 is closed.", "AX-156")
    assert not contains_exact_identifier("Ticket AX-156B is closed.", "AX-156")
    assert not contains_exact_identifier("Ticket AX-156-B is closed.", "AX-156")
    assert not contains_exact_identifier("Ticket AX-156.2 is closed.", "AX-156")
    assert not contains_exact_identifier("Ticket AX_156 is closed.", "AX-156")
    assert not contains_exact_identifier("Ticket AX 156 is closed.", "AX-156")
