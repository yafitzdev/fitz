from fitz_ingest.ingester.plugins.base import RawDocument
from fitz_ingest.ingester.validation import validate

def test_validation_filters_empty_documents():
    valid = RawDocument(path="a.txt", content="hello", metadata={})
    empty = RawDocument(path="b.txt", content="   ", metadata={})

    result = validate([valid, empty])

    assert len(result) == 1
    assert result[0].path == "a.txt"
