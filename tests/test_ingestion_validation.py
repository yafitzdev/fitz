import pytest
from fitz_ingest.ingester.validation import (
    IngestionValidator,
    IngestionValidationError,
    ChunkValidationConfig,
)


# Helper: minimal mock chunk
class MockChunk:
    def __init__(self, text, metadata):
        self.text = text
        self.metadata = metadata


# ---------------------------------------------------------
# Test: valid chunks pass
# ---------------------------------------------------------

def test_validation_passes_on_valid_chunks():
    validator = IngestionValidator()

    chunks = [
        MockChunk("hello world", {"doc_id": "1", "chunk_index": 0}),
        MockChunk("another chunk", {"doc_id": "1", "chunk_index": 1}),
    ]

    # Should not raise
    validator.validate_chunks(chunks, "file.txt")


# ---------------------------------------------------------
# Test: non-string text fails
# ---------------------------------------------------------

def test_validation_fails_non_string_text():
    validator = IngestionValidator()

    # text is not a string
    chunks = [MockChunk(12345, {})]

    with pytest.raises(
        IngestionValidationError,
        match="non-string text"
    ):
        validator.validate_chunks(chunks, "file.txt")


# ---------------------------------------------------------
# Test: text too short or too long
# ---------------------------------------------------------

def test_validation_fails_on_invalid_length():
    cfg = ChunkValidationConfig(min_chars=5, max_chars=10)
    validator = IngestionValidator(cfg)

    # too short
    chunks = [MockChunk("hi", {})]

    with pytest.raises(
        IngestionValidationError,
        match="invalid text length"
    ):
        validator.validate_chunks(chunks, "file.txt")

    # too long
    chunks = [MockChunk("this is way too long", {})]

    with pytest.raises(
        IngestionValidationError,
        match="invalid text length"
    ):
        validator.validate_chunks(chunks, "file.txt")


# ---------------------------------------------------------
# Test: metadata must be a mapping
# ---------------------------------------------------------

def test_validation_fails_invalid_metadata():
    validator = IngestionValidator()

    # metadata is not a dict/mapping
    chunks = [MockChunk("hello", metadata="not-a-dict")]

    with pytest.raises(
        IngestionValidationError,
        match="invalid metadata type"
    ):
        validator.validate_chunks(chunks, "file.txt")


# ---------------------------------------------------------
# Test: required metadata keys
# ---------------------------------------------------------

def test_validation_required_metadata_keys():
    cfg = ChunkValidationConfig(
        required_metadata_keys=("doc_id", "chunk_index")
    )
    validator = IngestionValidator(cfg)

    # Missing 'chunk_index'
    chunks = [MockChunk("hello", {"doc_id": "abc"})]

    with pytest.raises(
        IngestionValidationError,
        match="missing required metadata key 'chunk_index'"
    ):
        validator.validate_chunks(chunks, "file.txt")


# ---------------------------------------------------------
# Test: multiple chunks, failure on specific index
# ---------------------------------------------------------

def test_validation_identifies_correct_chunk_index():
    validator = IngestionValidator()

    chunks = [
        MockChunk("valid text", {"a": 1}),
        MockChunk(123, {"a": 2}),  # invalid text here
        MockChunk("valid again", {"a": 3}),
    ]

    with pytest.raises(
        IngestionValidationError,
        match="Chunk 1"
    ):
        validator.validate_chunks(chunks, "file.txt")
