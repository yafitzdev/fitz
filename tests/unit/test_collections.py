"""Collection identity tests."""

from pathlib import Path

import pytest

from fitz_sage.core.collections import collection_name_from_path, validate_collection_name
from fitz_sage.storage.sqlite import SqliteConnectionManager


@pytest.mark.parametrize("name", ["default", "bmw_tests", "release-15", "c1"])
def test_valid_collection_names_are_preserved(name: str) -> None:
    assert validate_collection_name(name) == name


@pytest.mark.parametrize("name", ["", "UPPER", "has space", "../escape", "foo/bar"])
def test_invalid_collection_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="Collection names"):
        validate_collection_name(name)


def test_generated_collection_name_is_valid(tmp_path: Path) -> None:
    source = tmp_path / "BMW Test Reports"
    source.mkdir()

    assert collection_name_from_path(source) == "bmw_test_reports"


def test_database_names_do_not_alias(tmp_path: Path) -> None:
    manager = SqliteConnectionManager()
    manager.config.storage_path = str(tmp_path)
    manager.start()

    assert manager.database_path("foo-bar") != manager.database_path("foo_bar")
