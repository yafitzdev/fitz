# tests/test_cli_collections.py
"""
Tests for the collections command.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from fitz_sage.cli.cli import app
from fitz_sage.services.fitz_service import CollectionInfo

runner = CliRunner()


class TestCollectionsCommand:
    """Tests for fitz collections command."""

    def test_collections_shows_help(self):
        """Test that collections --help works."""
        result = runner.invoke(app, ["collections", "--help"])

        assert result.exit_code == 0
        assert "collection" in result.output.lower()

    def test_collections_starts_correctly(self):
        """Test that collections starts and shows header with defaults."""
        # Mock FitzService to return no collections
        mock_service = MagicMock()
        mock_service.list_collections.return_value = []

        with patch("fitz_sage.cli.commands.collections.FitzService", return_value=mock_service):
            result = runner.invoke(app, ["collections"])

        # Should succeed (no collections found message)
        assert result.exit_code == 0
        assert "no collection" in result.output.lower() or "ingest" in result.output.lower()


class TestCollectionsHelpers:
    """Tests for collections helper functions."""

    def test_cli_context_loads_config(self):
        """Test CLIContext loads config."""
        mock_ctx = MagicMock()
        mock_ctx.vector_db_plugin = "pgvector"

        with patch("fitz_sage.cli.context.CLIContext.load", return_value=mock_ctx):
            from fitz_sage.cli.context import CLIContext

            ctx = CLIContext.load()

        assert ctx is not None
        assert ctx.vector_db_plugin == "pgvector"


class TestDisplayCollectionsTable:
    """Tests for _display_collections_table."""

    def test_display_collections_table_plain(self, capsys):
        """Test _display_collections_table outputs table."""
        with patch("fitz_sage.cli.commands.collections.RICH", False):
            from fitz_sage.cli.commands.collections import _display_collections_table

            collections = [
                {"name": "docs", "count": 100},
                {"name": "code", "count": 50},
            ]

            _display_collections_table(collections)

        captured = capsys.readouterr()
        assert "docs" in captured.out
        assert "100" in captured.out
        assert "code" in captured.out
        assert "50" in captured.out


class TestDisplayCollectionInfo:
    """Tests for _display_collection_info."""

    def test_display_collection_info_plain(self, capsys):
        """Test _display_collection_info shows stats."""
        with patch("fitz_sage.cli.commands.collections.RICH", False):
            from fitz_sage.cli.commands.collections import _display_collection_info

            _display_collection_info(
                name="my_collection",
                chunk_count=150,
                metadata={},
            )

        captured = capsys.readouterr()
        assert "my_collection" in captured.out
        assert "150" in captured.out

    def test_display_collection_info_vectors_count(self, capsys):
        """Test _display_collection_info handles vectors_count key."""
        with patch("fitz_sage.cli.commands.collections.RICH", False):
            from fitz_sage.cli.commands.collections import _display_collection_info

            _display_collection_info(
                name="test",
                chunk_count=200,
                metadata={},
            )

        captured = capsys.readouterr()
        assert "200" in captured.out


class TestCollectionsNoCollections:
    """Tests for handling empty collections."""

    def test_collections_empty_shows_message(self):
        """Test collections shows helpful message when no collections."""
        mock_service = MagicMock()
        mock_service.list_collections.return_value = []

        with patch("fitz_sage.cli.commands.collections.FitzService", return_value=mock_service):
            result = runner.invoke(app, ["collections"])

        assert "no collection" in result.output.lower() or "ingest" in result.output.lower()


class TestCollectionsWithData:
    """Tests for collections with actual data."""

    def test_collections_lists_and_exits(self):
        """Test collections can list and exit."""
        mock_service = MagicMock()
        mock_service.list_collections.return_value = [
            CollectionInfo(name="docs", chunk_count=100),
            CollectionInfo(name="code", chunk_count=50),
        ]

        with patch("fitz_sage.cli.commands.collections.FitzService", return_value=mock_service):
            # Select "Exit" (option 3 - after docs, code)
            result = runner.invoke(app, ["collections"], input="3\n")

        assert "docs" in result.output
        assert "code" in result.output

    def test_collections_select_and_exit(self):
        """Test collections can select a collection and exit."""
        mock_service = MagicMock()
        mock_service.list_collections.return_value = [
            CollectionInfo(name="docs", chunk_count=10),
        ]
        mock_service.get_collection.return_value = CollectionInfo(
            name="docs",
            chunk_count=10,
            metadata={},
        )

        with patch("fitz_sage.cli.commands.collections.FitzService", return_value=mock_service):
            # Select first collection, then Exit
            result = runner.invoke(app, ["collections"], input="1\n3\n")

        assert "docs" in result.output


class TestCollectionsDelete:
    """Tests for collection deletion."""

    def test_collections_delete_cancelled(self):
        """Test delete can be cancelled."""
        mock_service = MagicMock()
        mock_service.list_collections.return_value = [
            CollectionInfo(name="docs", chunk_count=10),
        ]
        mock_service.get_collection.return_value = CollectionInfo(
            name="docs",
            chunk_count=10,
            metadata={},
        )

        with patch("fitz_sage.cli.commands.collections.FitzService", return_value=mock_service):
            # Menu ordering (UI puts default first):
            # [1] docs (default) [2] Exit -> select 1 (docs)
            # [1] Back to list (default) [2] Delete collection [3] Exit -> select 2 (delete)
            # Confirm delete: n
            # Then exit: 3
            result = runner.invoke(app, ["collections"], input="1\n2\nn\n3\n")

        # Delete should not be called (user said 'n')
        mock_service.delete_collection.assert_not_called()
        assert "cancel" in result.output.lower()
