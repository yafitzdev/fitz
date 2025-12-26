# tests/test_ingest_scanner.py
"""
Tests for fitz_ai.ingest.diff.scanner module.
"""

from pathlib import Path

from fitz_ai.ingest.diff.scanner import (
    SUPPORTED_EXTENSIONS,
    FileScanner,
    ScannedFile,
    ScanResult,
    scan_directory,
)


class TestScannedFile:
    """Tests for ScannedFile dataclass."""

    def test_relative_path(self):
        """Test relative_path property."""
        scanned = ScannedFile(
            path="/root/docs/test.md",
            root="/root/docs",
            ext=".md",
            size_bytes=100,
            mtime_epoch=1234567890.0,
            content_hash="sha256:abc",
        )

        assert scanned.relative_path == "test.md"

    def test_nested_relative_path(self):
        """Test relative_path for nested file."""
        scanned = ScannedFile(
            path="/root/docs/subdir/test.md",
            root="/root/docs",
            ext=".md",
            size_bytes=100,
            mtime_epoch=1234567890.0,
            content_hash="sha256:abc",
        )

        # Use Path for cross-platform comparison
        assert Path(scanned.relative_path) == Path("subdir/test.md")


class TestScanResult:
    """Tests for ScanResult dataclass."""

    def test_totals(self):
        """Test total properties."""
        result = ScanResult(
            root="/root",
            files=[
                ScannedFile(
                    "/root/a.md", "/root", ".md", 100, 1234567890.0, "sha256:a"
                ),
                ScannedFile(
                    "/root/b.md", "/root", ".md", 100, 1234567890.0, "sha256:b"
                ),
            ],
            errors=[("/root/c.md", "error")],
            skipped_extensions={".jpg": 2, ".exe": 1},
        )

        assert result.total_scanned == 2
        assert result.total_errors == 1
        assert result.total_skipped == 3


class TestFileScanner:
    """Tests for FileScanner."""

    def test_scans_supported_extensions(self, tmp_path: Path):
        """Test that supported extensions are scanned."""
        # Create test files
        (tmp_path / "test.md").write_text("# Test")
        (tmp_path / "test.txt").write_text("Hello")
        (tmp_path / "test.py").write_text("print('hi')")

        scanner = FileScanner()
        result = scanner.scan(tmp_path)

        assert result.total_scanned == 3
        assert len(result.errors) == 0

        paths = {Path(f.path).name for f in result.files}
        assert paths == {"test.md", "test.txt", "test.py"}

    def test_skips_unsupported_extensions(self, tmp_path: Path):
        """Test that unsupported extensions are skipped."""
        (tmp_path / "test.md").write_text("# Test")
        (tmp_path / "image.jpg").write_bytes(b"\xff\xd8\xff")
        (tmp_path / "binary.exe").write_bytes(b"\x00\x01\x02")

        scanner = FileScanner()
        result = scanner.scan(tmp_path)

        assert result.total_scanned == 1
        assert result.total_skipped == 2
        assert ".jpg" in result.skipped_extensions
        assert ".exe" in result.skipped_extensions

    def test_scans_single_file(self, tmp_path: Path):
        """Test scanning a single file."""
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test")

        scanner = FileScanner()
        result = scanner.scan(test_file)

        assert result.total_scanned == 1
        assert result.files[0].path == str(test_file.resolve())

    def test_scans_recursively(self, tmp_path: Path):
        """Test that directories are scanned recursively."""
        (tmp_path / "a.md").write_text("# A")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "b.md").write_text("# B")
        subsubdir = subdir / "subsubdir"
        subsubdir.mkdir()
        (subsubdir / "c.md").write_text("# C")

        scanner = FileScanner()
        result = scanner.scan(tmp_path)

        assert result.total_scanned == 3

    def test_skips_hidden_files(self, tmp_path: Path):
        """Test that hidden files are skipped."""
        (tmp_path / "visible.md").write_text("# Visible")
        (tmp_path / ".hidden.md").write_text("# Hidden")

        scanner = FileScanner()
        result = scanner.scan(tmp_path)

        assert result.total_scanned == 1
        assert result.files[0].path.endswith("visible.md")

    def test_skips_hidden_directories(self, tmp_path: Path):
        """Test that hidden directories are skipped."""
        (tmp_path / "visible.md").write_text("# Visible")
        hidden_dir = tmp_path / ".hidden"
        hidden_dir.mkdir()
        (hidden_dir / "file.md").write_text("# In hidden dir")

        scanner = FileScanner()
        result = scanner.scan(tmp_path)

        assert result.total_scanned == 1

    def test_computes_content_hash(self, tmp_path: Path):
        """Test that content hash is computed."""
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test content")

        scanner = FileScanner()
        result = scanner.scan(tmp_path)

        assert result.files[0].content_hash.startswith("sha256:")

    def test_same_content_same_hash(self, tmp_path: Path):
        """Test that identical content produces identical hash."""
        content = "# Same content"
        (tmp_path / "file1.md").write_text(content)
        (tmp_path / "file2.md").write_text(content)

        scanner = FileScanner()
        result = scanner.scan(tmp_path)

        hashes = {f.content_hash for f in result.files}
        assert len(hashes) == 1  # Both files have same hash

    def test_custom_extensions(self, tmp_path: Path):
        """Test scanning with custom extensions."""
        (tmp_path / "test.md").write_text("# Test")
        (tmp_path / "test.custom").write_text("custom content")

        scanner = FileScanner(supported_extensions={".custom"})
        result = scanner.scan(tmp_path)

        assert result.total_scanned == 1
        assert result.files[0].path.endswith("test.custom")

    def test_empty_directory(self, tmp_path: Path):
        """Test scanning an empty directory."""
        scanner = FileScanner()
        result = scanner.scan(tmp_path)

        assert result.total_scanned == 0
        assert len(result.errors) == 0

    def test_captures_file_metadata(self, tmp_path: Path):
        """Test that file metadata is captured."""
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test with some content")

        scanner = FileScanner()
        result = scanner.scan(tmp_path)

        scanned = result.files[0]
        assert scanned.ext == ".md"
        assert scanned.size_bytes > 0
        assert scanned.mtime_epoch > 0


class TestScanDirectory:
    """Tests for scan_directory convenience function."""

    def test_scans_directory(self, tmp_path: Path):
        """Test the convenience function."""
        (tmp_path / "test.md").write_text("# Test")

        result = scan_directory(tmp_path)

        assert result.total_scanned == 1


class TestSupportedExtensions:
    """Tests for SUPPORTED_EXTENSIONS constant."""

    def test_contains_required_extensions(self):
        """Test that required extensions are supported."""
        assert ".md" in SUPPORTED_EXTENSIONS
        assert ".txt" in SUPPORTED_EXTENSIONS
        assert ".py" in SUPPORTED_EXTENSIONS
        assert ".pdf" in SUPPORTED_EXTENSIONS
