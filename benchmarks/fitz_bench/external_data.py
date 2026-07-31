"""Shared safety primitives for externally downloaded benchmark corpora."""

from __future__ import annotations

import hashlib
import shutil
import ssl
import stat
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import certifi

_CHUNK_BYTES = 1024 * 1024


def download_url(
    url: str,
    destination: Path,
    *,
    max_bytes: int,
    user_agent: str,
) -> int:
    """Download one URL atomically while enforcing a hard byte limit."""
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    tls_context = ssl.create_default_context(cafile=certifi.where())
    downloaded = 0
    try:
        with (
            urllib.request.urlopen(request, timeout=120, context=tls_context) as response,
            temporary.open("wb") as target,
        ):
            while chunk := response.read(_CHUNK_BYTES):
                downloaded += len(chunk)
                if downloaded > max_bytes:
                    raise ValueError(f"Download exceeds limit: {downloaded} bytes > {max_bytes}")
                target.write(chunk)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return downloaded


def file_digest(path: Path, algorithm: str) -> str:
    """Return a lowercase digest for a file using a named hashlib algorithm."""
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as source:
        while chunk := source.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def zip_uncompressed_bytes(path: Path) -> int:
    """Return the declared uncompressed byte total for regular ZIP members."""
    with zipfile.ZipFile(path) as archive:
        return sum(int(member.file_size) for member in archive.infolist() if not member.is_dir())


def safe_extract_zip(
    archive_path: Path,
    output_dir: Path,
    *,
    max_extracted_bytes: int,
) -> tuple[int, int]:
    """Extract a ZIP atomically after validating paths, symlinks, and size."""
    archive_path = Path(archive_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_parent = output_dir.parent.resolve()
    if output_dir == output_parent or output_parent not in output_dir.parents:
        raise ValueError(f"Unsafe extraction target: {output_dir}")

    with zipfile.ZipFile(archive_path) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        total_bytes = sum(int(member.file_size) for member in members)
        if total_bytes > max_extracted_bytes:
            raise ValueError(
                f"Archive expands to {total_bytes} bytes, over limit {max_extracted_bytes}"
            )
        for member in members:
            _validate_zip_member(member)

        output_parent.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=str(output_parent))
        ).resolve()
        try:
            for member in members:
                relative = PurePosixPath(member.filename)
                destination = temporary_dir.joinpath(*relative.parts).resolve()
                if temporary_dir not in destination.parents:
                    raise ValueError(f"Unsafe ZIP member path: {member.filename}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target, length=_CHUNK_BYTES)
            if output_dir.exists():
                shutil.rmtree(output_dir)
            _replace_directory_with_retry(temporary_dir, output_dir)
        except Exception:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise
    return len(members), total_bytes


def _validate_zip_member(member: zipfile.ZipInfo) -> None:
    path = PurePosixPath(member.filename)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or "\\" in member.filename
        or ":" in path.parts[0]
    ):
        raise ValueError(f"Unsafe ZIP member path: {member.filename}")
    unix_mode = (member.external_attr >> 16) & 0xFFFF
    if stat.S_ISLNK(unix_mode):
        raise ValueError(f"ZIP symlinks are not accepted: {member.filename}")


def _replace_directory_with_retry(
    source: Path,
    destination: Path,
    *,
    attempts: int = 60,
    delay_seconds: float = 1.0,
) -> None:
    """Retry a same-volume directory rename when Windows holds transient handles."""
    if attempts < 1:
        raise ValueError("Directory replacement attempts must be positive.")
    for attempt in range(1, attempts + 1):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt == attempts:
                raise
            time.sleep(delay_seconds)
