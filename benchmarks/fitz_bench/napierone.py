"""Download verified, unchanged slices of the public NapierOne corpus."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from benchmarks.fitz_bench.external_data import (
    download_url,
    file_digest,
    safe_extract_zip,
    zip_uncompressed_bytes,
)

DATASET_PAGE = "https://registry.opendata.aws/napierone/"
S3_BASE_URL = "https://s3.eu-north-1.amazonaws.com/napierone.com"
DEFAULT_TYPES = (
    "PDF",
    "DOCX",
    "PPTX",
    "CSV",
    "TXT",
    "JSON",
    "JAVASCRIPT",
    "HTML",
    "XML",
)
AVAILABLE_TYPES = frozenset(
    {
        "CSV",
        "DOCX",
        "HTML",
        "JAVASCRIPT",
        "JSON",
        "PDF",
        "PPTX",
        "TXT",
        "XLSX",
        "XML",
    }
)
PROFILES = frozenset({"tiny", "small", "total"})
VARIANTS = frozenset({"standard", "nomagic", "password"})
_VARIANT_TYPES = frozenset({"DOCX", "PDF", "PPTX", "XLSX"})
_SHA256_RE = re.compile(r"SHA256\s+32\s+([a-fA-F0-9]{64})")
_USER_AGENT = "fitz-sage-production-benchmark/1"


@dataclass(frozen=True)
class ArchiveSpec:
    """One official NapierOne archive and its published hash file."""

    file_type: str
    profile: str
    variant: str
    key: str
    hash_key: str

    @property
    def filename(self) -> str:
        return PurePosixPath(self.key).name


@dataclass(frozen=True)
class PreparedArchive:
    """Local state for one verified and extracted archive."""

    file_type: str
    profile: str
    variant: str
    archive: str
    sha256: str
    compressed_bytes: int
    extracted_files: int
    extracted_bytes: int


@dataclass(frozen=True)
class PreparedCorpus:
    """A local byte-preserving view over selected NapierOne archives."""

    source: str
    dataset_page: str
    profile: str
    variant: str
    corpus_dir: str
    file_types: tuple[str, ...]
    materialization: str
    archives: tuple[PreparedArchive, ...]

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["archives"] = [asdict(archive) for archive in self.archives]
        data["files"] = sum(archive.extracted_files for archive in self.archives)
        data["compressed_bytes"] = sum(archive.compressed_bytes for archive in self.archives)
        data["extracted_bytes"] = sum(archive.extracted_bytes for archive in self.archives)
        return data


def prepare_corpus(
    cache_dir: Path,
    *,
    file_types: Iterable[str] = DEFAULT_TYPES,
    profile: str = "tiny",
    variant: str = "standard",
    max_download_bytes: int = 5 * 1024**3,
    max_extracted_bytes: int = 20 * 1024**3,
    offline: bool = False,
    progress: Callable[[str], None] | None = None,
) -> PreparedCorpus:
    """Download, verify, and safely extract selected official archives."""
    cache_dir = Path(cache_dir).resolve()
    normalized_types = normalize_types(file_types)
    profile = profile.strip().lower()
    variant = variant.strip().lower()
    if profile not in PROFILES:
        raise ValueError(f"Unknown NapierOne profile: {profile}")
    if variant not in VARIANTS:
        raise ValueError(f"Unknown NapierOne variant: {variant}")

    specs = [archive_spec(file_type, profile, variant) for file_type in normalized_types]
    archives_dir = cache_dir / "archives"
    corpus_dir = cache_dir / "corpora" / profile / variant
    state_dir = cache_dir / "state" / profile / variant
    archives_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    emit = progress or (lambda _: None)

    prepared: list[PreparedArchive] = []
    download_budget = int(max_download_bytes)
    extract_budget = int(max_extracted_bytes)
    for spec in specs:
        archive_path = archives_dir / spec.filename
        expected_sha256 = _expected_sha256(spec, archives_dir, offline=offline)

        if archive_path.exists() and _sha256(archive_path) != expected_sha256:
            archive_path.unlink()
        if not archive_path.exists():
            if offline:
                raise FileNotFoundError(f"Archive is not cached for offline use: {archive_path}")
            remote_size = _remote_size(spec.key)
            if remote_size > download_budget:
                raise ValueError(
                    f"NapierOne download budget exceeded by {spec.filename}: "
                    f"{remote_size} bytes remain={download_budget}"
                )
            emit(f"Downloading {spec.filename} ({remote_size / 1024**2:.1f} MiB)...")
            downloaded_bytes = _download(
                spec.key,
                archive_path,
                max_bytes=download_budget,
            )
            download_budget -= downloaded_bytes

        actual_sha256 = _sha256(archive_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"SHA-256 mismatch for {archive_path.name}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

        output_dir = corpus_dir / spec.file_type
        marker_path = state_dir / f"{spec.file_type}.json"
        marker = _load_marker(marker_path)
        if (
            marker.get("sha256") == actual_sha256
            and output_dir.exists()
            and int(marker.get("extracted_files", 0)) > 0
        ):
            extracted_files = int(marker["extracted_files"])
            extracted_bytes = int(marker["extracted_bytes"])
            if extracted_bytes > extract_budget:
                raise ValueError(
                    f"NapierOne extraction budget exceeded by cached {spec.filename}: "
                    f"{extracted_bytes} bytes remain={extract_budget}"
                )
            extract_budget -= extracted_bytes
        else:
            extracted_size = _zip_uncompressed_bytes(archive_path)
            if extracted_size > extract_budget:
                raise ValueError(
                    f"NapierOne extraction budget exceeded by {spec.filename}: "
                    f"{extracted_size} bytes remain={extract_budget}"
                )
            emit(f"Extracting {spec.filename}...")
            extracted_files, extracted_bytes = safe_extract_zip(
                archive_path,
                output_dir,
                max_extracted_bytes=extract_budget,
            )
            extract_budget -= extracted_bytes
            _write_json_atomic(
                marker_path,
                {
                    "sha256": actual_sha256,
                    "extracted_files": extracted_files,
                    "extracted_bytes": extracted_bytes,
                },
            )

        prepared.append(
            PreparedArchive(
                file_type=spec.file_type,
                profile=spec.profile,
                variant=spec.variant,
                archive=str(archive_path),
                sha256=actual_sha256,
                compressed_bytes=archive_path.stat().st_size,
                extracted_files=extracted_files,
                extracted_bytes=extracted_bytes,
            )
        )

    selection_dir, materialization = _prepare_selection(
        cache_dir,
        corpus_dir=corpus_dir,
        profile=profile,
        variant=variant,
        file_types=normalized_types,
        archives=prepared,
    )
    return PreparedCorpus(
        source="NapierOne",
        dataset_page=DATASET_PAGE,
        profile=profile,
        variant=variant,
        corpus_dir=str(selection_dir),
        file_types=normalized_types,
        materialization=materialization,
        archives=tuple(prepared),
    )


def _prepare_selection(
    cache_dir: Path,
    *,
    corpus_dir: Path,
    profile: str,
    variant: str,
    file_types: tuple[str, ...],
    archives: list[PreparedArchive],
) -> tuple[Path, str]:
    ordered_types = tuple(sorted(file_types))
    selection_key = "-".join(value.lower() for value in ordered_types)
    selection_dir = cache_dir / "selections" / profile / variant / selection_key
    marker_path = (
        cache_dir / "state" / "selections" / profile / variant / f"{selection_key}.json"
    )
    signature = {
        "file_types": list(ordered_types),
        "archives": {
            archive.file_type: archive.sha256
            for archive in sorted(archives, key=lambda value: value.file_type)
        },
    }
    marker = _load_marker(marker_path)
    cached_materialization = marker.get("materialization")
    if (
        marker.get("signature") == signature
        and selection_dir.exists()
        and int(marker.get("files", 0)) > 0
        and cached_materialization in {"hardlink", "copy"}
    ):
        return selection_dir, str(cached_materialization)

    selection_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{selection_key}-", dir=str(selection_dir.parent))
    ).resolve()
    materialization = "hardlink"
    file_count = 0
    try:
        for file_type in ordered_types:
            source_root = corpus_dir / file_type
            for source in source_root.rglob("*"):
                if not source.is_file():
                    continue
                destination = temporary_dir / source.relative_to(corpus_dir)
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
                    materialization = "copy"
                file_count += 1
        if file_count == 0:
            raise ValueError("The selected NapierOne archives contain no files.")
        if selection_dir.exists():
            shutil.rmtree(selection_dir)
        temporary_dir.replace(selection_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    _write_json_atomic(
        marker_path,
        {
            "signature": signature,
            "files": file_count,
            "materialization": materialization,
        },
    )
    return selection_dir, materialization


def normalize_types(file_types: Iterable[str]) -> tuple[str, ...]:
    """Normalize and validate requested NapierOne file-type subsets."""
    normalized = tuple(dict.fromkeys(str(value).strip().upper() for value in file_types))
    if not normalized:
        raise ValueError("At least one NapierOne file type is required.")
    unknown = sorted(set(normalized) - AVAILABLE_TYPES)
    if unknown:
        raise ValueError(
            f"Unsupported NapierOne benchmark type(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(AVAILABLE_TYPES))}"
        )
    return normalized


def archive_spec(file_type: str, profile: str, variant: str = "standard") -> ArchiveSpec:
    """Resolve one archive using NapierOne's published key convention."""
    file_type = file_type.strip().upper()
    profile = profile.strip().lower()
    variant = variant.strip().lower()
    if file_type not in AVAILABLE_TYPES:
        raise ValueError(f"Unsupported NapierOne benchmark type: {file_type}")
    if profile not in PROFILES:
        raise ValueError(f"Unknown NapierOne profile: {profile}")
    if variant not in VARIANTS:
        raise ValueError(f"Unknown NapierOne variant: {variant}")
    if variant != "standard" and file_type not in _VARIANT_TYPES:
        raise ValueError(f"NapierOne has no {variant} archive for {file_type}")

    variant_part = "" if variant == "standard" else f"-{variant.upper()}"
    stem = f"{file_type}{variant_part}-{profile}"
    prefix = f"NapierOne/Data/{file_type}"
    return ArchiveSpec(
        file_type=file_type,
        profile=profile,
        variant=variant,
        key=f"{prefix}/{stem}.zip",
        hash_key=f"{prefix}/{stem}_zip_hashes.txt",
    )


def _expected_sha256(spec: ArchiveSpec, archives_dir: Path, *, offline: bool) -> str:
    hash_path = archives_dir / PurePosixPath(spec.hash_key).name
    if not hash_path.exists():
        if offline:
            raise FileNotFoundError(f"Hash manifest is not cached for offline use: {hash_path}")
        _download(spec.hash_key, hash_path, max_bytes=1024 * 1024)
    return parse_sha256_manifest(hash_path.read_text(encoding="utf-8", errors="replace"))


def parse_sha256_manifest(text: str) -> str:
    """Read the SHA-256 value from NapierOne's Hash Console report."""
    match = _SHA256_RE.search(text)
    if match is None:
        raise ValueError("NapierOne hash manifest does not contain a SHA-256 value.")
    return match.group(1).lower()


def _remote_size(key: str) -> int:
    request = urllib.request.Request(_url(key), method="HEAD", headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return int(response.headers.get("Content-Length", "0"))
    except (urllib.error.URLError, ValueError) as exc:
        raise RuntimeError(f"Could not inspect NapierOne object: {key}") from exc


def _download(key: str, destination: Path, *, max_bytes: int) -> int:
    return download_url(
        _url(key),
        destination,
        max_bytes=max_bytes,
        user_agent=_USER_AGENT,
    )


def _url(key: str) -> str:
    return f"{S3_BASE_URL}/{urllib.parse.quote(key, safe='/')}"


def _sha256(path: Path) -> str:
    return file_digest(path, "sha256")


def _zip_uncompressed_bytes(path: Path) -> int:
    return zip_uncompressed_bytes(path)


def _load_marker(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
