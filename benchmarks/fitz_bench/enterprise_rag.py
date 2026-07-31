"""Verified EnterpriseRAG-Bench release and byte-preserving corpus adapter."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from benchmarks.fitz_bench.external_data import (
    download_url,
    file_digest,
    safe_extract_zip,
    zip_uncompressed_bytes,
)

REPOSITORY_URL = "https://github.com/onyx-dot-app/EnterpriseRAG-Bench"
PAPER_URL = "https://arxiv.org/abs/2605.05253"
ADAPTER_SCHEMA_VERSION = 1
_USER_AGENT = "fitz-sage-enterprise-rag-benchmark/1"
_DOCUMENT_ID = re.compile(r"^(dsid_[0-9a-f]{32})__.+\.txt$")
_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class EnterpriseRagSpec:
    """Published release identity and expected corpus/question shape."""

    release: str
    archive_url: str
    archive_sha256: str
    archive_bytes: int
    questions_url: str
    questions_sha256: str
    questions_bytes: int
    archive_files: int
    extracted_bytes: int
    corpus_documents: int
    source_counts: dict[str, int]
    question_counts: dict[str, int]


SPEC = EnterpriseRagSpec(
    release="v1.0.0",
    archive_url=(
        "https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/"
        "download/v1.0.0/all_documents.zip"
    ),
    archive_sha256="9d1174928696ad08bc15f3f104739519de633c1605a4ec2034e0e3c0087bc5cd",
    archive_bytes=1_256_181_062,
    questions_url=(
        "https://github.com/onyx-dot-app/EnterpriseRAG-Bench/releases/"
        "download/v1.0.0/questions.jsonl"
    ),
    questions_sha256="f9524b9157cd43aae36b99333a124738804306ea6d07f332d49faa6d3d147905",
    questions_bytes=764_927,
    archive_files=511_963,
    extracted_bytes=2_474_399_575,
    corpus_documents=511_962,
    source_counts={
        "confluence": 5_189,
        "fireflies": 10_173,
        "github": 8_052,
        "gmail": 121_390,
        "google_drive": 25_108,
        "hubspot": 15_017,
        "jira": 6_120,
        "linear": 35_308,
        "slack": 285_605,
    },
    question_counts={
        "basic": 175,
        "completeness": 20,
        "conflicting_info": 20,
        "constrained": 30,
        "high_level": 10,
        "info_not_found": 20,
        "intra_document_reasoning": 40,
        "miscellaneous": 20,
        "project_related": 40,
        "semantic": 125,
    },
)


@dataclass(frozen=True)
class EnterpriseQuestion:
    """Retrieval-relevant fields from one official benchmark question."""

    question_id: str
    question_type: str
    source_types: tuple[str, ...]
    text: str
    expected_document_ids: tuple[str, ...]


@dataclass(frozen=True)
class PreparedEnterpriseRag:
    """Verified local paths and immutable release provenance."""

    spec: EnterpriseRagSpec
    archive: Path
    questions_path: Path
    corpus_dir: Path
    mapping_path: Path
    mapping_sha256: str
    unique_document_ids: int
    duplicate_document_ids: dict[str, tuple[str, ...]]

    @property
    def name(self) -> str:
        return "enterprise-rag-bench"

    @property
    def adapter_schema_version(self) -> int:
        return ADAPTER_SCHEMA_VERSION

    def fingerprint(self) -> dict[str, Any]:
        return {
            "release": self.spec.release,
            "archive_sha256": self.spec.archive_sha256,
            "questions_sha256": self.spec.questions_sha256,
            "mapping_sha256": self.mapping_sha256,
            "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": "EnterpriseRAG-Bench",
            "release": self.spec.release,
            "repository_url": REPOSITORY_URL,
            "paper_url": PAPER_URL,
            "archive_url": self.spec.archive_url,
            "archive": str(self.archive),
            "archive_sha256": self.spec.archive_sha256,
            "archive_bytes": self.spec.archive_bytes,
            "questions_url": self.spec.questions_url,
            "questions_path": str(self.questions_path),
            "questions_sha256": self.spec.questions_sha256,
            "corpus_dir": str(self.corpus_dir),
            "mapping_path": str(self.mapping_path),
            "mapping_sha256": self.mapping_sha256,
            "corpus_documents": self.spec.corpus_documents,
            "unique_document_ids": self.unique_document_ids,
            "duplicate_document_ids": {
                key: list(paths) for key, paths in self.duplicate_document_ids.items()
            },
            "source_counts": self.spec.source_counts,
            "question_counts": self.spec.question_counts,
            "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
            "adapter_projection": (
                "Original UTF-8 .txt bytes and source hierarchy; no rewriting, "
                "normalization, or generated metadata"
            ),
            "license": "MIT",
            "synthetic_corpus": True,
        }


def prepare_dataset(
    cache_dir: Path,
    *,
    spec: EnterpriseRagSpec = SPEC,
    max_download_bytes: int = 2 * 1024**3,
    max_extracted_bytes: int = 4 * 1024**3,
    offline: bool = False,
    progress: Callable[[str], None] | None = None,
) -> PreparedEnterpriseRag:
    """Download, verify, safely extract, and map the official release."""
    emit = progress or (lambda _message: None)
    release_dir = Path(cache_dir).resolve() / spec.release
    archive = release_dir / "all_documents.zip"
    questions_path = release_dir / "questions.jsonl"
    release_dir.mkdir(parents=True, exist_ok=True)
    _ensure_asset(
        archive,
        url=spec.archive_url,
        expected_sha256=spec.archive_sha256,
        expected_bytes=spec.archive_bytes,
        max_bytes=max_download_bytes,
        offline=offline,
        emit=emit,
    )
    _ensure_asset(
        questions_path,
        url=spec.questions_url,
        expected_sha256=spec.questions_sha256,
        expected_bytes=spec.questions_bytes,
        max_bytes=max_download_bytes,
        offline=offline,
        emit=emit,
    )
    questions = load_questions(questions_path, spec=spec)
    archive_shape = _validate_archive(archive, spec=spec)
    if archive_shape["extracted_bytes"] > max_extracted_bytes:
        raise ValueError(
            f"EnterpriseRAG-Bench expands to {archive_shape['extracted_bytes']} bytes, "
            f"over limit {max_extracted_bytes}"
        )

    adapter_dir = release_dir / f"adapter-v{ADAPTER_SCHEMA_VERSION}"
    payload_dir = adapter_dir / "payload"
    corpus_dir = payload_dir / "corpus"
    mapping_path = adapter_dir / "mapping.jsonl"
    state_path = adapter_dir / "state.json"
    state = _load_json_object(state_path)
    expected_state = {
        "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
        "archive_sha256": spec.archive_sha256,
        "questions_sha256": spec.questions_sha256,
        "archive_files": spec.archive_files,
        "extracted_bytes": spec.extracted_bytes,
        "corpus_documents": spec.corpus_documents,
        "source_counts": spec.source_counts,
    }
    corpus_ready = corpus_dir.is_dir() and all(
        (corpus_dir / source_type).is_dir() for source_type in spec.source_counts
    )
    state_matches = all(state.get(key) == value for key, value in expected_state.items())
    mapping_matches = mapping_path.is_file() and state.get("mapping_sha256") == file_digest(
        mapping_path, "sha256"
    )
    if not corpus_ready or not state_matches:
        emit(f"Extracting {spec.corpus_documents} EnterpriseRAG-Bench documents...")
        safe_extract_zip(
            archive,
            payload_dir,
            max_extracted_bytes=max_extracted_bytes,
        )
        _isolate_corpus(payload_dir, source_types=tuple(spec.source_counts))
        embedded_questions = payload_dir / "questions.jsonl"
        if file_digest(embedded_questions, "sha256") != spec.questions_sha256:
            raise ValueError("Embedded and standalone EnterpriseRAG-Bench questions differ.")
        mapping_matches = False
    if not mapping_matches:
        emit("Building byte-identity mapping for EnterpriseRAG-Bench...")
        mapping_state = _build_mapping(archive, mapping_path, spec=spec, emit=emit)
        state = {**expected_state, **mapping_state}
        _write_json_atomic(state_path, state)

    duplicate_document_ids = {
        str(document_id): tuple(str(path) for path in paths)
        for document_id, paths in state.get("duplicate_document_ids", {}).items()
    }
    if len(questions) != sum(spec.question_counts.values()):
        raise AssertionError("Validated question count changed unexpectedly.")
    return PreparedEnterpriseRag(
        spec=spec,
        archive=archive,
        questions_path=questions_path,
        corpus_dir=corpus_dir,
        mapping_path=mapping_path,
        mapping_sha256=str(state["mapping_sha256"]),
        unique_document_ids=int(state["unique_document_ids"]),
        duplicate_document_ids=duplicate_document_ids,
    )


def load_questions(
    path: Path,
    *,
    spec: EnterpriseRagSpec = SPEC,
) -> dict[str, EnterpriseQuestion]:
    """Load and validate only fields needed for retrieval evaluation."""
    questions: dict[str, EnterpriseQuestion] = {}
    categories: Counter[str] = Counter()
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid question JSON at {path}:{line_number}") from exc
            if not isinstance(raw, dict):
                raise TypeError(f"Expected question object at {path}:{line_number}")
            question_id = _required_string(raw, "question_id", path, line_number)
            question_type = _required_string(raw, "question_type", path, line_number)
            text = _required_string(raw, "question", path, line_number)
            source_types = _string_list(
                raw,
                "source_types",
                path,
                line_number,
                allow_empty=True,
            )
            expected = _string_list(
                raw,
                "expected_doc_ids",
                path,
                line_number,
                allow_empty=True,
            )
            if expected and not source_types:
                raise ValueError(f"Scored question has no source type: {question_id}")
            if question_id in questions:
                raise ValueError(f"Duplicate question ID: {question_id}")
            questions[question_id] = EnterpriseQuestion(
                question_id=question_id,
                question_type=question_type,
                source_types=tuple(source_types),
                text=text,
                expected_document_ids=tuple(expected),
            )
            categories[question_type] += 1
    if dict(categories) != spec.question_counts:
        raise ValueError(
            "EnterpriseRAG-Bench question category counts differ from the pinned release: "
            f"{dict(categories)}"
        )
    return questions


def queries_and_qrels(
    questions: dict[str, EnterpriseQuestion],
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Return all query text and only retrieval-scored relevance judgments."""
    queries = {question_id: question.text for question_id, question in questions.items()}
    qrels = {
        question_id: {document_id: 1 for document_id in question.expected_document_ids}
        for question_id, question in questions.items()
        if question.expected_document_ids
    }
    return queries, qrels


def iter_archive_documents(
    prepared: PreparedEnterpriseRag,
) -> Iterator[tuple[str, str, str]]:
    """Yield physical path, official ID, and unchanged UTF-8 text from the archive."""
    with zipfile.ZipFile(prepared.archive) as archive:
        for member in archive.infolist():
            if member.is_dir() or member.filename == "questions.jsonl":
                continue
            document_id = _document_id(member.filename)
            try:
                content = archive.read(member).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"Document is not UTF-8: {member.filename}") from exc
            yield member.filename, document_id, content


def _ensure_asset(
    path: Path,
    *,
    url: str,
    expected_sha256: str,
    expected_bytes: int,
    max_bytes: int,
    offline: bool,
    emit: Callable[[str], None],
) -> None:
    valid = (
        path.is_file()
        and path.stat().st_size == expected_bytes
        and file_digest(path, "sha256") == expected_sha256
    )
    if valid:
        return
    if path.exists() and offline:
        raise ValueError(f"Cached benchmark asset has the wrong identity: {path}")
    path.unlink(missing_ok=True)
    if offline:
        raise FileNotFoundError(f"Benchmark asset is not cached for offline use: {path}")
    emit(f"Downloading {path.name}...")
    download_url(url, path, max_bytes=max_bytes, user_agent=_USER_AGENT)
    if path.stat().st_size != expected_bytes or file_digest(path, "sha256") != expected_sha256:
        path.unlink(missing_ok=True)
        raise ValueError(f"Downloaded benchmark asset failed identity verification: {path}")


def _validate_archive(path: Path, *, spec: EnterpriseRagSpec) -> dict[str, Any]:
    source_counts: Counter[str] = Counter()
    file_count = 0
    questions = 0
    seen_paths: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            file_count += 1
            name = member.filename
            if name in seen_paths:
                raise ValueError(f"Duplicate ZIP member path: {name}")
            seen_paths.add(name)
            if name == "questions.jsonl":
                questions += 1
                continue
            relative = PurePosixPath(name)
            if len(relative.parts) < 2 or relative.parts[0] not in spec.source_counts:
                raise ValueError(f"Unexpected EnterpriseRAG-Bench archive path: {name}")
            _document_id(name)
            source_counts[relative.parts[0]] += 1
    extracted_bytes = zip_uncompressed_bytes(path)
    if file_count != spec.archive_files or questions != 1:
        raise ValueError(f"Unexpected archive shape: files={file_count}, questions={questions}")
    if dict(source_counts) != spec.source_counts:
        raise ValueError(f"Unexpected archive source counts: {dict(source_counts)}")
    if extracted_bytes != spec.extracted_bytes:
        raise ValueError(
            f"Unexpected archive byte count: {extracted_bytes} != {spec.extracted_bytes}"
        )
    return {
        "files": file_count,
        "extracted_bytes": extracted_bytes,
        "source_counts": dict(source_counts),
    }


def _isolate_corpus(payload_dir: Path, *, source_types: tuple[str, ...]) -> None:
    corpus_dir = payload_dir / "corpus"
    corpus_dir.mkdir()
    for source_type in source_types:
        source = payload_dir / source_type
        if not source.is_dir():
            raise FileNotFoundError(f"Extracted source directory is missing: {source}")
        source.replace(corpus_dir / source_type)


def _build_mapping(
    archive_path: Path,
    mapping_path: Path,
    *,
    spec: EnterpriseRagSpec,
    emit: Callable[[str], None],
) -> dict[str, Any]:
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = mapping_path.with_name(f".{mapping_path.name}.tmp")
    first_paths: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    count = 0
    try:
        with (
            zipfile.ZipFile(archive_path) as archive,
            temporary.open("w", encoding="utf-8", newline="\n") as output,
        ):
            for member in archive.infolist():
                if member.is_dir() or member.filename == "questions.jsonl":
                    continue
                document_id = _document_id(member.filename)
                with archive.open(member) as source:
                    digest = _stream_sha256(source)
                output.write(
                    json.dumps(
                        {
                            "document_id": document_id,
                            "relative_path": member.filename,
                            "content_sha256": digest,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                first = first_paths.setdefault(document_id, member.filename)
                if first != member.filename:
                    duplicates.setdefault(document_id, [first]).append(member.filename)
                count += 1
                if count % 25_000 == 0:
                    emit(f"  Mapped {count}/{spec.corpus_documents} documents")
        if count != spec.corpus_documents:
            raise ValueError(f"Mapping document count mismatch: {count} != {spec.corpus_documents}")
        temporary.replace(mapping_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    mapping_sha256 = file_digest(mapping_path, "sha256")
    return {
        "mapping_sha256": mapping_sha256,
        "unique_document_ids": len(first_paths),
        "duplicate_document_ids": dict(sorted(duplicates.items())),
    }


def _stream_sha256(source: Any) -> str:
    digest = hashlib.sha256()
    while chunk := source.read(_CHUNK_BYTES):
        digest.update(chunk)
    return digest.hexdigest()


def _document_id(path: str) -> str:
    match = _DOCUMENT_ID.fullmatch(PurePosixPath(path).name)
    if match is None:
        raise ValueError(f"Unexpected EnterpriseRAG-Bench document filename: {path}")
    return match.group(1)


def _required_string(
    raw: dict[str, Any],
    key: str,
    path: Path,
    line_number: int,
) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid {key!r} at {path}:{line_number}")
    return value


def _string_list(
    raw: dict[str, Any],
    key: str,
    path: Path,
    line_number: int,
    *,
    allow_empty: bool = False,
) -> list[str]:
    value = raw.get(key)
    if (
        not isinstance(value, list)
        or (not value and not allow_empty)
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"Invalid {key!r} at {path}:{line_number}")
    return value


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)
