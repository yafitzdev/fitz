"""Verified BEIR datasets and a transparent Fitz-Sage corpus projection."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.fitz_bench.external_data import (
    download_url,
    file_digest,
    safe_extract_zip,
)

DATASET_PAGE = "https://github.com/beir-cellar/beir"
PAPER_URL = "https://arxiv.org/abs/2104.08663"
DOWNLOAD_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
LICENSE_NOTICE = (
    "BEIR does not grant one umbrella license. Users remain responsible for "
    "the license of each underlying dataset."
)
ADAPTER_SCHEMA_VERSION = 1
_USER_AGENT = "fitz-sage-beir-benchmark/1"
DEFAULT_DATASETS = ("nfcorpus", "fiqa", "scifact")


@dataclass(frozen=True)
class DatasetSpec:
    """Published identity and expected shape of one BEIR archive."""

    name: str
    md5: str
    corpus_documents: int
    test_queries: int
    ignore_identical_ids: bool = False

    @property
    def url(self) -> str:
        return f"{DOWNLOAD_BASE_URL}/{self.name}.zip"


DATASETS: dict[str, DatasetSpec] = {
    "arguana": DatasetSpec(
        name="arguana",
        md5="8ad3e3c2a5867cdced806d6503f29b99",
        corpus_documents=8674,
        test_queries=1406,
        ignore_identical_ids=True,
    ),
    "nfcorpus": DatasetSpec(
        name="nfcorpus",
        md5="a89dba18a62ef92f7d323ec890a0d38d",
        corpus_documents=3633,
        test_queries=323,
    ),
    "fiqa": DatasetSpec(
        name="fiqa",
        md5="17918ed23cd04fb15047f73e6c3bd9d9",
        corpus_documents=57638,
        test_queries=648,
    ),
    "quora": DatasetSpec(
        name="quora",
        md5="18fb154900ba42a600f84b839c173167",
        corpus_documents=522931,
        test_queries=10000,
        ignore_identical_ids=True,
    ),
    "scifact": DatasetSpec(
        name="scifact",
        md5="5f7d1de60b170fc8027bb7898e2efca1",
        corpus_documents=5183,
        test_queries=300,
    ),
}


@dataclass(frozen=True)
class PreparedDataset:
    """Local paths and provenance for a verified, projected BEIR dataset."""

    name: str
    source: str
    dataset_page: str
    paper_url: str
    url: str
    archive: str
    md5: str
    compressed_bytes: int
    extracted_dir: str
    source_corpus: str
    source_queries: str
    source_qrels: str
    corpus_dir: str
    mapping_path: str
    corpus_documents: int
    empty_documents: int
    empty_judged_relevant_documents: int
    test_queries: int
    qrels: int
    ignore_identical_ids: bool
    adapter_schema_version: int
    adapter_projection: str
    license_notice: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorpusMapping:
    """Reversible identity mapping for one projected BEIR document."""

    document_id: str
    relative_path: str
    content_sha256: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def prepare_dataset(
    cache_dir: Path,
    dataset: str,
    *,
    max_download_bytes: int = 2 * 1024**3,
    max_extracted_bytes: int = 4 * 1024**3,
    offline: bool = False,
    progress: Callable[[str], None] | None = None,
) -> PreparedDataset:
    """Download, verify, safely extract, and project one official BEIR archive."""
    spec = dataset_spec(dataset)
    cache_dir = Path(cache_dir).resolve()
    archives_dir = cache_dir / "archives"
    extracted_dir = cache_dir / "extracted" / spec.name
    adapter_dir = cache_dir / "adapters" / spec.name / f"v{ADAPTER_SCHEMA_VERSION}"
    archive_path = archives_dir / f"{spec.name}.zip"
    emit = progress or (lambda _: None)

    archives_dir.mkdir(parents=True, exist_ok=True)
    if archive_path.exists() and file_digest(archive_path, "md5") != spec.md5:
        archive_path.unlink()
    if not archive_path.exists():
        if offline:
            raise FileNotFoundError(f"BEIR archive is not cached for offline use: {archive_path}")
        emit(f"Downloading BEIR {spec.name}...")
        download_url(
            spec.url,
            archive_path,
            max_bytes=max_download_bytes,
            user_agent=_USER_AGENT,
        )

    actual_md5 = file_digest(archive_path, "md5")
    if actual_md5 != spec.md5:
        raise ValueError(
            f"MD5 mismatch for {archive_path.name}: expected {spec.md5}, got {actual_md5}"
        )

    extraction_marker = extracted_dir.parent / f"{spec.name}.json"
    marker = _load_json_object(extraction_marker)
    extracted_now = marker.get("md5") != actual_md5 or not extracted_dir.exists()
    if extracted_now:
        emit(f"Extracting BEIR {spec.name}...")
        files, extracted_bytes = safe_extract_zip(
            archive_path,
            extracted_dir,
            max_extracted_bytes=max_extracted_bytes,
        )
    else:
        files = int(marker.get("files", 0))
        extracted_bytes = int(marker.get("extracted_bytes", 0))
        if extracted_bytes > max_extracted_bytes:
            raise ValueError(
                f"Cached BEIR extraction uses {extracted_bytes} bytes, "
                f"over limit {max_extracted_bytes}"
            )

    source_corpus = _single_path(extracted_dir, "corpus.jsonl")
    source_queries = _single_path(extracted_dir, "queries.jsonl")
    source_qrels = _single_path(extracted_dir, "test.tsv", parent_name="qrels")
    required_sha256 = _required_source_hashes(
        corpus=source_corpus,
        queries=source_queries,
        qrels=source_qrels,
    )
    if not extracted_now and marker.get("required_sha256") != required_sha256:
        emit(f"Re-extracting unverified or changed BEIR cache for {spec.name}...")
        files, extracted_bytes = safe_extract_zip(
            archive_path,
            extracted_dir,
            max_extracted_bytes=max_extracted_bytes,
        )
        source_corpus = _single_path(extracted_dir, "corpus.jsonl")
        source_queries = _single_path(extracted_dir, "queries.jsonl")
        source_qrels = _single_path(extracted_dir, "test.tsv", parent_name="qrels")
        required_sha256 = _required_source_hashes(
            corpus=source_corpus,
            queries=source_queries,
            qrels=source_qrels,
        )
    extraction_state = {
        "md5": actual_md5,
        "files": files,
        "extracted_bytes": extracted_bytes,
        "required_sha256": required_sha256,
    }
    if marker != extraction_state:
        _write_json_atomic(extraction_marker, extraction_state)

    corpus_documents, empty_document_ids = _corpus_shape(source_corpus)
    queries = load_queries(source_queries)
    qrels = load_qrels(source_qrels)
    _validate_shape(spec, corpus_documents=corpus_documents, queries=queries, qrels=qrels)

    mapping_path = adapter_dir / "mapping.jsonl"
    corpus_dir = adapter_dir / "corpus"
    adapter_marker = adapter_dir / "state.json"
    adapter_state = _load_json_object(adapter_marker)
    expected_state = {
        "source_md5": actual_md5,
        "source_corpus_sha256": required_sha256["corpus"],
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "documents": corpus_documents,
    }
    if (
        any(adapter_state.get(key) != value for key, value in expected_state.items())
        or not corpus_dir.exists()
        or not mapping_path.exists()
    ):
        emit(f"Projecting {corpus_documents} BEIR documents for Fitz-Sage...")
        _materialize_adapter(
            source_corpus,
            adapter_dir,
            source_md5=actual_md5,
            source_corpus_sha256=required_sha256["corpus"],
            expected_documents=corpus_documents,
        )

    return PreparedDataset(
        name=spec.name,
        source="BEIR",
        dataset_page=DATASET_PAGE,
        paper_url=PAPER_URL,
        url=spec.url,
        archive=str(archive_path),
        md5=actual_md5,
        compressed_bytes=archive_path.stat().st_size,
        extracted_dir=str(extracted_dir),
        source_corpus=str(source_corpus),
        source_queries=str(source_queries),
        source_qrels=str(source_qrels),
        corpus_dir=str(corpus_dir),
        mapping_path=str(mapping_path),
        corpus_documents=corpus_documents,
        empty_documents=len(empty_document_ids),
        empty_judged_relevant_documents=len(
            empty_document_ids.intersection(
                document_id
                for judgments in qrels.values()
                for document_id, score in judgments.items()
                if score > 0
            )
        ),
        test_queries=len(qrels),
        qrels=sum(len(values) for values in qrels.values()),
        ignore_identical_ids=spec.ignore_identical_ids,
        adapter_schema_version=ADAPTER_SCHEMA_VERSION,
        adapter_projection="UTF-8 title + blank line + text; no rewriting or metadata",
        license_notice=LICENSE_NOTICE,
    )


def dataset_spec(name: str) -> DatasetSpec:
    """Resolve a supported dataset name."""
    normalized = str(name).strip().lower()
    try:
        return DATASETS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown BEIR dataset: {name}. Available: {', '.join(sorted(DATASETS))}"
        ) from exc


def iter_corpus(path: Path) -> Iterator[dict[str, Any]]:
    """Yield validated BEIR corpus records without changing their fields."""
    yield from _iter_jsonl(path, required=("_id", "title", "text"))


def load_queries(path: Path) -> dict[str, str]:
    """Load BEIR query IDs and text in archive order."""
    queries: dict[str, str] = {}
    for record in _iter_jsonl(path, required=("_id", "text")):
        query_id = _required_string(record, "_id")
        if query_id in queries:
            raise ValueError(f"Duplicate BEIR query ID: {query_id}")
        queries[query_id] = _required_string(record, "text", allow_empty=False)
    return queries


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Load graded BEIR test judgments keyed by query and corpus ID."""
    qrels: dict[str, dict[str, int]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        required = {"query-id", "corpus-id", "score"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Invalid BEIR qrels header in {path}: {reader.fieldnames}")
        for row in reader:
            query_id = str(row["query-id"])
            corpus_id = str(row["corpus-id"])
            try:
                score = int(row["score"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid BEIR qrel score for {query_id}/{corpus_id}") from exc
            judgments = qrels.setdefault(query_id, {})
            if corpus_id in judgments:
                raise ValueError(f"Duplicate BEIR qrel: {query_id}/{corpus_id}")
            judgments[corpus_id] = score
    return qrels


def load_mapping(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Return relative-path and document-ID lookup tables."""
    by_path: dict[str, str] = {}
    by_document: dict[str, str] = {}
    for raw in _iter_jsonl(path, required=("document_id", "relative_path", "content_sha256")):
        document_id = _required_string(raw, "document_id")
        relative_path = _required_string(raw, "relative_path")
        if relative_path in by_path or document_id in by_document:
            raise ValueError(f"Duplicate BEIR adapter mapping: {document_id}/{relative_path}")
        by_path[relative_path.replace("\\", "/")] = document_id
        by_document[document_id] = relative_path
    return by_path, by_document


def projected_content(record: dict[str, Any]) -> str:
    """Project BEIR's title and text fields without semantic transformation."""
    title = _required_string(record, "title")
    text = _required_string(record, "text")
    if title and text:
        return f"{title}\n\n{text}"
    return title or text


def _materialize_adapter(
    source_corpus: Path,
    adapter_dir: Path,
    *,
    source_md5: str,
    source_corpus_sha256: str,
    expected_documents: int,
) -> None:
    adapter_dir.parent.mkdir(parents=True, exist_ok=True)
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    adapter_dir.mkdir()
    mapping_path = adapter_dir / "mapping.jsonl"
    document_count = 0
    seen_paths: set[str] = set()
    try:
        with mapping_path.open("w", encoding="utf-8", newline="\n") as mapping_file:
            for record in iter_corpus(source_corpus):
                document_id = _required_string(record, "_id")
                digest = hashlib.sha256(document_id.encode("utf-8")).hexdigest()
                relative_path = f"{digest[:2]}/{digest}.txt"
                if relative_path in seen_paths:
                    raise ValueError(f"BEIR document-ID hash collision: {document_id}")
                seen_paths.add(relative_path)
                content = projected_content(record)
                destination = adapter_dir / "corpus" / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(content, encoding="utf-8", newline="\n")
                mapping = CorpusMapping(
                    document_id=document_id,
                    relative_path=relative_path,
                    content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                )
                mapping_file.write(json.dumps(mapping.to_dict(), ensure_ascii=True) + "\n")
                document_count += 1

        if document_count != expected_documents:
            raise ValueError(
                f"BEIR adapter wrote {document_count} documents; expected {expected_documents}"
            )
        _write_json_atomic(
            adapter_dir / "state.json",
            {
                "source_md5": source_md5,
                "source_corpus_sha256": source_corpus_sha256,
                "schema_version": ADAPTER_SCHEMA_VERSION,
                "documents": document_count,
            },
        )
    except Exception:
        shutil.rmtree(adapter_dir, ignore_errors=True)
        raise


def _validate_shape(
    spec: DatasetSpec,
    *,
    corpus_documents: int,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
) -> None:
    if corpus_documents != spec.corpus_documents:
        raise ValueError(
            f"BEIR {spec.name} corpus has {corpus_documents} documents; "
            f"expected {spec.corpus_documents}"
        )
    if len(qrels) != spec.test_queries:
        raise ValueError(
            f"BEIR {spec.name} test split has {len(qrels)} judged queries; "
            f"expected {spec.test_queries}"
        )
    missing_queries = sorted(set(qrels) - set(queries))
    if missing_queries:
        raise ValueError(f"BEIR qrels reference missing queries: {missing_queries[:3]}")


def _iter_jsonl(path: Path, *, required: tuple[str, ...]) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise TypeError(f"Expected JSON object at {path}:{line_number}")
            missing = [key for key in required if key not in record]
            if missing:
                raise ValueError(f"Missing {missing} at {path}:{line_number}")
            yield record


def _required_string(
    record: dict[str, Any],
    key: str,
    *,
    allow_empty: bool = True,
) -> str:
    value = record.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(
            f"BEIR field '{key}' must be a{' non-empty' if not allow_empty else ''} string"
        )
    return value


def _single_path(root: Path, filename: str, *, parent_name: str | None = None) -> Path:
    matches = [
        path
        for path in root.rglob(filename)
        if path.is_file() and (parent_name is None or path.parent.name == parent_name)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {filename} below {root}, found {len(matches)}")
    return matches[0]


def _corpus_shape(path: Path) -> tuple[int, set[str]]:
    count = 0
    empty_document_ids: set[str] = set()
    for record in iter_corpus(path):
        count += 1
        if not projected_content(record).strip():
            empty_document_ids.add(_required_string(record, "_id"))
    return count, empty_document_ids


def _required_source_hashes(
    *,
    corpus: Path,
    queries: Path,
    qrels: Path,
) -> dict[str, str]:
    return {
        "corpus": file_digest(corpus, "sha256"),
        "queries": file_digest(queries, "sha256"),
        "qrels": file_digest(qrels, "sha256"),
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
