"""Versioned execution records for governed retrieval."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from .evidence import EvidencePack

RETRIEVAL_RUN_SCHEMA_VERSION = "2.0"


def _require_supported_schema(value: Any) -> str:
    version = str(value or "")
    if version.split(".", 1)[0] != RETRIEVAL_RUN_SCHEMA_VERSION.split(".", 1)[0]:
        raise ValueError(
            f"Unsupported retrieval-run schema {version!r}; "
            f"supported major version is {RETRIEVAL_RUN_SCHEMA_VERSION.split('.', 1)[0]}."
        )
    return version


def _content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, payload: str) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.write("\n")
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class QueryTerm:
    """One term used to prepare lexical retrieval."""

    text: str
    origin: str

    def to_dict(self) -> dict[str, str]:
        return {"text": self.text, "origin": self.origin}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "QueryTerm":
        return cls(text=str(raw.get("text") or ""), origin=str(raw.get("origin") or "unknown"))


@dataclass(frozen=True)
class QueryExecution:
    """Stable query-planning fields that materially affect retrieval."""

    source_text: str
    sanitized_text: str
    retrieval_text: str
    query_shape: str
    query_contract: str | None = None
    specificity: str | None = None
    answer_type: str | None = None
    retrieval_modality: str | None = None
    required_modalities: tuple[str, ...] = ()
    comparison_queries: tuple[str, ...] = ()
    comparison_entities: tuple[str, ...] = ()
    has_comparison_intent: bool = False
    has_aggregation_intent: bool = False
    has_temporal_intent: bool = False
    inject_corpus_summaries: bool = False
    terms: tuple[QueryTerm, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_text": self.source_text,
            "sanitized_text": self.sanitized_text,
            "retrieval_text": self.retrieval_text,
            "query_shape": self.query_shape,
            "query_contract": self.query_contract,
            "specificity": self.specificity,
            "answer_type": self.answer_type,
            "retrieval_modality": self.retrieval_modality,
            "required_modalities": list(self.required_modalities),
            "comparison_queries": list(self.comparison_queries),
            "comparison_entities": list(self.comparison_entities),
            "has_comparison_intent": self.has_comparison_intent,
            "has_aggregation_intent": self.has_aggregation_intent,
            "has_temporal_intent": self.has_temporal_intent,
            "inject_corpus_summaries": self.inject_corpus_summaries,
            "terms": [term.to_dict() for term in self.terms],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "QueryExecution":
        return cls(
            source_text=str(raw.get("source_text") or ""),
            sanitized_text=str(raw.get("sanitized_text") or ""),
            retrieval_text=str(raw.get("retrieval_text") or ""),
            query_shape=str(raw.get("query_shape") or "narrow"),
            query_contract=_optional_text(raw.get("query_contract")),
            specificity=_optional_text(raw.get("specificity")),
            answer_type=_optional_text(raw.get("answer_type")),
            retrieval_modality=_optional_text(raw.get("retrieval_modality")),
            required_modalities=_text_tuple(raw.get("required_modalities")),
            comparison_queries=_text_tuple(raw.get("comparison_queries")),
            comparison_entities=_text_tuple(raw.get("comparison_entities")),
            has_comparison_intent=bool(raw.get("has_comparison_intent", False)),
            has_aggregation_intent=bool(raw.get("has_aggregation_intent", False)),
            has_temporal_intent=bool(raw.get("has_temporal_intent", False)),
            inject_corpus_summaries=bool(raw.get("inject_corpus_summaries", False)),
            terms=tuple(QueryTerm.from_dict(item) for item in _dict_items(raw.get("terms"))),
        )


@dataclass(frozen=True)
class StrategyExecution:
    """One typed retrieval strategy invocation."""

    strategy: str
    query: str
    result_count: int
    succeeded: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "query": self.query,
            "result_count": self.result_count,
            "succeeded": self.succeeded,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StrategyExecution":
        return cls(
            strategy=str(raw.get("strategy") or ""),
            query=str(raw.get("query") or ""),
            result_count=_int(raw.get("result_count")),
            succeeded=bool(raw.get("succeeded", True)),
        )


@dataclass(frozen=True)
class CandidateReference:
    """Non-content identity and score for one candidate at one stage."""

    rank: int
    kind: str
    source_id: str
    location: str
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "kind": self.kind,
            "source_id": self.source_id,
            "location": self.location,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CandidateReference":
        return cls(
            rank=max(1, _int(raw.get("rank"), default=1)),
            kind=str(raw.get("kind") or ""),
            source_id=str(raw.get("source_id") or ""),
            location=str(raw.get("location") or ""),
            score=_optional_float(raw.get("score")),
        )


@dataclass(frozen=True)
class CandidateStage:
    """Ordered candidate references at a stable pipeline boundary."""

    name: str
    candidates: tuple[CandidateReference, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": len(self.candidates),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CandidateStage":
        return cls(
            name=str(raw.get("name") or ""),
            candidates=tuple(
                CandidateReference.from_dict(item) for item in _dict_items(raw.get("candidates"))
            ),
        )


@dataclass(frozen=True)
class PyrrhoExecution:
    """Pyrrho's authoritative decision over the delivered evidence."""

    verdict: str
    evidence_count: int
    reasons: tuple[str, ...] = ()
    decision: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "evidence_count": self.evidence_count,
            "reasons": list(self.reasons),
            "decision": dict(self.decision),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PyrrhoExecution":
        decision = raw.get("decision")
        if not isinstance(decision, dict):
            raise ValueError("Pyrrho execution is missing its serialized decision.")
        verdict = _required_text(raw, "verdict")
        if decision.get("verdict") != verdict:
            raise ValueError("Pyrrho execution verdict differs from its serialized decision.")
        return cls(
            verdict=verdict,
            evidence_count=max(0, _int(raw.get("evidence_count"))),
            reasons=_text_tuple(raw.get("reasons")),
            decision=dict(decision),
        )


@dataclass(frozen=True)
class FrozenEvidence:
    """Content-bearing evidence frozen at a named retrieval boundary."""

    rank: int
    source_id: str
    file_path: str
    address_kind: str
    address_location: str
    address_summary: str
    line_range: tuple[int, int] | None
    score: float | None
    content: str | None
    content_sha256: str
    content_chars: int
    compiler_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        rank: int,
        source_id: str,
        file_path: str,
        address_kind: str,
        address_location: str,
        address_summary: str,
        line_range: tuple[int, int] | None,
        score: float | None,
        content: str,
        compiler_metadata: dict[str, Any] | None = None,
    ) -> "FrozenEvidence":
        return cls(
            rank=rank,
            source_id=source_id,
            file_path=file_path,
            address_kind=address_kind,
            address_location=address_location,
            address_summary=address_summary,
            line_range=line_range,
            score=score,
            content=content,
            content_sha256=_content_sha256(content),
            content_chars=len(content),
            compiler_metadata=dict(compiler_metadata or {}),
        )

    def to_dict(self, *, include_content: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "rank": self.rank,
            "source_id": self.source_id,
            "file_path": self.file_path,
            "address_kind": self.address_kind,
            "address_location": self.address_location,
            "line_range": list(self.line_range) if self.line_range else None,
            "score": self.score,
            "content_sha256": self.content_sha256,
            "content_chars": self.content_chars,
            "compiler_metadata": (
                self.compiler_metadata
                if include_content
                else _redacted_compiler_metadata(self.compiler_metadata)
            ),
        }
        if include_content:
            payload["address_summary"] = self.address_summary
            payload["content"] = self.content
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "FrozenEvidence":
        line_range = raw.get("line_range")
        parsed_range = None
        if isinstance(line_range, (list, tuple)) and len(line_range) == 2:
            parsed_range = (_int(line_range[0]), _int(line_range[1]))
        content = raw.get("content")
        if not isinstance(content, str):
            content = None
        compiler = raw.get("compiler_metadata")
        return cls(
            rank=max(1, _int(raw.get("rank"), default=1)),
            source_id=str(raw.get("source_id") or ""),
            file_path=str(raw.get("file_path") or ""),
            address_kind=str(raw.get("address_kind") or ""),
            address_location=str(raw.get("address_location") or ""),
            address_summary=str(raw.get("address_summary") or ""),
            line_range=parsed_range,
            score=_optional_float(raw.get("score")),
            content=content,
            content_sha256=str(
                raw.get("content_sha256")
                or (_content_sha256(content) if content is not None else "")
            ),
            content_chars=max(
                0,
                _int(
                    raw.get("content_chars"),
                    default=len(content) if content is not None else 0,
                ),
            ),
            compiler_metadata=dict(compiler) if isinstance(compiler, dict) else {},
        )

    def verify_content(self) -> bool:
        """Return whether included content matches the recorded digest and length."""
        if self.content is None:
            return False
        return (
            len(self.content) == self.content_chars
            and _content_sha256(self.content) == self.content_sha256
        )


@dataclass(frozen=True)
class RunEnvironment:
    """Reproducibility metadata that does not expose provider credentials."""

    fitz_sage_version: str
    engine: str
    collection: str
    config_sha256: str
    collection_sha256: str | None
    components: dict[str, str] = field(default_factory=dict)
    indexing_status: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fitz_sage_version": self.fitz_sage_version,
            "engine": self.engine,
            "collection": self.collection,
            "config_sha256": self.config_sha256,
            "collection_sha256": self.collection_sha256,
            "components": dict(self.components),
            "indexing_status": self.indexing_status,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunEnvironment":
        components = raw.get("components")
        status = raw.get("indexing_status")
        return cls(
            fitz_sage_version=str(raw.get("fitz_sage_version") or "unknown"),
            engine=str(raw.get("engine") or "unknown"),
            collection=str(raw.get("collection") or "unknown"),
            config_sha256=str(raw.get("config_sha256") or ""),
            collection_sha256=_optional_text(raw.get("collection_sha256")),
            components={
                str(key): str(value)
                for key, value in (components.items() if isinstance(components, dict) else ())
            },
            indexing_status=dict(status) if isinstance(status, dict) else {},
        )


@dataclass
class RetrievalRun:
    """A complete, inspectable record of one governed retrieval execution."""

    run_id: str
    created_at: str
    query: QueryExecution
    evidence: EvidencePack
    strategies: tuple[StrategyExecution, ...]
    candidate_stages: tuple[CandidateStage, ...]
    pyrrho: PyrrhoExecution
    ranked_evidence: tuple[FrozenEvidence, ...]
    pyrrho_evidence: tuple[FrozenEvidence, ...]
    environment: RunEnvironment
    warnings: tuple[str, ...] = ()
    schema_version: str = RETRIEVAL_RUN_SCHEMA_VERSION
    content_included: bool = True

    def to_dict(self, *, include_content: bool = False) -> dict[str, Any]:
        """Return a stable JSON-compatible record.

        Source excerpts and content are redacted unless explicitly requested.
        A redacted record remains explainable but cannot be replayed.
        """
        evidence = self.evidence.to_dict()
        if not include_content:
            for item in evidence["items"]:
                content = str(item.get("content") or "")
                item["content_sha256"] = _content_sha256(content)
                item["content_chars"] = len(content)
                item["content"] = ""
                item["excerpt"] = ""
                item["metadata"] = _redacted_metadata(item.get("metadata"))
            evidence["metadata"] = _redacted_pack_metadata(evidence.get("metadata"))
        return {
            "schema_version": self.schema_version,
            "record_type": "retrieval_run",
            "run_id": self.run_id,
            "created_at": self.created_at,
            "content_included": include_content,
            "query": self.query.to_dict(),
            "evidence": evidence,
            "strategies": [strategy.to_dict() for strategy in self.strategies],
            "candidate_stages": [stage.to_dict() for stage in self.candidate_stages],
            "pyrrho": self.pyrrho.to_dict(),
            "ranked_evidence": [
                item.to_dict(include_content=include_content) for item in self.ranked_evidence
            ],
            "pyrrho_evidence": [
                item.to_dict(include_content=include_content) for item in self.pyrrho_evidence
            ],
            "environment": self.environment.to_dict(),
            "warnings": list(self.warnings),
        }

    def to_json(self, *, include_content: bool = False, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(include_content=include_content), **kwargs)

    def write(
        self,
        path: str | Path,
        *,
        include_content: bool = False,
        indent: int | None = 2,
    ) -> Path:
        """Atomically write a trace file and return its resolved path."""
        output = Path(path).expanduser().resolve()
        _write_json_atomic(
            output,
            self.to_json(include_content=include_content, indent=indent),
        )
        return output

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RetrievalRun":
        if raw.get("record_type") != "retrieval_run":
            raise ValueError("JSON record is not a retrieval run.")
        schema_version = _require_supported_schema(raw.get("schema_version"))
        evidence = raw.get("evidence")
        query = raw.get("query")
        pyrrho = raw.get("pyrrho")
        environment = raw.get("environment")
        if not all(isinstance(value, dict) for value in (evidence, query, pyrrho, environment)):
            raise ValueError("Retrieval run is missing required structured fields.")
        content_included = bool(raw.get("content_included", False))
        run = cls(
            run_id=_required_text(raw, "run_id"),
            created_at=_required_text(raw, "created_at"),
            query=QueryExecution.from_dict(cast(dict[str, Any], query)),
            evidence=EvidencePack.from_dict(cast(dict[str, Any], evidence)),
            strategies=tuple(
                StrategyExecution.from_dict(item) for item in _dict_items(raw.get("strategies"))
            ),
            candidate_stages=tuple(
                CandidateStage.from_dict(item) for item in _dict_items(raw.get("candidate_stages"))
            ),
            pyrrho=PyrrhoExecution.from_dict(cast(dict[str, Any], pyrrho)),
            ranked_evidence=tuple(
                FrozenEvidence.from_dict(item) for item in _dict_items(raw.get("ranked_evidence"))
            ),
            pyrrho_evidence=tuple(
                FrozenEvidence.from_dict(item) for item in _dict_items(raw.get("pyrrho_evidence"))
            ),
            environment=RunEnvironment.from_dict(cast(dict[str, Any], environment)),
            warnings=_text_tuple(raw.get("warnings")),
            schema_version=schema_version,
            content_included=content_included,
        )
        if content_included:
            if len(run.pyrrho_evidence) != run.pyrrho.evidence_count:
                raise ValueError("Pyrrho evidence count differs from the recorded decision input.")
            if len(run.evidence.items) != len(run.pyrrho_evidence):
                raise ValueError("Delivered evidence differs from the recorded Pyrrho input.")
            invalid = [
                item.rank
                for item in (*run.ranked_evidence, *run.pyrrho_evidence)
                if not item.verify_content()
            ]
            if invalid:
                raise ValueError(
                    "Retrieval-run evidence content failed integrity checks at ranks "
                    + ", ".join(str(rank) for rank in invalid)
                    + "."
                )
            mismatched = [
                frozen.rank
                for delivered, frozen in zip(
                    run.evidence.items,
                    run.pyrrho_evidence,
                    strict=True,
                )
                if delivered.source_id != frozen.source_id
                or _content_sha256(delivered.content) != frozen.content_sha256
            ]
            if mismatched:
                raise ValueError(
                    "Delivered evidence differs from Pyrrho input at ranks "
                    + ", ".join(str(rank) for rank in mismatched)
                    + "."
                )
        return run

    @classmethod
    def from_json(cls, payload: str) -> "RetrievalRun":
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid retrieval-run JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError("Retrieval-run JSON must contain an object.")
        return cls.from_dict(raw)

    @classmethod
    def read(cls, path: str | Path) -> "RetrievalRun":
        return cls.from_json(Path(path).expanduser().read_text(encoding="utf-8"))

    def explain(self) -> str:
        """Render a deterministic, content-free explanation of this execution."""
        term_groups: dict[str, list[str]] = {}
        for term in self.query.terms:
            term_groups.setdefault(term.origin, []).append(term.text)

        lines = [
            f"Retrieval run {self.run_id}",
            f"Query: {self.query.source_text}",
            (f"Plan: {self.query.query_shape}; retrieval query: {self.query.retrieval_text!r}"),
        ]
        for origin, terms in term_groups.items():
            lines.append(f"Terms ({origin}): {', '.join(terms)}")
        if self.strategies:
            strategy_text = ", ".join(
                f"{item.strategy} ({item.result_count})" for item in self.strategies
            )
            lines.append(f"Strategies: {strategy_text}")
        if self.candidate_stages:
            lines.append(
                "Candidates: "
                + " -> ".join(
                    f"{stage.name}={len(stage.candidates)}" for stage in self.candidate_stages
                )
            )
        lines.append(f"Pyrrho: {self.pyrrho.verdict}; evidence={self.pyrrho.evidence_count}")
        if self.evidence.items:
            lines.append("Evidence:")
            lines.extend(
                f"  {item.rank}. {item.file_path} :: {item.address_location}"
                for item in self.evidence.items
            )
        else:
            lines.append("Evidence: none")
        lines.append(
            f"Collection: {self.environment.collection}; "
            f"index complete={self.environment.indexing_status.get('complete', 'unknown')}"
        )
        if not self.content_included:
            lines.append("Replay: unavailable (trace content was redacted)")
        if self.warnings:
            lines.extend(f"Warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)


@dataclass
class PyrrhoReplay:
    """Comparison between recorded and replayed Pyrrho decisions."""

    replay_id: str
    source_run_id: str
    created_at: str
    pyrrho_spec: str
    source_fitz_sage_version: str
    replay_fitz_sage_version: str
    original: PyrrhoExecution
    replayed: PyrrhoExecution
    evidence: EvidencePack
    duration_seconds: float
    schema_version: str = RETRIEVAL_RUN_SCHEMA_VERSION
    content_included: bool = True

    def to_dict(self, *, include_content: bool = False) -> dict[str, Any]:
        evidence = self.evidence.to_dict()
        if not include_content:
            for item in evidence["items"]:
                content = str(item.get("content") or "")
                item["content_sha256"] = _content_sha256(content)
                item["content_chars"] = len(content)
                item["content"] = ""
                item["excerpt"] = ""
                item["metadata"] = _redacted_metadata(item.get("metadata"))
        return {
            "schema_version": self.schema_version,
            "record_type": "pyrrho_replay",
            "replay_id": self.replay_id,
            "source_run_id": self.source_run_id,
            "created_at": self.created_at,
            "pyrrho_spec": self.pyrrho_spec,
            "source_fitz_sage_version": self.source_fitz_sage_version,
            "replay_fitz_sage_version": self.replay_fitz_sage_version,
            "content_included": include_content,
            "original": self.original.to_dict(),
            "replayed": self.replayed.to_dict(),
            "changed": self.original.verdict != self.replayed.verdict,
            "duration_seconds": self.duration_seconds,
            "evidence": evidence,
        }

    def to_json(self, *, include_content: bool = False, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(include_content=include_content), **kwargs)

    def write(
        self,
        path: str | Path,
        *,
        include_content: bool = False,
        indent: int | None = 2,
    ) -> Path:
        output = Path(path).expanduser().resolve()
        _write_json_atomic(
            output,
            self.to_json(include_content=include_content, indent=indent),
        )
        return output

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PyrrhoReplay":
        if raw.get("record_type") != "pyrrho_replay":
            raise ValueError("JSON record is not a Pyrrho replay.")
        schema_version = _require_supported_schema(raw.get("schema_version"))
        original = raw.get("original")
        replayed = raw.get("replayed")
        evidence = raw.get("evidence")
        if not all(isinstance(value, dict) for value in (original, replayed, evidence)):
            raise ValueError("Pyrrho replay is missing required structured fields.")
        return cls(
            replay_id=_required_text(raw, "replay_id"),
            source_run_id=_required_text(raw, "source_run_id"),
            created_at=_required_text(raw, "created_at"),
            pyrrho_spec=_required_text(raw, "pyrrho_spec"),
            source_fitz_sage_version=_required_text(
                raw,
                "source_fitz_sage_version",
            ),
            replay_fitz_sage_version=_required_text(
                raw,
                "replay_fitz_sage_version",
            ),
            original=PyrrhoExecution.from_dict(cast(dict[str, Any], original)),
            replayed=PyrrhoExecution.from_dict(cast(dict[str, Any], replayed)),
            evidence=EvidencePack.from_dict(cast(dict[str, Any], evidence)),
            duration_seconds=_float(raw.get("duration_seconds")),
            schema_version=schema_version,
            content_included=bool(raw.get("content_included", False)),
        )

    @classmethod
    def from_json(cls, payload: str) -> "PyrrhoReplay":
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid Pyrrho-replay JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError("Pyrrho-replay JSON must contain an object.")
        return cls.from_dict(raw)

    @classmethod
    def read(cls, path: str | Path) -> "PyrrhoReplay":
        return cls.from_json(Path(path).expanduser().read_text(encoding="utf-8"))

    def explain(self) -> str:
        changed = self.original.verdict != self.replayed.verdict
        return "\n".join(
            [
                f"Pyrrho replay {self.replay_id}",
                f"Source run: {self.source_run_id}",
                f"Provider: {self.pyrrho_spec}",
                (f"Fitz: {self.source_fitz_sage_version} -> {self.replay_fitz_sage_version}"),
                (f"Original: {self.original.verdict}; evidence={self.original.evidence_count}"),
                (f"Replayed: {self.replayed.verdict}; evidence={self.replayed.evidence_count}"),
                f"Changed: {'yes' if changed else 'no'}",
            ]
        )


def _redacted_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compiler = value.get("evidence_compiler")
    safe_compiler = _redacted_compiler_metadata(compiler)
    return {"evidence_compiler": safe_compiler} if safe_compiler else {}


def _redacted_compiler_metadata(value: Any) -> dict[str, Any]:
    """Preserve mechanical compiler facts without source-derived role labels."""
    if not isinstance(value, dict):
        return {}
    safe: dict[str, Any] = {}
    for key in ("rank", "alignment_score"):
        field_value = value.get(key)
        if isinstance(field_value, (int, float)) and not isinstance(field_value, bool):
            safe[key] = field_value
    return safe


def _redacted_pack_metadata(value: Any) -> dict[str, Any]:
    """Keep only non-content pack metadata duplicated by the typed record."""
    if not isinstance(value, dict):
        return {}
    engine = value.get("engine")
    return {"engine": engine} if isinstance(engine, str) else {}


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _required_text(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Retrieval record is missing required field {key!r}.")
    return value


def _text_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple, set)):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str) and item)


def _dict_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, dict)]


def _int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "RETRIEVAL_RUN_SCHEMA_VERSION",
    "CandidateReference",
    "CandidateStage",
    "FrozenEvidence",
    "PyrrhoExecution",
    "PyrrhoReplay",
    "QueryExecution",
    "QueryTerm",
    "RetrievalRun",
    "RunEnvironment",
    "StrategyExecution",
]
