# fitz_sage/engines/fitz_krag/evidence_closure.py
"""Evidence-contract follow-up retrieval planning for evidence packs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from fitz_sage.engines.fitz_krag.evidence_compiler import EvidenceCompilation
from fitz_sage.engines.fitz_krag.evidence_contract import QueryContract, build_query_contract
from fitz_sage.engines.fitz_krag.types import ReadResult

_BRIDGE_IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])_?[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+"
    r"(?:[-_][A-Za-z0-9]+)*(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])(?=[A-Za-z0-9-]*\d)[A-Za-z][A-Za-z0-9]*"
    r"(?:-[A-Za-z0-9]+)+(?![A-Za-z0-9_])|"
    r"\b[A-Z]{2,}[A-Z0-9]*\d[A-Z0-9_-]*\b|"
    r"\b[A-Z]\d+\b"
)
_BACKTICK_PATTERN = re.compile(r"`([^`]{2,80})`")
_SNAKE_TOKEN_PATTERN = re.compile(r"\b[a-z][a-z0-9]+(?:_[a-z0-9]+)+\b")
_ACRONYM_PATTERN = re.compile(r"\b[A-Z]{2,6}\b")
_UUID_PATTERN = re.compile(
    r"\b[0-9a-f]{4,}(?:-[0-9a-f]{4,})+\b|" r"\b[0-9a-f]{8,}\b",
    re.IGNORECASE,
)
_SOURCE_REFERENCE_PATTERN = re.compile(
    r"\b(?:according\s+to|follow|following|follows|from|see|use|using)\s+(?:the\s+)?"
    r"([A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z][A-Za-z0-9-]*){0,3}\s+"
    r"(?:addendum|brief|contract|document|guide|matrix|notes?|playbook|policy|postmortem|"
    r"readme|report|rules?|sla|table))\b",
    re.IGNORECASE,
)
_SOURCE_FILE_TOKEN_PATTERN = re.compile(
    r"\b[A-Za-z0-9_.-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|rb|php|cs|cpp|c|h|hpp|md|csv|json|yaml|yml|toml)\b",
    re.IGNORECASE,
)
_EXPLICIT_DEFINITION_PATTERN = re.compile(
    r"\b(?P<label>[A-Z][A-Z0-9_-]{1,11})\s+"
    r"(?:means|stands\s+for|refers\s+to|is\s+short\s+for)\s+"
    r"(?P<definition>[^\n.;:!?]{3,120})"
)
_DEFINITION_CONTEXT_TAIL = re.compile(
    r"\s+(?:according\s+to|for\s+(?:this|the)|in\s+(?:this|the)|"
    r"throughout\s+(?:this|the)|within\s+(?:this|the))\b.*$",
    re.IGNORECASE,
)
_TABLE_HINTS = {
    "alert",
    "alerts",
    "asset",
    "assets",
    "deployment",
    "deployments",
    "experiment",
    "experiments",
    "feature flag",
    "feature flags",
    "invoice",
    "invoices",
    "release",
    "row",
    "table",
}
_DOCUMENT_HINTS = {
    "addendum",
    "brief",
    "contract",
    "document",
    "guide",
    "matrix",
    "note",
    "notes",
    "playbook",
    "policy",
    "postmortem",
    "readme",
    "report",
    "sla",
}
_QUERY_TERM_STOPWORDS = {
    "and",
    "are",
    "does",
    "for",
    "from",
    "how",
    "into",
    "the",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}
_GENERATED_BRIDGE_TERMS = {
    "aligned",
    "csv",
    "deterministic table",
    "required_section",
    "required_symbol",
    "required_table",
    "residual",
}


@dataclass(frozen=True)
class EvidenceClosureRequest:
    """One bounded follow-up retrieval request derived from a Pyrrho contract."""

    query: str
    modality: str
    role: str
    reason: str
    bridges: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceClosurePlan:
    """Follow-up retrieval plan plus serializable trace metadata."""

    requests: list[EvidenceClosureRequest] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def plan_evidence_closure(
    query: str,
    current_results: list[ReadResult],
    compilation: EvidenceCompilation,
    profile: Any = None,
    *,
    max_requests: int = 6,
) -> EvidenceClosurePlan:
    """Plan bounded follow-up searches for unresolved Pyrrho obligations."""
    contract = build_query_contract(query, profile)
    existing_roles = _existing_compiler_roles(compilation.results)
    definition_bridges = _definition_bridge_phrases(
        query,
        current_results,
        compilation.results,
    )
    bridge_terms = _bridge_terms(query, current_results, compilation.results, contract)
    existing_modalities = _existing_modalities(compilation.results or current_results)

    requests: list[EvidenceClosureRequest] = []

    def add(
        *,
        modality: str,
        role: str,
        reason: str,
        primary_terms: list[str],
    ) -> None:
        if len(requests) >= max_requests:
            return
        request_query = _request_query(
            primary_terms,
            contract=contract,
            bridge_terms=bridge_terms,
            modality=modality,
        )
        if not request_query:
            return
        key = (modality, role, _normalize(request_query))
        if key in {(item.modality, item.role, _normalize(item.query)) for item in requests}:
            return
        requests.append(
            EvidenceClosureRequest(
                query=request_query,
                modality=modality,
                role=role,
                reason=reason,
                bridges=tuple(bridge_terms[:12]),
            )
        )

    for modality in contract.required_modalities:
        role = f"required_{modality}"
        if role in existing_roles or modality in existing_modalities:
            continue
        add(
            modality=modality,
            role=role,
            reason="missing_required_modality",
            primary_terms=[modality],
        )

    for definition in definition_bridges:
        role = f"bridge_definition:{_normalize(definition)}"
        if role in existing_roles:
            continue
        add(
            modality="section",
            role=role,
            reason="bridge_definition",
            primary_terms=[definition],
        )

    if "section" in contract.required_modalities:
        for document in _document_bridge_terms(bridge_terms):
            role = f"bridge_document:{_normalize(document)}"
            if role in existing_roles:
                continue
            add(
                modality="section",
                role=role,
                reason="bridge_document",
                primary_terms=[document],
            )

    if "table" in contract.required_modalities:
        query_identifiers = {value.lower() for value in contract.identifiers}
        for identifier in _structured_bridge_identifiers(bridge_terms):
            if identifier.lower() in query_identifiers:
                continue
            role = f"bridge:{identifier}"
            if role in existing_roles:
                continue
            add(
                modality="table",
                role=role,
                reason="bridge_identifier",
                primary_terms=[identifier],
            )

    metadata = {
        "contract": {
            "query_contract": contract.query_contract,
            "answerability_shape": contract.answerability_shape,
            "retrieval_modality": contract.retrieval_modality,
            "retrieval_obligation": contract.retrieval_obligation,
            "identifiers": list(contract.identifiers),
            "required_modalities": list(contract.required_modalities),
            "temporal_policy": contract.temporal_policy,
        },
        "existing_roles": sorted(existing_roles),
        "existing_modalities": sorted(existing_modalities),
        "bridge_terms": bridge_terms,
        "definition_bridges": definition_bridges,
        "request_count": len(requests),
        "requests": [request_metadata(request) for request in requests],
    }
    return EvidenceClosurePlan(requests=requests, metadata=metadata)


def annotate_closure_result(
    result: ReadResult,
    request: EvidenceClosureRequest,
    *,
    contract: QueryContract,
    run_index: int,
) -> ReadResult:
    """Attach bridge provenance to a follow-up read result."""
    metadata = dict(result.metadata)
    metadata["evidence_closure"] = {
        "run": run_index,
        "role": request.role,
        "reason": request.reason,
        "modality": request.modality,
        "query": request.query,
        "bridges": list(request.bridges),
        "contract_identifiers": list(contract.identifiers),
    }
    return ReadResult(
        address=result.address,
        content=result.content,
        file_path=result.file_path,
        line_range=result.line_range,
        metadata=metadata,
    )


def request_metadata(request: EvidenceClosureRequest) -> dict[str, Any]:
    """Serialize one closure request."""
    return {
        "query": request.query,
        "modality": request.modality,
        "role": request.role,
        "reason": request.reason,
        "bridges": list(request.bridges),
    }


def _existing_compiler_roles(results: list[ReadResult]) -> set[str]:
    roles: set[str] = set()
    for result in results:
        compiler = result.metadata.get("evidence_compiler")
        if not isinstance(compiler, dict):
            continue
        values = compiler.get("roles", [])
        if isinstance(values, list):
            roles.update(str(value) for value in values)
    return roles


def _existing_modalities(results: list[ReadResult]) -> set[str]:
    return {result.address.kind.value for result in results}


def _bridge_terms(
    query: str,
    current_results: list[ReadResult],
    compiled_results: list[ReadResult],
    contract: QueryContract,
) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        text = str(value).strip().strip(".,;:()[]{}")
        if not text or "\n" in text:
            return
        normalized = _normalize(text)
        if (
            len(normalized) < 2
            or normalized in seen
            or normalized in _GENERATED_BRIDGE_TERMS
            or _UUID_PATTERN.fullmatch(normalized)
        ):
            return
        seen.add(normalized)
        terms.append(text)

    for value in contract.identifiers:
        add(value)
    for value in _specific_query_terms(contract):
        add(value)

    evidence = _bridge_seed_results(query, current_results, compiled_results)

    for value in _definition_bridge_phrases(query, current_results, compiled_results):
        add(value)

    for result in evidence[:10]:
        if _result_kind(result) == "table":
            continue
        text = _result_text(result)
        for match in _BRIDGE_IDENTIFIER_PATTERN.finditer(text):
            add(match.group(0))
        for match in _ACRONYM_PATTERN.finditer(text):
            add(match.group(0))
        for match in _BACKTICK_PATTERN.finditer(text):
            add(match.group(1))
        for match in _SNAKE_TOKEN_PATTERN.finditer(text):
            add(match.group(0))
        for match in _SOURCE_REFERENCE_PATTERN.finditer(text):
            add(match.group(1))
        for match in _SOURCE_FILE_TOKEN_PATTERN.finditer(text):
            add(match.group(0))
        for hint in (*_TABLE_HINTS, *_DOCUMENT_HINTS):
            if re.search(rf"\b{re.escape(hint)}\b", _normalize(text)):
                add(hint)

    for result in _reference_seed_results(compiled_results)[:10]:
        text = _result_text(result)
        for match in _SOURCE_REFERENCE_PATTERN.finditer(text):
            add(match.group(1))
        for match in _SOURCE_FILE_TOKEN_PATTERN.finditer(text):
            add(match.group(0))

    return terms[:32]


def _bridge_seed_results(
    query: str,
    current_results: list[ReadResult],
    compiled_results: list[ReadResult],
) -> list[ReadResult]:
    """Return only evidence selected for a real contract obligation."""
    evidence = [result for result in compiled_results if _result_has_bridge_seed_role(result)]
    for result in _query_referenced_results(query, compiled_results):
        if result not in evidence:
            evidence.append(result)
    if not evidence and not compiled_results:
        return list(current_results)
    return evidence


def _query_referenced_results(
    query: str,
    compiled_results: list[ReadResult],
) -> list[ReadResult]:
    """Return evidence whose source is explicitly named by the query."""
    references = {
        _normalize_document_bridge(match.group(1))
        for match in _SOURCE_REFERENCE_PATTERN.finditer(query)
    }
    if not references:
        return []

    matches: list[ReadResult] = []
    for result in compiled_results:
        labels = {
            _normalize_document_bridge(str(result.address.location)),
            _normalize_document_bridge(str(result.address.metadata.get("document_title") or "")),
        }
        if references & labels:
            matches.append(result)
    return matches


def _reference_seed_results(compiled_results: list[ReadResult]) -> list[ReadResult]:
    """Return evidence allowed to contribute explicit source references only."""
    return [
        result
        for result in compiled_results
        if _result_has_compiler_role(result, ("required_",))
    ]


def _definition_bridge_phrases(
    query: str,
    current_results: list[ReadResult],
    compiled_results: list[ReadResult],
) -> list[str]:
    """Extract corpus-stated expansions for labels that occur in the query."""
    phrases: list[str] = []
    seen: set[str] = set()
    for result in _bridge_seed_results(query, current_results, compiled_results)[:10]:
        if _result_kind(result) == "table":
            continue
        for match in _EXPLICIT_DEFINITION_PATTERN.finditer(_result_text(result)):
            label = match.group("label")
            if not re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(label)}(?![A-Za-z0-9_])",
                query,
                re.IGNORECASE,
            ):
                continue
            phrase = _DEFINITION_CONTEXT_TAIL.sub(
                "",
                match.group("definition").strip().strip("\"'()[]{}"),
            ).strip()
            normalized = _normalize(phrase)
            word_count = len(normalized.split())
            if (
                word_count < 2
                or word_count > 10
                or normalized in seen
                or normalized in _normalize(query)
            ):
                continue
            seen.add(normalized)
            phrases.append(phrase)
    return phrases[:8]


def _result_kind(result: ReadResult) -> str:
    return result.address.kind.value


def _result_has_bridge_seed_role(result: ReadResult) -> bool:
    """Return whether compilation selected this result for a real obligation."""
    return _result_has_compiler_role(
        result,
        (
            "anchor_identifier:",
            "anchor_keyword:",
        ),
    )


def _result_has_compiler_role(result: ReadResult, prefixes: tuple[str, ...]) -> bool:
    """Return whether evidence compilation assigned one of the given roles."""
    compiler = result.metadata.get("evidence_compiler")
    if not isinstance(compiler, dict):
        return False
    roles = compiler.get("roles")
    if not isinstance(roles, list):
        return False
    return any(str(role).startswith(prefixes) for role in roles)


def _specific_query_terms(contract: QueryContract) -> list[str]:
    terms: list[str] = []
    for term in contract.keyword_anchors:
        normalized = _normalize(term)
        if len(normalized) < 3 or normalized in _QUERY_TERM_STOPWORDS:
            continue
        terms.append(term)
    return terms[:12]


def _request_query(
    primary_terms: list[str],
    *,
    contract: QueryContract,
    bridge_terms: list[str],
    modality: str,
) -> str:
    if modality == "table":
        return _table_request_query(primary_terms, contract, bridge_terms)
    if modality == "symbol":
        return _symbol_request_query(primary_terms, contract, bridge_terms)
    return _section_request_query(primary_terms, contract, bridge_terms)


def _table_request_query(
    primary_terms: list[str],
    contract: QueryContract,
    bridge_terms: list[str],
) -> str:
    terms: list[str] = []
    add = _term_adder(terms)
    for value in primary_terms:
        add(value)
    for value in contract.identifiers:
        add(value)
    for value in _identifier_terms(bridge_terms):
        add(value)
    for value in _hint_terms(bridge_terms, _TABLE_HINTS):
        add(value)
    for value in _source_file_terms(bridge_terms):
        if any(hint.replace(" ", "_") in _normalize(value) for hint in _TABLE_HINTS):
            add(value)
    return " ".join(terms[:12])[:500]


def _symbol_request_query(
    primary_terms: list[str],
    contract: QueryContract,
    bridge_terms: list[str],
) -> str:
    terms: list[str] = []
    add = _term_adder(terms)
    for value in primary_terms:
        add(value)
    for value in _code_terms(bridge_terms):
        add(value)
    for value in contract.keyword_anchors:
        if value not in _QUERY_TERM_STOPWORDS:
            add(value)
    for value in contract.identifiers:
        add(value)
    return " ".join(terms[:14])[:500]


def _section_request_query(
    primary_terms: list[str],
    contract: QueryContract,
    bridge_terms: list[str],
) -> str:
    terms: list[str] = []
    add = _term_adder(terms)
    document_primary = _document_bridge_terms(primary_terms)
    if document_primary:
        for value in document_primary:
            add(value)
        return " ".join(terms[:8])[:500]

    for value in primary_terms:
        add(value)
    for value in contract.keyword_anchors:
        add(value)
    for value in _acronym_terms(bridge_terms):
        add(value)
    for value in _document_terms(bridge_terms):
        add(value)
    for value in contract.identifiers:
        add(value)
    return " ".join(terms[:14])[:500]


def _term_adder(terms: list[str]):
    seen: set[str] = set()

    def add(value: Any) -> None:
        text = str(value).strip().strip(".,;:()[]{}")
        normalized = _normalize(text)
        if (
            not text
            or "\n" in text
            or normalized in seen
            or normalized in _GENERATED_BRIDGE_TERMS
            or _UUID_PATTERN.fullmatch(normalized)
        ):
            return
        seen.add(normalized)
        terms.append(text)

    return add


def _identifier_terms(terms: list[str]) -> list[str]:
    values: list[str] = []
    for term in terms:
        if _ACRONYM_PATTERN.fullmatch(term) or (
            _BRIDGE_IDENTIFIER_PATTERN.fullmatch(term) and any(char.isdigit() for char in term)
        ):
            values.append(term)
    return values


def _hint_terms(terms: list[str], hints: set[str]) -> list[str]:
    normalized_hints = {_normalize(hint) for hint in hints}
    return [term for term in terms if _normalize(term) in normalized_hints]


def _source_file_terms(terms: list[str]) -> list[str]:
    return [term for term in terms if "." in term or "_" in term]


def _code_terms(terms: list[str]) -> list[str]:
    values: list[str] = []
    for term in terms:
        normalized = _normalize(term)
        if term.endswith(".py") or normalized in {"code", "function", "method"}:
            values.append(term)
    return values


def _document_terms(terms: list[str]) -> list[str]:
    values: list[str] = []
    for term in terms:
        normalized = _normalize(term)
        if normalized in {_normalize(item) for item in _DOCUMENT_HINTS}:
            values.append(term)
        elif any(hint in normalized.split() for hint in _DOCUMENT_HINTS):
            values.append(term)
    return values


def _acronym_terms(terms: list[str]) -> list[str]:
    return [term for term in terms if _ACRONYM_PATTERN.fullmatch(term)]


def _structured_bridge_identifiers(terms: list[str]) -> list[str]:
    values: list[str] = []
    for term in terms:
        if not _BRIDGE_IDENTIFIER_PATTERN.fullmatch(term):
            continue
        if not any(char.isdigit() for char in term):
            continue
        values.append(term)
    return list(dict.fromkeys(values))


def _document_bridge_terms(terms: list[str]) -> list[str]:
    values: list[str] = []
    for term in terms:
        normalized = _normalize_document_bridge(term)
        words = normalized.split()
        if len(words) < 2:
            continue
        if not any(hint in words for hint in _DOCUMENT_HINTS):
            continue
        if normalized.endswith(".md") or normalized.endswith(".py"):
            continue
        if normalized in {"deterministic table", "query grounded row filter"}:
            continue
        document_end = next(
            (index for index, word in enumerate(words) if word in _DOCUMENT_HINTS),
            len(words) - 1,
        )
        values.append(" ".join(words[: document_end + 1]))
    return list(dict.fromkeys(values))


def _normalize_document_bridge(term: str) -> str:
    normalized = _normalize(term)
    words = normalized.split()
    while words and words[0] in {"and", "from", "see", "the", "use", "using"}:
        words = words[1:]
    return " ".join(words)


def _result_text(result: ReadResult) -> str:
    """Return only evidence-bearing source text for bridge discovery."""
    return str(result.content)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9_/-]+", " ", value.lower())).strip()


__all__ = [
    "EvidenceClosurePlan",
    "EvidenceClosureRequest",
    "annotate_closure_result",
    "plan_evidence_closure",
    "request_metadata",
]
