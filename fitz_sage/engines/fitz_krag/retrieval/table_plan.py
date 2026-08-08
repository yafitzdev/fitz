# fitz_sage/engines/fitz_krag/retrieval/table_plan.py
"""Typed table query planning for deterministic evidence grounding."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from fitz_sage.core.identifiers import EXACT_IDENTIFIER_PATTERN as _IDENTIFIER_PATTERN
from fitz_sage.core.identifiers import (
    contains_exact_identifier,
    exact_identifiers,
)
from fitz_sage.tabular.value_semantics import normalize_table_value as _normalize
from fitz_sage.tabular.value_semantics import sortable_table_value as _sortable_value

_MAX_SUPERLATIVES = {"highest", "largest", "latest", "longest", "max", "maximum", "most", "newest"}
_MIN_SUPERLATIVES = {"earliest", "fastest", "least", "lowest", "min", "minimum", "shortest"}
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "using",
    "was",
    "were",
    "what",
    "which",
    "who",
    "with",
}
_BOOLEAN_TRUE = {"1", "active", "enabled", "true", "yes", "on"}
_BOOLEAN_FALSE = {"0", "disabled", "false", "inactive", "no", "off"}
_BOOLEAN_NEGATORS = {"no", "not", "without"}
_MIN_UNPREFIXED_ROOT_LENGTH = 4
_NEGATIVE_QUERY_TERMS = {
    "disabled",
    "false",
    "inactive",
    "no",
    "not",
    "off",
    "without",
}
_POSITIVE_QUERY_TERMS = {
    "active",
    "enabled",
    "encrypted",
    "managed",
    "on",
    "true",
    "visible",
    "yes",
}


class ColumnRole(str, Enum):
    """Semantic role assigned to a table column."""

    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    IDENTIFIER = "identifier"
    METRIC = "metric"


@dataclass(frozen=True)
class ColumnBinding:
    """Schema-level description of one table column."""

    index: int
    name: str
    role: ColumnRole
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class RowPredicate:
    """A typed row predicate derived from query intent and table schema."""

    column: ColumnBinding
    accepted_values: frozenset[str]
    source: str


@dataclass(frozen=True)
class SortClause:
    """A typed ordering clause for a table query."""

    column: ColumnBinding
    direction: str


@dataclass(frozen=True)
class TableQueryPlan:
    """Executable deterministic table plan."""

    columns: tuple[ColumnBinding, ...]
    identifiers: tuple[str, ...] = ()
    predicates: tuple[RowPredicate, ...] = ()
    sort: SortClause | None = None
    query_terms: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def build_table_query_plan(
    query: str,
    columns: list[str],
    rows: list[list[Any]],
) -> TableQueryPlan:
    """Build a typed table query plan from query text and concrete table schema."""
    query_terms = tuple(_query_terms(query))
    column_bindings = tuple(
        _bind_column(index, column, rows) for index, column in enumerate(columns)
    )
    identifiers = tuple(_query_identifiers(query))
    predicates = tuple(_row_predicates(query, query_terms, column_bindings, rows))
    sort = _sort_clause(query, query_terms, column_bindings, rows)
    return TableQueryPlan(
        columns=column_bindings,
        identifiers=identifiers,
        predicates=predicates,
        sort=sort,
        query_terms=query_terms,
        metadata={
            "columns": [
                {
                    "name": column.name,
                    "role": column.role.value,
                    "tokens": list(column.tokens),
                }
                for column in column_bindings
            ],
            "identifiers": list(identifiers),
            "predicates": [
                {
                    "column": predicate.column.name,
                    "accepted_values": sorted(predicate.accepted_values),
                    "source": predicate.source,
                }
                for predicate in predicates
            ],
            "sort": (
                {"column": sort.column.name, "direction": sort.direction}
                if sort is not None
                else None
            ),
        },
    )


def execute_table_query_plan(plan: TableQueryPlan, rows: list[list[Any]]) -> list[list[Any]]:
    """Execute a typed table plan against a bounded row scan."""
    if not rows:
        return []

    candidate_rows = list(rows)
    if plan.identifiers:
        candidate_rows = [
            row for row in candidate_rows if _row_matches_identifier(row, plan.identifiers)
        ]
    elif plan.predicates:
        candidate_rows = [
            row for row in candidate_rows if _row_satisfies_predicates(row, plan.predicates)
        ]
    if not candidate_rows:
        return []

    if plan.sort is not None:
        sortable = [
            (value, row)
            for row in candidate_rows
            if (value := _sortable_value(row[plan.sort.column.index])) is not None
        ]
        if sortable:
            return [
                sorted(
                    sortable,
                    key=lambda item: item[0],
                    reverse=plan.sort.direction == "max",
                )[
                    0
                ][1]
            ]

    scored = [(_row_score(plan, row), index, row) for index, row in enumerate(candidate_rows)]
    scored = [item for item in scored if item[0] > 0]
    if not scored:
        return []
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [row for _, _, row in scored]


def _bind_column(index: int, column: str, rows: list[list[Any]]) -> ColumnBinding:
    tokens = tuple(_tokenize(column))
    role = _column_role(column, tokens, [row[index] for row in rows if index < len(row)])
    return ColumnBinding(index=index, name=column, role=role, tokens=tokens)


def _column_role(column: str, tokens: tuple[str, ...], values: list[Any]) -> ColumnRole:
    normalized = _normalize(column)
    nonempty = [str(value).strip() for value in values if str(value).strip()]
    normalized_values = {_normalize(value) for value in nonempty}
    if normalized.endswith(" id") or normalized == "id" or normalized.endswith(" key"):
        return ColumnRole.IDENTIFIER
    if nonempty and sum(1 for value in nonempty if _IDENTIFIER_PATTERN.fullmatch(value)) >= max(
        1, int(len(nonempty) * 0.6)
    ):
        return ColumnRole.IDENTIFIER
    if normalized_values and normalized_values <= (_BOOLEAN_TRUE | _BOOLEAN_FALSE):
        return ColumnRole.BOOLEAN
    numeric_values = sum(1 for value in nonempty if _sortable_value(value) is not None)
    if nonempty and numeric_values >= max(1, int(len(nonempty) * 0.6)):
        return ColumnRole.METRIC
    return ColumnRole.CATEGORICAL


def _row_predicates(
    query: str,
    query_terms: tuple[str, ...],
    columns: tuple[ColumnBinding, ...],
    rows: list[list[Any]],
) -> list[RowPredicate]:
    predicates: list[RowPredicate] = []
    query_text = _normalize(query)
    for column in columns:
        if column.role is ColumnRole.BOOLEAN:
            accepted = _boolean_predicate_values(column, query_text)
            if accepted:
                predicates.append(
                    RowPredicate(
                        column=column,
                        accepted_values=frozenset(accepted),
                        source="boolean_column",
                    )
                )
            continue
        if column.role is not ColumnRole.CATEGORICAL:
            continue
        accepted_values = {
            _normalize(str(row[column.index]))
            for row in rows
            if column.index < len(row) and _normalize(str(row[column.index])) in set(query_terms)
        }
        if accepted_values:
            predicates.append(
                RowPredicate(
                    column=column,
                    accepted_values=frozenset(accepted_values),
                    source="categorical_value",
                )
            )
    return predicates


def _boolean_predicate_values(column: ColumnBinding, query_text: str) -> set[str]:
    polarity_terms = _NEGATIVE_QUERY_TERMS | _POSITIVE_QUERY_TERMS
    column_terms = {
        term for term in column.tokens if term not in _STOPWORDS or term in polarity_terms
    }
    query_terms = tuple(
        term for term in query_text.split() if term not in _STOPWORDS or term in polarity_terms
    )
    reference_indices = [
        index
        for index, term in enumerate(query_terms)
        if term in column_terms or _is_unprefixed_column_reference(term, column_terms)
    ]
    if not reference_indices:
        return set()

    if any(
        _is_unprefixed_column_reference(query_terms[index], column_terms)
        for index in reference_indices
    ):
        return _BOOLEAN_FALSE
    if any(
        index > 0 and query_terms[index - 1] in _BOOLEAN_NEGATORS for index in reference_indices
    ):
        return _BOOLEAN_FALSE

    direct_state_terms = {
        query_terms[index] for index in reference_indices if query_terms[index] in polarity_terms
    }
    if direct_state_terms:
        return _BOOLEAN_TRUE

    neighboring_terms = {
        query_terms[neighbor]
        for index in reference_indices
        for neighbor in (index - 1, index + 1)
        if 0 <= neighbor < len(query_terms)
    }
    if neighboring_terms & (_NEGATIVE_QUERY_TERMS - _BOOLEAN_NEGATORS):
        return _BOOLEAN_FALSE
    if neighboring_terms & _POSITIVE_QUERY_TERMS:
        return _BOOLEAN_TRUE
    return set()


def _is_unprefixed_column_reference(query_term: str, column_terms: set[str]) -> bool:
    if not query_term.startswith("un"):
        return False
    root = query_term.removeprefix("un")
    return len(root) >= _MIN_UNPREFIXED_ROOT_LENGTH and root in column_terms


def _sort_clause(
    query: str,
    query_terms: tuple[str, ...],
    columns: tuple[ColumnBinding, ...],
    rows: list[list[Any]],
) -> SortClause | None:
    direction = _superlative_direction(query)
    if direction is None:
        return None
    candidates: list[tuple[int, int, ColumnBinding]] = []
    for column in columns:
        if column.role is not ColumnRole.METRIC:
            continue
        if not any(
            _sortable_value(row[column.index]) is not None
            for row in rows
            if column.index < len(row)
        ):
            continue
        score = _column_query_score(column, query_terms)
        if score <= 0:
            continue
        candidates.append((score, -column.index, column))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], -item[1]))
    return SortClause(column=candidates[0][2], direction=direction)


def _column_query_score(column: ColumnBinding, query_terms: tuple[str, ...]) -> int:
    score = 0
    term_set = set(query_terms)
    for token in column.tokens:
        if token in term_set:
            score += 4
    joined = " ".join(column.tokens)
    for phrase in ("date", "time", "minute", "minutes", "day", "days", "rate", "percent", "score"):
        if phrase in joined and phrase in term_set:
            score += 2
    return score


def _row_score(plan: TableQueryPlan, row: list[Any]) -> int:
    value_text = _normalize(" ".join(str(value) for value in row if value is not None))
    pair_text = _normalize(
        " ".join(
            f"{column.name} {row[column.index]}"
            for column in plan.columns
            if column.index < len(row)
        )
    )
    score = 0
    for identifier in plan.identifiers:
        if _contains_identifier(value_text, identifier):
            score += 20
    for term in plan.query_terms:
        if _contains_term(value_text, term):
            score += 4
        elif _contains_term(pair_text, term):
            score += 2
    if plan.predicates and _row_satisfies_predicates(row, plan.predicates):
        score += 8
    return score


def _row_matches_identifier(row: list[Any], identifiers: tuple[str, ...]) -> bool:
    value_text = " ".join(str(value) for value in row if value is not None)
    return any(_contains_identifier(value_text, identifier) for identifier in identifiers)


def _row_satisfies_predicates(row: list[Any], predicates: tuple[RowPredicate, ...]) -> bool:
    for predicate in predicates:
        if predicate.column.index >= len(row):
            return False
        value = _normalize(str(row[predicate.column.index]))
        if value not in predicate.accepted_values:
            return False
    return True


def _query_identifiers(query: str) -> list[str]:
    return exact_identifiers(query)


def _query_terms(query: str) -> list[str]:
    terms = [
        term
        for term in _tokenize(query)
        if len(term) >= 3
        and term not in _STOPWORDS
        and term not in _MAX_SUPERLATIVES
        and term not in _MIN_SUPERLATIVES
    ]
    return list(dict.fromkeys(terms))


def _superlative_direction(query: str) -> str | None:
    terms = set(_tokenize(query))
    if terms & _MAX_SUPERLATIVES:
        return "max"
    if terms & _MIN_SUPERLATIVES:
        return "min"
    return None


def _contains_identifier(text: str, identifier: str) -> bool:
    return contains_exact_identifier(text, identifier)


def _contains_term(text: str, term: str) -> bool:
    normalized_term = _normalize(term)
    if not normalized_term:
        return False
    variants = {normalized_term}
    if normalized_term.endswith("s") and len(normalized_term) > 3:
        variants.add(normalized_term[:-1])
    else:
        variants.add(f"{normalized_term}s")
    return any(re.search(rf"\b{re.escape(variant)}\b", text) for variant in variants)


def _tokenize(value: str) -> list[str]:
    return _normalize(value).split()


__all__ = [
    "ColumnBinding",
    "ColumnRole",
    "RowPredicate",
    "SortClause",
    "TableQueryPlan",
    "build_table_query_plan",
    "execute_table_query_plan",
]
