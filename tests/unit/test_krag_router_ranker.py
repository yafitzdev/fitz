# tests/unit/test_krag_router_ranker.py
"""
Unit tests for RetrievalRouter and CrossStrategyRanker.

Tests routing logic (strategy dispatch, deduplication, fallback) and
cross-strategy ranking (weight application, entity bonus, ordering).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fitz_sage.engines.fitz_krag.query_planner import DeterministicQueryPlanner
from fitz_sage.engines.fitz_krag.retrieval.ranker import (
    ENTITY_MATCH_BONUS,
    CrossStrategyRanker,
)
from fitz_sage.engines.fitz_krag.retrieval.router import RetrievalRouter
from fitz_sage.engines.fitz_krag.retrieval_profile import RetrievalProfile
from fitz_sage.engines.fitz_krag.types import Address, AddressKind

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(top_addresses: int = 10) -> MagicMock:
    """Create a mock FitzKragConfig with the fields the router reads."""
    cfg = MagicMock()
    cfg.top_addresses = top_addresses
    cfg.retrieval_workers = 4
    return cfg


def _addr(
    kind: AddressKind = AddressKind.SYMBOL,
    source_id: str = "src",
    location: str = "mod.func",
    summary: str = "does something",
    score: float = 0.5,
    metadata: dict | None = None,
) -> Address:
    """Shortcut to build an Address."""
    return Address(
        kind=kind,
        source_id=source_id,
        location=location,
        summary=summary,
        score=score,
        metadata=metadata or {},
    )


def _code_profile(
    entities: tuple[str, ...] = (),
    top_k: int = 10,
) -> RetrievalProfile:
    """RetrievalProfile with CODE-like weights."""
    return RetrievalProfile(
        strategy_weights={"code": 0.83, "section": 0.11, "table": 0.06},
        entities=entities,
        top_k=top_k,
        analysis_type="code",
        analysis_confidence=0.9,
    )


def _custom_weight_profile(
    code: float,
    section: float,
    table: float,
    entities: tuple[str, ...] = (),
    top_k: int = 10,
) -> RetrievalProfile:
    """Build a RetrievalProfile with custom strategy weights."""
    return RetrievalProfile(
        strategy_weights={"code": code, "section": section, "table": table},
        entities=entities,
        top_k=top_k,
        analysis_type="general",
        analysis_confidence=0.8,
    )


# ---------------------------------------------------------------------------
# TestRetrievalRouter
# ---------------------------------------------------------------------------


class TestRetrievalRouter:
    """Tests for RetrievalRouter dispatch, fallback, dedup, and ranking."""

    # -- test_retrieve_code_only ------------------------------------------

    def test_retrieve_code_only(self):
        """Code strategy returns addresses; no section or chunk used."""
        code_strat = MagicMock()
        code_addrs = [_addr(score=0.9), _addr(score=0.7, location="mod.bar")]
        code_strat.retrieve.return_value = code_addrs

        config = _make_config(top_addresses=10)
        router = RetrievalRouter(
            code_strategy=code_strat,
            config=config,
            section_strategy=None,
        )

        result = router.retrieve("find func").addresses

        code_strat.retrieve.assert_called_once_with("find func", 10, detection=None)
        assert len(result) == 2
        # Without profile, sorted by score descending
        assert result[0].score >= result[1].score

    def test_retrieve_returns_profile_top_k_candidates(self):
        """Profile top_k should control the final candidate list size."""
        code_strat = MagicMock()
        code_strat.retrieve.return_value = [
            _addr(score=1.0 - (index * 0.01), location=f"mod.func{index}") for index in range(12)
        ]

        config = _make_config(top_addresses=5)
        router = RetrievalRouter(
            code_strategy=code_strat,
            config=config,
            section_strategy=None,
        )

        result = router.retrieve("find func", _code_profile(top_k=8)).addresses

        assert len(result) == 8

    # -- test_retrieve_with_section_strategy ------------------------------

    def test_retrieve_with_section_strategy(self):
        """Both code and section strategies contribute results."""
        code_strat = MagicMock()
        code_strat.retrieve.return_value = [
            _addr(AddressKind.SYMBOL, score=0.8, location="a.py:func"),
        ]
        section_strat = MagicMock()
        section_strat.retrieve.return_value = [
            _addr(AddressKind.SECTION, score=0.7, location="README#setup"),
        ]
        config = _make_config()

        router = RetrievalRouter(
            code_strategy=code_strat,
            config=config,
            section_strategy=section_strat,
        )

        result = router.retrieve("how to setup").addresses

        code_strat.retrieve.assert_called_once()
        section_strat.retrieve.assert_called_once()
        assert len(result) == 2

    # -- test_retrieve_skips_low_weight_strategy --------------------------

    def test_retrieve_skips_low_weight_strategy(self):
        """Strategy with weight <= 0.05 is skipped entirely."""
        code_strat = MagicMock()
        code_strat.retrieve.return_value = [
            _addr(score=0.9, location="a.func"),
        ]
        section_strat = MagicMock()
        section_strat.retrieve.return_value = [
            _addr(AddressKind.SECTION, score=0.6, location="doc#s"),
        ]

        config = _make_config()
        router = RetrievalRouter(
            code_strategy=code_strat,
            config=config,
            section_strategy=section_strat,
        )

        # Custom weights: section weight is 0.04 (below 0.05 threshold)
        profile = _custom_weight_profile(code=0.9, section=0.04, table=0.06)

        result = router.retrieve("find func", profile).addresses

        code_strat.retrieve.assert_called_once()
        section_strat.retrieve.assert_not_called()
        assert len(result) >= 1

    # -- test_retrieve_deduplicates ---------------------------------------

    def test_retrieve_deduplicates(self):
        """Same (source_id, location) from different strategies kept once."""
        code_strat = MagicMock()
        code_strat.retrieve.return_value = [
            _addr(
                AddressKind.SYMBOL,
                source_id="file.py",
                location="MyClass",
                score=0.9,
            ),
        ]
        section_strat = MagicMock()
        section_strat.retrieve.return_value = [
            _addr(
                AddressKind.SECTION,
                source_id="file.py",
                location="MyClass",
                score=0.7,
            ),
        ]

        config = _make_config()
        router = RetrievalRouter(
            code_strategy=code_strat,
            config=config,
            section_strategy=section_strat,
        )

        result = router.retrieve("query").addresses

        # Duplicate by (source_id, location) -- first one wins
        assert len(result) == 1
        assert result[0].score == 0.9

    # -- test_retrieve_with_profile_uses_ranker ---------------------------

    def test_retrieve_with_profile_uses_ranker(self):
        """When profile is provided, CrossStrategyRanker is used."""
        code_strat = MagicMock()
        sym_hi = _addr(AddressKind.SYMBOL, score=0.5, location="low_sym")
        sym_lo = _addr(AddressKind.SYMBOL, score=0.9, location="hi_sym")
        code_strat.retrieve.return_value = [sym_hi, sym_lo]

        config = _make_config()
        router = RetrievalRouter(
            code_strategy=code_strat,
            config=config,
        )

        profile = _code_profile(entities=("hi_sym",))
        result = router.retrieve("find hi_sym", profile).addresses

        # Ranker should apply entity bonus to hi_sym, boosting it
        assert len(result) == 2
        # hi_sym should rank first because of entity match bonus
        assert result[0].location == "hi_sym"

    # -- test_retrieve_without_profile_sorts_by_score --------------------

    def test_retrieve_without_profile_sorts_by_score(self):
        """No profile -> results sorted by raw score descending."""
        code_strat = MagicMock()
        code_strat.retrieve.return_value = [
            _addr(score=0.3, location="low"),
            _addr(score=0.9, location="high"),
            _addr(score=0.6, location="mid"),
        ]

        config = _make_config()
        router = RetrievalRouter(
            code_strategy=code_strat,
            config=config,
        )

        result = router.retrieve("query").addresses

        assert [a.location for a in result] == ["high", "mid", "low"]

    # -- test_retrieve_limits_results -------------------------------------

    def test_retrieve_limits_results(self):
        """More results than top_addresses -> truncated to limit."""
        code_strat = MagicMock()
        code_strat.retrieve.return_value = [
            _addr(score=1.0 - i * 0.05, location=f"f{i}") for i in range(15)
        ]

        config = _make_config(top_addresses=5)
        router = RetrievalRouter(
            code_strategy=code_strat,
            config=config,
        )

        result = router.retrieve("query").addresses

        assert len(result) == 5
        # Should be top-5 by score
        assert result[0].score == 1.0

    def test_retrieve_preserves_candidates_from_each_strategy(self):
        """A dominant section leg must not push every table candidate past the read limit."""
        code_strat = MagicMock()
        code_strat.retrieve.return_value = []
        section_strat = MagicMock()
        section_strat.retrieve.return_value = [
            _addr(
                AddressKind.SECTION,
                source_id=f"doc-{index}",
                location=f"section-{index}",
                score=1.0 - index * 0.01,
            )
            for index in range(10)
        ]
        table_strat = MagicMock()
        table_strat.retrieve.return_value = [
            _addr(
                AddressKind.TABLE,
                source_id="matrix.csv",
                location=f"table-{index}",
                score=0.1 - index * 0.01,
            )
            for index in range(2)
        ]
        router = RetrievalRouter(
            code_strategy=code_strat,
            config=_make_config(top_addresses=6),
            section_strategy=section_strat,
            table_strategy=table_strat,
        )
        profile = _custom_weight_profile(
            code=0.1,
            section=0.8,
            table=0.1,
            top_k=6,
        )

        result = router.retrieve("rollout status", profile).addresses

        assert len(result) == 6
        assert sum(address.kind == AddressKind.TABLE for address in result) == 2
        assert sum(address.kind == AddressKind.SECTION for address in result) == 4

    def test_strategy_coverage_does_not_invent_missing_modalities(self):
        """Coverage only applies to strategies that returned real candidates."""
        addresses = [
            _addr(
                AddressKind.SECTION,
                source_id=f"doc-{index}",
                location=f"section-{index}",
                score=1.0 - index * 0.01,
            )
            for index in range(10)
        ]

        result = RetrievalRouter._enforce_strategy_coverage(addresses, 5)

        assert result == addresses[:5]

    def test_compound_query_preserves_one_candidate_per_successful_leg(self):
        """Strong original-query hits must not erase explicit sub-question recall."""
        query = "What is the refund window, and who approves exceptions?"
        plan = DeterministicQueryPlanner().plan(query)
        assert plan.rewrite_result is not None
        refund_query, approver_query = plan.rewrite_result.decomposed_queries

        code_strat = MagicMock()

        def retrieve(strategy_query, _limit, detection=None):
            del detection
            if strategy_query == refund_query:
                return [_addr(source_id="refund", location="refund", score=0.10)]
            if strategy_query == approver_query:
                return [_addr(source_id="approver", location="approver", score=0.09)]
            return [
                _addr(source_id="noise-a", location="noise-a", score=0.99),
                _addr(source_id="noise-b", location="noise-b", score=0.98),
            ]

        code_strat.retrieve.side_effect = retrieve
        router = RetrievalRouter(
            code_strategy=code_strat,
            config=_make_config(top_addresses=2),
        )

        result = router.retrieve(
            query,
            _code_profile(top_k=2),
            rewrite_result=plan.rewrite_result,
        ).addresses

        assert len(result) == 2
        assert {address.source_id for address in result} == {"refund", "approver"}
        assert {call.args[0] for call in code_strat.retrieve.call_args_list} >= {
            query,
            refund_query,
            approver_query,
        }
        assert {tuple(address.metadata["retrieval_queries"]) for address in result} == {
            (refund_query,),
            (approver_query,),
        }


# ---------------------------------------------------------------------------
# TestCrossStrategyRanker
# ---------------------------------------------------------------------------


class TestCrossStrategyRanker:
    """Tests for CrossStrategyRanker scoring and ordering."""

    def setup_method(self):
        self.ranker = CrossStrategyRanker()

    # -- test_rank_applies_weights ----------------------------------------

    def test_rank_applies_weights(self):
        """CODE profile boosts SYMBOL addresses via higher weight."""
        sym_addr = _addr(AddressKind.SYMBOL, score=0.5, location="func")
        section_addr = _addr(AddressKind.SECTION, score=0.5, location="section1")

        profile = _code_profile()
        # CODE weights: code=0.8, section=0.1
        # sym: 0.5 * 0.8 = 0.40, section: 0.5 * 0.1 = 0.05
        result = self.ranker.rank([section_addr, sym_addr], profile)

        assert result[0].kind == AddressKind.SYMBOL
        assert result[1].kind == AddressKind.SECTION

    # -- test_rank_entity_match_bonus_in_location -------------------------

    def test_rank_entity_match_bonus(self):
        """Entity present in location earns ENTITY_MATCH_BONUS."""
        addr_match = _addr(
            AddressKind.SYMBOL,
            score=0.5,
            location="MyClass.do_work",
        )
        addr_no = _addr(
            AddressKind.SYMBOL,
            score=0.5,
            location="other_func",
        )

        profile = _code_profile(entities=("MyClass",))
        result = self.ranker.rank([addr_no, addr_match], profile)

        # addr_match gets bonus, should rank higher
        assert result[0].location == "MyClass.do_work"

        # Verify the bonus magnitude: both have same base * weight,
        # but match gets +ENTITY_MATCH_BONUS
        weights = profile.strategy_weights
        expected_match = 0.5 * weights["code"] + ENTITY_MATCH_BONUS
        expected_no = 0.5 * weights["code"]
        assert expected_match > expected_no

    # -- test_rank_entity_in_summary --------------------------------------

    def test_rank_entity_in_summary(self):
        """Entity present in summary (not location) also earns bonus."""
        addr = _addr(
            AddressKind.SYMBOL,
            score=0.5,
            location="some_func",
            summary="Handles MyClass initialization",
        )
        addr_no = _addr(
            AddressKind.SYMBOL,
            score=0.5,
            location="other_func",
            summary="unrelated work",
        )

        profile = _code_profile(entities=("MyClass",))
        result = self.ranker.rank([addr_no, addr], profile)

        assert result[0].summary == "Handles MyClass initialization"

    # -- test_rank_no_entity_match ----------------------------------------

    def test_rank_no_entity_match(self):
        """No entity match -> no bonus applied."""
        addr1 = _addr(AddressKind.SYMBOL, score=0.8, location="func_a")
        addr2 = _addr(AddressKind.SYMBOL, score=0.6, location="func_b")

        profile = _code_profile(entities=("NonExistent",))
        result = self.ranker.rank([addr2, addr1], profile)

        # Neither gets bonus; order by weighted score alone
        assert result[0].location == "func_a"
        assert result[1].location == "func_b"

    # -- test_rank_sorts_descending ---------------------------------------

    def test_rank_sorts_descending(self):
        """Results are sorted by computed score, highest first."""
        addrs = [
            _addr(AddressKind.SYMBOL, score=0.3, location="low"),
            _addr(AddressKind.SYMBOL, score=0.9, location="high"),
            _addr(AddressKind.SYMBOL, score=0.6, location="mid"),
        ]

        profile = _code_profile()
        result = self.ranker.rank(addrs, profile)

        scores = [a.score for a in result]
        # Original scores should be in descending order (all same weight)
        assert scores == sorted(scores, reverse=True)

    # -- test_rank_empty_addresses ----------------------------------------

    def test_rank_empty_addresses(self):
        """Empty address list returns empty list."""
        profile = _code_profile()
        result = self.ranker.rank([], profile)
        assert result == []

    # -- test_rank_without_profile ----------------------------------------

    def test_rank_without_profile(self):
        """When profile is None, rank by raw score only."""
        addrs = [
            _addr(AddressKind.SYMBOL, score=0.3, location="low"),
            _addr(AddressKind.SYMBOL, score=0.9, location="high"),
        ]
        result = self.ranker.rank(addrs, None)

        assert result[0].location == "high"
        assert result[1].location == "low"
