# tests/e2e_krag/scenarios.py
"""
E2E test scenario definitions.

Each scenario tests a specific retrieval intelligence feature with a query
and expected outcome validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class Feature(Enum):
    """Retrieval intelligence features under test."""

    MULTI_HOP = "multi_hop"
    ENTITY_GRAPH = "entity_graph"
    COMPARISON = "comparison"
    MULTI_QUERY = "multi_query"
    KEYWORD_EXACT = "keyword_exact"
    CONFLICT_AWARE = "conflict_aware"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CAUSAL_ATTRIBUTION = "causal_attribution"
    TABLE_SCHEMA = "table_schema"
    TABLE_QUERY = "table_query"
    CODE_SEARCH = "code_search"
    LONG_DOC = "long_doc"
    DEDUP = "dedup"
    BASIC_RETRIEVAL = "basic_retrieval"
    FRESHNESS = "freshness"
    SPARSE_SEARCH = "sparse_search"
    QUERY_EXPANSION = "query_expansion"
    TEMPORAL = "temporal"
    AGGREGATION = "aggregation"
    FIGURE_RETRIEVAL = "figure_retrieval"
    CONCEPTUAL_RETRIEVAL = "conceptual_retrieval"
    QUERY_REWRITING = "query_rewriting"
    PPTX_RETRIEVAL = "pptx_retrieval"
    XLSX_QUERY = "xlsx_query"
    SQL_SEARCH = "sql_search"


@dataclass
class TestScenario:
    """
    A single E2E test scenario.

    Attributes:
        id: Unique scenario identifier (e.g., "E01")
        name: Human-readable scenario name
        feature: The retrieval feature being tested
        query: The query to run against the RAG pipeline
        must_contain: Substrings that MUST appear in the answer (case-insensitive)
        must_contain_any: At least ONE of these must appear (case-insensitive)
        must_not_contain: Substrings that must NOT appear in the answer
        expected_mode: Expected AnswerMode (e.g., "HEDGED", "INSUFFICIENT")
        min_sources: Minimum number of source citations expected
        custom_validator: Optional custom validation function
    """

    id: str
    name: str
    feature: Feature
    query: str
    must_contain: list[str] = field(default_factory=list)
    must_contain_any: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    expected_mode: Optional[str] = None
    min_sources: int = 0
    custom_validator: Optional[Callable] = None


# =============================================================================
# Test Scenarios
# =============================================================================

SCENARIOS: list[TestScenario] = [
    # =========================================================================
    # Multi-Hop Retrieval
    # =========================================================================
    TestScenario(
        id="E01",
        name="Multi-hop: competitor product through person",
        feature=Feature.MULTI_HOP,
        query="What does Sarah Chen's company's main competitor manufacture?",
        must_contain_any=["hydrogen", "fuel cell"],
        min_sources=1,
    ),
    TestScenario(
        id="E02",
        name="Multi-hop: competitor CEO",
        feature=Feature.MULTI_HOP,
        query="Who is the CEO of TechCorp's main competitor?",
        must_contain=["Marcus Webb"],
        min_sources=1,
    ),
    # =========================================================================
    # Entity Graph Expansion
    # =========================================================================
    TestScenario(
        id="E03",
        name="Entity graph: company info includes related entities",
        feature=Feature.ENTITY_GRAPH,
        query="Tell me about TechCorp Industries",
        must_contain_any=["Sarah Chen", "electric vehicle", "GreenDrive"],
        min_sources=1,
    ),
    TestScenario(
        id="E04",
        name="Entity graph: person to company connection",
        feature=Feature.ENTITY_GRAPH,
        query="What company does Sarah Chen work for and what do they make?",
        must_contain=["TechCorp"],
        must_contain_any=["electric", "vehicle", "battery"],
        min_sources=1,
    ),
    # =========================================================================
    # Comparison Detection
    # =========================================================================
    TestScenario(
        id="E05",
        name="Comparison: two products",
        feature=Feature.COMPARISON,
        query="Compare the Model X100 vs Model Y200",
        must_contain=["X100", "Y200"],
        must_contain_any=["price", "range", "battery"],
        min_sources=1,
    ),
    TestScenario(
        id="E06",
        name="Comparison: price difference",
        feature=Feature.COMPARISON,
        query="What is the price difference between Model X100 and Model Z50?",
        must_contain_any=["45,000", "35,000", "10,000", "$10"],
        min_sources=1,
    ),
    # =========================================================================
    # Multi-Query Expansion
    # =========================================================================
    TestScenario(
        id="E07",
        name="Multi-query: complex question about multiple attributes",
        feature=Feature.MULTI_QUERY,
        query="What are the battery capacities and ranges for all TechCorp vehicle models?",
        must_contain_any=["75 kWh", "100 kWh", "50 kWh"],
        min_sources=1,
    ),
    # =========================================================================
    # Conflict Detection
    # =========================================================================
    TestScenario(
        id="E08",
        name="Conflict: employee count discrepancy",
        feature=Feature.CONFLICT_AWARE,
        query="How many employees does the company have?",
        # Should mention the discrepancy between different reports
        must_contain_any=["5,200", "4,800", "5,500", "differ", "conflict", "discrepancy", "varies"],
        min_sources=1,
    ),
    TestScenario(
        id="E09",
        name="Conflict: revenue figures",
        feature=Feature.CONFLICT_AWARE,
        query="What was the Q1 2024 revenue?",
        must_contain_any=["1.2 billion", "1.4 billion", "differ", "discrepancy"],
        min_sources=1,
    ),
    # =========================================================================
    # Insufficient Evidence
    # =========================================================================
    TestScenario(
        id="E10",
        name="Insufficient: missing budget information",
        feature=Feature.INSUFFICIENT_EVIDENCE,
        query="What is Project Alpha's budget?",
        # Should indicate information is not available
        must_contain_any=[
            "don't know",
            "not available",
            "not provided",
            "not found",
            "no information",
            "don't have",
            "classified",
            "restricted",
            "confidential",
            "not contain",  # Matches "do not contain" and "does not contain"
        ],
        min_sources=0,
    ),
    TestScenario(
        id="E11",
        name="Insufficient: missing timeline",
        feature=Feature.INSUFFICIENT_EVIDENCE,
        query="When will Project Alpha be completed?",
        must_contain_any=[
            "don't know",
            "not available",
            "not provided",
            "not found",
            "no information",
            "don't have",
            "classified",
            "restricted",
            "do not contain",
            "cannot answer",
            "unable to",
        ],
        min_sources=0,
    ),
    # =========================================================================
    # Causal Attribution
    # =========================================================================
    TestScenario(
        id="E12",
        name="Causal: stock price rise",
        feature=Feature.CAUSAL_ATTRIBUTION,
        query="Why did TechCorp's stock price rise 20% in Q1 2024?",
        # Should mention multiple factors, not claim single cause
        must_contain_any=[
            "product launch",
            "competitor",
            "tax credit",
            "analyst",
            "multiple",
            "factors",
            "several",
        ],
        min_sources=1,
    ),
    # =========================================================================
    # Table Schema / CSV Queries
    # =========================================================================
    TestScenario(
        id="E13",
        name="Table: column discovery",
        feature=Feature.TABLE_SCHEMA,
        query="What columns are in the employee data?",
        must_contain_any=["employee_id", "name", "department", "salary"],
        min_sources=1,
    ),
    TestScenario(
        id="E14",
        name="Table: department query",
        feature=Feature.TABLE_QUERY,
        query="How many employees are in the Engineering department?",
        must_contain_any=["5", "five", "Engineering"],
        min_sources=1,
    ),
    TestScenario(
        id="E15",
        name="Table: salary query",
        feature=Feature.TABLE_QUERY,
        query="What is the average salary in the Engineering department?",
        # Average of 95000, 105000, 98000, 110000, 88000 = 99200
        must_contain_any=["99", "100", "average"],
        min_sources=1,
    ),
    # =========================================================================
    # Code Search
    # =========================================================================
    TestScenario(
        id="E16",
        name="Code: class description",
        feature=Feature.CODE_SEARCH,
        query="What does the UserAuth class do?",
        must_contain_any=["authentication", "session", "login"],
        min_sources=1,
    ),
    TestScenario(
        id="E17",
        name="Code: method lookup",
        feature=Feature.CODE_SEARCH,
        query="How does the login method work in UserAuth?",
        must_contain_any=["authenticate", "password", "token", "session"],
        min_sources=1,
    ),
    # =========================================================================
    # Long Document Retrieval
    # =========================================================================
    TestScenario(
        id="E18",
        name="Long doc: authentication system",
        feature=Feature.LONG_DOC,
        query="How does the TechCorp authentication system work?",
        must_contain_any=["JWT", "token", "24-hour", "OAuth"],
        min_sources=1,
    ),
    TestScenario(
        id="E19",
        name="Long doc: database architecture",
        feature=Feature.LONG_DOC,
        query="How does the Data Service handle database connections?",
        must_contain_any=["PostgreSQL", "connection pool", "50 connections", "replica"],
        min_sources=1,
    ),
    # =========================================================================
    # Basic Retrieval (sanity checks)
    # =========================================================================
    TestScenario(
        id="E20",
        name="Basic: simple fact lookup",
        feature=Feature.BASIC_RETRIEVAL,
        query="Where is TechCorp Industries headquartered?",
        must_contain=["Austin"],
        min_sources=1,
    ),
    TestScenario(
        id="E21",
        name="Basic: product price",
        feature=Feature.BASIC_RETRIEVAL,
        query="What is the price of the Model Y200?",
        must_contain_any=["55,000", "$55"],
        min_sources=1,
    ),
    # =========================================================================
    # Keyword Exact Match
    # =========================================================================
    TestScenario(
        id="E22",
        name="Keyword: unique company name",
        feature=Feature.KEYWORD_EXACT,
        query="What is GreenDrive Inc's annual revenue?",
        must_contain_any=["2.3 billion", "$2.3"],
        min_sources=1,
    ),
    TestScenario(
        id="E23",
        name="Keyword: specific product model",
        feature=Feature.KEYWORD_EXACT,
        query="What is the 0-60 time for Model Z50?",
        must_contain_any=["6.8", "seconds"],
        min_sources=1,
    ),
    TestScenario(
        id="E24",
        name="Keyword: unique person name",
        feature=Feature.KEYWORD_EXACT,
        query="Where did Marcus Webb work before GreenDrive?",
        must_contain_any=["Toyota", "fuel cell"],
        min_sources=1,
    ),
    # =========================================================================
    # Deduplication
    # =========================================================================
    TestScenario(
        id="E25",
        name="Dedup: term appearing in multiple docs",
        feature=Feature.DEDUP,
        query="What products does TechCorp make?",
        # TechCorp mentioned in people.md, products.md, conflicts.md - should dedupe
        must_contain_any=["electric vehicle", "battery", "Model"],
        min_sources=1,
    ),
    # =========================================================================
    # Additional Table Queries
    # =========================================================================
    TestScenario(
        id="E26",
        name="Table: location filter",
        feature=Feature.TABLE_QUERY,
        query="Which employees work in the Nevada location?",
        must_contain_any=["Eva Brown", "Nathan Park", "Nevada"],
        min_sources=1,
    ),
    TestScenario(
        id="E27",
        name="Table: highest salary",
        feature=Feature.TABLE_QUERY,
        query="Who has the highest salary in the employee data?",
        # James Wilson at $145,000 or Peter Adams at $140,000
        must_contain_any=["James Wilson", "145,000", "Peter Adams", "140,000"],
        min_sources=1,
    ),
    TestScenario(
        id="E28",
        name="Table: department count",
        feature=Feature.TABLE_QUERY,
        query="How many departments are there in the employee data?",
        # Engineering, Marketing, Sales, HR, Finance = 5
        must_contain_any=["5", "five", "Engineering", "Marketing", "Sales", "HR", "Finance"],
        min_sources=1,
    ),
    # =========================================================================
    # Additional Code Search
    # =========================================================================
    TestScenario(
        id="E29",
        name="Code: exception classes",
        feature=Feature.CODE_SEARCH,
        query="What exceptions are defined in the authentication module?",
        must_contain_any=["AuthenticationError", "SessionExpiredError"],
        min_sources=1,
    ),
    TestScenario(
        id="E30",
        name="Code: RoleAuthorizer permissions",
        feature=Feature.CODE_SEARCH,
        query="What permissions does an admin role have?",
        must_contain_any=["read", "write", "delete", "manage_users"],
        min_sources=1,
    ),
    TestScenario(
        id="E31",
        name="Code: session expiration",
        feature=Feature.CODE_SEARCH,
        query="How long do sessions last in UserAuth by default?",
        must_contain_any=["24", "hour", "session_duration"],
        min_sources=1,
    ),
    # =========================================================================
    # Additional Long Document
    # =========================================================================
    TestScenario(
        id="E32",
        name="Long doc: vehicle telemetry",
        feature=Feature.LONG_DOC,
        query="How does TechCorp process vehicle telemetry data?",
        must_contain_any=["Kafka", "Flink", "TimescaleDB", "500,000"],
        min_sources=1,
    ),
    TestScenario(
        id="E33",
        name="Long doc: payment security",
        feature=Feature.LONG_DOC,
        query="What security measures does the Payment Service use?",
        must_contain_any=["PCI DSS", "tokenized", "Stripe", "3D Secure"],
        min_sources=1,
    ),
    TestScenario(
        id="E34",
        name="Long doc: notification channels",
        feature=Feature.LONG_DOC,
        query="What notification channels does TechCorp support?",
        must_contain_any=["push", "email", "SMS", "Firebase", "SendGrid", "Twilio"],
        min_sources=1,
    ),
    # =========================================================================
    # Additional Multi-Hop
    # =========================================================================
    TestScenario(
        id="E35",
        name="Multi-hop: person's previous employer product",
        feature=Feature.MULTI_HOP,
        query="What kind of vehicles does Sarah Chen's previous employer make?",
        # Sarah Chen → AutoMotors → gasoline-powered vehicles
        must_contain_any=["gasoline", "traditional", "AutoMotors"],
        min_sources=1,
    ),
    TestScenario(
        id="E36",
        name="Multi-hop: competitor headquarters",
        feature=Feature.MULTI_HOP,
        query="Where is TechCorp's main competitor headquartered?",
        # TechCorp → GreenDrive → San Jose
        must_contain=["San Jose"],
        min_sources=1,
    ),
    # =========================================================================
    # Additional Comparison
    # =========================================================================
    TestScenario(
        id="E37",
        name="Comparison: three models",
        feature=Feature.COMPARISON,
        query="Compare the range of all three TechCorp models",
        must_contain_any=["300", "400", "200"],  # miles for X100, Y200, Z50
        min_sources=1,
    ),
    TestScenario(
        id="E38",
        name="Comparison: feature difference",
        feature=Feature.COMPARISON,
        query="What features does Model Y200 have that Model Z50 doesn't?",
        must_contain_any=["self-driving", "solar", "Bang & Olufsen", "heads-up", "17-inch"],
        min_sources=1,
    ),
    # =========================================================================
    # Additional Causal Attribution
    # =========================================================================
    TestScenario(
        id="E39",
        name="Causal: competitor stock decline",
        feature=Feature.CAUSAL_ATTRIBUTION,
        query="Why did GreenDrive's stock decline in Q1 2024?",
        must_contain_any=["recall", "50,000", "cost-cutting", "confidence"],
        min_sources=1,
    ),
    # =========================================================================
    # Additional Basic Retrieval
    # =========================================================================
    TestScenario(
        id="E40",
        name="Basic: founding year",
        feature=Feature.BASIC_RETRIEVAL,
        query="When was TechCorp Industries founded?",
        must_contain=["2015"],
        min_sources=1,
    ),
    TestScenario(
        id="E41",
        name="Basic: battery warranty",
        feature=Feature.BASIC_RETRIEVAL,
        query="What is the warranty on TechCorp batteries?",
        must_contain_any=["8-year", "150,000 mile", "8 year"],
        min_sources=1,
    ),
    TestScenario(
        id="E42",
        name="Basic: Project Alpha team size",
        feature=Feature.BASIC_RETRIEVAL,
        query="How many engineers are on the Project Alpha team?",
        must_contain=["5"],
        min_sources=1,
    ),
    # =========================================================================
    # Edge Cases: Query Robustness
    # =========================================================================
    TestScenario(
        id="E43",
        name="Edge: case insensitivity",
        feature=Feature.BASIC_RETRIEVAL,
        query="WHERE IS TECHCORP INDUSTRIES HEADQUARTERED?",
        must_contain=["Austin"],
        min_sources=1,
    ),
    TestScenario(
        id="E44",
        name="Edge: typo in query",
        feature=Feature.BASIC_RETRIEVAL,
        query="What is the price of the Model Y2OO?",  # O instead of 0
        must_contain_any=["55,000", "$55", "Y200"],
        min_sources=1,
    ),
    TestScenario(
        id="E45",
        name="Edge: very short query",
        feature=Feature.BASIC_RETRIEVAL,
        query="TechCorp CEO",
        must_contain=["Sarah Chen"],
        min_sources=1,
    ),
    TestScenario(
        id="E46",
        name="Edge: question without question mark",
        feature=Feature.BASIC_RETRIEVAL,
        query="Tell me the headquarters location of TechCorp",
        must_contain=["Austin"],
        min_sources=1,
    ),
    # =========================================================================
    # Edge Cases: Negation and Exclusion
    # =========================================================================
    TestScenario(
        id="E47",
        name="Edge: negation query",
        feature=Feature.COMPARISON,
        query="Which TechCorp model does NOT have a solar roof option?",
        # Only Y200 has solar roof, so X100 and Z50 don't
        must_contain_any=["X100", "Z50"],
        min_sources=1,
    ),
    # =========================================================================
    # Edge Cases: Out of Domain
    # =========================================================================
    TestScenario(
        id="E48",
        name="Edge: completely out of domain",
        feature=Feature.INSUFFICIENT_EVIDENCE,
        query="What is the recipe for chocolate cake?",
        must_contain_any=[
            "don't know",
            "no information",
            "not found",
            "cannot",
            "don't have",
            "not available",
            "outside",
            "beyond",
            "unrelated",
            "no recipe",
            "not contain",
            "unable",
            "doesn't",
            "does not",
            "no mention",
            "no data",
        ],
        min_sources=0,
    ),
    TestScenario(
        id="E49",
        name="Edge: plausible but missing info",
        feature=Feature.INSUFFICIENT_EVIDENCE,
        query="What is TechCorp's stock ticker symbol?",
        # TECH is mentioned in causal_claims.md but only as hypothetical
        must_contain_any=[
            "TECH",
            "don't know",
            "not provided",
            "not specified",
            "not found",
        ],
        min_sources=0,
    ),
    # =========================================================================
    # Multi-Query Expansion (long complex queries)
    # =========================================================================
    TestScenario(
        id="E50",
        name="Multi-query: very long complex question",
        feature=Feature.MULTI_QUERY,
        query=(
            "I need a comprehensive analysis of TechCorp's product lineup including "
            "all vehicle models with their prices, ranges, battery capacities, and "
            "key distinguishing features. Also include information about the target "
            "market segment for each model."
        ),
        # Should retrieve info about all 3 models
        must_contain_any=["X100", "Y200", "Z50"],
        min_sources=1,
    ),
    # =========================================================================
    # Temporal Queries
    # =========================================================================
    TestScenario(
        id="E51",
        name="Temporal: specific quarter",
        feature=Feature.BASIC_RETRIEVAL,
        query="What happened in Q1 2024 at TechCorp?",
        must_contain_any=["revenue", "stock", "20%", "record"],
        min_sources=1,
    ),
    TestScenario(
        id="E52",
        name="Temporal: founding dates comparison",
        feature=Feature.COMPARISON,
        query="Which company was founded first, TechCorp or GreenDrive?",
        # GreenDrive 2012, TechCorp 2015
        must_contain_any=["GreenDrive", "2012", "first", "earlier", "before"],
        min_sources=1,
    ),
    # =========================================================================
    # Aggregation Queries
    # =========================================================================
    TestScenario(
        id="E53",
        name="Aggregation: count across sources",
        feature=Feature.MULTI_QUERY,
        query="How many different companies are mentioned in the documents?",
        # TechCorp, GreenDrive, AutoMotors, Toyota (mentioned)
        must_contain_any=["TechCorp", "GreenDrive", "AutoMotors", "3", "4", "three", "four"],
        min_sources=1,
    ),
    TestScenario(
        id="E54",
        name="Aggregation: list all products",
        feature=Feature.AGGREGATION,
        query="List all the TechCorp vehicle models",
        must_contain=["X100"],
        must_contain_any=["Y200", "Z50"],
        min_sources=1,
    ),
    # =========================================================================
    # Cross-Document Synthesis
    # =========================================================================
    TestScenario(
        id="E55",
        name="Cross-doc: info from multiple sources",
        feature=Feature.ENTITY_GRAPH,
        query="What is the complete profile of Sarah Chen including her education and career history?",
        # Info spread across people.md
        must_contain_any=["MIT", "PhD", "AutoMotors", "CEO", "2019"],
        min_sources=1,
    ),
    # =========================================================================
    # Numeric Precision
    # =========================================================================
    TestScenario(
        id="E56",
        name="Numeric: exact number lookup",
        feature=Feature.BASIC_RETRIEVAL,
        query="What is the exact battery capacity of the Model X100 in kWh?",
        must_contain=["75"],
        min_sources=1,
    ),
    TestScenario(
        id="E57",
        name="Numeric: calculation required",
        feature=Feature.COMPARISON,
        query="What is the total battery capacity if you bought one of each TechCorp model?",
        # 75 + 100 + 50 = 225 kWh
        must_contain_any=["225", "75", "100", "50"],
        min_sources=1,
    ),
    # =========================================================================
    # Ambiguous Queries
    # =========================================================================
    TestScenario(
        id="E58",
        name="Ambiguous: which 'company'",
        feature=Feature.BASIC_RETRIEVAL,
        query="When was the company founded?",
        # Ambiguous - could be TechCorp (2015) or GreenDrive (2012)
        must_contain_any=["2015", "2012", "TechCorp", "GreenDrive"],
        min_sources=1,
    ),
    # =========================================================================
    # Code: Edge Cases
    # =========================================================================
    TestScenario(
        id="E59",
        name="Code: method signature",
        feature=Feature.CODE_SEARCH,
        query="What parameters does the register_user method take?",
        must_contain_any=["username", "password", "email", "role"],
        min_sources=1,
    ),
    TestScenario(
        id="E60",
        name="Code: return type",
        feature=Feature.CODE_SEARCH,
        query="What does the login method return?",
        must_contain_any=["token", "str", "string", "session"],
        min_sources=1,
    ),
    # =========================================================================
    # Table: Edge Cases
    # =========================================================================
    TestScenario(
        id="E61",
        name="Table: manager lookup",
        feature=Feature.TABLE_QUERY,
        query="Who is the manager of Alice Wong?",
        # Alice Wong's manager_id is E010, which is James Wilson
        must_contain_any=["James Wilson", "E010", "manager"],
        min_sources=1,
    ),
    TestScenario(
        id="E62",
        name="Table: date-based query",
        feature=Feature.TABLE_QUERY,
        query="Who was hired in 2022?",
        # David Lee (2022-02-20), Nathan Park (2022-06-10)
        must_contain_any=["David Lee", "Nathan Park", "2022"],
        min_sources=1,
    ),
    # =========================================================================
    # Freshness / Authority Boosting
    # =========================================================================
    TestScenario(
        id="E63",
        name="Freshness: official spec keyword",
        feature=Feature.FRESHNESS,
        query="What is the official EPA range for the Model X100?",
        # Official spec says 298 miles (EPA certified), products.md says 300 miles
        must_contain_any=["298", "EPA", "certified", "official"],
        min_sources=1,
    ),
    TestScenario(
        id="E64",
        name="Freshness: spec keyword for warranty",
        feature=Feature.FRESHNESS,
        query="What does the specification say about battery warranty?",
        # Spec document has official warranty terms
        must_contain_any=["8 years", "150,000 miles", "official", "warranty"],
        min_sources=1,
    ),
    TestScenario(
        id="E65",
        name="Freshness: authoritative towing capacity",
        feature=Feature.FRESHNESS,
        query="What is the authoritative maximum towing capacity for Model Y200?",
        # Only in spec document: 5,500 lbs
        must_contain_any=["5,500", "5500", "towing"],
        min_sources=1,
    ),
    TestScenario(
        id="E66",
        name="Freshness: official service requirements",
        feature=Feature.FRESHNESS,
        query="What are the official service requirements for TechCorp vehicles?",
        # Only in spec document
        must_contain_any=["12,500 miles", "12 months", "brake fluid", "service"],
        min_sources=1,
    ),
    # =========================================================================
    # Sparse Search (exact tokens, acronyms, and identifiers)
    # =========================================================================
    TestScenario(
        id="E67",
        name="Sparse: exact model number",
        feature=Feature.SPARSE_SEARCH,
        query="X100 battery capacity",
        # Exact model number should be matched by sparse search
        must_contain_any=["75 kWh", "75kWh", "X100"],
        min_sources=1,
    ),
    TestScenario(
        id="E68",
        name="Sparse: technical identifier",
        feature=Feature.SPARSE_SEARCH,
        query="CCS Combo 1 charging standard",
        # Technical identifier from spec doc
        must_contain_any=["CCS", "DC Fast", "charging"],
        min_sources=1,
    ),
    TestScenario(
        id="E69",
        name="Sparse: specific department name",
        feature=Feature.SPARSE_SEARCH,
        query="employees in Engineering",
        # Department name should be exact matched
        must_contain_any=["Engineering", "employee"],
        min_sources=1,
    ),
    TestScenario(
        id="E70",
        name="Sparse: acronym lookup",
        feature=Feature.SPARSE_SEARCH,
        query="What is SAE Level 2+",
        # Technical acronym from spec
        must_contain_any=["SAE", "Level 2", "autonomous", "driving", "certified"],
        min_sources=1,
    ),
    # =========================================================================
    # Query Expansion (Synonym/Acronym Variations)
    # =========================================================================
    TestScenario(
        id="E71",
        name="Query expansion: synonym for retrieve",
        feature=Feature.QUERY_EXPANSION,
        query="How do I fetch employee data?",
        # "fetch" should expand to "retrieve/get" and find relevant content
        must_contain_any=["employee", "data", "retrieve", "get", "query"],
        min_sources=1,
    ),
    TestScenario(
        id="E72",
        name="Query expansion: synonym for create",
        feature=Feature.QUERY_EXPANSION,
        query="How do I add a new user account?",
        # "add" should expand to "create/register" and find auth module content
        must_contain_any=["register", "create", "user", "account"],
        min_sources=1,
    ),
    TestScenario(
        id="E73",
        name="Query expansion: acronym db to database",
        feature=Feature.QUERY_EXPANSION,
        query="How does the db connection work?",
        # "db" should expand to "database" and find Data Service content
        must_contain_any=["database", "connection", "PostgreSQL", "pool"],
        min_sources=1,
    ),
    TestScenario(
        id="E74",
        name="Query expansion: synonym for error",
        feature=Feature.QUERY_EXPANSION,
        query="What failures can occur in authentication?",
        # "failures" should expand to "errors/exceptions" and find auth content
        must_contain_any=["error", "exception", "authentication", "fail"],
        min_sources=1,
    ),
    # =========================================================================
    # Temporal Queries (Time-Based Comparisons and Periods)
    # =========================================================================
    TestScenario(
        id="E75",
        name="Temporal: Q1 2024 period query",
        feature=Feature.TEMPORAL,
        query="What happened at TechCorp in Q1 2024?",
        # Should retrieve Q1 2024 content from conflicts.md and causal_claims.md
        must_contain_any=["Q1 2024", "revenue", "stock", "20%", "record"],
        min_sources=1,
    ),
    TestScenario(
        id="E76",
        name="Temporal: year-based comparison",
        feature=Feature.TEMPORAL,
        query="Compare TechCorp's founding in 2015 with GreenDrive's founding in 2012",
        # Should find info about both companies and their founding years
        must_contain_any=["2015", "2012", "TechCorp", "GreenDrive", "founded"],
        min_sources=1,
    ),
    TestScenario(
        id="E77",
        name="Temporal: before a specific event",
        feature=Feature.TEMPORAL,
        query="What was Sarah Chen's role before becoming CEO in 2019?",
        # Should find info about her previous role at AutoMotors
        must_contain_any=["AutoMotors", "engineer", "2019", "CEO"],
        min_sources=1,
    ),
    TestScenario(
        id="E78",
        name="Temporal: what changed query",
        feature=Feature.TEMPORAL,
        query="What changed for TechCorp's stock price?",
        # Should detect "what changed" and find stock-related content
        must_contain_any=["stock", "20%", "rose", "price", "TECH"],
        min_sources=1,
    ),
    # =========================================================================
    # Aggregation Queries (List All, Count, Enumerate)
    # =========================================================================
    TestScenario(
        id="E79",
        name="Aggregation: list all people",
        feature=Feature.AGGREGATION,
        query="List all the people mentioned in the company documents",
        # Should fetch more chunks to find all people (Sarah Chen, Marcus Webb, etc.)
        must_contain_any=["Sarah Chen", "Marcus Webb"],
        min_sources=1,
    ),
    TestScenario(
        id="E80",
        name="Aggregation: what are the different companies",
        feature=Feature.AGGREGATION,
        query="What are the different companies mentioned?",
        # Should detect "what are the different" and find all company references
        must_contain_any=["TechCorp", "GreenDrive", "AutoMotors"],
        min_sources=1,
    ),
    TestScenario(
        id="E81",
        name="Aggregation: how many query",
        feature=Feature.AGGREGATION,
        query="How many products does TechCorp Industries have?",
        # Should fetch comprehensive results for accurate count
        must_contain_any=["vehicle", "product", "AI", "autonomous"],
        min_sources=1,
    ),
    TestScenario(
        id="E82",
        name="Aggregation: enumerate features",
        feature=Feature.AGGREGATION,
        query="Enumerate all the features of the autonomous vehicle system",
        # Should list out features comprehensively
        must_contain_any=["autonomous", "AI", "navigation", "sensor"],
        min_sources=1,
    ),
    # =========================================================================
    # Additional Multi-Hop - EDGE CASES
    # =========================================================================
    TestScenario(
        id="E83",
        name="Multi-hop: 3+ hop chain",
        feature=Feature.MULTI_HOP,
        query="What university did the CEO of the company that competes with GreenDrive attend?",
        # GreenDrive → competitor TechCorp → CEO Sarah Chen → MIT
        must_contain=["MIT"],
        min_sources=1,
    ),
    TestScenario(
        id="E84",
        name="Multi-hop: reverse direction chain",
        feature=Feature.MULTI_HOP,
        query="Which company hired someone from the fuel cell division of Toyota?",
        # Toyota fuel cell → Marcus Webb → GreenDrive
        must_contain=["GreenDrive"],
        min_sources=1,
    ),
    # =========================================================================
    # Additional Entity Graph - EDGE CASES
    # =========================================================================
    TestScenario(
        id="E85",
        name="Entity graph: location-based entity discovery",
        feature=Feature.ENTITY_GRAPH,
        query="What activities happen at the Nevada campus?",
        # Should connect Nevada → Gigafactory, Project Alpha, employees Eva Brown/Nathan Park
        must_contain_any=["Gigafactory", "Project Alpha", "battery", "research"],
        min_sources=1,
    ),
    TestScenario(
        id="E86",
        name="Entity graph: cross-document entity linking",
        feature=Feature.ENTITY_GRAPH,
        query="What connections exist between AutoMotors and current EV companies?",
        # AutoMotors → Sarah Chen → TechCorp; AutoMotors = traditional
        must_contain_any=["Sarah Chen", "TechCorp", "VP", "Engineering"],
        min_sources=1,
    ),
    TestScenario(
        id="E87",
        name="Entity graph: weak entity signal",
        feature=Feature.ENTITY_GRAPH,
        query="What do we know about the Nevada research campus?",
        # Mentioned briefly - tests weak entity extraction
        must_contain_any=["Project Alpha", "Gigafactory", "battery"],
        min_sources=1,
    ),
    # =========================================================================
    # Additional Multi-Query - EDGE CASES
    # =========================================================================
    TestScenario(
        id="E88",
        name="Multi-query: implicit AND across domains",
        feature=Feature.MULTI_QUERY,
        query="What security measures protect both user authentication and payment processing?",
        # Needs to retrieve from both auth service and payment service sections
        must_contain_any=["JWT", "PCI DSS", "token", "encryption"],
        min_sources=1,
    ),
    TestScenario(
        id="E89",
        name="Multi-query: mixed structured and unstructured",
        feature=Feature.MULTI_QUERY,
        query="Which employees work in Engineering and what products do they work on?",
        # Needs CSV employee data + product docs
        must_contain_any=["Engineering", "Alice", "Carol", "Model"],
        min_sources=1,
    ),
    TestScenario(
        id="E90",
        name="Multi-query: code and documentation",
        feature=Feature.MULTI_QUERY,
        query="How does the login method work and what are the session timeouts?",
        # Needs code_sample.py + technical.txt
        must_contain_any=["24", "hour", "authenticate", "token"],
        min_sources=1,
    ),
    # =========================================================================
    # Additional Keyword Exact - EDGE CASES
    # =========================================================================
    TestScenario(
        id="E91",
        name="Keyword: alphanumeric model ID lookup",
        feature=Feature.KEYWORD_EXACT,
        query="What are the specifications of the Model X100?",
        # Model X100 is in products.md - mid-range sedan
        must_contain_any=["45,000", "300 miles", "75 kWh", "sedan", "mid-range"],
        min_sources=1,
    ),
    TestScenario(
        id="E92",
        name="Keyword: version number lookup",
        feature=Feature.KEYWORD_EXACT,
        query="What does version 2.1 of the architecture document cover?",
        # technical.txt is Version 2.1
        must_contain_any=["microservices", "architecture", "January 2024"],
        min_sources=1,
    ),
    TestScenario(
        id="E93",
        name="Keyword: technical acronym in context",
        feature=Feature.KEYWORD_EXACT,
        query="What is the RBAC configuration?",
        # Role-based access control in technical.txt
        must_contain_any=["role", "access", "control", "admin"],
        min_sources=1,
    ),
    # =========================================================================
    # Additional Conflict Aware - EDGE CASES
    # =========================================================================
    TestScenario(
        id="E94",
        name="Conflict: implicit vs explicit numbers",
        feature=Feature.CONFLICT_AWARE,
        query="How many people support production at TechCorp?",
        # Operations breaks down (2100+350+280=2730) vs total employee counts
        must_contain_any=["2,100", "factory", "quality", "logistics", "2,730"],
        min_sources=1,
    ),
    TestScenario(
        id="E95",
        name="Conflict: percentage vs absolute",
        feature=Feature.CONFLICT_AWARE,
        query="What is the customer satisfaction at TechCorp?",
        # Marketing: 92%, HR: 4.2/5 (=84%) - different metrics
        must_contain_any=["92%", "4.2", "satisfaction"],
        min_sources=1,
    ),
    TestScenario(
        id="E96",
        name="Conflict: dated information",
        feature=Feature.CONFLICT_AWARE,
        query="What is TechCorp's market share?",
        # Marketing press release says 18%, may conflict with other sources
        must_contain_any=["18%", "market share"],
        min_sources=1,
    ),
    TestScenario(
        id="E97",
        name="Conflict: definitional differences",
        feature=Feature.CONFLICT_AWARE,
        query="What counts as an employee at TechCorp?",
        # HR includes contractors >6 months, Finance doesn't
        must_contain_any=["contractor", "full-time", "HR", "Finance"],
        min_sources=1,
    ),
    # =========================================================================
    # Additional Causal Attribution - EDGE CASES
    # =========================================================================
    TestScenario(
        id="E98",
        name="Causal: correlation not causation",
        feature=Feature.CAUSAL_ATTRIBUTION,
        query="Did the Model Z50 launch cause TechCorp's stock to rise?",
        # Multiple events happened - should hedge, not claim direct causation
        must_contain_any=["multiple", "factor", "also", "same time", "correlat"],
        min_sources=1,
    ),
    TestScenario(
        id="E99",
        name="Causal: contradicting expert opinions",
        feature=Feature.CAUSAL_ATTRIBUTION,
        query="According to analysts, what drove TechCorp's growth?",
        # Different analysts cite different reasons - should present multiple factors
        # Check for the actual opinions/factors, not just analyst names (LLM may summarize without attribution)
        must_contain_any=["product", "regulatory", "hydrogen", "tailwind", "market"],
        min_sources=1,
    ),
    TestScenario(
        id="E100",
        name="Causal: reverse causation question",
        feature=Feature.CAUSAL_ATTRIBUTION,
        query="What events might have influenced the EV tax credit expansion?",
        # Documents don't establish this causation - should insufficient or hedge
        must_contain_any=["not clear", "unknown", "cannot determine", "government", "policy"],
        min_sources=0,
    ),
    TestScenario(
        id="E101",
        name="Causal: spurious correlation",
        feature=Feature.CAUSAL_ATTRIBUTION,
        query="Did oil prices rising cause TechCorp's success?",
        # Mentioned as industry context but not direct cause
        must_contain_any=["oil", "factor", "industry", "tailwind", "multiple"],
        min_sources=1,
    ),
    # =========================================================================
    # Additional Table Schema - EDGE CASES
    # =========================================================================
    TestScenario(
        id="E102",
        name="Table schema: infer foreign keys",
        feature=Feature.TABLE_SCHEMA,
        query="How are employees linked to their managers in the data?",
        # Should identify manager_id as FK relationship
        must_contain_any=["manager_id", "E010", "E011", "reports to"],
        min_sources=1,
    ),
    TestScenario(
        id="E103",
        name="Table schema: null values handling",
        feature=Feature.TABLE_SCHEMA,
        query="Which employees don't have managers?",
        # James Wilson, Karen White, Leo Martinez, Maria Garcia, Peter Adams have no manager_id
        must_contain_any=["James Wilson", "Karen White", "Leo Martinez"],
        min_sources=1,
    ),
    TestScenario(
        id="E104",
        name="Table schema: data type inference",
        feature=Feature.TABLE_SCHEMA,
        query="What numeric fields are in the employee data?",
        # salary is numeric, employee_id has numbers but is ID
        must_contain_any=["salary", "numeric", "number"],
        min_sources=1,
    ),
    TestScenario(
        id="E105",
        name="Table schema: cardinality question",
        feature=Feature.TABLE_SCHEMA,
        query="How many unique values are in the department column?",
        # 5 departments: Engineering, Marketing, Sales, HR, Finance
        must_contain_any=["5", "five", "Engineering", "Marketing", "Sales"],
        min_sources=1,
    ),
    TestScenario(
        id="E106",
        name="Table schema: date format detection",
        feature=Feature.TABLE_SCHEMA,
        query="What date format is used in the employee hire dates?",
        # YYYY-MM-DD format
        must_contain_any=["YYYY", "2021", "2022", "date", "format"],
        min_sources=1,
    ),
    # =========================================================================
    # Additional Dedup - EDGE CASES
    # =========================================================================
    TestScenario(
        id="E107",
        name="Dedup: same fact different wording",
        feature=Feature.DEDUP,
        query="What is the price of the entry-level TechCorp vehicle?",
        # Model Z50 = $35,000 mentioned in products.md - should not duplicate
        must_contain_any=["35,000", "Z50", "entry-level"],
        min_sources=1,
    ),
    TestScenario(
        id="E108",
        name="Dedup: overlapping date ranges",
        feature=Feature.DEDUP,
        query="What happened at TechCorp in early 2024?",
        # Q1 2024 info in multiple docs - should consolidate
        must_contain_any=["Q1", "2024", "revenue", "stock"],
        min_sources=1,
    ),
    TestScenario(
        id="E109",
        name="Dedup: entity mentioned in many contexts",
        feature=Feature.DEDUP,
        query="In what contexts is Sarah Chen mentioned?",
        # CEO role, MIT, AutoMotors history, CNBC interview
        must_contain_any=["CEO", "MIT", "AutoMotors", "CNBC"],
        min_sources=1,
    ),
    TestScenario(
        id="E110",
        name="Dedup: technical term across docs",
        feature=Feature.DEDUP,
        query="What do the documents say about JWT?",
        # Mentioned in technical.txt and possibly code
        must_contain_any=["JWT", "token", "24", "authentication"],
        min_sources=1,
    ),
    TestScenario(
        id="E111",
        name="Dedup: conflicting info same entity",
        feature=Feature.DEDUP,
        query="What battery sizes are mentioned for TechCorp vehicles?",
        # 75 vs 74.8 kWh for X100 in different docs
        must_contain_any=["75", "74.8", "100", "50", "kWh"],
        min_sources=1,
    ),
    # =========================================================================
    # Additional Insufficient Evidence - EDGE CASES
    # =========================================================================
    TestScenario(
        id="E112",
        name="Insufficient: plausible but absent",
        feature=Feature.INSUFFICIENT_EVIDENCE,
        query="What is TechCorp's office address in Austin?",
        # Austin HQ mentioned, but no street address
        must_contain_any=[
            "not provided",
            "not specified",
            "no address",
            "don't have",
            "not found",
            "Austin",
            "headquarters",
        ],
        min_sources=0,
    ),
    TestScenario(
        id="E113",
        name="Insufficient: partial info available",
        feature=Feature.INSUFFICIENT_EVIDENCE,
        query="What is the full specification of Project Alpha's technology?",
        # Only limited info due to classification
        must_contain_any=[
            "classified",
            "restricted",
            "Phase 2",
            "confidential",
            "security clearance",
            "limited",
        ],
        min_sources=0,
    ),
    # =========================================================================
    # Additional Freshness - EDGE CASES
    # =========================================================================
    TestScenario(
        id="E114",
        name="Freshness: spec vs marketing discrepancy",
        feature=Feature.FRESHNESS,
        query="What is the precise battery capacity of Model X100?",
        # Spec: 74.8 kWh usable vs products.md: 75 kWh rounded
        must_contain_any=["74.8", "75", "kWh"],
        min_sources=1,
    ),
    TestScenario(
        id="E115",
        name="Freshness: authoritative source signal",
        feature=Feature.FRESHNESS,
        query="According to official specifications, what is the Z50's EPA range?",
        # Spec: 198 miles (certified) vs products.md: 200 miles
        must_contain_any=["198", "official", "EPA", "certified"],
        min_sources=1,
    ),
    # =========================================================================
    # Additional Sparse Search - EDGE CASES
    # =========================================================================
    TestScenario(
        id="E116",
        name="Sparse: rare technical term",
        feature=Feature.SPARSE_SEARCH,
        query="What uses Apache Flink?",
        # Exact term from technical.txt - stream processing
        must_contain_any=["Flink", "stream", "telemetry", "processing"],
        min_sources=1,
    ),
    TestScenario(
        id="E117",
        name="Sparse: code identifier lookup",
        feature=Feature.SPARSE_SEARCH,
        query="What does SessionExpiredError indicate?",
        # Exact exception class from code_sample.py
        must_contain_any=["session", "expired", "exception", "error"],
        min_sources=1,
    ),
    # =========================================================================
    # Additional Query Expansion - EDGE CASES
    # =========================================================================
    TestScenario(
        id="E118",
        name="Query expansion: informal term",
        feature=Feature.QUERY_EXPANSION,
        query="How do I log out of my account?",
        # "log out" should find logout method
        must_contain_any=["logout", "invalidate", "session", "token"],
        min_sources=1,
    ),
    TestScenario(
        id="E119",
        name="Query expansion: British vs American spelling",
        feature=Feature.QUERY_EXPANSION,
        query="What authorisation levels exist?",
        # British "authorisation" should find "authorization" content
        must_contain_any=["role", "admin", "manager", "permission", "user"],
        min_sources=1,
    ),
    # =========================================================================
    # Additional Temporal - EDGE CASES
    # =========================================================================
    TestScenario(
        id="E120",
        name="Temporal: implicit time reference",
        feature=Feature.TEMPORAL,
        query="What happened after the competitor recall?",
        # January 15 recall → subsequent events
        must_contain_any=["recall", "February", "March", "stock", "launch"],
        min_sources=1,
    ),
    TestScenario(
        id="E121",
        name="Temporal: relative time query",
        feature=Feature.TEMPORAL,
        query="What was TechCorp's status before Sarah Chen became CEO?",
        # Pre-2019 history
        must_contain_any=["2015", "founded", "before", "2019"],
        min_sources=1,
    ),
    # =========================================================================
    # Additional Aggregation - EDGE CASES
    # =========================================================================
    TestScenario(
        id="E122",
        name="Aggregation: count with filter",
        feature=Feature.AGGREGATION,
        query="How many employees earn over $100,000?",
        # Need to count from CSV: Carol (105k), Grace (110k), James (145k), Karen (125k), Leo (135k), Maria (115k), Peter (140k) = 7
        must_contain_any=["7", "seven", "Carol", "James", "Grace"],
        min_sources=1,
    ),
    # =========================================================================
    # Conceptual Retrieval (managed-Qwen keywords + exact evidence)
    # =========================================================================
    TestScenario(
        id="E150",
        name="Conceptual: abstract sustainability query",
        feature=Feature.CONCEPTUAL_RETRIEVAL,
        query="What is the philosophy behind TechCorp's approach to sustainable transportation?",
        # Query keywords should bridge the abstract query to concrete EV docs.
        must_contain_any=["electric", "sustainable", "green", "emission", "environment"],
        min_sources=1,
    ),
    TestScenario(
        id="E151",
        name="Conceptual: how it works explanation",
        feature=Feature.CONCEPTUAL_RETRIEVAL,
        query="How does the authentication flow ensure security?",
        # Query keywords should surface concrete auth-security evidence.
        must_contain_any=["JWT", "token", "session", "password", "encrypt"],
        min_sources=1,
    ),
    TestScenario(
        id="E152",
        name="Conceptual: domain expertise bridge",
        feature=Feature.CONCEPTUAL_RETRIEVAL,
        query="What architectural patterns ensure high availability in the system?",
        # Query keywords should bridge technical jargon to microservices docs.
        must_contain_any=["replica", "failover", "redundan", "load", "cluster"],
        min_sources=1,
    ),
    TestScenario(
        id="E153",
        name="Conceptual: vague product question",
        feature=Feature.CONCEPTUAL_RETRIEVAL,
        query="What makes TechCorp different from traditional automakers?",
        # Query keywords should surface EV-vs-traditional distinctions.
        must_contain_any=["electric", "battery", "software", "AutoMotors", "gasoline"],
        min_sources=1,
    ),
    TestScenario(
        id="E154",
        name="Conceptual: implicit requirements query",
        feature=Feature.CONCEPTUAL_RETRIEVAL,
        query="What would I need to know before buying a TechCorp vehicle?",
        # Query keywords should surface buyer-facing specs, price, and warranty.
        must_contain_any=["price", "range", "warranty", "charging", "battery"],
        min_sources=1,
    ),
    TestScenario(
        id="E155",
        name="Conceptual: short abstract query",
        feature=Feature.CONCEPTUAL_RETRIEVAL,
        query="TechCorp innovation strategy",
        # Short conceptual queries should still receive useful keyword expansion.
        must_contain_any=[
            "R&D",
            "research",
            "technology",
            "Gigafactory",
            "Project Alpha",
            "microservices",
            "architecture",
            "scalability",
        ],
        min_sources=1,
    ),
    # =========================================================================
    # Query Rewriting (Conversational Context Resolution)
    # =========================================================================
    TestScenario(
        id="E156",
        name="Query rewriting: typo correction",
        feature=Feature.QUERY_REWRITING,
        query="What is the price of the Modle X1OO?",
        # "Modle X1OO" should be corrected to "Model X100"
        must_contain_any=["45,000", "$45", "X100"],
        min_sources=1,
    ),
    TestScenario(
        id="E157",
        name="Query rewriting: filler word removal",
        feature=Feature.QUERY_REWRITING,
        query="So like, uhh, what does TechCorp basically manufacture, you know?",
        # Filler words removed, core question preserved
        must_contain_any=["electric", "vehicle", "battery", "EV"],
        min_sources=1,
    ),
    TestScenario(
        id="E158",
        name="Query rewriting: complex phrasing simplification",
        feature=Feature.QUERY_REWRITING,
        query="I was wondering if you could perhaps tell me about the individual who currently holds the position of chief executive officer at the company known as TechCorp Industries?",
        # Simplified to find CEO info
        must_contain=["Sarah Chen"],
        min_sources=1,
    ),
    TestScenario(
        id="E159",
        name="Query rewriting: question to statement optimization",
        feature=Feature.QUERY_REWRITING,
        query="What is the definition and overview of the Model Y200?",
        # Question rewritten to match document language
        must_contain_any=["Y200", "55,000", "400", "premium"],
        min_sources=1,
    ),
    TestScenario(
        id="E160",
        name="Query rewriting: multiple typos and noise",
        feature=Feature.QUERY_REWRITING,
        query="uhm wher is TechCrp headquaters locatd at??",
        # Multiple issues: filler, typos in "TechCorp", "headquarters", "located"
        must_contain=["Austin"],
        min_sources=1,
    ),
    TestScenario(
        id="E161",
        name="Query rewriting: informal to formal",
        feature=Feature.QUERY_REWRITING,
        query="yo whats the deal with that greendrive company",
        # Informal query about GreenDrive
        must_contain_any=["GreenDrive", "hydrogen", "fuel cell", "San Jose"],
        min_sources=1,
    ),
    TestScenario(
        id="E162",
        name="Query rewriting: abbreviated terms",
        feature=Feature.QUERY_REWRITING,
        query="TC CEO edu background?",
        # "TC" = TechCorp, "edu" = education
        must_contain_any=["Sarah Chen", "MIT", "PhD"],
        min_sources=1,
    ),
    TestScenario(
        id="E163",
        name="Query rewriting: run-on question",
        feature=Feature.QUERY_REWRITING,
        query="I need to know about batteries and also ranges and prices for all the cars",
        # Run-on converted to structured retrieval
        must_contain_any=["75 kWh", "100 kWh", "50 kWh", "300", "400", "200"],
        min_sources=1,
    ),
    TestScenario(
        id="E164",
        name="Query rewriting: mixed case and spacing issues",
        feature=Feature.QUERY_REWRITING,
        query="WHAT   is   the   warranty   on   TECHCORP   batteries???",
        # Normalize spacing, case
        must_contain_any=["8-year", "8 year", "150,000", "warranty"],
        min_sources=1,
    ),
    TestScenario(
        id="E165",
        name="Query rewriting: incomplete sentence",
        feature=Feature.QUERY_REWRITING,
        query="sarah chen previous job before",
        # Incomplete query about Sarah Chen's previous employment
        must_contain_any=["AutoMotors", "VP", "Engineering"],
        min_sources=1,
    ),
]


# =============================================================================
# PDF/DOCX Document Scenarios (Slow E2E - separate from main tests)
# =============================================================================
# These scenarios test PDF and DOCX parsing + retrieval.
# Run with: pytest -m e2e_parser
# Excluded from main e2e tests to keep them fast.

PDF_DOCX_SCENARIOS: list[TestScenario] = [
    # =========================================================================
    # PDF Document Queries (Nexus Robotics Q4 2025 Report)
    # =========================================================================
    TestScenario(
        id="E123",
        name="PDF: company founder lookup",
        feature=Feature.BASIC_RETRIEVAL,
        query="Who founded Nexus Robotics?",
        must_contain=["Sarah Chen"],
        min_sources=1,
    ),
    TestScenario(
        id="E124",
        name="PDF: product specification",
        feature=Feature.BASIC_RETRIEVAL,
        query="What is the maximum payload capacity of the NX-7 Navigator?",
        must_contain_any=["500 kg", "500kg"],
        min_sources=1,
    ),
    TestScenario(
        id="E125",
        name="PDF: financial data",
        feature=Feature.BASIC_RETRIEVAL,
        query="What was Nexus Robotics Q4 2025 revenue?",
        must_contain_any=["124.5 million", "$124.5"],
        min_sources=1,
    ),
    TestScenario(
        id="E126",
        name="PDF: battery specification",
        feature=Feature.BASIC_RETRIEVAL,
        query="How long can the NX-7 Navigator operate on a single charge?",
        must_contain_any=["16 hours", "16-hour"],
        min_sources=1,
    ),
    TestScenario(
        id="E127",
        name="PDF: future product launch",
        feature=Feature.BASIC_RETRIEVAL,
        query="When is the NX-8 Navigator launching?",
        must_contain_any=["March 2026"],
        min_sources=1,
    ),
    TestScenario(
        id="E128",
        name="PDF: customer data from table",
        feature=Feature.TABLE_QUERY,
        query="How many NX-7 units did Amazon order from Nexus Robotics?",
        must_contain_any=["312", "units"],
        min_sources=1,
    ),
    TestScenario(
        id="E129",
        name="PDF: technical accuracy",
        feature=Feature.BASIC_RETRIEVAL,
        query="What is the navigation accuracy of the NX-7 Navigator?",
        must_contain_any=["±2 cm", "2 cm", "centimeter"],
        min_sources=1,
    ),
    TestScenario(
        id="E130",
        name="PDF: regional sales comparison",
        feature=Feature.COMPARISON,
        query="Which region had the highest sales for Nexus Robotics in Q4 2025?",
        must_contain_any=["North America", "$67.2", "67.2 million"],
        min_sources=1,
    ),
    TestScenario(
        id="E131",
        name="PDF: employee count",
        feature=Feature.BASIC_RETRIEVAL,
        query="How many employees does Nexus Robotics have?",
        must_contain=["847"],
        min_sources=1,
    ),
    TestScenario(
        id="E132",
        name="PDF: market share",
        feature=Feature.BASIC_RETRIEVAL,
        query="What market share does the NX-7 Navigator have in North America?",
        must_contain_any=["28%", "28 percent"],
        min_sources=1,
    ),
    # =========================================================================
    # DOCX Document Queries (CloudScale Analytics Architecture)
    # =========================================================================
    TestScenario(
        id="E133",
        name="DOCX: company origin",
        feature=Feature.BASIC_RETRIEVAL,
        query="Who launched CloudScale Analytics and where is it headquartered?",
        must_contain=["DataFlow"],
        must_contain_any=["Seattle", "Washington"],
        min_sources=1,
    ),
    TestScenario(
        id="E134",
        name="DOCX: data processing scale",
        feature=Feature.BASIC_RETRIEVAL,
        query="How much data does CloudScale Analytics process daily?",
        must_contain_any=["2.3 petabytes", "2.3 PB"],
        min_sources=1,
    ),
    TestScenario(
        id="E135",
        name="DOCX: SLA metrics",
        feature=Feature.BASIC_RETRIEVAL,
        query="What is CloudScale's uptime SLA and current performance?",
        must_contain_any=["99.95%", "99.97%"],
        min_sources=1,
    ),
    TestScenario(
        id="E136",
        name="DOCX: query response time",
        feature=Feature.BASIC_RETRIEVAL,
        query="What is the average query response time for CloudScale?",
        must_contain_any=["340", "milliseconds", "ms"],
        min_sources=1,
    ),
    TestScenario(
        id="E137",
        name="DOCX: infrastructure component",
        feature=Feature.TABLE_QUERY,
        query="How many Kafka instances does CloudScale run?",
        must_contain=["36"],
        min_sources=1,
    ),
    TestScenario(
        id="E138",
        name="DOCX: security architecture",
        feature=Feature.BASIC_RETRIEVAL,
        query="What encryption does CloudScale use for data at rest and in transit?",
        must_contain_any=["AES-256", "TLS 1.3"],
        min_sources=1,
    ),
    TestScenario(
        id="E139",
        name="DOCX: pricing tier",
        feature=Feature.TABLE_QUERY,
        query="What is the monthly cost for CloudScale's Professional tier?",
        must_contain_any=["$8,500", "8,500"],
        min_sources=1,
    ),
    TestScenario(
        id="E140",
        name="DOCX: compliance certification",
        feature=Feature.BASIC_RETRIEVAL,
        query="When does CloudScale plan to achieve FedRAMP certification?",
        must_contain_any=["Q3 2026"],
        min_sources=1,
    ),
    TestScenario(
        id="E141",
        name="DOCX: compute capacity",
        feature=Feature.BASIC_RETRIEVAL,
        query="What is the total compute capacity of CloudScale's Spark clusters?",
        must_contain_any=["12,000 cores", "12000", "96 TB"],
        min_sources=1,
    ),
    TestScenario(
        id="E142",
        name="DOCX: R&D budget",
        feature=Feature.BASIC_RETRIEVAL,
        query="What is CloudScale's annual R&D budget?",
        must_contain_any=["$34 million", "34 million"],
        min_sources=1,
    ),
    # =========================================================================
    # Cross-Document Queries (PDF + DOCX combined)
    # =========================================================================
    TestScenario(
        id="E143",
        name="Cross-doc: compare company sizes",
        feature=Feature.COMPARISON,
        query="Compare the employee counts of Nexus Robotics and the number of engineers at CloudScale",
        # Nexus: 847 employees, CloudScale: 67 engineers
        must_contain_any=["847", "67"],
        min_sources=1,
    ),
    TestScenario(
        id="E144",
        name="Cross-doc: technology platforms",
        feature=Feature.MULTI_QUERY,
        query="What AI or machine learning technologies do Nexus Robotics and CloudScale use?",
        # Nexus: PathSense AI, CloudScale: ML inference
        must_contain_any=["PathSense", "ML inference", "machine learning"],
        min_sources=1,
    ),
    # =========================================================================
    # Figure/Picture Retrieval Tests (Quantum Computing Report PDF)
    # =========================================================================
    TestScenario(
        id="E145",
        name="Figure: chart data retrieval",
        feature=Feature.FIGURE_RETRIEVAL,
        query="What is the projected quantum computing market size in 2028?",
        # Figure 1 shows $42.7 billion by 2028
        must_contain_any=["42.7", "billion", "2028"],
        min_sources=1,
    ),
    TestScenario(
        id="E146",
        name="Figure: caption information",
        feature=Feature.FIGURE_RETRIEVAL,
        query="What is the CAGR for the quantum computing market shown in the figure?",
        # Figure caption mentions 72.3% CAGR
        must_contain_any=["72.3%", "72.3", "CAGR"],
        min_sources=1,
    ),
    TestScenario(
        id="E147",
        name="Figure: market growth context",
        feature=Feature.FIGURE_RETRIEVAL,
        query="According to the market analysis figure, what was the quantum computing market size in 2024?",
        # Figure shows $4.8 billion in 2024
        must_contain_any=["4.8", "billion", "2024"],
        min_sources=1,
    ),
    TestScenario(
        id="E148",
        name="Figure: related content retrieval",
        feature=Feature.BASIC_RETRIEVAL,
        query="What are the key drivers for quantum computing market growth?",
        # Text around the figure discusses drivers
        must_contain_any=["pharmaceutical", "financial", "cryptography", "supply chain"],
        min_sources=1,
    ),
]


# =============================================================================
# Format-Specific Scenarios (PPTX, XLSX, SQL, Go, Java, TypeScript)
# =============================================================================
# These scenarios test file type coverage beyond the core MD/PY/CSV/PDF/DOCX.
# Run with: pytest -m e2e_krag_formats

FORMAT_SCENARIOS: list[TestScenario] = [
    # =========================================================================
    # PPTX Scenarios (Meridian Health Q3 2025 Review)
    # =========================================================================
    TestScenario(
        id="E166",
        name="PPTX: revenue from slide content",
        feature=Feature.PPTX_RETRIEVAL,
        query="What was Meridian Health's Q3 revenue?",
        must_contain_any=["847", "million"],
        min_sources=1,
    ),
    TestScenario(
        id="E167",
        name="PPTX: patient count metric",
        feature=Feature.PPTX_RETRIEVAL,
        query="How many patients did Meridian Health serve in Q3?",
        must_contain_any=["2.4 million", "2,400,000", "2.4"],
        min_sources=1,
    ),
    TestScenario(
        id="E168",
        name="PPTX: expansion plans",
        feature=Feature.PPTX_RETRIEVAL,
        query="What are Meridian Health's expansion plans?",
        must_contain_any=["Austin", "Denver", "Portland", "clinics"],
        min_sources=1,
    ),
    TestScenario(
        id="E169",
        name="PPTX: key metrics aggregation",
        feature=Feature.PPTX_RETRIEVAL,
        query="List the key metrics from the Meridian Health quarterly review",
        must_contain_any=["satisfaction", "readmission", "wait time", "telehealth"],
        min_sources=1,
    ),
    TestScenario(
        id="E170",
        name="PPTX: presenter metadata",
        feature=Feature.BASIC_RETRIEVAL,
        query="Who presented the Meridian Health quarterly review?",
        must_contain_any=["Sarah Mitchell", "Chief Medical Officer"],
        min_sources=1,
    ),
    # =========================================================================
    # XLSX Scenarios (Engineering & Marketing Budgets)
    # =========================================================================
    TestScenario(
        id="E171",
        name="XLSX: engineering budget total",
        feature=Feature.XLSX_QUERY,
        query="What is the total engineering budget?",
        must_contain_any=["engineering", "salaries", "infrastructure", "budget"],
        min_sources=1,
    ),
    TestScenario(
        id="E172",
        name="XLSX: cross-sheet comparison",
        feature=Feature.XLSX_QUERY,
        query="Compare the engineering and marketing department budgets",
        must_contain_any=["engineering", "marketing"],
        min_sources=1,
    ),
    TestScenario(
        id="E173",
        name="XLSX: specific line item",
        feature=Feature.XLSX_QUERY,
        query="How much does the company spend on AWS cloud hosting?",
        must_contain_any=["85,000", "92,000", "380,000", "AWS", "cloud"],
        min_sources=1,
    ),
    TestScenario(
        id="E174",
        name="XLSX: category identification",
        feature=Feature.XLSX_QUERY,
        query="What are the main budget categories for engineering?",
        must_contain_any=["Salaries", "Infrastructure", "Tools", "Training", "Hardware"],
        min_sources=1,
    ),
    TestScenario(
        id="E175",
        name="XLSX: marketing specific data",
        feature=Feature.XLSX_QUERY,
        query="How much does marketing spend on Google Ads?",
        must_contain_any=["45,000", "52,000", "210,000", "Google"],
        min_sources=1,
    ),
    # =========================================================================
    # SQL Scenarios (NovaMart E-Commerce Schema)
    # =========================================================================
    TestScenario(
        id="E176",
        name="SQL: table listing",
        feature=Feature.SQL_SEARCH,
        query="What tables exist in the NovaMart ecommerce schema?",
        must_contain_any=["customers", "products", "orders", "order_items"],
        min_sources=1,
    ),
    TestScenario(
        id="E177",
        name="SQL: column details",
        feature=Feature.SQL_SEARCH,
        query="What columns does the orders table have?",
        must_contain_any=["order_id", "customer_id", "status", "total_cents"],
        min_sources=1,
    ),
    TestScenario(
        id="E178",
        name="SQL: foreign key relationships",
        feature=Feature.CODE_SEARCH,
        query="How are customers and orders related in the ecommerce schema?",
        must_contain_any=["customer_id", "REFERENCES", "foreign"],
        min_sources=1,
    ),
    TestScenario(
        id="E179",
        name="SQL: index detection",
        feature=Feature.SQL_SEARCH,
        query="What indexes exist in the NovaMart ecommerce database?",
        must_contain_any=["idx_", "INDEX", "index"],
        min_sources=1,
    ),
    # =========================================================================
    # Go Scenarios (TaskFlow HTTP Server)
    # =========================================================================
    TestScenario(
        id="E180",
        name="Go: HTTP route listing",
        feature=Feature.CODE_SEARCH,
        query="What HTTP routes does the Go TaskFlow server handle?",
        must_contain_any=["/health", "/api/tasks", "HandleFunc"],
        min_sources=1,
    ),
    TestScenario(
        id="E181",
        name="Go: authentication",
        feature=Feature.CODE_SEARCH,
        query="How does the Go TaskFlow server authenticate requests?",
        must_contain_any=["AuthMiddleware", "Bearer", "Authorization", "token"],
        min_sources=1,
    ),
    TestScenario(
        id="E182",
        name="Go: middleware",
        feature=Feature.CODE_SEARCH,
        query="What middleware does the Go TaskFlow server use?",
        must_contain_any=["LoggingMiddleware", "AuthMiddleware", "logging"],
        min_sources=1,
    ),
    # =========================================================================
    # Java Scenarios (Nexus Platform UserService)
    # =========================================================================
    TestScenario(
        id="E183",
        name="Java: CRUD operations",
        feature=Feature.CODE_SEARCH,
        query="What CRUD operations does the Java UserService provide?",
        must_contain_any=["createUser", "getUserById", "updateUser", "deactivateUser"],
        min_sources=1,
    ),
    TestScenario(
        id="E184",
        name="Java: exception handling",
        feature=Feature.CODE_SEARCH,
        query="What exceptions does the Java UserService throw?",
        must_contain_any=[
            "UserNotFoundException",
            "DuplicateEmailException",
            "IllegalArgumentException",
        ],
        min_sources=1,
    ),
    TestScenario(
        id="E185",
        name="Java: annotations",
        feature=Feature.CODE_SEARCH,
        query="What annotations are used on the Java UserService?",
        must_contain_any=["@Service", "@Transactional"],
        min_sources=1,
    ),
    # =========================================================================
    # TypeScript Scenarios (InventoryHub API Client)
    # =========================================================================
    TestScenario(
        id="E186",
        name="TypeScript: API endpoints",
        feature=Feature.CODE_SEARCH,
        query="What API endpoints does the TypeScript InventoryHub client call?",
        must_contain_any=["/api/products", "/api/warehouses", "/api/shipments"],
        min_sources=1,
    ),
    TestScenario(
        id="E187",
        name="TypeScript: interfaces",
        feature=Feature.CODE_SEARCH,
        query="What TypeScript interfaces are defined in the InventoryHub api_client?",
        must_contain_any=["Product", "Warehouse", "Shipment"],
        min_sources=1,
    ),
]


def get_scenarios_by_feature(feature: Feature) -> list[TestScenario]:
    """Get all scenarios for a specific feature."""
    return [s for s in SCENARIOS if s.feature == feature]


def get_scenario_by_id(scenario_id: str) -> TestScenario | None:
    """Get a scenario by its ID."""
    for s in SCENARIOS:
        if s.id == scenario_id:
            return s
    return None
