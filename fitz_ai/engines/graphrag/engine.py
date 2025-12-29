# fitz_ai/engines/graphrag/engine.py
"""
GraphRAG Engine Implementation.

This engine implements Microsoft's GraphRAG paradigm:
1. Extract entities and relationships from documents
2. Build a knowledge graph
3. Detect communities and generate summaries
4. Use local/global search for retrieval
5. Generate answers with graph context
"""

from typing import Any, Dict, List, Literal, Optional

from fitz_ai.core import Answer, KnowledgeError, Provenance, Query
from fitz_ai.engines.graphrag.config.schema import GraphRAGConfig, load_graphrag_config
from fitz_ai.engines.graphrag.graph.community import CommunityDetector, CommunitySummarizer
from fitz_ai.engines.graphrag.graph.extraction import EntityRelationshipExtractor
from fitz_ai.engines.graphrag.graph.storage import KnowledgeGraph
from fitz_ai.engines.graphrag.search.global_search import GlobalSearch, HybridSearch
from fitz_ai.engines.graphrag.search.local import LocalSearch
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


ANSWER_PROMPT = """Answer the question using ONLY the information provided in the context below.
If the context doesn't contain enough information, say "I don't know" or "The provided context doesn't contain this information."

## Context
{context}

## Question
{question}

## Instructions
- Use only facts from the context above
- Cite specific entities when relevant
- If unsure, acknowledge uncertainty

## Answer:"""


class GraphRAGEngine:
    """
    GraphRAG Engine for knowledge-graph-based retrieval.

    This engine extracts entities and relationships from documents,
    builds a knowledge graph with community structure, and uses
    local/global search for retrieval-augmented generation.

    Implements the KnowledgeEngine protocol.
    """

    def __init__(self, config: Optional[GraphRAGConfig] = None):
        """
        Initialize GraphRAG engine.

        Args:
            config: Engine configuration. Loads from default.yaml if None.
        """
        self.config = config if config is not None else load_graphrag_config()

        # Initialize components
        self._graph = KnowledgeGraph()
        self._extractor = EntityRelationshipExtractor(
            self.config.extraction,
            self.config.llm_provider,
        )
        self._community_detector = CommunityDetector(self.config.community)
        self._community_summarizer = CommunitySummarizer(self.config.llm_provider)

        # Search components (lazy initialized)
        self._local_search: Optional[LocalSearch] = None
        self._global_search: Optional[GlobalSearch] = None
        self._hybrid_search: Optional[HybridSearch] = None

        # Chat client for answer generation (uses 'smart' tier for user-facing responses)
        self._chat_client = None

        # Document tracking
        self._doc_texts: List[str] = []
        self._doc_ids: List[str] = []
        self._chunks: List[Dict[str, Any]] = []

        # State flags
        self._graph_built = False
        self._communities_built = False
        self._indexed = False

    def _get_chat_client(self):
        """Lazy load chat client with 'smart' tier for user-facing responses."""
        if self._chat_client is None:
            from fitz_ai.llm.registry import get_llm_plugin

            llm_provider = self.config.llm_provider or "cohere"
            self._chat_client = get_llm_plugin(
                plugin_type="chat",
                plugin_name=llm_provider,
                tier="smart",  # Use smart model for user-facing answers
            )
        return self._chat_client

    def _get_local_search(self) -> LocalSearch:
        """Lazy initialize local search."""
        if self._local_search is None:
            self._local_search = LocalSearch(
                self.config.search,
                self.config.embedding_provider,
            )
        return self._local_search

    def _get_global_search(self) -> GlobalSearch:
        """Lazy initialize global search."""
        if self._global_search is None:
            self._global_search = GlobalSearch(
                self.config.search,
                self.config.embedding_provider,
            )
        return self._global_search

    def _get_hybrid_search(self) -> HybridSearch:
        """Lazy initialize hybrid search."""
        if self._hybrid_search is None:
            self._hybrid_search = HybridSearch(
                self.config.search,
                self.config.embedding_provider,
            )
        return self._hybrid_search

    def add_documents(
        self,
        documents: List[str],
        doc_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents to build the knowledge graph.

        Args:
            documents: List of document texts
            doc_ids: Optional document IDs

        Returns:
            List of document IDs
        """
        if doc_ids is None:
            start_idx = len(self._doc_ids)
            doc_ids = [f"doc_{start_idx + i}" for i in range(len(documents))]

        self._doc_texts.extend(documents)
        self._doc_ids.extend(doc_ids)

        # Create chunks (simple: one chunk per document for now)
        for doc_text, doc_id in zip(documents, doc_ids):
            self._chunks.append(
                {
                    "id": doc_id,
                    "text": doc_text,
                }
            )

        # Mark graph as needing rebuild
        self._graph_built = False
        self._communities_built = False
        self._indexed = False

        logger.info(f"Added {len(documents)} documents (total: {len(self._doc_texts)})")
        return doc_ids

    def build_graph(self) -> None:
        """
        Build the knowledge graph from documents.

        Extracts entities and relationships from all chunks.
        """
        if self._graph_built:
            return

        if not self._chunks:
            raise KnowledgeError("No documents to build graph from")

        logger.info(f"Building knowledge graph from {len(self._chunks)} chunks...")

        # Extract entities and relationships
        self._graph = self._extractor.extract_from_chunks(self._chunks)

        self._graph_built = True
        logger.info(
            f"Graph built: {self._graph.num_entities} entities, "
            f"{self._graph.num_relationships} relationships"
        )

    def build_communities(self) -> None:
        """
        Detect communities and generate summaries.

        Must be called after build_graph().
        """
        if self._communities_built:
            return

        if not self._graph_built:
            self.build_graph()

        logger.info("Detecting communities...")

        # Detect communities
        communities = self._community_detector.detect_communities(
            self._graph,
            max_levels=self.config.community.max_hierarchy_levels,
        )

        if communities:
            # Generate summaries
            logger.info(f"Summarizing {len(communities)} communities...")
            communities = self._community_summarizer.summarize_communities(communities, self._graph)

            # Store in graph
            self._graph.set_communities(communities)

        self._communities_built = True
        logger.info(f"Communities built: {self._graph.num_communities} total")

    def index(self) -> None:
        """
        Build search indices.

        Must be called after build_communities().
        """
        if self._indexed:
            return

        if not self._communities_built:
            self.build_communities()

        logger.info("Building search indices...")

        # Index entities for local search
        local_search = self._get_local_search()
        local_search.index_entities(self._graph)

        # Index communities for global search
        global_search = self._get_global_search()
        global_search.index_communities(self._graph)

        self._indexed = True
        logger.info("Search indices built")

    def answer(self, query: Query) -> Answer:
        """
        Answer a question using the knowledge graph.

        Args:
            query: Query to answer

        Returns:
            Answer with provenance

        Raises:
            KnowledgeError: If no documents or graph not built
        """
        if not self._doc_texts:
            raise KnowledgeError("No documents in knowledge base")

        # Ensure graph and indices are built
        self.index()

        # Determine search mode
        search_mode = self.config.search.default_mode
        if query.constraints and hasattr(query.constraints, "search_mode"):
            search_mode = query.constraints.search_mode

        # Get context based on search mode
        context, provenance_entities = self._retrieve_context(query.text, search_mode)

        # Truncate context if needed
        max_tokens = self.config.search.max_context_tokens
        # Rough estimate: 4 chars per token
        max_chars = max_tokens * 4
        if len(context) > max_chars:
            context = context[:max_chars] + "\n\n[Context truncated...]"

        # Generate answer
        prompt = ANSWER_PROMPT.format(
            context=context,
            question=query.text,
        )

        chat_client = self._get_chat_client()
        response = chat_client.chat([{"role": "user", "content": prompt}])

        # Build provenance from entities
        chunk_lookup = {c["id"]: c["text"] for c in self._chunks}
        provenance = []
        seen_chunks = set()
        for entity in provenance_entities:
            for chunk_id in entity.source_chunks[:3]:  # Limit per entity
                if chunk_id in seen_chunks:
                    continue
                seen_chunks.add(chunk_id)
                chunk_text = chunk_lookup.get(chunk_id, "")
                excerpt = chunk_text[:200] if chunk_text else ""
                provenance.append(
                    Provenance(
                        source_id=chunk_id,
                        excerpt=excerpt,
                        metadata={
                            "entity": entity.name,
                            "type": entity.type,
                            "relevance_score": 0.0,
                        },
                    )
                )

        # Apply max_sources constraint
        if query.constraints and query.constraints.max_sources:
            provenance = provenance[: query.constraints.max_sources]

        return Answer(
            text=response.strip(),
            provenance=provenance,
            metadata={
                "engine": "graphrag",
                "search_mode": search_mode,
                "num_entities": self._graph.num_entities,
                "num_communities": self._graph.num_communities,
            },
        )

    def _retrieve_context(
        self,
        query_text: str,
        mode: Literal["local", "global", "hybrid"] = "local",
    ) -> tuple:
        """
        Retrieve context for a query.

        Args:
            query_text: Query string
            mode: Search mode

        Returns:
            Tuple of (context_string, list_of_entities)
        """
        if mode == "local":
            local_search = self._get_local_search()
            result = local_search.search(query_text, self._graph)
            return result.context, result.entities

        elif mode == "global":
            global_search = self._get_global_search()
            result = global_search.search(query_text, self._graph)
            # For global, we don't have direct entities, return empty
            return result.context, []

        else:  # hybrid
            hybrid_search = self._get_hybrid_search()
            context = hybrid_search.search(query_text, self._graph)
            # Get entities from local component
            local_result = hybrid_search.local_search.search(query_text, self._graph)
            return context, local_result.entities

    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        return {
            "num_documents": len(self._doc_texts),
            "num_chunks": len(self._chunks),
            "num_entities": self._graph.num_entities,
            "num_relationships": self._graph.num_relationships,
            "num_communities": self._graph.num_communities,
            "graph_built": self._graph_built,
            "communities_built": self._communities_built,
            "indexed": self._indexed,
            "search_mode": self.config.search.default_mode,
        }

    def clear_knowledge_base(self) -> None:
        """Clear all documents and the knowledge graph."""
        self._doc_texts.clear()
        self._doc_ids.clear()
        self._chunks.clear()
        self._graph.clear()
        self._graph_built = False
        self._communities_built = False
        self._indexed = False

        # Reset search components
        self._local_search = None
        self._global_search = None
        self._hybrid_search = None

        logger.info("Knowledge base cleared")

    def save_graph(self, path: str) -> None:
        """
        Save knowledge graph to file.

        Args:
            path: File path to save to
        """
        if not self._graph_built:
            raise KnowledgeError("No graph to save")
        self._graph.save(path)
        logger.info(f"Graph saved to {path}")

    def load_graph(self, path: str) -> None:
        """
        Load knowledge graph from file.

        Args:
            path: File path to load from
        """
        self._graph = KnowledgeGraph.load(path)
        self._graph_built = True
        self._communities_built = self._graph.num_communities > 0
        self._indexed = False
        logger.info(
            f"Graph loaded: {self._graph.num_entities} entities, "
            f"{self._graph.num_communities} communities"
        )
