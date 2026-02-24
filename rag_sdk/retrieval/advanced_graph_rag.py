"""AdvancedGraphRAGRetriever: Microsoft-style GraphRAG with local, global, and DRIFT search.

Ingestion is handled by ``GraphIndexer`` (rag_sdk.graph.indexer).
This module focuses purely on query-time retrieval.
"""

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import networkx as nx

    _NETWORKX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NETWORKX_AVAILABLE = False

from ..config import RetrievalConfig
from ..document import Document
from ..embeddings import EmbeddingProvider
from ..graph import Community, Entity, GraphIndexer, Relationship
from ..llm import LLMProvider
from ..vectorstore import VectorStoreProvider
from ..settings import Settings
from .base import BaseRetriever

logger = logging.getLogger(__name__)


class AdvancedGraphRAGRetriever(BaseRetriever):
    """Microsoft-style GraphRAG with community detection, summarization, and multi-modal search.

    Implements three retrieval modes inspired by Microsoft's GraphRAG paper:

    - **Local search**: Traverses from query-matched entities, passes the
      surrounding entity/relationship neighborhood (with descriptions) as
      structured context alongside dense-retrieved chunks.

    - **Global search**: Scores community summaries semantically (boosted by
      community rank). A map-reduce step generates a partial answer per top
      community, then synthesizes them into a single response document.

    - **DRIFT search**: Generates a hypothetical answer (HyDE) to identify
      entry communities, retrieves initial chunks, then iteratively generates
      follow-up questions and fetches additional supporting documents.

    Ingestion is delegated to ``GraphIndexer`` (``rag_sdk.graph``).
    Call ``build_graph(documents)`` once after ingesting documents into the
    vector store, then use ``retrieve()`` for query-time search.

    Requires networkx (``pip install rag_sdk[advanced-graph-rag]``).
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_store: Optional[VectorStoreProvider] = None,
        llm_provider: Optional[LLMProvider] = None,
        config: Optional[RetrievalConfig] = None,
    ):
        if not _NETWORKX_AVAILABLE:
            raise ImportError(
                "networkx is required for AdvancedGraphRAGRetriever. "
                "Install it with: pip install rag_sdk[advanced-graph-rag]"
            )
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._llm_provider = llm_provider
        self.config = config

        # Indexer owns all graph state; retriever reads from it at query time.
        # Pass through the explicit provider args (may be None → indexer reads Settings).
        self._indexer = GraphIndexer(embedding_provider, llm_provider, config)

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        provider = self._embedding_provider or Settings.embedding_provider
        if provider is None:
            raise RuntimeError(
                "No embedding provider available. Pass one to AdvancedGraphRAGRetriever() "
                "or set Settings.embedding_provider."
            )
        return provider

    @property
    def vector_store(self) -> VectorStoreProvider:
        if self._vector_store is None:
            raise RuntimeError(
                "No vector store available. Pass one to AdvancedGraphRAGRetriever()."
            )
        return self._vector_store

    @property
    def llm_provider(self) -> LLMProvider:
        provider = self._llm_provider or Settings.llm_provider
        if provider is None:
            raise RuntimeError(
                "No LLM provider available. Pass one to AdvancedGraphRAGRetriever() "
                "or set Settings.llm_provider."
            )
        return provider

    # ---------------------------------------------------------------------------
    # Convenience accessors — keeps external code that reads retriever.entities etc. working
    # ---------------------------------------------------------------------------

    @property
    def entities(self) -> Dict[str, Entity]:
        return self._indexer.entities

    @property
    def relationships(self) -> List[Relationship]:
        return self._indexer.relationships

    @property
    def graph(self) -> "nx.Graph":
        return self._indexer.graph

    @property
    def communities(self) -> Dict[str, Community]:
        return self._indexer.communities

    # ---------------------------------------------------------------------------
    # Graph Building (delegates entirely to GraphIndexer)
    # ---------------------------------------------------------------------------

    def build_graph(self, documents: List[Document]) -> None:
        """Extract entities/relationships, detect communities, and summarize them.

        Args:
            documents: Chunked documents to build the graph from.
        """
        self._indexer.build_graph(documents)
        logger.info(
            "Advanced knowledge graph ready: %d entities, %d communities",
            len(self.entities),
            len(self.communities),
        )

    # ---------------------------------------------------------------------------
    # Query-time helpers
    # ---------------------------------------------------------------------------

    def _extract_query_entities(self, query: str) -> List[str]:
        prompt = (
            "Extract key entity names from this query. "
            "Return ONLY a JSON array of lowercase strings.\n\n"
            f"Query: {query}\nResponse:"
        )
        try:
            response = self.llm_provider.generate(prompt=prompt)
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if match:
                return [
                    e.lower() for e in json.loads(match.group()) if isinstance(e, str)
                ]
        except Exception as e:
            logger.warning("Query entity extraction failed: %s", e)
        return [w.lower() for w in query.split() if len(w) > 3]

    def _match_entities(self, query_entities: List[str]) -> Set[str]:
        """Fuzzy-match query entity names to nodes in the graph."""
        matched: Set[str] = set()
        for qe in query_entities:
            for ge in self.entities:
                if qe in ge or ge in qe:
                    matched.add(ge)
        return matched

    def _get_neighborhood(self, seed_entities: Set[str], hops: int) -> Set[str]:
        """Return all nodes reachable from seed_entities within *hops* edges."""
        visited: Set[str] = set()
        frontier = set(seed_entities)
        for _ in range(hops):
            next_frontier: Set[str] = set()
            for node in frontier:
                if node in visited or node not in self.graph:
                    continue
                visited.add(node)
                next_frontier.update(set(self.graph.neighbors(node)) - visited)
            frontier = next_frontier
        return visited

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        arr_a = np.array(a)
        arr_b = np.array(b)
        denom = np.linalg.norm(arr_a) * np.linalg.norm(arr_b)
        return float(np.dot(arr_a, arr_b) / denom) if denom > 0 else 0.0

    def _score_communities(
        self, query_embedding: List[float]
    ) -> List[Tuple[Community, float]]:
        """Rank communities by cosine similarity weighted by community rank score."""
        scored = []
        for c in self.communities.values():
            if c.embedding:
                sim = self._cosine_similarity(query_embedding, c.embedding)
                # rank (1–10) provides a modest boost: rank=10 → 1.5x
                rank_factor = 1.0 + (c.rank / 20.0)
                scored.append((c, sim * rank_factor))
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def _make_doc(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        return Document(
            id=f"synth-{uuid.uuid4().hex[:8]}",
            content=content,
            metadata=metadata or {},
        )

    # ---------------------------------------------------------------------------
    # Search Modes
    # ---------------------------------------------------------------------------

    def _local_search(
        self, query: str, top_k: int, filters: Optional[Dict[str, Any]]
    ) -> List[Document]:
        """Graph-guided local search: entity neighborhood + dense chunks."""
        assert self.config is not None, "config is required"
        cfg = self.config.advanced_graph_rag

        query_embedding = self.embedding_provider.embed_query(query)
        dense_results = self.vector_store.search(
            query_embedding=query_embedding, top_k=top_k, filters=filters
        )

        query_entities = self._extract_query_entities(query)
        seed_entities = self._match_entities(query_entities)
        neighborhood = self._get_neighborhood(seed_entities, hops=cfg.max_graph_hops)

        if not neighborhood:
            return [doc for doc, _ in dense_results]

        neighbor_rels = [
            r
            for r in self.relationships
            if r.source in neighborhood or r.target in neighborhood
        ]

        entity_lines = []
        for name in list(neighborhood)[:20]:
            if name in self.entities:
                e = self.entities[name]
                desc_suffix = f": {e.description}" if e.description else ""
                entity_lines.append(f"- {name} ({e.entity_type}){desc_suffix}")

        rel_lines = []
        for r in neighbor_rels[:20]:
            desc_suffix = f" — {r.description}" if r.description else ""
            weight_suffix = f" (strength: {r.weight:.0f})" if r.weight != 1.0 else ""
            rel_lines.append(
                f"- {r.source} → [{r.relation}] → {r.target}{desc_suffix}{weight_suffix}"
            )

        sections = ["## Relevant Entities\n" + "\n".join(entity_lines)]
        if rel_lines:
            sections.append("## Relationships\n" + "\n".join(rel_lines))

        graph_doc = self._make_doc(
            "\n\n".join(sections),
            {"source": "knowledge_graph", "type": "graph_context"},
        )
        dense_docs = [doc for doc, _ in dense_results][: max(top_k - 1, 1)]
        return [graph_doc] + dense_docs

    def _global_search(
        self, query: str, top_k: int, filters: Optional[Dict[str, Any]]
    ) -> List[Document]:
        """Map-reduce over community summaries for broad, dataset-spanning queries."""
        assert self.config is not None, "config is required"
        cfg = self.config.advanced_graph_rag

        query_embedding = self.embedding_provider.embed_query(query)
        ranked = self._score_communities(query_embedding)
        top = ranked[: cfg.top_communities]

        if not top:
            results = self.vector_store.search(
                query_embedding=query_embedding, top_k=top_k, filters=filters
            )
            return [doc for doc, _ in results]

        partial_answers: List[str] = []
        for community, _score in top:
            prompt = (
                "Based on the following topic summary, answer the query as best you can.\n\n"
                f"Topic: {community.summary}\n\n"
                f"Query: {query}\n\n"
                "Provide a concise partial answer (2–4 sentences) using only the information above:"
            )
            try:
                partial_answers.append(
                    self.llm_provider.generate(prompt=prompt).strip()
                )
            except Exception as e:
                logger.warning("Global search map failed: %s", e)

        if not partial_answers:
            results = self.vector_store.search(
                query_embedding=query_embedding, top_k=top_k, filters=filters
            )
            return [doc for doc, _ in results]

        partials_text = "\n\n".join(
            f"Partial answer {i + 1}:\n{ans}" for i, ans in enumerate(partial_answers)
        )
        reduce_prompt = (
            f"You have multiple partial answers to the query: '{query}'\n\n"
            f"{partials_text}\n\n"
            "Synthesize these into a comprehensive, coherent answer:"
        )
        try:
            synthesis = self.llm_provider.generate(prompt=reduce_prompt).strip()
        except Exception as e:
            logger.warning("Global search reduce failed: %s", e)
            synthesis = "\n\n".join(partial_answers)

        synth_doc = self._make_doc(
            synthesis, {"source": "global_search", "type": "synthesized_answer"}
        )
        dense_results = self.vector_store.search(
            query_embedding=query_embedding, top_k=max(top_k - 1, 1), filters=filters
        )
        return [synth_doc] + [doc for doc, _ in dense_results][: max(top_k - 1, 1)]

    def _drift_search(
        self, query: str, top_k: int, filters: Optional[Dict[str, Any]]
    ) -> List[Document]:
        """DRIFT search: HyDE entry point + iterative follow-up retrieval."""
        assert self.config is not None, "config is required"
        cfg = self.config.advanced_graph_rag
        collected: Dict[str, Document] = {}

        hyde_prompt = (
            "Write a detailed hypothetical answer to the following question "
            "as if you had complete knowledge on the topic:\n\n"
            f"Question: {query}\n\nHypothetical answer:"
        )
        try:
            hypothetical = self.llm_provider.generate(prompt=hyde_prompt).strip()
        except Exception as e:
            logger.warning("DRIFT HyDE generation failed: %s", e)
            hypothetical = query

        hyp_embedding = self.embedding_provider.embed_query(hypothetical)
        top_communities = self._score_communities(hyp_embedding)[: cfg.top_communities]

        query_embedding = self.embedding_provider.embed_query(query)
        for doc, _ in self.vector_store.search(
            query_embedding=query_embedding, top_k=top_k, filters=filters
        ):
            collected[doc.id] = doc

        current_answer = hypothetical

        for round_num in range(cfg.drift_max_rounds):
            followup_prompt = (
                f"Given this partial answer to the query '{query}':\n\n"
                f"{current_answer}\n\n"
                f"Generate {cfg.drift_follow_up_questions} specific follow-up questions "
                "that would fill gaps or deepen the answer. "
                "Return ONLY a JSON array of question strings:"
            )
            try:
                response = self.llm_provider.generate(prompt=followup_prompt)
                match = re.search(r"\[.*\]", response, re.DOTALL)
                follow_ups: List[str] = json.loads(match.group()) if match else []
            except Exception as e:
                logger.warning(
                    "DRIFT follow-up generation failed (round %d): %s", round_num, e
                )
                break

            for fq in follow_ups[: cfg.drift_follow_up_questions]:
                try:
                    fq_embedding = self.embedding_provider.embed_query(fq)
                    for doc, _ in self.vector_store.search(
                        query_embedding=fq_embedding, top_k=2, filters=filters
                    ):
                        collected[doc.id] = doc
                except Exception as e:
                    logger.warning("DRIFT follow-up retrieval failed: %s", e)

            if top_communities:
                community_context = "\n\n".join(
                    c.summary for c, _ in top_communities[:2]
                )
                update_prompt = (
                    f"Using this background context:\n{community_context}\n\n"
                    f"Refine your answer to: {query}\n\nRefined answer:"
                )
                try:
                    current_answer = self.llm_provider.generate(
                        prompt=update_prompt
                    ).strip()
                except Exception as e:
                    logger.warning("DRIFT answer update failed: %s", e)

        return list(collected.values())[:top_k]

    # ---------------------------------------------------------------------------
    # BaseRetriever interface
    # ---------------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Dispatch to the configured search mode: local, global, or drift."""
        if not self.communities:
            logger.warning(
                "AdvancedGraphRAGRetriever: graph not built yet — falling back to dense retrieval"
            )
            query_embedding = self.embedding_provider.embed_query(query)
            results = self.vector_store.search(
                query_embedding=query_embedding, top_k=top_k, filters=filters
            )
            return [doc for doc, _ in results]

        assert self.config is not None, "config is required"
        mode = self.config.advanced_graph_rag.search_mode
        if mode == "global":
            return self._global_search(query, top_k, filters)
        elif mode == "drift":
            return self._drift_search(query, top_k, filters)
        else:
            return self._local_search(query, top_k, filters)
