import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set, Tuple
from .base import BaseRetriever
from ..document import Document
from ..embeddings import EmbeddingProvider
from ..vectorstore import VectorStoreProvider
from ..llm import LLMProvider
from ..config import RetrievalConfig

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    name: str
    entity_type: str = ""
    document_ids: List[str] = field(default_factory=list)


@dataclass
class Relationship:
    source: str
    target: str
    relation: str
    document_ids: List[str] = field(default_factory=list)


class GraphRAGRetriever(BaseRetriever):
    """Combines in-memory knowledge graph traversal with dense retrieval."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStoreProvider,
        llm_provider: LLMProvider,
        config: RetrievalConfig,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.config = config
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.adjacency: Dict[str, Set[str]] = {}

    def _extract_entities_and_relationships(
        self, text: str, document_id: str
    ) -> Tuple[List[Entity], List[Relationship]]:
        max_entities = self.config.graph_rag.max_entities_per_chunk
        max_rels = self.config.graph_rag.max_relationships_per_chunk

        prompt = (
            "Extract entities and relationships from the following text.\n\n"
            f"Text:\n{text}\n\n"
            f"Return ONLY a JSON object with:\n"
            f'- "entities": array of objects with "name" and "type" (max {max_entities})\n'
            f'- "relationships": array of objects with "source", "target", "relation" (max {max_rels})\n'
            "Response:"
        )

        try:
            response = self.llm_provider.generate(prompt=prompt)
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                entities = [
                    Entity(
                        name=e["name"].lower(),
                        entity_type=e.get("type", ""),
                        document_ids=[document_id],
                    )
                    for e in data.get("entities", [])[:max_entities]
                ]
                relationships = [
                    Relationship(
                        source=r["source"].lower(),
                        target=r["target"].lower(),
                        relation=r.get("relation", ""),
                        document_ids=[document_id],
                    )
                    for r in data.get("relationships", [])[:max_rels]
                ]
                return entities, relationships
        except Exception as e:
            logger.warning(f"Entity/relationship extraction failed: {e}")

        return [], []

    def build_graph(self, documents: List[Document]) -> None:
        for doc in documents:
            entities, relationships = self._extract_entities_and_relationships(
                doc.content, doc.id
            )

            for entity in entities:
                if entity.name in self.entities:
                    self.entities[entity.name].document_ids.append(doc.id)
                else:
                    self.entities[entity.name] = entity

                if entity.name not in self.adjacency:
                    self.adjacency[entity.name] = set()

            for rel in relationships:
                self.relationships.append(rel)
                if rel.source not in self.adjacency:
                    self.adjacency[rel.source] = set()
                if rel.target not in self.adjacency:
                    self.adjacency[rel.target] = set()
                self.adjacency[rel.source].add(rel.target)
                self.adjacency[rel.target].add(rel.source)

        logger.info(
            f"Built knowledge graph: {len(self.entities)} entities, "
            f"{len(self.relationships)} relationships"
        )

    def _extract_query_entities(self, query: str) -> List[str]:
        prompt = (
            "Extract the key entity names from this query. "
            "Return ONLY a JSON array of lowercase strings.\n\n"
            f"Query: {query}\n"
            "Response:"
        )

        try:
            response = self.llm_provider.generate(prompt=prompt)
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if match:
                entities = json.loads(match.group())
                return [e.lower() for e in entities if isinstance(e, str)]
        except Exception as e:
            logger.warning(f"Query entity extraction failed: {e}")

        # Fallback: use words longer than 3 chars
        return [w.lower() for w in query.split() if len(w) > 3]

    def _get_graph_document_ids(
        self, query_entities: List[str], max_hops: int = 2
    ) -> Set[str]:
        relevant_ids: Set[str] = set()
        visited: Set[str] = set()

        frontier = set()
        for entity_name in query_entities:
            # Fuzzy match: check if query entity is a substring of any graph entity
            for graph_entity_name in self.entities:
                if entity_name in graph_entity_name or graph_entity_name in entity_name:
                    frontier.add(graph_entity_name)

        for _ in range(max_hops):
            next_frontier: Set[str] = set()
            for entity_name in frontier:
                if entity_name in visited:
                    continue
                visited.add(entity_name)

                if entity_name in self.entities:
                    relevant_ids.update(self.entities[entity_name].document_ids)

                if entity_name in self.adjacency:
                    next_frontier.update(self.adjacency[entity_name] - visited)
            frontier = next_frontier

        return relevant_ids

    def retrieve(
        self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        # Dense retrieval
        query_embedding = self.embedding_provider.embed_query(query)
        dense_results = self.vector_store.search(
            query_embedding=query_embedding, top_k=top_k, filters=filters
        )

        # Graph-based retrieval
        query_entities = self._extract_query_entities(query)
        graph_doc_ids = self._get_graph_document_ids(query_entities)

        # Combine: boost graph-matched documents
        doc_scores: Dict[str, Tuple[Document, float]] = {}
        for doc, score in dense_results:
            boost = 1.2 if doc.id in graph_doc_ids else 1.0
            doc_scores[doc.id] = (doc, score * boost)

        # Sort by score and return top_k
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in sorted_docs[:top_k]]
