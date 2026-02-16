"""Chroma vector store provider.

Chroma stores documents, metadata, and embeddings natively — no side-storage
needed. Supports ephemeral (in-memory), persistent (on-disk), and HTTP
(client-server) modes.

Requires the optional `chromadb` dependency: pip install chromadb
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from ..config import ChromaConfig
from ..document import Document
from .base import VectorStoreProvider

logger = logging.getLogger(__name__)

try:
    import chromadb
except ImportError:
    chromadb = None  # type: ignore[assignment]


class ChromaVectorStore(VectorStoreProvider):
    """ChromaDB-backed vector store."""

    def __init__(self, config: Optional[ChromaConfig] = None) -> None:
        if chromadb is None:
            raise ImportError(
                "chromadb is required for ChromaVectorStore. "
                "Install it with: pip install chromadb"
            )
        self.config = config or ChromaConfig()
        self._client = self._create_client()
        self._collection = self._client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": self.config.distance_function},
        )

    def _create_client(self) -> Any:
        """Create a Chroma client based on config mode."""
        mode = self.config.mode
        if mode == "persistent":
            return chromadb.PersistentClient(path=self.config.persist_path)
        elif mode == "http":
            return chromadb.HttpClient(host=self.config.host, port=self.config.port)
        else:
            return chromadb.EphemeralClient()

    def _score_from_distance(self, distance: float) -> float:
        """Convert Chroma distance to a similarity score."""
        if self.config.distance_function == "cosine":
            return 1.0 - distance
        elif self.config.distance_function == "l2":
            return 1.0 / (1.0 + distance)
        else:
            return float(distance)

    def add_documents(
        self, documents: List[Document], embeddings: List[List[float]]
    ) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        ids = [doc.id for doc in documents]
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata if doc.metadata else {} for doc in documents]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        query_params: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            query_params["where"] = filters

        results = self._collection.query(**query_params)

        documents_out: List[Tuple[Document, float]] = []
        if not results["ids"] or not results["ids"][0]:
            return documents_out

        for i, doc_id in enumerate(results["ids"][0]):
            content = results["documents"][0][i] if results["documents"] else ""
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 0.0
            score = self._score_from_distance(distance)

            doc = Document(id=doc_id, content=content or "", metadata=metadata or {})
            documents_out.append((doc, score))

        return documents_out

    def delete(self, document_ids: List[str]) -> None:
        self._collection.delete(ids=document_ids)
