"""FAISS vector store provider.

Uses Facebook AI Similarity Search for fast nearest-neighbor lookup.
FAISS is an index-only library — documents and metadata are stored in
a side dict, similar to InMemoryVectorStore.

Requires the optional `faiss-cpu` dependency: pip install faiss-cpu
(GPU users can install `faiss-gpu` instead — same import path.)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import FAISSConfig
from ..document import Document
from .base import VectorStoreProvider

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore[assignment]


class FAISSVectorStore(VectorStoreProvider):
    """FAISS-backed vector store with side-storage for documents."""

    def __init__(self, config: Optional[FAISSConfig] = None) -> None:
        if faiss is None:
            raise ImportError(
                "faiss is required for FAISSVectorStore. "
                "Install it with: pip install faiss-cpu"
            )
        self.config = config or FAISSConfig()
        self._index: Optional[Any] = None
        self._id_map: List[str] = []
        self._doc_store: Dict[str, Document] = {}
        self._dimension: Optional[int] = None

    def _build_index(self, dimension: int) -> Any:
        """Create a FAISS index for the given dimension."""
        metric = self.config.metric
        if metric in ("cosine", "ip"):
            index = faiss.IndexFlatIP(dimension)
        else:
            index = faiss.IndexFlatL2(dimension)
        return index

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
        """L2-normalize vectors in-place for cosine similarity."""
        if self.config.metric == "cosine":
            faiss.normalize_L2(vectors)
        return vectors

    def add_documents(
        self, documents: List[Document], embeddings: List[List[float]]
    ) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        vectors = np.array(embeddings, dtype=np.float32)
        dimension = vectors.shape[1]

        if self._index is None:
            self._dimension = dimension
            self._index = self._build_index(dimension)

        self._normalize(vectors)
        self._index.add(vectors)

        for doc in documents:
            self._id_map.append(doc.id)
            self._doc_store[doc.id] = doc

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        if self._index is None or self._index.ntotal == 0:
            return []

        query = np.array([query_embedding], dtype=np.float32)
        self._normalize(query)

        distances, indices = self._index.search(query, min(top_k, self._index.ntotal))

        results: List[Tuple[Document, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            doc_id = self._id_map[idx]
            doc = self._doc_store[doc_id]
            score = float(dist)

            if filters:
                if not all(doc.metadata.get(k) == v for k, v in filters.items()):
                    continue

            results.append((doc, score))

        return results

    def delete(self, document_ids: List[str]) -> None:
        if self._index is None:
            return

        ids_to_delete = set(document_ids)
        surviving_indices = [
            i for i, doc_id in enumerate(self._id_map) if doc_id not in ids_to_delete
        ]

        if not surviving_indices:
            self._index.reset()
            self._id_map.clear()
            for doc_id in document_ids:
                self._doc_store.pop(doc_id, None)
            return

        surviving_vectors = np.array(
            [self._index.reconstruct(i) for i in surviving_indices], dtype=np.float32
        )
        surviving_ids = [self._id_map[i] for i in surviving_indices]

        self._index.reset()
        self._index.add(surviving_vectors)
        self._id_map = surviving_ids

        for doc_id in document_ids:
            self._doc_store.pop(doc_id, None)

    def save(self, path: str) -> None:
        """Save the FAISS index to disk."""
        if self._index is not None:
            faiss.write_index(self._index, path)

    def load(self, path: str) -> None:
        """Load a FAISS index from disk."""
        self._index = faiss.read_index(path)
