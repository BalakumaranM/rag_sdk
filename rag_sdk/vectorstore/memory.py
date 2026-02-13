import numpy as np
from typing import List, Dict, Optional, Tuple
from .base import VectorStoreProvider
from ..document import Document


class InMemoryVectorStore(VectorStoreProvider):
    """
    Simple in-memory vector store using numpy for cosine similarity.
    Good for testing and small datasets.
    """

    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        # We also keep lists for faster vector operations
        self._id_list: List[str] = []
        self._vector_matrix: Optional[np.ndarray] = None

    def add_documents(
        self, documents: List[Document], embeddings: List[List[float]]
    ) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        for doc, emb in zip(documents, embeddings):
            self.documents[doc.id] = doc
            self.embeddings[doc.id] = np.array(emb, dtype=np.float32)

            if doc.id not in self._id_list:
                self._id_list.append(doc.id)

        # Rebuild matrix
        self._rebuild_matrix()

    def _rebuild_matrix(self):
        if not self._id_list:
            self._vector_matrix = None
            return

        vectors = [self.embeddings[doc_id] for doc_id in self._id_list]
        self._vector_matrix = np.vstack(vectors)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[Document, float]]:
        if self._vector_matrix is None:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)

        # Calculate cosine similarity: (A . B) / (|A| * |B|)
        # Assuming vectors might not be normalized
        norm_query = np.linalg.norm(query_vec)
        norm_matrix = np.linalg.norm(self._vector_matrix, axis=1)

        # Avoid division by zero
        if norm_query == 0:
            return []

        dot_product = np.dot(self._vector_matrix, query_vec)
        cosine_similarities = dot_product / (norm_matrix * norm_query)

        # Get top k indices
        # argsort returns indices that would sort the array (ascending)
        # so we take the last k elements and reverse them
        top_k_indices = np.argsort(cosine_similarities)[-top_k:][::-1]

        results = []
        for idx in top_k_indices:
            doc_id = self._id_list[idx]
            doc = self.documents[doc_id]
            score = float(cosine_similarities[idx])

            # Apply filters if any (simple exact match metadata filter for now)
            if filters:
                match = True
                for key, value in filters.items():
                    if doc.metadata.get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            results.append((doc, score))

        return results

    def delete(self, document_ids: List[str]) -> None:
        for doc_id in document_ids:
            if doc_id in self.documents:
                del self.documents[doc_id]
            if doc_id in self.embeddings:
                del self.embeddings[doc_id]
            if doc_id in self._id_list:
                self._id_list.remove(doc_id)

        self._rebuild_matrix()
