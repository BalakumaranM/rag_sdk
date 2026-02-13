from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from ..document import Document


class VectorStoreProvider(ABC):
    """
    Abstract base class for vector store providers.
    """

    @abstractmethod
    def add_documents(
        self, documents: List[Document], embeddings: List[List[float]]
    ) -> None:
        """
        Add documents and their embeddings to the store.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[tuple[Document, float]]:
        """
        Search for similar documents using the query embedding.
        Returns a list of (Document, score) tuples.
        """
        pass

    @abstractmethod
    def delete(self, document_ids: List[str]) -> None:
        """
        Delete documents by ID.
        """
        pass
