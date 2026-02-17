from abc import ABC, abstractmethod
from typing import List, Tuple
from ..document import Document


class BaseReranker(ABC):
    """Abstract base class for all reranking strategies.

    Rerankers take a query and a list of documents, then return the documents
    reordered by relevance with scores.
    """

    @abstractmethod
    def rerank(
        self, query: str, documents: List[Document], top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Rerank documents by relevance to the query.

        Args:
            query: The search query.
            documents: List of documents to rerank.
            top_k: Maximum number of documents to return.

        Returns:
            List of (document, score) tuples, ordered by relevance descending.
        """
        pass
