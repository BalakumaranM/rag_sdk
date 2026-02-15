from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from ..document import Document


class BaseRetriever(ABC):
    """Base class for all retrieval strategies."""

    @abstractmethod
    def retrieve(
        self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Retrieve documents relevant to the query.

        Args:
            query: The search query.
            top_k: Maximum number of documents to return.
            filters: Optional metadata filters.

        Returns:
            List of relevant documents, ordered by relevance.
        """
        pass
