from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from ..document import Document


class BaseRetriever(ABC):
    """
    Abstract base class for retrievers.
    """

    @abstractmethod
    def retrieve(
        self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Retrieve documents relevant to the query.
        """
        pass
