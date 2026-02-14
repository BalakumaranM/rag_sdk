from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..document import Document


class GenerationStrategy(ABC):
    """
    Abstract base class for generation strategies.
    """

    @abstractmethod
    def generate(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """
        Generate an answer given a query and retrieved documents.

        Returns a dict with at least an 'answer' key.
        """
        pass
