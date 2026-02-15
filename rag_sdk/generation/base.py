from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..document import Document


class GenerationStrategy(ABC):
    """Base class for all generation strategies."""

    @abstractmethod
    def generate(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """Generate an answer from a query and retrieved documents.

        Args:
            query: The user's question.
            documents: Retrieved context documents.

        Returns:
            Dict with at least an 'answer' key. Strategies may include
            additional keys (e.g., 'citations', 'verification_qa').
        """
        pass
