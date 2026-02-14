from abc import ABC, abstractmethod
from typing import List
from .models import Document


class BaseTextSplitter(ABC):
    """
    Abstract base class for text splitters.
    """

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        Split a single text string into chunks.
        """
        pass

    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks, preserving metadata.
        """
        pass
