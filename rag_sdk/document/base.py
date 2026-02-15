from abc import ABC, abstractmethod
from typing import List
from .models import Document


class BaseTextSplitter(ABC):
    """Base class for all text splitting strategies."""

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split a single text string into chunks.

        Args:
            text: The text to split.

        Returns:
            List of text chunks.
        """
        pass

    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks, preserving metadata.

        Args:
            documents: Documents to split.

        Returns:
            List of chunked documents with parent metadata.
        """
        pass
