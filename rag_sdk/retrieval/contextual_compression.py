import logging
from typing import List, Optional, Dict, Any
from .base import BaseRetriever
from ..document import Document
from ..llm import LLMProvider
from ..config import ContextualCompressionConfig

logger = logging.getLogger(__name__)


class ContextualCompressionRetriever(BaseRetriever):
    """Wraps any retriever to compress documents, extracting only query-relevant content.

    Uses an LLM to extract the portions of each retrieved document that are
    relevant to the query, reducing noise for the generation step.

    Args:
        base_retriever: The underlying retriever to fetch documents from.
        llm_provider: LLM provider for compression.
        config: Contextual compression configuration.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm_provider: LLMProvider,
        config: ContextualCompressionConfig,
    ):
        self.base_retriever = base_retriever
        self.llm_provider = llm_provider
        self.config = config

    def _compress_document(self, query: str, document: Document) -> Optional[Document]:
        """Extract query-relevant content from a single document.

        Args:
            query: The search query.
            document: The document to compress.

        Returns:
            A compressed document, or None if no relevant content found.
        """
        prompt = (
            "Given the following question and document, extract only the parts of "
            "the document that are relevant to answering the question. "
            "If no part is relevant, respond with exactly 'NO_RELEVANT_CONTENT'.\n\n"
            f"Question: {query}\n\n"
            f"Document:\n{document.content}\n\n"
            "Relevant extract:"
        )

        try:
            result = self.llm_provider.generate(prompt=prompt)
            compressed = result.strip()

            if compressed == "NO_RELEVANT_CONTENT" or not compressed:
                return None

            return Document(
                content=compressed,
                metadata={
                    **document.metadata,
                    "compressed": True,
                    "original_length": len(document.content),
                },
            )
        except Exception as e:
            logger.warning(f"Compression failed for document: {e}")
            return document

    def retrieve(
        self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Retrieve and compress documents relevant to the query.

        Args:
            query: The search query.
            top_k: Maximum number of documents to return.
            filters: Optional metadata filters.

        Returns:
            List of compressed documents, ordered by relevance.
        """
        # Over-fetch to account for documents that get filtered out
        fetch_k = top_k * 2
        documents = self.base_retriever.retrieve(query, top_k=fetch_k, filters=filters)

        compressed: List[Document] = []
        for doc in documents:
            result = self._compress_document(query, doc)
            if result is not None:
                compressed.append(result)
            if len(compressed) >= top_k:
                break

        return compressed[:top_k]
