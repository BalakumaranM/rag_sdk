import logging
from typing import List, Tuple
from .base import BaseReranker
from ..document import Document
from ..config import CohereRerankConfig

logger = logging.getLogger(__name__)


class CohereReranker(BaseReranker):
    """Reranker using Cohere's Rerank API.

    Args:
        config: Cohere rerank configuration.
    """

    def __init__(self, config: CohereRerankConfig):
        import cohere

        self.client = cohere.Client(api_key=config.get_api_key())
        self.model = config.model

    def rerank(
        self, query: str, documents: List[Document], top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Rerank documents using Cohere Rerank API.

        Args:
            query: The search query.
            documents: List of documents to rerank.
            top_k: Maximum number of documents to return.

        Returns:
            List of (document, score) tuples, ordered by relevance descending.
        """
        if not documents:
            return []

        doc_texts = [doc.content for doc in documents]

        response = self.client.rerank(
            query=query,
            documents=doc_texts,
            model=self.model,
            top_n=min(top_k, len(documents)),
        )

        results: List[Tuple[Document, float]] = []
        for result in response.results:
            idx = result.index
            score = result.relevance_score
            results.append((documents[idx], score))

        return results
