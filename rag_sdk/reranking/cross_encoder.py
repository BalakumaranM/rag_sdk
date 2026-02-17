import logging
from typing import List, Tuple
from .base import BaseReranker
from ..document import Document
from ..config import CrossEncoderRerankConfig

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None  # type: ignore[assignment,misc]


class CrossEncoderReranker(BaseReranker):
    """Reranker using a cross-encoder model from sentence-transformers.

    Cross-encoders jointly encode the query and document together, producing
    more accurate relevance scores than bi-encoders at the cost of speed.

    Args:
        config: Cross-encoder rerank configuration.
    """

    def __init__(self, config: CrossEncoderRerankConfig):
        if CrossEncoder is None:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install it with: pip install sentence-transformers"
            )
        self.model = CrossEncoder(config.model)
        self.batch_size = config.batch_size

    def rerank(
        self, query: str, documents: List[Document], top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Rerank documents using a cross-encoder model.

        Args:
            query: The search query.
            documents: List of documents to rerank.
            top_k: Maximum number of documents to return.

        Returns:
            List of (document, score) tuples, ordered by relevance descending.
        """
        if not documents:
            return []

        pairs = [[query, doc.content] for doc in documents]
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        scored = list(zip(documents, [float(s) for s in scores]))
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:top_k]
