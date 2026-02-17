import logging
from typing import List, Optional, Dict, Any, Tuple
from .base import BaseRetriever
from .bm25 import BM25
from ..document import Document
from ..embeddings import EmbeddingProvider
from ..vectorstore import VectorStoreProvider
from ..config import RetrievalConfig

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """Combines dense vector retrieval with BM25 sparse retrieval using Reciprocal Rank Fusion.

    Dense retrieval captures semantic meaning while BM25 captures exact keyword
    matches. RRF merges both ranked lists without requiring score normalization.

    Args:
        embedding_provider: Embedding provider for dense retrieval.
        vector_store: Vector store for dense retrieval.
        config: Retrieval configuration.
        bm25_weight: Weight for BM25 results in RRF (0.0 to 1.0). Default 0.5.
        rrf_k: RRF constant to prevent high-ranked items from dominating. Default 60.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStoreProvider,
        config: RetrievalConfig,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.config = config
        self.bm25 = BM25(
            k1=config.hybrid.bm25_k1,
            b=config.hybrid.bm25_b,
        )
        self.bm25_weight = config.hybrid.bm25_weight
        self.rrf_k = config.hybrid.rrf_k
        self._indexed = False

    def index_documents(self, documents: List[Document]) -> None:
        """Build BM25 index from documents. Called during ingestion.

        Args:
            documents: List of documents to index for BM25.
        """
        self.bm25.index(documents)
        self._indexed = True

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: List[List[Tuple[Document, float]]],
        weights: List[float],
        k: int = 60,
    ) -> List[Tuple[Document, float]]:
        """Merge multiple ranked lists using weighted Reciprocal Rank Fusion.

        Args:
            ranked_lists: List of ranked result lists.
            weights: Weight for each ranked list.
            k: RRF constant.

        Returns:
            Merged list of (document, rrf_score) tuples.
        """
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for ranked_list, weight in zip(ranked_lists, weights):
            for rank, (doc, _score) in enumerate(ranked_list):
                doc_id = doc.id
                doc_map[doc_id] = doc
                rrf_score = weight / (k + rank + 1)
                scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score

        merged = [(doc_map[doc_id], score) for doc_id, score in scores.items()]
        merged.sort(key=lambda x: x[1], reverse=True)
        return merged

    def retrieve(
        self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Retrieve documents using hybrid dense + BM25 retrieval.

        Args:
            query: The search query.
            top_k: Maximum number of documents to return.
            filters: Optional metadata filters (applied to dense retrieval only).

        Returns:
            List of documents ranked by RRF score.
        """
        # Dense retrieval
        query_embedding = self.embedding_provider.embed_query(query)
        dense_results = self.vector_store.search(
            query_embedding=query_embedding, top_k=top_k, filters=filters
        )

        # BM25 retrieval
        if self._indexed:
            bm25_results = self.bm25.search(query, top_k=top_k)
        else:
            logger.warning(
                "BM25 index not built. Call index_documents() during ingestion. "
                "Falling back to dense-only retrieval."
            )
            return [doc for doc, _score in dense_results[:top_k]]

        # Merge with RRF
        dense_weight = 1.0 - self.bm25_weight
        merged = self._reciprocal_rank_fusion(
            [dense_results, bm25_results],
            [dense_weight, self.bm25_weight],
            k=self.rrf_k,
        )

        return [doc for doc, _score in merged[:top_k]]
