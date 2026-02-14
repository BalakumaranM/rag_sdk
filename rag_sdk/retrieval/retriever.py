from typing import List, Dict, Optional, Any
from .base import BaseRetriever
from ..embeddings import EmbeddingProvider
from ..vectorstore import VectorStoreProvider
from ..document import Document
from ..config import RetrievalConfig


class Retriever(BaseRetriever):
    """
    Retriever class that coordinates embedding and vector store search.
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

    def retrieve(
        self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Retrieve documents relevant to the query.
        """
        # 1. Embed query
        query_embedding = self.embedding_provider.embed_query(query)

        # 2. Search vector store
        k = top_k or self.config.top_k
        results = self.vector_store.search(
            query_embedding=query_embedding, top_k=k, filters=filters
        )

        # 3. Return documents (strip scores for now, or could return them)
        return [doc for doc, score in results]
