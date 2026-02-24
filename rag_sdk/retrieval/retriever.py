from typing import List, Dict, Optional, Any
from .base import BaseRetriever
from ..embeddings import EmbeddingProvider
from ..vectorstore import VectorStoreProvider
from ..document import Document
from ..config import RetrievalConfig
from ..settings import Settings


class Retriever(BaseRetriever):
    """
    Retriever class that coordinates embedding and vector store search.
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_store: Optional[VectorStoreProvider] = None,
        config: Optional[RetrievalConfig] = None,
    ):
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self.config = config

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        """Resolve lazily: explicit init arg, else module-level Settings."""
        provider = self._embedding_provider or Settings.embedding_provider
        if provider is None:
            raise RuntimeError(
                "No embedding provider available. Pass one to Retriever() "
                "or set Settings.embedding_provider."
            )
        return provider

    @property
    def vector_store(self) -> VectorStoreProvider:
        if self._vector_store is None:
            raise RuntimeError("No vector store available. Pass one to Retriever().")
        return self._vector_store

    def retrieve(
        self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Retrieve documents relevant to the query.
        """
        # 1. Embed query
        query_embedding = self.embedding_provider.embed_query(query)

        # 2. Search vector store
        assert self.config is not None, "config is required"
        k = top_k or self.config.top_k
        results = self.vector_store.search(
            query_embedding=query_embedding, top_k=k, filters=filters
        )

        # 3. Return documents (strip scores for now, or could return them)
        return [doc for doc, score in results]
