"""Weaviate vector store provider (stub).

Full implementation coming soon. For now, use ``memory``, ``faiss``,
``chroma``, or ``pinecone`` providers.

Requires: pip install weaviate-client>=4.0.0
"""

from typing import Any, Dict, List, Optional, Tuple

from ..config import WeaviateConfig
from ..document import Document
from .base import VectorStoreProvider

try:
    import weaviate  # noqa: F401

    _HAS_WEAVIATE = True
except ImportError:
    _HAS_WEAVIATE = False


class WeaviateVectorStore(VectorStoreProvider):
    """Weaviate vector store (not yet implemented)."""

    def __init__(self, config: WeaviateConfig) -> None:
        if not _HAS_WEAVIATE:
            raise ImportError(
                "weaviate-client is required for WeaviateVectorStore. "
                "Install it with: pip install weaviate-client>=4.0.0"
            )
        raise NotImplementedError(
            "WeaviateVectorStore is not yet implemented. "
            "Available providers: memory, faiss, chroma, pinecone"
        )

    def add_documents(
        self, documents: List[Document], embeddings: List[List[float]]
    ) -> None:
        raise NotImplementedError

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        raise NotImplementedError

    def delete(self, document_ids: List[str]) -> None:
        raise NotImplementedError
