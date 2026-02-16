"""Qdrant vector store provider (stub).

Full implementation coming soon. For now, use ``memory``, ``faiss``,
``chroma``, or ``pinecone`` providers.

Requires: pip install qdrant-client>=1.7.0
"""

from typing import Any, Dict, List, Optional, Tuple

from ..config import QdrantConfig
from ..document import Document
from .base import VectorStoreProvider

try:
    import qdrant_client  # noqa: F401

    _HAS_QDRANT = True
except ImportError:
    _HAS_QDRANT = False


class QdrantVectorStore(VectorStoreProvider):
    """Qdrant vector store (not yet implemented)."""

    def __init__(self, config: QdrantConfig) -> None:
        if not _HAS_QDRANT:
            raise ImportError(
                "qdrant-client is required for QdrantVectorStore. "
                "Install it with: pip install qdrant-client>=1.7.0"
            )
        raise NotImplementedError(
            "QdrantVectorStore is not yet implemented. "
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
