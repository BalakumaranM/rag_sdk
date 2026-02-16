"""Pinecone vector store provider.

Uses the Pinecone v5+ SDK (``pinecone`` package). Document content is stored
in metadata under the ``_content`` key since Pinecone has no native content
field.

Requires: pip install pinecone>=5.0.0
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..config import PineconeConfig
from ..document import Document
from .base import VectorStoreProvider

logger = logging.getLogger(__name__)

try:
    from pinecone import Pinecone
except ImportError:
    Pinecone = None  # type: ignore[assignment,misc]

_UPSERT_BATCH_SIZE = 100


class PineconeVectorStore(VectorStoreProvider):
    """Pinecone-backed vector store (v5+ SDK)."""

    def __init__(self, config: PineconeConfig) -> None:
        if Pinecone is None:
            raise ImportError(
                "pinecone is required for PineconeVectorStore. "
                "Install it with: pip install pinecone>=5.0.0"
            )
        self.config = config
        api_key = config.get_api_key()
        self._pc = Pinecone(api_key=api_key)

        host = config.index_host
        if not host:
            host = self._pc.describe_index(config.index_name).host
        self._index = self._pc.Index(host=host)
        self._namespace = config.namespace

    @staticmethod
    def _flatten_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten non-primitive metadata values to JSON strings."""
        flat: Dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                flat[key] = value
            else:
                flat[key] = json.dumps(value)
        return flat

    @staticmethod
    def _unflatten_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Restore JSON-serialized metadata values."""
        restored: Dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, (dict, list)):
                        restored[key] = parsed
                        continue
                except (json.JSONDecodeError, ValueError):
                    pass
            restored[key] = value
        return restored

    def add_documents(
        self, documents: List[Document], embeddings: List[List[float]]
    ) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        vectors = []
        for doc, emb in zip(documents, embeddings):
            meta = self._flatten_metadata(doc.metadata)
            meta["_content"] = doc.content
            vectors.append({"id": doc.id, "values": emb, "metadata": meta})

        for i in range(0, len(vectors), _UPSERT_BATCH_SIZE):
            batch = vectors[i : i + _UPSERT_BATCH_SIZE]
            self._index.upsert(vectors=batch, namespace=self._namespace)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        query_params: Dict[str, Any] = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True,
            "namespace": self._namespace,
        }
        if filters:
            query_params["filter"] = filters

        response = self._index.query(**query_params)

        results: List[Tuple[Document, float]] = []
        for match in response.get("matches", []):
            metadata = dict(match.get("metadata", {}))
            content = metadata.pop("_content", "")
            metadata = self._unflatten_metadata(metadata)
            doc = Document(id=match["id"], content=content, metadata=metadata)
            results.append((doc, float(match.get("score", 0.0))))

        return results

    def delete(self, document_ids: List[str]) -> None:
        self._index.delete(ids=document_ids, namespace=self._namespace)
