import logging
from typing import List
from .base import EmbeddingProvider
from ..config import LocalEmbeddingConfig

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment,misc]


class LocalEmbedding(EmbeddingProvider):
    """Local embedding provider using sentence-transformers (BGE, E5, etc.).

    Supports any model available via sentence-transformers, including:
    - BAAI/bge-small-en-v1.5, BAAI/bge-base-en-v1.5, BAAI/bge-large-en-v1.5
    - intfloat/e5-small-v2, intfloat/e5-base-v2, intfloat/e5-large-v2
    - sentence-transformers/all-MiniLM-L6-v2

    Args:
        config: Local embedding configuration.
    """

    def __init__(self, config: LocalEmbeddingConfig):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for LocalEmbedding. "
                "Install it with: pip install sentence-transformers"
            )
        self.model = SentenceTransformer(config.model)
        self.prefix = config.query_prefix
        self.doc_prefix = config.document_prefix
        self.batch_size = config.batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.

        Args:
            texts: List of document texts to embed.

        Returns:
            List of embedding vectors.
        """
        prefixed = (
            [f"{self.doc_prefix}{t}" for t in texts] if self.doc_prefix else texts
        )
        embeddings = self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector.
        """
        prefixed = f"{self.prefix}{text}" if self.prefix else text
        embedding = self.model.encode(
            [prefixed],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding[0].tolist()
