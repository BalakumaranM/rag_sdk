import re
import logging
from typing import List
import numpy as np
from .base import BaseTextSplitter
from .models import Document
from ..embeddings.base import EmbeddingProvider
from ..config import SemanticChunkingConfig

logger = logging.getLogger(__name__)


class SemanticSplitter(BaseTextSplitter):
    """Semantic chunking using embedding similarity to find natural breakpoints.

    Based on Greg Kamradt's approach: split text into sentences, embed each,
    compute cosine similarity between consecutive sentences, and split at
    points where similarity drops below a threshold.

    Args:
        embedding_provider: Embedding provider for sentence embeddings.
        config: Semantic chunking configuration.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        config: SemanticChunkingConfig,
    ):
        self.embedding_provider = embedding_provider
        self.config = config

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences using regex."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a_arr = np.array(a)
        b_arr = np.array(b)
        dot = np.dot(a_arr, b_arr)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def _find_breakpoints(self, similarities: List[float]) -> List[int]:
        """Find breakpoints where similarity drops below threshold.

        Uses percentile-based thresholding: the breakpoint threshold is the
        Nth percentile of all similarity scores.

        Args:
            similarities: List of consecutive sentence similarities.

        Returns:
            List of sentence indices where splits should occur.
        """
        if not similarities:
            return []

        threshold = float(
            np.percentile(similarities, self.config.breakpoint_percentile)
        )

        breakpoints = []
        for i, sim in enumerate(similarities):
            if sim < threshold:
                breakpoints.append(i + 1)  # Split after this sentence

        return breakpoints

    def split_text(self, text: str) -> List[str]:
        """Split text into semantic chunks based on embedding similarity.

        Args:
            text: The text to split.

        Returns:
            List of semantically coherent text chunks.
        """
        sentences = self._split_sentences(text)

        if len(sentences) <= 1:
            return [text] if text.strip() else []

        # Embed all sentences
        embeddings = self.embedding_provider.embed_documents(sentences)

        # Compute consecutive similarities
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Find breakpoints
        breakpoints = self._find_breakpoints(similarities)

        # Build chunks from breakpoints
        chunks: List[str] = []
        start = 0
        for bp in breakpoints:
            chunk_sentences = sentences[start:bp]
            if chunk_sentences:
                chunks.append(" ".join(chunk_sentences))
            start = bp

        # Add remaining sentences
        if start < len(sentences):
            chunks.append(" ".join(sentences[start:]))

        # Merge small chunks if below minimum size
        merged: List[str] = []
        for chunk in chunks:
            if merged and len(merged[-1]) + len(chunk) < self.config.min_chunk_size:
                merged[-1] = merged[-1] + " " + chunk
            else:
                merged.append(chunk)

        return merged if merged else [text]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into semantically coherent chunks.

        Args:
            documents: Documents to split.

        Returns:
            List of chunked documents with parent metadata.
        """
        return [
            Document(
                content=text,
                metadata={**doc.metadata, "chunk_index": i, "parent_id": doc.id},
            )
            for doc in documents
            for i, text in enumerate(self.split_text(doc.content))
        ]
