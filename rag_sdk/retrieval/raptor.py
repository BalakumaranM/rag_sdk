import logging
from typing import List, Optional, Dict, Any
import numpy as np
from .base import BaseRetriever
from ..document import Document
from ..embeddings import EmbeddingProvider
from ..vectorstore import VectorStoreProvider
from ..llm import LLMProvider
from ..config import RetrievalConfig

logger = logging.getLogger(__name__)


def _kmeans(vectors: np.ndarray, k: int, max_iter: int = 50) -> np.ndarray:
    n = vectors.shape[0]
    if n <= k:
        return np.arange(n)

    # Initialize centroids randomly
    rng = np.random.default_rng(42)
    indices = rng.choice(n, size=k, replace=False)
    centroids = vectors[indices].copy()

    assignments = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # Assign points to nearest centroid
        dists = np.linalg.norm(vectors[:, None] - centroids[None, :], axis=2)
        new_assignments = np.argmin(dists, axis=1)

        if np.array_equal(assignments, new_assignments):
            break
        assignments = new_assignments

        # Update centroids
        for j in range(k):
            mask = assignments == j
            if mask.any():
                centroids[j] = vectors[mask].mean(axis=0)

    return assignments


class RAPTORRetriever(BaseRetriever):
    """Builds a hierarchical tree of clustered document summaries for multi-level retrieval."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStoreProvider,
        llm_provider: LLMProvider,
        config: RetrievalConfig,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.config = config

    def _summarize_cluster(self, texts: List[str]) -> str:
        combined = "\n\n---\n\n".join(texts)
        if len(combined) > 8000:
            combined = combined[:8000] + "..."

        prompt = (
            "Summarize the following group of related text passages into a single "
            "coherent summary that captures the key information.\n\n"
            f"Passages:\n{combined}\n\n"
            "Summary:"
        )

        try:
            return self.llm_provider.generate(prompt=prompt)
        except Exception as e:
            logger.warning(f"Cluster summarization failed: {e}")
            # Fallback: concatenate first sentences
            return " ".join(t.split(".")[0] + "." for t in texts[:3])

    def build_tree(self, documents: List[Document]) -> None:
        num_levels = self.config.raptor.num_levels
        max_clusters = self.config.raptor.max_clusters_per_level

        current_docs = documents
        for level in range(1, num_levels + 1):
            if len(current_docs) <= 1:
                break

            logger.info(
                f"Building RAPTOR level {level} from {len(current_docs)} documents"
            )

            # Embed current level documents
            texts = [doc.content for doc in current_docs]
            embeddings = self.embedding_provider.embed_documents(texts)
            vectors = np.array(embeddings)

            # Determine number of clusters
            k = min(max_clusters, max(2, len(current_docs) // 3))

            # Cluster
            assignments = _kmeans(vectors, k)

            # Group documents by cluster
            clusters: Dict[int, List[Document]] = {}
            for idx, cluster_id in enumerate(assignments):
                clusters.setdefault(int(cluster_id), []).append(current_docs[idx])

            # Generate summaries for each cluster
            summary_docs = []
            summary_embeddings = []
            for cluster_id, cluster_docs in clusters.items():
                cluster_texts = [d.content for d in cluster_docs]
                summary_text = self._summarize_cluster(cluster_texts)

                summary_doc = Document(
                    content=summary_text,
                    metadata={
                        "raptor_level": level,
                        "cluster_id": cluster_id,
                        "source_doc_ids": [d.id for d in cluster_docs],
                    },
                )
                summary_docs.append(summary_doc)

            # Embed and store summaries
            if summary_docs:
                summary_texts = [d.content for d in summary_docs]
                summary_embeddings = self.embedding_provider.embed_documents(
                    summary_texts
                )
                self.vector_store.add_documents(summary_docs, summary_embeddings)

            current_docs = summary_docs

        logger.info("RAPTOR tree construction complete")

    def retrieve(
        self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        query_embedding = self.embedding_provider.embed_query(query)
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,
            filters=filters,
        )

        leaf_docs = [
            (doc, score)
            for doc, score in results
            if not doc.metadata.get("raptor_level")
        ]
        summary_docs = [
            (doc, score) for doc, score in results if doc.metadata.get("raptor_level")
        ]

        combined = leaf_docs[:top_k]
        remaining = top_k - len(combined)
        if remaining > 0:
            combined.extend(summary_docs[:remaining])

        return [doc for doc, _ in combined]
