import logging
from typing import List, Optional, Dict, Any, Set
from .base import BaseRetriever
from ..document import Document
from ..llm import LLMProvider
from ..config import MultiQueryConfig

logger = logging.getLogger(__name__)


class MultiQueryRetriever(BaseRetriever):
    """Generates multiple query variations using an LLM and merges retrieved results.

    This retriever expands a single query into multiple perspectives, retrieves
    documents for each variation, then deduplicates and returns the union.

    Args:
        base_retriever: The underlying retriever to fetch documents from.
        llm_provider: LLM provider for query generation.
        config: Multi-query configuration.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm_provider: LLMProvider,
        config: MultiQueryConfig,
    ):
        self.base_retriever = base_retriever
        self.llm_provider = llm_provider
        self.config = config

    def _generate_queries(self, query: str) -> List[str]:
        """Generate alternative query variations using an LLM.

        Args:
            query: The original search query.

        Returns:
            List of query variations including the original.
        """
        prompt = (
            f"Generate {self.config.num_queries} different versions of the following "
            "question to retrieve relevant documents from a vector database. "
            "Each version should approach the question from a different angle or "
            "use different keywords. Return ONLY the questions, one per line.\n\n"
            f"Original question: {query}\n\n"
            "Alternative questions:"
        )

        try:
            result = self.llm_provider.generate(prompt=prompt)
            lines = [
                line.strip().lstrip("0123456789.-) ")
                for line in result.strip().split("\n")
                if line.strip()
            ]
            # Filter empty lines and take only requested number
            queries = [q for q in lines if q][: self.config.num_queries]
            return queries
        except Exception as e:
            logger.warning(f"Query generation failed: {e}")
            return []

    def retrieve(
        self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Retrieve documents using multiple query variations.

        Args:
            query: The search query.
            top_k: Maximum number of documents to return.
            filters: Optional metadata filters.

        Returns:
            List of unique documents from all query variations.
        """
        # Generate alternative queries
        alt_queries = self._generate_queries(query)
        all_queries = [query] + alt_queries

        # Retrieve for each query and collect unique documents
        seen_ids: Set[str] = set()
        unique_docs: List[Document] = []

        for q in all_queries:
            docs = self.base_retriever.retrieve(q, top_k=top_k, filters=filters)
            for doc in docs:
                doc_id = doc.id
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_docs.append(doc)

        return unique_docs[:top_k]
