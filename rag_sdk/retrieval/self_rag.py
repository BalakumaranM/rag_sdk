import logging
from typing import List, Optional, Dict, Any
from .base import BaseRetriever
from ..document import Document
from ..llm import LLMProvider, extract_json_from_llm
from ..config import SelfRAGConfig

logger = logging.getLogger(__name__)


class SelfRAGRetriever(BaseRetriever):
    """Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.

    Implements a simplified Self-RAG that:
    1. Decides whether retrieval is needed for the query
    2. If needed, retrieves and evaluates document relevance
    3. Filters to only supported documents

    The full Self-RAG also includes generation-time critique (is the response
    supported? is it useful?), but those are generation concerns handled separately.

    Args:
        base_retriever: The underlying retriever to fetch documents from.
        llm_provider: LLM provider for self-reflection.
        config: Self-RAG configuration.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm_provider: LLMProvider,
        config: SelfRAGConfig,
    ):
        self.base_retriever = base_retriever
        self.llm_provider = llm_provider
        self.config = config

    def _needs_retrieval(self, query: str) -> bool:
        """Decide whether the query needs retrieval or can be answered from knowledge.

        Args:
            query: The search query.

        Returns:
            True if retrieval is needed, False if the LLM can answer directly.
        """
        prompt = (
            "Determine whether the following question requires retrieving external "
            "documents to answer accurately, or if it can be answered from general "
            "knowledge alone.\n\n"
            f"Question: {query}\n\n"
            'Respond with ONLY a JSON object: {{"needs_retrieval": true}} or '
            '{{"needs_retrieval": false}}'
        )

        result = extract_json_from_llm(self.llm_provider, prompt)
        if result is not None and isinstance(result, dict):
            return result.get("needs_retrieval", True)

        # Default to retrieval when uncertain
        return True

    def _evaluate_relevance(
        self, query: str, documents: List[Document]
    ) -> List[Document]:
        """Evaluate and filter documents by relevance to the query.

        Args:
            query: The search query.
            documents: List of candidate documents.

        Returns:
            List of documents deemed relevant.
        """
        if not documents:
            return []

        doc_summaries = "\n".join(
            [
                f"[{i}] {doc.content[:300]}{'...' if len(doc.content) > 300 else ''}"
                for i, doc in enumerate(documents)
            ]
        )

        prompt = (
            "Evaluate each document's relevance to the question. For each document, "
            "determine if it contains information that would help answer the question.\n\n"
            f"Question: {query}\n\n"
            f"Documents:\n{doc_summaries}\n\n"
            "Return ONLY a JSON array of objects with "
            '"index" (int) and "relevant" (boolean).'
        )

        result = extract_json_from_llm(self.llm_provider, prompt)
        if result is not None and isinstance(result, list):
            relevant_indices = {e["index"] for e in result if e.get("relevant", False)}
            return [doc for i, doc in enumerate(documents) if i in relevant_indices]

        # Fallback: return all
        return documents

    def _is_supported(self, query: str, document: Document) -> bool:
        """Check if a document provides sufficient support for answering the query.

        Args:
            query: The search query.
            document: The document to evaluate.

        Returns:
            True if the document supports answering the query.
        """
        prompt = (
            "Does the following document contain enough information to fully or "
            "partially support answering the question?\n\n"
            f"Question: {query}\n\n"
            f"Document:\n{document.content[:500]}\n\n"
            'Respond with ONLY: {{"supported": true}} or {{"supported": false}}'
        )

        result = extract_json_from_llm(self.llm_provider, prompt)
        if result is not None and isinstance(result, dict):
            return result.get("supported", True)

        return True

    def retrieve(
        self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Retrieve documents with self-reflection on retrieval necessity and relevance.

        Args:
            query: The search query.
            top_k: Maximum number of documents to return.
            filters: Optional metadata filters.

        Returns:
            List of relevant, supported documents.
        """
        # Step 1: Decide if retrieval is needed
        if not self._needs_retrieval(query):
            logger.info("Self-RAG: Retrieval deemed unnecessary for this query.")
            return []

        # Step 2: Retrieve candidates
        candidates = self.base_retriever.retrieve(
            query, top_k=top_k * 2, filters=filters
        )

        if not candidates:
            return []

        # Step 3: Filter by relevance
        relevant = self._evaluate_relevance(query, candidates)

        if not relevant:
            logger.info("Self-RAG: No relevant documents found after evaluation.")
            return []

        # Step 4: Check support (optional, for top candidates only)
        if self.config.check_support:
            supported = [doc for doc in relevant if self._is_supported(query, doc)]
            return supported[:top_k]

        return relevant[:top_k]
