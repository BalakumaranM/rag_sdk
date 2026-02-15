import json
import logging
import re
from typing import List, Optional, Dict, Any
from .base import BaseRetriever
from ..document import Document
from ..llm import LLMProvider
from ..config import CorrectiveRAGConfig

logger = logging.getLogger(__name__)


class CorrectiveRAGRetriever(BaseRetriever):
    """Wraps any retriever to evaluate relevance and refine queries when results are poor."""

    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm_provider: LLMProvider,
        config: CorrectiveRAGConfig,
    ):
        self.base_retriever = base_retriever
        self.llm_provider = llm_provider
        self.config = config

    def _evaluate_relevance(
        self, query: str, documents: List[Document]
    ) -> List[Document]:
        if not documents:
            return []

        doc_descriptions = "\n".join(
            [
                f"[{i}] {doc.content[:300]}{'...' if len(doc.content) > 300 else ''}"
                for i, doc in enumerate(documents)
            ]
        )

        prompt = (
            "Evaluate the relevance of each document to the given query.\n\n"
            f"Query: {query}\n\n"
            f"Documents:\n{doc_descriptions}\n\n"
            'Return ONLY a JSON array of objects with "index" (int) and "relevant" (boolean).\n'
            "Response:"
        )

        try:
            response = self.llm_provider.generate(prompt=prompt)
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if match:
                evaluations = json.loads(match.group())
                relevant_indices = {
                    e["index"] for e in evaluations if e.get("relevant", False)
                }
                return [doc for i, doc in enumerate(documents) if i in relevant_indices]
        except Exception as e:
            logger.warning(f"Relevance evaluation failed: {e}")

        # Fallback: return all documents
        return documents

    def _refine_query(self, original_query: str) -> str:
        prompt = (
            "The following query did not retrieve enough relevant results. "
            "Rewrite it to be more specific and likely to match relevant documents. "
            "Return ONLY the rewritten query, nothing else.\n\n"
            f"Original query: {original_query}\n"
            "Rewritten query:"
        )

        try:
            refined = self.llm_provider.generate(prompt=prompt)
            return refined.strip()
        except Exception as e:
            logger.warning(f"Query refinement failed: {e}")
            return original_query

    def retrieve(
        self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        current_query = query
        min_relevant = max(1, int(top_k * self.config.relevance_threshold))

        for attempt in range(self.config.max_refinement_attempts + 1):
            # Retrieve using base retriever
            documents = self.base_retriever.retrieve(
                current_query, top_k=top_k, filters=filters
            )

            if not documents:
                if attempt < self.config.max_refinement_attempts:
                    current_query = self._refine_query(current_query)
                    logger.info(
                        f"No documents found, refined query (attempt {attempt + 1}): "
                        f"{current_query}"
                    )
                    continue
                return []

            # Evaluate relevance
            relevant_docs = self._evaluate_relevance(current_query, documents)

            if len(relevant_docs) >= min_relevant:
                return relevant_docs[:top_k]

            # Not enough relevant docs - refine and retry
            if attempt < self.config.max_refinement_attempts:
                current_query = self._refine_query(current_query)
                logger.info(
                    f"Only {len(relevant_docs)} relevant docs (need {min_relevant}), "
                    f"refined query (attempt {attempt + 1}): {current_query}"
                )
            else:
                # Last attempt: return whatever we have
                return relevant_docs[:top_k] if relevant_docs else documents[:top_k]

        return []
