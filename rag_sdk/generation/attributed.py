import logging
import re
from typing import List, Dict, Any
from .base import GenerationStrategy
from ..document import Document
from ..llm import LLMProvider

logger = logging.getLogger(__name__)


class AttributedGeneration(GenerationStrategy):
    """Generates answers with inline [N] citations referencing source documents."""

    def __init__(self, llm_provider: LLMProvider, citation_style: str = "numeric"):
        self.llm_provider = llm_provider
        self.citation_style = citation_style

    def _build_numbered_context(self, documents: List[Document]) -> str:
        return "\n\n".join(
            f"[{i}] (Source: {doc.metadata.get('source', f'Document {i}')})\n{doc.content}"
            for i, doc in enumerate(documents, 1)
        )

    def _parse_citations(
        self, answer: str, documents: List[Document]
    ) -> List[Dict[str, Any]]:
        citation_numbers = set(int(m) for m in re.findall(r"\[(\d+)\]", answer))
        citations = []

        for num in sorted(citation_numbers):
            if 1 <= num <= len(documents):
                doc = documents[num - 1]
                citations.append(
                    {
                        "citation_number": num,
                        "document_id": doc.id,
                        "source": doc.metadata.get("source", ""),
                        "content_preview": doc.content[:200],
                    }
                )

        return citations

    def generate(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        numbered_context = self._build_numbered_context(documents)

        system_prompt = (
            "You are a helpful assistant. Use the following numbered sources to answer the user's question.\n"
            "IMPORTANT: Include inline citations using [N] notation to reference the source number "
            "for each claim or piece of information you use. For example: 'The capital of France is Paris [1].'\n"
            "If you don't know the answer, say so.\n\n"
            f"Sources:\n{numbered_context}"
        )

        answer = self.llm_provider.generate(prompt=query, system_prompt=system_prompt)
        citations = self._parse_citations(answer, documents)

        return {
            "answer": answer,
            "citations": citations,
        }
