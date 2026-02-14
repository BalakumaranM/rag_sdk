import json
import logging
import re
from typing import List
from .base import BaseTextSplitter
from .models import Document
from ..llm import LLMProvider

logger = logging.getLogger(__name__)


class PropositionSplitter(BaseTextSplitter):
    """
    Proposition-based chunking: uses an LLM to decompose text into atomic propositions,
    then groups propositions into chunks of configurable size.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        max_propositions_per_chunk: int = 5,
    ):
        self.llm_provider = llm_provider
        self.max_propositions_per_chunk = max_propositions_per_chunk

    def _extract_propositions(self, text: str) -> List[str]:
        """Use LLM to decompose text into atomic, self-contained propositions."""
        prompt = (
            "Decompose the following text into atomic, self-contained propositions. "
            "Each proposition should:\n"
            "- Express a single fact or claim\n"
            "- Be understandable without additional context\n"
            "- Include necessary entity references (no dangling pronouns)\n\n"
            f"Text:\n{text}\n\n"
            "Return ONLY a JSON array of strings, where each string is one proposition.\n"
            "Response:"
        )

        try:
            response = self.llm_provider.generate(prompt=prompt)
            # Try to extract JSON array from response
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if match:
                propositions = json.loads(match.group())
                if isinstance(propositions, list) and all(
                    isinstance(p, str) for p in propositions
                ):
                    return [p.strip() for p in propositions if p.strip()]
        except Exception as e:
            logger.warning(f"Proposition extraction failed: {e}")

        # Fallback: split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _group_propositions(self, propositions: List[str]) -> List[str]:
        """Group propositions into chunks."""
        chunks = []
        for i in range(0, len(propositions), self.max_propositions_per_chunk):
            group = propositions[i : i + self.max_propositions_per_chunk]
            chunk = " ".join(group)
            chunks.append(chunk)
        return chunks

    def split_text(self, text: str) -> List[str]:
        if not text.strip():
            return []

        propositions = self._extract_propositions(text)
        if not propositions:
            return [text]

        return self._group_propositions(propositions)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunks = []
        for doc in documents:
            text_chunks = self.split_text(doc.content)
            for i, text in enumerate(text_chunks):
                new_doc = Document(
                    content=text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "parent_id": doc.id,
                        "chunking_strategy": "proposition",
                    },
                )
                chunks.append(new_doc)
        return chunks
