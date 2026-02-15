import logging
import re
from typing import List
from .base import BaseTextSplitter
from .models import Document
from ..llm import LLMProvider, extract_json_from_llm

logger = logging.getLogger(__name__)


class AgenticSplitter(BaseTextSplitter):
    """Uses an LLM to determine semantic boundaries in text."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        max_chunk_size: int = 1000,
        similarity_threshold: float = 0.5,
    ):
        self.llm_provider = llm_provider
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold

    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _fallback_split(self, text: str) -> List[str]:
        return [
            text[i : i + self.max_chunk_size]
            for i in range(0, len(text), self.max_chunk_size)
        ]

    def _get_boundaries_from_llm(self, sentences: List[str]) -> List[int]:
        numbered_sentences = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
        num_sentences = len(sentences)

        prompt = (
            "You are a text segmentation assistant. Given the following numbered sentences, "
            "identify the sentence indices where a new semantic section begins. "
            "A new section starts when the topic or focus shifts meaningfully.\n\n"
            f"Sentences:\n{numbered_sentences}\n\n"
            "Return ONLY a JSON array of integer indices where new sections begin. "
            "Always include 0 as the first index. Example: [0, 4, 9]\n"
            "Do not include any explanation, markdown, or text outside the JSON array.\n"
            "Response:"
        )

        def _validate(parsed: list) -> str:
            if not all(isinstance(b, int) for b in parsed):
                return "Expected array of integers, got non-integer elements."
            return ""

        result = extract_json_from_llm(self.llm_provider, prompt, validate=_validate)
        if result is None:
            return []

        boundaries = sorted(set(b for b in result if 0 <= b < num_sentences))
        if not boundaries or boundaries[0] != 0:
            boundaries = [0] + boundaries
        return boundaries

    def split_text(self, text: str) -> List[str]:
        if len(text) <= self.max_chunk_size:
            return [text]

        sentences = self._split_into_sentences(text)
        if not sentences:
            return [text]

        # Try LLM-based boundary detection
        boundaries = self._get_boundaries_from_llm(sentences)
        if not boundaries:
            logger.info("Falling back to simple splitting for agentic chunker")
            return self._fallback_split(text)

        # Build chunks from boundaries
        chunks = []
        for idx in range(len(boundaries)):
            start = boundaries[idx]
            end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(sentences)
            chunk_text = " ".join(sentences[start:end])

            # If chunk is too large, sub-split it
            if len(chunk_text) > self.max_chunk_size * 1.5:
                sub_chunks = self._fallback_split(chunk_text)
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk_text)

        return [c for c in chunks if c.strip()]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        return [
            Document(
                content=text,
                metadata={
                    **doc.metadata,
                    "chunk_index": i,
                    "parent_id": doc.id,
                    "chunking_strategy": "agentic",
                },
            )
            for doc in documents
            for i, text in enumerate(self.split_text(doc.content))
        ]
