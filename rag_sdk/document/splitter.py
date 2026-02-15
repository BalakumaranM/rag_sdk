from typing import List, Optional
from .base import BaseTextSplitter
from .models import Document


class TextSplitter(BaseTextSplitter):
    """
    Splits text into chunks recursively based on separators.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        final_chunks = []

        separator = self._find_best_separator(text)
        splits = text.split(separator) if separator else list(text)

        current_chunk: List[str] = []
        current_length = 0
        sep_len = len(separator) if separator else 0

        for split in splits:
            if not split:
                continue

            split_len = len(split) + sep_len

            if current_length + split_len > self.chunk_size and current_chunk:
                final_chunks.append(separator.join(current_chunk))
                current_chunk, current_length = self._compute_overlap(
                    current_chunk, separator
                )

            current_chunk.append(split)
            current_length += split_len

        if current_chunk:
            final_chunks.append(separator.join(current_chunk))

        return final_chunks

    def _find_best_separator(self, text: str) -> str:
        for sep in self.separators:
            if sep == "" or sep in text:
                return sep
        return self.separators[-1]

    def _compute_overlap(
        self, current_chunk: List[str], separator: str
    ) -> tuple[List[str], int]:
        overlap_chunk: List[str] = []
        overlap_size = 0
        sep_len = len(separator)

        for i in range(len(current_chunk) - 1, -1, -1):
            piece_len = len(current_chunk[i]) + sep_len
            if overlap_size + piece_len > self.chunk_overlap:
                break
            overlap_chunk.insert(0, current_chunk[i])
            overlap_size += piece_len

        return overlap_chunk, overlap_size

    def split_documents(self, documents: List[Document]) -> List[Document]:
        return [
            Document(
                content=text,
                metadata={**doc.metadata, "chunk_index": i, "parent_id": doc.id},
            )
            for doc in documents
            for i, text in enumerate(self.split_text(doc.content))
        ]
