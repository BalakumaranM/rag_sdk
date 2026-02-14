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
        """
        Split text into chunks.
        This is a simplified recursive character splitter.
        """
        final_chunks = []
        separator = self.separators[-1]

        # Find the best separator
        for sep in self.separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break

        # Split
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)  # Split by character

        # Merge
        current_chunk: List[str] = []
        current_length = 0

        for split in splits:
            if not split:
                continue

            split_len = len(split) + (len(separator) if separator else 0)

            if current_length + split_len > self.chunk_size:
                if current_chunk:
                    doc_chunk = separator.join(current_chunk)
                    final_chunks.append(doc_chunk)

                    # Handle overlap (simplified: keep last n chars? No, keep last splits)
                    # For a proper overlap, we need to backtrack.
                    # Keeping it simple: start new chunk with current split

                    # Logic for overlap is complex. Let's do a simpler sliding window
                    # if we were strictly token based.
                    # For recursive:
                    # We accept that chunks might be slightly smaller or we just start fresh.
                    # To support overlap correctly, we'd need to keep some previous splits.

                    overlap_size = 0
                    overlap_chunk: List[str] = []
                    # Try to keep some of the previous splits that fit in overlap
                    for i in range(len(current_chunk) - 1, -1, -1):
                        s_len = len(current_chunk[i]) + len(separator)
                        if overlap_size + s_len <= self.chunk_overlap:
                            overlap_chunk.insert(0, current_chunk[i])
                            overlap_size += s_len
                        else:
                            break

                    current_chunk = overlap_chunk
                    current_length = overlap_size

            current_chunk.append(split)
            current_length += split_len

        if current_chunk:
            final_chunks.append(separator.join(current_chunk))

        return final_chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks, preserving metadata.
        """
        chunks = []
        for doc in documents:
            text_chunks = self.split_text(doc.content)
            for i, text in enumerate(text_chunks):
                new_doc = Document(
                    content=text,
                    metadata={**doc.metadata, "chunk_index": i, "parent_id": doc.id},
                )
                chunks.append(new_doc)
        return chunks
