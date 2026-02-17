import logging
from typing import List, Tuple
from .base import BaseTextSplitter
from .models import Document
from ..config import LateChunkingConfig

logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
except ImportError:
    AutoTokenizer = None  # type: ignore[assignment,misc]
    AutoModel = None  # type: ignore[assignment,misc]
    torch = None  # type: ignore[assignment]


class LateSplitter(BaseTextSplitter):
    """Late chunking: embed full document at token level, then chunk and pool.

    Based on Jina AI's approach: instead of chunking first then embedding,
    this embeds the full document through a transformer to get token-level
    contextual embeddings, then splits text into chunks and mean-pools the
    token embeddings per chunk. This preserves cross-chunk context.

    The chunk embeddings are stored in document metadata for later use.

    Args:
        config: Late chunking configuration.
    """

    def __init__(self, config: LateChunkingConfig):
        if AutoTokenizer is None or AutoModel is None:
            raise ImportError(
                "transformers and torch are required for LateSplitter. "
                "Install with: pip install transformers torch"
            )
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        self.model = AutoModel.from_pretrained(config.model)
        self.model.eval()

    def _get_token_embeddings(self, text: str) -> Tuple[List[int], "torch.Tensor"]:
        """Get token-level embeddings for full text.

        Args:
            text: The full document text.

        Returns:
            Tuple of (token_ids, token_embeddings tensor).
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_tokens,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Last hidden state: (1, seq_len, hidden_dim)
        token_embeddings = outputs.last_hidden_state.squeeze(0)
        token_ids = inputs["input_ids"].squeeze(0).tolist()

        return token_ids, token_embeddings

    def _chunk_by_sentences(self, text: str) -> List[str]:
        """Split text into sentence-based chunks of target size.

        Args:
            text: The text to chunk.

        Returns:
            List of text chunks.
        """
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for sentence in sentences:
            if (
                current_length + len(sentence) > self.config.chunk_size
                and current_chunk
            ):
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

            current_chunk.append(sentence)
            current_length += len(sentence) + 1  # +1 for space

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _map_chunks_to_tokens(
        self, text: str, chunks: List[str], token_ids: List[int]
    ) -> List[Tuple[int, int]]:
        """Map text chunks to token index ranges.

        Args:
            text: The original full text.
            chunks: The text chunks.
            token_ids: The token IDs from tokenization.

        Returns:
            List of (start_token_idx, end_token_idx) for each chunk.
        """
        ranges: List[Tuple[int, int]] = []
        char_pos = 0

        for chunk in chunks:
            # Find the chunk's character positions
            chunk_start = text.find(chunk, char_pos)
            if chunk_start == -1:
                chunk_start = char_pos
            chunk_end = chunk_start + len(chunk)

            # Map character positions to token positions
            token_start = None
            token_end = None

            decoded_pos = 0
            for t_idx in range(len(token_ids)):
                # Decode token to get its character span (approximate)
                token_text = self.tokenizer.decode([token_ids[t_idx]])
                token_char_end = decoded_pos + len(token_text)

                if token_start is None and token_char_end > chunk_start:
                    token_start = t_idx
                if token_char_end >= chunk_end:
                    token_end = t_idx + 1
                    break

                decoded_pos = token_char_end

            if token_start is None:
                token_start = 0
            if token_end is None:
                token_end = len(token_ids)

            ranges.append((token_start, token_end))
            char_pos = chunk_end

        return ranges

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks. Embeddings stored separately via split_documents.

        Args:
            text: The text to split.

        Returns:
            List of text chunks.
        """
        if not text.strip():
            return []
        return self._chunk_by_sentences(text)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents with late chunking: embed full doc, then chunk and pool.

        Each chunk's metadata includes a 'late_embedding' key with the pooled
        token embedding for that chunk.

        Args:
            documents: Documents to split.

        Returns:
            List of chunked documents with late embeddings in metadata.
        """
        result: List[Document] = []

        for doc in documents:
            if not doc.content.strip():
                continue

            chunks = self._chunk_by_sentences(doc.content)
            if not chunks:
                continue

            try:
                token_ids, token_embeddings = self._get_token_embeddings(doc.content)
                token_ranges = self._map_chunks_to_tokens(
                    doc.content, chunks, token_ids
                )

                for i, (chunk_text, (t_start, t_end)) in enumerate(
                    zip(chunks, token_ranges)
                ):
                    # Mean-pool token embeddings for this chunk
                    chunk_emb = token_embeddings[t_start:t_end].mean(dim=0)
                    embedding_list = chunk_emb.tolist()

                    result.append(
                        Document(
                            content=chunk_text,
                            metadata={
                                **doc.metadata,
                                "chunk_index": i,
                                "parent_id": doc.id,
                                "late_embedding": embedding_list,
                            },
                        )
                    )
            except Exception as e:
                logger.warning(
                    f"Late chunking failed, falling back to simple chunking: {e}"
                )
                for i, chunk_text in enumerate(chunks):
                    result.append(
                        Document(
                            content=chunk_text,
                            metadata={
                                **doc.metadata,
                                "chunk_index": i,
                                "parent_id": doc.id,
                            },
                        )
                    )

        return result
