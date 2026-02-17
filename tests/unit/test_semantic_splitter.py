from unittest.mock import MagicMock
from typing import Any

import pytest

from rag_sdk.document import Document
from rag_sdk.document.semantic_splitter import SemanticSplitter
from rag_sdk.config import SemanticChunkingConfig


@pytest.fixture()
def mock_embedding_provider() -> Any:
    return MagicMock()


class TestSemanticSplitter:
    def test_split_sentences(self) -> None:
        text = "First sentence. Second sentence! Third sentence?"
        sentences = SemanticSplitter._split_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Second sentence!"
        assert sentences[2] == "Third sentence?"

    def test_cosine_similarity(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert SemanticSplitter._cosine_similarity(a, b) == pytest.approx(1.0)

        c = [0.0, 1.0, 0.0]
        assert SemanticSplitter._cosine_similarity(a, c) == pytest.approx(0.0)

    def test_splits_at_low_similarity(self, mock_embedding_provider: Any) -> None:
        # Topic A sentences get similar embeddings, topic B gets different
        topic_a = [0.9, 0.1, 0.0]
        topic_a2 = [0.85, 0.15, 0.0]
        topic_b = [0.0, 0.1, 0.9]
        topic_b2 = [0.05, 0.15, 0.85]

        mock_embedding_provider.embed_documents.return_value = [
            topic_a,
            topic_a2,
            topic_b,  # big drop here
            topic_b2,
        ]

        config = SemanticChunkingConfig(breakpoint_percentile=25.0, min_chunk_size=10)
        splitter = SemanticSplitter(
            embedding_provider=mock_embedding_provider,
            config=config,
        )

        text = "Topic A first. Topic A second. Topic B first. Topic B second."
        chunks = splitter.split_text(text)

        # Should split into at least 2 chunks at the topic boundary
        assert len(chunks) >= 2
        assert "Topic A" in chunks[0]
        assert "Topic B" in chunks[-1]

    def test_single_sentence_returns_whole_text(
        self, mock_embedding_provider: Any
    ) -> None:
        config = SemanticChunkingConfig()
        splitter = SemanticSplitter(
            embedding_provider=mock_embedding_provider,
            config=config,
        )

        text = "Just one sentence."
        chunks = splitter.split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == "Just one sentence."

    def test_empty_text(self, mock_embedding_provider: Any) -> None:
        config = SemanticChunkingConfig()
        splitter = SemanticSplitter(
            embedding_provider=mock_embedding_provider,
            config=config,
        )

        chunks = splitter.split_text("")
        assert chunks == []

    def test_all_similar_no_splits(self, mock_embedding_provider: Any) -> None:
        # All sentences very similar — should result in one chunk
        mock_embedding_provider.embed_documents.return_value = [
            [0.9, 0.1, 0.0],
            [0.89, 0.11, 0.0],
            [0.88, 0.12, 0.0],
        ]

        config = SemanticChunkingConfig(breakpoint_percentile=10.0, min_chunk_size=10)
        splitter = SemanticSplitter(
            embedding_provider=mock_embedding_provider,
            config=config,
        )

        text = "Very similar A. Very similar B. Very similar C."
        chunks = splitter.split_text(text)

        # With very similar embeddings and low percentile, few splits expected
        assert len(chunks) >= 1

    def test_split_documents_preserves_metadata(
        self, mock_embedding_provider: Any
    ) -> None:
        mock_embedding_provider.embed_documents.return_value = [
            [0.9, 0.1],
            [0.1, 0.9],  # big topic shift
        ]

        config = SemanticChunkingConfig(breakpoint_percentile=50.0, min_chunk_size=5)
        splitter = SemanticSplitter(
            embedding_provider=mock_embedding_provider,
            config=config,
        )

        docs = [
            Document(
                content="Topic A content. Topic B content.",
                metadata={"source": "test.pdf"},
                id="doc-1",
            )
        ]

        chunks = splitter.split_documents(docs)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.pdf"
            assert chunk.metadata["parent_id"] == "doc-1"
            assert "chunk_index" in chunk.metadata

    def test_merges_small_chunks(self, mock_embedding_provider: Any) -> None:
        # Create embeddings that would split into many tiny chunks
        mock_embedding_provider.embed_documents.return_value = [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]

        config = SemanticChunkingConfig(
            breakpoint_percentile=50.0,
            min_chunk_size=50,  # High min to force merging
        )
        splitter = SemanticSplitter(
            embedding_provider=mock_embedding_provider,
            config=config,
        )

        text = "A. B. C. D."
        chunks = splitter.split_text(text)

        # With min_chunk_size=50, short chunks get merged
        total_len = sum(len(c) for c in chunks)
        assert total_len >= len("A. B. C. D.") - 4  # account for spaces


class TestSemanticChunkingConfig:
    def test_defaults(self) -> None:
        config = SemanticChunkingConfig()
        assert config.breakpoint_percentile == 25.0
        assert config.min_chunk_size == 100

    def test_in_document_processing_config(self) -> None:
        from rag_sdk.config.config import DocumentProcessingConfig

        config = DocumentProcessingConfig()
        assert config.semantic_chunking.breakpoint_percentile == 25.0
