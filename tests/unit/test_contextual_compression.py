from unittest.mock import MagicMock
from typing import Any

import pytest

from rag_sdk.document import Document
from rag_sdk.retrieval.contextual_compression import ContextualCompressionRetriever
from rag_sdk.config import ContextualCompressionConfig


@pytest.fixture()
def mock_retriever() -> Any:
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        Document(content="Full document about Python programming and web development."),
        Document(content="Document about machine learning algorithms."),
        Document(content="Irrelevant document about cooking recipes."),
    ]
    return retriever


@pytest.fixture()
def mock_llm() -> Any:
    return MagicMock()


class TestContextualCompressionRetriever:
    def test_compresses_documents(self, mock_retriever: Any, mock_llm: Any) -> None:
        mock_llm.generate.side_effect = [
            "Python programming is relevant to the query.",
            "Machine learning algorithms are useful.",
            "NO_RELEVANT_CONTENT",
        ]

        config = ContextualCompressionConfig()
        retriever = ContextualCompressionRetriever(
            base_retriever=mock_retriever,
            llm_provider=mock_llm,
            config=config,
        )

        results = retriever.retrieve("What is Python?", top_k=5)

        assert len(results) == 2
        assert results[0].content == "Python programming is relevant to the query."
        assert results[0].metadata["compressed"] is True
        assert results[1].content == "Machine learning algorithms are useful."

    def test_filters_no_relevant_content(
        self, mock_retriever: Any, mock_llm: Any
    ) -> None:
        mock_llm.generate.side_effect = [
            "NO_RELEVANT_CONTENT",
            "NO_RELEVANT_CONTENT",
            "NO_RELEVANT_CONTENT",
        ]

        config = ContextualCompressionConfig()
        retriever = ContextualCompressionRetriever(
            base_retriever=mock_retriever,
            llm_provider=mock_llm,
            config=config,
        )

        results = retriever.retrieve("Unrelated query", top_k=5)
        assert len(results) == 0

    def test_preserves_metadata(self, mock_retriever: Any, mock_llm: Any) -> None:
        mock_retriever.retrieve.return_value = [
            Document(
                content="Long document content here.",
                metadata={"source": "paper.pdf", "page": 3},
            ),
        ]
        mock_llm.generate.return_value = "Compressed content."

        config = ContextualCompressionConfig()
        retriever = ContextualCompressionRetriever(
            base_retriever=mock_retriever,
            llm_provider=mock_llm,
            config=config,
        )

        results = retriever.retrieve("test query", top_k=5)

        assert len(results) == 1
        assert results[0].metadata["source"] == "paper.pdf"
        assert results[0].metadata["page"] == 3
        assert results[0].metadata["compressed"] is True
        assert results[0].metadata["original_length"] == len(
            "Long document content here."
        )

    def test_top_k_limits_results(self, mock_retriever: Any, mock_llm: Any) -> None:
        mock_retriever.retrieve.return_value = [
            Document(content=f"Document {i}") for i in range(5)
        ]
        mock_llm.generate.side_effect = [f"Compressed {i}" for i in range(5)]

        config = ContextualCompressionConfig()
        retriever = ContextualCompressionRetriever(
            base_retriever=mock_retriever,
            llm_provider=mock_llm,
            config=config,
        )

        results = retriever.retrieve("test query", top_k=2)
        assert len(results) == 2

    def test_over_fetches_from_base_retriever(
        self, mock_retriever: Any, mock_llm: Any
    ) -> None:
        mock_llm.generate.return_value = "Compressed."

        config = ContextualCompressionConfig()
        retriever = ContextualCompressionRetriever(
            base_retriever=mock_retriever,
            llm_provider=mock_llm,
            config=config,
        )

        retriever.retrieve("test query", top_k=3)

        # Should over-fetch: top_k * 2 = 6
        mock_retriever.retrieve.assert_called_once_with(
            "test query", top_k=6, filters=None
        )

    def test_handles_llm_exception_gracefully(
        self, mock_retriever: Any, mock_llm: Any
    ) -> None:
        mock_llm.generate.side_effect = Exception("LLM error")

        config = ContextualCompressionConfig()
        retriever = ContextualCompressionRetriever(
            base_retriever=mock_retriever,
            llm_provider=mock_llm,
            config=config,
        )

        results = retriever.retrieve("test query", top_k=5)

        # Falls back to original documents on error
        assert len(results) == 3

    def test_empty_base_results(self, mock_llm: Any) -> None:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []

        config = ContextualCompressionConfig()
        retriever = ContextualCompressionRetriever(
            base_retriever=mock_retriever,
            llm_provider=mock_llm,
            config=config,
        )

        results = retriever.retrieve("test query", top_k=5)
        assert len(results) == 0


class TestContextualCompressionConfig:
    def test_defaults(self) -> None:
        config = ContextualCompressionConfig()
        assert config.enabled is False
