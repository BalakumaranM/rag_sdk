from unittest.mock import MagicMock, patch
from typing import Any

import pytest

from rag_sdk.document import Document
from rag_sdk.retrieval.self_rag import SelfRAGRetriever
from rag_sdk.config import SelfRAGConfig


@pytest.fixture()
def mock_retriever() -> Any:
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        Document(content="Relevant document about RAG systems."),
        Document(content="Another document about retrieval."),
        Document(content="Irrelevant document about cooking."),
    ]
    return retriever


@pytest.fixture()
def mock_llm() -> Any:
    return MagicMock()


class TestSelfRAGRetriever:
    @patch("rag_sdk.retrieval.self_rag.extract_json_from_llm")
    def test_retrieves_when_needed(
        self, mock_extract: Any, mock_retriever: Any, mock_llm: Any
    ) -> None:
        def side_effect(llm: Any, prompt: str, **kwargs: Any) -> Any:
            if "needs_retrieval" in prompt:
                return {"needs_retrieval": True}
            elif "evaluate" in prompt.lower():
                return [
                    {"index": 0, "relevant": True},
                    {"index": 1, "relevant": True},
                    {"index": 2, "relevant": False},
                ]
            return None

        mock_extract.side_effect = side_effect

        config = SelfRAGConfig(check_support=False)
        retriever = SelfRAGRetriever(
            base_retriever=mock_retriever,
            llm_provider=mock_llm,
            config=config,
        )

        results = retriever.retrieve("What is RAG?", top_k=5)

        assert len(results) == 2
        assert "RAG" in results[0].content
        assert "retrieval" in results[1].content

    @patch("rag_sdk.retrieval.self_rag.extract_json_from_llm")
    def test_skips_retrieval_when_not_needed(
        self, mock_extract: Any, mock_retriever: Any, mock_llm: Any
    ) -> None:
        mock_extract.return_value = {"needs_retrieval": False}

        config = SelfRAGConfig()
        retriever = SelfRAGRetriever(
            base_retriever=mock_retriever,
            llm_provider=mock_llm,
            config=config,
        )

        results = retriever.retrieve("What is 2+2?", top_k=5)

        assert len(results) == 0
        mock_retriever.retrieve.assert_not_called()

    @patch("rag_sdk.retrieval.self_rag.extract_json_from_llm")
    def test_support_check_filters_documents(
        self, mock_extract: Any, mock_retriever: Any, mock_llm: Any
    ) -> None:
        support_calls = 0

        def side_effect(llm: Any, prompt: str, **kwargs: Any) -> Any:
            nonlocal support_calls
            if "needs_retrieval" in prompt:
                return {"needs_retrieval": True}
            elif "evaluate" in prompt.lower():
                return [
                    {"index": 0, "relevant": True},
                    {"index": 1, "relevant": True},
                ]
            elif "supported" in prompt.lower():
                support_calls += 1
                return {"supported": support_calls == 1}
            return None

        mock_extract.side_effect = side_effect

        config = SelfRAGConfig(check_support=True)
        retriever = SelfRAGRetriever(
            base_retriever=mock_retriever,
            llm_provider=mock_llm,
            config=config,
        )

        results = retriever.retrieve("What is RAG?", top_k=5)

        assert len(results) == 1
        assert "RAG" in results[0].content

    @patch("rag_sdk.retrieval.self_rag.extract_json_from_llm")
    def test_empty_retrieval_results(self, mock_extract: Any, mock_llm: Any) -> None:
        mock_extract.return_value = {"needs_retrieval": True}

        empty_retriever = MagicMock()
        empty_retriever.retrieve.return_value = []

        config = SelfRAGConfig()
        retriever = SelfRAGRetriever(
            base_retriever=empty_retriever,
            llm_provider=mock_llm,
            config=config,
        )

        results = retriever.retrieve("test query", top_k=5)
        assert len(results) == 0

    @patch("rag_sdk.retrieval.self_rag.extract_json_from_llm")
    def test_no_relevant_after_evaluation(
        self, mock_extract: Any, mock_retriever: Any, mock_llm: Any
    ) -> None:
        def side_effect(llm: Any, prompt: str, **kwargs: Any) -> Any:
            if "needs_retrieval" in prompt:
                return {"needs_retrieval": True}
            elif "evaluate" in prompt.lower():
                return [
                    {"index": 0, "relevant": False},
                    {"index": 1, "relevant": False},
                    {"index": 2, "relevant": False},
                ]
            return None

        mock_extract.side_effect = side_effect

        config = SelfRAGConfig(check_support=False)
        retriever = SelfRAGRetriever(
            base_retriever=mock_retriever,
            llm_provider=mock_llm,
            config=config,
        )

        results = retriever.retrieve("Completely unrelated query", top_k=5)
        assert len(results) == 0


class TestSelfRAGConfig:
    def test_defaults(self) -> None:
        config = SelfRAGConfig()
        assert config.check_support is True

    def test_in_retrieval_config(self) -> None:
        from rag_sdk.config import RetrievalConfig

        config = RetrievalConfig()
        assert config.self_rag.check_support is True
