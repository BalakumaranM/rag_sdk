import sys
import importlib
from types import ModuleType
from unittest.mock import MagicMock
from typing import Any

import pytest

from rag_sdk.document import Document


# --- Cohere mock ---


class MockRerankResult:
    def __init__(self, index: int, relevance_score: float):
        self.index = index
        self.relevance_score = relevance_score


class MockRerankResponse:
    def __init__(self, results: list):  # type: ignore[type-arg]
        self.results = results


@pytest.fixture()
def mock_cohere() -> Any:
    """Inject a mock cohere module into sys.modules."""
    mock_module = ModuleType("cohere")

    mock_client_cls = MagicMock()
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_module.Client = mock_client_cls  # type: ignore[attr-defined]

    sys.modules["cohere"] = mock_module

    import rag_sdk.reranking.cohere_reranker as cohere_mod

    importlib.reload(cohere_mod)

    yield mock_client

    del sys.modules["cohere"]


# --- Tests ---


class TestBaseReranker:
    def test_cannot_instantiate_abc(self) -> None:
        from rag_sdk.reranking.base import BaseReranker

        with pytest.raises(TypeError):
            BaseReranker()  # type: ignore[abstract]


class TestCohereReranker:
    def test_rerank_returns_ordered_results(self, mock_cohere: Any) -> None:
        from rag_sdk.reranking.cohere_reranker import CohereReranker
        from rag_sdk.config import CohereRerankConfig

        mock_cohere.rerank.return_value = MockRerankResponse(
            [
                MockRerankResult(index=2, relevance_score=0.95),
                MockRerankResult(index=0, relevance_score=0.80),
                MockRerankResult(index=1, relevance_score=0.60),
            ]
        )

        config = CohereRerankConfig(api_key="test-key")  # type: ignore[arg-type]
        reranker = CohereReranker(config)

        docs = [
            Document(content="doc A"),
            Document(content="doc B"),
            Document(content="doc C"),
        ]

        results = reranker.rerank("test query", docs, top_k=3)

        assert len(results) == 3
        assert results[0][0].content == "doc C"
        assert results[0][1] == 0.95
        assert results[1][0].content == "doc A"
        assert results[1][1] == 0.80
        assert results[2][0].content == "doc B"
        assert results[2][1] == 0.60

        mock_cohere.rerank.assert_called_once_with(
            query="test query",
            documents=["doc A", "doc B", "doc C"],
            model="rerank-v3.5",
            top_n=3,
        )

    def test_rerank_empty_documents(self, mock_cohere: Any) -> None:
        from rag_sdk.reranking.cohere_reranker import CohereReranker
        from rag_sdk.config import CohereRerankConfig

        config = CohereRerankConfig(api_key="test-key")  # type: ignore[arg-type]
        reranker = CohereReranker(config)

        results = reranker.rerank("test query", [], top_k=5)
        assert results == []
        mock_cohere.rerank.assert_not_called()

    def test_rerank_top_k_limits_results(self, mock_cohere: Any) -> None:
        from rag_sdk.reranking.cohere_reranker import CohereReranker
        from rag_sdk.config import CohereRerankConfig

        mock_cohere.rerank.return_value = MockRerankResponse(
            [
                MockRerankResult(index=0, relevance_score=0.9),
                MockRerankResult(index=1, relevance_score=0.8),
            ]
        )

        config = CohereRerankConfig(api_key="test-key")  # type: ignore[arg-type]
        reranker = CohereReranker(config)

        docs = [
            Document(content="doc A"),
            Document(content="doc B"),
            Document(content="doc C"),
        ]

        results = reranker.rerank("test query", docs, top_k=2)
        assert len(results) == 2

        mock_cohere.rerank.assert_called_once_with(
            query="test query",
            documents=["doc A", "doc B", "doc C"],
            model="rerank-v3.5",
            top_n=2,
        )


@pytest.fixture()
def mock_sentence_transformers() -> Any:
    """Inject a mock sentence_transformers module into sys.modules."""
    mock_st = ModuleType("sentence_transformers")
    mock_cross_encoder_cls = MagicMock()
    mock_st.CrossEncoder = mock_cross_encoder_cls  # type: ignore[attr-defined]

    sys.modules["sentence_transformers"] = mock_st

    import rag_sdk.reranking.cross_encoder as ce_mod

    importlib.reload(ce_mod)

    yield mock_cross_encoder_cls

    del sys.modules["sentence_transformers"]


class TestCrossEncoderReranker:
    def test_rerank_returns_sorted_by_score(
        self, mock_sentence_transformers: Any
    ) -> None:
        from rag_sdk.reranking.cross_encoder import CrossEncoderReranker
        from rag_sdk.config import CrossEncoderRerankConfig

        import numpy as np

        mock_model = mock_sentence_transformers.return_value
        mock_model.predict.return_value = np.array([0.3, 0.9, 0.6])

        config = CrossEncoderRerankConfig()
        reranker = CrossEncoderReranker(config)

        docs = [
            Document(content="low relevance"),
            Document(content="high relevance"),
            Document(content="medium relevance"),
        ]

        results = reranker.rerank("test query", docs, top_k=3)

        assert len(results) == 3
        assert results[0][0].content == "high relevance"
        assert results[1][0].content == "medium relevance"
        assert results[2][0].content == "low relevance"

        mock_model.predict.assert_called_once()
        pairs = mock_model.predict.call_args[0][0]
        assert len(pairs) == 3
        assert pairs[0] == ["test query", "low relevance"]

    def test_rerank_empty_documents(self, mock_sentence_transformers: Any) -> None:
        from rag_sdk.reranking.cross_encoder import CrossEncoderReranker
        from rag_sdk.config import CrossEncoderRerankConfig

        config = CrossEncoderRerankConfig()
        reranker = CrossEncoderReranker(config)

        results = reranker.rerank("test query", [], top_k=5)
        assert results == []

    def test_rerank_top_k_limits(self, mock_sentence_transformers: Any) -> None:
        from rag_sdk.reranking.cross_encoder import CrossEncoderReranker
        from rag_sdk.config import CrossEncoderRerankConfig

        import numpy as np

        mock_model = mock_sentence_transformers.return_value
        mock_model.predict.return_value = np.array([0.9, 0.8, 0.7])

        config = CrossEncoderRerankConfig()
        reranker = CrossEncoderReranker(config)

        docs = [
            Document(content="doc A"),
            Document(content="doc B"),
            Document(content="doc C"),
        ]

        results = reranker.rerank("test query", docs, top_k=2)
        assert len(results) == 2

    def test_import_error_without_dep(self) -> None:
        """CrossEncoderReranker raises ImportError if sentence-transformers missing."""
        original = sys.modules.pop("sentence_transformers", None)
        try:
            import rag_sdk.reranking.cross_encoder as ce_mod

            importlib.reload(ce_mod)

            from rag_sdk.config import CrossEncoderRerankConfig

            with pytest.raises(ImportError, match="sentence-transformers"):
                ce_mod.CrossEncoderReranker(CrossEncoderRerankConfig())
        finally:
            if original is not None:
                sys.modules["sentence_transformers"] = original


class TestCrossEncoderRerankConfig:
    def test_defaults(self) -> None:
        from rag_sdk.config import CrossEncoderRerankConfig

        config = CrossEncoderRerankConfig()
        assert config.model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert config.batch_size == 32


class TestRerankingConfig:
    def test_default_config(self) -> None:
        from rag_sdk.config import RerankingConfig

        config = RerankingConfig()
        assert config.enabled is False
        assert config.provider == "cohere"

    def test_cohere_rerank_config_defaults(self) -> None:
        from rag_sdk.config import CohereRerankConfig

        config = CohereRerankConfig()
        assert config.model == "rerank-v3.5"
        assert config.top_n == 5

    def test_reranking_in_retrieval_config(self) -> None:
        from rag_sdk.config import RetrievalConfig

        config = RetrievalConfig()
        assert config.reranking.enabled is False
        assert config.reranking.provider == "cohere"
