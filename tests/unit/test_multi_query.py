from unittest.mock import MagicMock
from typing import Any

import pytest

from rag_sdk.document import Document
from rag_sdk.retrieval.multi_query import MultiQueryRetriever
from rag_sdk.config import MultiQueryConfig


@pytest.fixture()
def mock_retriever() -> Any:
    return MagicMock()


@pytest.fixture()
def mock_llm() -> Any:
    return MagicMock()


class TestMultiQueryRetriever:
    def test_generates_and_merges_results(
        self, mock_retriever: Any, mock_llm: Any
    ) -> None:
        mock_llm.generate.return_value = (
            "1. What are Python's main features?\n"
            "2. Describe key characteristics of Python\n"
            "3. What makes Python popular?"
        )

        doc_a = Document(content="doc A", id="a")
        doc_b = Document(content="doc B", id="b")
        doc_c = Document(content="doc C", id="c")
        doc_d = Document(content="doc D", id="d")

        # Each query returns some overlapping docs
        mock_retriever.retrieve.side_effect = [
            [doc_a, doc_b],  # original query
            [doc_b, doc_c],  # alt query 1
            [doc_c, doc_d],  # alt query 2
            [doc_a, doc_d],  # alt query 3
        ]

        config = MultiQueryConfig(num_queries=3)
        retriever = MultiQueryRetriever(
            base_retriever=mock_retriever,
            llm_provider=mock_llm,
            config=config,
        )

        results = retriever.retrieve("What is Python?", top_k=10)

        # Should deduplicate: a, b, c, d
        assert len(results) == 4
        assert results[0].id == "a"
        assert results[1].id == "b"
        assert results[2].id == "c"
        assert results[3].id == "d"

        # Should have called retrieve 4 times (original + 3 alt)
        assert mock_retriever.retrieve.call_count == 4

    def test_top_k_limits_merged_results(
        self, mock_retriever: Any, mock_llm: Any
    ) -> None:
        mock_llm.generate.return_value = "Alt query 1\nAlt query 2"

        mock_retriever.retrieve.side_effect = [
            [Document(content=f"doc {i}", id=str(i)) for i in range(3)],
            [Document(content=f"doc {i}", id=str(i + 3)) for i in range(3)],
            [Document(content=f"doc {i}", id=str(i + 6)) for i in range(3)],
        ]

        config = MultiQueryConfig(num_queries=2)
        retriever = MultiQueryRetriever(
            base_retriever=mock_retriever,
            llm_provider=mock_llm,
            config=config,
        )

        results = retriever.retrieve("test", top_k=5)
        assert len(results) == 5

    def test_handles_llm_failure(self, mock_retriever: Any, mock_llm: Any) -> None:
        mock_llm.generate.side_effect = Exception("LLM error")

        doc_a = Document(content="doc A", id="a")
        mock_retriever.retrieve.return_value = [doc_a]

        config = MultiQueryConfig(num_queries=3)
        retriever = MultiQueryRetriever(
            base_retriever=mock_retriever,
            llm_provider=mock_llm,
            config=config,
        )

        # Should still work with just the original query
        results = retriever.retrieve("test", top_k=5)
        assert len(results) == 1
        assert results[0].id == "a"

        # Only called once (original query, no alt queries generated)
        assert mock_retriever.retrieve.call_count == 1

    def test_deduplicates_by_id(self, mock_retriever: Any, mock_llm: Any) -> None:
        mock_llm.generate.return_value = "Alt query 1"

        shared_doc = Document(content="shared doc", id="shared")

        mock_retriever.retrieve.side_effect = [
            [shared_doc, Document(content="unique A", id="a")],
            [shared_doc, Document(content="unique B", id="b")],
        ]

        config = MultiQueryConfig(num_queries=1)
        retriever = MultiQueryRetriever(
            base_retriever=mock_retriever,
            llm_provider=mock_llm,
            config=config,
        )

        results = retriever.retrieve("test", top_k=10)

        # shared_doc should appear only once
        assert len(results) == 3
        ids = [doc.id for doc in results]
        assert ids.count("shared") == 1


class TestMultiQueryConfig:
    def test_defaults(self) -> None:
        config = MultiQueryConfig()
        assert config.num_queries == 3

    def test_in_retrieval_config(self) -> None:
        from rag_sdk.config import RetrievalConfig

        config = RetrievalConfig()
        assert config.multi_query.num_queries == 3
