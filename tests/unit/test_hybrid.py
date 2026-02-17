from unittest.mock import MagicMock
from typing import Any

import pytest

from rag_sdk.document import Document
from rag_sdk.retrieval.bm25 import BM25
from rag_sdk.retrieval.hybrid import HybridRetriever
from rag_sdk.config import RetrievalConfig, HybridRetrievalConfig


class TestBM25:
    def test_tokenize(self) -> None:
        tokens = BM25._tokenize("Hello, World! This is a test.")
        assert tokens == ["hello", "world", "this", "is", "a", "test"]

    def test_index_and_search(self) -> None:
        bm25 = BM25()
        docs = [
            Document(content="Python is a programming language"),
            Document(content="Java is another programming language"),
            Document(content="Cooking recipes and food preparation"),
        ]
        bm25.index(docs)

        results = bm25.search("Python programming", top_k=2)

        assert len(results) == 2
        assert results[0][0].content == "Python is a programming language"
        assert results[0][1] > results[1][1]  # Python doc should score higher

    def test_search_empty_index(self) -> None:
        bm25 = BM25()
        results = bm25.search("test query", top_k=5)
        assert results == []

    def test_exact_keyword_match(self) -> None:
        bm25 = BM25()
        docs = [
            Document(content="The quick brown fox jumps over the lazy dog"),
            Document(content="A slow red cat sleeps under the active tree"),
        ]
        bm25.index(docs)

        results = bm25.search("quick fox", top_k=2)
        assert results[0][0].content.startswith("The quick brown fox")

    def test_idf_scoring(self) -> None:
        """Common terms should have lower IDF weight."""
        bm25 = BM25()
        docs = [
            Document(content="the cat sat on the mat"),
            Document(content="the dog ran on the field"),
            Document(content="rare unique special terms here"),
        ]
        bm25.index(docs)

        # "the" appears in all docs (low IDF), "rare" appears in one (high IDF)
        results = bm25.search("rare", top_k=3)
        assert results[0][0].content.startswith("rare unique")


class TestHybridRetriever:
    @pytest.fixture()
    def mock_embedding_provider(self) -> Any:
        provider = MagicMock()
        provider.embed_query.return_value = [0.1, 0.2, 0.3]
        return provider

    @pytest.fixture()
    def mock_vector_store(self) -> Any:
        store = MagicMock()
        return store

    def test_hybrid_retrieval_merges_results(
        self, mock_embedding_provider: Any, mock_vector_store: Any
    ) -> None:
        config = RetrievalConfig(strategy="hybrid")

        doc_a = Document(content="Dense match about Python", id="a")
        doc_b = Document(content="BM25 match about Python programming", id="b")
        doc_c = Document(content="Both methods find this Python document", id="c")

        mock_vector_store.search.return_value = [
            (doc_a, 0.95),
            (doc_c, 0.80),
        ]

        retriever = HybridRetriever(
            embedding_provider=mock_embedding_provider,
            vector_store=mock_vector_store,
            config=config,
        )

        # Index for BM25
        retriever.index_documents([doc_a, doc_b, doc_c])

        results = retriever.retrieve("Python programming", top_k=3)

        assert len(results) > 0
        # All three docs should appear since they all contain "Python"
        result_ids = {doc.id for doc in results}
        assert "a" in result_ids  # dense match
        assert "b" in result_ids  # BM25 match

    def test_dense_only_fallback_without_bm25_index(
        self, mock_embedding_provider: Any, mock_vector_store: Any
    ) -> None:
        config = RetrievalConfig(strategy="hybrid")

        doc_a = Document(content="doc A", id="a")
        mock_vector_store.search.return_value = [(doc_a, 0.95)]

        retriever = HybridRetriever(
            embedding_provider=mock_embedding_provider,
            vector_store=mock_vector_store,
            config=config,
        )

        # Don't call index_documents — should fall back to dense-only
        results = retriever.retrieve("test query", top_k=5)

        assert len(results) == 1
        assert results[0].id == "a"

    def test_rrf_fusion(self) -> None:
        """Test Reciprocal Rank Fusion directly."""
        doc_a = Document(content="doc A", id="a")
        doc_b = Document(content="doc B", id="b")
        doc_c = Document(content="doc C", id="c")

        list1 = [(doc_a, 0.9), (doc_b, 0.8)]
        list2 = [(doc_c, 0.9), (doc_a, 0.7)]

        merged = HybridRetriever._reciprocal_rank_fusion(
            [list1, list2], [0.5, 0.5], k=60
        )

        # doc_a appears in both lists, should score highest
        assert merged[0][0].id == "a"
        assert len(merged) == 3


class TestHybridRetrievalConfig:
    def test_defaults(self) -> None:
        config = HybridRetrievalConfig()
        assert config.bm25_weight == 0.5
        assert config.rrf_k == 60
        assert config.bm25_k1 == 1.5
        assert config.bm25_b == 0.75

    def test_in_retrieval_config(self) -> None:
        config = RetrievalConfig()
        assert config.hybrid.bm25_weight == 0.5
