import sys
import importlib
from types import ModuleType
from unittest.mock import MagicMock
from typing import Any

import pytest
import numpy as np

from rag_sdk.config import LocalEmbeddingConfig


@pytest.fixture()
def mock_sentence_transformers() -> Any:
    """Inject a mock sentence_transformers module."""
    mock_st = ModuleType("sentence_transformers")
    mock_model_cls = MagicMock()
    mock_st.SentenceTransformer = mock_model_cls  # type: ignore[attr-defined]

    sys.modules["sentence_transformers"] = mock_st

    import rag_sdk.embeddings.local as local_mod

    importlib.reload(local_mod)

    yield mock_model_cls

    del sys.modules["sentence_transformers"]


class TestLocalEmbedding:
    def test_embed_documents(self, mock_sentence_transformers: Any) -> None:
        from rag_sdk.embeddings.local import LocalEmbedding

        mock_model = mock_sentence_transformers.return_value
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        config = LocalEmbeddingConfig(model="BAAI/bge-small-en-v1.5")
        provider = LocalEmbedding(config)

        results = provider.embed_documents(["hello", "world"])

        assert len(results) == 2
        assert results[0] == pytest.approx([0.1, 0.2, 0.3])
        assert results[1] == pytest.approx([0.4, 0.5, 0.6])

        mock_model.encode.assert_called_once()

    def test_embed_query(self, mock_sentence_transformers: Any) -> None:
        from rag_sdk.embeddings.local import LocalEmbedding

        mock_model = mock_sentence_transformers.return_value
        mock_model.encode.return_value = np.array([[0.7, 0.8, 0.9]])

        config = LocalEmbeddingConfig()
        provider = LocalEmbedding(config)

        result = provider.embed_query("test query")

        assert result == pytest.approx([0.7, 0.8, 0.9])

    def test_query_prefix(self, mock_sentence_transformers: Any) -> None:
        from rag_sdk.embeddings.local import LocalEmbedding

        mock_model = mock_sentence_transformers.return_value
        mock_model.encode.return_value = np.array([[0.1, 0.2]])

        config = LocalEmbeddingConfig(
            query_prefix="query: ",
            document_prefix="passage: ",
        )
        provider = LocalEmbedding(config)

        provider.embed_query("test")
        call_args = mock_model.encode.call_args[0][0]
        assert call_args == ["query: test"]

    def test_document_prefix(self, mock_sentence_transformers: Any) -> None:
        from rag_sdk.embeddings.local import LocalEmbedding

        mock_model = mock_sentence_transformers.return_value
        mock_model.encode.return_value = np.array([[0.1, 0.2]])

        config = LocalEmbeddingConfig(document_prefix="passage: ")
        provider = LocalEmbedding(config)

        provider.embed_documents(["doc text"])
        call_args = mock_model.encode.call_args[0][0]
        assert call_args == ["passage: doc text"]

    def test_import_error_without_dep(self) -> None:
        original = sys.modules.pop("sentence_transformers", None)
        try:
            import rag_sdk.embeddings.local as local_mod

            importlib.reload(local_mod)

            with pytest.raises(ImportError, match="sentence-transformers"):
                local_mod.LocalEmbedding(LocalEmbeddingConfig())
        finally:
            if original is not None:
                sys.modules["sentence_transformers"] = original


class TestLocalEmbeddingConfig:
    def test_defaults(self) -> None:
        config = LocalEmbeddingConfig()
        assert config.model == "BAAI/bge-small-en-v1.5"
        assert config.query_prefix == ""
        assert config.document_prefix == ""
        assert config.batch_size == 32

    def test_in_embedding_config(self) -> None:
        from rag_sdk.config.config import EmbeddingConfig

        config = EmbeddingConfig()
        assert config.local is not None
        assert config.local.model == "BAAI/bge-small-en-v1.5"  # type: ignore[union-attr]
