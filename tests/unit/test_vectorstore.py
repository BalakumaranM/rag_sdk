"""Unit tests for vector store providers.

All external dependencies (faiss, chromadb, pinecone) are mocked so tests
run without optional packages installed.
"""

import importlib
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from rag_sdk.config import (
    ChromaConfig,
    FAISSConfig,
    PineconeConfig,
    QdrantConfig,
    WeaviateConfig,
)
from rag_sdk.config.config import VectorStoreConfig
from rag_sdk.document import Document
from rag_sdk.vectorstore.memory import InMemoryVectorStore


# ---------------------------------------------------------------------------
# InMemoryVectorStore
# ---------------------------------------------------------------------------
class TestInMemoryVectorStore:
    def test_add_search_delete_roundtrip(self) -> None:
        store = InMemoryVectorStore()
        docs = [
            Document(id="a", content="hello", metadata={"k": "v1"}),
            Document(id="b", content="world", metadata={"k": "v2"}),
        ]
        embs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        store.add_documents(docs, embs)

        results = store.search([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0][0].id == "a"
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_search_with_filter(self) -> None:
        store = InMemoryVectorStore()
        docs = [
            Document(id="a", content="hello", metadata={"k": "v1"}),
            Document(id="b", content="world", metadata={"k": "v2"}),
        ]
        embs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        store.add_documents(docs, embs)

        results = store.search([1.0, 0.0, 0.0], top_k=2, filters={"k": "v2"})
        assert len(results) == 1
        assert results[0][0].id == "b"

    def test_delete(self) -> None:
        store = InMemoryVectorStore()
        docs = [Document(id="a", content="hello"), Document(id="b", content="world")]
        embs = [[1.0, 0.0], [0.0, 1.0]]
        store.add_documents(docs, embs)
        store.delete(["a"])
        results = store.search([1.0, 0.0], top_k=2)
        assert len(results) == 1
        assert results[0][0].id == "b"

    def test_search_empty_store(self) -> None:
        store = InMemoryVectorStore()
        assert store.search([1.0, 0.0]) == []

    def test_mismatched_lengths_raises(self) -> None:
        store = InMemoryVectorStore()
        with pytest.raises(ValueError, match="Number of documents"):
            store.add_documents([Document(content="x")], [[1.0], [2.0]])


# ---------------------------------------------------------------------------
# FAISSVectorStore (mocked faiss)
# ---------------------------------------------------------------------------
class TestFAISSVectorStore:
    @pytest.fixture(autouse=True)
    def _setup_faiss_mock(self) -> None:  # type: ignore[misc,return]
        """Install a mock faiss module before each test."""
        mock_faiss = MagicMock()

        # IndexFlatIP / IndexFlatL2 return a mock index
        def _make_index(dim: int) -> MagicMock:
            idx = MagicMock()
            idx.ntotal = 0
            _vectors: list[np.ndarray] = []  # type: ignore[type-arg]

            def _add(vecs: np.ndarray) -> None:  # type: ignore[type-arg]
                _vectors.append(vecs.copy())
                idx.ntotal += vecs.shape[0]

            def _search(q: np.ndarray, k: int) -> tuple:  # type: ignore[type-arg]
                if not _vectors:
                    return np.array([[-1.0] * k]), np.array([[-1] * k])
                all_vecs = np.vstack(_vectors)
                scores = all_vecs @ q.T
                scores = scores.flatten()
                top = min(k, len(scores))
                indices = np.argsort(scores)[::-1][:top]
                dists = scores[indices]
                # pad to k
                pad = k - top
                if pad > 0:
                    indices = np.concatenate([indices, np.full(pad, -1)])
                    dists = np.concatenate([dists, np.full(pad, -1.0)])
                return np.array([dists]), np.array([indices])

            def _reconstruct(i: int) -> np.ndarray:  # type: ignore[type-arg]
                all_vecs = np.vstack(_vectors)
                return all_vecs[i]

            def _reset() -> None:
                _vectors.clear()
                idx.ntotal = 0

            idx.add = _add
            idx.search = _search
            idx.reconstruct = _reconstruct
            idx.reset = _reset
            return idx

        mock_faiss.IndexFlatIP = _make_index
        mock_faiss.IndexFlatL2 = _make_index
        mock_faiss.normalize_L2 = lambda x: x.__setitem__(
            slice(None), x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10)
        )
        mock_faiss.write_index = MagicMock()
        mock_faiss.read_index = MagicMock()

        sys.modules["faiss"] = mock_faiss
        # Reload the module so it picks up the mock
        import rag_sdk.vectorstore.faiss_store as fs_mod

        importlib.reload(fs_mod)
        yield
        sys.modules.pop("faiss", None)

    def _make_store(self, **kwargs: str) -> "FAISSVectorStore":  # type: ignore[name-defined] # noqa: F821
        from rag_sdk.vectorstore.faiss_store import FAISSVectorStore

        return FAISSVectorStore(FAISSConfig(**kwargs))

    def test_add_and_search(self) -> None:
        store = self._make_store()
        docs = [Document(id="1", content="foo"), Document(id="2", content="bar")]
        embs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        store.add_documents(docs, embs)

        results = store.search([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0][0].id == "1"

    def test_delete_and_rebuild(self) -> None:
        store = self._make_store()
        docs = [Document(id="1", content="a"), Document(id="2", content="b")]
        embs = [[1.0, 0.0], [0.0, 1.0]]
        store.add_documents(docs, embs)
        store.delete(["1"])
        results = store.search([0.0, 1.0], top_k=2)
        assert len(results) == 1
        assert results[0][0].id == "2"

    def test_search_empty(self) -> None:
        store = self._make_store()
        assert store.search([1.0, 0.0]) == []

    def test_delete_all(self) -> None:
        store = self._make_store()
        docs = [Document(id="1", content="x")]
        store.add_documents(docs, [[1.0]])
        store.delete(["1"])
        assert store.search([1.0]) == []

    def test_cosine_normalization(self) -> None:
        store = self._make_store(metric="cosine")
        docs = [Document(id="1", content="hi")]
        store.add_documents(docs, [[3.0, 4.0]])
        results = store.search([3.0, 4.0], top_k=1)
        assert len(results) == 1
        # After normalization, score should be ~1.0
        assert results[0][1] == pytest.approx(1.0, abs=0.01)

    def test_filter(self) -> None:
        store = self._make_store()
        docs = [
            Document(id="1", content="a", metadata={"tag": "x"}),
            Document(id="2", content="b", metadata={"tag": "y"}),
        ]
        embs = [[1.0, 0.0], [0.9, 0.1]]
        store.add_documents(docs, embs)
        results = store.search([1.0, 0.0], top_k=2, filters={"tag": "y"})
        assert len(results) == 1
        assert results[0][0].id == "2"

    def test_mismatched_lengths_raises(self) -> None:
        store = self._make_store()
        with pytest.raises(ValueError, match="Number of documents"):
            store.add_documents([Document(content="x")], [[1.0], [2.0]])


# ---------------------------------------------------------------------------
# ChromaVectorStore (mocked chromadb)
# ---------------------------------------------------------------------------
class TestChromaVectorStore:
    @pytest.fixture(autouse=True)
    def _setup_chroma_mock(self) -> None:  # type: ignore[misc,return]
        """Install a mock chromadb module."""
        mock_chromadb = types.ModuleType("chromadb")
        _storage: dict[str, dict] = {}  # type: ignore[type-arg]

        def _make_collection(**kwargs: str) -> MagicMock:
            col = MagicMock()

            def upsert(
                ids: list, embeddings: list, documents: list, metadatas: list
            ) -> None:  # type: ignore[type-arg]
                for i, doc_id in enumerate(ids):
                    _storage[doc_id] = {
                        "embedding": embeddings[i],
                        "document": documents[i],
                        "metadata": metadatas[i],
                    }

            def query(
                query_embeddings: list, n_results: int, include: list, **kw: dict
            ) -> dict:  # type: ignore[type-arg]
                q = np.array(query_embeddings[0])
                scored = []
                where = kw.get("where")
                for doc_id, data in _storage.items():
                    if where:
                        skip = False
                        for k, v in where.items():  # type: ignore[union-attr]
                            if data["metadata"].get(k) != v:
                                skip = True
                                break
                        if skip:
                            continue
                    vec = np.array(data["embedding"])
                    # cosine distance
                    cos_sim = np.dot(q, vec) / (
                        np.linalg.norm(q) * np.linalg.norm(vec) + 1e-10
                    )
                    dist = 1.0 - cos_sim
                    scored.append((doc_id, data, dist))
                scored.sort(key=lambda x: x[2])
                scored = scored[:n_results]
                return {
                    "ids": [[s[0] for s in scored]],
                    "documents": [[s[1]["document"] for s in scored]],
                    "metadatas": [[s[1]["metadata"] for s in scored]],
                    "distances": [[s[2] for s in scored]],
                }

            def delete(ids: list) -> None:  # type: ignore[type-arg]
                for doc_id in ids:
                    _storage.pop(doc_id, None)

            col.upsert = upsert
            col.query = query
            col.delete = delete
            return col

        def _ephemeral() -> MagicMock:
            client = MagicMock()
            client.get_or_create_collection = _make_collection
            return client

        mock_chromadb.EphemeralClient = _ephemeral  # type: ignore[attr-defined]
        mock_chromadb.PersistentClient = _ephemeral  # type: ignore[attr-defined]
        mock_chromadb.HttpClient = lambda **kw: _ephemeral()  # type: ignore[attr-defined]

        sys.modules["chromadb"] = mock_chromadb
        import rag_sdk.vectorstore.chroma_store as cs_mod

        importlib.reload(cs_mod)
        yield
        sys.modules.pop("chromadb", None)

    def _make_store(self, **kwargs: Any) -> "ChromaVectorStore":  # type: ignore[name-defined] # noqa: F821
        from rag_sdk.vectorstore.chroma_store import ChromaVectorStore

        return ChromaVectorStore(ChromaConfig(**kwargs))

    def test_add_and_search(self) -> None:
        store = self._make_store()
        docs = [Document(id="1", content="hello"), Document(id="2", content="world")]
        embs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        store.add_documents(docs, embs)

        results = store.search([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0][0].content == "hello"
        # cosine: score = 1 - distance ≈ 1.0 for same vector
        assert results[0][1] == pytest.approx(1.0, abs=0.01)

    def test_delete(self) -> None:
        store = self._make_store()
        docs = [Document(id="1", content="a"), Document(id="2", content="b")]
        embs = [[1.0, 0.0], [0.0, 1.0]]
        store.add_documents(docs, embs)
        store.delete(["1"])
        results = store.search([1.0, 0.0], top_k=2)
        assert len(results) == 1
        assert results[0][0].id == "2"

    def test_filter(self) -> None:
        store = self._make_store()
        docs = [
            Document(id="1", content="a", metadata={"tag": "x"}),
            Document(id="2", content="b", metadata={"tag": "y"}),
        ]
        embs = [[1.0, 0.0], [0.9, 0.1]]
        store.add_documents(docs, embs)
        results = store.search([1.0, 0.0], top_k=2, filters={"tag": "y"})
        assert len(results) == 1
        assert results[0][0].id == "2"

    def test_upsert_overwrites(self) -> None:
        store = self._make_store()
        store.add_documents([Document(id="1", content="old")], [[1.0, 0.0]])
        store.add_documents([Document(id="1", content="new")], [[1.0, 0.0]])
        results = store.search([1.0, 0.0], top_k=1)
        assert results[0][0].content == "new"

    def test_mismatched_lengths_raises(self) -> None:
        store = self._make_store()
        with pytest.raises(ValueError, match="Number of documents"):
            store.add_documents([Document(content="x")], [[1.0], [2.0]])


# ---------------------------------------------------------------------------
# PineconeVectorStore (mocked pinecone)
# ---------------------------------------------------------------------------
class TestPineconeVectorStore:
    @pytest.fixture(autouse=True)
    def _setup_pinecone_mock(self) -> None:  # type: ignore[misc,return]
        """Install a mock pinecone module."""
        mock_pinecone = types.ModuleType("pinecone")
        _storage: dict[str, dict] = {}  # type: ignore[type-arg]

        mock_index = MagicMock()

        def _upsert(vectors: list, namespace: str = "") -> None:  # type: ignore[type-arg]
            for v in vectors:
                _storage[v["id"]] = {"values": v["values"], "metadata": v["metadata"]}

        def _query(
            vector: list,  # type: ignore[type-arg]
            top_k: int = 5,
            include_metadata: bool = True,
            namespace: str = "",
            **kw: dict,
        ) -> dict:
            q = np.array(vector)
            scored = []
            for doc_id, data in _storage.items():
                vec = np.array(data["values"])
                score = float(
                    np.dot(q, vec) / (np.linalg.norm(q) * np.linalg.norm(vec) + 1e-10)
                )
                scored.append(
                    {"id": doc_id, "score": score, "metadata": dict(data["metadata"])}
                )
            scored.sort(key=lambda x: x["score"], reverse=True)  # type: ignore[arg-type,return-value]
            return {"matches": scored[:top_k]}

        def _delete(ids: list, namespace: str = "") -> None:  # type: ignore[type-arg]
            for doc_id in ids:
                _storage.pop(doc_id, None)

        mock_index.upsert = _upsert
        mock_index.query = _query
        mock_index.delete = _delete

        mock_describe = MagicMock()
        mock_describe.host = "test-host.pinecone.io"

        mock_pc_class = MagicMock()
        mock_pc_class.return_value.Index.return_value = mock_index
        mock_pc_class.return_value.describe_index.return_value = mock_describe

        mock_pinecone.Pinecone = mock_pc_class  # type: ignore[attr-defined]
        sys.modules["pinecone"] = mock_pinecone
        import rag_sdk.vectorstore.pinecone as pc_mod

        importlib.reload(pc_mod)
        yield
        sys.modules.pop("pinecone", None)
        _storage.clear()

    def _make_store(self) -> "PineconeVectorStore":  # type: ignore[name-defined] # noqa: F821
        from rag_sdk.vectorstore.pinecone import PineconeVectorStore

        config = PineconeConfig(api_key="test-key", index_host="test-host")  # type: ignore[arg-type]
        return PineconeVectorStore(config)

    def test_add_and_search(self) -> None:
        store = self._make_store()
        docs = [Document(id="1", content="hello", metadata={"k": "v"})]
        store.add_documents(docs, [[1.0, 0.0, 0.0]])

        results = store.search([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0][0].content == "hello"
        assert results[0][0].metadata == {"k": "v"}

    def test_content_in_metadata(self) -> None:
        """Content should be stored under _content key in metadata."""
        store = self._make_store()
        docs = [Document(id="1", content="my content")]
        store.add_documents(docs, [[1.0]])

        # Verify _content is stored in metadata
        results = store.search([1.0], top_k=1)
        assert results[0][0].content == "my content"

    def test_metadata_flattening(self) -> None:
        from rag_sdk.vectorstore.pinecone import PineconeVectorStore

        meta = {"simple": "val", "nested": {"a": 1}}
        flat = PineconeVectorStore._flatten_metadata(meta)
        assert flat["simple"] == "val"
        assert isinstance(flat["nested"], str)

        restored = PineconeVectorStore._unflatten_metadata(flat)
        assert restored["nested"] == {"a": 1}

    def test_delete(self) -> None:
        store = self._make_store()
        store.add_documents([Document(id="1", content="x")], [[1.0]])
        store.delete(["1"])
        results = store.search([1.0], top_k=1)
        assert len(results) == 0

    def test_host_resolution(self) -> None:
        """If index_host is empty, host should be resolved via describe_index."""
        from rag_sdk.vectorstore.pinecone import PineconeVectorStore

        config = PineconeConfig(api_key="test-key", index_host="")  # type: ignore[arg-type]
        store = PineconeVectorStore(config)
        # Should not raise — host resolved via mock describe_index
        assert store is not None

    def test_mismatched_lengths_raises(self) -> None:
        store = self._make_store()
        with pytest.raises(ValueError, match="Number of documents"):
            store.add_documents([Document(content="x")], [[1.0], [2.0]])


# ---------------------------------------------------------------------------
# Stub providers: import guard + NotImplementedError
# ---------------------------------------------------------------------------
class TestWeaviateStub:
    def test_import_error_without_dep(self) -> None:
        sys.modules.pop("weaviate", None)
        from rag_sdk.vectorstore.weaviate_store import WeaviateVectorStore

        with pytest.raises(ImportError, match="weaviate-client"):
            WeaviateVectorStore(WeaviateConfig())

    def test_not_implemented_with_dep(self) -> None:
        sys.modules["weaviate"] = types.ModuleType("weaviate")
        try:
            # Need to reimport to pick up the module
            import importlib

            import rag_sdk.vectorstore.weaviate_store as ws

            importlib.reload(ws)
            with pytest.raises(NotImplementedError, match="not yet implemented"):
                ws.WeaviateVectorStore(WeaviateConfig())
        finally:
            sys.modules.pop("weaviate", None)


class TestQdrantStub:
    def test_import_error_without_dep(self) -> None:
        sys.modules.pop("qdrant_client", None)
        from rag_sdk.vectorstore.qdrant_store import QdrantVectorStore

        with pytest.raises(ImportError, match="qdrant-client"):
            QdrantVectorStore(QdrantConfig())

    def test_not_implemented_with_dep(self) -> None:
        sys.modules["qdrant_client"] = types.ModuleType("qdrant_client")
        try:
            import importlib

            import rag_sdk.vectorstore.qdrant_store as qs

            importlib.reload(qs)
            with pytest.raises(NotImplementedError, match="not yet implemented"):
                qs.QdrantVectorStore(QdrantConfig())
        finally:
            sys.modules.pop("qdrant_client", None)


# ---------------------------------------------------------------------------
# Config defaults and YAML parsing
# ---------------------------------------------------------------------------
class TestVectorStoreConfig:
    def test_defaults(self) -> None:
        cfg = VectorStoreConfig()
        assert cfg.provider == "memory"
        assert cfg.faiss is not None
        assert cfg.chroma is not None
        assert cfg.pinecone is not None
        assert cfg.weaviate is not None
        assert cfg.qdrant is not None

    def test_faiss_config_defaults(self) -> None:
        cfg = FAISSConfig()
        assert cfg.index_type == "Flat"
        assert cfg.metric == "cosine"
        assert cfg.persist_path is None

    def test_chroma_config_defaults(self) -> None:
        cfg = ChromaConfig()
        assert cfg.mode == "ephemeral"
        assert cfg.collection_name == "rag-collection"
        assert cfg.distance_function == "cosine"

    def test_pinecone_config_defaults(self) -> None:
        cfg = PineconeConfig()
        assert cfg.index_name == "rag-index"
        assert cfg.index_host == ""
        assert cfg.namespace == "default"

    def test_yaml_parsing(self) -> None:
        from rag_sdk.config import Config

        cfg = Config(
            **{  # type: ignore[arg-type]
                "vectorstore": {
                    "provider": "faiss",
                    "faiss": {"metric": "l2", "persist_path": "/tmp/idx"},
                }
            }
        )
        assert cfg.vectorstore.provider == "faiss"
        assert cfg.vectorstore.faiss is not None
        assert cfg.vectorstore.faiss.metric == "l2"
        assert cfg.vectorstore.faiss.persist_path == "/tmp/idx"
