"""Unit tests for BasicGraphRAGRetriever — edge cases and individual methods."""

from unittest.mock import MagicMock


from rag_sdk.config import Config
from rag_sdk.document import Document
from rag_sdk.graph.models import Entity
from rag_sdk.retrieval.graph_rag import BasicGraphRAGRetriever


def _make_retriever(llm_response=None, embed_return=None, search_return=None):
    mock_llm = MagicMock()
    if llm_response is not None:
        mock_llm.generate.return_value = llm_response
    mock_embedding = MagicMock()
    mock_embedding.embed_query.return_value = embed_return or [0.1] * 8
    mock_vectorstore = MagicMock()
    mock_vectorstore.search.return_value = search_return or []
    config = Config().retrieval
    retriever = BasicGraphRAGRetriever(
        embedding_provider=mock_embedding,
        vector_store=mock_vectorstore,
        llm_provider=mock_llm,
        config=config,
    )
    return retriever, mock_llm, mock_embedding, mock_vectorstore


class TestBasicGraphRAGRetrieverInit:
    def test_empty_entities(self):
        retriever, *_ = _make_retriever()
        assert retriever.entities == {}

    def test_empty_relationships(self):
        retriever, *_ = _make_retriever()
        assert retriever.relationships == []

    def test_empty_adjacency(self):
        retriever, *_ = _make_retriever()
        assert retriever.adjacency == {}

    def test_providers_stored(self):
        retriever, mock_llm, mock_embedding, mock_vectorstore = _make_retriever()
        assert retriever.llm_provider is mock_llm
        assert retriever.embedding_provider is mock_embedding
        assert retriever.vector_store is mock_vectorstore


class TestExtractEntitiesAndRelationships:
    def test_successful_extraction(self):
        response = (
            '{"entities": [{"name": "Alice", "type": "person"}],'
            '"relationships": [{"source": "Alice", "target": "Lab", "relation": "works_at"}]}'
        )
        retriever, *_ = _make_retriever(llm_response=response)
        entities, rels = retriever._extract_entities_and_relationships("text", "doc1")
        assert len(entities) == 1
        assert entities[0].name == "alice"
        assert entities[0].entity_type == "person"
        assert entities[0].document_ids == ["doc1"]
        assert len(rels) == 1
        assert rels[0].source == "alice"
        assert rels[0].target == "lab"
        assert rels[0].relation == "works_at"

    def test_entity_name_lowercased(self):
        response = '{"entities": [{"name": "UPPER_CASE", "type": "concept"}], "relationships": []}'
        retriever, *_ = _make_retriever(llm_response=response)
        entities, _ = retriever._extract_entities_and_relationships("text", "doc1")
        assert entities[0].name == "upper_case"

    def test_relationship_names_lowercased(self):
        response = (
            '{"entities": [], "relationships":'
            '[{"source": "Alice", "target": "BOB", "relation": "knows"}]}'
        )
        retriever, *_ = _make_retriever(llm_response=response)
        _, rels = retriever._extract_entities_and_relationships("text", "doc1")
        assert rels[0].source == "alice"
        assert rels[0].target == "bob"

    def test_max_entities_respected(self):
        entities_json = ", ".join(
            f'{{"name": "e{i}", "type": "concept"}}' for i in range(20)
        )
        response = f'{{"entities": [{entities_json}], "relationships": []}}'
        retriever, *_ = _make_retriever(llm_response=response)
        max_e = retriever.config.graph_rag.max_entities_per_chunk
        entities, _ = retriever._extract_entities_and_relationships("text", "doc1")
        assert len(entities) <= max_e

    def test_max_relationships_respected(self):
        rels_json = ", ".join(
            f'{{"source": "a", "target": "b{i}", "relation": "r"}}' for i in range(20)
        )
        response = f'{{"entities": [], "relationships": [{rels_json}]}}'
        retriever, *_ = _make_retriever(llm_response=response)
        max_r = retriever.config.graph_rag.max_relationships_per_chunk
        _, rels = retriever._extract_entities_and_relationships("text", "doc1")
        assert len(rels) <= max_r

    def test_llm_exception_returns_empty(self):
        retriever, mock_llm, *_ = _make_retriever()
        mock_llm.generate.side_effect = Exception("LLM error")
        entities, rels = retriever._extract_entities_and_relationships("text", "doc1")
        assert entities == []
        assert rels == []

    def test_invalid_json_returns_empty(self):
        retriever, *_ = _make_retriever(llm_response="not json at all")
        entities, rels = retriever._extract_entities_and_relationships("text", "doc1")
        assert entities == []
        assert rels == []

    def test_missing_type_defaults_to_empty(self):
        response = '{"entities": [{"name": "alice"}], "relationships": []}'
        retriever, *_ = _make_retriever(llm_response=response)
        entities, _ = retriever._extract_entities_and_relationships("text", "doc1")
        assert entities[0].entity_type == ""

    def test_document_id_assigned(self):
        response = (
            '{"entities": [{"name": "alice", "type": "person"}], "relationships": []}'
        )
        retriever, *_ = _make_retriever(llm_response=response)
        entities, _ = retriever._extract_entities_and_relationships("text", "my_doc")
        assert entities[0].document_ids == ["my_doc"]

    def test_empty_arrays_returns_empty_lists(self):
        response = '{"entities": [], "relationships": []}'
        retriever, *_ = _make_retriever(llm_response=response)
        entities, rels = retriever._extract_entities_and_relationships("text", "doc1")
        assert entities == []
        assert rels == []


class TestExtractQueryEntities:
    def test_successful_json_extraction(self):
        retriever, mock_llm, *_ = _make_retriever()
        mock_llm.generate.return_value = '["python", "guido"]'
        result = retriever._extract_query_entities("Who created Python?")
        assert "python" in result
        assert "guido" in result

    def test_results_lowercased(self):
        retriever, mock_llm, *_ = _make_retriever()
        mock_llm.generate.return_value = '["Python", "GUIDO"]'
        result = retriever._extract_query_entities("query")
        assert all(e == e.lower() for e in result)

    def test_fallback_on_llm_exception(self):
        retriever, mock_llm, *_ = _make_retriever()
        mock_llm.generate.side_effect = Exception("fail")
        result = retriever._extract_query_entities("What is machine learning about?")
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(len(e) > 3 for e in result)

    def test_fallback_on_invalid_json(self):
        retriever, mock_llm, *_ = _make_retriever()
        mock_llm.generate.return_value = "not a json array"
        result = retriever._extract_query_entities("What about Python programming?")
        assert isinstance(result, list)

    def test_filters_non_strings_from_json(self):
        retriever, mock_llm, *_ = _make_retriever()
        mock_llm.generate.return_value = '["python", 42, null, "guido"]'
        result = retriever._extract_query_entities("query")
        assert all(isinstance(e, str) for e in result)

    def test_empty_json_array(self):
        retriever, mock_llm, *_ = _make_retriever()
        mock_llm.generate.return_value = "[]"
        result = retriever._extract_query_entities("query")
        assert result == []


class TestGetGraphDocumentIds:
    def _make_retriever_with_graph(self):
        retriever, *_ = _make_retriever()
        retriever.entities = {
            "alice": Entity(name="alice", document_ids=["doc1"]),
            "lab": Entity(name="lab", document_ids=["doc2"]),
            "project": Entity(name="project", document_ids=["doc3"]),
        }
        retriever.adjacency = {
            "alice": {"lab"},
            "lab": {"alice", "project"},
            "project": {"lab"},
        }
        return retriever

    def test_direct_entity_match_returns_doc(self):
        retriever = self._make_retriever_with_graph()
        ids = retriever._get_graph_document_ids(["alice"])
        assert "doc1" in ids

    def test_one_hop_visits_seed_collects_neighbor_on_next(self):
        retriever = self._make_retriever_with_graph()
        # max_hops=1: alice is visited, lab queued but not visited → only doc1
        ids = retriever._get_graph_document_ids(["alice"], max_hops=1)
        assert "doc1" in ids
        # lab (1-hop neighbor) is NOT yet visited with only 1 iteration
        assert "doc2" not in ids

    def test_two_hops_reaches_one_hop_neighbor(self):
        retriever = self._make_retriever_with_graph()
        # max_hops=2: alice (hop1) + lab (hop2) → doc1 and doc2
        ids = retriever._get_graph_document_ids(["alice"], max_hops=2)
        assert "doc1" in ids
        assert "doc2" in ids

    def test_three_hops_reaches_two_hop_neighbor(self):
        retriever = self._make_retriever_with_graph()
        # max_hops=3: alice → lab → project → doc1, doc2, doc3
        ids = retriever._get_graph_document_ids(["alice"], max_hops=3)
        assert "doc3" in ids

    def test_empty_query_entities_returns_empty(self):
        retriever = self._make_retriever_with_graph()
        ids = retriever._get_graph_document_ids([])
        assert ids == set()

    def test_unknown_entity_returns_empty(self):
        retriever = self._make_retriever_with_graph()
        ids = retriever._get_graph_document_ids(["completely_unknown"])
        assert ids == set()

    def test_substring_matching(self):
        retriever = self._make_retriever_with_graph()
        # "alic" is a substring of "alice"
        ids = retriever._get_graph_document_ids(["alic"])
        assert "doc1" in ids

    def test_zero_hops_returns_empty(self):
        retriever = self._make_retriever_with_graph()
        ids = retriever._get_graph_document_ids(["alice"], max_hops=0)
        assert ids == set()

    def test_circular_graph_no_infinite_loop(self):
        retriever, *_ = _make_retriever()
        retriever.entities = {
            "a": Entity(name="a", document_ids=["d1"]),
            "b": Entity(name="b", document_ids=["d2"]),
        }
        retriever.adjacency = {"a": {"b"}, "b": {"a"}}  # circular
        ids = retriever._get_graph_document_ids(["a"], max_hops=5)
        # Should terminate without infinite loop
        assert "d1" in ids
        assert "d2" in ids


class TestBuildGraph:
    def test_entities_populated(self):
        response = (
            '{"entities": [{"name": "Alice", "type": "person"}], "relationships": []}'
        )
        retriever, *_ = _make_retriever(llm_response=response)
        retriever.build_graph([Document(content="Alice is a scientist.", metadata={})])
        assert "alice" in retriever.entities

    def test_relationships_populated(self):
        response = (
            '{"entities": [{"name": "alice", "type": "person"}, {"name": "lab", "type": "org"}],'
            '"relationships": [{"source": "alice", "target": "lab", "relation": "works_at"}]}'
        )
        retriever, *_ = _make_retriever(llm_response=response)
        retriever.build_graph([Document(content="Alice works at Lab.", metadata={})])
        assert len(retriever.relationships) == 1
        assert retriever.relationships[0].relation == "works_at"

    def test_adjacency_populated(self):
        response = (
            '{"entities": [{"name": "alice", "type": "person"}, {"name": "lab", "type": "org"}],'
            '"relationships": [{"source": "alice", "target": "lab", "relation": "works_at"}]}'
        )
        retriever, *_ = _make_retriever(llm_response=response)
        retriever.build_graph([Document(content="Alice works at Lab.", metadata={})])
        assert "lab" in retriever.adjacency.get("alice", set())
        assert "alice" in retriever.adjacency.get("lab", set())

    def test_duplicate_entity_appends_doc_id(self):
        response = (
            '{"entities": [{"name": "Alice", "type": "person"}], "relationships": []}'
        )
        retriever, *_ = _make_retriever(llm_response=response)
        doc1 = Document(content="Alice A.", metadata={}, id="d1")
        doc2 = Document(content="Alice B.", metadata={}, id="d2")
        retriever.build_graph([doc1, doc2])
        assert "d1" in retriever.entities["alice"].document_ids
        assert "d2" in retriever.entities["alice"].document_ids

    def test_empty_documents_list(self):
        retriever, *_ = _make_retriever()
        retriever.build_graph([])
        assert retriever.entities == {}
        assert retriever.relationships == []


class TestRetrieve:
    def test_graph_boosted_doc_scores_higher(self):
        retriever, mock_llm, mock_embedding, mock_vectorstore = _make_retriever()
        retriever.entities = {
            "python": Entity(name="python", document_ids=["graph_doc"])
        }
        retriever.adjacency = {"python": set()}
        mock_llm.generate.return_value = '["python"]'
        mock_embedding.embed_query.return_value = [0.1] * 8

        graph_doc = Document(content="Python doc.", metadata={}, id="graph_doc")
        other_doc = Document(content="Other doc.", metadata={}, id="other_doc")
        # graph_doc has lower raw score but gets 1.2x boost: 0.8 * 1.2 = 0.96 > 0.9
        mock_vectorstore.search.return_value = [
            (other_doc, 0.9),
            (graph_doc, 0.8),
        ]
        results = retriever.retrieve("Tell me about Python", top_k=2)
        assert results[0].id == "graph_doc"

    def test_non_graph_doc_not_boosted(self):
        retriever, mock_llm, mock_embedding, mock_vectorstore = _make_retriever()
        mock_llm.generate.return_value = "[]"
        mock_embedding.embed_query.return_value = [0.1] * 8
        doc1 = Document(content="Doc 1.", metadata={}, id="d1")
        doc2 = Document(content="Doc 2.", metadata={}, id="d2")
        mock_vectorstore.search.return_value = [(doc1, 0.9), (doc2, 0.7)]
        results = retriever.retrieve("query", top_k=2)
        assert results[0].id == "d1"
        assert results[1].id == "d2"

    def test_top_k_limits_results(self):
        retriever, mock_llm, mock_embedding, mock_vectorstore = _make_retriever()
        mock_llm.generate.return_value = "[]"
        mock_embedding.embed_query.return_value = [0.1] * 8
        docs = [Document(content=f"Doc {i}", metadata={}, id=f"d{i}") for i in range(5)]
        mock_vectorstore.search.return_value = [
            (d, 0.9 - i * 0.1) for i, d in enumerate(docs)
        ]
        results = retriever.retrieve("query", top_k=2)
        assert len(results) <= 2

    def test_filters_forwarded_to_vectorstore(self):
        retriever, mock_llm, mock_embedding, mock_vectorstore = _make_retriever()
        mock_llm.generate.return_value = "[]"
        mock_embedding.embed_query.return_value = [0.1] * 8
        mock_vectorstore.search.return_value = []
        retriever.retrieve("query", filters={"source": "wiki"})
        call_kwargs = mock_vectorstore.search.call_args.kwargs
        assert call_kwargs.get("filters") == {"source": "wiki"}

    def test_returns_list(self):
        retriever, mock_llm, mock_embedding, mock_vectorstore = _make_retriever()
        mock_llm.generate.return_value = "[]"
        mock_embedding.embed_query.return_value = [0.1] * 8
        mock_vectorstore.search.return_value = []
        result = retriever.retrieve("query")
        assert isinstance(result, list)
